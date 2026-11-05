"""User-triggered Chapter 4-style experiments. No manuscript metrics are hardcoded.

All tables are calculated from a saved, user-submitted and labeled run. Uses
exactly the thesis project's baseline, classifier, debounce and correlation
functions, re-running them whenever parameters change. Outputs are stored in
runs/<run_id>/experiment_tables/ and never overwrite published results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from dashboard_engine import ROOT, evaluate, load_labels

sys.path.insert(0, str(ROOT / 'src'))
from baseline import run_baseline_over_stream
from zscore_detector import classify
from debounce import apply_debounce
from log_correlation import correlate_and_score

TABLE_DESCRIPTIONS = {
    '4.1': 'Z-Score Threshold Sensitivity Sweep (no ceiling)',
    '4.2': 'Window-Size Sensitivity Sweep (no ceiling)',
    '4.3': 'Joint Z-Score × Safety-Ceiling Sensitivity Sweep',
    '4.4': 'Static Threshold vs. Enhanced and Hybrid Comparison',
    '4.5': 'Bootstrap 95% Confidence Intervals',
}

JOINT_SELECTED = [(4.0,85), (4.5,85), (5.0,85), (5.5,85), (6.0,85),
                  (5.0,90), (5.0,None), (5.5,80), (5.5,82), (5.5,88), (5.5,None)]


def _read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec['timestamp'] = datetime.fromisoformat(rec['timestamp'])
                rows.append(rec)
    return rows


def _scores(m):
    return { 'TP':m['tp'], 'FP':m['fp'], 'TN':m['tn'], 'FN':m['fn'],
             'Precision':round(m['precision'],4), 'Recall':round(m['recall'],4),
             'F1':round(m['f1'],4), 'FPR':round(m['fpr'],6), 'Matched':m['matched'] }


def _save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _evaluate_variant(clean, parsed, labels, cfg, window, min_obs, theta, ceiling, debounce,
                      suppression, sigma_floor=0):
    # The original modules mutate entries in place; copy source observations.
    baseline = run_baseline_over_stream([dict(e) for e in clean],window_size=window,
                  min_observations=min_obs,sigma_floor=sigma_floor)
    classified = classify(baseline,theta_z=theta,hard_ceiling=ceiling)
    debounced = apply_debounce(classified,min_consecutive=debounce)
    scored = correlate_and_score(debounced,parsed,cfg)
    if not suppression:
        for e in scored:
            if e['classification']=='ABNORMAL':
                e['suppressed']=False
    metrics=evaluate(scored,labels)
    return metrics, scored


def _matched_predictions(scored,labels,static_threshold=None):
    """Match one baseline-ready reading per unique labeled timestamp."""
    result=[];seen=set()
    for e in scored:
        if not e.get('baseline_ready'):continue
        stamp=e['timestamp'].isoformat()
        if stamp in seen or stamp not in labels:continue
        seen.add(stamp)
        predicted=(e['disk_usage_value']>=static_threshold if static_threshold is not None
                else e['classification']=='ABNORMAL' and not e.get('suppressed'))
        result.append((stamp,bool(labels[stamp]),bool(predicted)))
    return result


def _bootstrap(matched_static, matched_enhanced, n_resamples, seed):
    """Exact observation bootstrap via multinomial counts across joint outcomes.

    Resampling counts from the 8 (truth, static prediction, enhanced prediction)
    categories is distributionally equivalent to resampling the N matched rows
    with replacement, without allocating an N × resamples matrix.
    """
    indexed={t:(y,p) for t,y,p in matched_enhanced}
    paired=[(y,int(p),int(indexed[t][1])) for t,y,p in matched_static
            if t in indexed and y==indexed[t][0]]
    if not paired:raise ValueError('No matched observations available for paired bootstrap.')
    cnt=Counter(paired)
    categories=list(cnt)
    probs=np.array([cnt[c] for c in categories],dtype=float)/len(paired)
    rng=np.random.default_rng(seed)
    draws=rng.multinomial(len(paired),probs,size=n_resamples)

    def score(pred_idx):
        y=np.array([c[0] for c in categories],dtype=bool)
        pred=np.array([c[pred_idx] for c in categories],dtype=bool)
        tp=draws[:,y&pred].sum(axis=1)
        fp=draws[:,~y&pred].sum(axis=1)
        fn=draws[:,y&~pred].sum(axis=1)
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0)
        recall=np.divide(tp,tp+fn,out=np.zeros_like(tp,dtype=float),where=(tp+fn)>0)
        f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
        return {'Precision':precision,'Recall':recall,'F1':f1}
    static=score(1);enhanced=score(2)
    rows=[]
    for metric in ('Precision','Recall','F1'):
        a=np.quantile(static[metric],[0.025,0.975]);b=np.quantile(enhanced[metric],[0.025,0.975])
        rows.append({'Metric':metric,'Static 80% CI lower':round(float(a[0]),4),
                     'Static 80% CI upper':round(float(a[1]),4),
                     'Enhanced CI lower':round(float(b[0]),4),
                     'Enhanced CI upper':round(float(b[1]),4)})
    diff=enhanced['F1']-static['F1']
    extra={'resamples':n_resamples,'paired_samples':len(paired),
           'seed':seed,'f1_difference_mean':float(diff.mean()),
           'f1_difference_95_ci':[float(x) for x in np.quantile(diff,[.025,.975])],
           'probability_enhanced_f1_greater':float(np.mean(diff>0))}
    return rows,extra


def generate_experiment_tables(run_folder, *, selected=('4.1','4.2','4.3','4.4','4.5'),
                               window=50,min_obs=10,theta_final=5.5,ceiling_final=85.0,
                               delta=30,debounce=1,sigma_floor=0,
                               theta_sweep=(1.5,2,2.5,3,3.5,4),
                               window_sweep=(10,15,20,30,50,75,100,150,200),
                               window_theta=2.0, window_warmup_ratio=0.2, joint_full=False,
                               bootstrap_resamples=5000, bootstrap_seed=42, callback=None):
    run_folder=Path(run_folder).resolve()
    if run_folder.parent != (ROOT/'runs').resolve():
        raise ValueError('Select an experiment stored under the project runs/ directory.')
    summary_path=run_folder/'summary.json'
    labels_path=run_folder/'ground_truth_uploaded'
    if not summary_path.exists() or not labels_path.exists():
        raise ValueError('This experiment has no saved uploaded labels. Re-run Dataset Laboratory with a label file.')
    source_summary=json.loads(summary_path.read_text(encoding='utf-8'))
    name=source_summary.get('labels_filename','')
    labels=load_labels(labels_path.read_bytes(),name)
    clean=_read_jsonl(run_folder/'clean.jsonl')
    parsed=_read_jsonl(run_folder/'parsed.jsonl')
    cfg=yaml.safe_load((run_folder/'config.yaml').read_text(encoding='utf-8'))
    cfg['correlation']['delta_t_seconds']=delta
    if not labels:raise ValueError('Uploaded label file contains no valid labels.')
    # Protect from zero/poor label matches, which can create misleading F1 values.
    matched_timestamps={e['timestamp'].isoformat() for e in clean if e['timestamp'].isoformat() in labels}
    matched=len(matched_timestamps)
    if matched<3:raise ValueError('Fewer than 3 uploaded ground-truth timestamps match the disk readings.')
    eligible={e['timestamp'].isoformat() for e in clean[min_obs:] if e['timestamp'].isoformat() in labels}
    positive_count=sum(bool(labels[t]) for t in eligible)
    negative_count=len(eligible)-positive_count
    if positive_count==0 or negative_count==0:
        raise ValueError('Matched baseline-ready labels must include both anomalous and normal observations to calculate comparable precision/recall/F1 tables.')
    selected=tuple(s for s in selected if s in TABLE_DESCRIPTIONS)
    if not selected:raise ValueError('Select at least one results table to compute.')
    if min_obs>window:raise ValueError('Warm-up observations cannot exceed the rolling window.')
    # Each button click creates a separate immutable results batch. Failed or
    # interrupted sweeps cannot overwrite an earlier validated experiment.
    batch_id=datetime.utcnow().strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]
    target=run_folder/'experiment_tables'/batch_id
    target.mkdir(parents=True,exist_ok=False)
    reported={}
    def variant(w,t,c,suppress=True,mo=None):
        return _evaluate_variant(clean,parsed,labels,cfg,w,mo if mo is not None else min(min_obs,w),
                                  t,c,debounce,suppress,sigma_floor)
    def notify(detail):
        if callback:callback(detail)
    static80=None
    def static():
        nonlocal static80
        if static80 is None:
            m,stream=variant(window,theta_final,ceiling_final,True)
            static80=_scores(evaluate(stream,labels,lambda e:e['disk_usage_value']>=80))
        return static80

    if '4.1' in selected:
        rows=[]
        for t in theta_sweep:
            notify(f'Table 4.1 — threshold {t:g}')
            m,_=variant(window,t,None,True)
            rows.append({'θ_z':t,**_scores(m)})
        rows.append({'θ_z':'STATIC 80%',**static()})
        _save_csv(target/'table_4_1.csv',rows);reported['4.1']={'rows':len(rows)}

    if '4.2' in selected:
        rows=[]
        for w in window_sweep:
            mo=min(int(w),max(5,int(w*window_warmup_ratio)))
            notify(f'Table 4.2 — window {w}')
            m,_=variant(w,window_theta,None,True,mo=mo)
            rows.append({'W':w,'min_observations':mo,**_scores(m)})
        _save_csv(target/'table_4_2.csv',rows);reported['4.2']={'rows':len(rows)}

    if '4.3' in selected:
        combinations=([(t,c) for t in (4.0,4.5,5.0,5.5,6.0,6.5,7.0)
                         for c in (80,82,84,85,86,88,90,92,None)] if joint_full else JOINT_SELECTED)
        rows=[]
        for idx,(t,c) in enumerate(combinations,1):
            notify(f'Table 4.3 — combination {idx}/{len(combinations)}: θ={t:g}, ceiling={c}')
            m,_=variant(window,t,c,True)
            rows.append({'θ_z':t,'θ_ceiling':f'{c:g}%' if c is not None else 'none',**_scores(m)})
        rows.append({'θ_z':'STATIC 80%','θ_ceiling':'—',**static()})
        _save_csv(target/'table_4_3.csv',rows);reported['4.3']={'rows':len(rows)}

    enhanced_stream=None;hybrid_stream=None
    if '4.4' in selected or '4.5' in selected:
        notify('Table 4.4 — final enhanced configuration and priority-only variant')
        enhanced_m,enhanced_stream=variant(window,theta_final,ceiling_final,True)
        hybrid_m,hybrid_stream=variant(window,theta_final,ceiling_final,False)
        static_80=evaluate(enhanced_stream,labels,lambda e:e['disk_usage_value']>=80)
        static_85=evaluate(enhanced_stream,labels,lambda e:e['disk_usage_value']>=85)
        variants=[('Static 80%',static_80),('Static 85%',static_85),
                  ('Enhanced (suppression on)',enhanced_m),('Hybrid (priority only)',hybrid_m)]
        rows=[]
        for name,m in variants:
            a=_scores(m)
            a={'Method':name,**a,'Δ F1 vs Static 80%':round(m['f1']-static_80['f1'],4)}
            rows.append(a)
        if '4.4' in selected:
            _save_csv(target/'table_4_4.csv',rows);reported['4.4']={'rows':len(rows)}

    if '4.5' in selected:
        notify('Table 4.5 — paired observation bootstrap')
        a=_matched_predictions(enhanced_stream,labels,80)
        b=_matched_predictions(enhanced_stream,labels)
        rows,extra=_bootstrap(a,b,int(bootstrap_resamples),int(bootstrap_seed))
        _save_csv(target/'table_4_5.csv',rows)
        (target/'bootstrap_detail.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
        reported['4.5']={'rows':len(rows),**extra}

    manifest={'table_batch_id':batch_id,
              'source_run_id':source_summary['run_id'],
              'source_files':source_summary['uploaded_files'],
              'labels_filename':name,
              'label_provenance':source_summary.get('label_provenance','Not specified'),
              'label_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
              'matched_input_timestamps':matched,
              'total_disk_readings':len(clean),
              'label_count':len(labels),
              'eligible_matched_positive':positive_count,
              'eligible_matched_negative':negative_count,
              'computed_at_utc':datetime.utcnow().isoformat(timespec='seconds')+'Z',
              'parameter_settings':{'window':window,'min_observations':min_obs,
                    'theta_final':theta_final,'ceiling_final':ceiling_final,'delta':delta,
                    'debounce':debounce,'sigma_floor':sigma_floor,
                    'theta_sweep':list(theta_sweep),'window_sweep':list(window_sweep),
                    'window_theta':window_theta,'window_warmup_ratio':window_warmup_ratio,'joint_full':joint_full,
                    'bootstrap_resamples':bootstrap_resamples,'bootstrap_seed':bootstrap_seed},
              'tables':reported,
              'caution':'Results are computed only for this uploaded dataset and its labels, not manuscript transcriptions. Labels derived from the same ceiling can bias comparisons.'}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    notify('All selected tables saved to '+str(target))
    return target,manifest


def load_table_csv(folder,number):
    import pandas as pd
    return pd.read_csv(Path(folder)/f'table_{number.replace(".","_")}.csv')
