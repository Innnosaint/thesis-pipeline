"""Exploratory detector-stage comparison of existing vs original/strengthened SRGG.

Run in the thesis repo (where src/generate_synthetic_disk_usage.py and
original Loghub file are available):

  python benchmark_srgg_stages.py --repo /path/to/thesis --out srgg_results

Requires pandas, numpy. Uses generator injection labels, NOT independently
verified operational incidents. Results are not a replacement for Chapter 4.
"""
import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from src.seasonal_growth_guard import SRGGConfig, apply_srgg


def metrics(labels, predictions, warmup=10):
    truth=np.asarray(labels,dtype=bool)[warmup:]
    pred=np.asarray(predictions,dtype=bool)[warmup:]
    tp=int(np.sum(truth & pred));fp=int(np.sum(~truth & pred))
    fn=int(np.sum(truth & ~pred));tn=int(np.sum(~truth & ~pred))
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,
                Precision=tp/(tp+fp) if tp+fp else 0,
                Recall=tp/(tp+fn) if tp+fn else 0,
                F1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0)


def classify_old(values):
    s=pd.Series(values,dtype='float64')
    mu=s.shift(1).rolling(50,min_periods=10).mean()
    sigma=s.shift(1).rolling(50,min_periods=10).std(ddof=0).clip(lower=1e-6)
    z=((s-mu)/sigma).fillna(0).to_numpy()
    old=((np.abs(z)>=5.5)|(s.to_numpy()>=85))
    return old,z


def candidate(series,old,mode):
    if mode == 'one-day':
        cfg=SRGGConfig(reference_days=(1,),persistence_readings=1)
    else:
        cfg=SRGGConfig(reference_days=(1,2,3),persistence_readings=3)
    rows=[{'timestamp':r['dt'],'disk_usage_value':r['usage'],
           'classification':'ABNORMAL' if bool(old[i]) else 'NORMAL',
           'zscore':0., 'severity_score':0., 'triggered_by':'original' if old[i] else None}
          for i,r in enumerate(series)]
    result=apply_srgg(rows,cfg)
    return np.array([r['classification']=='ABNORMAL' for r in result],dtype=bool),sum(r['srgg_new_detection'] for r in result)


def run(repo,out, raw_override=None):
    sys.path.insert(0,str(repo/'src'))
    from generate_synthetic_disk_usage import parse_real_log_timestamps,generate_synthetic_series
    choices = [repo/'data/raw_backup/Linux_full_original.log',
               repo/'data/external/loghub/Linux.log',
               repo/'data/raw/Linux_full_original.log']
    raw=Path(raw_override) if raw_override else next((q for q in choices if q.exists()), None)
    if raw is None or not raw.exists():
        raise FileNotFoundError('Original Linux log not found; provide --raw /path/to/Linux.log')
    start,end,*_=parse_real_log_timestamps(str(raw),2024)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in [42,43,44,45]:
        series=generate_synthetic_series(start,end,5,seed)
        values=np.array([r['usage'] for r in series],float)
        labels=np.array([r['is_anomaly'] for r in series],bool)
        old,z=classify_old(values)
        predicates={
          'V0 - Static 80%':values>=80,
          'V1 - Static 85%':values>=85,
          'V2 - Z-score + 85% ceiling':old,
        }
        for name,mode in [('V3 - V2 + original SRGG','one-day'),
                          ('V4 - V2 + strengthened SRGG','three-day')]:
            predicates[name],_=candidate(series,old,mode)
        for method,pred in predicates.items():
            records.append({'seed':seed,'split':'held-out seed (same generator)' if seed==45 else 'development seed',
                            'scenario':'injected anomalies', 'method':method,**metrics(labels,pred)})
        if seed==45:
            # Deliberate counterexample: planned temporary growth NOT labeled anomalous.
            stressed=values.copy()
            for i in (22000,45000,70000):
                stressed[i:i+72]+=np.arange(1,73)*18/72
                stressed[i+72:i+168]+=18
                stressed[i+168:i+240]+=18*(1-np.arange(1,73)/72)
            stressed=np.clip(stressed,0,100)
            altered=[dict(r,usage=float(v)) for r,v in zip(series,stressed)]
            old2,z2=classify_old(stressed)
            stress_preds={'V2 - Z-score + 85% ceiling':old2}
            for name,mode in [('V3 - V2 + original SRGG','one-day'),
                              ('V4 - V2 + strengthened SRGG','three-day')]:
                stress_preds[name],_=candidate(altered,old2,mode)
            for method,pred in stress_preds.items():
                records.append({'seed':seed,'split':'stress scenario','scenario':'planned temporary growth',
                                'method':method,**metrics(labels,pred)})
    output=out/'incremental_results.csv'
    with output.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    for r in records:
        if r['seed']==45:
            print(f"{r['scenario']:<26} {r['method']:<35} TP {r['TP']:<4} FP {r['FP']:<4} F1 {r['F1']:.4f}")
    print('Saved', output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('srgg_results'))
    p.add_argument('--raw',type=Path,default=None,help='Original Loghub Linux.log path')
    a=p.parse_args();run(a.repo.resolve(),a.out.resolve(),a.raw)
