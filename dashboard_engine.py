"""Isolated runs of the original thesis detection modules; no demo/fake alerts."""
import csv
import io
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from ingest import ingest_directory
from preprocess import preprocess_pipeline
from baseline import run_baseline_over_stream
from zscore_detector import classify
from debounce import apply_debounce
from log_correlation import correlate_and_score
from alert_scoring import rank_alerts, write_audit_log


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as f:
        for obj in records:
            row = dict(obj)
            if isinstance(row.get('timestamp'), datetime):
                row['timestamp'] = row['timestamp'].isoformat()
            f.write(json.dumps(row, default=str) + '\n')


def parse_upload(file_name, data):
    """Return plain syslog-style lines. CSV requires timestamp and disk_usage_value columns."""
    name = Path(file_name).name
    ext = Path(name).suffix.lower()
    if ext not in {'.log', '.txt', '.csv'}:
        raise ValueError('Dataset must be .log, .txt, or .csv')
    if len(data) > 60 * 1024 * 1024:
        raise ValueError('Each dataset must be at most 60 MB')
    contents = data.decode('utf-8-sig', errors='replace')
    if ext != '.csv':
        return contents
    reader = csv.DictReader(io.StringIO(contents))
    if not reader.fieldnames or not {'timestamp', 'disk_usage_value'}.issubset(reader.fieldnames):
        raise ValueError('CSV needs columns: timestamp, disk_usage_value; optional: message, log_source')
    lines = []
    for n, row in enumerate(reader, 2):
        try:
            dt = datetime.fromisoformat((row['timestamp'] or '').strip().replace('Z', '+00:00'))
            value = float(row['disk_usage_value'])
        except (ValueError, TypeError) as ex:
            raise ValueError(f'Invalid timestamp/disk usage in CSV line {n}') from ex
        if not 0 <= value <= 100:
            raise ValueError(f'Disk usage must be between 0 and 100 (line {n})')
        # Ingest recognizes second-resolution ISO-like timestamp without T or offset.
        stamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        safe_message = re.sub(r'[\r\n]+', ' ', row.get('message') or '').strip()
        lines.append(f'{stamp} dataset diskmon[1000]: /dev/dataset {value:.4f}% used {safe_message}')
    return '\n'.join(lines) + '\n'


def load_labels(data, file_name):
    if not data:
        return None
    text = data.decode('utf-8-sig')
    rows = list(csv.DictReader(io.StringIO(text))) if file_name.lower().endswith('.csv') else [json.loads(line) for line in text.splitlines() if line.strip()]
    labels = {}
    for i, row in enumerate(rows, 1):
        if 'timestamp' not in row or 'is_anomaly' not in row:
            raise ValueError('Labels must contain timestamp and is_anomaly')
        key = datetime.fromisoformat(str(row['timestamp']).replace('Z', '+00:00')).replace(tzinfo=None).isoformat()
        flag = row['is_anomaly']
        if isinstance(flag, bool):
            actual = flag
        elif str(flag).strip().lower() in {'true','1','yes'}:
            actual = True
        elif str(flag).strip().lower() in {'false','0','no'}:
            actual = False
        else:
            raise ValueError(f'Invalid is_anomaly value on label line {i}')
        labels[key] = actual
    return labels


def evaluate(rows, labels, prediction=None):
    if labels is None:
        return None
    tp = fp = tn = fn = 0
    seen = set()
    for row in rows:
        if not row.get('baseline_ready'):
            continue
        ts = row['timestamp'].isoformat()
        if ts not in labels or ts in seen:
            continue
        seen.add(ts)
        pred = prediction(row) if prediction else (row['classification'] == 'ABNORMAL' and not row.get('suppressed'))
        actual = labels[ts]
        if pred and actual: tp += 1
        elif pred: fp += 1
        elif actual: fn += 1
        else: tn += 1
    n = tp + fp + tn + fn
    return {'matched':n, 'labels_count':len(labels), 'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,
            'fpr':fp/(fp+tn) if fp+tn else 0,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


def run_dataset(datasets, labels_data=None, labels_name='', window=50, min_obs=10,
                theta=5.5, ceiling=85.0, debounce=1, delta=30, suppress_benign=True):
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    folder = ROOT / 'runs' / run_id
    raw_dir = folder / 'raw'
    raw_dir.mkdir(parents=True)
    if not datasets:
        raise ValueError('Choose at least one dataset')
    for n, (filename, data) in enumerate(datasets):
        normalized = parse_upload(filename, data)
        (raw_dir / f'upload_{n:02d}.log').write_text(normalized, encoding='utf8')
    cfg = yaml.safe_load((ROOT / 'config/settings.yaml').read_text())
    cfg['baseline']['window_size'] = window
    cfg['baseline']['min_observations'] = min_obs
    cfg['zscore']['theta_z'] = theta
    cfg['zscore']['hard_ceiling'] = ceiling
    cfg['debounce']['min_consecutive'] = debounce
    cfg['correlation']['delta_t_seconds'] = delta
    (folder / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf8')
    parsed = ingest_directory(str(raw_dir), str(folder / 'drain_state.json'))
    if not parsed:
        raise ValueError('No valid timestamps recognized. Use syslog timestamps (Jan 1 04:02:05) or YYYY-MM-DD HH:MM:SS.')
    clean = preprocess_pipeline(parsed)
    if not clean:
        raise ValueError('No disk readings found. Log lines need "disk usage: 75%" or "75% used".')
    if len(clean) < min_obs:
        raise ValueError(f'Only {len(clean)} unique disk readings, but min_observations={min_obs}. Add more readings or lower the minimum.')
    baselined = run_baseline_over_stream(clean, window, min_obs, cfg['baseline'].get('sigma_floor',0))
    classified = classify(baselined, theta, ceiling)
    classified = apply_debounce(classified, debounce)
    scored = correlate_and_score(classified, parsed, cfg)
    if not suppress_benign:
        # A posteriori study ablation: correlation prioritizes, without suppressing.
        for entry in scored:
            if entry['classification'] == 'ABNORMAL':
                entry['suppressed'] = False
    alerts = rank_alerts(scored)
    write_jsonl(folder / 'parsed.jsonl', parsed)
    write_jsonl(folder / 'clean.jsonl', clean)
    write_jsonl(folder / 'final_scored_stream.jsonl', scored)
    write_jsonl(folder / 'ranked_alerts.jsonl', alerts)
    write_audit_log(scored, str(folder / 'audit_trail.jsonl'))
    labels = load_labels(labels_data, labels_name) if labels_data else None
    metrics = evaluate(scored, labels)
    static_80_metrics = evaluate(scored, labels, lambda r: r['disk_usage_value'] >= 80)
    static_85_metrics = evaluate(scored, labels, lambda r: r['disk_usage_value'] >= 85)
    summary = {'run_id':run_id, 'uploaded_files':[name for name,_ in datasets],
               'parsed':len(parsed), 'readings':len(scored), 'alerts':len(alerts),
               'suppressed':sum(bool(r.get('suppressed')) for r in scored),
               'high':sum(r.get('priority')=='HIGH' and r['classification']=='ABNORMAL' for r in scored),
               'metrics':metrics,
               'static_80_metrics': static_80_metrics,
               'static_85_metrics': static_85_metrics,
               'settings': {'window':window, 'min_obs':min_obs, 'theta':theta,
                            'ceiling':ceiling, 'debounce':debounce,
                            'delta':delta, 'suppress_benign':suppress_benign}}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    return folder, summary


def load_run(folder):
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder / 'final_scored_stream.jsonl').open(encoding='utf8')]
    return json.loads((folder / 'summary.json').read_text()), rows
