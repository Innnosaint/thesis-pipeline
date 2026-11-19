"""Validation and incremental A/B/C comparison for Table 4.6.

Metrics JSON are outputs of dashboard_engine.evaluate. The file format does
not encode run parameters or dataset identity: counts alone cannot prove
that experiments used identical data/configuration.
"""
from __future__ import annotations

import math

REQUIRED = ('matched', 'labels_count', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'fpr', 'f1')
EXPERIMENTS = (
    ('A', 'Existing detector (SRGG OFF, suppression OFF)'),
    ('B', 'Existing + Original SRGG (suppression OFF)'),
    ('C', 'Full pipeline (Original SRGG + suppression ON)'),
)

class Table46Error(ValueError):
    pass


def check_metrics(raw, label):
    if not isinstance(raw, dict):
        raise Table46Error(f'Experiment {label}: expected a JSON object.')
    missing = [key for key in REQUIRED if key not in raw]
    if missing:
        raise Table46Error(f'Experiment {label}: missing {", ".join(missing)}. Export current_evaluation.json from Research Evaluation.')
    for key in ('matched', 'labels_count', 'tp', 'fp', 'tn', 'fn'):
        val=raw[key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise Table46Error(f'Experiment {label}: {key} must be a non-negative integer.')
    n=sum(raw[k] for k in ('tp','fp','tn','fn'))
    if n != raw['matched'] or n == 0:
        raise Table46Error(f'Experiment {label}: TP + FP + TN + FN must equal matched (and be nonzero).')
    if raw['matched'] > raw['labels_count']:
        raise Table46Error(f'Experiment {label}: matched cannot exceed labels_count.')
    tp,fp,tn,fn=(raw[k] for k in ('tp','fp','tn','fn'))
    calculated={
        'precision':tp/(tp+fp) if tp+fp else 0.0,
        'recall':tp/(tp+fn) if tp+fn else 0.0,
        'fpr':fp/(fp+tn) if fp+tn else 0.0,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
    }
    for key,expected in calculated.items():
        val=raw[key]
        if isinstance(val,bool) or not isinstance(val,(float,int)) or not math.isfinite(val) or abs(val-expected) > 0.00001:
            raise Table46Error(f'Experiment {label}: {key} disagrees with its confusion matrix.')
    return dict(raw)


def build_comparison(a,b,c):
    metrics={key:check_metrics(raw,key) for key,raw in zip(('A','B','C'),(a,b,c))}
    matched={m['matched'] for m in metrics.values()}
    label_counts={m['labels_count'] for m in metrics.values()}
    positives={m['tp']+m['fn'] for m in metrics.values()}
    negatives={m['tn']+m['fp'] for m in metrics.values()}
    if len(matched)!=1 or len(label_counts)!=1 or len(positives)!=1 or len(negatives)!=1:
        raise Table46Error('The runs have different evaluated totals or label counts. Use the same dataset and the same label file for A/B/C.')
    rows=[]
    for key, desc in EXPERIMENTS:
        m=metrics[key]
        rows.append({
            'Experiment':key,
            'Configuration':desc,
            'TP':m['tp'], 'FP':m['fp'], 'FN':m['fn'], 'TN':m['tn'],
            'Precision (%)':round(100*m['precision'],4),
            'Recall (%)':round(100*m['recall'],4),
            'F1-score (%)':round(100*m['f1'],4),
            'False-positive rate (%)':round(100*m['fpr'],6),
            'Evaluated observations':m['matched'],
            'Positive labels':m['tp']+m['fn'],
        })
    return rows
