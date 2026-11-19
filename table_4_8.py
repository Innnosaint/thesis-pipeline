"""Evidence validators for Table 4.8 and optional D row in Table 4.6.

No synthetic or missing values are generated. Source files remain unchanged.
Confusion matrices prove observed differences; the small audit CSV verifies
which C->D alerts changed and whether they meet the 85% critical ceiling.
"""
from __future__ import annotations

import csv
from datetime import datetime
import io
import math

from table_4_6 import check_metrics, Table46Error


class Table48Error(ValueError):
    pass


def _validate_pair(c, d):
    try:
        c = check_metrics(c, 'C')
        d = check_metrics(d, 'D')
    except Table46Error as exc:
        raise Table48Error(str(exc)) from exc
    for key in ('matched', 'labels_count'):
        if c[key] != d[key]:
            raise Table48Error(f'C and D differ in {key}; upload comparable experiment metrics.')
    if c['tp'] + c['fn'] != d['tp'] + d['fn'] or c['tn'] + c['fp'] != d['tn'] + d['fp']:
        raise Table48Error('C and D do not have the same number of positive and negative ground-truth labels.')
    return c, d


def compare_c_d(c_raw, d_raw):
    """Return rows and directly measured differences for C vs D."""
    c, d = _validate_pair(c_raw, d_raw)
    descriptions = {'C': 'Original SRGG + contextual suppression',
                    'D': 'Original SRGG + suppression + critical-ceiling override'}
    rows = []
    for label, m in [('C', c), ('D', d)]:
        rows.append({
            'Experiment': label,
            'Configuration': descriptions[label],
            'TP': m['tp'], 'FP': m['fp'], 'FN': m['fn'], 'TN': m['tn'],
            'Precision (%)': round(m['precision'] * 100, 4),
            'Recall (%)': round(m['recall'] * 100, 4),
            'F1-score (%)': round(m['f1'] * 100, 4),
            'Evaluated observations': m['matched'],
            'Positive labels': m['tp'] + m['fn'],
        })
    delta = {
        'tp': d['tp'] - c['tp'], 'fp': d['fp'] - c['fp'],
        'fn': d['fn'] - c['fn'], 'tn': d['tn'] - c['tn'],
        'precision_pp': (d['precision']-c['precision'])*100,
        'recall_pp': (d['recall']-c['recall'])*100,
        'f1_pp': (d['f1']-c['f1'])*100,
    }
    return rows, delta


def append_experiment_d(abc_rows, d_raw):
    """Append validated D to the prior A/B/C comparison (does not mutate it)."""
    if len(abc_rows) != 3 or [r.get('Experiment') for r in abc_rows] != ['A', 'B', 'C']:
        raise Table48Error('Expected Table 4.6 rows in A, B, C order.')
    try:
        d = check_metrics(d_raw, 'D')
    except Table46Error as exc:
        raise Table48Error(str(exc)) from exc
    for row in abc_rows:
        if (row['Evaluated observations'] != d['matched'] or
            row['Positive labels'] != d['tp'] + d['fn'] or
            row['TN'] + row['FP'] != d['tn'] + d['fp']):
            raise Table48Error('Experiment D totals differ from A/B/C. Use the same dataset and labels.')
    result = list(abc_rows)
    result.append({
        'Experiment': 'D',
        'Configuration': 'Original SRGG + suppression + critical-ceiling override',
        'TP': d['tp'], 'FP': d['fp'], 'FN': d['fn'], 'TN': d['tn'],
        'Precision (%)': round(d['precision']*100, 4),
        'Recall (%)': round(d['recall']*100, 4),
        'F1-score (%)': round(d['f1']*100, 4),
        'False-positive rate (%)': round(d['fpr']*100, 6),
        'Evaluated observations': d['matched'],
        'Positive labels': d['tp']+d['fn'],
    })
    return result


_AUDIT_FIELDS = (
    'timestamp', 'usage_pct', 'anomaly_type', 'matched_keyword',
    'C_suppressed', 'D_suppressed', 'D_critical_override_applied'
)


def _bool(value, field, line):
    if not isinstance(value, str) or value.strip().lower() not in ('true', 'false'):
        raise Table48Error(f'Row {line}: {field} must be True or False.')
    return value.strip().lower() == 'true'


def parse_override_audit(csv_bytes, ceiling=85.0):
    """Review *only restored* alerts in the C->D audit export.

    These are already classified relative to synthetic injection labels by the
    audit generator. The CSV alone cannot establish ground truth of the events.
    """
    if not csv_bytes:
        raise Table48Error('Audit CSV is empty.')
    try:
        text = csv_bytes.decode('utf-8-sig')
    except UnicodeError as exc:
        raise Table48Error('Audit must be a UTF-8 CSV file.') from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise Table48Error('Audit CSV has no headers.')
    missing = sorted(set(_AUDIT_FIELDS) - set(reader.fieldnames))
    if missing:
        raise Table48Error('Audit is missing fields: ' + ', '.join(missing))
    try:
        ceiling = float(ceiling)
    except (ValueError, TypeError) as exc:
        raise Table48Error('Critical ceiling must be numeric.') from exc
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise Table48Error('Critical ceiling must be positive and finite.')
    entries, keywords, types, timestamps = [], {}, {}, set()
    for line, source in enumerate(reader, start=2):
        try:
            timestamp = datetime.fromisoformat(source['timestamp'])
            usage = float(source['usage_pct'])
        except (TypeError, ValueError) as exc:
            raise Table48Error(f'Row {line}: invalid timestamp or disk utilization.') from exc
        if timestamp in timestamps:
            raise Table48Error(f'Row {line}: duplicated timestamp.')
        timestamps.add(timestamp)
        if not math.isfinite(usage) or usage < 0 or usage > 100.0001:
            raise Table48Error(f'Row {line}: disk utilization must be between 0 and 100%.')
        if usage + 1e-9 < ceiling:
            raise Table48Error(f'Row {line}: a restored critical alert is below the {ceiling}% ceiling.')
        if not _bool(source['C_suppressed'], 'C_suppressed', line):
            raise Table48Error(f'Row {line}: alert must have been suppressed in C.')
        if _bool(source['D_suppressed'], 'D_suppressed', line):
            raise Table48Error(f'Row {line}: restored alert must not be suppressed in D.')
        if not _bool(source['D_critical_override_applied'], 'D_critical_override_applied', line):
            raise Table48Error(f'Row {line}: missing override marker.')
        keyword = (source['matched_keyword'] or '').strip() or '(none)'
        anomaly_type = (source['anomaly_type'] or '').strip() or '(not specified)'
        keywords[keyword] = keywords.get(keyword, 0)+1
        types[anomaly_type] = types.get(anomaly_type, 0)+1
        entries.append({
            'Timestamp': timestamp.isoformat(sep=' '),
            'Disk usage (%)': usage,
            'Anomaly type (from labels)': anomaly_type,
            'Matched keyword': keyword,
            'C suppressed': True,
            'D suppressed': False,
            'Critical override applied': True,
        })
    if not entries:
        raise Table48Error('Audit CSV has no restored alerts.')
    return entries, {'restored':len(entries), 'keywords':keywords,
                     'types':types, 'critical_ceiling_pct':ceiling}


def verify_audit_against_metrics(delta, audit_summary):
    """Disallow presenting an inconsistent audit as complete proof of C->D changes."""
    restored = audit_summary['restored']
    # One flip from suppressed->active increases either TP or FP for the SAME labels.
    newly_active = delta['tp'] + delta['fp']
    if newly_active != restored:
        raise Table48Error(
            f'Audit lists {restored} restored alert readings, but C->D increases '
            f'total predicted positives by {newly_active}. Check the run pair and audit.')
    return True
