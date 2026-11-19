"""Validate label-verified episode results for Chapter Four, Table 4.7.

No data are invented or filled in: callers must upload a previously generated
CSV derived from aligned ground-truth and experiment A/B/C streams.
"""
import csv
import io
from datetime import datetime
from statistics import mean

REQUIRED = (
    'event_id', 'labeled_onset', 'end', 'labeled_readings',
    'existing_first_labeled_alert', 'existing_delay_min',
    'original_srgg_first_labeled_alert', 'original_srgg_delay_min',
    'improvement_in_label_confirmed_delay_min',
    'SRGG_first_unlabeled_preonset_alert', 'preonset_advance_min',
    'preonset_false_positive_readings', 'existing_true_positive_readings',
    'srgg_true_positive_readings', 'full_pipeline_true_positive_readings',
)


class Table47Error(ValueError):
    pass


def _date(value, label):
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise Table47Error(f'{label}: expected ISO 8601 timestamp, got {value!r}') from exc


def _integer(value, label):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise Table47Error(f'{label}: expected integer, got {value!r}') from exc
    if number < 0 or str(number) != str(value).strip():
        raise Table47Error(f'{label}: expected nonnegative integer, got {value!r}')
    return number


def parse_table47(csv_bytes):
    """Return (validated rows, summary), refusing mismatched onset/delay data.

    The function checks arithmetic and chronology, not the scientific provenance.
    The provenance must be confirmed from scored streams and labels separately.
    """
    if not csv_bytes:
        raise Table47Error('CSV upload is empty.')
    try:
        decoded = csv_bytes.decode('utf-8-sig')
    except UnicodeError as exc:
        raise Table47Error('Upload a UTF-8 CSV file.') from exc
    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise Table47Error('CSV header is missing.')
    missing = sorted(set(REQUIRED) - set(reader.fieldnames))
    if missing:
        raise Table47Error('Missing required columns: ' + ', '.join(missing))
    out = []
    seen_events = set()
    for index, raw in enumerate(reader, 1):
        if not any((v or '').strip() for v in raw.values() if isinstance(v, str)):
            continue
        eid = _integer(raw['event_id'], f'row {index} event_id')
        if eid in seen_events:
            raise Table47Error(f'Duplicate event_id {eid}')
        seen_events.add(eid)
        onset = _date(raw['labeled_onset'], f'row {index} labeled_onset')
        end = _date(raw['end'], f'row {index} end')
        if end < onset:
            raise Table47Error(f'row {index}: episode ends before onset')
        a = _date(raw['existing_first_labeled_alert'], f'row {index} existing_first_labeled_alert')
        b = _date(raw['original_srgg_first_labeled_alert'], f'row {index} original_srgg_first_labeled_alert')
        if not (onset <= a <= end and onset <= b <= end):
            raise Table47Error(f'row {index}: labeled alerts must fall inside the labeled episode')
        a_delay = _integer(raw['existing_delay_min'], f'row {index} existing_delay_min')
        b_delay = _integer(raw['original_srgg_delay_min'], f'row {index} original_srgg_delay_min')
        improvement = int(raw['improvement_in_label_confirmed_delay_min'])
        if abs((a - onset).total_seconds() / 60 - a_delay) > 0.001:
            raise Table47Error(f'row {index}: existing_delay_min disagrees with timestamps')
        if abs((b - onset).total_seconds() / 60 - b_delay) > 0.001:
            raise Table47Error(f'row {index}: original_srgg_delay_min disagrees with timestamps')
        if improvement != a_delay - b_delay:
            raise Table47Error(f'row {index}: delay improvement must equal existing minus SRGG delay')
        advance = _integer(raw['preonset_advance_min'], f'row {index} preonset_advance_min')
        fp = _integer(raw['preonset_false_positive_readings'], f'row {index} preonset_false_positive_readings')
        early_s = (raw['SRGG_first_unlabeled_preonset_alert'] or '').strip()
        if early_s:
            early_dt = _date(early_s, f'row {index} SRGG_first_unlabeled_preonset_alert')
            if early_dt >= onset or abs((onset - early_dt).total_seconds()/60 - advance) > 0.001:
                raise Table47Error(f'row {index}: pre-onset alert timing disagrees with advance')
        elif advance or fp:
            raise Table47Error(f'row {index}: missing pre-onset timestamp for nonzero pre-onset values')
        labeled = _integer(raw['labeled_readings'], f'row {index} labeled_readings')
        if labeled < 1:
            raise Table47Error(f'row {index}: labeled_readings must be at least 1')
        captures = []
        for column in ('existing_true_positive_readings','srgg_true_positive_readings','full_pipeline_true_positive_readings'):
            count = _integer(raw[column], f'row {index} {column}')
            if count > labeled:
                raise Table47Error(f'row {index}: {column} cannot exceed labeled_readings')
            captures.append(count)
        out.append({
            'Episode': f'{eid} — {onset.strftime("%b %d, %Y")}',
            'Labeled onset': onset.isoformat(sep=' '),
            'Labeled readings': labeled,
            'Existing delay (min)': a_delay,
            'SRGG delay (min)': b_delay,
            'Earlier detection (min)': improvement,
            'Pre-onset unlabeled alerts': fp,
            'Pre-onset advance (min)': advance,
            'A detected readings': captures[0],
            'B detected readings': captures[1],
            'C detected readings': captures[2],
        })
    if not out:
        raise Table47Error('CSV contains no labeled episodes.')
    stats = {
        'episodes': len(out),
        'mean_existing_delay_min': mean(r['Existing delay (min)'] for r in out),
        'mean_srgg_delay_min': mean(r['SRGG delay (min)'] for r in out),
        'mean_improvement_min': mean(r['Earlier detection (min)'] for r in out),
        'total_preonset_false_positives': sum(r['Pre-onset unlabeled alerts'] for r in out),
        'total_labeled_readings': sum(r['Labeled readings'] for r in out),
        'total_A_detected': sum(r['A detected readings'] for r in out),
        'total_B_detected': sum(r['B detected readings'] for r in out),
        'total_C_detected': sum(r['C detected readings'] for r in out),
    }
    return out, stats

STRESS_REQUIRED = (
    'scenario','method','observations_evaluated','ground_truth',
    'false_alert_observations','false_alerts_per_day',
    'SRGG_only_false_alerts','days','seed',
)


def parse_benign_stress(csv_bytes):
    """Validate synthetic normal-workload stress results for display only."""
    if not csv_bytes:
        raise Table47Error('Empty stress test CSV.')
    try:
        csv_text=csv_bytes.decode('utf-8-sig')
    except UnicodeError as exc:
        raise Table47Error('Stress test CSV must be UTF-8.') from exc
    reader=csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise Table47Error('Stress test CSV has no header.')
    missing=set(STRESS_REQUIRED)-set(reader.fieldnames)
    if missing:
        raise Table47Error('Missing stress CSV columns: '+', '.join(sorted(missing)))
    records=[]
    for index,row in enumerate(reader,1):
        if row['ground_truth']!='all_normal_by_test_design':
            raise Table47Error(f'row {index}: only all-normal stress scenarios are supported')
        evaluated=_integer(row['observations_evaluated'],f'row {index} evaluated')
        fp=_integer(row['false_alert_observations'],f'row {index} false_alert_observations')
        srgg=_integer(row['SRGG_only_false_alerts'],f'row {index} SRGG_only_false_alerts')
        days=_integer(row['days'],f'row {index} days')
        if fp>evaluated or srgg>fp or days<=0:
            raise Table47Error(f'row {index}: inconsistent false-alert totals')
        try:rate=float(row['false_alerts_per_day'])
        except (ValueError,TypeError) as exc:raise Table47Error(f'row {index}: invalid false_alerts_per_day') from exc
        if abs(rate-(fp/days)) > 0.011:
            raise Table47Error(f'row {index}: false_alerts_per_day disagrees with count/days')
        records.append({'Scenario':row['scenario'],'Method':row['method'],
                        'False-alert observations':fp,'False-alerts per day':rate,
                        'SRGG-only false alerts':srgg,'Evaluated observations':evaluated})
    if not records:
        raise Table47Error('Stress CSV has no rows.')
    return records
