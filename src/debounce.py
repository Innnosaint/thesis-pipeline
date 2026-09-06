"""
debounce.py
Step 4.5 (new): Consecutive-Readings Debounce Filter.

Addresses the false-positive pattern diagnosed in diagnose_false_positives.py,
where isolated single-reading noise (one reading briefly deviating from a
low-variance baseline) was being classified ABNORMAL despite representing no
sustained or genuine disk usage problem.

Requires N consecutive ABNORMAL classifications (default 2) before a reading
is allowed to remain ABNORMAL; isolated single-reading spikes are reclassified
back to NORMAL. This does not touch the Z-Score computation itself -- it is
a post-classification filter, applied after zscore_detector.classify() and
before log_correlation.correlate_and_score(), consistent with keeping the
core Z-Score math (Equation 3.4-3.5) unchanged.

Genuine sustained anomalies (multiple consecutive high-Z-score readings, as
seen in real disk-full/degradation events) are unaffected, since they satisfy
the consecutive-count requirement naturally.
"""


def apply_debounce(entries, min_consecutive: int = 2):
    """
    entries: time-ordered list of dicts, each with a 'classification' field
    ('ABNORMAL' or 'NORMAL'), as produced by zscore_detector.classify().

    Mutates and returns the same list: any ABNORMAL entry that is not part
    of a run of at least `min_consecutive` consecutive ABNORMAL entries is
    reclassified to NORMAL (severity_score and zscore are left untouched
    for audit/debugging purposes; only 'classification' and a new
    'debounced' flag are added).

    min_consecutive=1 disables debouncing entirely (every ABNORMAL entry
    passes through unchanged), useful for A/B comparison in Chapter 4.
    """
    if min_consecutive <= 1:
        for e in entries:
            e["debounced"] = False
        return entries

    n = len(entries)
    i = 0
    while i < n:
        if entries[i]["classification"] == "ABNORMAL":
            # find the full run of consecutive ABNORMAL entries starting at i
            j = i
            while j < n and entries[j]["classification"] == "ABNORMAL":
                j += 1
            run_length = j - i

            if run_length < min_consecutive:
                # isolated spike (or short run) -- reclassify back to NORMAL
                for k in range(i, j):
                    entries[k]["classification"] = "NORMAL"
                    entries[k]["debounced"] = True
            else:
                for k in range(i, j):
                    entries[k]["debounced"] = False

            i = j
        else:
            entries[i]["debounced"] = False
            i += 1

    return entries


if __name__ == "__main__":
    import json
    import sys
    import yaml
    from pathlib import Path
    from datetime import datetime

    sys.path.insert(0, str(Path(__file__).parent))
    from zscore_detector import classify

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    in_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "baseline_stream.jsonl"
    entries = []
    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)

    classified = classify(entries, theta_z=cfg["zscore"]["theta_z"])
    min_consecutive = cfg.get("debounce", {}).get("min_consecutive", 2)
    debounced = apply_debounce(classified, min_consecutive=min_consecutive)

    n_before = sum(1 for e in entries if e.get("classification") == "ABNORMAL")
    n_after = sum(1 for e in debounced if e["classification"] == "ABNORMAL")
    print(f"Before debounce: {n_before} ABNORMAL")
    print(f"After debounce (min_consecutive={min_consecutive}): {n_after} ABNORMAL")
    print(f"Suppressed as isolated noise: {sum(1 for e in debounced if e.get('debounced'))}")

    out_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "debounced_stream.jsonl"
    with open(out_path, "w") as f:
        for e in debounced:
            rec = dict(e)
            rec["timestamp"] = rec["timestamp"].isoformat()
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote debounced stream to {out_path}")
