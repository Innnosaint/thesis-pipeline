"""
log_correlation.py
Objective 4 support: Contextual Log Correlation (Equations 3.6-3.7).
For every ABNORMAL observation, retrieves concurrent log entries within a
+/- delta_t window and applies keyword matching to classify priority,
avoiding full cross-source regex matching for computational efficiency.
"""

from bisect import bisect_left, bisect_right


def build_time_index(all_entries):
    """all_entries must be sorted by timestamp. Returns the same list plus
    a parallel list of timestamps for fast bisect-based window lookups."""
    timestamps = [e["timestamp"] for e in all_entries]
    return all_entries, timestamps


def get_log_window(all_entries, timestamps, center_time, delta_t_seconds):
    from datetime import timedelta

    lo_time = center_time - timedelta(seconds=delta_t_seconds)
    hi_time = center_time + timedelta(seconds=delta_t_seconds)

    lo_idx = bisect_left(timestamps, lo_time)
    hi_idx = bisect_right(timestamps, hi_time)

    return all_entries[lo_idx:hi_idx]


def classify_priority(log_window, benign_keywords, failure_keywords):
    """
    Scans raw text of all entries in the window for keyword matches.
    Failure keywords take precedence over benign (a real failure signature
    should never be suppressed just because a backup ran nearby).
    Returns (priority, matched_keyword_or_None).
    """
    text_blob = " ".join(e["raw"].lower() for e in log_window)

    for kw in failure_keywords:
        if kw.lower() in text_blob:
            return "HIGH", kw

    for kw in benign_keywords:
        if kw.lower() in text_blob:
            return "LOW", kw

    return "MEDIUM", None


def correlate_and_score(classified_entries, all_entries_for_context, cfg):
    """
    For each ABNORMAL entry, run contextual correlation and compute the
    final Enhanced Alert Score (Equation 3.8). NORMAL entries pass through
    unchanged with priority=None.
    """
    delta_t = cfg["correlation"]["delta_t_seconds"]
    benign_kw = cfg["correlation"]["benign_keywords"]
    failure_kw = cfg["correlation"]["failure_keywords"]
    weights = cfg["priority_weights"]

    all_entries_sorted, timestamps = build_time_index(all_entries_for_context)

    for e in classified_entries:
        if e["classification"] != "ABNORMAL":
            e["priority"] = None
            e["matched_keyword"] = None
            e["enhanced_alert_score"] = 0.0
            e["suppressed"] = False
            continue

        window = get_log_window(all_entries_sorted, timestamps, e["timestamp"], delta_t)
        priority, matched_kw = classify_priority(window, benign_kw, failure_kw)

        e["priority"] = priority
        e["matched_keyword"] = matched_kw
        e["enhanced_alert_score"] = e["zscore"] * weights[priority]
        e["suppressed"] = (priority == "LOW")

    return classified_entries


if __name__ == "__main__":
    import json
    import yaml
    from pathlib import Path
    from datetime import datetime

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    in_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "classified_stream.jsonl"
    entries = []
    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)

    # For context lookups we use the same stream (already time-sorted).
    scored = correlate_and_score(entries, entries, cfg)

    n_suppressed = sum(1 for e in scored if e.get("suppressed"))
    n_escalated = sum(1 for e in scored if e.get("priority") == "HIGH")
    print(f"Correlated {len(scored)} entries: {n_suppressed} suppressed, {n_escalated} escalated.")

    out_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "scored_stream.jsonl"
    with open(out_path, "w") as f:
        for e in scored:
            rec = dict(e)
            rec["timestamp"] = rec["timestamp"].isoformat()
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote final scored stream to {out_path}")
