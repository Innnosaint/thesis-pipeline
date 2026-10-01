"""
preprocess.py
Objective 1 & 2 support: Filter, deduplicate, and time-normalize parsed log
entries; segment into a clean time-ordered stream ready for sliding window
baseline computation.
"""

from datetime import timedelta


def filter_disk_related(entries, require_disk_value: bool = False):
    """
    Keep entries relevant to disk/filesystem monitoring.
    If require_disk_value is True, only entries with a numeric disk_usage_value
    survive (used for baseline/Z-Score); otherwise all entries are kept
    (needed later for keyword-based contextual correlation).
    """
    if require_disk_value:
        return [e for e in entries if e["disk_usage_value"] is not None]
    return entries


def remove_duplicates(entries, time_tolerance_seconds: int = 1):
    """
    Removes near-duplicate log entries (e.g., log rotation artifacts, repeated
    daemon output) — same event_type + same disk_usage_value within a small
    time tolerance is considered a duplicate.
    """
    cleaned = []
    last_seen = {}

    for e in entries:
        key = (e["log_source"], e["event_type"], e["disk_usage_value"])
        prev_time = last_seen.get(key)
        if prev_time and abs((e["timestamp"] - prev_time).total_seconds()) <= time_tolerance_seconds:
            continue
        last_seen[key] = e["timestamp"]
        cleaned.append(e)

    return cleaned


def segment_stream(entries):
    """
    Confirms entries are strictly time-ordered and assigns a sequential index.
    This is the structured input handed to the Dynamic Baseline module.
    """
    entries = sorted(entries, key=lambda e: e["timestamp"])
    for i, e in enumerate(entries):
        e["sequence_index"] = i
    return entries


def preprocess_pipeline(entries, require_disk_value_for_baseline=True, dedup_tolerance=1):
    """Full preprocessing chain: filter -> dedup -> segment."""
    disk_entries = filter_disk_related(entries, require_disk_value=require_disk_value_for_baseline)
    disk_entries = remove_duplicates(disk_entries, time_tolerance_seconds=dedup_tolerance)
    disk_entries = segment_stream(disk_entries)
    return disk_entries


if __name__ == "__main__":
    import json
    import yaml
    from pathlib import Path
    from datetime import datetime

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    in_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "parsed_entries.jsonl"
    entries = []
    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)

    clean = preprocess_pipeline(entries)
    print(f"Preprocessed down to {len(clean)} disk-usage-bearing entries (from {len(entries)}).")

    out_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "clean_stream.jsonl"
    with open(out_path, "w") as f:
        for e in clean:
            rec = dict(e)
            rec["timestamp"] = rec["timestamp"].isoformat()
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote cleaned stream to {out_path}")
