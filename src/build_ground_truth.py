"""
build_ground_truth.py
Objective 5 support: Constructs ground-truth anomaly labels for the real
Loghub Linux diskmon dataset using a heuristic INDEPENDENT of the detection
algorithm under evaluation (no Z-Score, no dynamic baseline, no sliding
window is used here) — avoiding circular evaluation.

A diskmon reading is labeled TRUE ANOMALY if EITHER:
  (a) the raw disk usage percentage itself crosses a fixed domain-expert
      critical threshold (default: >= 85%), OR
  (b) a genuine failure-signature log line (e.g. "disk full", "I/O error",
      "inode exhausted", "filesystem read-only", "no space left") occurs
      within a fixed time window around that reading, scanned directly
      from the raw log file text — not from the algorithm's own
      Contextual Log Correlation module.

This mirrors how prior work constructs ground truth when no official
labels exist (e.g., Xu et al., 2009; Oliner & Stearley, 2007) and should
be cited/described in the thesis Methodology as an independent labeling
procedure, separate from the algorithm being evaluated.
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timedelta


DISKMON_PATTERN = re.compile(
    r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+diskmon\[\d+\]:\s*"
    r"\S+\s+(\d{1,3}(?:\.\d+)?)\s*%\s*used",
    re.IGNORECASE,
)

TIMESTAMP_PATTERN = re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")

DEFAULT_FAILURE_KEYWORDS = [
    "disk full",
    "i/o error",
    "write failure",
    "inode exhausted",
    "filesystem read-only",
    "device error",
    "no space left",
    "disk quota",
]


def parse_line_timestamp(line: str, fallback_year: int):
    m = TIMESTAMP_PATTERN.search(line)
    if not m:
        return None
    raw = m.group(1)
    try:
        return datetime.strptime(f"{fallback_year} {raw}", "%Y %b %d %H:%M:%S")
    except ValueError:
        return None


def load_raw_lines_with_timestamps(log_path: str, fallback_year: int):
    """Returns a list of (timestamp, raw_line) tuples, sorted by time,
    for ALL lines in the file (not just diskmon lines) — needed so we
    can scan for failure keywords in the surrounding context."""
    entries = []
    with open(log_path, errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            ts = parse_line_timestamp(line, fallback_year)
            if ts:
                entries.append((ts, line))
    entries.sort(key=lambda x: x[0])
    return entries


def extract_diskmon_readings(all_entries, log_source_name: str):
    """Filters the full entry list down to just diskmon readings,
    extracting the numeric disk usage value for each."""
    readings = []
    for ts, line in all_entries:
        m = DISKMON_PATTERN.search(line)
        if m:
            value = float(m.group(3))
            readings.append({
                "timestamp": ts,
                "disk_usage_value": value,
                "log_source": log_source_name,
                "raw": line,
            })
    return readings


def has_failure_signature_nearby(all_entries, center_time, delta_t_seconds, failure_keywords):
    """Scans the raw line text (not the diskmon line itself) within
    +/- delta_t of center_time for any failure keyword. Independent of
    the detection algorithm's own correlation module."""
    lo = center_time - timedelta(seconds=delta_t_seconds)
    hi = center_time + timedelta(seconds=delta_t_seconds)

    for ts, line in all_entries:
        if ts < lo:
            continue
        if ts > hi:
            break
        line_lower = line.lower()
        for kw in failure_keywords:
            if kw.lower() in line_lower:
                return True, kw
    return False, None


def build_labels(
    log_path: str,
    output_path: str,
    log_source_name: str = None,
    critical_pct: float = 85.0,
    delta_t_seconds: int = 60,
    failure_keywords=None,
    fallback_year: int = 2004,
):
    if failure_keywords is None:
        failure_keywords = DEFAULT_FAILURE_KEYWORDS
    if log_source_name is None:
        log_source_name = Path(log_path).stem

    print(f"Loading and timestamp-parsing all lines from {log_path} ...")
    all_entries = load_raw_lines_with_timestamps(log_path, fallback_year)
    print(f"  {len(all_entries)} timestamped lines loaded.")

    print("Extracting diskmon readings ...")
    readings = extract_diskmon_readings(all_entries, log_source_name)
    print(f"  {len(readings)} diskmon readings found.")

    print(f"Labeling: critical_pct >= {critical_pct}% OR failure keyword within +/-{delta_t_seconds}s ...")
    n_by_pct = 0
    n_by_keyword = 0
    n_anomaly = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for r in readings:
            reason = None
            is_anomaly = False

            if r["disk_usage_value"] >= critical_pct:
                is_anomaly = True
                reason = "threshold"
                n_by_pct += 1
            else:
                found, kw = has_failure_signature_nearby(
                    all_entries, r["timestamp"], delta_t_seconds, failure_keywords
                )
                if found:
                    is_anomaly = True
                    reason = f"keyword:{kw}"
                    n_by_keyword += 1

            if is_anomaly:
                n_anomaly += 1

            record = {
                "log_source": r["log_source"],
                "timestamp": r["timestamp"].isoformat(),
                "is_anomaly": is_anomaly,
                "anomaly_type": reason,
            }
            out.write(json.dumps(record) + "\n")

    print(f"\nDone. Wrote {len(readings)} labels to {output_path}")
    print(f"  Anomalies flagged: {n_anomaly} ({n_anomaly / len(readings) * 100:.2f}%)")
    print(f"    - by critical threshold (>= {critical_pct}%): {n_by_pct}")
    print(f"    - by nearby failure keyword: {n_by_keyword}")
    print(f"  Normal: {len(readings) - n_anomaly}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build independent ground-truth labels for diskmon readings."
    )
    parser.add_argument("--log", required=True, help="Path to raw log file (e.g., Linux_full.log)")
    parser.add_argument("--output", required=True, help="Path to write labels.jsonl")
    parser.add_argument("--log-source-name", default=None,
                         help="Value to use for log_source field (default: filename stem)")
    parser.add_argument("--critical-pct", type=float, default=85.0,
                         help="Disk usage %% at or above which a reading is auto-labeled anomalous")
    parser.add_argument("--delta-t", type=int, default=60,
                         help="Seconds +/- around a reading to scan for failure keywords")
    parser.add_argument("--fallback-year", type=int, default=2004,
                         help="Year to assume for logs with no year in timestamp")
    args = parser.parse_args()

    build_labels(
        log_path=args.log,
        output_path=args.output,
        log_source_name=args.log_source_name,
        critical_pct=args.critical_pct,
        delta_t_seconds=args.delta_t,
        fallback_year=args.fallback_year,
    )
