"""
generate_synthetic_disk_usage.py

Generates a synthetic disk usage percentage time series aligned to the
real timestamps found in a raw Linux syslog file, and interleaves it as
realistic 'diskmon' daemon log lines. Produces:

  1. An augmented raw log (real lines + synthetic diskmon lines, sorted
     chronologically) that can be fed directly into the existing
     ingest.py -> Drain3 -> preprocess.py -> baseline -> Z-Score pipeline.

  2. A ground truth labels.jsonl file marking exactly which synthetic
     entries are anomalous, matching the schema expected by
     replay_simulator.py::load_ground_truth().

WHY THIS EXISTS:
Real syslog/kernel logs (like the Loghub-2.0 Linux dataset) record
*events* (mounts, device detection, SMART status) rather than *metrics*
(disk usage percentages). Since the enhanced algorithm's baseline and
Z-Score computations require a continuous numeric metric stream, this
script synthesizes that stream while preserving real log timestamps and
real log context, so Contextual Log Correlation still operates against
genuine system activity.

Usage:
    python generate_synthetic_disk_usage.py \
        --input ../data/raw/Linux_full.log \
        --output ../data/raw/Linux_full_augmented.log \
        --labels ../data/ground_truth/labels.jsonl \
        --interval-minutes 5 \
        --fallback-year 2004
"""

import argparse
import json
import random
import re
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# CONFIGURATION (tune these to change the simulated disk behavior)
# ---------------------------------------------------------------------------

BASELINE_MEAN = 45.0          # center of normal disk usage (%)
BASELINE_AMPLITUDE = 8.0      # daily circadian swing (%)
BASELINE_NOISE_STD = 2.5      # random noise added each reading (%)

SPIKE_COUNT = 12              # number of sudden-spike events to inject
SPIKE_MIN, SPIKE_MAX = 90, 98 # spike magnitude range (%)
SPIKE_DURATION_READINGS = 2   # how many consecutive readings stay spiked

DEGRADATION_EVENTS = 3        # number of gradual-degradation episodes
DEGRADATION_HOURS = 6         # how long each degradation ramp lasts
DEGRADATION_TARGET = 88.0     # peak usage reached at the end of the ramp

BENIGN_KEYWORDS = ["backup", "rsync", "cron", "apt", "yum",
                    "package install", "scheduled task"]
FAILURE_KEYWORDS = ["disk full", "I/O error", "write failure",
                     "inode exhausted", "filesystem read-only",
                     "device error", "no space left"]

DEVICE = "/dev/hda1"
DAEMON_PID = 2048

# Matches classic syslog format: "Jun  9 06:06:20 combo kernel: message"
SYSLOG_TS_RE = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s"
)

# ---------------------------------------------------------------------------


def parse_real_log_timestamps(path: str, fallback_year: int):
    """Parse the raw log and return (min_dt, max_dt, hostname, raw_lines)."""
    raw_lines = []
    timestamps = []
    hostname = "combo"

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            match = SYSLOG_TS_RE.match(line)
            if match:
                ts_str = match.group("ts")
                hostname = match.group("host")
                try:
                    dt = datetime.strptime(
                        f"{fallback_year} {ts_str}", "%Y %b %d %H:%M:%S"
                    )
                    timestamps.append(dt)
                except ValueError:
                    pass
            raw_lines.append(line)

    if not timestamps:
        raise RuntimeError(
            "No parseable timestamps found in the input log. "
            "Check that --fallback-year matches the dataset."
        )

    return min(timestamps), max(timestamps), hostname, raw_lines


def circadian_baseline(dt: datetime, start: datetime) -> float:
    """Smooth daily wave, e.g. lower usage at night, higher in daytime."""
    hours_elapsed = (dt - start).total_seconds() / 3600.0
    phase = (hours_elapsed % 24) / 24.0 * 2 * 3.14159265
    import math
    wave = math.sin(phase - 1.2)  # peak mid-afternoon
    return BASELINE_MEAN + BASELINE_AMPLITUDE * wave


def generate_synthetic_series(start: datetime, end: datetime,
                               interval_minutes: int, seed: int = 42):
    """
    Build a list of (datetime, usage_pct, is_anomaly, anomaly_type,
    context_keyword_or_None) tuples covering the full time range.
    """
    random.seed(seed)

    timestamps = []
    t = start
    step = timedelta(minutes=interval_minutes)
    while t <= end:
        timestamps.append(t)
        t += step

    n = len(timestamps)
    series = [
        {
            "dt": ts,
            "usage": round(
                max(0.0, min(100.0,
                    circadian_baseline(ts, start)
                    + random.gauss(0, BASELINE_NOISE_STD))),
                1,
            ),
            "is_anomaly": False,
            "anomaly_type": None,
            "keyword": None,
        }
        for ts in timestamps
    ]

    # --- Inject sudden spikes ---
    spike_indices = random.sample(
        range(SPIKE_DURATION_READINGS, n - SPIKE_DURATION_READINGS),
        min(SPIKE_COUNT, max(0, n - 2 * SPIKE_DURATION_READINGS)),
    )
    for idx in spike_indices:
        magnitude = random.uniform(SPIKE_MIN, SPIKE_MAX)
        # roughly half the spikes are benign (paired with benign keyword),
        # the rest are genuine failures (paired with failure keyword)
        is_benign = random.random() < 0.5
        keyword = (random.choice(BENIGN_KEYWORDS) if is_benign
                   else random.choice(FAILURE_KEYWORDS))
        for offset in range(SPIKE_DURATION_READINGS):
            i = idx + offset
            if i >= n:
                continue
            series[i]["usage"] = round(magnitude + random.gauss(0, 1.0), 1)
            series[i]["is_anomaly"] = True
            series[i]["anomaly_type"] = "sudden_spike"
            series[i]["keyword"] = keyword

    # --- Inject gradual degradation episodes ---
    readings_per_hour = max(1, 60 // interval_minutes)
    ramp_len = DEGRADATION_HOURS * readings_per_hour
    if ramp_len < n:
        possible_starts = range(0, n - ramp_len)
        degradation_starts = random.sample(
            list(possible_starts), min(DEGRADATION_EVENTS, len(list(possible_starts)))
        ) if len(list(possible_starts)) >= DEGRADATION_EVENTS else []

        for start_idx in degradation_starts:
            base_usage = series[start_idx]["usage"]
            for step_i in range(ramp_len):
                i = start_idx + step_i
                if i >= n:
                    break
                progress = step_i / ramp_len
                target = base_usage + (DEGRADATION_TARGET - base_usage) * progress
                series[i]["usage"] = round(target + random.gauss(0, 1.0), 1)
                # Mark as anomaly only once it departs meaningfully from
                # baseline (progress > 0.35), so early ramp still looks
                # "normal-ish" -- this is what tests whether Z-Score catches
                # drift earlier than a static 80% threshold would.
                if progress > 0.35:
                    series[i]["is_anomaly"] = True
                    series[i]["anomaly_type"] = "gradual_degradation"
                    series[i]["keyword"] = None  # no clean signature; genuine

    return series


def format_diskmon_line(entry, hostname: str) -> str:
    """Render one synthetic reading as a syslog-style diskmon log line."""
    ts_str = entry["dt"].strftime("%b %e %H:%M:%S")
    usage = entry["usage"]

    if entry["keyword"]:
        message = (f"diskmon[{DAEMON_PID}]: {DEVICE} {usage:.1f}% used "
                    f"(context: {entry['keyword']})")
    else:
        message = f"diskmon[{DAEMON_PID}]: {DEVICE} {usage:.1f}% used"

    return f"{ts_str} {hostname} {message}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                         help="Path to the raw real log file")
    parser.add_argument("--output", required=True,
                         help="Path to write the augmented log")
    parser.add_argument("--labels", required=True,
                         help="Path to write ground_truth labels.jsonl")
    parser.add_argument("--interval-minutes", type=int, default=5,
                         help="Sampling interval for synthetic readings")
    parser.add_argument("--fallback-year", type=int, default=2004,
                         help="Year to assume for year-less timestamps")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Reading real log from {args.input} ...")
    start, end, hostname, raw_lines = parse_real_log_timestamps(
        args.input, args.fallback_year
    )
    print(f"Real log spans {start} to {end} (host: {hostname})")

    print(f"Generating synthetic disk usage series "
          f"(every {args.interval_minutes} min) ...")
    series = generate_synthetic_series(
        start, end, args.interval_minutes, seed=args.seed
    )
    anomaly_count = sum(1 for e in series if e["is_anomaly"])
    print(f"Generated {len(series)} synthetic readings "
          f"({anomaly_count} marked anomalous).")

    synthetic_lines = [format_diskmon_line(e, hostname) for e in series]

    # Merge and sort chronologically. Real lines don't carry a parsed dt
    # object here, so we re-derive a sortable key from their prefix.
    def sort_key_real(line):
        m = SYSLOG_TS_RE.match(line)
        if not m:
            return datetime.max
        try:
            return datetime.strptime(
                f"{args.fallback_year} {m.group('ts')}", "%Y %b %d %H:%M:%S"
            )
        except ValueError:
            return datetime.max

    combined = [(sort_key_real(l), l) for l in raw_lines]
    combined += [(e["dt"], sl) for e, sl in zip(series, synthetic_lines)]
    combined.sort(key=lambda pair: pair[0])

    print(f"Writing augmented log to {args.output} ...")
    with open(args.output, "w") as f:
        for _, line in combined:
            f.write(line + "\n")

    print(f"Writing ground truth labels to {args.labels} ...")
    with open(args.labels, "w") as f:
        for e in series:
            record = {
                "log_source": "linux_diskmon",
                "timestamp": e["dt"].isoformat(),
                "is_anomaly": e["is_anomaly"],
                "anomaly_type": e["anomaly_type"],
            }
            f.write(json.dumps(record) + "\n")

    print("Done.")
    print(f"  Total lines written (real + synthetic): {len(combined)}")
    print(f"  Synthetic readings: {len(series)} "
          f"({anomaly_count} anomalous, "
          f"{len(series) - anomaly_count} normal)")


if __name__ == "__main__":
    main()
