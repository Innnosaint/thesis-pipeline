"""
bench_correlation.py  (Section 4.7 runtime claim)
Times the windowed correlation lookup (bisect, O(log n)) against a naive
full-corpus scan for 200 sampled ABNORMAL alerts, using YOUR log_correlation.py.

Usage: python bench_correlation.py [--parsed ../data/processed/parsed_entries.jsonl]
                                   [--final  ../data/processed/final_scored_stream.jsonl]
                                   [--config ../config/settings.yaml]
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from log_correlation import build_time_index, get_log_window, classify_priority  # noqa: E402


def load(path):
    out = []
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            r["timestamp"] = datetime.fromisoformat(r["timestamp"])
            out.append(r)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--parsed", default="../data/processed/parsed_entries.jsonl")
ap.add_argument("--final", default="../data/processed/final_scored_stream.jsonl")
ap.add_argument("--config", default="../config/settings.yaml")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
dt = cfg["correlation"]["delta_t_seconds"]
ben, fail = cfg["correlation"]["benign_keywords"], cfg["correlation"]["failure_keywords"]

corpus = load(a.parsed)
corpus.sort(key=lambda e: e["timestamp"])
entries, ts = build_time_index(corpus)
alerts = [e["timestamp"] for e in load(a.final) if e.get("classification") == "ABNORMAL"]
random.Random(a.seed).shuffle(alerts)
sample = alerts[:a.n] or [e["timestamp"] for e in corpus[:a.n]]
print(f"corpus: {len(corpus):,} log lines   sampled alerts: {len(sample)}   window: +/-{dt}s")

REPS = 200
t0 = time.perf_counter()
for _ in range(REPS):
    for t in sample:
        classify_priority(get_log_window(entries, ts, t, dt), ben, fail)
fast_ms = (time.perf_counter() - t0) / (REPS * len(sample)) * 1000

t0 = time.perf_counter()
for t in sample:
    win = [e for e in corpus if abs((e["timestamp"] - t).total_seconds()) <= dt]
    classify_priority(win, ben, fail)
slow_ms = (time.perf_counter() - t0) / len(sample) * 1000

print(f"windowed lookup : {fast_ms:.4f} ms per alert")
print(f"full-corpus scan: {slow_ms:.2f} ms per alert")
print(f"speedup         : about {slow_ms / fast_ms:,.0f}x")
print("paper: 0.006 ms vs 37.8 ms, about 6,000x (129,043 lines, +/-60 s). Absolute times depend on your CPU; the ratio should be of the same order.")
