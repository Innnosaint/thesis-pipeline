"""
replay_simulator.py  (Section 3.6.2 -- real-time replay)
Replays the cleaned disk-usage stream one entry at a time, in timestamp order,
optionally paced at accelerated real time, running baseline -> Z-Score (+ hybrid
ceiling) -> correlation -> live rolling evaluation with NO lookahead:
the baseline is read before the current reading is added, and correlation only
sees log lines that have already "arrived" (timestamp <= current reading).

Usage:
  python replay_simulator.py                 # uses config/settings.yaml
  python replay_simulator.py --speed 3600    # 1 hour of log time per real second
"""
import argparse
import sys
import time
from bisect import bisect_left
from datetime import timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from baseline import DynamicBaseline
from zscore_detector import classify
from log_correlation import classify_priority
from alert_scoring import write_audit_log
from evaluation_harness import RollingEvaluator
from common import load_stream, load_labels

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def replay(clean_path, context_path, cfg, labels, speed=None, max_lines=None):
    entries = sorted(load_stream(clean_path), key=lambda e: e["timestamp"])
    if max_lines:
        entries = entries[:max_lines]

    # Context lines (all log lines if available, else the disk stream itself).
    ctx = sorted(load_stream(context_path), key=lambda e: e["timestamp"]) \
        if context_path and Path(context_path).exists() else entries
    ctx_ts = [c["timestamp"] for c in ctx]

    speed = cfg["replay"]["speed_multiplier"] if speed is None else speed
    theta = cfg["zscore"]["theta_z"]
    ceiling = cfg["zscore"].get("hard_ceiling")
    delta_t = cfg["correlation"]["delta_t_seconds"]
    benign = cfg["correlation"]["benign_keywords"]
    failure = cfg["correlation"]["failure_keywords"]
    weights = cfg["priority_weights"]
    win = cfg["evaluation"]["rolling_window_for_metrics"]

    baseline = DynamicBaseline(window_size=cfg["baseline"]["window_size"],
                               min_observations=cfg["baseline"]["min_observations"],
                               sigma_floor=cfg["baseline"].get("sigma_floor", 0.0))
    evaluator = RollingEvaluator(rolling_window=win)
    processed, prev_ts = [], None
    print(f"Replaying {len(entries)} entries (speed_multiplier={speed}, "
          f"theta_z={theta}, ceiling={ceiling}) ...\n")

    for e in entries:
        if prev_ts is not None and speed > 0:
            gap = (e["timestamp"] - prev_ts).total_seconds() / speed
            if gap > 0:
                time.sleep(min(gap, 2.0))
        prev_ts = e["timestamp"]

        # Causal baseline: computed BEFORE adding this observation.
        mu, sigma = baseline.get_baseline()
        e.update(baseline_mu=mu, baseline_sigma=sigma, baseline_ready=baseline.is_ready())
        baseline.update(e["disk_usage_value"])

        classify([e], theta_z=theta, hard_ceiling=ceiling)  # same rule as batch mode

        if e["classification"] == "ABNORMAL":
            # Only lines that have already arrived: ts in [t - dt, t]
            lo = bisect_left(ctx_ts, e["timestamp"] - timedelta(seconds=delta_t))
            hi = bisect_left(ctx_ts, e["timestamp"] + timedelta(microseconds=1))
            prio, kw = classify_priority(ctx[lo:hi], benign, failure)
            e.update(priority=prio, matched_keyword=kw,
                     enhanced_alert_score=e["zscore"] * weights[prio],
                     suppressed=(prio == "LOW"))
        else:
            e.update(priority=None, matched_keyword=None,
                     enhanced_alert_score=0.0, suppressed=False)

        true_label = labels.get((e["log_source"], e["timestamp"].isoformat()))
        if e["baseline_ready"]:  # exclude warm-up, same as the batch evaluation
            evaluator.record(e["classification"] == "ABNORMAL" and not e["suppressed"],
                             true_label)
        processed.append(e)

        if len(processed) % win == 0:
            m = evaluator.rolling_metrics()
            print(f"[{e['timestamp']}] n={len(processed)} precision={m['precision']:.3f} "
                  f"recall={m['recall']:.3f} fpr={m['fpr']:.4f}")
    return processed, evaluator


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--speed", type=float, default=None, help="overrides replay.speed_multiplier")
    ap.add_argument("--max-lines", type=int, default=None)
    a = ap.parse_args()

    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    labels = load_labels(str(PROJECT_ROOT / cfg["paths"]["ground_truth_dir"] / "labels.jsonl"))

    processed, ev = replay(str(proc / "clean_stream.jsonl"), str(proc / "parsed_entries.jsonl"),
                           cfg, labels, a.speed, a.max_lines or cfg["replay"].get("max_lines"))
    write_audit_log(processed, str(PROJECT_ROOT / cfg["paths"]["audit_log"]))

    m = ev.rolling_metrics()
    print("\n=== Final Real-Time Evaluation Summary ===")
    print(f"TP={ev.tp} FP={ev.fp} FN={ev.fn} TN={ev.tn}")
    print(f"Precision: {m['precision']:.4f}\nRecall:    {m['recall']:.4f}\nFPR:       {m['fpr']:.4f}")
    out = PROJECT_ROOT / cfg["paths"]["results_dir"] / "rolling_metrics_history.csv"
    ev.save_history(out)
    print(f"Rolling metrics saved to {out}")
