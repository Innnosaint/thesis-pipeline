"""
replay_simulator.py
Objective 5 support: Simulates real-time log arrival by replaying historical
Loghub entries at (accelerated) original timestamp intervals, feeding them
through the full detection pipeline one entry at a time with no lookahead —
enforcing true online/causal processing, as described in Section 3.6.2.
"""

import time
import json
import yaml
from pathlib import Path
from datetime import datetime

from baseline import DynamicBaseline
from zscore_detector import compute_zscore
from log_correlation import get_log_window, classify_priority, build_time_index
from alert_scoring import write_audit_log
from evaluation_harness import RollingEvaluator


def load_ground_truth(gt_path: str):
    """
    Expects a JSONL or CSV mapping some identifying key (e.g., timestamp+source)
    to a boolean/label 'is_anomaly'. Adjust the loader to match your actual
    Loghub-2.0 label file format.
    Returns dict keyed by (log_source, timestamp_iso) -> bool
    """
    labels = {}
    gt_file = Path(gt_path)
    if not gt_file.exists():
        print(f"WARNING: ground truth file not found at {gt_path}. "
              f"Evaluation will run without ground-truth scoring.")
        return labels

    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["log_source"], rec["timestamp"])
            labels[key] = bool(rec["is_anomaly"])
    return labels


def replay(clean_stream_path: str, cfg: dict, ground_truth: dict):
    """
    Streams the pre-parsed, pre-cleaned entries in timestamp order, pacing
    delivery according to cfg['replay']['speed_multiplier'] to simulate
    real-time arrival, and runs the full pipeline (baseline -> zscore ->
    correlation -> scoring) on each entry as it "arrives."
    """
    entries = []
    with open(clean_stream_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)

    entries.sort(key=lambda e: e["timestamp"])

    max_lines = cfg["replay"].get("max_lines")
    if max_lines:
        entries = entries[:max_lines]

    speed = cfg["replay"]["speed_multiplier"]
    baseline = DynamicBaseline(
        window_size=cfg["baseline"]["window_size"],
        min_observations=cfg["baseline"]["min_observations"],
    )
    evaluator = RollingEvaluator(rolling_window=cfg["evaluation"]["rolling_window_for_metrics"])

    theta_z = cfg["zscore"]["theta_z"]
    delta_t = cfg["correlation"]["delta_t_seconds"]
    benign_kw = cfg["correlation"]["benign_keywords"]
    failure_kw = cfg["correlation"]["failure_keywords"]
    weights = cfg["priority_weights"]

    # Context window for correlation needs recent history; keep a rolling buffer.
    context_buffer = []
    context_buffer_seconds = max(delta_t * 4, 120)

    processed = []
    prev_ts = None

    print(f"Starting real-time replay of {len(entries)} entries "
          f"(speed_multiplier={speed})...\n")

    for e in entries:
        # --- Pace the stream to simulate real-time arrival ---
        if prev_ts is not None and speed > 0:
            real_gap = (e["timestamp"] - prev_ts).total_seconds() / speed
            if real_gap > 0:
                time.sleep(min(real_gap, 2.0))  # capped so demos don't stall
        prev_ts = e["timestamp"]

        # --- Baseline (causal: computed BEFORE this observation is added) ---
        mu, sigma = baseline.get_baseline()
        e["baseline_mu"] = mu
        e["baseline_sigma"] = sigma
        e["baseline_ready"] = baseline.is_ready()
        baseline.update(e["disk_usage_value"])

        # --- Z-Score classification ---
        if e["baseline_ready"]:
            z = compute_zscore(e["disk_usage_value"], mu, sigma)
            e["zscore"] = z
            e["classification"] = "ABNORMAL" if abs(z) >= theta_z else "NORMAL"
            e["severity_score"] = abs(z)
        else:
            e["zscore"] = 0.0
            e["classification"] = "NORMAL"
            e["severity_score"] = 0.0

        # --- Contextual correlation (only for ABNORMAL) ---
        context_buffer.append(e)
        context_buffer = [
            c for c in context_buffer
            if (e["timestamp"] - c["timestamp"]).total_seconds() <= context_buffer_seconds
        ]

        if e["classification"] == "ABNORMAL":
            sorted_ctx, timestamps = build_time_index(context_buffer)
            window = get_log_window(sorted_ctx, timestamps, e["timestamp"], delta_t)
            priority, matched_kw = classify_priority(window, benign_kw, failure_kw)
            e["priority"] = priority
            e["matched_keyword"] = matched_kw
            e["enhanced_alert_score"] = e["zscore"] * weights[priority]
            e["suppressed"] = (priority == "LOW")
        else:
            e["priority"] = None
            e["matched_keyword"] = None
            e["enhanced_alert_score"] = 0.0
            e["suppressed"] = False

        # --- Live evaluation against ground truth (if available) ---
        gt_key = (e["log_source"], e["timestamp"].isoformat())
        true_label = ground_truth.get(gt_key)  # None if unknown
        predicted_abnormal = (e["classification"] == "ABNORMAL") and not e.get("suppressed")
        evaluator.record(predicted_abnormal, true_label)

        processed.append(e)

        if len(processed) % cfg["evaluation"]["rolling_window_for_metrics"] == 0:
            metrics = evaluator.rolling_metrics()
            print(f"[{e['timestamp']}] n={len(processed)} "
                  f"rolling_precision={metrics['precision']:.3f} "
                  f"rolling_recall={metrics['recall']:.3f} "
                  f"rolling_fpr={metrics['fpr']:.3f}")

    return processed, evaluator


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    clean_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "clean_stream.jsonl"
    gt_path = PROJECT_ROOT / cfg["paths"]["ground_truth_dir"] / "labels.jsonl"
    audit_path = PROJECT_ROOT / cfg["paths"]["audit_log"]
    results_path = PROJECT_ROOT / cfg["paths"]["results_dir"]

    ground_truth = load_ground_truth(str(gt_path))
    processed, evaluator = replay(str(clean_path), cfg, ground_truth)

    write_audit_log(processed, str(audit_path))

    final_metrics = evaluator.rolling_metrics()
    print("\n=== Final Real-Time Evaluation Summary ===")
    print(f"Precision: {final_metrics['precision']:.3f}")
    print(f"Recall:    {final_metrics['recall']:.3f}")
    print(f"FPR:       {final_metrics['fpr']:.3f}")
    print(f"Total processed: {len(processed)}")

    evaluator.save_history(results_path / "rolling_metrics_history.csv")
    print(f"\nRolling metrics history saved to {results_path / 'rolling_metrics_history.csv'}")
