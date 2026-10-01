"""
window_sweep.py
Diagnostic sweep over Dynamic Baseline window_size (W) to test whether a
smaller/larger sliding window improves recall on slow, gradual disk-usage
climbs without sacrificing precision, per Chapter 4 diagnostic discussion.

Unlike a theta_z sweep (which only changes the classification cutoff on
an already-computed baseline), window_size changes the baseline itself,
so this script re-runs baseline -> zscore -> correlation for each W
tested, using a fixed theta_z you specify (default 2.0, the thesis's
stated default).

Usage:
  python window_sweep.py \
    --clean-stream ../data/processed/clean_stream.jsonl \
    --labels ../data/ground_truth/labels.jsonl \
    --windows 10 20 30 50 75 100 150 200 \
    --theta-z 2.0 \
    --delta-t 60 \
    --output-csv ../results/window_sweep.csv
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from baseline import run_baseline_over_stream
from zscore_detector import classify
from log_correlation import correlate_and_score


DEFAULT_CFG_CORRELATION = {
    "correlation": {
        "delta_t_seconds": 60,
        "benign_keywords": [
            "backup", "rsync", "cron", "apt", "yum",
            "package install", "scheduled task",
        ],
        "failure_keywords": [
            "disk full", "i/o error", "write failure", "inode exhausted",
            "filesystem read-only", "device error", "no space left", "disk quota",
        ],
    },
    "priority_weights": {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5},
}


def load_clean_stream(path: str):
    entries = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)
    return entries


def load_labels(path: str):
    """Returns dict keyed by (log_source, timestamp_iso) -> bool"""
    labels = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["log_source"], rec["timestamp"])
            labels[key] = bool(rec["is_anomaly"])
    return labels


def compute_confusion(scored_entries, labels, min_observations: int):
    """
    Computes TP/FP/TN/FN comparing algorithm's non-suppressed ABNORMAL
    classification against ground truth labels. Entries whose baseline
    was not yet ready (insufficient history) are excluded, matching how
    the main pipeline treats warm-up entries.
    """
    tp = fp = tn = fn = 0
    unlabeled = 0

    for e in scored_entries:
        if not e.get("baseline_ready", False):
            continue  # exclude warm-up period, consistent across all window sizes

        key = (e["log_source"], e["timestamp"].isoformat())
        true_label = labels.get(key)
        if true_label is None:
            unlabeled += 1
            continue

        predicted_abnormal = (e["classification"] == "ABNORMAL") and not e.get("suppressed", False)

        if predicted_abnormal and true_label:
            tp += 1
        elif predicted_abnormal and not true_label:
            fp += 1
        elif not predicted_abnormal and true_label:
            fn += 1
        else:
            tn += 1

    return tp, fp, tn, fn, unlabeled


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_sweep(clean_stream_path, labels_path, windows, theta_z, delta_t, output_csv,
              min_observations_ratio=0.2):
    print(f"Loading clean stream from {clean_stream_path} ...")
    raw_entries = load_clean_stream(clean_stream_path)
    print(f"  {len(raw_entries)} entries loaded.")

    print(f"Loading ground truth from {labels_path} ...")
    labels = load_labels(labels_path)
    print(f"  {len(labels)} labeled entries loaded.\n")

    cfg = dict(DEFAULT_CFG_CORRELATION)
    cfg["correlation"]["delta_t_seconds"] = delta_t

    header = f"{'W':>6} {'min_obs':>8} {'TP':>5} {'FP':>7} {'TN':>8} {'FN':>5} " \
             f"{'Precision':>10} {'Recall':>8} {'FPR':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for W in windows:
        min_obs = max(5, int(W * min_observations_ratio))

        # Fresh deep-ish copy per run (entries are dicts; re-baseline mutates in place)
        entries = [dict(e) for e in raw_entries]

        baselined = run_baseline_over_stream(entries, window_size=W, min_observations=min_obs)
        classified = classify(baselined, theta_z=theta_z)
        scored = correlate_and_score(classified, classified, cfg)

        tp, fp, tn, fn, unlabeled = compute_confusion(scored, labels, min_obs)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append({
            "window_size": W, "min_observations": min_obs,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
        })

        print(f"{W:>6} {min_obs:>8} {tp:>5} {fp:>7} {tn:>8} {fn:>5} "
              f"{precision:>10.4f} {recall:>8.4f} {fpr:>8.4f} {f1:>8.4f}")

    best = max(rows, key=lambda r: r["f1"])
    print(f"\nBest F1 among tested window sizes: W={best['window_size']} "
          f"(F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f})")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("window_size,min_observations,TP,FP,TN,FN,precision,recall,fpr,f1\n")
        for r in rows:
            f.write(f"{r['window_size']},{r['min_observations']},{r['TP']},{r['FP']},"
                     f"{r['TN']},{r['FN']},{r['precision']:.4f},{r['recall']:.4f},"
                     f"{r['fpr']:.4f},{r['f1']:.4f}\n")

    print(f"\nSweep results written to {output_csv}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep dynamic baseline window_size and report P/R/F1.")
    parser.add_argument("--clean-stream", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--windows", nargs="+", type=int, required=True)
    parser.add_argument("--theta-z", type=float, default=2.0)
    parser.add_argument("--delta-t", type=int, default=60)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-obs-ratio", type=float, default=0.2,
                         help="min_observations = max(5, W * ratio); default 0.2")
    args = parser.parse_args()

    run_sweep(
        clean_stream_path=args.clean_stream,
        labels_path=args.labels,
        windows=args.windows,
        theta_z=args.theta_z,
        delta_t=args.delta_t,
        output_csv=args.output_csv,
        min_observations_ratio=args.min_obs_ratio,
    )
