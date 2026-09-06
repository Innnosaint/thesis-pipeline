"""
window_x_theta_sweep.py
Joint sensitivity analysis over Dynamic Baseline window_size (W) AND
Z-Score anomaly threshold (theta_z) simultaneously, since the two
parameters interact: window size determines how stable/noisy sigma(t)
is, which in turn determines what a given theta_z actually means in
absolute terms. Tuning either parameter alone (as in theta_sweep.py or
window_sweep.py individually) can mislead, since the "best" theta_z at
one window size may be far from optimal at another.

For each window_size in --windows, this script re-runs baseline
computation ONCE (the expensive part), then re-classifies with EVERY
theta_z in --thetas against that same baseline (cheap), avoiding
redundant baseline recomputation across the theta dimension.

Usage:
  python window_x_theta_sweep.py \
    --clean-stream ../data/processed/clean_stream.jsonl \
    --labels ../data/ground_truth/labels.jsonl \
    --windows 10 20 30 50 75 100 150 200 \
    --thetas 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
    --delta-t 60 \
    --output-csv ../results/grid_sweep.csv \
    --top-n 15
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
    labels = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["log_source"], rec["timestamp"])
            labels[key] = bool(rec["is_anomaly"])
    return labels


def compute_confusion(scored_entries, labels):
    """Excludes entries whose baseline was not yet ready (warm-up period),
    consistent across all (W, theta) combinations."""
    tp = fp = tn = fn = 0

    for e in scored_entries:
        if not e.get("baseline_ready", False):
            continue

        key = (e["log_source"], e["timestamp"].isoformat())
        true_label = labels.get(key)
        if true_label is None:
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

    return tp, fp, tn, fn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_grid_sweep(clean_stream_path, labels_path, windows, thetas, delta_t,
                    output_csv, min_observations_ratio=0.2, top_n=15):
    print(f"Loading clean stream from {clean_stream_path} ...")
    raw_entries = load_clean_stream(clean_stream_path)
    print(f"  {len(raw_entries)} entries loaded.")

    print(f"Loading ground truth from {labels_path} ...")
    labels = load_labels(labels_path)
    n_true_positive_total = sum(1 for v in labels.values() if v)
    print(f"  {len(labels)} labeled entries loaded ({n_true_positive_total} true anomalies).\n")

    cfg = dict(DEFAULT_CFG_CORRELATION)
    cfg["correlation"]["delta_t_seconds"] = delta_t

    rows = []
    total_runs = len(windows) * len(thetas)
    run_i = 0

    for W in windows:
        min_obs = max(5, int(W * min_observations_ratio))
        entries = [dict(e) for e in raw_entries]

        # Baseline computed ONCE per window size (expensive step)
        baselined = run_baseline_over_stream(entries, window_size=W, min_observations=min_obs)

        for theta in thetas:
            run_i += 1
            # classify() and correlate_and_score() are cheap; re-run per theta
            classified_copy = [dict(e) for e in baselined]
            classified = classify(classified_copy, theta_z=theta)
            scored = correlate_and_score(classified, classified, cfg)

            tp, fp, tn, fn = compute_confusion(scored, labels)
            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            rows.append({
                "window_size": W, "theta_z": theta, "min_observations": min_obs,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
            })

            print(f"[{run_i}/{total_runs}] W={W:>4} theta_z={theta:>4.1f}  "
                  f"TP={tp:>3} FP={fp:>6} FN={fn:>3}  "
                  f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    rows.sort(key=lambda r: r["f1"], reverse=True)

    print(f"\n=== Top {top_n} (window_size, theta_z) combinations by F1 ===\n")
    header = f"{'Rank':>4} {'W':>6} {'theta_z':>8} {'TP':>5} {'FP':>7} {'FN':>5} " \
              f"{'Precision':>10} {'Recall':>8} {'FPR':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(rows[:top_n], 1):
        print(f"{rank:>4} {r['window_size']:>6} {r['theta_z']:>8.1f} "
              f"{r['TP']:>5} {r['FP']:>7} {r['FN']:>5} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['fpr']:>8.4f} {r['f1']:>8.4f}")

    best = rows[0]
    print(f"\nBest overall: W={best['window_size']}, theta_z={best['theta_z']} "
          f"(F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, "
          f"FP={best['FP']}, FN={best['FN']})")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("window_size,theta_z,min_observations,TP,FP,TN,FN,precision,recall,fpr,f1\n")
        for r in rows:
            f.write(f"{r['window_size']},{r['theta_z']},{r['min_observations']},"
                     f"{r['TP']},{r['FP']},{r['TN']},{r['FN']},"
                     f"{r['precision']:.4f},{r['recall']:.4f},{r['fpr']:.4f},{r['f1']:.4f}\n")

    print(f"\nFull grid results ({len(rows)} combinations) written to {output_csv}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint window_size x theta_z sensitivity grid sweep.")
    parser.add_argument("--clean-stream", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--windows", nargs="+", type=int, required=True)
    parser.add_argument("--thetas", nargs="+", type=float, required=True)
    parser.add_argument("--delta-t", type=int, default=60)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-obs-ratio", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    run_grid_sweep(
        clean_stream_path=args.clean_stream,
        labels_path=args.labels,
        windows=args.windows,
        thetas=args.thetas,
        delta_t=args.delta_t,
        output_csv=args.output_csv,
        min_observations_ratio=args.min_obs_ratio,
        top_n=args.top_n,
    )
