"""
ceiling_sweep.py
Sweeps theta_z AND hard_ceiling together to find the best hybrid
safety-net configuration (see zscore_detector.py's classify() docstring
for the rationale). Uses an already-computed baseline_stream.jsonl (or
clean_stream.jsonl, which will have baseline recomputed once) so this
is fast -- classification is cheap, unlike full baseline recomputation.

Usage:
  python ceiling_sweep.py \
    --clean-stream ../data/processed/clean_stream.jsonl \
    --labels ../data/ground_truth/labels.jsonl \
    --window-size 50 \
    --thetas 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
    --ceilings 80 82 85 88 90 95 none \
    --output-csv ../results/ceiling_sweep.csv \
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


def parse_ceiling(val: str):
    if val.lower() == "none":
        return None
    return float(val)


def run_sweep(clean_stream_path, labels_path, window_size, thetas, ceilings,
              delta_t, output_csv, min_observations_ratio=0.2, top_n=15):
    print(f"Loading clean stream from {clean_stream_path} ...")
    raw_entries = load_clean_stream(clean_stream_path)
    print(f"  {len(raw_entries)} entries loaded.")

    print(f"Loading ground truth from {labels_path} ...")
    labels = load_labels(labels_path)
    n_true = sum(1 for v in labels.values() if v)
    print(f"  {len(labels)} labeled entries loaded ({n_true} true anomalies).\n")

    min_obs = max(5, int(window_size * min_observations_ratio))
    print(f"Computing baseline ONCE at window_size={window_size}, min_observations={min_obs} ...")
    baselined = run_baseline_over_stream([dict(e) for e in raw_entries],
                                          window_size=window_size, min_observations=min_obs)
    print("Baseline computed.\n")

    cfg = dict(DEFAULT_CFG_CORRELATION)
    cfg["correlation"]["delta_t_seconds"] = delta_t

    rows = []
    total = len(thetas) * len(ceilings)
    i = 0
    for theta in thetas:
        for ceiling in ceilings:
            i += 1
            classified = classify([dict(e) for e in baselined], theta_z=theta, hard_ceiling=ceiling)
            scored = correlate_and_score(classified, classified, cfg)

            tp, fp, tn, fn = compute_confusion(scored, labels)
            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            ceiling_label = "none" if ceiling is None else f"{ceiling:.0f}"
            rows.append({
                "theta_z": theta, "hard_ceiling": ceiling_label,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
            })

            print(f"[{i}/{total}] theta_z={theta:>4.1f} ceiling={ceiling_label:>4}  "
                  f"TP={tp:>3} FP={fp:>5} FN={fn:>3}  "
                  f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    rows.sort(key=lambda r: r["f1"], reverse=True)

    print(f"\n=== Top {top_n} (theta_z, hard_ceiling) combinations by F1 ===\n")
    header = f"{'Rank':>4} {'theta_z':>8} {'ceiling':>8} {'TP':>5} {'FP':>7} {'FN':>5} " \
              f"{'Precision':>10} {'Recall':>8} {'FPR':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(rows[:top_n], 1):
        print(f"{rank:>4} {r['theta_z']:>8.1f} {r['hard_ceiling']:>8} "
              f"{r['TP']:>5} {r['FP']:>7} {r['FN']:>5} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['fpr']:>8.4f} {r['f1']:>8.4f}")

    best = rows[0]
    print(f"\nBest overall: theta_z={best['theta_z']}, hard_ceiling={best['hard_ceiling']} "
          f"(F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, "
          f"FP={best['FP']}, FN={best['FN']})")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("theta_z,hard_ceiling,TP,FP,TN,FN,precision,recall,fpr,f1\n")
        for r in rows:
            f.write(f"{r['theta_z']},{r['hard_ceiling']},{r['TP']},{r['FP']},"
                     f"{r['TN']},{r['FN']},{r['precision']:.4f},{r['recall']:.4f},"
                     f"{r['fpr']:.4f},{r['f1']:.4f}\n")

    print(f"\nFull results ({len(rows)} combinations) written to {output_csv}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep theta_z x hard_ceiling for the hybrid safety-net rule.")
    parser.add_argument("--clean-stream", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--thetas", nargs="+", type=float, required=True)
    parser.add_argument("--ceilings", nargs="+", type=str, required=True,
                         help="List of ceiling percentages, or 'none' to disable for that run")
    parser.add_argument("--delta-t", type=int, default=60)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-obs-ratio", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    ceilings_parsed = [parse_ceiling(c) for c in args.ceilings]

    run_sweep(
        clean_stream_path=args.clean_stream,
        labels_path=args.labels,
        window_size=args.window_size,
        thetas=args.thetas,
        ceilings=ceilings_parsed,
        delta_t=args.delta_t,
        output_csv=args.output_csv,
        min_observations_ratio=args.min_obs_ratio,
        top_n=args.top_n,
    )
