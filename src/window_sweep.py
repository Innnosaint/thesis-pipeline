"""
window_sweep.py  (Section 4.4 -- window-size diagnostic)
Sweeps the Dynamic Baseline window size W at a fixed theta_z (no ceiling).
window_size changes the baseline itself, so baseline -> z-score -> correlation
are re-run for every W.

Usage:
  python window_sweep.py --clean-stream ../data/processed/clean_stream.jsonl \
    --labels ../data/ground_truth/labels.jsonl \
    --windows 10 15 20 30 50 75 100 150 200 --theta-z 2.0 \
    --output-csv ../results/window_sweep.csv
"""
import argparse
from baseline import run_baseline_over_stream
from zscore_detector import classify
from log_correlation import correlate_and_score
from common import (load_stream, load_context, load_labels, make_cfg,
                    compute_confusion, precision_recall_f1, write_csv)


def run_sweep(clean_stream_path, labels_path, windows, theta_z, delta_t,
              output_csv, min_obs_ratio=0.2, context_path=None):
    raw = load_stream(clean_stream_path)
    labels = load_labels(labels_path)
    context = load_context(clean_stream_path, context_path)
    cfg = make_cfg(delta_t)
    print(f"{len(raw)} entries, {len(labels)} labels, "
          f"context={'all log lines' if context else 'disk-only clean stream'}\n")

    header = f"{'W':>6} {'min_obs':>8} {'TP':>5} {'FP':>7} {'TN':>8} {'FN':>5} " \
             f"{'Precision':>10} {'Recall':>8} {'FPR':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for W in windows:
        min_obs = max(5, int(W * min_obs_ratio))
        base = run_baseline_over_stream([dict(e) for e in raw], window_size=W,
                                        min_observations=min_obs)
        cls = classify(base, theta_z=theta_z)
        scored = correlate_and_score(cls, context or cls, cfg)
        tp, fp, tn, fn = compute_confusion(scored, labels)
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({"window_size": W, "min_observations": min_obs, "TP": tp,
                     "FP": fp, "TN": tn, "FN": fn, "precision": round(p, 4),
                     "recall": round(r, 4), "fpr": round(fpr, 4), "f1": round(f1, 4)})
        print(f"{W:>6} {min_obs:>8} {tp:>5} {fp:>7} {tn:>8} {fn:>5} "
              f"{p:>10.4f} {r:>8.4f} {fpr:>8.4f} {f1:>8.4f}")

    best = max(rows, key=lambda r: r["f1"])
    print(f"\nBest F1: W={best['window_size']} (F1={best['f1']}, "
          f"P={best['precision']}, R={best['recall']})")
    write_csv(output_csv, list(rows[0].keys()), rows)
    print(f"Written to {output_csv}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean-stream", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--windows", nargs="+", type=int, required=True)
    ap.add_argument("--theta-z", type=float, default=2.0)
    ap.add_argument("--delta-t", type=int, default=60)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--min-obs-ratio", type=float, default=0.2)
    ap.add_argument("--context", default=None, help="Optional parsed_entries.jsonl for correlation")
    a = ap.parse_args()
    run_sweep(a.clean_stream, a.labels, a.windows, a.theta_z, a.delta_t,
              a.output_csv, a.min_obs_ratio, a.context)
