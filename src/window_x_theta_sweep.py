"""
window_x_theta_sweep.py
Joint W x theta_z grid. Baseline is computed once per W; every theta_z is then
re-classified against it (cheap). No hard ceiling here (pure Z-Score grid).

Usage:
  python window_x_theta_sweep.py --clean-stream ../data/processed/clean_stream.jsonl \
    --labels ../data/ground_truth/labels.jsonl \
    --windows 10 20 30 50 75 100 150 200 --thetas 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
    --output-csv ../results/grid_sweep.csv --top-n 15
"""
import argparse
from baseline import run_baseline_over_stream
from zscore_detector import classify
from log_correlation import correlate_and_score
from common import (load_stream, load_context, load_labels, make_cfg,
                    compute_confusion, precision_recall_f1, write_csv)


def run_grid_sweep(clean_stream_path, labels_path, windows, thetas, delta_t,
                   output_csv, min_obs_ratio=0.2, top_n=15, context_path=None):
    raw = load_stream(clean_stream_path)
    labels = load_labels(labels_path)
    context = load_context(clean_stream_path, context_path)
    cfg = make_cfg(delta_t)
    print(f"{len(raw)} entries, {len(labels)} labels ({sum(labels.values())} anomalies)\n")

    rows, total, i = [], len(windows) * len(thetas), 0
    for W in windows:
        min_obs = max(5, int(W * min_obs_ratio))
        base = run_baseline_over_stream([dict(e) for e in raw], window_size=W,
                                        min_observations=min_obs)
        for theta in thetas:
            i += 1
            cls = classify([dict(e) for e in base], theta_z=theta)
            scored = correlate_and_score(cls, context or cls, cfg)
            tp, fp, tn, fn = compute_confusion(scored, labels)
            p, r, f1 = precision_recall_f1(tp, fp, fn)
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            rows.append({"window_size": W, "theta_z": theta, "min_observations": min_obs,
                         "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                         "precision": round(p, 4), "recall": round(r, 4),
                         "fpr": round(fpr, 4), "f1": round(f1, 4)})
            print(f"[{i}/{total}] W={W:>4} theta_z={theta:>4.1f}  TP={tp:>3} FP={fp:>6} "
                  f"FN={fn:>3}  P={p:.4f} R={r:.4f} F1={f1:.4f}")

    rows.sort(key=lambda r: r["f1"], reverse=True)
    print(f"\n=== Top {top_n} (W, theta_z) by F1 ===")
    print(f"{'Rank':>4} {'W':>5} {'theta_z':>8} {'TP':>4} {'FP':>7} {'FN':>4} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7}")
    for k, r in enumerate(rows[:top_n], 1):
        print(f"{k:>4} {r['window_size']:>5} {r['theta_z']:>8.1f} {r['TP']:>4} {r['FP']:>7} "
              f"{r['FN']:>4} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    write_csv(output_csv, list(rows[0].keys()), rows)
    print(f"\n{len(rows)} combinations written to {output_csv}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean-stream", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--windows", nargs="+", type=int, required=True)
    ap.add_argument("--thetas", nargs="+", type=float, required=True)
    ap.add_argument("--delta-t", type=int, default=60)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--min-obs-ratio", type=float, default=0.2)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--context", default=None)
    a = ap.parse_args()
    run_grid_sweep(a.clean_stream, a.labels, a.windows, a.thetas, a.delta_t,
                   a.output_csv, a.min_obs_ratio, a.top_n, a.context)
