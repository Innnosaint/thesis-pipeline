"""
ceiling_sweep.py  (Section 4.5 -- Table 4.2)
Joint theta_z x hard_ceiling sweep for the hybrid safety-net rule (Eq. 4.1).
Baseline is computed once at --window-size; classification is re-run per pair.

Usage:
  python ceiling_sweep.py --clean-stream ../data/processed/clean_stream.jsonl \
    --labels ../data/ground_truth/labels.jsonl --window-size 50 \
    --thetas 2.0 3.0 4.0 4.5 5.0 5.5 6.0 --ceilings 80 85 90 95 none \
    --output-csv ../results/ceiling_sweep.csv --top-n 15
"""
import argparse
from baseline import run_baseline_over_stream
from zscore_detector import classify
from log_correlation import correlate_and_score
from common import (load_stream, load_context, load_labels, make_cfg,
                    compute_confusion, precision_recall_f1, write_csv)


def parse_ceiling(v: str):
    return None if v.lower() == "none" else float(v)


def run_sweep(clean_stream_path, labels_path, window_size, thetas, ceilings,
              delta_t, output_csv, min_obs_ratio=0.2, top_n=15, context_path=None):
    raw = load_stream(clean_stream_path)
    labels = load_labels(labels_path)
    context = load_context(clean_stream_path, context_path)
    cfg = make_cfg(delta_t)
    min_obs = max(5, int(window_size * min_obs_ratio))
    print(f"{len(raw)} entries, {len(labels)} labels ({sum(labels.values())} anomalies)")
    print(f"Baseline once at W={window_size}, min_observations={min_obs}\n")
    base = run_baseline_over_stream([dict(e) for e in raw], window_size=window_size,
                                    min_observations=min_obs)

    rows, total, i = [], len(thetas) * len(ceilings), 0
    for theta in thetas:
        for ceil in ceilings:
            i += 1
            cls = classify([dict(e) for e in base], theta_z=theta, hard_ceiling=ceil)
            scored = correlate_and_score(cls, context or cls, cfg)
            tp, fp, tn, fn = compute_confusion(scored, labels)
            p, r, f1 = precision_recall_f1(tp, fp, fn)
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            label = "none" if ceil is None else f"{ceil:.0f}"
            rows.append({"theta_z": theta, "hard_ceiling": label, "TP": tp, "FP": fp,
                         "TN": tn, "FN": fn, "precision": round(p, 4),
                         "recall": round(r, 4), "fpr": round(fpr, 4), "f1": round(f1, 4)})
            print(f"[{i}/{total}] theta_z={theta:>4.1f} ceiling={label:>4}  TP={tp:>3} "
                  f"FP={fp:>5} FN={fn:>3}  P={p:.4f} R={r:.4f} F1={f1:.4f}")

    rows.sort(key=lambda r: r["f1"], reverse=True)
    print(f"\n=== Top {top_n} (theta_z, ceiling) by F1 ===")
    print(f"{'Rank':>4} {'theta_z':>8} {'ceiling':>8} {'TP':>4} {'FP':>6} {'FN':>4} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7}")
    for k, r in enumerate(rows[:top_n], 1):
        print(f"{k:>4} {r['theta_z']:>8.1f} {r['hard_ceiling']:>8} {r['TP']:>4} {r['FP']:>6} "
              f"{r['FN']:>4} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    b = rows[0]
    print(f"\nBest: theta_z={b['theta_z']}, ceiling={b['hard_ceiling']} "
          f"(F1={b['f1']}, P={b['precision']}, R={b['recall']}, FP={b['FP']}, FN={b['FN']})")
    write_csv(output_csv, list(rows[0].keys()), rows)
    print(f"{len(rows)} combinations written to {output_csv}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean-stream", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--thetas", nargs="+", type=float, required=True)
    ap.add_argument("--ceilings", nargs="+", required=True, help="percentages or 'none'")
    ap.add_argument("--delta-t", type=int, default=60)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--min-obs-ratio", type=float, default=0.2)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--context", default=None)
    a = ap.parse_args()
    run_sweep(a.clean_stream, a.labels, a.window_size, a.thetas,
              [parse_ceiling(c) for c in a.ceilings], a.delta_t, a.output_csv,
              a.min_obs_ratio, a.top_n, a.context)
