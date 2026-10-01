"""
make_tables.py
Regenerates Tables 4.1, 4.2 and 4.3 of the thesis directly from the data
(clean stream + ground-truth labels + full log context), prints them in the
paper's layout, and compares every cell with the value PRINTED IN THE PAPER
so any mismatch is visible before the defense.

Outputs: results/tables/table_4_1.csv, table_4_2.csv, table_4_3.csv, tables.md

Usage (from src/, after run_pipeline.py and build_ground_truth.py):
  python make_tables.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from baseline import run_baseline_over_stream
from zscore_detector import classify
from log_correlation import correlate_and_score
from common import (load_stream, load_context, load_labels, make_cfg,
                    compute_confusion, precision_recall_f1)

ROOT = Path(__file__).resolve().parent.parent

# Values exactly as printed in the paper: (TP, FP, FN)
PAPER_41 = {1.5: (43, 34390, 2), 2.0: (35, 12862, 10), 2.5: (29, 2675, 16),
            3.0: (24, 263, 21), 3.5: (24, 13, 21), 4.0: (24, 0, 21)}
PAPER_42 = [(4.0, 85.0, (27, 59, 18)), (4.5, 85.0, (27, 14, 18)),
            (5.0, 85.0, (27, 3, 18)), (5.5, 85.0, (27, 0, 18)),
            (6.0, 85.0, (27, 0, 18)), (5.0, 90.0, (12, 3, 33)),
            (5.0, None, (12, 3, 33))]
PAPER_STATIC = (39, 25, 6)
PAPER_FINAL = (27, 0, 18)


def run_cfg(base, theta, ceiling, context, cfg, labels):
    cls = classify([dict(e) for e in base], theta_z=theta, hard_ceiling=ceiling)
    scored = correlate_and_score(cls, context or cls, cfg)
    return compute_confusion(scored, labels)


def static_cm(base, labels, thr):
    tp = fp = tn = fn = 0
    for e in base:
        if not e["baseline_ready"]:
            continue
        lab = labels.get((e["log_source"], e["timestamp"].isoformat()))
        if lab is None:
            continue
        pred = e["disk_usage_value"] >= thr
        if pred and lab: tp += 1
        elif pred: fp += 1
        elif lab: fn += 1
        else: tn += 1
    return tp, fp, tn, fn


def check(got, paper):
    return "OK" if tuple(got) == tuple(paper) else f"DIFF (paper TP/FP/FN={paper})"


def row_md(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean-stream", default=str(ROOT / "data/processed/clean_stream.jsonl"))
    ap.add_argument("--labels", default=str(ROOT / "data/ground_truth/labels.jsonl"))
    ap.add_argument("--context", default=None)
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--static-threshold", type=float, default=80.0)
    ap.add_argument("--delta-t", type=int, default=60)
    ap.add_argument("--out-dir", default=str(ROOT / "results/tables"))
    a = ap.parse_args()

    raw = load_stream(a.clean_stream)
    labels = load_labels(a.labels)
    context = load_context(a.clean_stream, a.context)
    if context:
        context.sort(key=lambda e: e["timestamp"])
    cfg = make_cfg(a.delta_t)
    min_obs = max(5, int(a.window_size * 0.2))
    base = run_baseline_over_stream([dict(e) for e in raw], window_size=a.window_size,
                                    min_observations=min_obs)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = []

    print(f"{len(raw)} readings | {len(labels)} labels ({sum(labels.values())} anomalies) | "
          f"W={a.window_size} | context={'all log lines' if context else 'disk-only'}\n")

    # ---------------- Table 4.1 ----------------
    hdr = ["theta_z", "TP", "FP", "FN", "Precision", "Recall", "F1", "vs paper"]
    md += ["### Table 4.1  Z-Score Threshold Sensitivity Sweep (W = 50, no ceiling)", "",
           row_md(hdr), row_md(["---"] * len(hdr))]
    csv41 = [",".join(hdr)]
    print("Table 4.1  Z-Score Threshold Sensitivity Sweep (W = 50, no ceiling)")
    print(f"{'theta_z':>8}{'TP':>6}{'FP':>8}{'FN':>5}{'Precision':>11}{'Recall':>8}{'F1':>8}  vs paper")
    for th in sorted(PAPER_41):
        tp, fp, tn, fn = run_cfg(base, th, None, context, cfg, labels)
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        c = check((tp, fp, fn), PAPER_41[th])
        print(f"{th:>8.1f}{tp:>6}{fp:>8}{fn:>5}{p:>11.4f}{r:>8.4f}{f1:>8.4f}  {c}")
        cells = [th, tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", c]
        md.append(row_md(cells)); csv41.append(",".join(str(x).replace(",", ";") for x in cells))
    tp, fp, tn, fn = static_cm(base, labels, a.static_threshold)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    c = check((tp, fp, fn), PAPER_STATIC)
    print(f"{'STATIC':>8}{tp:>6}{fp:>8}{fn:>5}{p:>11.4f}{r:>8.4f}{f1:>8.4f}  {c}\n")
    cells = [f"STATIC {a.static_threshold:.0f}%", tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", c]
    md.append(row_md(cells)); csv41.append(",".join(str(x).replace(",", ";") for x in cells))
    (out / "table_4_1.csv").write_text("\n".join(csv41) + "\n")
    static_row = (tp, fp, fn, p, r, f1, tn)

    # ---------------- Table 4.2 ----------------
    hdr = ["theta_z", "theta_ceiling", "TP", "FP", "FN", "Precision", "Recall", "F1", "vs paper"]
    md += ["", "### Table 4.2  Joint theta_z x theta_ceiling Sweep (W = 50)", "",
           row_md(hdr), row_md(["---"] * len(hdr))]
    csv42 = [",".join(hdr)]
    print("Table 4.2  Joint theta_z x theta_ceiling Sweep (W = 50)")
    print(f"{'theta_z':>8}{'ceiling':>9}{'TP':>6}{'FP':>6}{'FN':>5}{'Precision':>11}{'Recall':>8}{'F1':>8}  vs paper")
    for th, ceil, paper in PAPER_42:
        tp, fp, tn, fn = run_cfg(base, th, ceil, context, cfg, labels)
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        c = check((tp, fp, fn), paper)
        cl = "none" if ceil is None else f"{ceil:.0f}%"
        print(f"{th:>8.1f}{cl:>9}{tp:>6}{fp:>6}{fn:>5}{p:>11.4f}{r:>8.4f}{f1:>8.4f}  {c}")
        cells = [th, cl, tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", c]
        md.append(row_md(cells)); csv42.append(",".join(str(x).replace(",", ";") for x in cells))
    stp, sfp, sfn, sp, sr, sf1, _ = static_row
    print(f"{'STATIC':>8}{'80%':>9}{stp:>6}{sfp:>6}{sfn:>5}{sp:>11.4f}{sr:>8.4f}{sf1:>8.4f}\n")
    cells = ["STATIC", "80%", stp, sfp, sfn, f"{sp:.4f}", f"{sr:.4f}", f"{sf1:.4f}", ""]
    md.append(row_md(cells)); csv42.append(",".join(str(x) for x in cells))
    (out / "table_4_2.csv").write_text("\n".join(csv42) + "\n")

    # ---------------- Table 4.3 ----------------
    etp, efp, etn, efn = run_cfg(base, 5.5, 85.0, context, cfg, labels)
    ep, er, ef1 = precision_recall_f1(etp, efp, efn)
    hdr = ["Metric", "Static Threshold (80%)", "Enhanced (theta_z=5.5, ceiling=85%)", "Delta", "vs paper"]
    rows = [("True Positives", stp, etp), ("False Positives", sfp, efp),
            ("False Negatives", sfn, efn), ("Precision", sp, ep),
            ("Recall", sr, er), ("F1-Score", sf1, ef1)]
    paper_ok = check((etp, efp, efn), PAPER_FINAL) + " / static " + check((stp, sfp, sfn), PAPER_STATIC)
    md += ["", "### Table 4.3  Final Comparative Performance", "", row_md(hdr), row_md(["---"] * len(hdr))]
    csv43 = [",".join(hdr)]
    print("Table 4.3  Final Comparative Performance: Static Threshold vs. Enhanced Algorithm")
    print(f"{'Metric':<18}{'Static (80%)':>14}{'Enhanced':>12}{'Delta':>10}")
    for name, s, e in rows:
        isint = isinstance(s, int)
        fmt = (lambda v: f"{v}") if isint else (lambda v: f"{v:.4f}")
        d = (e - s)
        dtxt = f"{d:+d}" if isint else f"{d:+.4f}"
        print(f"{name:<18}{fmt(s):>14}{fmt(e):>12}{dtxt:>10}")
        cells = [name, fmt(s), fmt(e), dtxt, paper_ok if name == "True Positives" else ""]
        md.append(row_md(cells)); csv43.append(",".join(cells))
    print(f"\nvs paper: {paper_ok}")
    (out / "table_4_3.csv").write_text("\n".join(csv43) + "\n")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print(f"\nWritten to {out}/ (table_4_1.csv, table_4_2.csv, table_4_3.csv, tables.md)")


if __name__ == "__main__":
    main()
