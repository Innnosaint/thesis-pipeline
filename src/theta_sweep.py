"""
theta_sweep.py

Sweeps a range of theta_z (Z-Score anomaly threshold) values against
the already-computed 'zscore' field in final_scored_stream.jsonl,
re-classifying each entry as ABNORMAL/NORMAL for each candidate theta_z
without re-running the full pipeline (Drain3 parsing, baseline
calculation, etc. only need to happen once).

Also computes the static 80% threshold baseline once for reference,
since it doesn't depend on theta_z.

An entry counts as a "predicted positive" (actionable alert) if EITHER:
  (a) abs(zscore) >= theta, OR
  (b) --hard-ceiling is set AND disk_usage_value >= hard_ceiling
(matching the hybrid safety-net rule in zscore_detector.py's classify()),
AND the entry was not suppressed by Contextual Log Correlation as a
benign event (e.g. a scheduled backup). Ignoring the suppressed flag
would count silently-suppressed alerts as if they were active
detections, which does not reflect what an administrator would
actually see or act on -- this was a bug in an earlier version of this
script and has been fixed here.

If --hard-ceiling is omitted, this reproduces the original pure
Z-Score-only sensitivity curve (condition (a) alone), useful for
isolating the Z-Score mechanism's standalone contribution in Chapter 4,
separate from the hybrid ceiling's added effect (see also
ceiling_sweep.py for a full theta_z x hard_ceiling grid).

Also excludes entries where baseline_ready == False (insufficient
history for a valid baseline), consistent with how the main pipeline
and window_x_theta_sweep.py treat the warm-up period. Note: the ceiling
condition still applies even when baseline_ready == False, matching
zscore_detector.py's behavior (a 95% reading is dangerous regardless of
whether enough history exists yet for a statistical baseline) -- but
such entries are still excluded from the confusion matrix here by
default, since ground truth ceiling-only entries during warm-up are a
separate edge case worth checking manually if it affects results.

This produces the sensitivity curve needed to justify the final theta_z
choice in Chapter 4 / Objective 3, and to compare against the static
threshold baseline at each sensitivity level.

Usage:
    python theta_sweep.py \
        --scored ../data/processed/final_scored_stream.jsonl \
        --labels ../data/ground_truth/labels.jsonl \
        --thetas 1.5 2.0 2.5 3.0 3.5 4.0 \
        --static-threshold 80.0 \
        --hard-ceiling 85.0 \
        --output-csv ../results/theta_sweep.csv
"""

import argparse
import json


def load_jsonl(path):
    records = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["timestamp"]] = obj
    return records


def safe_div(a, b):
    return a / b if b else 0.0


def confusion_for_predicted_set(pred_positive_ts, ground_truth, scored, require_baseline_ready=True):
    tp = fp = tn = fn = 0
    for ts, gt in ground_truth.items():
        entry = scored.get(ts)
        if entry is None:
            continue
        if require_baseline_ready and not entry.get("baseline_ready", False):
            continue

        actual = gt["is_anomaly"]
        predicted = ts in pred_positive_ts
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def metrics_from_confusion(tp, fp, tn, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "fpr": fpr, "accuracy": accuracy, "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--thetas", type=float, nargs="+",
                         default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    parser.add_argument("--static-threshold", type=float, default=80.0)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--hard-ceiling", type=float, default=None,
                         help="If set, an entry with disk_usage_value >= this value "
                              "is also counted as predicted positive, matching the "
                              "hybrid safety-net rule in zscore_detector.py. "
                              "Omit to sweep pure Z-Score sensitivity only.")
    parser.add_argument("--include-warmup", action="store_true",
                         help="Include entries where baseline_ready == False "
                              "(default: excluded, matching main pipeline behavior)")
    args = parser.parse_args()

    require_baseline_ready = not args.include_warmup

    print(f"Loading scored stream from {args.scored} ...")
    scored = load_jsonl(args.scored)
    print(f"  {len(scored)} scored entries loaded.")

    print(f"Loading ground truth from {args.labels} ...")
    ground_truth = load_jsonl(args.labels)
    print(f"  {len(ground_truth)} labeled entries loaded.\n")

    if not require_baseline_ready:
        print("NOTE: --include-warmup set; entries with baseline_ready=False are included.\n")

    if args.hard_ceiling is not None:
        print(f"NOTE: --hard-ceiling={args.hard_ceiling} set; predicted positives include "
              f"the hybrid safety-net ceiling condition alongside each theta_z.\n")

    # --- Static baseline (theta-independent, computed once) ---
    # Static threshold has no "suppression" concept -- it is a raw metric
    # comparison, unaffected by Contextual Log Correlation.
    static_positive = {
        ts for ts, e in scored.items()
        if e.get("disk_usage_value") is not None
        and e["disk_usage_value"] >= args.static_threshold
    }
    static_metrics = metrics_from_confusion(
        *confusion_for_predicted_set(static_positive, ground_truth, scored, require_baseline_ready)
    )

    # --- Theta sweep ---
    # An entry is a "predicted positive" (actionable alert) if its
    # Z-Score clears theta OR (when set) its raw disk usage clears the
    # hybrid safety-net ceiling, AND it was not suppressed as benign.
    results = []
    for theta in args.thetas:
        predicted_positive = set()
        for ts, e in scored.items():
            if e.get("suppressed", False):
                continue

            zscore_hit = e.get("zscore") is not None and abs(e["zscore"]) >= theta
            ceiling_hit = (args.hard_ceiling is not None
                            and e.get("disk_usage_value") is not None
                            and e["disk_usage_value"] >= args.hard_ceiling)

            if zscore_hit or ceiling_hit:
                predicted_positive.add(ts)

        m = metrics_from_confusion(
            *confusion_for_predicted_set(predicted_positive, ground_truth, scored, require_baseline_ready)
        )
        m["theta_z"] = theta
        results.append(m)

    # --- Print table ---
    header = (f"{'theta_z':>8}{'TP':>7}{'FP':>9}{'TN':>9}{'FN':>7}"
              f"{'Precision':>12}{'Recall':>10}{'FPR':>10}{'F1':>10}")
    print(header)
    print("-" * len(header))
    for m in results:
        print(f"{m['theta_z']:>8.2f}{m['tp']:>7}{m['fp']:>9}{m['tn']:>9}"
              f"{m['fn']:>7}{m['precision']:>12.4f}{m['recall']:>10.4f}"
              f"{m['fpr']:>10.4f}{m['f1']:>10.4f}")

    print(f"\n{'STATIC 80%':>8}{static_metrics['tp']:>7}"
          f"{static_metrics['fp']:>9}{static_metrics['tn']:>9}"
          f"{static_metrics['fn']:>7}{static_metrics['precision']:>12.4f}"
          f"{static_metrics['recall']:>10.4f}{static_metrics['fpr']:>10.4f}"
          f"{static_metrics['f1']:>10.4f}")

    best_f1 = max(results, key=lambda m: m["f1"])
    print(f"\nBest F1 among tested theta_z values: "
          f"theta_z={best_f1['theta_z']} (F1={best_f1['f1']:.4f}, "
          f"Precision={best_f1['precision']:.4f}, "
          f"Recall={best_f1['recall']:.4f})")

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            fieldnames = ["theta_z", "tp", "fp", "tn", "fn",
                          "precision", "recall", "fpr", "accuracy", "f1"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in results:
                writer.writerow(m)
            static_row = dict(static_metrics)
            static_row["theta_z"] = "static_80pct"
            writer.writerow(static_row)
        print(f"\nSweep results written to {args.output_csv}")


if __name__ == "__main__":
    main()
