"""
evaluate.py

Joins the enhanced algorithm's scored output against the ground truth
labels and computes:

  1. Precision, Recall, and False Positive Rate for the ENHANCED
     algorithm (dynamic baseline + Z-Score + contextual correlation).
  2. The same metrics for a STATIC 80% THRESHOLD baseline, computed
     directly from disk_usage_value, for direct comparison.
  3. A side-by-side summary table.

IMPORTANT: An entry counts as a "predicted positive" (actionable alert)
for the Enhanced Algorithm only if classification == "ABNORMAL" AND the
entry was NOT suppressed by Contextual Log Correlation as a benign
event. Ignoring the suppressed flag would count silently-suppressed
alerts (e.g. a scheduled backup correctly identified as benign) as if
they were active detections -- this was a bug in an earlier version of
this script and has been fixed here.

Entries where baseline_ready == False (insufficient history for a
valid baseline) are excluded by default, consistent with how the main
pipeline and window_x_theta_sweep.py treat the warm-up period.

This gives the empirical figures for Chapter 4 / Objective 1 (baseline
weaknesses) and the overall comparative evaluation described in
Section 3.1 (Research Design).

Usage:
    python evaluate.py \
        --scored ../data/processed/final_scored_stream.jsonl \
        --labels ../data/ground_truth/labels.jsonl \
        --static-threshold 80.0
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


def compute_confusion(pred_positive_ts, ground_truth, scored, require_baseline_ready=True):
    """
    pred_positive_ts: set of timestamps the detector flagged as anomalous
    ground_truth: dict[timestamp] -> {"is_anomaly": bool, ...}
    scored: dict[timestamp] -> scored entry, used to check baseline_ready

    Only evaluates timestamps present in ground_truth AND (by default)
    where baseline_ready == True in the scored stream, since entries
    during the warm-up period have no valid baseline to judge against.
    """
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


def safe_div(a, b):
    return a / b if b else 0.0


def print_report(name, tp, fp, tn, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)

    print(f"\n=== {name} ===")
    print(f"  True Positives  (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  ------------------------------")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  FPR       : {fpr:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")

    return {
        "name": name, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "fpr": fpr, "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True,
                         help="Path to final_scored_stream.jsonl")
    parser.add_argument("--labels", required=True,
                         help="Path to ground_truth/labels.jsonl")
    parser.add_argument("--static-threshold", type=float, default=80.0,
                         help="Static threshold percentage for baseline comparison")
    parser.add_argument("--output-csv", default=None,
                         help="Optional path to write a summary CSV")
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
    print(f"  {len(ground_truth)} labeled entries loaded.")

    matched = sum(1 for ts in ground_truth if ts in scored)
    print(f"  {matched} / {len(ground_truth)} ground truth timestamps "
          f"matched against the scored stream.")

    if matched < len(ground_truth):
        missing = len(ground_truth) - matched
        print(f"  WARNING: {missing} ground truth entries have no "
              f"matching scored entry. These are excluded from the "
              f"confusion matrix -- check timestamp/log_source formatting "
              f"if this number is large.")

    if not require_baseline_ready:
        print("  NOTE: --include-warmup set; entries with baseline_ready=False are included.")

    # --- Enhanced algorithm predictions ---
    # A predicted positive is an actionable alert: classification ==
    # ABNORMAL AND not suppressed by Contextual Log Correlation.
    enhanced_positive = {
        ts for ts, e in scored.items()
        if e.get("classification") == "ABNORMAL"
        and not e.get("suppressed", False)
    }
    enhanced_results = compute_confusion(enhanced_positive, ground_truth, scored, require_baseline_ready)

    # --- Static 80% threshold baseline predictions ---
    # No suppression concept applies to the static baseline.
    static_positive = {
        ts for ts, e in scored.items()
        if e.get("disk_usage_value") is not None
        and e["disk_usage_value"] >= args.static_threshold
    }
    static_results = compute_confusion(static_positive, ground_truth, scored, require_baseline_ready)

    r1 = print_report("Enhanced Algorithm (Dynamic Baseline + Z-Score + Correlation)",
                       *enhanced_results)
    r2 = print_report(f"Static Threshold Baseline (>= {args.static_threshold}%)",
                       *static_results)

    print("\n=== Summary Comparison ===")
    print(f"{'Metric':<12}{'Static':>12}{'Enhanced':>12}{'Delta':>12}")
    for key in ["precision", "recall", "fpr", "accuracy"]:
        delta = r1[key] - r2[key]
        print(f"{key:<12}{r2[key]:>12.4f}{r1[key]:>12.4f}{delta:>+12.4f}")

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "tp", "fp", "tn", "fn",
                               "precision", "recall", "fpr", "accuracy"]
            )
            writer.writeheader()
            writer.writerow(r2)
            writer.writerow(r1)
        print(f"\nSummary written to {args.output_csv}")


if __name__ == "__main__":
    main()
