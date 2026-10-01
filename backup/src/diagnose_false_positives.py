"""
diagnose_false_positives.py

Inspects the entries the enhanced algorithm flagged as ABNORMAL but
that are actually NORMAL per ground truth (false positives), to
determine the root cause of an unexpectedly high false positive rate.

Specifically checks:
  1. How many false positives occurred while baseline_ready == False
     (i.e. during the baseline warm-up period, where sigma may still
     be a near-zero placeholder).
  2. The distribution of baseline_sigma among false positives vs.
     true positives vs. all entries, to detect a sigma-near-zero
     division blowup.
  3. Prints a handful of raw false-positive records for manual review.

Usage:
    python diagnose_false_positives.py \
        --scored ../data/processed/final_scored_stream.jsonl \
        --labels ../data/ground_truth/labels.jsonl
"""

import argparse
import json
import statistics


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


def summarize(values, label):
    if not values:
        print(f"  {label}: no data")
        return
    print(f"  {label}:")
    print(f"    n      = {len(values)}")
    print(f"    min    = {min(values):.6f}")
    print(f"    median = {statistics.median(values):.6f}")
    print(f"    mean   = {statistics.mean(values):.6f}")
    print(f"    max    = {max(values):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--sample-size", type=int, default=10,
                         help="Number of raw false-positive records to print")
    args = parser.parse_args()

    scored = load_jsonl(args.scored)
    ground_truth = load_jsonl(args.labels)

    all_sigmas = []
    fp_sigmas = []
    tp_sigmas = []
    fp_records = []

    fp_warmup_count = 0
    fp_total = 0
    tp_total = 0

    for ts, gt in ground_truth.items():
        entry = scored.get(ts)
        if entry is None:
            continue

        sigma = entry.get("baseline_sigma")
        if sigma is not None:
            all_sigmas.append(sigma)

        predicted = entry.get("classification") == "ABNORMAL"
        actual = gt["is_anomaly"]

        if predicted and not actual:
            fp_total += 1
            if sigma is not None:
                fp_sigmas.append(sigma)
            if entry.get("baseline_ready") is False:
                fp_warmup_count += 1
            if len(fp_records) < args.sample_size:
                fp_records.append(entry)

        if predicted and actual:
            tp_total += 1
            if sigma is not None:
                tp_sigmas.append(sigma)

    print("=== False Positive Diagnostic Report ===\n")
    print(f"Total false positives: {fp_total}")
    print(f"Total true positives:  {tp_total}\n")

    print(f"False positives occurring while baseline_ready == False: "
          f"{fp_warmup_count} / {fp_total} "
          f"({(fp_warmup_count / fp_total * 100) if fp_total else 0:.1f}%)\n")

    print("--- baseline_sigma distribution ---")
    summarize(all_sigmas, "All entries")
    summarize(fp_sigmas, "False positives")
    summarize(tp_sigmas, "True positives")

    print("\n--- Sample false positive records ---")
    for i, rec in enumerate(fp_records, start=1):
        print(f"\n[{i}] {json.dumps(rec, indent=2)}")

    print("\n=== Interpretation guide ===")
    print("- If most false positives have baseline_ready == False and/or")
    print("  a near-zero baseline_sigma, the fix is to either (a) skip")
    print("  classification entirely until baseline_ready == True, or")
    print("  (b) add an epsilon floor to sigma in the Z-Score denominator")
    print("  (e.g. sigma = max(sigma, some_minimum_value)).")
    print("- If false positive sigma values look similar to true positive")
    print("  sigma values (no near-zero cluster), the issue is more likely")
    print("  theta_z being too sensitive for this dataset's noise level,")
    print("  and raising theta_z in config/settings.yaml would help.")


if __name__ == "__main__":
    main()
