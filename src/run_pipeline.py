"""
run_pipeline.py
One-command run of the full Enhanced Threshold-Based Alert Algorithm
(Section 3.2.1, Steps 1-6):

  1 ingest (Drain3) -> 2 preprocess -> 3 dynamic baseline -> 4 Z-Score (+ hybrid
  ceiling, Eq. 4.1) -> 5 contextual log correlation -> 6 ranked alerts + audit trail

Writes every intermediate stage to data/processed/ and the final result to
data/processed/final_scored_stream.jsonl (the file evaluate.py, theta_sweep.py
and diagnose_false_positives.py read).

Usage (final configuration from config/settings.yaml):
  python run_pipeline.py
Override parameters, e.g. the permissive stream used by theta_sweep.py:
  python run_pipeline.py --theta-z 1.5 --hard-ceiling 85 \
      --output final_scored_stream_permissive.jsonl
Reproduce the pre-fix behaviour (correlate against diskmon lines only):
  python run_pipeline.py --disk-only-context
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from ingest import ingest_directory
from preprocess import preprocess_pipeline
from baseline import run_baseline_over_stream
from zscore_detector import classify
from log_correlation import correlate_and_score
from alert_scoring import rank_alerts, write_audit_log, print_top_alerts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dump(entries, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            rec = dict(e)
            rec["timestamp"] = rec["timestamp"].isoformat()
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config" / "settings.yaml"))
    ap.add_argument("--theta-z", type=float, default=None)
    ap.add_argument("--hard-ceiling", default=None, help="percentage, or 'none' to disable")
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument("--output", default="final_scored_stream.jsonl",
                    help="file name inside processed_dir")
    ap.add_argument("--disk-only-context", action="store_true",
                    help="correlate against the disk-usage-bearing stream only")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]

    theta = a.theta_z if a.theta_z is not None else cfg["zscore"]["theta_z"]
    if a.hard_ceiling is None:
        ceiling = cfg["zscore"].get("hard_ceiling")
    else:
        ceiling = None if a.hard_ceiling.lower() == "none" else float(a.hard_ceiling)
    W = a.window_size or cfg["baseline"]["window_size"]

    print(f"[1/6] Ingesting + Drain3 parsing from {cfg['paths']['raw_log_dir']} ...")
    parsed = ingest_directory(str(PROJECT_ROOT / cfg["paths"]["raw_log_dir"]),
                              str(PROJECT_ROOT / cfg["parsing"]["drain_state_file"]))
    print(f"      {len(parsed)} timestamped log lines parsed")
    dump(parsed, proc / "parsed_entries.jsonl")

    print("[2/6] Preprocessing (filter, dedup, segment) ...")
    clean = preprocess_pipeline(parsed)
    print(f"      {len(clean)} disk-usage readings kept")
    if not clean:
        sys.exit("No disk usage readings found -- check the raw log format "
                 "(expects e.g. 'diskmon[123]: /dev/hda1 45.0% used').")
    dump(clean, proc / "clean_stream.jsonl")

    print(f"[3/6] Dynamic baseline (W={W}) ...")
    baselined = run_baseline_over_stream(
        [dict(e) for e in clean], window_size=W,
        min_observations=cfg["baseline"]["min_observations"],
        sigma_floor=cfg["baseline"].get("sigma_floor", 0.0))
    dump(baselined, proc / "baseline_stream.jsonl")

    print(f"[4/6] Z-Score detection (theta_z={theta}, hard_ceiling={ceiling}) ...")
    classified = classify(baselined, theta_z=theta, hard_ceiling=ceiling)
    n_abn = sum(e["classification"] == "ABNORMAL" for e in classified)
    print(f"      {n_abn} readings flagged ABNORMAL")
    dump(classified, proc / "classified_stream.jsonl")

    print("[5/6] Contextual log correlation "
          f"(+/-{cfg['correlation']['delta_t_seconds']}s, "
          f"{'disk-only' if a.disk_only_context else 'all log lines'}) ...")
    context = classified if a.disk_only_context else parsed
    scored = correlate_and_score(classified, context, cfg)
    n_supp = sum(bool(e.get("suppressed")) for e in scored)
    n_high = sum(e.get("priority") == "HIGH" for e in scored)
    print(f"      {n_supp} suppressed as benign, {n_high} escalated HIGH")
    dump(scored, proc / "scored_stream.jsonl")
    dump(scored, proc / a.output)

    print("[6/6] Ranking alerts + writing audit trail ...")
    alerts = rank_alerts(scored)
    print_top_alerts(alerts, a.top)
    write_audit_log(scored, str(PROJECT_ROOT / cfg["paths"]["audit_log"]))
    print(f"\nFinal stream : {proc / a.output}\nAudit trail  : {cfg['paths']['audit_log']}")


if __name__ == "__main__":
    main()
