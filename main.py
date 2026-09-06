"""
main.py
Orchestrates the full offline pipeline end-to-end (Objectives 1-4):
ingest -> preprocess -> baseline -> zscore -> debounce -> correlation -> alert ranking.

For the real-time replay + live evaluation (Objective 5), run
src/replay_simulator.py instead (it re-uses the same baseline/zscore/
correlation logic in streaming form).
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import ingest_directory
from preprocess import preprocess_pipeline
from baseline import run_baseline_over_stream
from zscore_detector import classify
from debounce import apply_debounce
from log_correlation import correlate_and_score
from alert_scoring import rank_alerts, write_audit_log, print_top_alerts


def load_config(path="config/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def save_jsonl(entries, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            rec = dict(e)
            rec["timestamp"] = rec["timestamp"].isoformat()
            f.write(json.dumps(rec) + "\n")


def main():
    cfg = load_config()

    print("=== Step 1-2: Ingest & Parse (Drain3) ===")
    parsed = ingest_directory(cfg["paths"]["raw_log_dir"], cfg["parsing"]["drain_state_file"])
    print(f"Parsed {len(parsed)} entries.\n")

    print("=== Step 2: Preprocess (filter, dedup, segment) ===")
    clean = preprocess_pipeline(parsed)
    print(f"{len(clean)} entries remain after preprocessing.\n")
    save_jsonl(clean, Path(cfg["paths"]["processed_dir"]) / "clean_stream.jsonl")

    print("=== Step 3: Dynamic Baseline Calculation ===")
    baselined = run_baseline_over_stream(
        clean,
        window_size=cfg["baseline"]["window_size"],
        min_observations=cfg["baseline"]["min_observations"],
        sigma_floor=cfg["baseline"].get("sigma_floor", 0.0),
    )
    print("Baseline computed for all entries.\n")

    print("=== Step 4: Z-Score Anomaly Detection (with hybrid safety-net ceiling) ===")
    hard_ceiling = cfg["zscore"].get("hard_ceiling")
    classified = classify(baselined, theta_z=cfg["zscore"]["theta_z"], hard_ceiling=hard_ceiling)
    n_abnormal = sum(1 for e in classified if e["classification"] == "ABNORMAL")
    n_by_ceiling = sum(1 for e in classified if e.get("triggered_by") == "ceiling")
    print(f"{n_abnormal} entries flagged ABNORMAL out of {len(classified)} "
          f"({n_by_ceiling} via ceiling safety-net alone) (before debounce).\n")

    print("=== Step 4.5: Consecutive-Readings Debounce ===")
    min_consecutive = cfg.get("debounce", {}).get("min_consecutive", 1)
    classified = apply_debounce(classified, min_consecutive=min_consecutive)
    n_abnormal_after = sum(1 for e in classified if e["classification"] == "ABNORMAL")
    print(f"{n_abnormal_after} entries remain ABNORMAL after debounce "
          f"(min_consecutive={min_consecutive}).\n")

    print("=== Step 5: Contextual Log Correlation ===")
    scored = correlate_and_score(classified, classified, cfg)
    n_suppressed = sum(1 for e in scored if e.get("suppressed"))
    n_escalated = sum(1 for e in scored if e.get("priority") == "HIGH")
    print(f"{n_suppressed} suppressed (benign), {n_escalated} escalated (failure signature).\n")

    print("=== Step 6: Alert Ranking & Audit Log ===")
    alerts = rank_alerts(scored)
    print_top_alerts(alerts, n=10)
    write_audit_log(scored, cfg["paths"]["audit_log"])
    print(f"\nAudit trail written to {cfg['paths']['audit_log']}")

    save_jsonl(scored, Path(cfg["paths"]["processed_dir"]) / "final_scored_stream.jsonl")
    print(f"\nPipeline complete. {len(alerts)} actionable alerts generated.")


if __name__ == "__main__":
    main()
