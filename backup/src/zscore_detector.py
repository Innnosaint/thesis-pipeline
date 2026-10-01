"""
zscore_detector.py
Objective 3 support: Z-Score Anomaly Detection (Equations 3.4-3.5), extended
with an optional hybrid safety-net ceiling rule.

Rationale for the safety-net rule (see Chapter 4 diagnostic findings):
on real Loghub Linux diskmon data, the dynamic baseline can "absorb" a slow,
sustained climb in disk usage into its own moving average before the
Z-Score crosses theta_z -- since mu(t) rises alongside the genuine anomaly
itself. This caused a measurable recall gap versus a naive static threshold,
which never adapts and therefore never misses an absolute-ceiling breach.

The hybrid rule below classifies an observation as ABNORMAL if EITHER:
  (a) |Z(t)| >= theta_z  (the original statistical deviation rule), OR
  (b) disk_usage_value >= hard_ceiling  (an absolute safety-net ceiling)

This does not replace or weaken the Z-Score mechanism -- condition (a) is
unchanged and still fires independently for genuine statistical anomalies
below the ceiling (e.g. a sudden spike from 40% to 60%). Condition (b) is
a strictly additive OR: it can only ever flag MORE observations as
ABNORMAL than Z-Score alone, never fewer, so it cannot suppress a
detection the pure Z-Score rule would have made. Setting hard_ceiling to
None (or a very high value like 100.1) disables the safety net entirely,
recovering the original pure Z-Score behavior for A/B comparison.
"""


def compute_zscore(value: float, mu: float, sigma: float) -> float:
    return (value - mu) / sigma


def classify(entries, theta_z: float = 2.0, hard_ceiling: float = None):
    """
    Adds 'zscore', 'classification' (NORMAL/ABNORMAL), 'severity_score',
    and 'triggered_by' (one of "zscore", "ceiling", "both", or None) to
    each entry. Entries where the baseline is not yet ready (insufficient
    history) are classified NORMAL by default and flagged 'baseline_ready'=False
    so they can be excluded from precision/recall scoring if desired.

    hard_ceiling: if set (e.g. 85.0), any entry with disk_usage_value >=
    hard_ceiling is classified ABNORMAL regardless of its Z-Score, acting
    as a safety net against baseline drift on slow, sustained climbs.
    Pass None to disable (pure Z-Score behavior, unchanged from before).
    """
    for e in entries:
        ceiling_hit = (hard_ceiling is not None
                        and e.get("disk_usage_value") is not None
                        and e["disk_usage_value"] >= hard_ceiling)

        if not e.get("baseline_ready", False):
            # Even during warm-up, the safety-net ceiling still applies --
            # a reading at 95% disk usage is dangerous regardless of
            # whether enough history exists yet for a statistical baseline.
            e["zscore"] = 0.0
            if ceiling_hit:
                e["classification"] = "ABNORMAL"
                e["severity_score"] = (e["disk_usage_value"] - hard_ceiling) + theta_z
                e["triggered_by"] = "ceiling"
            else:
                e["classification"] = "NORMAL"
                e["severity_score"] = 0.0
                e["triggered_by"] = None
            continue

        z = compute_zscore(e["disk_usage_value"], e["baseline_mu"], e["baseline_sigma"])
        e["zscore"] = z
        zscore_hit = abs(z) >= theta_z

        if zscore_hit and ceiling_hit:
            e["classification"] = "ABNORMAL"
            e["triggered_by"] = "both"
        elif zscore_hit:
            e["classification"] = "ABNORMAL"
            e["triggered_by"] = "zscore"
        elif ceiling_hit:
            e["classification"] = "ABNORMAL"
            e["triggered_by"] = "ceiling"
        else:
            e["classification"] = "NORMAL"
            e["triggered_by"] = None

        e["severity_score"] = abs(z)

    return entries


if __name__ == "__main__":
    import json
    import yaml
    from pathlib import Path
    from datetime import datetime

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    in_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "baseline_stream.jsonl"
    entries = []
    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)

    hard_ceiling = cfg["zscore"].get("hard_ceiling")
    classified = classify(entries, theta_z=cfg["zscore"]["theta_z"], hard_ceiling=hard_ceiling)

    n_abnormal = sum(1 for e in classified if e["classification"] == "ABNORMAL")
    n_by_ceiling = sum(1 for e in classified if e.get("triggered_by") == "ceiling")
    print(f"Classified {len(classified)} entries: {n_abnormal} flagged ABNORMAL "
          f"({n_by_ceiling} via ceiling safety-net alone).")

    out_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "classified_stream.jsonl"
    with open(out_path, "w") as f:
        for e in classified:
            rec = dict(e)
            rec["timestamp"] = rec["timestamp"].isoformat()
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote classified stream to {out_path}")
