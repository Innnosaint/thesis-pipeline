"""
alert_scoring.py
Final alert output stage: ranks non-suppressed ABNORMAL alerts in descending
order of Enhanced Alert Score and writes the persistent audit trail
(including suppressed LOW priority events, per Section 3.2.1 Step 6).
"""

import json
from pathlib import Path


def rank_alerts(scored_entries):
    """Returns only non-suppressed ABNORMAL alerts, ranked by score desc."""
    alerts = [e for e in scored_entries if e["classification"] == "ABNORMAL" and not e.get("suppressed")]
    alerts.sort(key=lambda e: e["enhanced_alert_score"], reverse=True)
    return alerts


def write_audit_log(all_entries, audit_log_path: str):
    """Writes every processed entry (including NORMAL and suppressed) to a
    persistent JSONL audit trail for post-hoc review."""
    Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(audit_log_path, "w") as f:
        for e in all_entries:
            record = {
                "timestamp": e["timestamp"].isoformat() if hasattr(e["timestamp"], "isoformat") else e["timestamp"],
                "log_source": e.get("log_source"),
                "disk_usage_value": e.get("disk_usage_value"),
                "zscore": e.get("zscore"),
                "classification": e.get("classification"),
                "priority": e.get("priority"),
                "matched_keyword": e.get("matched_keyword"),
                "enhanced_alert_score": e.get("enhanced_alert_score"),
                "suppressed": e.get("suppressed"),
            }
            f.write(json.dumps(record) + "\n")


def print_top_alerts(alerts, n: int = 10):
    print(f"\nTop {min(n, len(alerts))} alerts by Enhanced Alert Score:\n")
    for i, a in enumerate(alerts[:n], 1):
        ts = a["timestamp"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else ts
        print(
            f"{i:>2}. [{a['priority']:<6}] score={a['enhanced_alert_score']:.2f} "
            f"z={a['zscore']:.2f} disk={a['disk_usage_value']}% @ {ts_str} "
            f"kw={a.get('matched_keyword')}"
        )


if __name__ == "__main__":
    import yaml
    from datetime import datetime

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    in_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "scored_stream.jsonl"
    entries = []
    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)

    alerts = rank_alerts(entries)
    print_top_alerts(alerts)
    write_audit_log(entries, str(PROJECT_ROOT / cfg["paths"]["audit_log"]))
    print(f"\nFull audit trail written to {cfg['paths']['audit_log']}")
