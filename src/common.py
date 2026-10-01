"""
common.py
Shared helpers for the sweep scripts (window / theta / ceiling): data loading,
confusion-matrix and metric computation, and the default correlation config.
Keeping these in one place guarantees every sweep scores results identically.
"""

import json
from datetime import datetime
from pathlib import Path

DEFAULT_CFG = {
    "correlation": {
        "delta_t_seconds": 60,
        "benign_keywords": ["backup", "rsync", "cron", "apt", "yum",
                            "package install", "scheduled task"],
        "failure_keywords": ["disk full", "i/o error", "write failure",
                             "inode exhausted", "filesystem read-only",
                             "device error", "no space left", "disk quota"],
    },
    "priority_weights": {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5},
}


def make_cfg(delta_t: int = 60) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CFG))  # deep copy
    cfg["correlation"]["delta_t_seconds"] = delta_t
    return cfg


def load_stream(path: str):
    """Loads a JSONL stream, parsing 'timestamp' into datetime."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
                entries.append(rec)
    return entries


def load_context(clean_stream_path: str, context_path: str = None):
    """Log lines used for Contextual Log Correlation (Eq. 3.6).
    Preference: explicit --context file, else parsed_entries.jsonl next to the
    clean stream (ALL log lines, so failure/benign keywords outside the
    diskmon lines can be found), else None (caller falls back to the
    disk-only clean stream)."""
    if context_path:
        return load_stream(context_path)
    guess = Path(clean_stream_path).parent / "parsed_entries.jsonl"
    if guess.exists():
        ctx = load_stream(str(guess))
        ctx.sort(key=lambda e: e["timestamp"])
        return ctx
    return None


def load_labels(path: str):
    """Returns dict keyed by (log_source, timestamp_iso) -> bool."""
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels[(rec["log_source"], rec["timestamp"])] = bool(rec["is_anomaly"])
    return labels


def compute_confusion(scored_entries, labels):
    """TP/FP/TN/FN of non-suppressed ABNORMAL predictions vs ground truth.
    Warm-up entries (baseline_ready == False) and unlabeled entries are skipped."""
    tp = fp = tn = fn = 0
    for e in scored_entries:
        if not e.get("baseline_ready", False):
            continue
        true_label = labels.get((e["log_source"], e["timestamp"].isoformat()))
        if true_label is None:
            continue
        pred = e["classification"] == "ABNORMAL" and not e.get("suppressed", False)
        if pred and true_label:
            tp += 1
        elif pred:
            fp += 1
        elif true_label:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def write_csv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
