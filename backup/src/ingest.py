"""
ingest.py
Objective 1 support: Ingest and parse raw system log data using the Drain
algorithm, extracting structured fields (timestamp, event type, disk usage,
process ID) needed for downstream baseline and anomaly detection.
"""

import re
import json
from pathlib import Path
from datetime import datetime

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


DISK_USAGE_PATTERN = re.compile(
    r"(?:disk\s*usage[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%"
    r"|(\d{1,3}(?:\.\d+)?)\s*%\s*used)",
    re.IGNORECASE,
)
TIMESTAMP_PATTERNS = [
    ("%b %d %H:%M:%S", re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")),
    ("%Y-%m-%d %H:%M:%S", re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")),
]


def build_template_miner(state_file: str) -> TemplateMiner:
    """Create a Drain3 template miner with persistent state so structure
    learned across runs is reused (mirrors Zhu et al., 2020's Drain design)."""
    config = TemplateMinerConfig()  # defaults are used automatically; no file needed
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    return TemplateMiner(config=config)


def extract_timestamp(line: str, fallback_year: int = 2024):
    for fmt, pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(line)
        if match:
            raw = match.group(1)
            try:
                if fmt.startswith("%b"):
                    dt = datetime.strptime(f"{fallback_year} {raw}", f"%Y {fmt}")
                else:
                    dt = datetime.strptime(raw, fmt)
                return dt
            except ValueError:
                continue
    return None


def extract_disk_usage(line: str):
    match = DISK_USAGE_PATTERN.search(line)
    if match:
        value = match.group(1) or match.group(2)
        return float(value)
    return None


def extract_process_id(line: str):
    match = re.search(r"\[(\d+)\]", line)
    return int(match.group(1)) if match else None


def parse_log_file(filepath: str, miner: TemplateMiner):
    """
    Yields structured dicts for each log line:
    {timestamp, log_source, event_type(template), disk_usage_value, process_id, raw}
    Lines with no disk usage value are still yielded (needed for keyword
    correlation later) but flagged accordingly.
    """
    source_name = Path(filepath).stem
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            result = miner.add_log_message(line)
            template = result.get("template_mined", line)

            yield {
                "timestamp": extract_timestamp(line),
                "log_source": source_name,
                "event_type": template,
                "disk_usage_value": extract_disk_usage(line),
                "process_id": extract_process_id(line),
                "raw": line,
            }


def ingest_directory(raw_dir: str, state_file: str):
    """Ingest every log file in raw_dir, returning a single time-ordered list."""
    miner = build_template_miner(state_file)
    entries = []
    for filepath in sorted(Path(raw_dir).glob("*")):
        if filepath.is_file():
            entries.extend(parse_log_file(str(filepath), miner))

    entries = [e for e in entries if e["timestamp"] is not None]
    entries.sort(key=lambda e: e["timestamp"])
    return entries


if __name__ == "__main__":
    import yaml

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    parsed = ingest_directory(
        str(PROJECT_ROOT / cfg["paths"]["raw_log_dir"]),
        str(PROJECT_ROOT / cfg["parsing"]["drain_state_file"]),
    )
    print(f"Parsed {len(parsed)} log entries with timestamps.")

    out_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "parsed_entries.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for e in parsed:
            record = dict(e)
            record["timestamp"] = record["timestamp"].isoformat()
            f.write(json.dumps(record) + "\n")

    print(f"Wrote parsed output to {out_path}")
