"""
baseline.py
Objective 2 & 3 support: Dynamic Baseline Calculation using sliding window
mean and standard deviation (Equations 3.2-3.3), continuously updated to
adapt to evolving workload patterns and time-of-day variations.
"""

from collections import deque
import math


class DynamicBaseline:
    """
    Maintains a rolling window of the W most recent disk usage observations
    and computes mu(t), sigma(t) on demand. Designed for streaming / online
    use: call update() once per new observation, then get_baseline().
    """

    def __init__(self, window_size: int = 50, min_observations: int = 10, sigma_floor: float = 0.0):
        self.window_size = window_size
        self.min_observations = min_observations
        self.sigma_floor = sigma_floor  # minimum meaningful sigma (percentage points)
        self.window = deque(maxlen=window_size)

    def update(self, value: float):
        self.window.append(value)

    def is_ready(self) -> bool:
        return len(self.window) >= self.min_observations

    def get_baseline(self):
        """Returns (mu, sigma). sigma is floored to self.sigma_floor (if set)
        to prevent hypersensitivity during naturally low-variance periods,
        where trivial fluctuations would otherwise produce inflated Z-scores.
        Always additionally floored at 1e-6 to guard against true
        division-by-zero regardless of the configured sigma_floor."""
        n = len(self.window)
        if n == 0:
            return 0.0, max(self.sigma_floor, 1e-6)

        mu = sum(self.window) / n
        variance = sum((x - mu) ** 2 for x in self.window) / n
        sigma = math.sqrt(variance)

        return mu, max(sigma, self.sigma_floor, 1e-6)


def run_baseline_over_stream(entries, window_size: int = 50, min_observations: int = 10,
                               sigma_floor: float = 0.0):
    """
    Applies DynamicBaseline sequentially over a time-ordered stream of entries
    (each dict must have 'disk_usage_value'). Attaches mu(t), sigma(t) to each
    entry as computed *before* incorporating the current observation, so the
    baseline reflects only past data (causal/online, no lookahead).

    sigma_floor: minimum sigma value (in the same units as disk_usage_value,
    e.g. percentage points) to prevent hypersensitivity during unusually
    stable/low-variance stretches of the data. Default 0.0 preserves prior
    behavior (only the 1e-6 numerical guard applies).
    """
    baseline = DynamicBaseline(window_size=window_size, min_observations=min_observations,
                                sigma_floor=sigma_floor)
    enriched = []

    for e in entries:
        mu, sigma = baseline.get_baseline()
        e["baseline_mu"] = mu
        e["baseline_sigma"] = sigma
        e["baseline_ready"] = baseline.is_ready()

        baseline.update(e["disk_usage_value"])
        enriched.append(e)

    return enriched


if __name__ == "__main__":
    import json
    import yaml
    from pathlib import Path
    from datetime import datetime

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    in_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "clean_stream.jsonl"
    entries = []
    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
            entries.append(rec)

    enriched = run_baseline_over_stream(
        entries,
        window_size=cfg["baseline"]["window_size"],
        min_observations=cfg["baseline"]["min_observations"],
        sigma_floor=cfg["baseline"].get("sigma_floor", 0.0),
    )

    out_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "baseline_stream.jsonl"
    with open(out_path, "w") as f:
        for e in enriched:
            rec = dict(e)
            rec["timestamp"] = rec["timestamp"].isoformat()
            f.write(json.dumps(rec) + "\n")

    print(f"Computed rolling baseline for {len(enriched)} entries -> {out_path}")
