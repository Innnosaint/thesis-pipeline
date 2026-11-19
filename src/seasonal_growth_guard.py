"""Experimental Seasonal-Reference Growth Guard (SRGG).

Applies an optional, additive detection condition to *chronologically sorted*
entries already classified by the existing zscore_detector.classify().

No new metric is inferred from logs. The guard uses a causal, evenly sampled
five-minute disk usage series, one-hour means, and same-time historic means.
The algorithm does not read future observations or truth labels.

This is an experimental candidate, not a proven production detector.
"""
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite


@dataclass(frozen=True)
class SRGGConfig:
    interval_minutes: int = 5
    window_readings: int = 12  # one-hour smoothing at 5-minute intervals
    reference_days: tuple[int, ...] = (1, 2, 3)
    comparison_hours: int = 1
    seasonal_diff_pp: float = 11.0
    hourly_growth_pp: float = 3.0
    min_usage_pct: float = 55.0
    persistence_readings: int = 3
    cadence_tolerance_seconds: int = 30

    def validate(self):
        if self.interval_minutes <= 0 or self.window_readings <= 0:
            raise ValueError("Sampling interval and smoothing window must be positive")
        if self.persistence_readings < 1:
            raise ValueError("persistence_readings must be >= 1")
        if not self.reference_days or any(int(d) != d or d < 1 for d in self.reference_days):
            raise ValueError("reference_days must contain positive integer days")
        if self.comparison_hours * 60 % self.interval_minutes != 0:
            raise ValueError("comparison_hours must align with interval_minutes")
        if self.seasonal_diff_pp <= 0 or self.hourly_growth_pp <= 0:
            raise ValueError("Deviation and growth thresholds must be positive")


def apply_srgg(entries, cfg=SRGGConfig(), enabled=True):
    """Modify entries in place. Preserve the old detector's positive alerts.

    Adds srgg_eligible/reference/deviation/hourly_growth/streak/alert/
    new_detection fields and, for SRGG-only hits, changes classification to
    ABNORMAL and triggered_by to 'srgg'. Assumes datetime timestamps.

    If readings are missing or irregular, the rolling smoothing resets and
    no seasonal guard is generated until sufficient usable history exists.
    """
    cfg.validate()
    step = timedelta(minutes=cfg.interval_minutes)
    hour_lag = timedelta(hours=cfg.comparison_hours)
    recent = deque(maxlen=cfg.window_readings)
    smoothed_by_time = {}
    previous_dt = None
    streak = 0

    for row in entries:
        t = row['timestamp']
        if not isinstance(t, datetime):
            raise TypeError('timestamp must be datetime; parse uploaded JSONL before applying SRGG')
        x = float(row['disk_usage_value'])
        if not isfinite(x):
            raise ValueError(f'Non-finite disk usage at {t}')
        if previous_dt is not None:
            dt_seconds = (t - previous_dt).total_seconds()
            if abs(dt_seconds - step.total_seconds()) > cfg.cadence_tolerance_seconds:
                recent.clear()
                streak = 0
        previous_dt = t
        recent.append(x)
        ready = len(recent) == cfg.window_readings
        smoothed = sum(recent) / cfg.window_readings if ready else None
        if ready:
            smoothed_by_time[t] = smoothed

        reference_values = [smoothed_by_time.get(t-timedelta(days=d))
                            for d in cfg.reference_days]
        prev_hour = smoothed_by_time.get(t-hour_lag)
        eligible = (ready and prev_hour is not None and
                    all(v is not None for v in reference_values))
        reference = sorted(reference_values)[len(reference_values)//2] if eligible else None
        # For even-sized reference sets, use the arithmetic median.
        if eligible and len(reference_values) % 2 == 0:
            mid = len(reference_values)//2
            vals = sorted(reference_values)
            reference = (vals[mid-1]+vals[mid])/2
        seasonal_delta = smoothed - reference if eligible else None
        hourly_growth = smoothed - prev_hour if eligible else None
        qualifying = bool(eligible and seasonal_delta >= cfg.seasonal_diff_pp
                          and hourly_growth >= cfg.hourly_growth_pp
                          and x >= cfg.min_usage_pct)
        streak = streak + 1 if qualifying else 0
        guard = bool(enabled and qualifying and streak >= cfg.persistence_readings)
        previous_alert = row.get('classification') == 'ABNORMAL'
        row['srgg_eligible'] = bool(eligible)
        row['srgg_reference'] = reference
        row['srgg_seasonal_delta'] = seasonal_delta
        row['srgg_hourly_growth'] = hourly_growth
        row['srgg_streak'] = streak
        row['srgg_alert'] = guard
        row['srgg_new_detection'] = bool(guard and not previous_alert)
        if guard and not previous_alert:
            row['classification'] = 'ABNORMAL'
            row['triggered_by'] = 'srgg'
            row['severity_score'] = max(
                float(row.get('severity_score') or 0.0),
                seasonal_delta / cfg.seasonal_diff_pp,
                hourly_growth / cfg.hourly_growth_pp)
    return entries
