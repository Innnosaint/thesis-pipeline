# Enhanced Threshold-Based Alert Algorithm — Pipeline

Implementation of the disk usage anomaly detection pipeline described in
Chapter 3 of the thesis: Drain3 parsing -> Dynamic Baseline -> Z-Score
Detection -> Contextual Log Correlation -> Ranked Alert Output, plus a
real-time replay simulator for Objective 5 (live dataset testing).

## 1. Setup (WSL2 / Ubuntu)

```bash
# From Windows PowerShell (Admin), one-time only:
wsl --install -d Ubuntu-22.04

# Inside the Ubuntu terminal:
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3-pip python3-venv git

# Set up the project
cd ~/thesis
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Add your data

- Place raw Loghub log files (syslog, kern.log, etc.) into `data/raw/`
- Place Loghub-2.0 ground-truth labels into `data/ground_truth/labels.jsonl`

**Ground truth format expected** (one JSON object per line):
```json
{"log_source": "syslog", "timestamp": "2024-03-01T02:47:43", "is_anomaly": true}
```
Adjust `src/replay_simulator.py::load_ground_truth()` if your actual
Loghub-2.0 export uses a different schema (e.g., CSV, different keys) —
this is the one place you'll likely need to adapt to the exact file you
download.

## 3. Run the offline pipeline (Objectives 1-4)

```bash
python main.py
```

This runs the full batch pipeline: ingest -> preprocess -> baseline ->
Z-Score -> correlation -> ranked alerts, and writes:
- `data/processed/final_scored_stream.jsonl` — every entry with full scoring
- `logs/audit_trail.jsonl` — persistent audit trail (includes suppressed alerts)
- Console output — top 10 ranked alerts

## 4. Run the real-time replay + live evaluation (Objective 5)

```bash
cd src
python replay_simulator.py
```

This replays `data/processed/clean_stream.jsonl` (generated automatically
by `main.py` on its first run) in timestamp order, paced by
`config/settings.yaml -> replay.speed_multiplier`, and prints rolling
Precision/Recall/FPR every N entries (configurable via
`evaluation.rolling_window_for_metrics`).

- `speed_multiplier: 1` = true real-time pacing
- `speed_multiplier: 60` = 1 simulated hour per 1 real minute (default, for demos)
- `replay.max_lines` = cap entries for quick test runs; set to `null` for full dataset

Output:
- Console: live rolling metrics as the stream progresses
- `results/rolling_metrics_history.csv` — the full rolling-metrics curve, ready to plot
- `logs/audit_trail.jsonl` — overwritten with the real-time run's results

## 5. Tune detection parameters

All thresholds live in `config/settings.yaml` — no code changes needed:
- `baseline.window_size` — sliding window size (W)
- `zscore.theta_z` — anomaly sensitivity (default 2.0 std deviations)
- `correlation.delta_t_seconds` — contextual correlation window (±Δt)
- `correlation.benign_keywords` / `failure_keywords` — event signatures

## 6. Project structure

```
thesis-pipeline/
├── data/
│   ├── raw/              # put Loghub log files here
│   ├── ground_truth/     # put Loghub-2.0 labels here
│   └── processed/        # pipeline intermediate + final outputs
├── src/                  # one module per pipeline stage
├── config/settings.yaml  # all tunable parameters
├── logs/audit_trail.jsonl
├── results/              # metrics CSV, plots
├── notebooks/            # exploration notebook
└── main.py               # offline pipeline entry point
```

## 7. Plotting the rolling performance curve (optional)

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/rolling_metrics_history.csv")
df.plot(x="n", y=["precision", "recall", "fpr"], marker="o")
plt.title("Rolling Detection Performance Over Real-Time Replay")
plt.xlabel("Entries Processed")
plt.savefig("results/rolling_performance.png")
```
