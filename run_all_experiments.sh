#!/usr/bin/env bash
# Reproduces every table in Chapter 4. Put your raw log (e.g. Linux_full.log)
# in data/raw/ first, then:   bash run_all_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/src"
RAW=$(ls ../data/raw/*.log | head -1)
P=../data/processed; L=../data/ground_truth/labels.jsonl

echo "### 4.2 Ground truth (independent of the algorithm)"
python -u build_ground_truth.py --log "$RAW" --output $L

echo "### Final pipeline (theta_z=5.5, ceiling=85, W=50) + permissive stream for theta_sweep"
python -u run_pipeline.py
python -u run_pipeline.py --theta-z 1.5 --hard-ceiling 85 --output final_scored_stream_permissive.jsonl >/dev/null

echo "### Table 4.3 -- static vs enhanced"
python -u evaluate.py --scored $P/final_scored_stream.jsonl --labels $L --output-csv ../results/final_comparison.csv

echo "### Table 4.1 -- theta_z sweep (W=50, no ceiling)"
python -u theta_sweep.py --scored $P/final_scored_stream_permissive.jsonl --labels $L \
  --thetas 1.5 2.0 2.5 3.0 3.5 4.0 --output-csv ../results/theta_sweep.csv

echo "### 4.4 -- window-size sweep (theta_z=2.0)"
python -u window_sweep.py --clean-stream $P/clean_stream.jsonl --labels $L \
  --windows 10 15 20 30 50 75 100 150 200 --theta-z 2.0 --output-csv ../results/window_sweep.csv

echo "### Joint W x theta_z grid"
python -u window_x_theta_sweep.py --clean-stream $P/clean_stream.jsonl --labels $L \
  --windows 10 20 30 50 75 100 150 200 --thetas 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
  --output-csv ../results/grid_sweep.csv

echo "### Table 4.2 -- theta_z x ceiling sweep (W=50)"
python -u ceiling_sweep.py --clean-stream $P/clean_stream.jsonl --labels $L --window-size 50 \
  --thetas 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 --ceilings 80 82 85 88 90 95 none \
  --output-csv ../results/ceiling_sweep.csv

echo "### 4.3 -- false-positive diagnosis at default theta_z=2.0"
python -u run_pipeline.py --theta-z 2.0 --hard-ceiling none --output final_scored_stream_default.jsonl >/dev/null
python -u diagnose_false_positives.py --scored $P/final_scored_stream_default.jsonl --labels $L

echo "### 3.6.2 -- real-time replay (no lookahead)"
python -u run_pipeline.py >/dev/null   # restore final stream/audit trail
python -u replay_simulator.py --speed 1e9

echo "### Tables 4.1-4.3 regenerated + checked against the paper"
python -u make_tables.py | tee ../results/tables_check.txt

echo "### Supporting evidence for Sections 4.1-4.8"
python -u supporting_evidence.py | tee ../results/evidence_report.txt

echo "All done. CSVs in results/, audit trail in logs/."
