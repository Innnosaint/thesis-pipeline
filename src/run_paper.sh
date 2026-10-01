#!/usr/bin/env bash
# run_paper.sh -- reproduces every results table of the defense paper.
# Run from  ~/thesis/src  with the venv active:   bash run_paper.sh
set -e
LOG=../data/raw/Linux_full.log
LAB=../data/ground_truth/labels.jsonl
CLEAN=../data/processed/clean_stream.jsonl
CTX=../data/processed/parsed_entries.jsonl
mkdir -p ../results ../data/ground_truth

echo "##### 0. Preflight: files the pipeline needs"
for f in ingest.py preprocess.py debounce.py common.py baseline.py zscore_detector.py log_correlation.py \
         alert_scoring.py build_ground_truth.py run_pipeline.py theta_sweep.py window_sweep.py \
         window_x_theta_sweep.py ceiling_sweep.py paper_check.py bench_correlation.py; do
  [ -f "$f" ] && echo "ok       $f" || { echo "MISSING  $f"; MISSING=1; }
done
grep -q preprocess_pipeline preprocess.py || { echo "preprocess.py is not your original (no preprocess_pipeline)"; MISSING=1; }
[ -z "$MISSING" ] || { echo "Fix the items above, then rerun."; exit 1; }
[ -f "$LOG" ] || { echo "Missing $LOG (the merged Loghub + synthetic diskmon log)"; exit 1; }
sha256sum "$LOG"

echo; echo "##### 1. Section 4.2 ground truth (expect 45 anomalies: 39 threshold + 6 keyword)"
python build_ground_truth.py --log "$LOG" --output "$LAB"

echo; echo "##### 2. Permissive run (theta_z=1.5, no ceiling) for Table 4.1"
python run_pipeline.py --theta-z 1.5 --hard-ceiling none --window-size 50 \
       --output final_scored_stream_permissive.jsonl

echo; echo "##### 3. Table 4.1  theta_z sweep (W=50, no ceiling) + STATIC 80% row"
python theta_sweep.py --scored ../data/processed/final_scored_stream_permissive.jsonl --labels "$LAB" \
       --thetas 1.5 2.0 2.5 3.0 3.5 4.0 --static-threshold 80 --output-csv ../results/table_4_1.csv

echo; echo "##### 4. FINAL configuration run (theta_z=5.5, ceiling=85, W=50): audit log + final_scored_stream.jsonl"
python run_pipeline.py --theta-z 5.5 --hard-ceiling 85 --window-size 50

echo; echo "##### 5. Table 4.2  window-size sweep (theta_z=2.0, no ceiling)"
python window_sweep.py --clean-stream "$CLEAN" --labels "$LAB" --context "$CTX" \
       --windows 10 15 20 30 50 75 100 150 200 --theta-z 2.0 --output-csv ../results/table_4_2.csv

echo; echo "##### 6. Section 4.4  W x theta_z grid, 64 combinations, no ceiling (expect best F1 0.4211, 12 TP, 0 FP)"
python window_x_theta_sweep.py --clean-stream "$CLEAN" --labels "$LAB" --context "$CTX" \
       --windows 10 20 30 50 75 100 150 200 --thetas 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
       --output-csv ../results/grid_sweep.csv --top-n 15

echo; echo "##### 7. Table 4.3  joint theta_z x ceiling sweep (63 combinations)"
python ceiling_sweep.py --clean-stream "$CLEAN" --labels "$LAB" --context "$CTX" --window-size 50 \
       --thetas 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 --ceilings 80 82 85 88 90 95 none \
       --output-csv ../results/table_4_3_ceiling_sweep.csv --top-n 15
echo; echo "--- the 11 rows printed in Table 4.3 ---"
python - << 'PY'
import csv
want=[(4.0,'85'),(4.5,'85'),(5.0,'85'),(5.5,'85'),(6.0,'85'),(5.0,'90'),(5.0,'none'),(5.5,'80'),(5.5,'82'),(5.5,'88'),(5.5,'none')]
rows={(float(r['theta_z']),r['hard_ceiling']):r for r in csv.DictReader(open('../results/table_4_3_ceiling_sweep.csv'))}
print(f"{'theta_z':>7} {'ceiling':>8} {'TP':>4} {'FP':>6} {'FN':>4} {'Prec':>7} {'Recall':>7} {'F1':>7}")
for k in want:
    r=rows[k]; print(f"{k[0]:>7.1f} {k[1]:>8} {r['TP']:>4} {r['FP']:>6} {r['FN']:>4} {r['precision']:>7} {r['recall']:>7} {r['f1']:>7}")
PY

echo; echo "##### 8. Sections 4.1-4.7, Table 4.4, Table 4.5 (bootstrap)"
python paper_check.py --final ../data/processed/final_scored_stream.jsonl --labels "$LAB"

echo; echo "##### 9. Section 4.7 runtime claim (windowed lookup vs full scan)"
python bench_correlation.py

echo; echo "All done. CSVs are in ../results/"
# OPTIONAL (Section 4.7 streaming replay: 9.95 s, ~10,500 readings/s, 319 MB peak). I have not seen your
# replay_simulator.py arguments, so run   python replay_simulator.py --help   then, for example:
#   /usr/bin/time -v python replay_simulator.py <your args> 2>&1 | grep -E "Elapsed|Maximum resident|TP|FP|FN"
