#!/usr/bin/env bash
# Single entry point: full pipeline in the order used for the paper (replaces the original queue.sh / queue3.sh).
# Run from anywhere:  bash scripts/run_all.sh        Every step skips outputs that already exist, so it is resumable.
# Needs: raw data (python scripts/download_data.py), pip install -r requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
export TF_ENABLE_ONEDNN_OPTS=0          # oneDNN OFF for all runs unless stated (bitwise-closer run-to-run behaviour)
PY=${PYTHON:-python3}

echo "== 1. cached datasets (v2 + causal front ends, INCART) and beat times"
$PY code/prep.py
$PY code/beattimes.py

echo "== 2. DS1 4-fold CV for lambda selection (folds 0-3, seeds 42 and 1)"
for s in 42 1; do for k in 0 1 2 3; do $PY code/run_cv.py $k $s; done; done

echo "== 3. CV selection rule -> results/cv_selection.json"
$PY code/analyze_cv.py

echo "== 4. final DS1->DS2 models, both front ends, seeds 42 and 1..9 (oneDNN off)"
for s in 42 1 2 3 4 5 6 7 8 9; do $PY code/run_final.py v2 $s; done
for s in 42 1 2 3 4 5 6 7 8 9; do $PY code/run_final.py causal $s; done
echo "== 4b. oneDNN-ON variance study (v2, seeds 42 1 2 3 4) -> results/final_v2dnn_s*.npz"
for s in 42 1 2 3 4; do TAG=dnn TF_ENABLE_ONEDNN_OPTS=1 $PY code/run_final.py v2 $s; done

echo "== 5. 20 random patient-wise splits (v2 front end)"
for i in $(seq 0 19); do $PY code/run_split.py $i; done

echo "== 6. INT8 quantisation of causal models + TFLite Micro host check (needs the tflite-micro pip package)"
$PY code/quant_causal.py

echo "== 7. prior analyses and all numbers"
$PY code/analyze_prior.py cv
$PY code/analyze_prior.py ds2
$PY code/analyze_all.py

echo "== 8. LaTeX tables and figures -> paper_out/"
$PY code/gen_tables.py
$PY code/make_figs_v3.py
echo ALL_DONE
