#!/usr/bin/env bash
# Rebuttal rerun queue: every external-cohort analysis on the corrected BRFSS (v2).
# Waits for 33 (results_v2, results_sr) and 56 to finish, then runs readouts and sweeps.
set -u
cd "$(dirname "$0")"
PY=/c/Users/Asus/miniconda3/python.exe
export BRFSS_PATH="D:/Projects/diabetes_prediction_project/federated/data/brfss_final_v2.csv"
export FL_RESULTS_SUBDIR=results_v2 PYTHONUNBUFFERED=1
L=results_v2

running() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*$1*' }).Count" | tr -d '\r'; }
while [ "$(running 33_multiseed)" != "0" ] || [ "$(running 56_budget)" != "0" ]; do sleep 60; done
echo "[queue] training done $(date)" >> $L/queue.log

run() { echo "[queue] start $* $(date)" >> $L/queue.log; $PY -W ignore "$@" > "$L/log_${1%%_*}${3:+_$3}.txt" 2>&1; echo "[queue] exit $? $* $(date)" >> $L/queue.log; }

run 35_delong_deltas.py
run 44_clinical_translation.py
run 47_xauc.py
run 48_capacity_curve.py
run 57_budget_bn_readout.py
run 61_label_holdout_readout.py results_v2 results_sr
run 59_rebuttal_threshold.py --boot 2000 &
run 43_partition_sweep.py --seeds 5 &
run 49_prevalence_sweep.py --seeds 10 &
wait
run 52_covariate_sweep.py &
run 52_covariate_sweep.py --covariate RIDAGEYR --out covariate_sweep_age.json &
run 53_factorial.py &
wait
FL_RESULTS_SUBDIR=results_sr $PY -W ignore 59_rebuttal_threshold.py --boot 200 > results_sr/log_59.txt 2>&1
echo "[queue] ALL DONE $(date)" >> $L/queue.log
