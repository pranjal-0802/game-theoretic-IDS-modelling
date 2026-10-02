#!/bin/bash
# Script to run evaluation and collate results for an algorithm
ALGO=$1 # sac, td3, or ddpg
JOBID=$2

if [ -z "$ALGO" ]; then
    echo "Usage: ./run_evaluation_and_collect.sh <sac|td3|ddpg> [jobid]"
    exit 1
fi

BASE_DIR="$HOME/research/DDPG_MQTT_125"
SRC_DIR="$BASE_DIR/src"
RESULTS_DIR="$BASE_DIR/results/$ALGO"
mkdir -p "$RESULTS_DIR"

cd "$SRC_DIR"

N_EXP=${3:-1}

echo "=== Running evaluation for $ALGO (n_experiment=$N_EXP) at $(date) ==="
if [ "$ALGO" == "sac" ]; then
    ~/.conda/envs/Alerts/bin/python -u evaluate_sac.py snort sac 1000 125 sac 125 $N_EXP > "$RESULTS_DIR/evaluate_console.log" 2>&1
elif [ "$ALGO" == "td3" ]; then
    ~/.conda/envs/Alerts/bin/python -u evaluate_td3.py snort td3 1000 125 td3 125 $N_EXP > "$RESULTS_DIR/evaluate_console.log" 2>&1
elif [ "$ALGO" == "ddpg" ]; then
    ~/.conda/envs/Alerts/bin/python -u evaluate_ddpg.py snort rl 1000 125 rl 125 $N_EXP > "$RESULTS_DIR/evaluate_console.log" 2>&1
fi
EVAL_EXIT=$?
echo "Evaluation finished with exit code $EVAL_EXIT at $(date)"

# Copy rewards_per_attack.csv
if [ -f "$SRC_DIR/rewards_per_attack.csv" ]; then
    cp "$SRC_DIR/rewards_per_attack.csv" "$RESULTS_DIR/rewards_per_attack.csv"
fi

# Copy console logs
if [ -n "$JOBID" ]; then
    [ -f "$SRC_DIR/do_${ALGO}.${JOBID}.out" ] && cp "$SRC_DIR/do_${ALGO}.${JOBID}.out" "$RESULTS_DIR/double_oracle_console.log"
    [ -f "$SRC_DIR/do_${ALGO}.${JOBID}.err" ] && cp "$SRC_DIR/do_${ALGO}.${JOBID}.err" "$RESULTS_DIR/double_oracle_stderr.log"
else
    LATEST_OUT=$(ls -t "$SRC_DIR/do_${ALGO}."*.out 2>/dev/null | head -1)
    [ -n "$LATEST_OUT" ] && cp "$LATEST_OUT" "$RESULTS_DIR/double_oracle_console.log"
    LATEST_ERR=$(ls -t "$SRC_DIR/do_${ALGO}."*.err 2>/dev/null | head -1)
    [ -n "$LATEST_ERR" ] && cp "$LATEST_ERR" "$RESULTS_DIR/double_oracle_stderr.log"
fi

# Copy / archive converge directory
CONV_DIR="$BASE_DIR/model/converge/snort_1000_125_${ALGO}_do"
if [ -d "$CONV_DIR" ]; then
    mkdir -p "$RESULTS_DIR/double_oracle/snort"
    mkdir -p "$RESULTS_DIR/snort"
    cp -r "$CONV_DIR"/* "$RESULTS_DIR/double_oracle/snort/"
    cp -r "$CONV_DIR"/* "$RESULTS_DIR/snort/"
    tar -czf "$RESULTS_DIR/snort_checkpoints.tar.gz" -C "$BASE_DIR/model/converge" "snort_1000_125_${ALGO}_do"
fi

echo "All results collected in $RESULTS_DIR"
