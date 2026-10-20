#!/bin/bash
# ============================================================
# run_ablation.sh  (v2 — drives dueling_ddqn_multiseed.py)
#
# Three reward-weight configurations, each run through the Python
# training script. The training script spawns ns-3 as a child
# process, trains Dueling DDQN, and writes a tagged CSV per seed
# into DUELING_CSV_DIR.
#
# Prerequisites:
#   1. mec_simulator_v8_robust.cc rebuilt with --w_* args
#   2. dueling_ddqn_multiseed.py updated (Fix 2.1, 2.2)
#   3. env_config.py updated (Fix 1.1 – 1.4)
# ============================================================

set -e

# Run from the directory that contains dueling_ddqn_multiseed.py
cd "$(dirname "$0")"

mkdir -p logs

run_config () {
    local name="$1"; shift
    echo "=============================================="
    echo "Configuration: $name"
    echo "  args: $@"
    echo "=============================================="

    python dueling_ddqn_multiseed.py "$@" \
        > "logs/ablation_${name}.log" 2>&1

    echo "  done → logs/ablation_${name}.log"
}

# Baseline
run_config baseline \
    --w_deadline=0.55 --w_delay=0.10 --w_energy=0.05 \
    --w_load=0.10 --w_penalty=0.05 --w_backhaul=0.15

# Deadline-heavy
run_config deadline_heavy \
    --w_deadline=0.75 --w_delay=0.05 --w_energy=0.03 \
    --w_load=0.05 --w_penalty=0.02 --w_backhaul=0.10

# Energy-heavy
run_config energy_heavy \
    --w_deadline=0.45 --w_delay=0.10 --w_energy=0.20 \
    --w_load=0.10 --w_penalty=0.05 --w_backhaul=0.10

echo ""
echo "=============================================="
echo "Ablation complete."
echo "CSVs are in DUELING_CSV_DIR (see env_config.py)."
echo "Logs are in logs/."
echo "=============================================="