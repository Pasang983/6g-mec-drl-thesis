# 6G MEC Task Offloading — Code

Reference code for the Master's thesis:

> **Task Offloading Optimization in 6G Mobile Edge Computing Based on Artificial Intelligence**
> Pasang Lama — Instituto Superior Técnico, University of Lisbon, October 2026

## Contents

- `mec_simulator_v8_robust.cc` — ns-3 simulator source (place in `$NS3_PATH/scratch/`)
- `env_config.py` — shared configuration
- `double_dqn_multiseed.py` — DDQN training
- `dueling_ddqn_multiseed.py` — Dueling DDQN training
- `ppo_agent_multiseed.py` — PPO training
- `random_baseline_multiseed.py` — Random baseline
- `greedy_baseline_multiseed.py` — Greedy Load and Greedy Ratio baselines
- `evaluate_all_drl.py` — Evaluation harness
- `complete_data_analysis.py` — Analysis pipeline

## ⚠️ Configuration Note — Paths Must Be Set Before Running

The code was written for a specific experimental setup. **Before running anything, you must update the path variables** in `env_config.py` and `complete_data_analysis.py` to match your own system.

### Step-by-step

**1. In `env_config.py`, find:**

```python
BASE_DIR    = "/mnt/sdb1/users/plama/research/ns-3.38/scratch"
NS3_PATH    = "/mnt/sdb1/users/plama/research/ns-3.38"
