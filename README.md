# 6G MEC Task Offloading — Code

Reference code for the Master's thesis:

> **Task Offloading Optimization in 6G Mobile Edge Computing Based on Artificial Intelligence**
>
> Pasang Lama — Instituto Superior Técnico, University of Lisbon, October 2026

## Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/Pasang983/6g-mec-drl-thesis.git
cd 6g-mec-drl-thesis

# 2. Fix paths in env_config.py and complete_data_analysis.py (see below)

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run a training experiment
python double_dqn_multiseed.py
```

**Full setup instructions are provided below.**

## Contents

### Core simulator and configuration

- `mec_simulator_v8_robust.cc` — ns-3 simulator source (place in `$NS3_PATH/scratch/`)
- `env_config.py` — shared configuration

### DRL agents

- `double_dqn_multiseed.py` — DDQN training
- `dueling_ddqn_multiseed.py` — Dueling DDQN training
- `ppo_agent_multiseed.py` — PPO training

### Baselines

- `random_baseline_multiseed.py` — Random baseline
- `greedy_baseline_multiseed.py` — Greedy Load and Greedy Ratio baselines

### Evaluation and analysis

- `evaluate_all_drl.py` — Evaluation harness for trained DRL policies
- `complete_data_analysis.py` — Full analysis pipeline (tables, figures, statistical tests)

### Reward-weight ablation study

- `run_ablation.sh` — Shell script that launches the three reward-weight configurations (baseline, deadline-heavy, energy-heavy) sequentially
- `dueling_ddqn_multiseed_ablation.py` — Dueling DDQN training script with configurable reward weights (used by `run_ablation.sh`)
- `analyse_ablation.py` — Aggregates and plots the ablation results across the three configurations

## Repository Structure

6g-mec-drl-thesis/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── env_config.py
│
├── simulator/
│   └── mec_simulator_v8_robust.cc
│
├── agents/
│   ├── double_dqn_multiseed.py
│   ├── dueling_ddqn_multiseed.py
│   ├── ppo_agent_multiseed.py
│   ├── random_baseline_multiseed.py
│   └── greedy_baseline_multiseed.py
│
├── evaluation/
│   └── evaluate_all_drl.py
│
├── analysis/
│   ├── complete_data_analysis.py
│   └── analyse_ablation.py
│
├── ablation/
│   ├── run_ablation.sh
│   └── dueling_ddqn_multiseed_ablation.py
│
└── results/
    └── phase3_v8_robust_fixed/
        ├── raw/
        ├── analysis/
        │   ├── tables/
        │   ├── figures/
        │   └── statistics/
        └── checkpoints/

## ⚠️ Configuration Note — Paths Must Be Set Before Running

The code was written for a specific experimental setup. **Before running anything, you must update the path variables** in `env_config.py` and `complete_data_analysis.py` to match your own system.

### Step-by-step

**1. In `env_config.py`, find:**

```python
BASE_DIR    = "/mnt/sdb1/users/plama/research/ns-3.38/scratch"
NS3_PATH    = "/mnt/sdb1/users/plama/research/ns-3.38"
```

**2. Replace with your own paths, for example:**

```python
BASE_DIR    = "/home/your_username/6g-mec-drl-thesis"
NS3_PATH    = "/home/your_username/ns-3.38"
```

**3. In `complete_data_analysis.py`, find:**

```python
BASE_DIR   = "/mnt/sdb1/users/plama/research/ns-3.38/scratch/results"
```

**4. Replace with:**

```python
BASE_DIR   = "/home/your_username/6g-mec-drl-thesis/results"
```

**5. In `run_ablation.sh`, find:**

```bash
BASE_DIR=/mnt/sdb1/users/plama/research/ns-3.38/scratch
```

**6. Replace with:**

```bash
BASE_DIR=/home/your_username/6g-mec-drl-thesis
```

**7. Verify no hardcoded paths remain:**

```bash
grep -rn "/mnt/sdb1" *.py *.cc *.sh
```

This must return **zero lines**. If any line appears, update that file the same way.

## Setup

### Prerequisites

- **ns-3.38** or later with the **ns3-gym** module (bundled with ns-3 ≥ 3.30)
- **Python 3.9+** with pip
- **PyTorch 2.0+** (CPU or CUDA)
- **GCC 9+** or **Clang 12+**
- **Linux** or **macOS** (Windows users should use WSL2)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Pasang983/6g-mec-drl-thesis.git
   cd 6g-mec-drl-thesis
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the simulator source into your ns-3 installation:

   ```bash
   cp mec_simulator_v8_robust.cc $NS3_PATH/scratch/
   ```

4. Build the simulator:

   ```bash
   cd $NS3_PATH
   ./ns3 build scratch/mec_simulator_v8_robust
   ```

5. Set path variables in `env_config.py`, `complete_data_analysis.py` and `run_ablation.sh` (see the Configuration Note above).

## Usage

### Train DRL agents (600 episodes × 5 seeds per algorithm)

```bash
python double_dqn_multiseed.py
python dueling_ddqn_multiseed.py
python ppo_agent_multiseed.py
```

Training takes approximately **8 hours per seed per algorithm** on a single GPU. Adjust `MULTI_SEEDS` in `env_config.py` for a faster smoke test.

### Run baselines (evaluation only)

```bash
python random_baseline_multiseed.py
python greedy_baseline_multiseed.py both
```

### Evaluate trained DRL policies

```bash
python evaluate_all_drl.py
```

### Generate tables, figures and statistical tests

```bash
python complete_data_analysis.py
```

Outputs are written to `results/phase3_v8_robust_fixed/analysis/`:

- `tables/` — comparison tables (CSV + LaTeX)
- `figures/` — publication-quality PDFs and PNGs
- `statistics/` — statistical test results

## Reward-Weight Ablation Study

The thesis reports a reward-weight sensitivity campaign in §5.9. This repository includes the scripts to reproduce it.

### What it does

Trains three Dueling DDQN configurations that differ only in reward weights:

| Configuration | `w_deadline` | `w_backhaul` | `w_delay` | `w_load` | `w_energy` | `w_penalty` |
|---|---|---|---|---|---|---|
| Baseline | 0.55 | 0.15 | 0.10 | 0.10 | 0.05 | 0.05 |
| Deadline-heavy | 0.75 | 0.10 | 0.05 | 0.05 | 0.03 | 0.02 |
| Energy-heavy | 0.45 | 0.10 | 0.10 | 0.10 | 0.20 | 0.05 |

Each configuration is trained under all five seeds.

### Running the ablation

```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

Total runtime: **~120 hours** (3 configurations × 5 seeds × 8 hours per training run).

### Analysing the ablation

```bash
python analyse_ablation.py
```

Produces per-configuration comparison tables and figures showing the effect of reward-weight changes on DL%, delay, energy and fairness.

### Expected output

The ablation is documented in thesis §5.9 and Appendix B.3. The energy-heavy configuration shows approximately **8.3% lower estimated energy** than the baseline, while DL%, delay and fairness remain relatively stable across configurations.

## Reproducibility

All experiments use **fixed seeds** `{1, 7, 21, 42, 84}`. The simulator uses an explicitly seeded PRNG, and all training is deterministic given a seed and PyTorch's `torch.use_deterministic_algorithms(True)`.

The results reported in the thesis are seed-level means with seed-level 95% confidence intervals, computed across the five seeds. Pairwise comparisons use the Wilcoxon signed-rank test on matched seed identifiers with Holm–Bonferroni correction.

Raw evaluation data and trained model checkpoints are not distributed with this repository due to size. They are archived with the thesis materials and available on request.

## Reference-Only Notice

This is a **reference implementation**, not a turnkey reproduction package. It assumes:

- ns-3.38 with the ns3-gym module compiled and enabled
- Python 3.9+ with PyTorch and the packages listed in `requirements.txt`
- A Linux or macOS environment (Windows via WSL2)
- Familiarity with the setup described in the thesis §5.2

Adapting it to a different topology, workload, reward function or simulator version may require non-trivial changes.

The code is provided to support inspection and adaptation. Results may differ if the code is run under different conditions, hardware or library versions.

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{Lama2026MEC,
  author = {Pasang Lama},
  title  = {Task Offloading Optimization in 6G Mobile Edge Computing Based on Artificial Intelligence},
  school = {Instituto Superior Técnico, University of Lisbon},
  year   = {2026},
  month  = {October},
}
```

## License

MIT License — see [`LICENSE`](LICENSE) for details.

## Contact

**Author:** Pasang Lama
**Email:** pasang.lama@tecnico.ulisboa.pt
**Supervisors:** Prof. Antonio Grilo, Prof. Naercio Magaia
**Institution:** Instituto Superior Técnico, University of Lisbon
