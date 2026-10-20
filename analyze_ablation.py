#!/usr/bin/env python3
"""
analyze_ablation.py  (corrected for ablation-tagged CSVs)

Reads the per-configuration CSVs produced by dueling_ddqn_multiseed.py
when invoked with --w_deadline / --w_backhaul / etc. and produces:

  1. Per-configuration means ± 95% CI for DL%, delay, energy, fairness
  2. A ready-to-paste LaTeX table
  3. A Friedman test across the three configurations

Filenames are matched by the cfg_tag embedded in the CSV name:
   dueling_ddqn_wd{WD}_wh{WH}_we{WE}_wl{WL}_seed{N}.csv

where WD, WH, WE, WL are the weights formatted to two decimals, as
produced by train_one_seed() in dueling_ddqn_multiseed.py.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Paths (must match env_config.py) ----------
# If env_config is importable, use it; otherwise fall back to a default.
try:
    from env_config import DUELING_CSV_DIR
except Exception:
    DUELING_CSV_DIR = "dueling_csv"

# ---------- Configuration mapping ----------
# Maps a human-readable label to the (w_deadline, w_backhaul, w_energy, w_load)
# tuple that the training script embeds in the filename.
# Add or edit entries here to match whatever configurations you ran.
CONFIGS = {
    "Baseline":       (0.55, 0.15, 0.05, 0.10),
    "Deadline-heavy": (0.75, 0.10, 0.03, 0.05),
    "Energy-heavy":   (0.45, 0.10, 0.20, 0.10),
}

SEEDS = [1, 7, 21, 42, 84]

# Columns of interest in the training CSV.
# Adjust `dl_rate` -> whatever your CSV actually calls it if different.
METRICS = {
    "dl_rate":       "DL%",
    "avg_delay_ms":  "Delay (ms)",
    "avg_energy_j":  "Energy (J)",
    "jains_fairness": "Fairness",
}


# ---------- Loading ----------
def _cfg_tag(wd, wh, we, wl):
    """Reproduce the tag used by train_one_seed()."""
    return f"wd{wd:.2f}_wh{wh:.2f}_we{we:.2f}_wl{wl:.2f}"


def load_config(label):
    """Load and concatenate all seed CSVs for one configuration."""
    wd, wh, we, wl = CONFIGS[label]
    tag = _cfg_tag(wd, wh, we, wl)

    frames = []
    for seed in SEEDS:
        # Match by glob to be robust to stray files with the same tag
        pattern = os.path.join(
            DUELING_CSV_DIR,
            f"dueling_ddqn_{tag}_seed{seed}.csv",
        )
        if not os.path.exists(pattern):
            print(f"WARNING: missing {pattern}")
            continue
        df = pd.read_csv(pattern)
        df["seed"] = seed
        frames.append(df)

    if not frames:
        print(f"No CSVs found for {label} (tag={tag})")
        return None
    return pd.concat(frames, ignore_index=True)


def per_seed_means(df, column):
    """One scalar per seed: mean of `column` over the LAST 100 rows for that seed.
    Matches the tail-100 convention used inside dueling_ddqn_multiseed.py."""
    if column not in df.columns:
        return np.array([])
    return df.groupby("seed")[column].apply(
        lambda s: s.tail(100).mean()
    ).values


def mean_ci(values, confidence=0.95):
    """Returns (mean, 95% CI half-width). Handles n=1 gracefully."""
    n = len(values)
    if n == 0:
        return float("nan"), 0.0
    mean = float(np.mean(values))
    if n == 1:
        return mean, 0.0
    sem = stats.sem(values)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, float(h)


# ---------- Analysis ----------
print("=" * 70)
print("Reward Weight Sensitivity Analysis")
print("=" * 70)
print(f"Reading from: {os.path.abspath(DUELING_CSV_DIR)}")
print()

results = {}
for label in CONFIGS:
    df = load_config(label)
    if df is None:
        continue

    results[label] = {}
    print(f"\n{label}")
    print("-" * 70)

    for col, pretty in METRICS.items():
        if col not in df.columns:
            print(f"  {pretty:15s}: column '{col}' not present in CSV")
            continue
        values = per_seed_means(df, col)
        m, ci = mean_ci(values)
        results[label][pretty] = (m, ci)
        print(f"  {pretty:15s}: {m:8.3f} ± {ci:.3f}   (n={len(values)} seeds)")


# ---------- LaTeX Table ----------
print("\n" + "=" * 70)
print("LaTeX Table for Chapter 5")
print("=" * 70)
print()
print(r"\begin{table}[htbp]")
print(r"    \centering")
print(r"    \caption{Reward weight sensitivity analysis (Dueling DDQN, "
      r"training-episode means $\pm$ 95\% CI over 5 seeds).}")
print(r"    \label{tab:weight_sensitivity}")
print(r"    \begin{tabular}{lcccc}")
print(r"        \toprule")
print(r"        \textbf{Configuration} & \textbf{DL\%} & \textbf{Delay (ms)} "
      r"& \textbf{Energy (J)} & \textbf{Fairness} \\")
print(r"        \midrule")

def _fmt(v):
    """Format (mean, ci) as 'm.xx $\pm$ c.yy' or '--'."""
    if v is None or not np.isfinite(v[0]):
        return "--"
    return f"{v[0]:.2f} $\\pm$ {v[1]:.2f}"

for label in CONFIGS:
    if label not in results:
        print(f"        {label} & -- & -- & -- & -- \\\\")
        continue
    r = results[label]
    print(f"        {label} & "
          f"{_fmt(r.get('DL%'))} & "
          f"{_fmt(r.get('Delay (ms)'))} & "
          f"{_fmt(r.get('Energy (J)'))} & "
          f"{_fmt(r.get('Fairness'))} \\\\")

print(r"        \bottomrule")
print(r"    \end{tabular}")
print(r"\end{table}")


# ---------- Statistical Test ----------
print("\n" + "=" * 70)
print("Friedman test across configurations (on DL%)")
print("=" * 70)

# Friedman requires the same number of observations per group.
# Keep only seeds present in every configuration.
per_cfg_seeds = {}
for label in CONFIGS:
    df = load_config(label)
    if df is None:
        continue
    per_cfg_seeds[label] = df.groupby("seed")["dl_rate"].mean()

if len(per_cfg_seeds) >= 3:
    common_seeds = set.intersection(*[set(s.index) for s in per_cfg_seeds.values()])
    common_seeds = sorted(common_seeds)

    if len(common_seeds) >= 2:
        samples = [
            per_cfg_seeds[label].loc[common_seeds].values
            for label in CONFIGS
            if label in per_cfg_seeds
        ]
        F, p = stats.friedmanchisquare(*samples)
        print(f"Configurations compared: {list(per_cfg_seeds.keys())}")
        print(f"Common seeds used: {common_seeds}")
        print(f"Friedman statistic = {F:.3f}, p = {p:.4f}")
        if p < 0.05:
            print("  -> Statistically significant difference across configurations")
        else:
            print("  -> No statistically significant difference across configurations")
    else:
        print("Fewer than 2 seeds shared across all configurations; "
              "cannot run Friedman test.")
else:
    print("Fewer than 3 configurations loaded; skipping Friedman test.")