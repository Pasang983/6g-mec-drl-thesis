#!/usr/bin/env python3
"""
COMPLETE DATA ANALYSIS PIPELINE FOR THESIS
6G MEC Task Offloading Optimization
Author: Pasang Lama

Generates:
  1. Comparison tables (with 95% confidence intervals)
  2. Statistical tests (parametric + non-parametric)
  3. Publication-quality figures
  4. Summary statistics, correlations, convergence analysis
  5. Handover / energy breakdowns
  6. Per-seed results table
  7. DL% vs Delay Pareto scatter
  8. RSU artifact analysis

Input layout (matches the actual results tree):
  results/phase3_v8_robust_fixed/
    ddqn/        csv/seed_N/double_dqn_seedN.csv   +  eval/seed_N/...csv
    dueling_ddqn/csv/seed_N/dueling_ddqn_seedN.csv +  eval/seed_N/...
    ppo/         csv/seed_N/ppo_seedN.csv          +  eval/seed_N/...
    random/      csv/random_seedN.csv
    greedy/
      load/      csv/greedy_load_seedN.csv
      ratio/     csv/greedy_ratio_seedN.csv

The loader prefers EVALUATION CSVs when present, and falls back to
TRAINING CSVs (last N episodes) when eval data is not available.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from scipy import stats
from scipy.stats import (
    mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    ttest_ind, f_oneway, pearsonr, spearmanr,
    ttest_rel, shapiro, normaltest, t
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION  (ALIGNED WITH THE RESULTS TREE)
# ============================================================
BASE_DIR   = "/mnt/sdb1/users/plama/research/ns-3.38/scratch/results"
PHASE_DIR_NAME = "phase3_v8_robust_fixed"
PHASE_DIR  = os.path.join(BASE_DIR, PHASE_DIR_NAME)

ANALYSIS_DIR = os.path.join(PHASE_DIR, "analysis")
TABLES_DIR   = os.path.join(ANALYSIS_DIR, "tables")
FIGURES_DIR  = os.path.join(ANALYSIS_DIR, "figures")
STATS_DIR    = os.path.join(ANALYSIS_DIR, "statistics")
DATA_DIR     = os.path.join(ANALYSIS_DIR, "data")
for d in [ANALYSIS_DIR, TABLES_DIR, FIGURES_DIR, STATS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# Evaluation protocol
EVAL_EPISODES = 600
SEEDS = [1, 7, 21, 42, 84]

# Algorithm registry
ALGORITHMS = {
    'random': {
        'label': 'Random',
        'color': '#6C757D',
        'type': 'Baseline',
        'marker': 'o',
        'csv_dir_train': os.path.join(PHASE_DIR, 'random', 'csv'),
        'csv_dir_eval':  os.path.join(PHASE_DIR, 'random', 'eval'),
        'csv_patterns': [
            'random_seed_{seed}.csv',
            'random_seed{seed}.csv',
        ],
    },
    'greedy_load': {
        'label': 'Greedy (Load)',
        'color': '#28A745',
        'type': 'Heuristic',
        'marker': 's',
        'csv_dir_train': os.path.join(PHASE_DIR, 'greedy', 'load', 'csv'),
        'csv_dir_eval':  os.path.join(PHASE_DIR, 'greedy', 'load', 'eval'),
        'csv_patterns': [
            'greedy_load_seed{seed}.csv',
        ],
    },
    'greedy_ratio': {
        'label': 'Greedy (Ratio)',
        'color': '#FD7E14',
        'type': 'Heuristic',
        'marker': '^',
        'csv_dir_train': os.path.join(PHASE_DIR, 'greedy', 'ratio', 'csv'),
        'csv_dir_eval':  os.path.join(PHASE_DIR, 'greedy', 'ratio', 'eval'),
        'csv_patterns': [
            'greedy_ratio_seed{seed}.csv',
        ],
    },
    'ddqn': {
        'label': 'DDQN',
        'color': '#A23B72',
        'type': 'RL',
        'marker': 'D',
        'csv_dir_train': os.path.join(PHASE_DIR, 'ddqn', 'csv'),
        'csv_dir_eval':  os.path.join(PHASE_DIR, 'ddqn', 'eval'),
        'csv_patterns': [
            'ddqn_seed{seed}_eval.csv',
            'double_dqn_seed{seed}.csv',
        ],
    },
    'dueling_ddqn': {
        'label': 'Dueling DDQN',
        'color': '#F18F01',
        'type': 'RL',
        'marker': 'p',
        'csv_dir_train': os.path.join(PHASE_DIR, 'dueling_ddqn', 'csv'),
        'csv_dir_eval':  os.path.join(PHASE_DIR, 'dueling_ddqn', 'eval'),
        'csv_patterns': [
            'dueling_ddqn_seed{seed}_eval.csv',
            'dueling_ddqn_seed{seed}.csv',
        ],
    },
    'ppo': {
        'label': 'PPO',
        'color': '#2E86AB',
        'type': 'RL',
        'marker': '*',
        'csv_dir_train': os.path.join(PHASE_DIR, 'ppo', 'csv'),
        'csv_dir_eval':  os.path.join(PHASE_DIR, 'ppo', 'eval'),
        'csv_patterns': [
            'ppo_seed{seed}_eval.csv',
            'ppo_seed{seed}.csv',
        ],
    },
}

ACTION_NAMES = ['LOCAL', 'MEC0', 'MEC1', 'MEC2', 'RSU0', 'RSU1', 'Satellite']
NUM_ACTIONS = len(ACTION_NAMES)

RL_ALGORITHMS      = ['ddqn', 'dueling_ddqn', 'ppo']
BASELINE_ALGORITHMS = ['random', 'greedy_load', 'greedy_ratio']

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ============================================================
# 1. DATA LOADING
# ============================================================
def _resolve_csv(config, seed, prefer_eval=True):
    candidates = []
    dirs = []
    if prefer_eval and config.get('csv_dir_eval'):
        dirs.append(config['csv_dir_eval'])
    if config.get('csv_dir_train'):
        dirs.append(config['csv_dir_train'])
    if not prefer_eval and config.get('csv_dir_eval'):
        dirs.append(config['csv_dir_eval'])

    for base in dirs:
        if not os.path.exists(base):
            continue
        for sub in [f"seed_{seed}", ""]:
            for pat in config['csv_patterns']:
                fname = pat.format(seed=seed)
                p = os.path.join(base, sub, fname) if sub else os.path.join(base, fname)
                if os.path.exists(p):
                    candidates.append(p)

    return candidates[0] if candidates else None


def load_algorithm_data(algorithm_name):
    config = ALGORITHMS[algorithm_name]
    seed_data = []

    for seed in SEEDS:
        csv_path = _resolve_csv(config, seed, prefer_eval=True)
        if csv_path is None:
            print(f"  ⚠️ No CSV found for {algorithm_name} seed {seed}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ❌ Failed to read {csv_path}: {e}")
            continue

        if 'dl_rate' not in df.columns:
            print(f"  ⚠️ No 'dl_rate' column in {csv_path}")
            continue

        # Prefer evaluation phase rows if a phase column exists
        if 'phase' in df.columns:
            eval_df = df[df['phase'] == 'evaluation']
            if len(eval_df) > 0:
                df = eval_df

        # Cap to EVAL_EPISODES
        if len(df) > EVAL_EPISODES:
            df = df.tail(EVAL_EPISODES).reset_index(drop=True)

        entry = {
            'seed': seed,
            'source_csv': csv_path,
            'episodes': len(df),
        }

        entry['dl_rates']   = df['dl_rate'].values.astype(float)
        if 'env_reward' in df.columns:
            entry['rewards'] = df['env_reward'].values.astype(float)
        elif 'env_reward_raw' in df.columns:
            entry['rewards'] = df['env_reward_raw'].values.astype(float)
        elif 'reward' in df.columns:
            entry['rewards'] = df['reward'].values.astype(float)
        else:
            entry['rewards'] = np.zeros(len(df))

        if 'avg_delay_ms' in df.columns:
            entry['avg_delay'] = df['avg_delay_ms'].values.astype(float) / 1000.0
        else:
            entry['avg_delay'] = np.zeros(len(df))

        entry['p95_delay'] = entry['avg_delay']

        if 'jains_fairness' in df.columns:
            entry['fairness'] = df['jains_fairness'].values.astype(float)
        elif 'fairness' in df.columns:
            entry['fairness'] = df['fairness'].values.astype(float)
        else:
            entry['fairness'] = np.ones(len(df))

        if 'handover_rate' in df.columns:
            entry['handover_rate'] = df['handover_rate'].values.astype(float)
        if 'avg_energy_j' in df.columns:
            entry['energy'] = df['avg_energy_j'].values.astype(float)
        elif 'energy_j' in df.columns:
            entry['energy'] = df['energy_j'].values.astype(float)
        if 'throughput_tps' in df.columns:
            entry['throughput'] = df['throughput_tps'].values.astype(float)

        # Try "act_*" columns first (DRL eval CSVs)
        act_cols = ['act_local', 'act_mec0', 'act_mec1', 'act_mec2',
                    'act_rsu0', 'act_rsu1', 'act_sat']
        # Fallback: "pct_*" columns (baseline CSVs)
        pct_cols = ['pct_local', 'pct_mec0', 'pct_mec1', 'pct_mec2',
                    'pct_rsu0', 'pct_rsu1', 'pct_sat']

        if all(c in df.columns for c in act_cols):
            entry['action_distributions'] = df[act_cols].values.astype(float)
        elif all(c in df.columns for c in pct_cols):
            entry['action_distributions'] = df[pct_cols].values.astype(float)
        else:
            entry['action_distributions'] = np.full((len(df), NUM_ACTIONS), 100.0 / NUM_ACTIONS)

        # Also stash the raw dataframe for later (RSU analysis etc.)
        entry['_df'] = df

        seed_data.append(entry)
        print(f"  ✅ {algorithm_name} seed {seed}: {len(df)} eps from {os.path.basename(csv_path)}, "
              f"DL={np.mean(entry['dl_rates']):.1f}%")

    return seed_data


def load_all_algorithms():
    print("\n" + "="*70)
    print("LOADING ALGORITHM DATA")
    print("="*70)
    results = {}
    for alg in ALGORITHMS:
        print(f"\nLoading {alg}...")
        data = load_algorithm_data(alg)
        results[alg] = data if data else None
    return results

# ============================================================
# 2. AGGREGATION
# ============================================================
def compute_aggregate_stats(seed_data):
    if not seed_data:
        return None

    final = {}
    curves = {}

    def _per_seed(key):
        vals = []
        for d in seed_data:
            if key in d and len(d[key]) > 0:
                vals.append(float(np.mean(d[key])))
        return vals

    _dl = _per_seed('dl_rates')
    _rw = _per_seed('rewards')
    _dy = _per_seed('avg_delay')
    _fv = _per_seed('fairness')
    _en = _per_seed('energy')
    _ho = _per_seed('handover_rate')

    final['dl_rate']       = float(np.mean(_dl)) if _dl else 0.0
    final['dl_rate_std']   = float(np.std (_dl)) if _dl else 0.0
    final['reward']        = float(np.mean(_rw)) if _rw else 0.0
    final['reward_std']    = float(np.std (_rw)) if _rw else 0.0
    final['avg_delay']     = float(np.mean(_dy)) if _dy else 0.0
    final['avg_delay_std'] = float(np.std (_dy)) if _dy else 0.0
    final['p95_delay']     = final['avg_delay']
    final['p95_delay_std'] = final['avg_delay_std']
    final['fairness']      = float(np.mean(_fv)) if _fv else 1.0
    final['fairness_std']  = float(np.std (_fv)) if _fv else 0.0
    final['energy']        = float(np.mean(_en)) if _en else 0.0
    final['energy_std']    = float(np.std (_en)) if _en else 0.0
    final['handovers']     = float(np.mean(_ho)) * 10 if _ho else 0.0
    final['handovers_std'] = float(np.std (_ho)) * 10 if _ho else 0.0
    final['inference']     = 0.0
    final['inference_std'] = 0.0

    for key in ['dl_rates', 'rewards', 'avg_delay', 'fairness']:
        max_len = max((len(d[key]) for d in seed_data if key in d), default=0)
        if max_len == 0:
            continue
        padded = []
        for d in seed_data:
            if key not in d:
                continue
            arr = np.asarray(d[key], dtype=float)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
            padded.append(arr[:max_len])
        padded = np.stack(padded)
        curves[key] = {
            'mean': np.nanmean(padded, axis=0),
            'std':  np.nanstd (padded, axis=0),
        }

    ad_list = [d['action_distributions'].mean(axis=0) for d in seed_data
               if 'action_distributions' in d]
    action_mean = np.mean(np.stack(ad_list), axis=0) if ad_list else np.full(NUM_ACTIONS, np.nan)

    return {'final': final, 'curves': curves, 'action_dist': action_mean}


def compute_confidence_intervals(seed_data, metric='dl_rates', confidence=0.95):
    if not seed_data:
        return None
    vals = []
    for d in seed_data:
        if metric in d and len(d[metric]) > 0:
            vals.extend(np.asarray(d[metric]).tolist())
    if not vals:
        return None
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 0 else 0.0
    tcrit = t.ppf((1 + confidence)/2, max(n - 1, 1))
    margin = tcrit * se
    return {'mean': m, 'lower': m - margin, 'upper': m + margin,
            'std': s, 'se': se, 'n': n}


def compute_all_confidence_intervals(results):
    out = {}
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        out[alg] = {}
        for metric in ['dl_rates', 'rewards', 'avg_delay', 'fairness']:
            ci = compute_confidence_intervals(seed_data, metric)
            if ci:
                out[alg][metric] = ci
    return out

# ============================================================
# 3. STATISTICAL TESTS
# ============================================================
def run_statistical_tests(results):
    test_results = []

    per_seed = {}
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        dl_means = [float(np.mean(d['dl_rates'])) for d in seed_data if 'dl_rates' in d]
        if dl_means:
            per_seed[alg] = dl_means

    if not per_seed:
        return pd.DataFrame()

    for alg, vals in per_seed.items():
        if len(vals) >= 3:
            stat, p = shapiro(vals)
            test_results.append({
                'Test': 'Shapiro-Wilk',
                'Description': f'{ALGORITHMS[alg]["label"]} normality',
                'Statistic': f"{stat:.3f}",
                'p_value': p,
                'Significant': p < 0.05,
                'Interpretation': 'Normal' if p > 0.05 else 'Non-normal'
            })

    if len(per_seed) >= 2:
        groups = list(per_seed.values())
        h, p = kruskal(*groups)
        test_results.append({
            'Test': 'Kruskal-Wallis',
            'Description': 'Across all algorithms',
            'Statistic': f"{h:.4f}",
            'p_value': p,
            'Significant': p < 0.05,
            'Interpretation': 'Significant difference' if p < 0.05 else 'No significant difference'
        })

    alg_list = [a for a in per_seed if len(per_seed[a]) >= 3]
    if len(alg_list) >= 3:
        n = min(len(per_seed[a]) for a in alg_list)
        matrix = np.array([[per_seed[a][i] for a in alg_list] for i in range(n)])
        try:
            stat, p = friedmanchisquare(*matrix.T)
            test_results.append({
                'Test': 'Friedman',
                'Description': 'Repeated measures across seeds',
                'Statistic': f"{stat:.4f}",
                'p_value': p,
                'Significant': p < 0.05,
                'Interpretation': 'Significant difference' if p < 0.05 else 'No significant difference'
            })
        except Exception:
            pass

    rl_avail = [a for a in RL_ALGORITHMS if a in per_seed]
    bl_avail = [a for a in BASELINE_ALGORITHMS if a in per_seed]
    for rl in rl_avail:
        for bl in bl_avail:
            x, y = per_seed[rl], per_seed[bl]
            if len(x) < 3 or len(y) < 3:
                continue
            u, p = mannwhitneyu(x, y, alternative='greater')
            n = len(x) + len(y)
            z = (u - len(x)*len(y)/2) / np.sqrt(len(x)*len(y)*(n+1)/12)
            r = abs(z) / np.sqrt(n)
            test_results.append({
                'Test': 'Mann-Whitney U',
                'Description': f'{ALGORITHMS[rl]["label"]} vs {ALGORITHMS[bl]["label"]}',
                'Statistic': f"{u:.2f}",
                'p_value': p,
                'Significant': p < 0.05,
                'Effect_Size': f"{r:.3f}",
                'Effect_Size_Interpretation': ('Large' if r > 0.5 else
                                               'Medium' if r > 0.3 else 'Small'),
                'Improvement': f"{np.mean(x) - np.mean(y):.1f}%"
            })

    for i, a in enumerate(rl_avail):
        for b in rl_avail[i+1:]:
            x, y = per_seed[a], per_seed[b]
            n = min(len(x), len(y))
            if n < 3:
                continue
            try:
                w, p = wilcoxon(x[:n], y[:n])
                test_results.append({
                    'Test': 'Wilcoxon signed-rank',
                    'Description': f'{ALGORITHMS[a]["label"]} vs {ALGORITHMS[b]["label"]}',
                    'Statistic': f"{w:.2f}",
                    'p_value': p,
                    'Significant': p < 0.05,
                    'Effect_Size': 'N/A',
                    'Effect_Size_Interpretation': 'N/A',
                    'Improvement': f"{np.mean(x) - np.mean(y):.1f}%"
                })
            except Exception:
                pass

    if len(alg_list) >= 3:
        try:
            all_vals, all_labels = [], []
            for a in alg_list:
                all_vals.extend(per_seed[a])
                all_labels.extend([ALGORITHMS[a]['label']] * len(per_seed[a]))
            tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
            for g1, g2, md, padj, lo, hi, rej in zip(
                tukey.group1, tukey.group2, tukey.meandiff,
                tukey.p_adj, tukey.lower, tukey.upper, tukey.reject):
                if rej:
                    test_results.append({
                        'Test': 'Tukey HSD',
                        'Description': f'{g1} vs {g2}',
                        'Statistic': f"{md:.3f}",
                        'p_value': padj,
                        'Significant': True,
                        'Effect_Size': 'N/A',
                        'Effect_Size_Interpretation': 'N/A',
                        'Confidence_Interval': f"[{lo:.1f}, {hi:.1f}]"
                    })
        except Exception:
            pass

    df = pd.DataFrame(test_results)
    if len(df) > 0:
        df.to_csv(os.path.join(STATS_DIR, 'statistical_tests.csv'), index=False)
        print(f"✅ Statistical tests → {STATS_DIR}/statistical_tests.csv")
    return df

# ============================================================
# 4. HELPERS FOR FIGURES
# ============================================================
def _alg_color(alg):
    return ALGORITHMS[alg]['color']

def _alg_label(alg):
    return ALGORITHMS[alg]['label']

# ============================================================
# 5. FIGURES
# ============================================================
def plot_learning_curves(results, ci_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    avail = [a for a in results if results[a]]

    for ax, key, title, ylabel in [
        (axes[0], 'rewards',  'Learning Curves — Reward', 'Average Reward'),
        (axes[1], 'dl_rates', 'Learning Curves — DL%',    'Deadline Success (%)'),
    ]:
        for alg in avail:
            agg = compute_aggregate_stats(results[alg])
            if agg is None or key not in agg['curves']:
                continue
            m = agg['curves'][key]['mean']
            s = agg['curves'][key]['std']
            x = np.arange(len(m))
            ax.plot(x, m, label=_alg_label(alg), color=_alg_color(alg), linewidth=2.5)
            ax.fill_between(x, m - s, m + s, color=_alg_color(alg), alpha=0.2)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        if key == 'dl_rates':
            ax.set_ylim([0, 105])

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'learning_curves.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: learning_curves")


def plot_final_performance(results, ci_results):
    avail = [a for a in results if results[a]]
    labels  = [_alg_label(a) for a in avail]
    colors  = [_alg_color(a) for a in avail]

    dl, dl_lo, dl_hi = [], [], []
    rw, rw_lo, rw_hi = [], [], []
    dy, dy_lo, dy_hi = [], [], []
    fv, fv_lo, fv_hi = [], [], []

    for a in avail:
        agg = compute_aggregate_stats(results[a])['final']
        dl.append(agg['dl_rate'])
        rw.append(agg['reward'])
        dy.append(agg['avg_delay'] * 1000)
        fv.append(agg['fairness'])

        ci = (ci_results or {}).get(a, {})
        dl_ci = ci.get('dl_rates');  rw_ci = ci.get('rewards')
        dy_ci = ci.get('avg_delay'); fv_ci = ci.get('fairness')

        dl_lo.append(dl_ci['lower'] if dl_ci else agg['dl_rate'] - agg['dl_rate_std'])
        dl_hi.append(dl_ci['upper'] if dl_ci else agg['dl_rate'] + agg['dl_rate_std'])
        rw_lo.append(rw_ci['lower'] if rw_ci else agg['reward'] - agg['reward_std'])
        rw_hi.append(rw_ci['upper'] if rw_ci else agg['reward'] + agg['reward_std'])
        dy_lo.append((dy_ci['lower'] if dy_ci else agg['avg_delay'] - agg['avg_delay_std']) * 1000)
        dy_hi.append((dy_ci['upper'] if dy_ci else agg['avg_delay'] + agg['avg_delay_std']) * 1000)
        fv_lo.append(fv_ci['lower'] if fv_ci else agg['fairness'] - agg['fairness_std'])
        fv_hi.append(fv_ci['upper'] if fv_ci else agg['fairness'] + agg['fairness_std'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def _bar(ax, vals, lo, hi, ylabel, title, fmt, ymax=None):
        x = np.arange(len(vals))
        err = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
        bars = ax.bar(x, vals, yerr=err, capsize=5, color=colors,
                      alpha=0.85, edgecolor='black', linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + (0.02 if ymax and ymax <= 1.1 else 1),
                    fmt.format(v), ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        if ymax: ax.set_ylim([0, ymax])

    _bar(axes[0, 0], dl, dl_lo, dl_hi, 'Deadline Success (%)',
         'Final DL% (95% CI)', '{:.1f}%', ymax=105)
    _bar(axes[0, 1], rw, rw_lo, rw_hi, 'Average Reward',
         'Final Reward', '{:.1f}')
    _bar(axes[1, 0], dy, dy_lo, dy_hi, 'Average Delay (ms)',
         'Final Delay', '{:.1f}')
    _bar(axes[1, 1], fv, fv_lo, fv_hi, "Jain's Fairness",
         'Final Fairness', '{:.3f}', ymax=1.05)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'final_performance.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: final_performance")


def plot_action_distribution(results):
    avail = [a for a in results if results[a]]
    if not avail:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(NUM_ACTIONS)
    width = 0.8 / max(len(avail), 1)

    for i, alg in enumerate(avail):
        agg = compute_aggregate_stats(results[alg])
        if agg is None:
            continue
        ad = np.asarray(agg['action_dist'])
        if np.nansum(ad) > 0:
            ad = np.nan_to_num(ad)
            if not np.isclose(ad.sum(), 100.0):
                ad = 100.0 * ad / ad.sum()
        offset = (i - len(avail)/2) * width + width/2
        ax.bar(x + offset, ad, width, label=_alg_label(alg),
               color=_alg_color(alg), alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Action')
    ax.set_ylabel('Selection (%)')
    ax.set_title('Action Distribution by Algorithm', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(ACTION_NAMES, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'action_distribution.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: action_distribution")


def plot_boxplot_comparison(results):
    avail = [a for a in results if results[a]]
    if not avail:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dl_data, rw_data, labels, colors = [], [], [], []
    for a in avail:
        seed_dls = [float(np.mean(d['dl_rates'])) for d in results[a] if 'dl_rates' in d]
        seed_rws = [float(np.mean(d['rewards']))  for d in results[a] if 'rewards'  in d]
        if seed_dls:
            dl_data.append(seed_dls)
            rw_data.append(seed_rws)
            labels.append(_alg_label(a))
            colors.append(_alg_color(a))

    def _box(ax, data, ylabel, title):
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        meanprops=dict(color='red', linestyle='--', linewidth=1.5))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)

    _box(axes[0], dl_data, 'Deadline Success (%)', 'DL% Distribution Across Seeds')
    axes[0].set_ylim([0, 105])
    _box(axes[1], rw_data, 'Reward', 'Reward Distribution Across Seeds')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'boxplot_comparison.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: boxplot_comparison")


def plot_radar_chart(results):
    selected = [a for a in ['random', 'greedy_ratio', 'ddqn', 'dueling_ddqn', 'ppo']
                if a in results and results[a]]
    if len(selected) < 3:
        print("⚠️ Radar chart skipped (need ≥3 algorithms)")
        return

    metrics = ['DL%', 'Reward', 'Delay', 'Fairness']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))

    all_delays = [compute_aggregate_stats(results[a])['final']['avg_delay'] for a in selected]
    all_rewards= [compute_aggregate_stats(results[a])['final']['reward']    for a in selected]
    dmax = max(all_delays) if max(all_delays) > 0 else 1.0
    rmax = max(all_rewards) if max(all_rewards) > 0 else 1.0

    for alg in selected:
        f = compute_aggregate_stats(results[alg])['final']
        vals = [
            f['dl_rate'] / 100.0,
            max(0.0, min(1.0, f['reward'] / rmax)),
            max(0.0, min(1.0, 1.0 - f['avg_delay'] / dmax)),
            f['fairness'],
        ]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2.5,
                label=_alg_label(alg), color=_alg_color(alg))
        ax.fill(angles, vals, alpha=0.1, color=_alg_color(alg))

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('Multi-Metric Algorithm Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'radar_chart.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: radar_chart")


def plot_delay_cdf(results):
    fig, ax = plt.subplots(figsize=(12, 8))
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        vals = []
        for d in seed_data:
            vals.extend((np.asarray(d['avg_delay']) * 1000).tolist())
        if not vals:
            continue
        s = np.sort(vals)
        cdf = np.arange(1, len(s)+1) / len(s)
        ax.plot(s, cdf, label=_alg_label(alg), color=_alg_color(alg), linewidth=2.5)
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Task Completion Delay CDF', fontweight='bold')
    ax.set_xlim([0, 500]); ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3); ax.legend(loc='lower right')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'delay_cdf.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: delay_cdf")


def plot_heatmap_comparison(results):
    avail = [a for a in results if results[a]]
    if not avail:
        return

    dl_vals, rw_vals, dy_vals, fv_vals = [], [], [], []
    for a in avail:
        f = compute_aggregate_stats(results[a])['final']
        dl_vals.append(f['dl_rate'] / 100.0)
        rw_vals.append(f['reward'])
        dy_vals.append(f['avg_delay'] * 1000)
        fv_vals.append(f['fairness'])

    rmax = max(rw_vals) if max(rw_vals) > 0 else 1.0
    dmax = max(dy_vals) if max(dy_vals) > 0 else 1.0

    matrix = []
    for i, a in enumerate(avail):
        matrix.append([
            dl_vals[i],
            max(0.0, min(1.0, rw_vals[i] / rmax)),
            max(0.0, min(1.0, 1.0 - dy_vals[i] / dmax)),
            fv_vals[i],
        ])
    matrix = np.array(matrix)
    cols = ['DL%', 'Reward', 'Delay', 'Fairness']

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(avail))); ax.set_yticklabels([_alg_label(a) for a in avail])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                    color='black' if matrix[i,j] > 0.5 else 'white',
                    fontsize=11, fontweight='bold')
    ax.set_title('Algorithm Performance Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized Score (1 = Best)')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_comparison.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: heatmap_comparison")


def plot_energy_vs_performance(results):
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = False
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        energies, dls = [], []
        for d in seed_data:
            if 'energy' in d and 'dl_rates' in d:
                energies.append(float(np.mean(d['energy'])))
                dls.append(float(np.mean(d['dl_rates'])))
        if not energies:
            continue
        ax.errorbar(np.mean(energies), np.mean(dls),
                    xerr=np.std(energies), yerr=np.std(dls),
                    label=_alg_label(alg), color=_alg_color(alg),
                    marker=ALGORITHMS[alg]['marker'], markersize=12,
                    capsize=5, capthick=2, elinewidth=2, linewidth=2)
        plotted = True

    if not plotted:
        plt.close()
        print("⚠️ energy_vs_performance skipped (no energy data)")
        return

    ax.set_xlabel('Energy Consumption (J)')
    ax.set_ylabel('Deadline Success (%)')
    ax.set_title('Energy vs Performance Trade-off', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper right')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'energy_vs_performance.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: energy_vs_performance")


def plot_convergence_analysis(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    avail = [a for a in results if results[a]]
    win = 10

    for alg in avail:
        agg = compute_aggregate_stats(results[alg])
        if not agg or 'rewards' not in agg['curves']:
            continue
        m = agg['curves']['rewards']['mean']
        if len(m) >= win:
            ma = np.convolve(m, np.ones(win)/win, mode='valid')
            axes[0].plot(ma, label=_alg_label(alg), color=_alg_color(alg), linewidth=2)
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel(f'MA Reward (w={win})')
    axes[0].set_title('Convergence Speed', fontweight='bold')
    axes[0].grid(True, alpha=0.3); axes[0].legend(loc='lower right')

    stab, lbls, cols_ = [], [], []
    for alg in avail:
        variances = []
        for d in results[alg]:
            r = d['rewards']
            if len(r) >= 50:
                variances.append(float(np.var(r[-50:])))
        if variances:
            stab.append(np.mean(variances)); lbls.append(_alg_label(alg)); cols_.append(_alg_color(alg))
    if stab:
        axes[1].bar(lbls, stab, color=cols_, alpha=0.85, edgecolor='black')
        axes[1].set_ylabel('Reward Variance (last 50 eps)')
        axes[1].set_title('Stability', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'convergence_analysis.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: convergence_analysis")


def plot_correlation_matrix(results):
    rows = []
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        for d in seed_data:
            n = len(d['dl_rates'])
            for ep in range(n):
                rows.append({
                    'Reward':   float(d['rewards'][ep])    if 'rewards'    in d else 0.0,
                    'DL%':      float(d['dl_rates'][ep]),
                    'Delay':    float(d['avg_delay'][ep]) * 1000,
                    'Fairness': float(d['fairness'][ep])   if 'fairness'   in d else 1.0,
                    'Energy':   float(d['energy'][ep])     if 'energy'     in d and ep < len(d['energy']) else 0.0,
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            v = corr.iloc[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='white' if abs(v) > 0.5 else 'black',
                    fontsize=11, fontweight='bold')
    ax.set_title('Metric Correlation Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Pearson correlation')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'correlation_matrix.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: correlation_matrix")


# ============================================================
# 5b. ADDED FIGURES
# ============================================================
def plot_pareto_dl_vs_delay(results):
    """DL% vs Delay scatter — best is upper-left."""
    fig, ax = plt.subplots(figsize=(10, 8))
    xs, ys, labels_, colors_ = [], [], [], []
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        f = compute_aggregate_stats(seed_data)['final']
        xs.append(f['avg_delay'] * 1000)
        ys.append(f['dl_rate'])
        labels_.append(_alg_label(alg))
        colors_.append(_alg_color(alg))
    if not xs:
        plt.close()
        return

    ax.scatter(xs, ys, c=colors_, s=280, edgecolor='black', linewidth=1.5, zorder=3)
    for x, y, l in zip(xs, ys, labels_):
        ax.annotate(l, (x, y), xytext=(8, 8),
                    textcoords='offset points', fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Delay (ms)')
    ax.set_ylabel('Deadline Success (%)')
    ax.set_title('DL% vs Delay — Pareto Frontier', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(FIGURES_DIR, f'pareto_dl_delay.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure: pareto_dl_delay")


# ============================================================
# 6. TABLES
# ============================================================
def generate_comparison_table(results, ci_results):
    rows = []
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        agg = compute_aggregate_stats(seed_data)['final']
        ci = (ci_results or {}).get(alg, {})
        dl_ci = ci.get('dl_rates')
        rw_ci = ci.get('rewards')
        fv_ci = ci.get('fairness')
        rows.append({
            'Algorithm':    ALGORITHMS[alg]['label'],
            'Type':         ALGORITHMS[alg]['type'],
            'DL%':          f"{agg['dl_rate']:.2f}",
            'DL%_std':      f"{agg['dl_rate_std']:.2f}",
            'DL%_CI':       (f"[{dl_ci['lower']:.2f}, {dl_ci['upper']:.2f}]" if dl_ci else "N/A"),
            'Reward':       f"{agg['reward']:.2f}",
            'Reward_std':   f"{agg['reward_std']:.2f}",
            'Reward_CI':    (f"[{rw_ci['lower']:.2f}, {rw_ci['upper']:.2f}]" if rw_ci else "N/A"),
            'Delay_ms':     f"{agg['avg_delay']*1000:.2f}",
            'Delay_std':    f"{agg['avg_delay_std']*1000:.2f}",
            'Fairness':     f"{agg['fairness']:.4f}",
            'Fairness_std': f"{agg['fairness_std']:.4f}",
            'Fairness_CI':  (f"[{fv_ci['lower']:.4f}, {fv_ci['upper']:.4f}]" if fv_ci else "N/A"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, 'comparison_table.csv'), index=False)
    try:
        with open(os.path.join(TABLES_DIR, 'comparison_table.tex'), 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
    except Exception:
        pass
    print("✅ Table: comparison_table")
    return df


def generate_action_table(results):
    rows = []
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        agg = compute_aggregate_stats(seed_data)
        ad = np.asarray(agg['action_dist'])
        if np.nansum(ad) == 0:
            ad = np.full(NUM_ACTIONS, np.nan)
        row = {'Algorithm': ALGORITHMS[alg]['label']}
        for i, name in enumerate(ACTION_NAMES):
            row[name] = f"{ad[i]:.2f}" if i < len(ad) else "0.00"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, 'action_distribution.csv'), index=False)
    print("✅ Table: action_distribution")
    return df


def generate_per_seed_table(results):
    """Table with one row per seed per algorithm."""
    rows = []
    for alg, seed_data in results.items():
        if not seed_data:
            continue
        for d in seed_data:
            dl = float(np.mean(d['dl_rates'])) if 'dl_rates' in d else 0.0
            rw = float(np.mean(d['rewards']))  if 'rewards'  in d else 0.0
            dy = float(np.mean(d['avg_delay']))*1000 if 'avg_delay' in d else 0.0
            fv = float(np.mean(d['fairness'])) if 'fairness' in d else 1.0
            rows.append({
                'Algorithm': ALGORITHMS[alg]['label'],
                'Seed':      d['seed'],
                'DL%':       f"{dl:.2f}",
                'Reward':    f"{rw:.2f}",
                'Delay_ms':  f"{dy:.2f}",
                'Fairness':  f"{fv:.4f}",
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, 'per_seed_results.csv'), index=False)

    try:
        piv = df.pivot(index='Seed', columns='Algorithm', values='DL%')
        piv.to_csv(os.path.join(TABLES_DIR, 'per_seed_DL_pivot.csv'))
    except Exception:
        pass
    print("✅ Table: per_seed_results")
    return df


def generate_rsu_artifact_table(results):
    """
    Report how often each algorithm picks RSUs and, where the columns
    exist, the coverage/success metrics that bound the artifact effect.
    """
    rows = []
    for alg, seed_data in results.items():
        if not seed_data:
            continue

        rsu_picks, rsu_coverage, rsu_success = [], [], []
        for d in seed_data:
            df = d.get('_df')
            if df is None:
                continue
            if 'act_rsu0' in df.columns and 'act_rsu1' in df.columns:
                rsu_picks.append((df['act_rsu0'] + df['act_rsu1']).mean())
            if 'rsu_coverage_pct' in df.columns:
                rsu_coverage.append(df['rsu_coverage_pct'].mean())
            if 'rsu_offload_success_rate' in df.columns:
                rsu_success.append(df['rsu_offload_success_rate'].mean())

        def _fmt(vals, fmt):
            return fmt.format(np.mean(vals)) if vals else "N/A"

        rows.append({
            'Algorithm':        ALGORITHMS[alg]['label'],
            'RSU_picks_%':      _fmt(rsu_picks,    "{:.1f}"),
            'RSU_coverage_%':   _fmt(rsu_coverage, "{:.1f}"),
            'RSU_success_%':    _fmt(rsu_success,  "{:.1f}"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, 'rsu_artifact_analysis.csv'), index=False)
    print("✅ Table: rsu_artifact_analysis")
    return df


# ============================================================
# 7. SUMMARY REPORT
# ============================================================
def generate_summary_report(results, ci_results, test_results):
    path = os.path.join(ANALYSIS_DIR, 'summary_report.txt')
    with open(path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("  THESIS RESULTS SUMMARY\n")
        f.write("  6G MEC Task Offloading Optimization\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Episodes per seed (cap): {EVAL_EPISODES}\n\n")

        rows = []
        for alg, seed_data in results.items():
            if not seed_data:
                continue
            agg = compute_aggregate_stats(seed_data)['final']
            rows.append((alg, agg))
        rows.sort(key=lambda t: t[1]['dl_rate'], reverse=True)

        f.write("RANKING (by mean DL%)\n")
        f.write("-"*40 + "\n")
        for i, (alg, agg) in enumerate(rows, 1):
            f.write(f"  {i}. {ALGORITHMS[alg]['label']:<18} "
                    f"DL={agg['dl_rate']:.2f}%  "
                    f"Reward={agg['reward']:.2f}  "
                    f"Delay={agg['avg_delay']*1000:.1f} ms  "
                    f"Fairness={agg['fairness']:.4f}\n")

        if test_results is not None and len(test_results) > 0:
            sig = test_results[test_results['Significant'] == True]
            f.write(f"\nSIGNIFICANT TESTS: {len(sig)}\n")
            for _, r in sig.head(20).iterrows():
                f.write(f"  - {r['Test']:<22} {r['Description']:<45} p={r['p_value']:.4e}\n")
    print(f"✅ Summary report → {path}")


# ============================================================
# 8. MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("COMPLETE DATA ANALYSIS PIPELINE")
    print(f"Phase: {PHASE_DIR_NAME}")
    print(f"Algorithms: {list(ALGORITHMS.keys())}")
    print("="*70)

    results = load_all_algorithms()
    ci_results = compute_all_confidence_intervals(results)
    test_results = run_statistical_tests(results)

    print("\nGenerating figures...")
    plot_learning_curves(results, ci_results)
    plot_final_performance(results, ci_results)
    plot_radar_chart(results)
    plot_action_distribution(results)
    plot_delay_cdf(results)
    plot_boxplot_comparison(results)
    plot_heatmap_comparison(results)
    plot_energy_vs_performance(results)
    plot_convergence_analysis(results)
    plot_correlation_matrix(results)
    plot_pareto_dl_vs_delay(results)

    print("\nGenerating tables...")
    generate_comparison_table(results, ci_results)
    generate_action_table(results)
    generate_per_seed_table(results)
    generate_rsu_artifact_table(results)

    print("\nGenerating summary report...")
    generate_summary_report(results, ci_results, test_results)

    print("\n" + "="*70)
    print(f"✅ ANALYSIS COMPLETE")
    print(f"  Figures:  {FIGURES_DIR}")
    print(f"  Tables:   {TABLES_DIR}")
    print(f"  Stats:    {STATS_DIR}")
    print(f"  Data:     {DATA_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()