#!/usr/bin/env python3
"""
UNIFIED EVALUATION SCRIPT FOR DRL ALGORITHMS
6G MEC Task Offloading Optimization
Author: Pasang Lama

Evaluates:
- DDQN
- Dueling DDQN
- PPO

Baselines (Random, Greedy-Load, Greedy-Ratio) are read from their
existing CSVs — they are NOT re-run here.

PROTOCOL (matches baselines):
- Simulator mode: --evaluate=1 (held-out evaluation)
- Episodes per seed: 600
- Seeds: [1, 7, 21, 42, 84]
- Epsilon: 0.0 (greedy action selection, no exploration)
- Learning disabled
- Load best checkpoint per seed

OUTPUT:
- Per-seed eval CSVs
- Aggregate comparison table
- Action distribution table
- Summary report
"""

import numpy as np
import pandas as pd
import signal
import csv
import os
import random
import subprocess
import sys
import time
import json
import re
import socket
import torch
import warnings
import gymnasium as gym
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore", message="Overriding environment")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
sys.modules['gym'] = gym

from ns3gym import ns3env

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================
from env_config import *

# ============================================================
# EVALUATION SETTINGS — MUST MATCH BASELINES
# ============================================================
EVAL_EPISODES = 600      # MUST match baselines (they used 600)
EVAL_SEEDS = [1, 7, 21, 42, 84]
BASE_PORT = 15000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# IMPORT DRL AGENTS FROM YOUR ACTUAL SCRIPT FILENAMES
# ============================================================
try:
    from double_dqn_multiseed import DQNAgent as DDQNAgent
    print("✅ DDQN imported")
except ImportError as e:
    print(f"⚠️ DDQN import failed: {e}")
    DDQNAgent = None

try:
    from dueling_ddqn_multiseed import DuelingDQNAgent
    print("✅ Dueling DDQN imported")
except ImportError as e:
    print(f"⚠️ Dueling DDQN import failed: {e}")
    DuelingDQNAgent = None

try:
    from ppo_agent_multiseed import PPOAgent
    print("✅ PPO imported")
except ImportError as e:
    print(f"⚠️ PPO import failed: {e}")
    PPOAgent = None

# ============================================================
# ALGORITHM CONFIGURATION — MATCHES YOUR ACTUAL DIRECTORIES
# ============================================================
ALGORITHMS = {
    'ddqn': {
        'label': 'DDQN',
        'agent_class': DDQNAgent,
        'model_dir': MODELS_DIR,           # results/phase3_v8_robust_fixed/ddqn/models
        'eval_dir': DDQN_EVAL_DIR,
        'csv_dir': DDQN_CSV_DIR,
        'type': 'rl',
        'model_patterns': [
            'double_dqn_seed{seed}_final.pt',
            'double_dqn_seed{seed}_best.pt',
        ],
    },
    'dueling_ddqn': {
        'label': 'Dueling DDQN',
        'agent_class': DuelingDQNAgent,
        'model_dir': DUELING_MODELS_DIR,
        'eval_dir': DUELING_EVAL_DIR,
        'csv_dir': DUELING_CSV_DIR,
        'type': 'rl',
        'model_patterns': [
            'dueling_ddqn_seed{seed}_final.pt',
            'dueling_ddqn_seed{seed}_best.pt',
        ],
    },
    'ppo': {
        'label': 'PPO',
        'agent_class': PPOAgent,
        'model_dir': PPO_MODELS_DIR,           # results/phase3_v8_robust_fixed/ppo/models
        'eval_dir': PPO_EVAL_DIR,
        'csv_dir': PPO_CSV_DIR,
        'type': 'rl',
        'model_patterns': [
            # v8.6.29: prefer _final.pt for PPO because the _best.pt
            # checkpoint was saved at the training peak (~99% DL) while
            # _final.pt reflects the converged policy (~86% DL mean).
            # Using _best.pt would overstate PPO's held-out performance.
            'ppo_seed{seed}_final.pt',
            'ppo_seed{seed}_best.pt',
        ],
    },
}

# ============================================================
# BASELINE RESULTS — READ FROM EXISTING SUMMARIES, NOT RE-RUN
# ============================================================
# These are the values from your already-completed baseline runs.
# Fill these in from your random/greedy summaries.
BASELINE_RESULTS = {
    'random': {
        'label': 'Random',
        'dl_rate': 55.99,
        'dl_rate_std': 0.26,
        'fairness': 0.9856,
        'avg_delay': 100.33,
        'avg_delay_std': 0.39,
    },
    'greedy_load': {
        'label': 'Greedy (Load)',
        'dl_rate': 68.60,
        'dl_rate_std': 0.11,
        'fairness': 0.786,
        'avg_delay': 90.65,
        'avg_delay_std': 0.05,
    },
    'greedy_ratio': {
        'label': 'Greedy (Ratio)',
        'dl_rate': 78.81,
        'dl_rate_std': 0.06,
        'fairness': 0.709,
        'avg_delay': 82.44,
        'avg_delay_std': 0.02,
    },
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def kill_process_on_port(port):
    if not _PSUTIL:
        return
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                try:
                    psutil.Process(conn.pid).kill()
                except Exception:
                    pass
    except Exception:
        pass

def kill_zombie_ns3():
    if not _PSUTIL:
        return
    try:
        for proc in psutil.process_iter(['pid', 'cmdline']):
            cmdline = proc.info.get('cmdline') or []
            if any('mec_simulator' in str(c) for c in cmdline):
                try:
                    proc.kill()
                except Exception:
                    pass
    except Exception:
        pass

def wait_port_free(port, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        if not is_port_in_use(port):
            time.sleep(2)
            return True
        time.sleep(1)
    return False

def kill_ns3(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    except Exception:
        pass

# ============================================================
# NS-3 LAUNCHER — CRITICAL: --evaluate=1
# ============================================================
def launch_ns3(seed, port):
    """Launch ns-3 in EVALUATION mode."""
    if is_port_in_use(port):
        print(f"  ⚠️ Port {port} in use, killing...")
        kill_process_on_port(port)
        time.sleep(2)

    cmd = [
        f"{NS3_PATH}/ns3", "run",
        f"mec_simulator_v8_robust_fixed --seed={seed} --port={port} --evaluate=1"
    ]
    #                                                                         ^^^^^^^^^^^^
    #                                                                         CRITICAL: EVAL MODE

    env = os.environ.copy()
    if "PATH" not in env:
        env["PATH"] = "/usr/bin:/bin:/usr/local/bin"

    proc = subprocess.Popen(
        cmd,
        cwd=NS3_PATH,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env
    )

    # Wait for startup
    time.sleep(8)
    if proc.poll() is not None:
        raise RuntimeError(f"ns-3 crashed immediately (seed={seed})")

    return proc

# ============================================================
# OBSERVATION HELPERS
# ============================================================
def sanitize_obs(obs):
    if obs is None:
        return np.zeros(OBS_SIZE, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32).flatten()
    if len(obs) < OBS_SIZE:
        obs = np.pad(obs, (0, OBS_SIZE - len(obs)), 'constant')
    return np.nan_to_num(obs[:OBS_SIZE], nan=0.0, posinf=1.0, neginf=-1.0)

def safe_reset(env):
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return sanitize_obs(obs) if obs is not None else np.zeros(OBS_SIZE, dtype=np.float32)
    except Exception:
        return np.zeros(OBS_SIZE, dtype=np.float32)

# ============================================================
# TIMEOUT HANDLING
# ============================================================
class StepTimeout(Exception):
    pass

class EnvCloseTimeout(Exception):
    pass

def _timeout_handler(s, f):
    raise StepTimeout()

def _close_timeout_handler(s, f):
    raise EnvCloseTimeout()

def step_with_timeout(env, action, timeout_sec=STEP_TIMEOUT):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            obs, reward, done, info = result
            done = bool(done)
        else:
            obs, reward, done = result
            info = None
        return sanitize_obs(obs), float(reward), bool(done), info
    except StepTimeout:
        return np.zeros(OBS_SIZE, dtype=np.float32), -1.0, True, None
    except Exception:
        return None, -2.0, True, None
    finally:
        signal.alarm(0)

def close_env_safe(env, timeout_sec=5):
    if env is None:
        return
    signal.signal(signal.SIGALRM, _close_timeout_handler)
    signal.alarm(timeout_sec)
    try:
        env.close()
    except Exception:
        pass
    finally:
        signal.alarm(0)

# ============================================================
# INFO PARSER
# ============================================================
def parse_info(info):
    """Parse info dict or JSON string for key metrics."""
    if info is None:
        return {}
    if isinstance(info, dict):
        return info
    if isinstance(info, str):
        s = info.strip()
        if s.startswith('{'):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    return {}

# ============================================================
# EPISODE STATS
# ============================================================
class EpisodeStats:
    def __init__(self):
        self.reward = 0.0
        self.step_count = 0
        self.delays = []
        self.action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self.dl_rate = 0.0
        self.handovers = 0
        # v8.6.30: track max deadlines_met / total_steps seen this episode.
        # The C++ resets accumulators at episode boundaries, so the
        # last-reported dl_rate is ~0 right after reset. The maximum
        # over the episode captures the true episode DL.
        self._max_dm = 0
        self._max_ts = 0
        # v8.6.30: accumulate fairness/delay/energy across steps, then
        # average at the end. The C++ values evolve during an episode;
        # taking max or last gives wrong answers.
        self._fairness_sum = 0.0
        self._fairness_count = 0
        self._delay_sum = 0.0
        self._delay_count = 0
        self._energy_sum = 0.0
        self._energy_count = 0

    def record_step(self, action, reward, info):
        self.reward += reward
        self.step_count += 1
        self.action_counts[action] += 1

        d = parse_info(info)

        # v8.6.30: use deadlines_met / total_steps (monotonic during
        # an episode) to compute the running DL. Take the max so the
        # post-reset zeros at episode end don't overwrite the true DL.
        dm = d.get('deadlines_met')
        ts = d.get('total_steps')
        if dm is not None and ts is not None:
            try:
                dm_i = int(dm)
                ts_i = int(ts)
                if ts_i > self._max_ts:
                    self._max_ts = ts_i
                    self._max_dm = dm_i
                    self.dl_rate = 100.0 * dm_i / ts_i
            except (TypeError, ValueError):
                pass
        elif 'dl_rate' in d:
            # Fallback if deadlines_met/total_steps are missing.
            try:
                v = float(d['dl_rate'])
                if v > self.dl_rate:
                    self.dl_rate = v
            except (TypeError, ValueError):
                pass

        # v8.6.30: fairness and delay must be AVERAGED across the
        # episode, not maxed. The C++ values evolve during an episode
        # (fairness starts at 1.0 and declines as load concentrates;
        # delay starts near 0 and climbs as tasks accumulate). Taking
        # the max or last value misrepresents the episode.
        if 'jains_fairness' in d:
            try:
                v = float(d['jains_fairness'])
                if 0.0 < v <= 1.0:
                    self._fairness_sum += v
                    self._fairness_count += 1
            except (TypeError, ValueError):
                pass

        if 'handover_count' in d:
            try:
                self.handovers += int(d['handover_count'])
            except (TypeError, ValueError):
                pass

        if 'avg_energy_j' in d:
            try:
                v = float(d['avg_energy_j'])
                if v > 0:
                    self._energy_sum += v
                    self._energy_count += 1
            except (TypeError, ValueError):
                pass

        if 'avg_delay_ms' in d:
            try:
                v = float(d['avg_delay_ms'])
                if v > 0:
                    self._delay_sum += v
                    self._delay_count += 1
            except (TypeError, ValueError):
                pass

    @property
    def fairness(self):
        if self._fairness_count == 0:
            return 1.0
        return self._fairness_sum / self._fairness_count

    @property
    def avg_delay_ms(self):
        if self._delay_count == 0:
            return 0.0
        return self._delay_sum / self._delay_count

    @property
    def energy(self):
        if self._energy_count == 0:
            return 0.0
        return self._energy_sum / self._energy_count

    def action_distribution(self):
        total = self.action_counts.sum()
        if total == 0:
            return np.zeros(NUM_ACTIONS)
        return 100.0 * self.action_counts / total

# ============================================================
# MODEL LOADING
# ============================================================
def find_model_path(algorithm_name, seed, config):
    """Find model checkpoint path."""
    model_dir = config['model_dir']
    if not os.path.exists(model_dir):
        return None

    # Try with seed subdirectory first
    for pattern in config['model_patterns']:
        p = os.path.join(model_dir, f"seed_{seed}", pattern.format(seed=seed))
        if os.path.exists(p):
            return p

    # Try without seed subdirectory
    for pattern in config['model_patterns']:
        p = os.path.join(model_dir, pattern.format(seed=seed))
        if os.path.exists(p):
            return p

    # Fallback: search recursively
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith('.pt') and f'seed{seed}' in f:
                return os.path.join(root, f)
            if f.endswith('.pt') and f'seed_{seed}' in f:
                return os.path.join(root, f)

    return None

def load_agent(algorithm_name, config, seed):
    """Load agent with trained model."""
    agent_class = config['agent_class']
    if agent_class is None:
        print(f"  ❌ Agent class not available for {algorithm_name}")
        return None

    try:
        agent = agent_class()
    except Exception as e:
        print(f"  ❌ Failed to instantiate agent: {e}")
        return None

    model_path = find_model_path(algorithm_name, seed, config)
    if model_path is None:
        print(f"  ⚠️ No model found for {algorithm_name} seed {seed}")
        print(f"     Looking in: {config['model_dir']}")
        return None

    try:
        print(f"  Loading: {os.path.basename(model_path)}")
        checkpoint = torch.load(model_path, map_location=DEVICE)

        if isinstance(checkpoint, dict):
            # DDQN / Dueling DDQN format: policy / target
            if 'policy' in checkpoint:
                agent.policy_net.load_state_dict(checkpoint['policy'], strict=False)
                if hasattr(agent, 'target_net') and 'target' in checkpoint:
                    agent.target_net.load_state_dict(checkpoint['target'], strict=True)
                if hasattr(agent, 'policy_net'):
                    agent.policy_net.eval()
                if hasattr(agent, 'target_net'):
                    agent.target_net.eval()

            # PPO format: actor / critic
            elif 'actor' in checkpoint:
                agent.actor.load_state_dict(checkpoint['actor'], strict=True)
                if hasattr(agent, 'critic') and 'critic' in checkpoint:
                    agent.critic.load_state_dict(checkpoint['critic'], strict=True)
                if hasattr(agent, 'target_critic') and 'target_critic' in checkpoint:
                    agent.target_critic.load_state_dict(checkpoint['target_critic'], strict=True)
                if hasattr(agent, 'actor'):
                    agent.actor.eval()
                if hasattr(agent, 'critic'):
                    agent.critic.eval()
            else:
                print(f"  ⚠️ Unknown checkpoint keys: {list(checkpoint.keys())}")
                return None

        # Freeze exploration for all RL agents
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0

        print(f"  ✅ Model loaded, epsilon frozen at 0.0")
        return agent

    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# EVALUATE ONE SEED
# ============================================================
def evaluate_seed(seed, algorithm_name, agent, config):
    """Run one seed's evaluation and save results."""
    port = BASE_PORT + seed
    print(f"\n  Seed {seed} (port {port})...")

    kill_zombie_ns3()
    kill_process_on_port(port)
    wait_port_free(port)

    ns3_proc = launch_ns3(seed, port)
    if ns3_proc is None:
        print(f"  ❌ Failed to launch ns-3")
        return None

    env = None
    for attempt in range(10):
        try:
            env = ns3env.Ns3Env(port=port, startSim=False, simSeed=seed)
            break
        except Exception as e:
            time.sleep(3)

    if env is None:
        print(f"  ❌ Failed to connect to ns-3")
        kill_ns3(ns3_proc)
        return None

    # Results
    episode_results = []

    # Output CSV
    eval_dir = os.path.join(config['eval_dir'], f"seed_{seed}")
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, f"{algorithm_name}_seed{seed}_eval.csv")

    obs = safe_reset(env)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'reward', 'dl_rate', 'fairness', 'avg_delay_ms',
            'handover_count', 'energy_j',
            'act_local', 'act_mec0', 'act_mec1', 'act_mec2',
            'act_rsu0', 'act_rsu1', 'act_sat',
        ])

        for ep in tqdm(range(EVAL_EPISODES), desc=f"Seed {seed}", leave=False):
            stats = EpisodeStats()
            ep_done = False

            while stats.step_count < MAX_STEPS and not ep_done:
                try:
                    action_result = agent.select_action(obs)
                    if isinstance(action_result, tuple):
                        action = action_result[0]
                    else:
                        action = action_result
                except Exception as e:
                    print(f"  ⚠️ Action failed: {e}, using random")
                    action = random.randint(0, NUM_ACTIONS - 1)

                result = step_with_timeout(env, action)
                if result[0] is None:
                    break

                next_obs, reward, done, info = result
                stats.record_step(action, reward, info)

                obs = next_obs
                if done:
                    ep_done = True
                    break

            # Log episode
            ad = stats.action_distribution()
            writer.writerow([
                ep,
                round(stats.reward, 4),
                round(stats.dl_rate, 2),
                round(stats.fairness, 4),
                round(stats.avg_delay_ms, 2),
                stats.handovers,
                round(stats.energy, 4),
                round(ad[0], 2), round(ad[1], 2), round(ad[2], 2), round(ad[3], 2),
                round(ad[4], 2), round(ad[5], 2), round(ad[6], 2),
            ])
            f.flush()

            episode_results.append({
                'dl_rate': stats.dl_rate,
                'fairness': stats.fairness,
                'avg_delay_ms': stats.avg_delay_ms,
                'reward': stats.reward,
                'handovers': stats.handovers,
                'energy': stats.energy,
                'action_dist': ad,
            })

    close_env_safe(env)
    kill_ns3(ns3_proc)
    kill_process_on_port(port)

    # Aggregate across episodes
    dls = [e['dl_rate'] for e in episode_results]
    fairs = [e['fairness'] for e in episode_results]
    delays = [e['avg_delay_ms'] for e in episode_results]
    rewards = [e['reward'] for e in episode_results]
    adists = np.array([e['action_dist'] for e in episode_results])

    seed_summary = {
        'seed': seed,
        'dl_mean': float(np.mean(dls)),
        'dl_std': float(np.std(dls)),
        'fairness_mean': float(np.mean(fairs)),
        'delay_mean': float(np.mean(delays)),
        'reward_mean': float(np.mean(rewards)),
        'action_dist': np.mean(adists, axis=0).tolist(),
        'n_episodes': len(episode_results),
    }

    print(f"  ✅ Seed {seed}: DL={seed_summary['dl_mean']:.2f}%, "
          f"Fair={seed_summary['fairness_mean']:.3f}, "
          f"Delay={seed_summary['delay_mean']:.1f}ms")

    return seed_summary

# ============================================================
# EVALUATE ALGORITHM ACROSS ALL SEEDS
# ============================================================
def evaluate_algorithm(algorithm_name, config):
    print(f"\n{'='*70}")
    print(f"EVALUATING: {algorithm_name.upper()} ({config['label']})")
    print(f"{'='*70}")

    seed_summaries = []

    for seed in EVAL_SEEDS:
        agent = load_agent(algorithm_name, config, seed)
        if agent is None:
            print(f"  ⚠️ Skipping seed {seed}")
            continue

        summary = evaluate_seed(seed, algorithm_name, agent, config)
        if summary is not None:
            seed_summaries.append(summary)

    if not seed_summaries:
        return None

    # Aggregate across seeds
    dl_means = [s['dl_mean'] for s in seed_summaries]
    fair_means = [s['fairness_mean'] for s in seed_summaries]
    delay_means = [s['delay_mean'] for s in seed_summaries]
    reward_means = [s['reward_mean'] for s in seed_summaries]
    act_dists = np.array([s['action_dist'] for s in seed_summaries])

    return {
        'label': config['label'],
        'seeds': [s['seed'] for s in seed_summaries],
        'dl_rate': float(np.mean(dl_means)),
        'dl_rate_std': float(np.std(dl_means)),
        'fairness': float(np.mean(fair_means)),
        'fairness_std': float(np.std(fair_means)),
        'avg_delay': float(np.mean(delay_means)),
        'avg_delay_std': float(np.std(delay_means)),
        'reward': float(np.mean(reward_means)),
        'action_dist': np.mean(act_dists, axis=0).tolist(),
        'per_seed': seed_summaries,
    }

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
def build_comparison_table(drl_results):
    """Combine baselines (static) with DRL eval results (fresh)."""
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE — Held-out Evaluation (600 episodes × 5 seeds)")
    print("="*80)
    print(f"{'Algorithm':<18} {'DL%':>16} {'Fairness':>12} {'Delay(ms)':>14}")
    print("-"*80)

    rows = []

    # Baselines (from static values)
    for key in ['random', 'greedy_load', 'greedy_ratio']:
        b = BASELINE_RESULTS[key]
        print(f"{b['label']:<18} "
              f"{b['dl_rate']:>7.2f}±{b['dl_rate_std']:<6.2f} "
              f"{b['fairness']:>11.3f} "
              f"{b['avg_delay']:>8.2f}±{b['avg_delay_std']:<4.2f}")
        rows.append({
            'Algorithm': b['label'],
            'Type': 'baseline',
            'DL%': b['dl_rate'],
            'DL%_std': b['dl_rate_std'],
            'Fairness': b['fairness'],
            'Delay_ms': b['avg_delay'],
            'Delay_std': b['avg_delay_std'],
        })

    # DRL results (from fresh evaluation)
    for name, r in drl_results.items():
        if r is None:
            continue
        print(f"{r['label']:<18} "
              f"{r['dl_rate']:>7.2f}±{r['dl_rate_std']:<6.2f} "
              f"{r['fairness']:>11.3f} "
              f"{r['avg_delay']:>8.2f}±{r['avg_delay_std']:<4.2f}")
        rows.append({
            'Algorithm': r['label'],
            'Type': 'rl',
            'DL%': r['dl_rate'],
            'DL%_std': r['dl_rate_std'],
            'Fairness': r['fairness'],
            'Delay_ms': r['avg_delay'],
            'Delay_std': r['avg_delay_std'],
        })

    print("="*80)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "final_comparison_eval.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Comparison table saved: {out_csv}")

    return df

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*80)
    print("DRL EVALUATION — Held-out (600 episodes × 5 seeds, --evaluate=1)")
    print(f"Seeds: {EVAL_SEEDS}   Episodes/seed: {EVAL_EPISODES}")
    print("="*80)

    available = []
    if DDQNAgent is not None:
        available.append('ddqn')
    if DuelingDQNAgent is not None:
        available.append('dueling_ddqn')
    if PPOAgent is not None:
        available.append('ppo')

    print(f"\nDRL agents to evaluate: {available}\n")

    drl_results = {}

    for alg_name in available:
        config = ALGORITHMS[alg_name]
        result = evaluate_algorithm(alg_name, config)
        if result is not None:
            drl_results[alg_name] = result

            # Save per-algorithm summary JSON
            out_json = os.path.join(RESULTS_DIR, f"eval_{alg_name}.json")
            with open(out_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  ✅ Saved: {out_json}")

    if not drl_results:
        print("\n❌ No DRL results collected.")
        return

    # Final comparison table
    df = build_comparison_table(drl_results)

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()