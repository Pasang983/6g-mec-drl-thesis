# =============================================================================
# ppo_agent_multiseed.py  —  Proximal Policy Optimization (v8.6.29)
#
# Master's Thesis: Task Offloading Optimization in 6G MEC/VEC/Satellite
# Author : Pasang Lama — IST Lisboa
#
# v8.6.29 REWARD-SHAPING MERGE:
#   - Replaced identity reward with per-component shaped reward
#     (deadline 0.60, delay 0.15, energy 0.10, load 0.10,
#      penalty 0.05, backhaul 0.05). This is the reward the
#      policy-gradient method actually needs; the C++ composite
#      reward is dense enough for value-based methods but not
#      for PPO.
#   - Reward clipped to [-1, +1] for policy-gradient stability.
#   - Multi-episode accumulation: 5 episodes per PPO update
#     (~1000 samples/update vs 200 previously).
#
# v8.6.28 (retained):
#   - target_kl = 0.15 (relaxed from 0.06)
#   - kl_early_stop_patience = 15
#   - actor_lr = 1e-4, critic_lr = 3e-4
#   - entropy_coef = 0.02
#   - epochs_per_update = 10
#   - minibatch_size = 64
#   - Stagnation heuristic REMOVED (was reducing exploration)
#
# Cite: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
# =============================================================================

from env_config import *

import numpy as np
import signal, csv, os, random, subprocess, sys, time, json, re, socket
import pandas as pd
from collections import deque
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
import gymnasium as gym
warnings.filterwarnings("ignore", message="Overriding environment")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
sys.modules['gym'] = gym

from ns3gym import ns3env

# ============================================================
# HYPERPARAMETERS
# ============================================================
GAMMA = 0.99
LR_END = 5.0e-5
MAX_GRAD_NORM = 1.0
REWARD_LOW = -10.0
REWARD_HIGH = 10.0
USE_CYCLIC_EXPLORATION = False

# ============================================================
# CONSTANTS
# ============================================================
kGnbHomeMec = [0, 0, 1, 1, 2, 2]

# ============================================================
# OVERRIDES
# ============================================================
MAX_STEPS = 200
NUM_EPISODES = 600
EPISODES_PER_UPDATE = 5          # ★ v8.6.29: multi-episode accumulation

# ============================================================
# CONVERGENCE DETECTION
# ============================================================
CONVERGENCE_WINDOW = 50
CONVERGENCE_STDDEV_THRESHOLD = 4.0
CONVERGENCE_MIN_EPISODES = 500
CONVERGENCE_CONSECUTIVE_STABLE = 8

# ============================================================
# DEVICE & CONFIG
# ============================================================
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
print(f"[PPO] Training device: {DEVICE}")
print(f"[PPO] OBS_SIZE={OBS_SIZE}  NUM_ACTIONS={NUM_ACTIONS}")
print(f"[PPO] MAX_STEPS={MAX_STEPS}  NUM_EPISODES={NUM_EPISODES}")
print(f"[PPO] EPISODES_PER_UPDATE={EPISODES_PER_UPDATE}")

assert OBS_SIZE == 48, f"OBS_SIZE should be 48, got {OBS_SIZE}"

# ============================================================
# PPO HYPERPARAMETERS (v8.6.29 — shaped-reward-tuned)
# ============================================================
PPO_CONFIG = {
    # Learning
    'actor_lr': 1.0e-4,
    'critic_lr': 3.0e-4,
    'lr_warmup_steps': 300,
    'lr_decay': True,
    'adaptive_lr': True,
    'lr_min': 5.0e-5,
    'lr_max': 3.0e-4,
    'lr_reduction_factor': 0.9995,
    'loss_improvement_threshold': 0.01,

    # PPO core
    'gamma': 0.99,
    'gae_lambda': 0.96,
    'clip_epsilon': 0.08,

    # Exploration
    'entropy_coef': 0.02,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.4,

    # Training
    'epochs_per_update': 10,
    'minibatch_size': 64,
    'buffer_size': 8192,

    # KL
    'target_kl': 0.15,
    'kl_early_stop_patience': 15,

    # Target network
    'tau': 0.002,
    'update_target_every': 5,
}

# =============================================================================
# SEED CONTROL
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if TORCH_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# PROCESS / PORT UTILITIES
# =============================================================================
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    print("WARNING: psutil not installed — zombie cleanup disabled")

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

def find_free_port(start_port, max_tries=50):
    for offset in range(max_tries):
        port = start_port + offset
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No free port near {start_port} after {max_tries} tries")

# =============================================================================
# NS-3 LAUNCHER
# =============================================================================
def launch_ns3(seed, port):
    cmd = [
        f"{NS3_PATH}/ns3", "run",
        f"mec_simulator_v8_robust_fixed --seed={seed} --port={port} --evaluate=0"
    ]
    print(f"Launching ns-3 v8_robust_fixed for seed {seed} on port {port}...")

    env = os.environ.copy()
    if "PATH" not in env:
        env["PATH"] = "/usr/bin:/bin:/usr/local/bin"

    proc = subprocess.Popen(
        cmd, cwd=NS3_PATH, text=True, bufsize=1,
        universal_newlines=True, env=env
    )

    print(f"  ns-3 PID={proc.pid}")
    time.sleep(8)
    if proc.poll() is not None:
        raise RuntimeError(f"ns-3 crashed immediately (seed={seed})")
    return proc

# =============================================================================
# OBSERVATION HELPERS
# =============================================================================
def sanitize_obs(obs):
    if obs is None:
        return np.zeros(OBS_SIZE, dtype=np.float32)
    if isinstance(obs, (int, float, np.number)):
        obs = np.array([obs], dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32).flatten()
    if len(obs) < OBS_SIZE:
        obs = np.pad(obs, (0, OBS_SIZE - len(obs)), 'constant')
    return np.nan_to_num(obs[:OBS_SIZE], nan=0.0, posinf=1.0, neginf=-1.0)

def safe_index(arr, idx, default=0.0):
    arr = np.asarray(arr).flatten()
    if idx < 0 or idx >= len(arr):
        return default
    return float(arr[idx])

def decode_ue_type(obs):
    if safe_index(obs, 2) > 0.5:
        return 0
    elif safe_index(obs, 3) > 0.5:
        return 1
    elif safe_index(obs, 4) > 0.5:
        return 2
    return 0

def compute_jains_fairness(obs):
    loads = [
        safe_index(obs, 7),
        safe_index(obs, 8),
        safe_index(obs, 9),
        safe_index(obs, 10) * 0.5,
        safe_index(obs, 11) * 0.5,
    ]
    sum_loads = sum(loads)
    if sum_loads < 1e-6:
        return 1.0
    sum_sq = sum([l * l for l in loads])
    n = len(loads)
    fairness = (sum_loads * sum_loads) / (n * sum_sq)
    return float(np.clip(fairness, 0.0, 1.0))

# =============================================================================
# CONVERGENCE DETECTION
# =============================================================================
class ConvergenceDetector:
    def __init__(self, window=CONVERGENCE_WINDOW,
                 stddev_threshold=CONVERGENCE_STDDEV_THRESHOLD,
                 min_episodes=CONVERGENCE_MIN_EPISODES):
        self.window = window
        self.stddev_threshold = stddev_threshold
        self.min_episodes = min_episodes
        self.history = deque(maxlen=window)
        self.converged = False
        self.convergence_episode = -1
        self.consecutive_stable = 0

    def update(self, dl_rate, episode):
        self.history.append(dl_rate)
        if len(self.history) < self.window or episode < self.min_episodes:
            return False, "insufficient_data"
        stddev_dl = np.std(self.history)
        if stddev_dl < self.stddev_threshold:
            self.consecutive_stable += 1
            if self.consecutive_stable >= CONVERGENCE_CONSECUTIVE_STABLE:
                if not self.converged:
                    self.converged = True
                    self.convergence_episode = episode
                return True, "converged"
        else:
            self.consecutive_stable = 0
            self.converged = False
        return self.converged, f"std={stddev_dl:.2f}%"

    def get_stats(self):
        if len(self.history) < self.window:
            return {'converged': False, 'convergence_episode': -1,
                    'window_size': len(self.history), 'mean_dl': 0.0,
                    'stddev_dl': 0.0, 'stability': 'insufficient_data'}
        return {'converged': self.converged,
                'convergence_episode': self.convergence_episode,
                'window_size': self.window,
                'mean_dl': np.mean(self.history),
                'stddev_dl': np.std(self.history),
                'stability': 'converged' if self.converged else 'not_converged'}

# ==============================================================
# RUNNING STATS
# ==============================================================
class RunningStats:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0; self.std = 1.0; self.count = 0
        self.M2 = 0.0; self.epsilon = epsilon; self.warmup_count = 100

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            self.std = np.sqrt(self.M2 / (self.count - 1))

    def normalize(self, x):
        if self.count < self.warmup_count:
            return x
        std = max(self.std, 0.1)
        return np.clip((x - self.mean) / std, -3.0, 3.0)

    def reset(self):
        self.mean = 0.0; self.std = 1.0; self.count = 0; self.M2 = 0.0

# ==============================================================
# SHAPED REWARD (v8.6.29 — from the shaped-reward reference)
# ==============================================================
class RewardShaper:
    """
    Per-component shaped reward for policy-gradient training.

    Weights (sum to 1.0):
        deadline 0.60  — primary task completion signal
        delay    0.15  — latency
        energy   0.10  — efficiency
        load     0.10  — load balancing
        penalty  0.05  — bad decisions
        backhaul 0.05  — network cost

    The C++ composite reward is dense enough for value-based
    methods (DDQN/Dueling) but too sparse/noisy for policy-gradient
    methods. This shaper decomposes the reward into smooth
    functions of the observation that are directly attributable
    to the chosen action.
    """

    def __init__(self):
        self.weights = {
            'deadline': 0.60,
            'delay':    0.15,
            'energy':   0.10,
            'load':     0.10,
            'penalty':  0.05,
            'backhaul': 0.05,
        }
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total

        self.stats = {k: RunningStats() for k in self.weights}

        # Reward smoothing
        self.smooth_factor = 0.5
        self.smoothed_reward = 0.0

        # Consistency tracking
        self.last_action = -1
        self.consecutive_count = 0
        self.action_history = deque(maxlen=30)

        self.episode_count = 0

        # ★ v8.6.29: tighter clipping for policy-gradient stability
        self.reward_min = -1.0
        self.reward_max =  1.0

        self.raw_components = {}

        print("✅ RewardShaper initialized (SHAPED, v8.6.29)")
        print(f"   Weights: {self.weights}")

    # ---- component functions ----------------------------------------

    def compute_deadline_reward(self, chosen_ratio, urgency, task_size):
        if chosen_ratio <= 1.0:
            margin = 1.0 - chosen_ratio
            base = 0.5 + 0.5 * min(margin, 1.0)
            offload_bonus = 0.1 * task_size if task_size > 0.5 else 0.0
            urgency_bonus = 0.1 * urgency if urgency > 0.5 else 0.0
            return float(np.clip(base + offload_bonus + urgency_bonus, 0.0, 1.5))
        else:
            violation = min(chosen_ratio - 1.0, 2.0)
            base = -violation
            urgency_penalty = -0.5 * urgency * min(violation, 0.5)
            return float(np.clip(base + urgency_penalty, -2.0, 0.0))

    def compute_delay_reward(self, avg_delay_ms, urgency):
        if urgency > 0.7:
            if avg_delay_ms < 50:
                return 0.8 * (1.0 - avg_delay_ms / 50.0)
            elif avg_delay_ms < 100:
                return -0.3 * ((avg_delay_ms - 50.0) / 50.0)
            else:
                return -0.6
        else:
            if avg_delay_ms < 100:
                return 0.4 * (1.0 - avg_delay_ms / 100.0)
            elif avg_delay_ms < 200:
                return -0.15 * ((avg_delay_ms - 100.0) / 100.0)
            else:
                return -0.4

    def compute_energy_reward(self, energy_j, action, obs):
        if action == 0:
            cpu = safe_index(obs, 0); data = safe_index(obs, 1)
            task_size = (cpu + data) / 2.0
            if task_size < 0.3: return 0.5
            elif task_size < 0.6: return 0.0
            else: return -0.5
        elif 1 <= action <= 3:
            load = safe_index(obs, 7 + (action - 1))
            if load < 0.3: return 0.4
            elif load < 0.6: return 0.0
            else: return -0.3
        elif 4 <= action <= 5:
            load = safe_index(obs, 10 + (action - 4))
            if load < 0.3: return 0.0
            else: return -0.3
        else:
            return -0.8

    def compute_load_reward(self, chosen_load, action, obs):
        mec_loads = [safe_index(obs, 7), safe_index(obs, 8), safe_index(obs, 9)]
        penalty = -0.6 * min(chosen_load, 1.0)
        if 1 <= action <= 3:
            avg_mec = np.mean(mec_loads)
            if avg_mec > 0.1:
                normalized = chosen_load / max(avg_mec, 0.01)
                if normalized < 0.8: penalty += 0.3
                elif normalized > 1.2: penalty -= 0.3
        return float(np.clip(penalty, -1.0, 1.0))

    def compute_penalty_reward(self, action, obs, task_size, urgency, handover):
        penalty = 0.0
        if action == 0:
            if task_size > 0.6: penalty -= 0.3 * (task_size - 0.6) / 0.4
            if urgency > 0.7: penalty -= 0.15
        if action == 6: penalty -= 0.2
        if handover > 0.5: penalty -= 0.1
        if 1 <= action <= 3:
            gnb = int(round(safe_index(obs, 37) * 5))
            gnb = max(0, min(5, gnb))
            if kGnbHomeMec[gnb] != (action - 1): penalty -= 0.05
        return 1.0 + min(penalty, 0.0)

    def compute_backhaul_reward(self, action, obs):
        cost = 0.0
        if 1 <= action <= 3:
            cost = min(safe_index(obs, 40 + (action - 1)), 1.0)
            gnb = int(round(safe_index(obs, 37) * 5))
            gnb = max(0, min(5, gnb))
            if kGnbHomeMec[gnb] != (action - 1): cost += 0.2
        elif 4 <= action <= 5:
            cost = 0.1
        elif action == 6:
            cost = 0.8
        return 1.0 - min(cost, 1.0)

    def compute_offload_incentive(self, action, obs, task_size, urgency):
        if action == 0:
            penalty = 0.0
            if task_size > 0.7: penalty = -0.3 * (task_size - 0.7) / 0.3
            if urgency > 0.8: penalty -= 0.2
            return float(np.clip(penalty, -0.6, 0.0))
        elif 1 <= action <= 3:
            load = safe_index(obs, 7 + (action - 1))
            return float(np.clip((1.0 - load) * 0.3 + 0.3 * task_size, 0.0, 0.6))
        elif 4 <= action <= 5:
            return 0.25 if decode_ue_type(obs) == 2 else 0.0
        elif action == 6:
            return 0.15 if (task_size > 0.6 and urgency > 0.7) else -0.25
        return 0.0

    def compute_consistency_bonus(self, action):
        self.action_history.append(action)
        if len(self.action_history) < 10:
            return 0.0
        last5 = list(self.action_history)[-5:]
        if len(set(last5)) == 1: return 0.08
        last10 = list(self.action_history)[-10:]
        if len(set(last10)) == 1: return 0.15
        return 0.0

    # ---- main shape ------------------------------------------------

    def shape(self, env_reward, obs, action):
        obs = sanitize_obs(obs)

        chosen_ratio = safe_index(obs, 12 + action) * 3.0
        urgency = safe_index(obs, 6)
        task_size = (safe_index(obs, 0) + safe_index(obs, 1)) / 2.0
        handover = safe_index(obs, 39)
        avg_delay_ms = safe_index(obs, 5) * 200.0

        chosen_load = 0.0
        if 1 <= action <= 3:
            chosen_load = safe_index(obs, 7 + (action - 1))
        elif 4 <= action <= 5:
            chosen_load = safe_index(obs, 10 + (action - 4))

        energy_j = 0.1 + 0.5 * chosen_load
        if action == 0:
            energy_j = 0.05 + 0.3 * task_size

        components = {
            'deadline': self.compute_deadline_reward(chosen_ratio, urgency, task_size),
            'delay':    self.compute_delay_reward(avg_delay_ms, urgency),
            'energy':   self.compute_energy_reward(energy_j, action, obs),
            'load':     self.compute_load_reward(chosen_load, action, obs),
            'penalty':  self.compute_penalty_reward(action, obs, task_size, urgency, handover),
            'backhaul': self.compute_backhaul_reward(action, obs),
        }
        self.raw_components = components

        offload_bonus = self.compute_offload_incentive(action, obs, task_size, urgency)
        consistency_bonus = self.compute_consistency_bonus(action)

        r = 0.0
        for key, value in components.items():
            self.stats[key].update(value)
            normalized = self.stats[key].normalize(value)
            r += self.weights[key] * normalized

        r += offload_bonus * 0.08
        r += consistency_bonus * 0.02

        if action == self.last_action:
            self.consecutive_count += 1
            if self.consecutive_count > 5:
                r += 0.05
        else:
            self.consecutive_count = 1
        self.last_action = action

        if self.episode_count == 0:
            self.smoothed_reward = r
        else:
            self.smoothed_reward = (1 - self.smooth_factor) * self.smoothed_reward + self.smooth_factor * r

        return float(np.clip(self.smoothed_reward, self.reward_min, self.reward_max))

    def reset_episode(self):
        self.consecutive_count = 0
        self.last_action = -1
        self.action_history.clear()
        self.episode_count += 1
        self.raw_components = {}

    def get_weight_summary(self):
        return self.weights

    def get_stats_summary(self):
        return {k: {'mean': s.mean, 'std': s.std, 'count': s.count}
                for k, s in self.stats.items()}

# =============================================================================
# NS-3 INFO PARSER
# =============================================================================
_info_debug_printed = False

def _parse_ns3_info(info):
    try:
        if info is None:
            return None
        if isinstance(info, dict):
            return {k: info.get(k, 0.0) for k in [
                'dl_rate','step','episode','phase','deadlines_met','total_steps',
                'serving_gnb','avg_gnb_idx','handover','avg_delay_ms','avg_bh_ms',
                'avg_air_ms','avg_comp_ms','avg_queue_ms','home_mec_pct',
                'cross_zone_pct','handover_count','handover_rate','act_local',
                'act_mec0','act_mec1','act_mec2','act_rsu0','act_rsu1','act_sat',
                'avg_gnb_load','avg_mec_load','avg_energy_j','throughput_tps',
                'jains_fairness','converged']}
        if isinstance(info, str):
            s = info.strip()
            if s.startswith('{'):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    pass
            result = {}
            patterns = {
                'dl_rate': r'"dl_rate"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_delay_ms': r'"avg_delay_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_energy_j': r'"avg_energy_j"\s*:\s*([0-9]+\.?[0-9]*)',
                'jains_fairness': r'"jains_fairness"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_local': r'"act_local"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_mec0': r'"act_mec0"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_mec1': r'"act_mec1"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_mec2': r'"act_mec2"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_rsu0': r'"act_rsu0"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_rsu1': r'"act_rsu1"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_sat': r'"act_sat"\s*:\s*([0-9]+\.?[0-9]*)',
            }
            for key, pattern in patterns.items():
                m = re.search(pattern, s)
                result[key] = float(m.group(1)) if m else 0.0
            return result
        if isinstance(info, (int, float)):
            return {'dl_rate': float(info)}
        return None
    except Exception as e:
        print(f"⚠️ Error parsing ns-3 info: {e}")
        return None

def _parse_ns3_dl_rate(info):
    parsed = _parse_ns3_info(info)
    if parsed is None:
        return None, None
    return parsed.get('dl_rate', None), parsed.get('step', None)

# =============================================================================
# TIMEOUT / SAFE CLOSE
# =============================================================================
class StepTimeout(Exception): pass
class EnvCloseTimeout(Exception): pass

def _timeout_handler(s, f): raise StepTimeout()
def _close_timeout_handler(s, f): raise EnvCloseTimeout()

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
    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ("accomplished","closed","timeout")):
            return None, -2.0, True, None
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

def safe_reset(env):
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        if obs is None:
            return np.zeros(OBS_SIZE, dtype=np.float32)
        return sanitize_obs(obs)
    except Exception:
        return np.zeros(OBS_SIZE, dtype=np.float32)

# =============================================================================
# GET DL STATS FROM CSV
# =============================================================================
def get_dl_stats_from_csv(seed, csv_dir):
    train_csv_path = os.path.join(csv_dir, f"seed_{seed}", f"ppo_seed{seed}.csv")
    if not os.path.exists(train_csv_path):
        return {'mean_dl': 0.0, 'best_dl': 0.0, 'std_dl': 0.0}
    try:
        train_df = pd.read_csv(train_csv_path)
        best_dl = train_df['dl_rate'].max()
        eval_df = train_df.tail(100)
        return {'mean_dl': eval_df['dl_rate'].mean(),
                'best_dl': best_dl,
                'std_dl': eval_df['dl_rate'].std()}
    except Exception as e:
        print(f"Error reading CSV for seed {seed}: {e}")
        return {'mean_dl': 0.0, 'best_dl': 0.0, 'std_dl': 0.0}

# =============================================================================
# PPO NETWORKS
# =============================================================================
class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBS_SIZE, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
    def forward(self, x): return self.network(x)

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBS_SIZE, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.network(x)

# =============================================================================
# PPO BUFFER
# =============================================================================
class PPOBuffer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.states = []; self.actions = []; self.rewards = []
        self.dones = []; self.log_probs = []; self.values = []
        self.advantages = []; self.returns = []
    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(sanitize_obs(state))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
    def __len__(self): return len(self.states)
    def get_tensors(self):
        return {
            'states': torch.tensor(np.stack(self.states), dtype=torch.float32).to(DEVICE),
            'actions': torch.tensor(self.actions, dtype=torch.long).to(DEVICE),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32).to(DEVICE),
            'values': torch.tensor(self.values, dtype=torch.float32).to(DEVICE),
            'advantages': torch.tensor(self.advantages, dtype=torch.float32).to(DEVICE),
            'returns': torch.tensor(self.returns, dtype=torch.float32).to(DEVICE),
        }

# =============================================================================
# PPO AGENT (v8.6.29)
# =============================================================================
class PPOAgent:
    def __init__(self):
        self.actor = ActorNetwork().to(DEVICE)
        self.critic = CriticNetwork().to(DEVICE)
        self.target_critic = CriticNetwork().to(DEVICE)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_CONFIG['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_CONFIG['critic_lr'])

        self.actor_scheduler = optim.lr_scheduler.LinearLR(
            self.actor_optimizer, start_factor=0.1,
            total_iters=PPO_CONFIG['lr_warmup_steps'])
        self.critic_scheduler = optim.lr_scheduler.LinearLR(
            self.critic_optimizer, start_factor=0.1,
            total_iters=PPO_CONFIG['lr_warmup_steps'])

        self.buffer = PPOBuffer()
        self.reward_shaper = RewardShaper()

        self.epsilon = 1.0
        self.global_step = 0
        self.total_updates = 0
        self.update_counter = 0

        self.prev_actor_loss = float('inf')
        self.prev_critic_loss = float('inf')
        self.lr_reduction_counter = 0

        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"  Actor Network:   {actor_params:,} parameters")
        print(f"  Critic Network:  {critic_params:,} parameters")
        print(f"  Total:           {actor_params + critic_params:,} parameters")

    def select_action(self, state):
        state = sanitize_obs(state)
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.actor(t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(t).squeeze(0)
        return int(action.item()), log_prob.item(), value.item()

    def remember(self, state, action, reward, done, log_prob, value):
        self.buffer.push(state, action, reward, done, log_prob, value)

    def compute_gae(self, next_value):
        gamma = PPO_CONFIG['gamma']
        gae_lambda = PPO_CONFIG['gae_lambda']
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        returns = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = 0.0 if dones[t] else next_value
            else:
                next_v = values[t + 1] if not dones[t] else 0.0
            delta = rewards[t] + gamma * next_v - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.append(gae)
            returns.append(gae + values[t])

        advantages = list(reversed(advantages))
        returns = list(reversed(returns))
        advantages = np.array(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        self.buffer.returns = returns

    def soft_update_target(self):
        tau = PPO_CONFIG['tau']
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def update(self, next_value):
        buffer_size = len(self.buffer)
        if buffer_size < 64:
            return None

        self.compute_gae(next_value)
        data = self.buffer.get_tensors()
        states, actions = data['states'], data['actions']
        old_log_probs = data['log_probs']
        advantages, returns = data['advantages'], data['returns']

        if self.global_step < PPO_CONFIG['lr_warmup_steps']:
            f = max(0.1, self.global_step / PPO_CONFIG['lr_warmup_steps'])
            for pg in self.actor_optimizer.param_groups:
                pg['lr'] = PPO_CONFIG['actor_lr'] * f
            for pg in self.critic_optimizer.param_groups:
                pg['lr'] = PPO_CONFIG['critic_lr'] * f

        indices = np.arange(buffer_size)
        np.random.shuffle(indices)

        total_a_loss = total_c_loss = total_ent = 0.0
        num_updates = 0
        kl_violations = 0

        epochs = PPO_CONFIG['epochs_per_update']
        mb = PPO_CONFIG['minibatch_size']
        clip = PPO_CONFIG['clip_epsilon']
        ent_coef = PPO_CONFIG['entropy_coef']
        max_gn = PPO_CONFIG['max_grad_norm']
        target_kl = PPO_CONFIG['target_kl']
        kl_pat = PPO_CONFIG['kl_early_stop_patience']

        for epoch in range(epochs):
            epoch_kl = 0.0; epoch_kl_n = 0
            for start in range(0, buffer_size, mb):
                end = min(start + mb, buffer_size)
                idx = indices[start:end]
                bs = states[idx]; ba = actions[idx]
                bolp = old_log_probs[idx]; badv = advantages[idx]; bret = returns[idx]

                logits = self.actor(bs)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                clp = dist.log_prob(ba)
                ent = dist.entropy().mean()

                ratio = torch.exp(clp - bolp)
                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * badv
                a_loss = -torch.min(surr1, surr2).mean() - ent_coef * ent

                cv = self.critic(bs).squeeze()
                cv = torch.clamp(cv, -5.0, 5.0)
                c_loss = F.mse_loss(cv, bret)

                self.actor_optimizer.zero_grad()
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_gn)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_gn)
                self.critic_optimizer.step()

                total_a_loss += a_loss.item()
                total_c_loss += c_loss.item()
                total_ent += ent.item()
                num_updates += 1

                with torch.no_grad():
                    epoch_kl += (bolp - clp).mean().item()
                    epoch_kl_n += 1

            if epoch_kl_n > 0:
                avg_kl = epoch_kl / epoch_kl_n
                if avg_kl > target_kl:
                    kl_violations += 1
                    if kl_violations > kl_pat:
                        print(f"  Early stopping at epoch {epoch} (KL: {avg_kl:.4f})")
                        break
                else:
                    kl_violations = max(0, kl_violations - 1)

        avg_a = total_a_loss / max(num_updates, 1)
        avg_c = total_c_loss / max(num_updates, 1)
        avg_e = total_ent / max(num_updates, 1)

        if PPO_CONFIG['adaptive_lr']:
            if (self.prev_actor_loss - avg_a) < PPO_CONFIG['loss_improvement_threshold'] and \
               (self.prev_critic_loss - avg_c) < PPO_CONFIG['loss_improvement_threshold']:
                self.lr_reduction_counter += 1
                if self.lr_reduction_counter > 10:
                    for pg in self.actor_optimizer.param_groups:
                        pg['lr'] = max(pg['lr'] * PPO_CONFIG['lr_reduction_factor'], PPO_CONFIG['lr_min'])
                    for pg in self.critic_optimizer.param_groups:
                        pg['lr'] = max(pg['lr'] * PPO_CONFIG['lr_reduction_factor'], PPO_CONFIG['lr_min'])
                    self.lr_reduction_counter = 0
            else:
                self.lr_reduction_counter = max(0, self.lr_reduction_counter - 1)
            self.prev_actor_loss = avg_a
            self.prev_critic_loss = avg_c

        self.buffer.reset()
        self.global_step += 1
        self.total_updates += 1
        self.update_counter += 1

        if PPO_CONFIG['lr_decay']:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        if self.update_counter % PPO_CONFIG['update_target_every'] == 0:
            self.soft_update_target()

        return avg_a, avg_c, avg_e

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'epsilon': self.epsilon,
            'global_step': self.global_step,
            'total_updates': self.total_updates,
            'architecture': 'PPO_v8.6.29_shaped',
            'reward_weights': self.reward_shaper.weights,
        }, path)
        print(f"  Model saved → {path}")

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ck['actor'])
        self.critic.load_state_dict(ck['critic'])
        if 'target_critic' in ck:
            self.target_critic.load_state_dict(ck['target_critic'])
        self.actor_optimizer.load_state_dict(ck['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ck['critic_optimizer'])
        self.epsilon = ck.get('epsilon', 1.0)
        self.global_step = ck.get('global_step', 0)
        self.total_updates = ck.get('total_updates', 0)
        print(f"  Model loaded from {path}")

# =============================================================================
# TRAIN ONE SEED
# =============================================================================
def train_one_seed(seed):
    set_seed(seed)

    print(f"\n{'='*85}")
    print(f"  PPO — SEED {seed}  (v8.6.29 SHAPED REWARD)")
    print(f"{'='*85}\n")

    BASE_PORT = 15000
    port = find_free_port(BASE_PORT + seed)

    kill_zombie_ns3()
    kill_process_on_port(port)
    time.sleep(5)
    wait_port_free(port, timeout=15)

    ns3_proc = launch_ns3(seed, port)
    env = None

    try:
        for attempt in range(10):
            try:
                env = ns3env.Ns3Env(port=port, startSim=False, simSeed=seed)
                print(f"  Connected to ns-3 on port {port}  (attempt {attempt+1})")
                break
            except Exception as e:
                print(f"Connection attempt {attempt+1}/10 failed: {e}")
                time.sleep(4)
        if env is None:
            raise RuntimeError(f"Failed to connect to ns-3 (seed={seed})")

        agent = PPOAgent()
        convergence_detector = ConvergenceDetector()

        alg_name = "ppo"
        train_dir = os.path.join(PPO_TRAIN_DIR, f"seed_{seed}")
        csv_dir = os.path.join(PPO_CSV_DIR, f"seed_{seed}")
        models_dir = os.path.join(PPO_MODELS_DIR, f"seed_{seed}")
        for d in [train_dir, csv_dir, models_dir]:
            os.makedirs(d, exist_ok=True)

        best_dl = 0.0
        best_path = os.path.join(models_dir, f"{alg_name}_seed{seed}_best.pt")
        final_path = os.path.join(models_dir, f"{alg_name}_seed{seed}_final.pt")
        csv_path = os.path.join(csv_dir, f"{alg_name}_seed{seed}.csv")

        episode_rewards = []
        episode_dl_rates = []
        episode_moving_avgs = []
        episode_moving_dls = []
        episode_fairness = []
        episode_converged = []
        all_episode_rewards = []
        all_episode_dl_rates = []

        # ★ v8.6.29: buffer accumulates across EPISODES_PER_UPDATE episodes
        agent.buffer.reset()
        episodes_since_update = 0

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "env_reward", "shaped_reward",
                "moving_avg", "dl_rate", "moving_dl", "best_dl",
                "actor_loss", "critic_loss", "entropy",
                "avg_delay_ms", "avg_bh_ms", "avg_air_ms", "avg_comp_ms", "avg_queue_ms",
                "avg_energy_j", "throughput_tps", "handover_rate",
                "home_mec_pct", "cross_zone_pct", "avg_gnb_load", "avg_mec_load",
                "act_local", "act_mec0", "act_mec1", "act_mec2",
                "act_rsu0", "act_rsu1", "act_sat",
                "avg_gnb_idx", "jains_fairness", "converged", "info_available",
            ])

            obs = safe_reset(env)
            obs = sanitize_obs(obs)

            for ep in range(NUM_EPISODES):
                agent.reward_shaper.reset_episode()
                total_env_reward = 0.0
                total_shaped_reward = 0.0
                steps = 0
                last_valid_dl = None
                simulator_ended = False
                info_available = False
                next_value = 0.0
                done = False

                while steps < MAX_STEPS:
                    action, log_prob, value = agent.select_action(obs)
                    result = step_with_timeout(env, action)

                    if result[0] is None:
                        simulator_ended = True
                        break

                    next_obs, env_reward, done, info = result
                    next_obs = sanitize_obs(next_obs)

                    shaped_reward = agent.reward_shaper.shape(env_reward, obs, action)
                    agent.remember(obs, action, shaped_reward, done, log_prob, value)

                    total_env_reward += env_reward
                    total_shaped_reward += shaped_reward

                    parsed_dl, step_num = _parse_ns3_dl_rate(info)
                    if parsed_dl is not None and (step_num is None or step_num > 0):
                        info_available = True
                        last_valid_dl = parsed_dl

                    obs = next_obs
                    steps += 1
                    if done:
                        with torch.no_grad():
                            st = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                            next_value = agent.critic(st).item()
                        break

                if not done and steps > 0:
                    with torch.no_grad():
                        st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        next_value = agent.critic(st).item()

                # ★ v8.6.29: accumulate episodes, update once every N
                episodes_since_update += 1
                actor_loss = critic_loss = entropy = 0.0

                if episodes_since_update >= EPISODES_PER_UPDATE:
                    if len(agent.buffer) >= 64:
                        update_result = agent.update(next_value)
                        if update_result:
                            actor_loss, critic_loss, entropy = update_result
                    episodes_since_update = 0
                    agent.buffer.reset()

                dl_rate = last_valid_dl if last_valid_dl is not None else 0.0
                fairness = compute_jains_fairness(obs)
                converged, stability_info = convergence_detector.update(dl_rate, ep)

                episode_rewards.append(total_env_reward)
                episode_dl_rates.append(dl_rate)
                episode_fairness.append(fairness)
                episode_converged.append(1 if converged else 0)
                all_episode_rewards.append(total_env_reward)
                all_episode_dl_rates.append(dl_rate)

                moving_avg = float(np.mean(all_episode_rewards[-20:])) if len(all_episode_rewards) >= 20 else float(np.mean(all_episode_rewards))
                moving_dl = float(np.mean(all_episode_dl_rates[-20:])) if len(all_episode_dl_rates) >= 20 else float(np.mean(all_episode_dl_rates))
                episode_moving_avgs.append(moving_avg)
                episode_moving_dls.append(moving_dl)

                if moving_dl > best_dl:
                    best_dl = moving_dl
                    agent.save(best_path)

                parsed_info = _parse_ns3_info(info) if info else {}

                writer.writerow([
                    ep,
                    round(total_env_reward, 4),
                    round(total_shaped_reward, 4),
                    round(moving_avg, 4),
                    round(dl_rate, 2),
                    round(moving_dl, 2),
                    round(best_dl, 2),
                    round(actor_loss, 5),
                    round(critic_loss, 5),
                    round(entropy, 5),
                    parsed_info.get('avg_delay_ms', 0.0),
                    parsed_info.get('avg_bh_ms', 0.0),
                    parsed_info.get('avg_air_ms', 0.0),
                    parsed_info.get('avg_comp_ms', 0.0),
                    parsed_info.get('avg_queue_ms', 0.0),
                    parsed_info.get('avg_energy_j', 0.0),
                    parsed_info.get('throughput_tps', 0.0),
                    parsed_info.get('handover_rate', 0.0),
                    parsed_info.get('home_mec_pct', 0.0),
                    parsed_info.get('cross_zone_pct', 0.0),
                    parsed_info.get('avg_gnb_load', 0.0),
                    parsed_info.get('avg_mec_load', 0.0),
                    parsed_info.get('act_local', 0.0),
                    parsed_info.get('act_mec0', 0.0),
                    parsed_info.get('act_mec1', 0.0),
                    parsed_info.get('act_mec2', 0.0),
                    parsed_info.get('act_rsu0', 0.0),
                    parsed_info.get('act_rsu1', 0.0),
                    parsed_info.get('act_sat', 0.0),
                    parsed_info.get('avg_gnb_idx', 0),
                    round(fairness, 4),
                    1 if converged else 0,
                    int(info_available)
                ])
                f.flush()

                log_freq = 10 if ep < 50 else 50
                if ep % log_freq == 0 or ep >= NUM_EPISODES - 10:
                    status = "🎯CONVERGED" if converged else f"({stability_info})"
                    print(
                        f"Seed {seed} | EP {ep:4d} | "
                        f"EnvR {total_env_reward:8.2f} | "
                        f"ShapedR {total_shaped_reward:7.2f} | "
                        f"DL {dl_rate:6.1f}% | "
                        f"DL20 {moving_dl:6.1f}% | "
                        f"BestDL {best_dl:6.1f}% | "
                        f"Fair {fairness:.3f} | "
                        f"{status}"
                    )

                if simulator_ended:
                    print(f"Simulator terminated.")
                    break

        agent.save(final_path)

        np.save(os.path.join(train_dir, f"rewards_seed{seed}.npy"), np.array(episode_rewards, dtype=np.float32))
        np.save(os.path.join(train_dir, f"dl_rates_seed{seed}.npy"), np.array(episode_dl_rates, dtype=np.float32))
        np.save(os.path.join(train_dir, f"moving_reward_seed{seed}.npy"), np.array(episode_moving_avgs, dtype=np.float32))
        np.save(os.path.join(train_dir, f"moving_dl_seed{seed}.npy"), np.array(episode_moving_dls, dtype=np.float32))
        np.save(os.path.join(train_dir, f"fairness_seed{seed}.npy"), np.array(episode_fairness, dtype=np.float32))
        np.save(os.path.join(train_dir, f"converged_seed{seed}.npy"), np.array(episode_converged, dtype=np.int32))

        conv_stats = convergence_detector.get_stats()
        if conv_stats['converged']:
            dl_stats = get_dl_stats_from_csv(seed, PPO_CSV_DIR)
            print(f"\n🎯 SEED {seed} CONVERGED at episode {conv_stats['convergence_episode']}")
            print(f"   Mean DL%: {conv_stats['mean_dl']:.2f}%, StdDev: {conv_stats['stddev_dl']:.2f}%")
            print(f"   Best DL%: {dl_stats['best_dl']:.2f}%")
        else:
            print(f"\n⚠️ SEED {seed} did not converge within {NUM_EPISODES} episodes")

    finally:
        print(f"\nCleaning up seed {seed} ...")
        if env is not None:
            close_env_safe(env, timeout_sec=5)
        kill_ns3(ns3_proc)
        kill_process_on_port(port)
        time.sleep(5)
        wait_port_free(port, timeout=20)
        time.sleep(5)
        print(f"\nCleaning finished for seed {seed}\n")

    print(f"Seed {seed} complete.\n")
    return episode_rewards, episode_dl_rates, episode_fairness, episode_converged

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 85)
    print("  PPO — MULTI-SEED TRAINING (v8.6.29 SHAPED REWARD)")
    print(f"  Seeds: {MULTI_SEEDS}   Episodes: {NUM_EPISODES}   Steps: {MAX_STEPS}")
    print(f"  Episodes per PPO update: {EPISODES_PER_UPDATE}")
    print(f"  Simulator: mec_simulator_v8_robust_fixed")
    print(f"  OBS_SIZE={OBS_SIZE}   NUM_ACTIONS={NUM_ACTIONS}")
    print(f"  Actor LR: {PPO_CONFIG['actor_lr']}   Critic LR: {PPO_CONFIG['critic_lr']}")
    print(f"  Entropy: {PPO_CONFIG['entropy_coef']}   Epochs/update: {PPO_CONFIG['epochs_per_update']}")
    print(f"  Target KL: {PPO_CONFIG['target_kl']} (patience={PPO_CONFIG['kl_early_stop_patience']})")
    print(f"  Reward: SHAPED (deadline 0.60, delay 0.15, energy 0.10, load 0.10,")
    print(f"          penalty 0.05, backhaul 0.05), clipped to [-1, +1]")
    print("=" * 85 + "\n")

    all_rewards = {}
    all_dl_rates = {}
    all_fairness = {}
    all_converged = {}

    for seed in MULTI_SEEDS:
        if seed in EXCLUDE_SEEDS:
            continue
        print(f"\n{'='*50} SEED {seed} {'='*50}\n")
        try:
            rewards, dl_rates, fairness, converged = train_one_seed(seed)
            all_rewards[seed] = rewards
            all_dl_rates[seed] = dl_rates
            all_fairness[seed] = fairness
            all_converged[seed] = converged
            print(f"\nFINISHED SEED {seed}\n")
        except Exception as e:
            print(f"\nSeed {seed} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_rewards:
        print("No seeds completed.")
        sys.exit(1)

    completed_r = [r for r in all_rewards.values() if len(r) == NUM_EPISODES]
    completed_dl = [dl for dl in all_dl_rates.values() if len(dl) == NUM_EPISODES]
    completed_fair = [f for f in all_fairness.values() if len(f) == NUM_EPISODES]

    if completed_r:
        np.save(os.path.join(RESULTS_DIR, "ppo_rewards_thesis_aligned.npy"),
                np.stack(completed_r).mean(axis=0))
    if completed_dl:
        np.save(os.path.join(RESULTS_DIR, "ppo_dl_rates_thesis_aligned.npy"),
                np.stack(completed_dl).mean(axis=0))
    if completed_fair:
        np.save(os.path.join(RESULTS_DIR, "ppo_fairness_thesis_aligned.npy"),
                np.stack(completed_fair).mean(axis=0))

    print("\n" + "=" * 85)
    print("PPO — MULTI-SEED SUMMARY (v8.6.29 SHAPED REWARD)")
    print("=" * 85)

    convergence_summary = []
    for seed in MULTI_SEEDS:
        if seed in EXCLUDE_SEEDS or seed not in all_rewards:
            print(f"Seed {seed:2d} | FAILED")
            continue

        dl_stats = get_dl_stats_from_csv(seed, PPO_CSV_DIR)
        best_dl = dl_stats['best_dl']
        mean_dl = dl_stats['mean_dl']

        mean_last50 = np.mean(all_rewards[seed][-50:]) if len(all_rewards[seed]) >= 50 else np.mean(all_rewards[seed])
        best_r = max(all_rewards[seed])
        mean_fairness = np.mean(all_fairness[seed][-50:]) if seed in all_fairness else 0.0
        converged = np.any(all_converged[seed]) if seed in all_converged else False

        conv_ep = -1
        if converged:
            conv_indices = np.where(np.array(all_converged[seed]) == 1)[0]
            if len(conv_indices) > 0:
                conv_ep = conv_indices[0]

        print(f"Seed {seed:2d} | Best Reward: {best_r:8.2f} | Mean L50: {mean_last50:8.2f} | "
              f"Best DL: {best_dl:6.2f}% | Mean DL: {mean_dl:6.2f}% | "
              f"Fairness: {mean_fairness:.3f} | "
              f"Converged: {'✅' if converged else '❌'}"
              + (f" at ep {conv_ep}" if converged else ""))

        convergence_summary.append({
            'seed': seed, 'converged': converged,
            'convergence_episode': conv_ep,
            'best_dl': best_dl, 'mean_dl': mean_dl,
            'mean_fairness': mean_fairness,
        })

    print("\n" + "=" * 85)
    print("DL% SUMMARY TABLE")
    print("=" * 85)
    print("Seed | Best DL% | Mean DL% (Eval) | Fairness | Converged")
    print("-" * 60)
    for s in convergence_summary:
        print(f"{s['seed']:4d} | {s['best_dl']:8.2f}% | "
              f"{s['mean_dl']:14.2f}% | {s['mean_fairness']:8.3f} | "
              f"{'✅' if s['converged'] else '❌'}")

    print("=" * 85)
    print("\nPPO training complete! 🚀")