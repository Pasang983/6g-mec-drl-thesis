# double_dqn_multiseed.py
"""
Double DQN v18 - Multi-Seed Training with gNB/Backhaul Awareness
Master's Thesis: Task Offloading Optimization in 6G MEC/VEC/Satellite
Author : Pasang Lama — IST Lisboa

Patch v8.6.17.1 (Python-only):
  - step_with_timeout now retries and returns TRANSIENT_FAIL sentinel
  - MIN_REPLAY_SIZE lowered 20000 -> 5000 (learning starts ~ep 25)
  - Cyclic exploration disabled; linear epsilon decay over 300 eps
  - LR scheduler T_max matches actual scheduler.step() calls
  - Best-model selection on 50-ep window + rolling snapshots
  - Jain's fairness matches C++ (5 servers); prefer C++ value
  - Debug print guard fixed
"""

from env_config import *

import numpy as np
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
import pandas as pd

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
import gymnasium as gym
warnings.filterwarnings("ignore", message="Overriding environment")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
sys.modules['gym'] = gym

from ns3gym import ns3env

# ==============================
# CONSTANTS
# ==============================
kGnbHomeMec = [0, 0, 1, 1, 2, 2]
BASE_PORT = 15000                              # ★ PATCHED (P2-9): single source of truth

# ==============================
# CONVERGENCE DETECTION CONSTANTS
# ==============================
CONVERGENCE_WINDOW = 50
CONVERGENCE_STDDEV_THRESHOLD = 4.0
CONVERGENCE_MIN_EPISODES = 500

# ==============================
# HYPERPARAMETERS (patched)
# ==============================
GAMMA = 0.99
LR_START = 3.0e-4
LR_END = 5.0e-5
BATCH_SIZE = 256
MEMORY_SIZE = 200000
MIN_REPLAY_SIZE = 5000                         # ★ PATCHED (P0-2): was 20000
TAU = 0.01
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 2000
MAX_GRAD_NORM = 1.0
REWARD_LOW = -10.0
REWARD_HIGH = 10.0
USE_PRIORITIZED_REPLAY = False
USE_CYCLIC_EXPLORATION = False                 # ★ PATCHED (P0-3): was True
EPS_LINEAR_DECAY_EPISODES = 300                # ★ PATCHED (P0-3): eps 1.0->0.01 over 300 eps

# ==============================
# DEVICE
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================
# SEED
# ==============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if TORCH_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==============================
# PSUTIL — optional
# ==============================
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    print("WARNING: psutil not installed — zombie cleanup disabled")

# ==============================
# PORT / PROCESS UTILITIES
# ==============================
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
                    print(f"  Killed PID={conn.pid} on port {port}")
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
                    print(f"  Killed zombie ns-3 PID={proc.pid}")
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
    except Exception as e:
        print(e)

# ==============================
# NS-3 LAUNCHER
# ==============================
def launch_ns3(seed, port):
    cmd = [
        f"{NS3_PATH}/ns3", "run",
        f"mec_simulator_v8_robust_fixed --seed={seed} --port={port} --evaluate=0"
    ]
    print(f"Launching ns-3 v8_robust_fixed (TRAINING) for seed {seed} on port {port}...")

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

    print(f"  ns-3 PID={proc.pid}")
    time.sleep(8)

    if proc.poll() is not None:
        raise RuntimeError(f"ns-3 crashed immediately (seed={seed})")
    return proc

# ==============================
# HELPERS
# ==============================
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
    """Decode one-hot UE type from indices 2, 3, 4"""
    mobile = safe_index(obs, 2)
    iot = safe_index(obs, 3)
    vehicle = safe_index(obs, 4)

    total = mobile + iot + vehicle
    if total < 0.1:
        return 0
    if mobile > iot and mobile > vehicle:
        return 0
    elif iot > vehicle:
        return 1
    else:
        return 2

def estimate_delay(obs, action):
    ratio_idx = 12 + action
    delay_ratio = safe_index(obs, ratio_idx) * 3.0
    deadline_s = safe_index(obs, 5) * 1.0
    return float(delay_ratio * deadline_s)

def get_position_indices(action):
    if action == 0:
        return 19, 20
    elif 1 <= action <= 3:
        idx = 19 + (action - 1) * 2
        return idx, idx + 1
    elif 4 <= action <= 5:
        idx = 25 + (action - 4) * 2
        return idx, idx + 1
    else:
        return 19, 20

def get_server_load(obs, action):
    if action == 0:
        return 0.0
    elif 1 <= action <= 3:
        return safe_index(obs, 7 + (action - 1))
    elif 4 <= action <= 5:
        return safe_index(obs, 10 + (action - 4))
    else:
        return 0.0

def get_backhaul_delay(obs, action):
    if action == 0:
        return 0.0
    elif 1 <= action <= 3:
        return safe_index(obs, 40 + (action - 1))
    elif 4 <= action <= 5:
        return 0.0
    elif action == 6:
        return safe_index(obs, 29) * 0.5
    return 0.0

def compute_jains_fairness(obs):
    """
    ★ PATCHED (P1-6): match C++ ComputeJainsFairnessIndex() which uses
    5 servers (3 MEC + 2 RSU with RSU loads scaled by 0.5).
    """
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

# ==============================
# PRIORITIZED REPLAY BUFFER
# ==============================
class PrioritizedReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done, td_error=None):
        self.buffer.append((
            sanitize_obs(state),
            int(action),
            float(reward),
            sanitize_obs(next_state),
            bool(done)
        ))
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = 1.0
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        self.beta = min(1.0, self.beta + self.beta_increment)

        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        weights = (1.0 / (len(self.buffer) * probs[indices])) ** self.beta
        weights = weights / weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.stack(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
            torch.tensor(np.stack(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(DEVICE),
            torch.tensor(weights, dtype=torch.float32).to(DEVICE),
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# ==============================
# STANDARD REPLAY BUFFER
# ==============================
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            sanitize_obs(state),
            int(action),
            float(reward),
            sanitize_obs(next_state),
            bool(done)
        ))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
            torch.tensor(np.stack(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

# ==============================
# Q-NETWORK with Dropout for Better Generalization
# ==============================
class QNetwork(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.net = nn.Sequential(
            nn.Linear(OBS_SIZE, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )

    def forward(self, x, training=True):
        return self.net(x)

# ==============================
# CONVERGENCE DETECTION CLASS
# ==============================
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
        self.best_dl = 0.0
        self.best_episode = -1

    def update(self, dl_rate, episode):
        self.history.append(dl_rate)

        if dl_rate > self.best_dl:
            self.best_dl = dl_rate
            self.best_episode = episode

        if len(self.history) < self.window or episode < self.min_episodes:
            return False, "insufficient_data"

        stddev_dl = np.std(self.history)

        if stddev_dl < self.stddev_threshold:
            self.consecutive_stable += 1
            if self.consecutive_stable >= 10:
                if not self.converged:
                    self.converged = True
                    self.convergence_episode = episode
                return True, f"converged (std={stddev_dl:.2f}%)"
        else:
            self.consecutive_stable = 0
            self.converged = False

        return self.converged, f"std={stddev_dl:.2f}%"

    def get_stats(self):
        if len(self.history) < self.window:
            return {
                'converged': False,
                'convergence_episode': -1,
                'window_size': len(self.history),
                'mean_dl': 0.0,
                'stddev_dl': 0.0,
                'best_dl': self.best_dl,
                'best_episode': self.best_episode,
                'stability': 'insufficient_data'
            }

        return {
            'converged': self.converged,
            'convergence_episode': self.convergence_episode,
            'window_size': self.window,
            'mean_dl': np.mean(self.history),
            'stddev_dl': np.std(self.history),
            'best_dl': self.best_dl,
            'best_episode': self.best_episode,
            'stability': 'converged' if self.converged else 'not_converged'
        }

# ==============================
# DOUBLE DQN AGENT
# ==============================
class DQNAgent:
    def __init__(self, use_prioritized=False):
        self.use_prioritized = use_prioritized
        self.epsilon = EPS_START
        self.epsilon_decay_steps = EPS_DECAY_STEPS
        self.steps_since_epsilon_update = 0

        self.policy_net = QNetwork(dropout_rate=0.1).to(DEVICE)
        self.target_net = QNetwork(dropout_rate=0.1).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LR_START,
            weight_decay=1e-5,
            eps=1e-8
        )

        # ★ PATCHED (P0-4): T_max matches actual scheduler.step() calls
        # (once per episode after warmup). Was NUM_EPISODES * MAX_STEPS,
        # which made LR stay near LR_START for the entire run.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, NUM_EPISODES - 50),
            eta_min=LR_END
        )

        if use_prioritized:
            self.memory = PrioritizedReplayBuffer()
        else:
            self.memory = ReplayBuffer()

        self.global_step = 0
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)

    def get_epsilon_cyclic(self, episode):
        """Legacy helper — only used if USE_CYCLIC_EXPLORATION is True."""
        if not USE_CYCLIC_EXPLORATION:
            return self.epsilon

        if episode < 50:
            return max(EPS_END, EPS_START * (0.99 ** (episode * 2)))

        cycle_length = 100
        cycle_pos = (episode - 50) % cycle_length
        cycle_frac = cycle_pos / cycle_length
        cycle_epsilon = EPS_END + (0.3 - EPS_END) * (1.0 - cycle_frac)
        return cycle_epsilon

    def get_lr(self, episode):
        if episode < 50:
            return LR_START * (episode / 50)
        elif episode < 200:
            return LR_START
        else:
            return self.scheduler.get_last_lr()[0]

    def select_action(self, state, episode=0):
        state = sanitize_obs(state)

        epsilon = self.get_epsilon_cyclic(episode) if USE_CYCLIC_EXPLORATION else self.epsilon

        if random.random() < epsilon:
            if random.random() < 0.6:
                loads = np.array([
                    max(float(state[i]), 1e-6) for i in range(7, 12)
                ])
                probs = 1.0 / loads
                probs /= probs.sum()
                candidates = [1, 2, 3, 4, 5]
                return int(np.random.choice(candidates, p=probs))
            return random.randint(0, NUM_ACTIONS - 1)

        self.policy_net.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = self.policy_net(t, training=False).argmax(dim=1).item()
        self.policy_net.train()
        return int(action)

    def update(self, episode=0):
        if len(self.memory) < max(BATCH_SIZE, MIN_REPLAY_SIZE):
            return None

        if episode < 50:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_lr(episode)

        if self.use_prioritized:
            batch = self.memory.sample(BATCH_SIZE)
            if batch is None:
                return None
            states, actions, rewards, next_states, dones, weights, indices = batch
            actions = actions.unsqueeze(1)
        else:
            batch = self.memory.sample(BATCH_SIZE)
            if batch is None:
                return None
            states, actions, rewards, next_states, dones = batch
            actions = actions.unsqueeze(1)
            weights = torch.ones_like(rewards).to(DEVICE)
            indices = None

        current_q = self.policy_net(states, training=True).gather(1, actions).squeeze(1)
        avg_q = current_q.mean().item()
        max_q = current_q.max().item()

        with torch.no_grad():
            next_actions = self.policy_net(next_states, training=False).argmax(1, keepdim=True)
            next_q = self.target_net(next_states, training=False).gather(1, next_actions).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)
            target_q = torch.clamp(target_q, REWARD_LOW, REWARD_HIGH)

        td_errors = target_q - current_q

        loss = (weights * nn.HuberLoss(delta=1.0)(current_q, target_q)).mean()

        if self.use_prioritized and indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        if episode >= 50:
            self.scheduler.step()

        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(TAU * pp.data + (1.0 - TAU) * tp.data)

        self.global_step += 1
        self.loss_history.append(float(loss.item()))
        self.q_value_history.append(avg_q)

        return float(loss.item()), avg_q, max_q

    def decay_epsilon(self, episode=0):
        """
        ★ PATCHED (P0-3): Linear decay over EPS_LINEAR_DECAY_EPISODES.
        Previous version multiplied by EPS_DECAY (0.9975/ep) which would
        take ~1840 episodes to reach EPS_END — far beyond our 600-ep budget.
        """
        if USE_CYCLIC_EXPLORATION:
            return
        frac = min(1.0, episode / float(EPS_LINEAR_DECAY_EPISODES))
        self.epsilon = EPS_START + (EPS_END - EPS_START) * frac

    def save(self, path):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epsilon": self.epsilon,
            "global_step": self.global_step,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint["policy"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epsilon = checkpoint.get("epsilon", EPS_START)
        self.global_step = checkpoint.get("global_step", 0)
        print(f"Model loaded → {path}")

# ==============================
# REWARD SHAPING (identity — matches C++)
# ==============================
def shape_reward(env_reward, obs, action):
    """
    v8.6.17: Return the raw environment reward unchanged.

    The simulator's ComputeReward already combines deadline (0.55),
    delay (0.10), energy (0.05), load (0.10), penalty (0.05), and
    backhaul (0.15) into a single scalar. DDQN trains on this scalar
    so its objective matches the evaluation metric used for the
    Random and Greedy baselines.

    ★ PATCHED (P2-8): The CSV still has both 'env_reward' and
    'shaped_reward' columns. They are IDENTICAL by design. If you
    re-add shaping in a future version, only 'shaped_reward' goes
    into the replay buffer; 'env_reward' remains the reporting metric.
    """
    return float(env_reward)

# ==============================
# NS-3 INFO PARSER
# ==============================
_info_debug_printed = False

def _parse_ns3_info(info):
    """Parse ALL metrics from ns-3 GetExtraInfo()"""
    global _info_debug_printed

    try:
        if info is None:
            return None

        if not _info_debug_printed:
            print(f"\n✅ ns3gym info received → type={type(info).__name__}")
            if isinstance(info, str) and len(info) < 500:
                print(f"   value={info[:200]}\n")
            _info_debug_printed = True

        if isinstance(info, dict):
            return {
                'dl_rate': float(info.get('dl_rate', 0.0)),
                'step': int(info.get('step', 0)),
                'episode': int(info.get('episode', 0)),
                'phase': info.get('phase', 'unknown'),
                'deadlines_met': int(info.get('deadlines_met', 0)),
                'total_steps': int(info.get('total_steps', 0)),
                'serving_gnb': int(info.get('serving_gnb', 0)),
                'avg_gnb_idx': float(info.get('avg_gnb_idx', 0.0)),
                'handover': int(info.get('handover', 0)),
                'avg_delay_ms': float(info.get('avg_delay_ms', 0.0)),
                'avg_bh_ms': float(info.get('avg_bh_ms', 0.0)),
                'avg_air_ms': float(info.get('avg_air_ms', 0.0)),
                'avg_comp_ms': float(info.get('avg_comp_ms', 0.0)),
                'avg_queue_ms': float(info.get('avg_queue_ms', 0.0)),
                'home_mec_pct': float(info.get('home_mec_pct', 0.0)),
                'cross_zone_pct': float(info.get('cross_zone_pct', 0.0)),
                'handover_count': int(info.get('handover_count', 0)),
                'handover_rate': float(info.get('handover_rate', 0.0)),
                'act_local': float(info.get('act_local', 0.0)),
                'act_mec0': float(info.get('act_mec0', 0.0)),
                'act_mec1': float(info.get('act_mec1', 0.0)),
                'act_mec2': float(info.get('act_mec2', 0.0)),
                'act_rsu0': float(info.get('act_rsu0', 0.0)),
                'act_rsu1': float(info.get('act_rsu1', 0.0)),
                'act_sat': float(info.get('act_sat', 0.0)),
                'avg_gnb_load': float(info.get('avg_gnb_load', 0.0)),
                'avg_mec_load': float(info.get('avg_mec_load', 0.0)),
                'avg_energy_j': float(info.get('avg_energy_j', 0.0)),
                'throughput_tps': float(info.get('throughput_tps', 0.0)),
                'jains_fairness': float(info.get('jains_fairness', 1.0)),
                'converged': int(info.get('converged', 0)),
            }

        if isinstance(info, str):
            s = info.strip()

            if s.startswith('{'):
                try:
                    d = json.loads(s)
                    dl_rate = float(d.get('dl_rate', 0.0))

                    # ★ PATCHED (P1-7): mark debug flag so this print fires once
                    if dl_rate > 0 and not _info_debug_printed:
                        print(f"    Parsed DL% from JSON: {dl_rate}%")
                        _info_debug_printed = True

                    return {
                        'dl_rate': dl_rate,
                        'step': int(d.get('step', 0)),
                        'episode': int(d.get('episode', 0)),
                        'phase': d.get('phase', 'unknown'),
                        'deadlines_met': int(d.get('deadlines_met', 0)),
                        'total_steps': int(d.get('total_steps', 0)),
                        'serving_gnb': int(d.get('serving_gnb', 0)),
                        'avg_gnb_idx': float(d.get('avg_gnb_idx', 0.0)),
                        'handover': int(d.get('handover', 0)),
                        'avg_delay_ms': float(d.get('avg_delay_ms', 0.0)),
                        'avg_bh_ms': float(d.get('avg_bh_ms', 0.0)),
                        'avg_air_ms': float(d.get('avg_air_ms', 0.0)),
                        'avg_comp_ms': float(d.get('avg_comp_ms', 0.0)),
                        'avg_queue_ms': float(d.get('avg_queue_ms', 0.0)),
                        'home_mec_pct': float(d.get('home_mec_pct', 0.0)),
                        'cross_zone_pct': float(d.get('cross_zone_pct', 0.0)),
                        'handover_count': int(d.get('handover_count', 0)),
                        'handover_rate': float(d.get('handover_rate', 0.0)),
                        'act_local': float(d.get('act_local', 0.0)),
                        'act_mec0': float(d.get('act_mec0', 0.0)),
                        'act_mec1': float(d.get('act_mec1', 0.0)),
                        'act_mec2': float(d.get('act_mec2', 0.0)),
                        'act_rsu0': float(d.get('act_rsu0', 0.0)),
                        'act_rsu1': float(d.get('act_rsu1', 0.0)),
                        'act_sat': float(d.get('act_sat', 0.0)),
                        'avg_gnb_load': float(d.get('avg_gnb_load', 0.0)),
                        'avg_mec_load': float(d.get('avg_mec_load', 0.0)),
                        'avg_energy_j': float(d.get('avg_energy_j', 0.0)),
                        'throughput_tps': float(d.get('throughput_tps', 0.0)),
                        'jains_fairness': float(d.get('jains_fairness', 1.0)),
                        'converged': int(d.get('converged', 0)),
                    }
                except json.JSONDecodeError:
                    pass

            # Regex fallback
            result = {
                'dl_rate': 0.0, 'step': 0, 'episode': 0, 'phase': 'unknown',
                'deadlines_met': 0, 'total_steps': 0, 'serving_gnb': 0,
                'avg_gnb_idx': 0.0, 'handover': 0, 'avg_delay_ms': 0.0,
                'avg_bh_ms': 0.0, 'avg_air_ms': 0.0, 'avg_comp_ms': 0.0,
                'avg_queue_ms': 0.0, 'home_mec_pct': 0.0, 'cross_zone_pct': 0.0,
                'handover_count': 0, 'handover_rate': 0.0, 'act_local': 0.0,
                'act_mec0': 0.0, 'act_mec1': 0.0, 'act_mec2': 0.0,
                'act_rsu0': 0.0, 'act_rsu1': 0.0, 'act_sat': 0.0,
                'avg_gnb_load': 0.0, 'avg_mec_load': 0.0, 'avg_energy_j': 0.0,
                'throughput_tps': 0.0, 'jains_fairness': 1.0, 'converged': 0,
            }

            patterns = {
                'dl_rate': r'"dl_rate"\s*:\s*([0-9]+\.?[0-9]*)',
                'step': r'"step"\s*:\s*([0-9]+)',
                'episode': r'"episode"\s*:\s*([0-9]+)',
                'phase': r'"phase"\s*:\s*"([^"]+)"',
                'deadlines_met': r'"deadlines_met"\s*:\s*([0-9]+)',
                'total_steps': r'"total_steps"\s*:\s*([0-9]+)',
                'serving_gnb': r'"serving_gnb"\s*:\s*([0-9]+)',
                'avg_gnb_idx': r'"avg_gnb_idx"\s*:\s*([0-9]+\.?[0-9]*)',
                'handover': r'"handover"\s*:\s*([0-9]+)',
                'avg_delay_ms': r'"avg_delay_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_bh_ms': r'"avg_bh_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_air_ms': r'"avg_air_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_comp_ms': r'"avg_comp_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_queue_ms': r'"avg_queue_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'home_mec_pct': r'"home_mec_pct"\s*:\s*([0-9]+\.?[0-9]*)',
                'cross_zone_pct': r'"cross_zone_pct"\s*:\s*([0-9]+\.?[0-9]*)',
                'handover_count': r'"handover_count"\s*:\s*([0-9]+)',
                'handover_rate': r'"handover_rate"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_local': r'"act_local"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_mec0': r'"act_mec0"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_mec1': r'"act_mec1"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_mec2': r'"act_mec2"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_rsu0': r'"act_rsu0"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_rsu1': r'"act_rsu1"\s*:\s*([0-9]+\.?[0-9]*)',
                'act_sat': r'"act_sat"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_gnb_load': r'"avg_gnb_load"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_mec_load': r'"avg_mec_load"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_energy_j': r'"avg_energy_j"\s*:\s*([0-9]+\.?[0-9]*)',
                'throughput_tps': r'"throughput_tps"\s*:\s*([0-9]+\.?[0-9]*)',
                'jains_fairness': r'"jains_fairness"\s*:\s*([0-9]+\.?[0-9]*)',
                'converged': r'"converged"\s*:\s*([0-9]+)',
            }

            for key, pattern in patterns.items():
                m = re.search(pattern, s)
                if m:
                    try:
                        val = m.group(1)
                        if key == 'phase':
                            result[key] = val
                        elif '.' in val or 'e' in val:
                            result[key] = float(val)
                        else:
                            result[key] = int(val)
                    except Exception:
                        result[key] = 0.0

            # ★ PATCHED (P1-7): mark debug flag so this print fires once
            if result['dl_rate'] > 0 and not _info_debug_printed:
                print(f"   ✅ Parsed DL% (regex): {result['dl_rate']}%")
                _info_debug_printed = True

            return result

        if isinstance(info, (int, float)):
            return {
                'dl_rate': float(info), 'step': 0, 'episode': 0, 'phase': 'unknown',
                'deadlines_met': 0, 'total_steps': 0, 'serving_gnb': 0,
                'avg_gnb_idx': 0.0, 'handover': 0, 'avg_delay_ms': 0.0,
                'avg_bh_ms': 0.0, 'avg_air_ms': 0.0, 'avg_comp_ms': 0.0,
                'avg_queue_ms': 0.0, 'home_mec_pct': 0.0, 'cross_zone_pct': 0.0,
                'handover_count': 0, 'handover_rate': 0.0, 'act_local': 0.0,
                'act_mec0': 0.0, 'act_mec1': 0.0, 'act_mec2': 0.0,
                'act_rsu0': 0.0, 'act_rsu1': 0.0, 'act_sat': 0.0,
                'avg_gnb_load': 0.0, 'avg_mec_load': 0.0, 'avg_energy_j': 0.0,
                'throughput_tps': 0.0, 'jains_fairness': 1.0, 'converged': 0,
            }

        return None

    except Exception as e:
        print(f"⚠️ Error parsing ns-3 info: {e}")
        return None

# ==============================
# BACKWARD COMPATIBILITY
# ==============================
def _parse_ns3_dl_rate(info):
    parsed = _parse_ns3_info(info)
    if parsed is None:
        return None, None
    return parsed.get('dl_rate', None), parsed.get('step', None)

# ==============================
# TIMEOUT HANDLERS
# ==============================
class StepTimeout(Exception):
    pass

class EnvCloseTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise StepTimeout()

def _close_timeout_handler(signum, frame):
    raise EnvCloseTimeout()

# ★ PATCHED (P0-1): retries + TRANSIENT_FAIL sentinel
def step_with_timeout(env, action, timeout_sec=STEP_TIMEOUT, retries=2):
    """
    Step the environment with timeout protection and retries.

    Returns:
      (obs, reward, done, info) on success
      ("TRANSIENT_FAIL", -1.0, False, None) if all retries exhausted
      (None, -2.0, True, None) on genuine simulator death
    """
    for attempt in range(retries + 1):
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
            if attempt < retries:
                print(f"  Step timeout, retry {attempt + 1}/{retries}")
                time.sleep(2)
                continue
            return "TRANSIENT_FAIL", -1.0, False, None
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ("cannot be accomplished", "closed", "timeout")):
                print("Simulator has terminated")
                return None, -2.0, True, None
            print(f"Step error: {e}")
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
        print("  env.close() completed")
    except EnvCloseTimeout:
        print(f"  env.close() timed out after {timeout_sec}s — skipping")
    except Exception as e:
        print(f"  env.close() error (ignored): {e}")
    finally:
        signal.alarm(0)

def safe_reset(env):
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        if obs is None:
            return np.zeros(OBS_SIZE, dtype=np.float32)
        return sanitize_obs(obs)
    except Exception as e:
        print(f"Reset error: {e}")
        return np.zeros(OBS_SIZE, dtype=np.float32)

# ==============================
# GET DL STATS FROM CSV
# ==============================
def get_dl_stats_from_csv(seed, csv_dir):
    train_csv_path = os.path.join(csv_dir, f"seed_{seed}", f"double_dqn_seed{seed}.csv")

    if not os.path.exists(train_csv_path):
        return {'mean_dl': 0.0, 'best_dl': 0.0, 'std_dl': 0.0}

    try:
        train_df = pd.read_csv(train_csv_path)
        best_dl = train_df['dl_rate'].max()
        eval_df = train_df.tail(100)
        return {
            'mean_dl': eval_df['dl_rate'].mean(),
            'best_dl': best_dl,
            'std_dl': eval_df['dl_rate'].std(),
        }
    except Exception as e:
        print(f"Error reading CSV for seed {seed}: {e}")
        return {'mean_dl': 0.0, 'best_dl': 0.0, 'std_dl': 0.0}

# ==============================
# TRAINING — ONE SEED
# ==============================
def train_one_seed(seed):
    set_seed(seed)

    global _info_debug_printed
    _info_debug_printed = False

    print(f"\n{'='*85}\nTRAINING SEED {seed} (Phase 3 - v8_robust_fixed)\n{'='*85}\n")

    port = BASE_PORT + seed

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
                print(f"Connected to ns-3 on port {port}")
                break
            except Exception as e:
                print(f"Connection attempt {attempt+1}/10 failed: {e}")
                time.sleep(4)

        if env is None:
            raise RuntimeError("Failed to connect to ns-3")

        agent = DQNAgent(use_prioritized=USE_PRIORITIZED_REPLAY)
        convergence_detector = ConvergenceDetector()

        train_dir = os.path.join(DDQN_TRAIN_DIR, f"seed_{seed}")
        csv_dir = os.path.join(DDQN_CSV_DIR, f"seed_{seed}")
        models_dir = os.path.join(MODELS_DIR, f"seed_{seed}")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        episode_rewards = []
        episode_dl_rates = []
        episode_moving_avgs = []
        episode_moving_dls = []
        episode_moving_dls_50 = []      # ★ PATCHED (P1-5): new 50-ep moving avg
        episode_fairness = []
        episode_converged = []
        episode_losses = []
        episode_q_values = []
        episode_epsilons = []
        all_episode_rewards = []
        all_episode_dl_rates = []

        best_dl = 0.0
        best_dl_actual = 0.0

        best_model_path = os.path.join(models_dir, f"double_dqn_seed{seed}_best.pt")
        final_model_path = os.path.join(models_dir, f"double_dqn_seed{seed}_final.pt")
        csv_path = os.path.join(csv_dir, f"double_dqn_seed{seed}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # ★ PATCHED (P1-5 / P2-8): added moving_dl_50 column;
            # 'shaped_reward' intentionally equals 'env_reward' in v8.6.17
            writer.writerow([
                "episode", "env_reward", "shaped_reward",
                "moving_avg", "dl_rate", "moving_dl", "moving_dl_50", "best_dl",
                "avg_loss", "avg_q", "epsilon",
                "avg_delay_ms", "avg_bh_ms", "avg_air_ms", "avg_comp_ms", "avg_queue_ms",
                "avg_energy_j", "throughput_tps", "handover_rate",
                "home_mec_pct", "cross_zone_pct", "avg_gnb_load", "avg_mec_load",
                "act_local", "act_mec0", "act_mec1", "act_mec2", "act_rsu0", "act_rsu1", "act_sat",
                "avg_gnb_idx", "jains_fairness", "converged", "info_available"
            ])

            obs = safe_reset(env)

            for ep in range(NUM_EPISODES):
                total_env_reward = 0.0
                total_shaped_reward = 0.0
                losses = []
                q_vals = []
                step_count = 0
                last_valid_dl = None
                simulator_ended = False
                info_available = False

                while step_count < MAX_STEPS:
                    action = agent.select_action(obs, episode=ep)
                    result = step_with_timeout(env, action)

                    # Genuine simulator death
                    if result[0] is None:
                        simulator_ended = True
                        break

                    # Transient failure — end episode only, not the run
                    # NOTE: isinstance(str), NOT ==, because on success
                    # result[0] is a NumPy array and `array == "string"`
                    # raises ValueError on truth-value coercion.
                    if isinstance(result[0], str) and result[0] == "TRANSIENT_FAIL":
                        print(f"  EP {ep}: transient step failure at step "
                              f"{step_count}, ending episode")
                        break

                    next_obs, reward, done, info = result

                    shaped_reward = shape_reward(reward, obs, action)
                    agent.memory.push(obs, action, shaped_reward, next_obs, done)

                    update_result = agent.update(episode=ep)
                    if update_result is not None:
                        loss, avg_q, max_q = update_result
                        losses.append(loss)
                        q_vals.append(avg_q)

                    total_env_reward += reward
                    total_shaped_reward += shaped_reward

                    parsed_dl, step_num = _parse_ns3_dl_rate(info)
                    if parsed_dl is not None and (step_num is None or step_num > 0):
                        info_available = True
                        last_valid_dl = parsed_dl

                    obs = next_obs
                    step_count += 1

                    if done:
                        break

                # === END OF EPISODE ===
                if not USE_CYCLIC_EXPLORATION:
                    agent.decay_epsilon(episode=ep)

                dl_rate = last_valid_dl if last_valid_dl is not None else 0.0
                avg_loss = float(np.mean(losses)) if losses else 0.0
                avg_q_ep = float(np.mean(q_vals)) if q_vals else 0.0

                # Parse info first so we can prefer C++ fairness
                parsed_info = _parse_ns3_info(info) if info else {}
                if parsed_info is None:
                    parsed_info = {}

                # ★ PATCHED (P1-6): prefer C++ Jain's fairness; fall back to local
                cpp_fairness = parsed_info.get('jains_fairness', None)
                if cpp_fairness is not None and cpp_fairness > 0:
                    fairness = float(cpp_fairness)
                else:
                    fairness = compute_jains_fairness(obs)

                converged, stability_info = convergence_detector.update(dl_rate, ep)

                episode_rewards.append(total_env_reward)
                episode_dl_rates.append(dl_rate)
                episode_fairness.append(fairness)
                episode_converged.append(1 if converged else 0)
                episode_losses.append(avg_loss)
                episode_q_values.append(avg_q_ep)
                episode_epsilons.append(agent.get_epsilon_cyclic(ep) if USE_CYCLIC_EXPLORATION else agent.epsilon)

                all_episode_rewards.append(total_env_reward)
                all_episode_dl_rates.append(dl_rate)

                moving_avg = float(np.mean(all_episode_rewards[-20:])) if len(all_episode_rewards) >= 20 else float(np.mean(all_episode_rewards))
                moving_dl = float(np.mean(all_episode_dl_rates[-20:])) if len(all_episode_dl_rates) >= 20 else float(np.mean(all_episode_dl_rates))

                # ★ PATCHED (P1-5): 50-ep moving avg used for best-model selection
                if len(all_episode_dl_rates) >= CONVERGENCE_WINDOW:
                    moving_dl_50 = float(np.mean(all_episode_dl_rates[-CONVERGENCE_WINDOW:]))
                else:
                    moving_dl_50 = moving_dl

                episode_moving_avgs.append(moving_avg)
                episode_moving_dls.append(moving_dl)
                episode_moving_dls_50.append(moving_dl_50)

                # ★ PATCHED (P1-5): best-model on 50-ep window
                if moving_dl_50 > best_dl:
                    best_dl = moving_dl_50
                    best_dl_actual = dl_rate
                    agent.save(best_model_path)
                    print(f"   💾 New best DL%(50): {best_dl:.2f}% | ep {ep} | instant {dl_rate:.2f}%")

                # ★ PATCHED (P1-5): rolling snapshots every 50 eps
                if ep > 0 and ep % 50 == 0:
                    snap = os.path.join(models_dir, f"double_dqn_seed{seed}_ep{ep}.pt")
                    agent.save(snap)

                current_epsilon = agent.get_epsilon_cyclic(ep) if USE_CYCLIC_EXPLORATION else agent.epsilon

                writer.writerow([
                    ep,
                    round(total_env_reward, 4),
                    round(total_shaped_reward, 4),
                    round(moving_avg, 4),
                    round(dl_rate, 2),
                    round(moving_dl, 2),
                    round(moving_dl_50, 2),         # ★ PATCHED (P1-5)
                    round(best_dl, 2),
                    round(avg_loss, 5),
                    round(avg_q_ep, 4),
                    round(current_epsilon, 4),
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

                if ep % 20 == 0 or ep >= NUM_EPISODES - 10:
                    status = "🎯CONVERGED" if converged else f"({stability_info})"
                    print(
                        f"Seed {seed} | EP {ep:4d} | "
                        f"EnvR {total_env_reward:8.2f} | "
                        f"MA20 {moving_avg:7.2f} | "
                        f"DL {dl_rate:6.1f}% | "
                        f"DL20 {moving_dl:6.1f}% | "
                        f"DL50 {moving_dl_50:6.1f}% | "     # ★ PATCHED (P1-5)
                        f"BestDL {best_dl:6.1f}% | "
                        f"Fair {fairness:.3f} | "
                        f"{status} | "
                        f"eps {agent.epsilon:.3f}"
                    )

                if simulator_ended:
                    print(f"Simulator terminated.")
                    break

        # Final save
        agent.save(final_model_path)

        # Save numpy arrays
        np.save(os.path.join(train_dir, f"rewards_seed{seed}.npy"),
                np.array(episode_rewards, dtype=np.float32))
        np.save(os.path.join(train_dir, f"dl_rates_seed{seed}.npy"),
                np.array(episode_dl_rates, dtype=np.float32))
        np.save(os.path.join(train_dir, f"moving_reward_seed{seed}.npy"),
                np.array(episode_moving_avgs, dtype=np.float32))
        np.save(os.path.join(train_dir, f"moving_dl_seed{seed}.npy"),
                np.array(episode_moving_dls, dtype=np.float32))
        np.save(os.path.join(train_dir, f"moving_dl_50_seed{seed}.npy"),  # ★ PATCHED (P1-5)
                np.array(episode_moving_dls_50, dtype=np.float32))
        np.save(os.path.join(train_dir, f"fairness_seed{seed}.npy"),
                np.array(episode_fairness, dtype=np.float32))
        np.save(os.path.join(train_dir, f"converged_seed{seed}.npy"),
                np.array(episode_converged, dtype=np.int32))
        np.save(os.path.join(train_dir, f"loss_seed{seed}.npy"),
                np.array(episode_losses, dtype=np.float32))
        np.save(os.path.join(train_dir, f"q_values_seed{seed}.npy"),
                np.array(episode_q_values, dtype=np.float32))
        np.save(os.path.join(train_dir, f"epsilon_seed{seed}.npy"),
                np.array(episode_epsilons, dtype=np.float32))

        conv_stats = convergence_detector.get_stats()
        if conv_stats['converged']:
            dl_stats = get_dl_stats_from_csv(seed, DDQN_CSV_DIR)
            print(f"\n🎯 SEED {seed} CONVERGED at episode {conv_stats['convergence_episode']}")
            print(f"   Mean DL%: {conv_stats['mean_dl']:.2f}%, StdDev: {conv_stats['stddev_dl']:.2f}%")
            print(f"   Best DL%: {dl_stats['best_dl']:.2f}%")
            print(f"   Best DL% (window): {conv_stats['best_dl']:.2f}% at episode {conv_stats['best_episode']}")
        else:
            print(f"\n⚠️ SEED {seed} did not converge within {NUM_EPISODES} episodes")
            if conv_stats['window_size'] >= CONVERGENCE_WINDOW:
                print(f"   Final Mean DL%: {conv_stats['mean_dl']:.2f}%, StdDev: {conv_stats['stddev_dl']:.2f}%")
                print(f"   Best DL% (window): {conv_stats['best_dl']:.2f}% at episode {conv_stats['best_episode']}")

    finally:
        print(f"\nCleaning up seed {seed} ...")
        if env is not None:
            close_env_safe(env, timeout_sec=5)
        kill_ns3(ns3_proc)
        kill_process_on_port(port)
        time.sleep(5)
        if wait_port_free(port, timeout=20):
            print(f"Port {port} released.")
        else:
            print(f"WARNING: Port {port} still busy.")
        time.sleep(5)
        print(f"\nCleaning finished for seed {seed}\n")

    print(f"Seed {seed} complete.\n")
    return episode_rewards, episode_dl_rates, episode_fairness, episode_converged

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 3: DDQN TRAINING (v8.6.17.1 — patched)")
    print(f"OBS_SIZE: {OBS_SIZE}, ACTIONS: {NUM_ACTIONS}")
    print(f"Convergence: Window={CONVERGENCE_WINDOW}, StdDev<{CONVERGENCE_STDDEV_THRESHOLD}%")
    print(f"Cyclic Exploration: {USE_CYCLIC_EXPLORATION}")
    print(f"Prioritized Replay: {USE_PRIORITIZED_REPLAY}")
    print(f"MIN_REPLAY_SIZE: {MIN_REPLAY_SIZE}")
    print(f"EPS decay: linear over {EPS_LINEAR_DECAY_EPISODES} episodes")
    print("="*80 + "\n")

    all_rewards = {}
    all_dl_rates = {}
    all_fairness = {}
    all_converged = {}

    for seed in MULTI_SEEDS:
        if seed in EXCLUDE_SEEDS:
            print(f"\nSkipping seed {seed} (in EXCLUDE_SEEDS)")
            continue
        print(f"\n{'='*50} SEED {seed} {'='*50}\n")

        port = BASE_PORT + seed

        print(f"Checking port {port}...")
        if is_port_in_use(port):
            print(f"⚠️ Port {port} is in use. Killing process...")
            kill_process_on_port(port)
            time.sleep(3)
            if is_port_in_use(port):
                print(f"❌ Port {port} still in use. Forcing kill...")
                subprocess.run(f"fuser -k {port}/tcp", shell=True)
                time.sleep(3)

        try:
            rewards, dl_rates, fairness, converged = train_one_seed(seed)
            all_rewards[seed] = rewards
            all_dl_rates[seed] = dl_rates
            all_fairness[seed] = fairness
            all_converged[seed] = converged
            print(f"\nFINISHED SEED {seed}\n")
        except Exception as e:
            print(f"\n Seed {seed} failed: {e}\n")
            import traceback
            traceback.print_exc()
            kill_process_on_port(port)
            time.sleep(5)
            continue

        print(f"Waiting 10 seconds before next seed...")
        time.sleep(10)

    if not all_rewards:
        print("No seeds completed successfully.")
        sys.exit(1)

    completed = [r for r in all_rewards.values() if len(r) == NUM_EPISODES]
    completed_dl = [d for d in all_dl_rates.values() if len(d) == NUM_EPISODES]
    completed_fair = [f for f in all_fairness.values() if len(f) == NUM_EPISODES]

    if completed:
        reward_matrix = np.stack(completed)
        np.save(os.path.join(RESULTS_DIR, "double_dqn_rewards_phase3.npy"),
                reward_matrix.mean(axis=0))
    else:
        print("WARNING: no seed completed all episodes — aggregate reward skipped")

    if completed_dl:
        dl_matrix = np.stack(completed_dl)
        np.save(os.path.join(RESULTS_DIR, "double_dqn_dl_rates_phase3.npy"),
                dl_matrix.mean(axis=0))
    else:
        print("WARNING: no seed completed all episodes — aggregate dl_rate skipped")

    if completed_fair:
        fairness_matrix = np.stack(completed_fair)
        np.save(os.path.join(RESULTS_DIR, "double_dqn_fairness_phase3.npy"),
                fairness_matrix.mean(axis=0))

    print("\n" + "="*80)
    print("DDQN MULTI-SEED TRAINING SUMMARY")
    print("="*80)

    convergence_summary = []
    for seed in MULTI_SEEDS:
        if seed in EXCLUDE_SEEDS:
            continue
        if seed not in all_rewards:
            print(f"Seed {seed:2d} | FAILED")
            continue

        mean_last50 = np.mean(all_rewards[seed][-50:])
        best_r = max(all_rewards[seed])

        dl_stats = get_dl_stats_from_csv(seed, DDQN_CSV_DIR)
        best_dl = dl_stats['best_dl']
        mean_dl = dl_stats['mean_dl']

        mean_fairness = np.mean(all_fairness[seed][-50:]) if seed in all_fairness else 0.0
        converged = np.any(all_converged[seed]) if seed in all_converged else False

        conv_ep = -1
        if converged:
            conv_indices = np.where(np.array(all_converged[seed]) == 1)[0]
            if len(conv_indices) > 0:
                conv_ep = conv_indices[0]

        print(
            f"Seed {seed:2d} | "
            f"Best Reward: {best_r:8.2f} | "
            f"Mean L50: {mean_last50:8.2f} | "
            f"Best DL: {best_dl:6.2f}% | "
            f"Mean DL: {mean_dl:6.2f}% | "
            f"Fairness: {mean_fairness:.3f} | "
            f"Converged: {'✅' if converged else '❌'}"
            + (f" at ep {conv_ep}" if converged else "")
        )

        convergence_summary.append({
            'seed': seed,
            'converged': converged,
            'convergence_episode': conv_ep,
            'best_dl': best_dl,
            'mean_dl': mean_dl,
            'mean_fairness': mean_fairness,
        })

    print("\n" + "="*80)
    print("CONVERGENCE SUMMARY")
    print("="*80)
    num_converged = sum(1 for s in convergence_summary if s['converged'])
    print(f"Seeds converged: {num_converged}/{len(MULTI_SEEDS)}")
    if num_converged > 0:
        conv_eps = [s['convergence_episode'] for s in convergence_summary if s['converged']]
        avg_conv_ep = np.mean(conv_eps)
        print(f"Average convergence episode: {avg_conv_ep:.1f}")

    print("\n" + "="*80)
    print("DL% SUMMARY TABLE")
    print("="*80)
    print("Seed | Best DL% | Mean DL% (Eval) | Fairness | Converged")
    print("-"*60)
    for s in convergence_summary:
        print(f"{s['seed']:4d} | {s['best_dl']:8.2f}% | {s['mean_dl']:14.2f}% | {s['mean_fairness']:8.3f} | {'✅' if s['converged'] else '❌'}")

    print("="*80)
    print("\nDDQN training finished! 🚀")