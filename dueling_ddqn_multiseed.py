# dueling_ddqn_multiseed.py
"""
Double Dueling DQN - Multi-Seed Training (Phase 3 - v8_6.27)
Master's Thesis: Task Offloading Optimization in 6G MEC/VEC/Satellite
Author : Pasang Lama — IST Lisboa

Architecture:
  Input (OBS_SIZE=48)
      │
  shared_net: Linear(48,256) LayerNorm ReLU → Linear(256,256) LayerNorm ReLU
      │
      ├─── value_stream:     Linear(256,128) ReLU → Linear(128,1)
      │                                              V(s)  — scalar
      │
      └─── advantage_stream: Linear(256,128) ReLU → Linear(128,7)
                                                      A(s,a) — per-action

  Aggregation (Wang et al. 2016, Eq. 9):
      Q(s,a) = V(s) + A(s,a) − (1/|A|) Σ_a' A(s,a')

UPDATES (v8.6.27):
- Identity reward shaping — trains on the C++ ComputeReward scalar
  so DDQN and Dueling DDQN are directly comparable.
- Linear epsilon decay over EPS_LINEAR_DECAY_EPISODES, matching
  double_dqn_multiseed.py.
- LR cosine annealing LR_START → LR_END.
- Best-model guard requires ≥100 episodes of history.
- Fairness averaged over per-step C++ values (5-server Jain's index).
- Uses mec_simulator_v8_robust_fixed --evaluate=0.
- Port collision handling via find_free_port().
"""

from env_config import *

# ==============================
# HYPERPARAMETERS (v8.6.27)
# Must be defined locally — env_config.py does not contain these.
# Values match double_dqn_multiseed.py for direct comparability.
# ==============================
GAMMA = 0.99
LR_START = 3.0e-4
LR_END = 5.0e-5
BATCH_SIZE = 256
MEMORY_SIZE = 200000
MIN_REPLAY_SIZE = 5000
TAU = 0.01
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 2000
MAX_GRAD_NORM = 1.0
REWARD_LOW = -10.0
REWARD_HIGH = 10.0
USE_PRIORITIZED_REPLAY = False
USE_CYCLIC_EXPLORATION = False
EPS_LINEAR_DECAY_EPISODES = 250

import numpy as np
import signal, csv, os, random, subprocess, sys, time, json, re, socket
import pandas as pd
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
import gymnasium as gym
warnings.filterwarnings("ignore", message="Overriding environment")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
sys.modules['gym'] = gym

from ns3gym import ns3env

# ============================================================
# CONSTANTS - UPDATED FOR 48 DIMS
# ============================================================
kGnbHomeMec = [0, 0, 1, 1, 2, 2]  # gNB 0-1 → MEC0, gNB 2-3 → MEC1, gNB 4-5 → MEC2

# ============================================================
# CONVERGENCE DETECTION CONSTANTS - RELAXED FOR DUELING DDQN
# ============================================================
CONVERGENCE_WINDOW = 50
CONVERGENCE_STDDEV_THRESHOLD = 5.0  # Relaxed from 2.0 for better convergence
CONVERGENCE_MIN_EPISODES = 50

# ============================================================
# DEVICE & CONFIG
# ============================================================
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
print(f"[Dueling DDQN] Training device: {DEVICE}")
print(f"[Dueling DDQN] OBS_SIZE={OBS_SIZE}  NUM_ACTIONS={NUM_ACTIONS}")

# Verify OBS_SIZE
assert OBS_SIZE == 48, f"OBS_SIZE should be 48, got {OBS_SIZE}"

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

def find_free_port(start_port, max_tries=50):
    """v8.6.27: bump the port until a free one is found."""
    for offset in range(max_tries):
        port = start_port + offset
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No free port near {start_port} after {max_tries} tries")

# ==============================
# NS-3 LAUNCHER - FIXED: Use v8_fixed
# ==============================
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

# =============================================================================
# OBSERVATION HELPERS - UPDATED FOR 48 DIMS
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
    """Decode one-hot UE type from indices 2, 3, 4"""
    if safe_index(obs, 2) > 0.5:  # MOBILE
        return 0
    elif safe_index(obs, 3) > 0.5:  # IOT
        return 1
    elif safe_index(obs, 4) > 0.5:  # VEHICLE
        return 2
    return 0  # Default to MOBILE

def get_position_indices(action):
    """Return (dx_idx, dy_idx) for a given action (48-dim v8 indices)."""
    if action == 0:  # LOCAL - use MEC0 position as fallback
        return 19, 20
    elif 1 <= action <= 3:  # MEC0-2
        idx = 19 + (action - 1) * 2
        return idx, idx + 1
    elif 4 <= action <= 5:  # RSU0-1
        idx = 25 + (action - 4) * 2
        return idx, idx + 1
    else:  # Satellite - use MEC0 as fallback
        return 19, 20

def get_server_load(obs, action):
    """Get load for a given action (48-dim v8 indices)."""
    if action == 0:  # LOCAL
        return 0.0
    elif 1 <= action <= 3:  # MEC0-2
        return safe_index(obs, 7 + (action - 1))  # indices 7-9
    elif 4 <= action <= 5:  # RSU0-1
        return safe_index(obs, 10 + (action - 4))  # indices 10-11
    else:  # Satellite
        return 0.0

def estimate_delay(obs, action):
    """
    Delay ratios at obs[12..18] (local=12, mec0=13, mec1=14, mec2=15, 
    rsu0=16, rsu1=17, sat=18). ratio stored as delay/deadline/3.
    deadline at obs[5] is normalised by MAX_DEADLINE=1.0.
    """
    ratio_idx = 12 + action
    delay_ratio = safe_index(obs, ratio_idx) * 3.0
    deadline_s = safe_index(obs, 5) * 1.0  # FIXED: was *6.0
    return float(delay_ratio * deadline_s)

def get_backhaul_delay(obs, action):
    """Get backhaul delay for a given action (48-dim v8 indices)."""
    if action == 0:  # LOCAL - no backhaul
        return 0.0
    elif 1 <= action <= 3:  # MEC0-2
        return safe_index(obs, 40 + (action - 1))  # indices 40-42
    elif 4 <= action <= 5:  # RSU0-1
        return 0.0
    elif action == 6:  # Satellite
        return safe_index(obs, 29) * 0.5  # sat_available * 0.5
    return 0.0

def compute_jains_fairness(obs):
    """
    v8.6.27: match C++ ComputeJainsFairnessIndex() — 5 servers
    (3 MEC + 2 RSU with RSU loads scaled by 0.5). Previously this
    used only the 3 MEC loads, which was inconsistent with the C++
    simulator and with double_dqn_multiseed.py.
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

# =============================================================================
# CONVERGENCE DETECTION CLASS
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
        
        mean_dl = np.mean(self.history)
        stddev_dl = np.std(self.history)
        
        if stddev_dl < self.stddev_threshold:
            self.consecutive_stable += 1
            if self.consecutive_stable >= 8:  # Reduced from 10
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
            return {
                'converged': False,
                'convergence_episode': -1,
                'window_size': len(self.history),
                'mean_dl': 0.0,
                'stddev_dl': 0.0,
                'stability': 'insufficient_data'
            }
        
        return {
            'converged': self.converged,
            'convergence_episode': self.convergence_episode,
            'window_size': self.window,
            'mean_dl': np.mean(self.history),
            'stddev_dl': np.std(self.history),
            'stability': 'converged' if self.converged else 'not_converged'
        }

# =============================================================================
# DUELING Q-NETWORK (48 -> 256 -> 256 -> [V:128->1 | A:128->7])
# =============================================================================
class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(OBS_SIZE, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_net(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q

    def get_advantage(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_net(x)
        return self.advantage_stream(shared)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_net(x)
        return self.value_stream(shared)

# =============================================================================
# REPLAY BUFFER
# =============================================================================
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

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = BATCH_SIZE
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

# =============================================================================
# DUELING DOUBLE DQN AGENT
# =============================================================================
class DuelingDQNAgent:
    def __init__(self):
        self.epsilon = EPS_START
        self.policy_net = DuelingQNetwork().to(DEVICE)
        self.target_net = DuelingQNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LR_START,
            weight_decay=1e-5,
            eps=1e-8
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, 300),
            eta_min=LR_END
        )

        self.memory = ReplayBuffer()
        self.global_step = 0

        n_params = sum(p.numel() for p in self.policy_net.parameters())
        print(f"  DuelingQNetwork: {n_params:,} parameters")

    def select_action(self, state):
        state = sanitize_obs(state)

        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        self.policy_net.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = self.policy_net(t).argmax(dim=1).item()
        self.policy_net.train()
        return int(a)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self, episode=0):
        if len(self.memory) < max(BATCH_SIZE, MIN_REPLAY_SIZE):
            return None

        batch = self.memory.sample(BATCH_SIZE)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch
        actions = actions.unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        avg_q = current_q.mean().item()
        max_q = current_q.max().item()

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_vals = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            # v8.6.27: use named constants from env_config.
            target_q = torch.clamp(
                rewards + GAMMA * next_q_vals * (1 - dones),
                min=REWARD_LOW, max=REWARD_HIGH
            )

        loss = nn.HuberLoss(delta=1.0)(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        # v8.6.27: step LR schedule only after warmup (matches double_dqn).
        if episode >= 50:
            self.scheduler.step()

        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(TAU * pp.data + (1.0 - TAU) * tp.data)

        self.global_step += 1
        return float(loss.item()), float(avg_q), float(max_q)

    def decay_epsilon(self, episode=0):
        """
        v8.6.27: linear epsilon decay over EPS_LINEAR_DECAY_EPISODES,
        matching double_dqn_multiseed.py. The previous multiplicative
        decay (EPS_DECAY=0.9975/ep) would take ~1840 episodes to reach
        EPS_END, far beyond the 600-episode budget.
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
            "epsilon": self.epsilon,
            "architecture": "DuelingDoubleDQN",
        }, path)
        print(f"  Model saved → {path}")

    def inspect_state(self, state):
        state = sanitize_obs(state)
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        self.policy_net.eval()
        with torch.no_grad():
            Q = self.policy_net(t).cpu().numpy().flatten()
            V = self.policy_net.get_value(t).cpu().item()
            A = self.policy_net.get_advantage(t).cpu().numpy().flatten()
        self.policy_net.train()
        print(f"  V(s)    = {V:.4f}")
        print(f"  A(s,a)  = {[f'{x:.3f}' for x in A]}")
        print(f"  Q(s,a)  = {[f'{x:.3f}' for x in Q]}")
        print(f"  Best a  = {int(np.argmax(Q))}  ({ACTION_NAMES[int(np.argmax(Q))]})")

# =============================================================================
# THESIS-ALIGNED REWARD SHAPING
# =============================================================================
def shape_reward(env_reward, obs, action):
    """
    v8.6.27: identity function — return the environment reward unchanged.

    The simulator's ComputeReward already combines deadline (0.55),
    delay (0.10), energy (0.05), load (0.10), penalty (0.05), and
    backhaul (0.15) into a single scalar. Training on this scalar
    keeps Dueling DDQN's objective consistent with Double DQN, Random,
    and Greedy — enabling direct architecture comparison.

    The previous version applied thesis-weighted shaping on top of the
    env reward, using reward ratios that did not match the C++
    ComputeReward. That made Dueling DDQN's reward incompatible with
    the baselines and with Double DQN.
    """
    return float(env_reward)

# =============================================================================
# NS-3 INFO PARSER (COMPREHENSIVE)
# =============================================================================
_info_debug_printed = False

def _parse_ns3_info(info):
    """Parse ALL metrics from ns-3 GetExtraInfo()"""
    try:
        if info is None:
            return None
        
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
            
            # Try JSON parsing first
            if s.startswith('{'):
                try:
                    d = json.loads(s)
                    return {
                        'dl_rate': float(d.get('dl_rate', 0.0)),
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
                'dl_rate': 0.0,
                'step': 0,
                'episode': 0,
                'phase': 'unknown',
                'deadlines_met': 0,
                'total_steps': 0,
                'serving_gnb': 0,
                'avg_gnb_idx': 0.0,
                'handover': 0,
                'avg_delay_ms': 0.0,
                'avg_bh_ms': 0.0,
                'avg_air_ms': 0.0,
                'avg_comp_ms': 0.0,
                'avg_queue_ms': 0.0,
                'home_mec_pct': 0.0,
                'cross_zone_pct': 0.0,
                'handover_count': 0,
                'handover_rate': 0.0,
                'act_local': 0.0,
                'act_mec0': 0.0,
                'act_mec1': 0.0,
                'act_mec2': 0.0,
                'act_rsu0': 0.0,
                'act_rsu1': 0.0,
                'act_sat': 0.0,
                'avg_gnb_load': 0.0,
                'avg_mec_load': 0.0,
                'avg_energy_j': 0.0,
                'throughput_tps': 0.0,
                'jains_fairness': 1.0,
                'converged': 0,
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
                    except:
                        result[key] = 0.0
            
            return result

        if isinstance(info, (int, float)):
            return {
                'dl_rate': float(info),
                'step': 0,
                'episode': 0,
                'phase': 'unknown',
                'deadlines_met': 0,
                'total_steps': 0,
                'serving_gnb': 0,
                'avg_gnb_idx': 0.0,
                'handover': 0,
                'avg_delay_ms': 0.0,
                'avg_bh_ms': 0.0,
                'avg_air_ms': 0.0,
                'avg_comp_ms': 0.0,
                'avg_queue_ms': 0.0,
                'home_mec_pct': 0.0,
                'cross_zone_pct': 0.0,
                'handover_count': 0,
                'handover_rate': 0.0,
                'act_local': 0.0,
                'act_mec0': 0.0,
                'act_mec1': 0.0,
                'act_mec2': 0.0,
                'act_rsu0': 0.0,
                'act_rsu1': 0.0,
                'act_sat': 0.0,
                'avg_gnb_load': 0.0,
                'avg_mec_load': 0.0,
                'avg_energy_j': 0.0,
                'throughput_tps': 0.0,
                'jains_fairness': 1.0,
                'converged': 0,
            }

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
class StepTimeout(Exception): 
    pass

class EnvCloseTimeout(Exception): 
    pass

def _timeout_handler(s, f): 
    raise StepTimeout()

def _close_timeout_handler(s,f): 
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
        print(f"Step timeout (>{timeout_sec}s)")
        return np.zeros(OBS_SIZE, dtype=np.float32), -1.0, True, None
    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ("accomplished","closed","timeout")):
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

# =============================================================================
# GET DL STATS FROM CSV - FIXED
# =============================================================================
def get_dl_stats_from_csv(seed, csv_dir):
    """Retrieve DL% statistics from CSV file."""
    train_csv_path = os.path.join(csv_dir, f"dueling_ddqn_seed{seed}.csv")  # FIXED: no seed_ dir

    if not os.path.exists(train_csv_path):
        return {'mean_dl': 0.0, 'best_dl': 0.0, 'std_dl': 0.0}

    try:
        train_df = pd.read_csv(train_csv_path)
        best_dl = train_df['dl_rate'].max()

        # Use last 100 training episodes for evaluation-like stats
        eval_df = train_df.tail(100)
        return {
            'mean_dl': eval_df['dl_rate'].mean(),
            'best_dl': best_dl,
            'std_dl': eval_df['dl_rate'].std(),
        }
    except Exception as e:
        print(f"Error reading CSV for seed {seed}: {e}")
        return {'mean_dl': 0.0, 'best_dl': 0.0, 'std_dl': 0.0}

# =============================================================================
# TRAIN ONE SEED
# =============================================================================
def train_one_seed(seed):
    set_seed(seed)
    global _info_debug_printed
    _info_debug_printed = False

    print(f"\n{'='*85}")
    print(f"  DUELING DOUBLE DQN — SEED {seed}  (Phase 3: v8_fixed simulator)")
    print(f"{'='*85}\n")
    BASE_PORT = 15000
    base_port = BASE_PORT + seed
    port = find_free_port(base_port)
    if port != base_port:
        print(f"  Port {base_port} busy → using {port}")

    kill_zombie_ns3()
    kill_process_on_port(port)
    time.sleep(5)
    wait_port_free(port, timeout=15)

    ns3_proc = launch_ns3(seed, port)
    env = None

    try:
        # Connect to ns-3
        for attempt in range(10):
            try:
                env = ns3env.Ns3Env(port=port, startSim=False, simSeed=seed)
                print(f"  Connected to ns-3 v8_fixed on port {port}  (attempt {attempt+1})")
                break
            except Exception as e:
                print(f"Connection attempt {attempt+1}/10 failed: {e}")
                time.sleep(4)
        if env is None:
            raise RuntimeError(f"Failed to connect to ns-3 v8_fixed (seed={seed})")

        # Initialize agent
        agent = DuelingDQNAgent()
        convergence_detector = ConvergenceDetector()

        # Reset shape_reward's persistent state
        if hasattr(shape_reward, 'last_action'):
            del shape_reward.last_action
            del shape_reward.consecutive_count

        # Use DUELING directories from env_config.py
        train_dir = DUELING_TRAIN_DIR
        csv_dir = DUELING_CSV_DIR
        models_dir = DUELING_MODELS_DIR
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        best_dl = 0.0
        best_dl_actual = 0.0
        best_path = os.path.join(models_dir, f"dueling_ddqn_seed{seed}_best.pt")
        final_path = os.path.join(models_dir, f"dueling_ddqn_seed{seed}_final.pt")
        csv_path = os.path.join(csv_dir, f"dueling_ddqn_seed{seed}.csv")  # FIXED: no seed_ dir

        episode_rewards = []
        episode_dl_rates = []
        all_episode_rewards = []
        all_episode_dl_rates = []
        episode_moving_avgs = []
        episode_moving_dls = []
        episode_fairness = []
        episode_converged = []
        episode_losses = []
        episode_q_values = []
        episode_epsilons = []

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "env_reward", "shaped_reward",
                "moving_avg", "dl_rate", "moving_dl", "best_dl",
                "avg_loss", "avg_q", "epsilon",
                "avg_delay_ms", "avg_bh_ms", "avg_air_ms", "avg_comp_ms", "avg_queue_ms",
                "avg_energy_j", "throughput_tps", "handover_rate",
                "home_mec_pct", "cross_zone_pct", "avg_gnb_load", "avg_mec_load",
                "act_local", "act_mec0", "act_mec1", "act_mec2", 
                "act_rsu0", "act_rsu1", "act_sat",
                "avg_gnb_idx", "jains_fairness", "converged", "info_available",
            ])

            obs = safe_reset(env)

            for ep in range(NUM_EPISODES):
                total_env_reward = 0.0
                total_shaped_reward = 0.0
                losses = []
                q_vals = []
                steps = 0
                last_valid_dl = None
                simulator_ended = False
                info_available = False
                episode_fairness_per_step = []
                _parsed_step = None   # v8.6.27: hoisted for reuse after the loop

                while steps < MAX_STEPS:
                    action = agent.select_action(obs)
                    result = step_with_timeout(env, action)

                    if result[0] is None:
                        simulator_ended = True
                        break

                    next_obs, env_reward, done, info = result
                    next_obs = sanitize_obs(next_obs)

                    shaped_reward = shape_reward(env_reward, obs, action)
                    agent.memory.push(obs, action, shaped_reward, next_obs, done)

                    update_result = agent.update(episode=ep)
                    if update_result:
                        loss, avg_q, max_q = update_result
                        losses.append(loss)
                        q_vals.append(avg_q)

                    total_env_reward += env_reward
                    total_shaped_reward += shaped_reward

                    # v8.6.27: parse info ONCE per step, reuse for DL and fairness.
                    _parsed_step = _parse_ns3_info(info) if info is not None else None
                    if _parsed_step is not None:
                        parsed_dl = _parsed_step.get('dl_rate')
                        step_num = _parsed_step.get('step')
                        if parsed_dl is not None and (step_num is None or step_num > 0):
                            info_available = True
                            last_valid_dl = parsed_dl
                        jf = _parsed_step.get('jains_fairness', 0.0)
                        if 0.0 < jf <= 1.0:
                            episode_fairness_per_step.append(jf)

                    obs = next_obs
                    steps += 1

                    if done:
                        break

                agent.decay_epsilon(episode=ep)

                dl_rate = last_valid_dl if last_valid_dl is not None else 0.0
                avg_loss = float(np.mean(losses)) if losses else 0.0
                avg_q_ep = float(np.mean(q_vals)) if q_vals else 0.0

                # v8.6.27: reuse the per-step parse of the final info.
                # The loop already parsed it; re-parsing here is redundant.
                parsed_info = _parsed_step if _parsed_step is not None else {}

                # v8.6.27: prefer per-step mean; fall back to local.
                if episode_fairness_per_step:
                    fairness = float(np.mean(episode_fairness_per_step))
                elif parsed_info.get('jains_fairness', 0.0) > 0:
                    fairness = float(parsed_info['jains_fairness'])
                else:
                    fairness = compute_jains_fairness(obs)
                converged, stability_info = convergence_detector.update(dl_rate, ep)

                episode_rewards.append(total_env_reward)
                episode_dl_rates.append(dl_rate)
                episode_fairness.append(fairness)
                episode_converged.append(1 if converged else 0)
                episode_losses.append(avg_loss)
                episode_q_values.append(avg_q_ep)
                episode_epsilons.append(agent.epsilon)

                all_episode_rewards.append(total_env_reward)
                all_episode_dl_rates.append(dl_rate)

                moving_avg = float(np.mean(all_episode_rewards[-20:])) if len(all_episode_rewards) >= 20 else float(np.mean(all_episode_rewards))
                moving_dl = float(np.mean(all_episode_dl_rates[-20:])) if len(all_episode_dl_rates) >= 20 else float(np.mean(all_episode_dl_rates))

                episode_moving_avgs.append(moving_avg)
                episode_moving_dls.append(moving_dl)

                # v8.6.27: require ≥100 episodes so best-model is not
                # saved during the random-exploration phase.
                if len(all_episode_dl_rates) >= 100 and moving_dl > best_dl + 0.5:
                    best_dl = moving_dl
                    best_dl_actual = dl_rate
                    agent.save(best_path)

                writer.writerow([
                    ep, 
                    round(total_env_reward, 4),
                    round(total_shaped_reward, 4),
                    round(moving_avg, 4),
                    round(dl_rate, 2),
                    round(moving_dl, 2),
                    round(best_dl, 2),
                    round(avg_loss, 5),
                    round(avg_q_ep, 4), 
                    round(agent.epsilon, 4),
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
                        f"BestDL {best_dl:6.1f}% | "
                        f"Fair {fairness:.3f} | "
                        f"{status} | "
                        f"eps {agent.epsilon:.3f}"
                    )

                if simulator_ended:
                    print(f"Simulator terminated.")
                    break
                    
        # Final save
        agent.save(final_path)

        # Save numpy arrays
        seed_train_dir = os.path.join(train_dir, f"seed_{seed}")
        os.makedirs(seed_train_dir, exist_ok=True)

        np.save(os.path.join(seed_train_dir, f"rewards_seed{seed}.npy"),
                np.array(episode_rewards, dtype=np.float32))
        np.save(os.path.join(seed_train_dir, f"dl_rates_seed{seed}.npy"),
                np.array(episode_dl_rates, dtype=np.float32))
        np.save(os.path.join(seed_train_dir, f"moving_reward_seed{seed}.npy"),
                np.array(episode_moving_avgs, dtype=np.float32))
        np.save(os.path.join(seed_train_dir, f"moving_dl_seed{seed}.npy"),
                np.array(episode_moving_dls, dtype=np.float32))
        np.save(os.path.join(seed_train_dir, f"fairness_seed{seed}.npy"),
                np.array(episode_fairness, dtype=np.float32))
        np.save(os.path.join(seed_train_dir, f"converged_seed{seed}.npy"),
                np.array(episode_converged, dtype=np.int32))
        np.save(os.path.join(seed_train_dir, f"loss_seed{seed}.npy"),
                np.array(episode_losses, dtype=np.float32))
        np.save(os.path.join(seed_train_dir, f"q_values_seed{seed}.npy"),
                np.array(episode_q_values, dtype=np.float32))
        np.save(os.path.join(seed_train_dir, f"epsilon_seed{seed}.npy"),
                np.array(episode_epsilons, dtype=np.float32))

        # Print convergence summary
        conv_stats = convergence_detector.get_stats()
        if conv_stats['converged']:
            dl_stats = get_dl_stats_from_csv(seed, DUELING_CSV_DIR)
            print(f"\n🎯 SEED {seed} CONVERGED at episode {conv_stats['convergence_episode']}")
            print(f"   Mean DL%: {conv_stats['mean_dl']:.2f}%, StdDev: {conv_stats['stddev_dl']:.2f}%")
            print(f"   Best DL%: {dl_stats['best_dl']:.2f}%")
            print(f"   Eval DL%: {dl_stats['mean_dl']:.2f}% ± {dl_stats['std_dl']:.2f}%")
        else:
            print(f"\n⚠️ SEED {seed} did not converge within {NUM_EPISODES} episodes")

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

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 85)
    print("  DOUBLE DUELING DQN — MULTI-SEED TRAINING")
    print(f"  Seeds: {MULTI_SEEDS}   Episodes: {NUM_EPISODES}   Steps: {MAX_STEPS}")
    print(f"  Simulator: mec_simulator_v8_robust_fixed.cc   OBS_SIZE={OBS_SIZE}   NUM_ACTIONS={NUM_ACTIONS}")
    print(f"  Network:   {OBS_SIZE}→256→256→[V:128→1 | A:128→{NUM_ACTIONS}]  +  Double DQN targets")
    print("=" * 85 + "\n")

    all_rewards = {}
    all_dl_rates = {}
    all_fairness = {}
    all_converged = {}

    for seed in MULTI_SEEDS:
        if seed in EXCLUDE_SEEDS:
            print(f"\nSkipping seed {seed} (in EXCLUDE_SEEDS)")
            continue
        print(f"\n{'='*50} SEED {seed} {'='*50}\n")
        
        port = 15000 + seed
        print(f"Checking port {port}...")
        if is_port_in_use(port):
            print(f"⚠️ Port {port} is in use. Killing process...")
            kill_process_on_port(port)
            time.sleep(3)
            if is_port_in_use(port):
                print(f"❌ Port {port} still in use. Forcing kill...")
                import subprocess
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
            print(f"\nSeed {seed} FAILED: {e}")
            import traceback
            traceback.print_exc()
            kill_process_on_port(port)
            time.sleep(5)
            continue

    if not all_rewards:
        print("No seeds completed.")
        sys.exit(1)

    # Save aggregate arrays
    completed_r = [r for r in all_rewards.values() if len(r) == NUM_EPISODES]
    completed_dl = [dl for dl in all_dl_rates.values() if len(dl) == NUM_EPISODES]
    completed_fair = [f for f in all_fairness.values() if len(f) == NUM_EPISODES]

    if completed_r:
        reward_matrix = np.stack(completed_r)
        np.save(os.path.join(DUELING_SUMMARY_DIR, "dueling_ddqn_rewards_phase3.npy"),
                reward_matrix.mean(axis=0))
    if completed_dl:
        dl_matrix = np.stack(completed_dl)
        np.save(os.path.join(DUELING_SUMMARY_DIR, "dueling_ddqn_dl_rates_phase3.npy"),
                dl_matrix.mean(axis=0))
    if completed_fair:
        fairness_matrix = np.stack(completed_fair)
        np.save(os.path.join(DUELING_SUMMARY_DIR, "dueling_ddqn_fairness_phase3.npy"),
                fairness_matrix.mean(axis=0))

    # Summary
    print("\n" + "=" * 85)
    print("DUELING DDQN — MULTI-SEED SUMMARY")
    print("=" * 85)
    
    convergence_summary = []
    for seed in MULTI_SEEDS:
        if seed in EXCLUDE_SEEDS:
            continue
        if seed not in all_rewards:
            print(f"Seed {seed:2d} | FAILED")
            continue
        
        dl_stats = get_dl_stats_from_csv(seed, DUELING_CSV_DIR)
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
    
    print("\n" + "=" * 85)
    print("CONVERGENCE SUMMARY")
    print("=" * 85)
    num_converged = sum(1 for s in convergence_summary if s['converged'])
    print(f"Seeds converged: {num_converged}/{len(MULTI_SEEDS)}")
    if num_converged > 0:
        avg_conv_ep = np.mean([s['convergence_episode'] for s in convergence_summary if s['converged']])
        print(f"Average convergence episode: {avg_conv_ep:.1f}")
    
    print("\n" + "=" * 85)
    print("DL% SUMMARY TABLE")
    print("=" * 85)
    print("Seed | Best DL% | Mean DL% (Eval) | Fairness | Converged")
    print("-" * 60)
    for s in convergence_summary:
        print(f"{s['seed']:4d} | {s['best_dl']:8.2f}% | {s['mean_dl']:14.2f}% | {s['mean_fairness']:8.3f} | {'✅' if s['converged'] else '❌'}")
    
    print("=" * 85)
    print("\nDueling DDQN training complete! 🚀")