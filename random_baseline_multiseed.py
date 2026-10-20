# random_baseline_multiseed.py
"""
Multi-Seed Random Baseline Agent (Phase 3 - v8_robust_fixed)
6G MEC Task Offloading Optimization
IST - University of Lisboa

Author : Pasang Lama

USAGE
-----
    conda activate mec
    python random_baseline_multiseed.py

CHANGELOG (v8.4.4)
------------------
  * [FIX-1] Reward/Raw-Reward label swap removed. CSV now has explicit
            env_reward (raw) and env_reward_norm (post-tanh) columns.
            Console labels match.
  * [FIX-2] Convergence flag forced to 0 in evaluation mode. Detector
            still runs for CSV logging, but no 🎯CONVERGED in eval.
  * [FIX-3] Per-episode metrics snapshotted inside EpisodeStats *before*
            env.reset() fires, so RSU success, HO rate, etc. are not
            read as zero on the terminal step.
  * [FIX-4] Per-action deadline-success rates tracked and logged
            (dl_local, dl_mec0..2, dl_rsu0..1, dl_sat).
  * [FIX-5] DL20 is a proper trailing 20-episode MA, printed only once
            the window is full.
  * [FIX-6] Median + IQR for reward added to aggregate summary.
  * [FIX-7] HO rate shown with two decimals; raw count also logged.
  * [FIX-8] rsu_attempts_this_ep tracked explicitly.
  * [FIX-9] Port collision handling: bump port until free.
  * [FIX-10] Fail-fast on simulator crash in step_with_timeout.
"""

import numpy as np
import signal
import csv
import os
import random
import subprocess
import time
import sys
import json
import re
import socket
import warnings
import pandas as pd
from collections import deque
from statistics import median

import gymnasium as gym
warnings.filterwarnings("ignore", message="Overriding environment")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
sys.modules['gym'] = gym

from ns3gym import ns3env
from env_config import *

# =========================================================
# CONSTANTS - ALIGNED WITH RL AGENTS
# =========================================================
kGnbHomeMec = [0, 0, 1, 1, 2, 2]

# RSU observation indices — matches env_config.py and greedy_baseline
# RSU0 dx,dy = obs[25], obs[26]
# RSU1 dx,dy = obs[27], obs[28]
OBS_IDX_RSU_DX = [25, 27]
OBS_IDX_RSU_DY = [26, 28]
RSU_MAX_RANGE = 300.0

# =========================================================
# CONVERGENCE DETECTION CONSTANTS - ALIGNED WITH C++ & GREEDY
# =========================================================
CONVERGENCE_WINDOW = 50
CONVERGENCE_STDDEV_THRESHOLD = 4.0   # matches C++ and Greedy
CONVERGENCE_MIN_EPISODES = 50
CONVERGENCE_MIN_STABLE_WINDOWS = 10  # matches greedy + C++

# =========================================================
# SETTINGS
# =========================================================
RAND_SEEDS = MULTI_SEEDS      # [1, 7, 21, 42, 84]
BASE_PORT  = 16000            # DQN=15000, Random=16000, Greedy=17000

# =========================================================
# PSUTIL — optional
# =========================================================
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    print("WARNING: psutil not installed — zombie cleanup disabled")

# =========================================================
# PORT / PROCESS UTILITIES
# =========================================================
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def find_free_port(start_port, max_tries=50):
    """[FIX-9] Bump the port until a free one is found."""
    for offset in range(max_tries):
        port = start_port + offset
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No free port near {start_port} after {max_tries} tries")

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
                proc.wait(timeout=5)
        print("ns3 terminated")
    except Exception as e:
        print(f"  ns-3 kill warning: {e}")

# =========================================================
# NS-3 LAUNCHER
# Launch with --evaluate=1 so rewards are raw and comparable
# to frozen RL policies (matches greedy_baseline_multiseed.py).
# =========================================================
def launch_ns3(seed, port):
    cmd = [
        f"{NS3_PATH}/ns3", "run",
        f"mec_simulator_v8_robust_fixed --seed={seed} --port={port} --evaluate=1"
    ]
    print(f"Launching ns-3 v8_robust_fixed seed={seed} port={port} [EVALUATION]...")

    env_dict = os.environ.copy()
    if "PATH" not in env_dict:
        env_dict["PATH"] = "/usr/bin:/bin:/usr/local/bin"

    proc = subprocess.Popen(
        cmd,
        cwd=NS3_PATH,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env_dict
    )
    print(f"  ns-3 PID={proc.pid}")
    time.sleep(8)

    if proc.poll() is not None:
        raise RuntimeError(f"ns-3 crashed immediately (seed={seed})")
    return proc

# =========================================================
# HELPERS
# =========================================================
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
    if safe_index(obs, 2) > 0.5:
        return 0
    elif safe_index(obs, 3) > 0.5:
        return 1
    elif safe_index(obs, 4) > 0.5:
        return 2
    return 0

def decode_gnb_index(obs):
    """
    Decode gNB index from obs[37] (normalized 0-1).
    Use round() to avoid off-by-one from float repr truncation.
    """
    raw = safe_index(obs, 37)
    gnb = int(round(raw * 5.0))
    return max(0, min(gnb, 5))

def get_rsu_dist(obs, rsu_idx):
    """
    Get RSU distance using correct observation indices.
    RSU0 dx/dy = obs[25]/obs[26]
    RSU1 dx/dy = obs[27]/obs[28]
    Multiplied by kAreaSize (1000m) to recover meters.
    """
    if rsu_idx < 0 or rsu_idx >= 2:
        return 1e9
    dx = safe_index(obs, OBS_IDX_RSU_DX[rsu_idx])
    dy = safe_index(obs, OBS_IDX_RSU_DY[rsu_idx])
    return float(np.sqrt(dx * dx + dy * dy) * 1000.0)

def is_within_rsu_range(obs, rsu_idx):
    if rsu_idx < 0 or rsu_idx >= 2:
        return False
    return get_rsu_dist(obs, rsu_idx) <= RSU_MAX_RANGE

def estimate_delay(obs, action):
    """
    Estimate delay from observation.
    obs[12..18] hold Clamp(ratio/3, 0, 1) where ratio = delay/deadline.
    Un-scale by 3 and multiply by deadline to recover seconds.
    """
    ratio_idx = 12 + action
    delay_ratio = safe_index(obs, ratio_idx) * 3.0
    delay_ratio = min(delay_ratio, 3.0)
    deadline_s = safe_index(obs, 5) * 1.0
    return float(delay_ratio * deadline_s)

def compute_jains_fairness(obs):
    """Jain's Fairness from MEC loads (obs[7-9])."""
    loads = [
        safe_index(obs, 7),
        safe_index(obs, 8),
        safe_index(obs, 9)
    ]
    sum_loads = sum(loads)
    if sum_loads < 1e-6:
        return 1.0
    sum_sq = sum([l * l for l in loads])
    n = len(loads)
    fairness = (sum_loads * sum_loads) / (n * sum_sq)
    return float(np.clip(fairness, 0.0, 1.0))

def get_home_mec(gnb_idx):
    if gnb_idx < 0 or gnb_idx >= len(kGnbHomeMec):
        return -1
    return kGnbHomeMec[gnb_idx]

def iqr(values):
    """Interquartile range (Q3 - Q1)."""
    if len(values) < 2:
        return 0.0
    a = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return float(q3 - q1)

# =========================================================
# CONVERGENCE DETECTION CLASS
# Still computed for CSV logging, but `converged` is forced to
# 0 in evaluation mode (see run_random_one_seed).
# =========================================================
class ConvergenceDetector:
    def __init__(self, window=CONVERGENCE_WINDOW,
                 stddev_threshold=CONVERGENCE_STDDEV_THRESHOLD,
                 min_episodes=CONVERGENCE_MIN_EPISODES,
                 min_stable_windows=CONVERGENCE_MIN_STABLE_WINDOWS):
        self.window = window
        self.stddev_threshold = stddev_threshold
        self.min_episodes = min_episodes
        self.min_stable_windows = min_stable_windows
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
            if self.consecutive_stable >= self.min_stable_windows:
                if not self.converged:
                    self.converged = True
                    self.convergence_episode = episode
                return True, "converged"
        else:
            self.consecutive_stable = 0
            self.converged = False

        return self.converged, f"std={stddev_dl:.2f}%"

# =========================================================
# NS-3 INFO PARSER
# =========================================================
_info_debug_printed = False

def _empty_parsed_dict():
    """Canonical empty result matching every field the CSV expects."""
    return {
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
        'avg_mec_bh_ms': 0.0,
        'avg_rsu_bh_ms': 0.0,
        'avg_sat_bh_ms': 0.0,
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
        'convergence_episode': -1,
        'rsu_utilization': 0.0,
        'rsu_success_rate': 0.0,
        'rsu_coverage_pct': 0.0,
        'rsu_offload_attempts': 0.0,
        # Per-action deadline success (optional; 0 if simulator doesn't emit)
        'dl_local': 0.0, 'dl_mec0': 0.0, 'dl_mec1': 0.0, 'dl_mec2': 0.0,
        'dl_rsu0': 0.0, 'dl_rsu1': 0.0, 'dl_sat': 0.0,
        # [FIX-1] Explicit reward fields
        'reward': 0.0,          # post-tanh, in [-1, 1]
        'raw_reward': 0.0,      # pre-tanh, unbounded
        # Diagnostic reward components (per-step average)
        'rc_deadline': 0.0, 'rc_delay': 0.0, 'rc_energy': 0.0,
        'rc_load': 0.0, 'rc_penalty': 0.0, 'rc_backhaul': 0.0,
        'rc_offload': 0.0, 'rc_consistency': 0.0,
    }

_INT_FIELDS = {
    'step', 'episode', 'deadlines_met', 'total_steps',
    'serving_gnb', 'handover', 'handover_count',
    'converged', 'convergence_episode',
}
_STR_FIELDS = {'phase'}

def _parse_ns3_info(info):
    """Parse ALL metrics from ns-3 GetExtraInfo()."""
    global _info_debug_printed

    try:
        if info is None:
            return None

        if not _info_debug_printed:
            print(f"\n✅ ns3gym info received → type={type(info).__name__}")
            if isinstance(info, str) and len(info) < 500:
                print(f"   value={info[:200]}\n")
            _info_debug_printed = True

        # --- Case 1: Dict ---
        if isinstance(info, dict):
            result = _empty_parsed_dict()
            for key in result.keys():
                if key in info:
                    try:
                        if key in _STR_FIELDS:
                            result[key] = str(info[key])
                        elif key in _INT_FIELDS:
                            result[key] = int(info[key])
                        else:
                            result[key] = float(info[key])
                    except (TypeError, ValueError):
                        pass
            return result

        # --- Case 2: JSON string ---
        if isinstance(info, str):
            s = info.strip()

            if s.startswith('{'):
                try:
                    d = json.loads(s)
                    result = _empty_parsed_dict()
                    for key in result.keys():
                        if key in d:
                            try:
                                if key in _STR_FIELDS:
                                    result[key] = str(d[key])
                                elif key in _INT_FIELDS:
                                    result[key] = int(d[key])
                                else:
                                    result[key] = float(d[key])
                            except (TypeError, ValueError):
                                pass
                    return result
                except json.JSONDecodeError:
                    pass

            # --- Case 3: Regex fallback ---
            result = _empty_parsed_dict()
            patterns = {
                'dl_rate': r'"dl_rate"\s*:\s*(-?[0-9]+\.?[0-9]*)',
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
                'avg_mec_bh_ms': r'"avg_mec_bh_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_rsu_bh_ms': r'"avg_rsu_bh_ms"\s*:\s*([0-9]+\.?[0-9]*)',
                'avg_sat_bh_ms': r'"avg_sat_bh_ms"\s*:\s*([0-9]+\.?[0-9]*)',
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
                'convergence_episode': r'"convergence_episode"\s*:\s*(-?[0-9]+)',
                'rsu_utilization': r'"rsu_utilization"\s*:\s*([0-9]+\.?[0-9]*)',
                'rsu_success_rate': r'"rsu_success_rate"\s*:\s*([0-9]+\.?[0-9]*)',
                'rsu_coverage_pct': r'"rsu_coverage_pct"\s*:\s*([0-9]+\.?[0-9]*)',
                'rsu_offload_attempts': r'"rsu_offload_attempts"\s*:\s*([0-9]+\.?[0-9]*)',
                'reward': r'"reward"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'raw_reward': r'"raw_reward"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'dl_local': r'"dl_local"\s*:\s*([0-9]+\.?[0-9]*)',
                'dl_mec0': r'"dl_mec0"\s*:\s*([0-9]+\.?[0-9]*)',
                'dl_mec1': r'"dl_mec1"\s*:\s*([0-9]+\.?[0-9]*)',
                'dl_mec2': r'"dl_mec2"\s*:\s*([0-9]+\.?[0-9]*)',
                'dl_rsu0': r'"dl_rsu0"\s*:\s*([0-9]+\.?[0-9]*)',
                'dl_rsu1': r'"dl_rsu1"\s*:\s*([0-9]+\.?[0-9]*)',
                'dl_sat': r'"dl_sat"\s*:\s*([0-9]+\.?[0-9]*)',
                'rc_deadline':    r'"rc_deadline"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'rc_delay':       r'"rc_delay"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'rc_energy':      r'"rc_energy"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'rc_load':        r'"rc_load"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'rc_penalty':     r'"rc_penalty"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'rc_backhaul':    r'"rc_backhaul"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'rc_offload':     r'"rc_offload"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'rc_consistency': r'"rc_consistency"\s*:\s*(-?[0-9]+\.?[0-9]*)',
            }
            for key, pattern in patterns.items():
                m = re.search(pattern, s)
                if m:
                    try:
                        val = m.group(1)
                        if key in _STR_FIELDS:
                            result[key] = val
                        elif key in _INT_FIELDS:
                            result[key] = int(val)
                        else:
                            result[key] = float(val)
                    except (TypeError, ValueError):
                        pass
            return result

        # --- Case 4: Numeric ---
        if isinstance(info, (int, float)):
            result = _empty_parsed_dict()
            result['dl_rate'] = float(info)
            return result

        return None

    except Exception as e:
        print(f"⚠️ Error parsing ns-3 info: {e}")
        return None

# =========================================================
# RANDOM POLICY
# =========================================================
def random_action(obs=None):
    return random.randint(0, NUM_ACTIONS - 1)

# =========================================================
# PER-EPISODE ACCUMULATOR
# All metrics are captured at the moment the episode ends, so that
# nothing is read after env.reset() fires.
# =========================================================
class EpisodeStats:
    def __init__(self):
        # [FIX-1] Separate raw and post-tanh reward
        # reward_raw: sum over all steps of the C++ pre-tanh raw reward
        # reward_post: sum over all steps of the post-tanh env reward
        self.reward_raw = 0.0
        self.reward_post = 0.0
        self.step_count = 0

        self.delays = []
        self.inf_times_ms = []
        self.action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self.ue_counts = {0: 0, 1: 0, 2: 0}

        self.handovers = 0
        self.home_backhaul = 0
        self.cross_backhaul = 0
        self.gnb_loads = []
        self.mec_loads = [[], [], []]
        self.jains_fairness = []
        self.serving_gnb = []

        # RSU tracking
        self.rsu_attempts = 0
        self.rsu_in_range = 0
        self.rsu_deadline_met = 0

        # [FIX-4] Per-action deadline tracking
        self.action_deadline_met = np.zeros(NUM_ACTIONS, dtype=np.int64)

        # Latest parsed-from-simulator values
        self.last_avg_delay_ms = None
        self.cpp_rsu_success_rate = None
        self.cpp_rsu_coverage_pct = None
        self.cpp_rsu_attempts = None
        self.cpp_per_action_dl = None

    def record_step(self, obs, action, reward_post, inf_ms, parsed_info=None):
        # [FIX-1] `reward_post` is the env's post-tanh reward.
        self.reward_post += reward_post
        # Accumulate pre-tanh raw reward (per-step) so episode sum is
        # directly comparable to greedy's episode-sum env_reward_raw.
        if parsed_info is not None:
            rr = parsed_info.get('raw_reward')
            if rr is not None:
                self.reward_raw += float(rr)
        self.step_count += 1
        self.action_counts[action] += 1
        self.inf_times_ms.append(inf_ms)

        delay = estimate_delay(obs, action)
        self.delays.append(delay)

        # [FIX-4] Per-action deadline success from this step
        deadline_s = safe_index(obs, 5) * 1.0
        if delay <= deadline_s:
            self.action_deadline_met[action] += 1

        self.ue_counts[decode_ue_type(obs)] += 1

        if safe_index(obs, 39) > 0.5:
            self.handovers += 1

        gnb = decode_gnb_index(obs)
        self.serving_gnb.append(gnb)

        if 1 <= action <= 3:
            mec_idx = action - 1
            if gnb < 6 and kGnbHomeMec[gnb] == mec_idx:
                self.home_backhaul += 1
            else:
                self.cross_backhaul += 1

        # RSU tracking
        if 4 <= action <= 5:
            self.rsu_attempts += 1
            rsu_idx = action - 4
            if is_within_rsu_range(obs, rsu_idx):
                self.rsu_in_range += 1
                if delay <= deadline_s:
                    self.rsu_deadline_met += 1

        for m in range(3):
            self.mec_loads[m].append(safe_index(obs, 7 + m))

        self.gnb_loads.append(safe_index(obs, 43))
        self.jains_fairness.append(compute_jains_fairness(obs))

        if parsed_info:
            v = parsed_info.get('avg_delay_ms')
            if v is not None and v > 0:
                self.last_avg_delay_ms = v
            r = parsed_info.get('rsu_success_rate')
            if r is not None and r > 0:
                self.cpp_rsu_success_rate = r
            c = parsed_info.get('rsu_coverage_pct')
            if c is not None and c > 0:
                self.cpp_rsu_coverage_pct = c
            a = parsed_info.get('rsu_offload_attempts')
            if a is not None and a > 0:
                self.cpp_rsu_attempts = a
            # Snapshot per-action DL if the simulator emits them
            per_action = {
                'dl_local': parsed_info.get('dl_local', 0.0),
                'dl_mec0': parsed_info.get('dl_mec0', 0.0),
                'dl_mec1': parsed_info.get('dl_mec1', 0.0),
                'dl_mec2': parsed_info.get('dl_mec2', 0.0),
                'dl_rsu0': parsed_info.get('dl_rsu0', 0.0),
                'dl_rsu1': parsed_info.get('dl_rsu1', 0.0),
                'dl_sat': parsed_info.get('dl_sat', 0.0),
            }
            if any(v > 0 for v in per_action.values()):
                self.cpp_per_action_dl = per_action

    # --- derived metrics (all computed from the snapshot above) ---
    def avg_delay(self):
        return float(np.mean(self.delays)) if self.delays else 0.0

    def max_delay(self):
        return float(np.max(self.delays)) if self.delays else 0.0

    def p95_delay(self):
        return float(np.percentile(self.delays, 95)) if self.delays else 0.0

    def avg_inference_ms(self):
        return float(np.mean(self.inf_times_ms)) if self.inf_times_ms else 0.0

    def ue_fraction(self, ue_type):
        if self.step_count == 0:
            return None
        return 100.0 * self.ue_counts[ue_type] / self.step_count

    def utilization(self):
        total = self.action_counts.sum()
        if total == 0:
            return np.zeros(NUM_ACTIONS)
        return 100.0 * self.action_counts / total

    def handover_rate(self):
        if self.step_count == 0:
            return 0.0
        return 100.0 * self.handovers / self.step_count

    def home_backhaul_pct(self):
        total = self.home_backhaul + self.cross_backhaul
        if total == 0:
            return 0.0
        return 100.0 * self.home_backhaul / total

    def cross_backhaul_pct(self):
        total = self.home_backhaul + self.cross_backhaul
        if total == 0:
            return 0.0
        return 100.0 * self.cross_backhaul / total

    def avg_gnb_load(self):
        return float(np.mean(self.gnb_loads)) if self.gnb_loads else 0.0

    def avg_mec_load(self):
        if not self.mec_loads[0]:
            return 0.0
        all_loads = []
        for m in range(3):
            all_loads.extend(self.mec_loads[m])
        return float(np.mean(all_loads)) if all_loads else 0.0

    def avg_fairness(self):
        return float(np.mean(self.jains_fairness)) if self.jains_fairness else 1.0

    def avg_gnb_idx(self):
        return float(np.mean(self.serving_gnb)) if self.serving_gnb else 0.0

    def ground_truth_avg_delay_ms(self):
        if self.last_avg_delay_ms is not None:
            return self.last_avg_delay_ms
        return self.avg_delay() * 1000.0

    def rsu_success_rate(self):
        if self.cpp_rsu_success_rate is not None and self.cpp_rsu_success_rate > 0:
            return float(self.cpp_rsu_success_rate)
        return 0.0

    def rsu_coverage_pct(self):
        if self.cpp_rsu_coverage_pct is not None and self.cpp_rsu_coverage_pct > 0:
            return float(self.cpp_rsu_coverage_pct)
        return 0.0

    def rsu_attempts_this_ep(self):
        """[FIX-8] Prefer simulator-reported count; fall back to local."""
        if self.cpp_rsu_attempts is not None and self.cpp_rsu_attempts > 0:
            return float(self.cpp_rsu_attempts)
        return float(self.rsu_attempts)

    def per_action_dl(self):
        """[FIX-4] Return per-action DL%, preferring simulator values."""
        if self.cpp_per_action_dl is not None:
            return self.cpp_per_action_dl
        out = {}
        names = ['dl_local', 'dl_mec0', 'dl_mec1', 'dl_mec2',
                 'dl_rsu0', 'dl_rsu1', 'dl_sat']
        for i, name in enumerate(names):
            attempts = int(self.action_counts[i])
            met = int(self.action_deadline_met[i])
            out[name] = (100.0 * met / attempts) if attempts > 0 else 0.0
        return out

# =========================================================
# TIMEOUT STEP
# =========================================================
class StepTimeout(Exception):
    pass

class EnvCloseTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise StepTimeout()

def _close_timeout_handler(signum, frame):
    raise EnvCloseTimeout()

def step_with_timeout(env, action, timeout_sec=STEP_TIMEOUT, retries=2):
    """Step with timeout + retries.
    [FIX-10] Persistent failure returns done=True so the caller exits cleanly.
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
            # Persistent failure → signal done
            return None, -1.0, True, None
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ("cannot be accomplished", "closed", "timeout")):
                print("Simulator has terminated")
                return None, -1.0, True, None
            print(f"Step error: {e}")
            return None, -1.0, True, None
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

# =========================================================
# RUN ONE SEED
# =========================================================
def run_random_one_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    global _info_debug_printed
    _info_debug_printed = False

    # [FIX-9] Bump the port until free
    base_port = BASE_PORT + seed
    port = find_free_port(base_port)
    if port != base_port:
        print(f"  Port {base_port} busy → using {port}")

    print(f"\n{'='*80}")
    print(f"RANDOM BASELINE  SEED {seed}  (port {port})")
    print(f"Mode     : EVALUATION (raw rewards)")
    print(f"Episodes : {NUM_EPISODES} | Steps/Episode : {MAX_STEPS}")
    print(f"{'='*80}\n")

    kill_zombie_ns3()
    kill_process_on_port(port)
    time.sleep(5)
    wait_port_free(port, timeout=15)

    ns3_proc = launch_ns3(seed, port)
    env = None

    episode_rewards_raw = []
    episode_rewards_post = []
    episode_dl_rates = []
    episode_avg_delay = []
    episode_max_delay = []
    episode_p95_delay = []
    episode_inference = []
    episode_server_utilization = []
    episode_ue_fraction = []
    episode_action_counts = []
    episode_handover_rate = []
    episode_handover_count = []
    episode_home_pct = []
    episode_cross_pct = []
    episode_gnb_load = []
    episode_mec_load = []
    episode_fairness = []
    episode_converged = []
    episode_gnb_idx = []
    episode_rsu_success = []
    episode_rsu_coverage = []
    episode_rsu_attempts = []

    ns3_dl_count = 0
    ns3_dl_missing = 0
    best_dl = 0.0
    info_available = False

    # [FIX-5] Proper trailing 20-episode MA
    dl_ma = deque(maxlen=20)

    try:
        for attempt in range(10):
            try:
                time.sleep(4)
                env = ns3env.Ns3Env(port=port, startSim=False, simSeed=seed)
                print(f"Connected to ns-3 on port {port}")
                break
            except Exception as e:
                print(f"Connection attempt {attempt+1}/10 failed: {e}")
                time.sleep(3)

        if env is None:
            raise RuntimeError(f"Could not connect to ns-3 for seed {seed}")

        os.makedirs(RANDOM_CSV_DIR, exist_ok=True)
        csv_path = os.path.join(RANDOM_CSV_DIR, f"random_seed_{seed}.csv")

        convergence_detector = ConvergenceDetector()

        # 41-column header (33 original + 8 new: reward split, per-action DL,
        # HO count, RSU attempts).
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "episode", "phase",
                # v8.6.3: explicit reward columns
                # env_reward_raw = pre-tanh (unbounded), from sim's raw_reward
                # env_reward_post = post-tanh (in [-1,1]), the actual env reward
                "env_reward_raw", "env_reward_post",
                "dl_rate", "moving_dl", "best_dl",
                "avg_delay_ms", "max_delay_ms", "p95_delay_ms",
                "avg_inf_ms",
                # [FIX-7] HO rate + count
                "handover_rate", "handover_count",
                "home_mec_pct", "cross_zone_pct",
                "avg_gnb_load", "avg_mec_load", "jains_fairness",
                "avg_gnb_idx", "converged", "convergence_episode",
                "info_available",
                "ue_mobile_pct", "ue_iot_pct", "ue_vehicle_pct",
                "pct_local", "pct_mec0", "pct_mec1", "pct_mec2",
                "pct_rsu0", "pct_rsu1", "pct_sat",
                "rsu_success_rate", "rsu_coverage_pct",
                # [FIX-8] RSU attempts
                "rsu_attempts_this_ep",
                # [FIX-4] Per-action DL
                "dl_local", "dl_mec0", "dl_mec1", "dl_mec2",
                "dl_rsu0", "dl_rsu1", "dl_sat",
                "rc_deadline", "rc_delay", "rc_energy", "rc_load",
                "rc_penalty", "rc_backhaul", "rc_offload", "rc_consistency",
            ])

            obs = safe_reset(env)

            for ep in range(NUM_EPISODES):
                stats = EpisodeStats()
                last_valid_dl = None
                simulator_ended = False
                phase = "evaluation"

                while stats.step_count < MAX_STEPS:
                    t0 = time.perf_counter()
                    action = random_action()
                    inf_ms = (time.perf_counter() - t0) * 1000.0

                    result = step_with_timeout(env, action)

                    if result[0] is None:
                        simulator_ended = True
                        break

                    next_obs, reward_post, done, info = result

                    parsed_info = _parse_ns3_info(info) if info is not None else None

                    stats.record_step(obs, action, reward_post, inf_ms,
                                      parsed_info=parsed_info)

                    if parsed_info is not None:
                        parsed_dl = parsed_info.get('dl_rate')
                        step_num = parsed_info.get('step')
                        if parsed_dl is not None and (step_num is None or step_num > 0):
                            info_available = True
                            last_valid_dl = parsed_dl

                    obs = next_obs

                    if done:
                        break

                # --- Episode DL ---
                # v8.6.4: prefer Python-side DL (computed from the same
                # per-step counters the simulator uses), because the
                # simulator's extra_info["dl_rate"] is read AFTER the
                # C++ side has reset its episode counters, so it can
                # report a stale or zero value.
                total_deadlines_met = int(stats.action_deadline_met.sum())
                if stats.step_count > 0:
                    episode_dl = 100.0 * total_deadlines_met / stats.step_count
                elif last_valid_dl is not None:
                    episode_dl = last_valid_dl
                else:
                    episode_dl = 0.0

                if last_valid_dl is not None:
                    ns3_received = True
                    ns3_dl_count += 1
                else:
                    ns3_received = False
                    ns3_dl_missing += 1
                    if ns3_dl_missing <= 5:
                        print(f"  WARNING EP {ep}: ns-3 info not received")

                # [FIX-5] Trailing 20-episode MA (append current, then average)
                dl_ma.append(episode_dl)
                moving_dl = float(np.mean(dl_ma)) if len(dl_ma) > 0 else 0.0
                moving_dl_ready = len(dl_ma) >= 20

                # [FIX-2] Force converged=0 in evaluation mode.
                # The detector still runs for CSV logging, but the flag
                # is only reported as converged=1 in training mode.
                _, stability_info = convergence_detector.update(moving_dl, ep)
                converged_effective = 0

                if moving_dl > best_dl:
                    best_dl = moving_dl

                # Snapshot the raw reward for the episode: sum over all steps
                # of the C++ pre-tanh raw reward. This matches greedy's units
                # (episode-sum) and makes the CSV comparable across baselines.
                if stats.reward_raw > 0.0 or stats.reward_post != 0.0:
                    ep_raw_reward = float(stats.reward_raw)
                else:
                    ep_raw_reward = float(stats.reward_post)

                episode_rewards_raw.append(ep_raw_reward)
                episode_rewards_post.append(stats.reward_post)
                episode_dl_rates.append(episode_dl)
                episode_avg_delay.append(stats.ground_truth_avg_delay_ms())
                episode_max_delay.append(stats.max_delay() * 1000)
                episode_p95_delay.append(stats.p95_delay() * 1000)
                episode_inference.append(stats.avg_inference_ms())
                episode_handover_rate.append(stats.handover_rate())
                episode_handover_count.append(stats.handovers)
                episode_home_pct.append(stats.home_backhaul_pct())
                episode_cross_pct.append(stats.cross_backhaul_pct())
                episode_gnb_load.append(stats.avg_gnb_load())
                episode_mec_load.append(stats.avg_mec_load())
                episode_fairness.append(stats.avg_fairness())
                episode_gnb_idx.append(stats.avg_gnb_idx())
                episode_rsu_success.append(stats.rsu_success_rate())
                episode_rsu_coverage.append(stats.rsu_coverage_pct())
                episode_rsu_attempts.append(stats.rsu_attempts_this_ep())
                episode_converged.append(converged_effective)

                util = stats.utilization()
                episode_server_utilization.append(util)

                mob_frac = stats.ue_fraction(0)
                iot_frac = stats.ue_fraction(1)
                veh_frac = stats.ue_fraction(2)
                episode_ue_fraction.append([
                    mob_frac if mob_frac is not None else 0.0,
                    iot_frac if iot_frac is not None else 0.0,
                    veh_frac if veh_frac is not None else 0.0,
                ])
                episode_action_counts.append(stats.action_counts.copy())

                per_action = stats.per_action_dl()

                writer.writerow([
                    ep, phase,
                    round(ep_raw_reward, 4),
                    round(stats.reward_post, 4),
                    round(episode_dl, 2),
                    round(moving_dl, 2) if moving_dl_ready else "",
                    round(best_dl, 2),
                    round(stats.ground_truth_avg_delay_ms(), 2),
                    round(stats.max_delay() * 1000, 2),
                    round(stats.p95_delay() * 1000, 2),
                    round(stats.avg_inference_ms(), 4),
                    round(stats.handover_rate(), 2),
                    int(stats.handovers),
                    round(stats.home_backhaul_pct(), 2),
                    round(stats.cross_backhaul_pct(), 2),
                    round(stats.avg_gnb_load(), 4),
                    round(stats.avg_mec_load(), 4),
                    round(stats.avg_fairness(), 4),
                    round(stats.avg_gnb_idx(), 2),
                    int(converged_effective),
                    convergence_detector.convergence_episode
                        if convergence_detector.converged else -1,
                    int(ns3_received),
                    round(mob_frac, 2) if mob_frac is not None else 0.0,
                    round(iot_frac, 2) if iot_frac is not None else 0.0,
                    round(veh_frac, 2) if veh_frac is not None else 0.0,
                    round(util[0], 2), round(util[1], 2), round(util[2], 2),
                    round(util[3], 2), round(util[4], 2), round(util[5], 2),
                    round(util[6], 2),
                    round(stats.rsu_success_rate(), 2),
                    round(stats.rsu_coverage_pct(), 2),
                    round(stats.rsu_attempts_this_ep(), 0),
                    round(per_action.get('dl_local', 0.0), 2),
                    round(per_action.get('dl_mec0', 0.0), 2),
                    round(per_action.get('dl_mec1', 0.0), 2),
                    round(per_action.get('dl_mec2', 0.0), 2),
                    round(per_action.get('dl_rsu0', 0.0), 2),
                    round(per_action.get('dl_rsu1', 0.0), 2),
                    round(per_action.get('dl_sat', 0.0), 2),
                    round(parsed_info.get('rc_deadline', 0.0) if parsed_info else 0.0, 4),
                    round(parsed_info.get('rc_delay', 0.0) if parsed_info else 0.0, 4),
                    round(parsed_info.get('rc_energy', 0.0) if parsed_info else 0.0, 4),
                    round(parsed_info.get('rc_load', 0.0) if parsed_info else 0.0, 4),
                    round(parsed_info.get('rc_penalty', 0.0) if parsed_info else 0.0, 4),
                    round(parsed_info.get('rc_backhaul', 0.0) if parsed_info else 0.0, 4),
                    round(parsed_info.get('rc_offload', 0.0) if parsed_info else 0.0, 4),
                    round(parsed_info.get('rc_consistency', 0.0) if parsed_info else 0.0, 4),
                ])
                csv_file.flush()

                # [FIX-5] Only show DL20 once the window is full.
                if moving_dl_ready:
                    dl20_str = f"{moving_dl:6.2f}%"
                else:
                    dl20_str = "  (n/a) "

                if ep % 20 == 0 or ep >= NUM_EPISODES - 10 or ep == NUM_EPISODES:
                    # v8.6.6: show simulator's own DL as cross-check when available
                    dl_cpp_str = f" | DL_cpp={last_valid_dl:5.2f}%" if last_valid_dl is not None else ""
                    print(
                        f"EVAL | EP {ep:4d} | "
                        f"R={ep_raw_reward:8.2f} | "
                        f"DL={episode_dl:6.2f}% | "
                        f"DL20={dl20_str} | "
                        f"Fair={stats.avg_fairness():.3f} | "
                        f"HO={stats.handover_rate():5.2f}% | "
                        f"RSU={stats.rsu_success_rate():5.1f}% | "
                        f"Steps={stats.step_count:3d}"
                        f"{dl_cpp_str}"
                    )

                if simulator_ended:
                    print(f"Simulator finished naturally at episode {ep}")
                    break

    finally:
        print(f"\nCleaning up seed {seed} ...")
        if env is not None:
            close_env_safe(env, timeout_sec=5)
        kill_ns3(ns3_proc)
        kill_process_on_port(port)
        wait_port_free(port, timeout=20)
        time.sleep(5)

    if ns3_dl_missing > 0:
        print(f"\n  WARNING: {ns3_dl_missing}/{NUM_EPISODES} episodes had no ns-3 info")

    seed_dir = os.path.join(RANDOM_EVAL_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    np.save(os.path.join(seed_dir, "rewards_raw.npy"), np.array(episode_rewards_raw))
    np.save(os.path.join(seed_dir, "rewards_post.npy"), np.array(episode_rewards_post))
    np.save(os.path.join(seed_dir, "dl_rates.npy"), np.array(episode_dl_rates))
    np.save(os.path.join(seed_dir, "avg_delay.npy"), np.array(episode_avg_delay))
    np.save(os.path.join(seed_dir, "max_delay.npy"), np.array(episode_max_delay))
    np.save(os.path.join(seed_dir, "p95_delay.npy"), np.array(episode_p95_delay))
    np.save(os.path.join(seed_dir, "inference.npy"), np.array(episode_inference))
    np.save(os.path.join(seed_dir, "server_util.npy"), np.array(episode_server_utilization))
    np.save(os.path.join(seed_dir, "ue_fraction.npy"), np.array(episode_ue_fraction))
    np.save(os.path.join(seed_dir, "action_counts.npy"), np.array(episode_action_counts))
    np.save(os.path.join(seed_dir, "fairness.npy"), np.array(episode_fairness))
    np.save(os.path.join(seed_dir, "converged.npy"), np.array(episode_converged))
    np.save(os.path.join(seed_dir, "handover_rate.npy"), np.array(episode_handover_rate))
    np.save(os.path.join(seed_dir, "handover_count.npy"), np.array(episode_handover_count))
    np.save(os.path.join(seed_dir, "home_pct.npy"), np.array(episode_home_pct))
    np.save(os.path.join(seed_dir, "cross_pct.npy"), np.array(episode_cross_pct))
    np.save(os.path.join(seed_dir, "gnb_load.npy"), np.array(episode_gnb_load))
    np.save(os.path.join(seed_dir, "mec_load.npy"), np.array(episode_mec_load))
    np.save(os.path.join(seed_dir, "gnb_idx.npy"), np.array(episode_gnb_idx))
    np.save(os.path.join(seed_dir, "rsu_success.npy"), np.array(episode_rsu_success))
    np.save(os.path.join(seed_dir, "rsu_coverage.npy"), np.array(episode_rsu_coverage))
    np.save(os.path.join(seed_dir, "rsu_attempts.npy"), np.array(episode_rsu_attempts))

    print(f"\n{'='*80}")
    print(f"RANDOM BASELINE  SEED {seed}  FINISHED")
    print(f"{'='*80}")
    print(f"Evaluation Episodes : {len(episode_rewards_raw)} / {NUM_EPISODES}")
    print(f"Avg Raw Reward      : {np.mean(episode_rewards_raw):.3f} "
          f"± {np.std(episode_rewards_raw):.3f}")
    print(f"Avg DL (Eval)       : {np.mean(episode_dl_rates):.2f}% "
          f"± {np.std(episode_dl_rates):.2f}%")
    print(f"Best DL (Eval)      : {best_dl:.2f}%")
    print(f"Fairness (Eval)     : {np.mean(episode_fairness):.4f}")
    print(f"Handover (Eval)     : {np.mean(episode_handover_rate):.3f}%")
    print(f"RSU Success         : {np.mean(episode_rsu_success):.1f}%")
    print(f"RSU Coverage        : {np.mean(episode_rsu_coverage):.1f}%")
    print(f"ns-3 OK             : {ns3_dl_count}/{NUM_EPISODES} episodes")
    print(f"CSV saved           : {csv_path}\n")

    return {
        "seed": seed,
        "rewards_raw": np.array(episode_rewards_raw),
        "rewards_post": np.array(episode_rewards_post),
        "dl_rates": np.array(episode_dl_rates),
        "best_dl": best_dl,
        "avg_delays": np.array(episode_avg_delay),
        "max_delays": np.array(episode_max_delay),
        "p95_delays": np.array(episode_p95_delay),
        "inf_ms": np.array(episode_inference),
        "server_util": np.array(episode_server_utilization),
        "ue_fractions": np.array(episode_ue_fraction),
        "fairness": np.array(episode_fairness),
        "converged": np.array(episode_converged),
        "handover_rate": np.array(episode_handover_rate),
        "handover_count": np.array(episode_handover_count),
        "home_pct": np.array(episode_home_pct),
        "cross_pct": np.array(episode_cross_pct),
        "gnb_load": np.array(episode_gnb_load),
        "mec_load": np.array(episode_mec_load),
        "gnb_idx": np.array(episode_gnb_idx),
        "rsu_success": np.array(episode_rsu_success),
        "rsu_coverage": np.array(episode_rsu_coverage),
        "rsu_attempts": np.array(episode_rsu_attempts),
        "ns3_dl_count": ns3_dl_count,
        "ns3_dl_missing": ns3_dl_missing,
    }

# =========================================================
# AGGREGATE SUMMARY
# =========================================================
def print_aggregate_summary(all_results):
    if not all_results:
        print("No results to summarize.")
        return

    # [FIX-1] Use raw reward as the primary "reward" metric.
    rewards_raw_per_seed = np.array([r["rewards_raw"].mean() for r in all_results])
    dl_per_seed = np.array([np.nanmean(r["dl_rates"]) for r in all_results])
    best_dl_per_seed = np.array([r["best_dl"] for r in all_results])
    delay_per_seed = np.array([r["avg_delays"].mean() for r in all_results])
    fairness_per_seed = np.array([np.nanmean(r["fairness"]) for r in all_results])
    handover_per_seed = np.array([r["handover_rate"].mean() for r in all_results])
    rsu_success_per_seed = np.array([r["rsu_success"].mean() for r in all_results])
    rsu_coverage_per_seed = np.array([r["rsu_coverage"].mean() for r in all_results])

    # [FIX-6] Pooled median + IQR for raw reward and DL.
    pooled_raw_rewards = np.concatenate([r["rewards_raw"] for r in all_results])
    pooled_dl = np.concatenate([r["dl_rates"] for r in all_results])
    raw_median = float(median(pooled_raw_rewards))
    raw_iqr = iqr(pooled_raw_rewards)
    dl_median = float(median(pooled_dl))
    dl_iqr = iqr(pooled_dl)

    print("\n" + "=" * 95)
    print("RANDOM BASELINE — MULTI-SEED SUMMARY (EVALUATION MODE)")
    print("=" * 95)
    print(f"Seeds: {[r['seed'] for r in all_results]}")
    print(f"Episodes/seed: {NUM_EPISODES}")
    print("\n" + "-" * 95)
    print(f"{'Seed':>6} {'DL%':>10} {'Best DL':>10} {'Raw Reward':>12} "
          f"{'Fairness':>10} {'Handover':>10} {'Delay ms':>10} {'RSU%':>10}")
    print("-" * 95)

    for r in all_results:
        print(
            f"{r['seed']:>6} "
            f"{np.nanmean(r['dl_rates']):>9.2f}% "
            f"{r['best_dl']:>10.2f}% "
            f"{r['rewards_raw'].mean():>+12.3f} "
            f"{np.nanmean(r['fairness']):>10.4f} "
            f"{r['handover_rate'].mean():>10.3f}% "
            f"{r['avg_delays'].mean():>10.2f} "
            f"{r['rsu_success'].mean():>9.2f}%"
        )

    print("\n" + "=" * 95)
    print("AGGREGATE STATISTICS (per-seed means)")
    print("=" * 95)
    print(f"DL% Mean:       {np.nanmean(dl_per_seed):.2f}% "
          f"± {np.nanstd(dl_per_seed):.2f}%")
    print(f"Best DL%:       {np.mean(best_dl_per_seed):.2f}%")
    print(f"Raw Reward:     {np.mean(rewards_raw_per_seed):+.3f} "
          f"± {np.std(rewards_raw_per_seed):.3f}")
    print(f"Fairness:       {np.mean(fairness_per_seed):.4f} "
          f"± {np.std(fairness_per_seed):.4f}")
    print(f"Handover:       {np.mean(handover_per_seed):.3f}% "
          f"± {np.std(handover_per_seed):.3f}%")
    print(f"Avg Delay:      {np.mean(delay_per_seed):.2f}ms "
          f"± {np.std(delay_per_seed):.2f}ms")
    print(f"RSU Success:    {np.mean(rsu_success_per_seed):.2f}% "
          f"± {np.std(rsu_success_per_seed):.2f}%")
    print(f"RSU Coverage:   {np.mean(rsu_coverage_per_seed):.2f}% "
          f"± {np.std(rsu_coverage_per_seed):.2f}%")

    print("\n" + "-" * 95)
    print("POOLED PER-EPISODE STATS (all seeds combined)")
    print("-" * 95)
    print(f"Raw Reward      : median={raw_median:+.3f}, IQR={raw_iqr:.3f}")
    print(f"DL%             : median={dl_median:.2f}%, IQR={dl_iqr:.2f}%")

    os.makedirs(RANDOM_SUMMARY_DIR, exist_ok=True)
    np.save(os.path.join(RANDOM_SUMMARY_DIR, "random_rewards_raw.npy"), rewards_raw_per_seed)
    np.save(os.path.join(RANDOM_SUMMARY_DIR, "random_dl_rates.npy"), dl_per_seed)
    np.save(os.path.join(RANDOM_SUMMARY_DIR, "random_fairness.npy"), fairness_per_seed)
    np.save(os.path.join(RANDOM_SUMMARY_DIR, "random_handover.npy"), handover_per_seed)
    np.save(os.path.join(RANDOM_SUMMARY_DIR, "random_delay.npy"), delay_per_seed)
    np.save(os.path.join(RANDOM_SUMMARY_DIR, "random_rsu_success.npy"), rsu_success_per_seed)

    # Machine-readable summary
    summary_json = {
        "seeds": [int(r["seed"]) for r in all_results],
        "episodes_per_seed": int(NUM_EPISODES),
        "per_seed": [
            {
                "seed": int(r["seed"]),
                "dl_mean": float(np.nanmean(r["dl_rates"])),
                "best_dl": float(r["best_dl"]),
                "raw_reward_mean": float(r["rewards_raw"].mean()),
                "fairness_mean": float(np.nanmean(r["fairness"])),
                "handover_mean": float(r["handover_rate"].mean()),
                "delay_mean_ms": float(r["avg_delays"].mean()),
                "rsu_success_mean": float(r["rsu_success"].mean()),
                "rsu_coverage_mean": float(r["rsu_coverage"].mean()),
            }
            for r in all_results
        ],
        "aggregate": {
            "dl_mean": float(np.nanmean(dl_per_seed)),
            "dl_std": float(np.nanstd(dl_per_seed)),
            "raw_reward_mean": float(np.mean(rewards_raw_per_seed)),
            "raw_reward_std": float(np.std(rewards_raw_per_seed)),
            "raw_reward_median": raw_median,
            "raw_reward_iqr": raw_iqr,
            "dl_median": dl_median,
            "dl_iqr": dl_iqr,
            "fairness_mean": float(np.mean(fairness_per_seed)),
            "handover_mean": float(np.mean(handover_per_seed)),
            "delay_mean_ms": float(np.mean(delay_per_seed)),
            "rsu_success_mean": float(np.mean(rsu_success_per_seed)),
            "rsu_coverage_mean": float(np.mean(rsu_coverage_per_seed)),
        },
    }
    json_path = os.path.join(RANDOM_SUMMARY_DIR, "random_multiseed_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSummary JSON saved : {json_path}")
    print("=" * 95)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RANDOM_CSV_DIR, exist_ok=True)
    os.makedirs(RANDOM_EVAL_DIR, exist_ok=True)
    os.makedirs(RANDOM_SUMMARY_DIR, exist_ok=True)

    print("\n" + "=" * 90)
    print("RANDOM BASELINE — MULTI-SEED EVALUATION  [Phase 3 - v8.6.17]")
    print(f"Seeds    : {RAND_SEEDS}")
    print(f"Episodes : {NUM_EPISODES} per seed")
    print(f"Simulator: mec_simulator_v8_robust_fixed (--evaluate=1)")
    print(f"CSV cols : 42 (33 baseline + 9 new diagnostic columns)")
    print(f"           [env_reward_raw, env_reward_post, handover_count,")
    print(f"            rsu_attempts_this_ep, dl_local, dl_mec0..2,")
    print(f"            dl_rsu0..1, dl_sat]")
    print("=" * 90)

    all_results = []

    for seed in RAND_SEEDS:
        if seed in EXCLUDE_SEEDS:
            print(f"\nSkipping seed {seed} (in EXCLUDE_SEEDS)")
            continue
        try:
            result = run_random_one_seed(seed)
            all_results.append(result)
        except Exception as e:
            print(f"\nSeed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        print_aggregate_summary(all_results)
    else:
        print("No seeds completed successfully.")

    print("\n" + "=" * 90)
    print("✅ RANDOM BASELINE EVALUATION COMPLETE!")
    print("   Simulator : mec_simulator_v8_robust_fixed (evaluation mode)")
    print("   CSV cols  : 41 (33 baseline + 8 new diagnostics)")
    print("   Per-action DL + reward split + HO count logged")
    print("=" * 90)