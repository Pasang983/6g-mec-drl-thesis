#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
greedy_baseline_multiseed.py - FIXED & ALIGNED WITH OPTIMIZED SIMULATOR
Multi-Seed Greedy Baseline Agents for 6G MEC Task Offloading
Master's Thesis: Task Offloading Optimization in 6G MEC/VEC/Satellite
IST - University of Lisboa

Author: Pasang Lama

Agents:
  - Greedy Load  : picks lowest-load server with realistic penalties
  - Greedy Ratio : picks best delay/deadline ratio with realistic adjustments

Greedy and Random are NON-LEARNING baselines. They run in evaluation mode
only (--evaluate=1 on the C++ side), which returns raw (unsmoothed) rewards
so their numbers are directly comparable to frozen DDQN/Dueling/PPO policies.

USAGE
-----
    conda activate mec
    python greedy_baseline_multiseed.py load     # Greedy-Load only
    python greedy_baseline_multiseed.py ratio    # Greedy-Ratio only
    python greedy_baseline_multiseed.py both     # Load then Ratio (default)
"""

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
import warnings
import pandas as pd
from collections import deque
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
warnings.filterwarnings("ignore", message="Overriding environment")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
sys.modules['gym'] = gym

# =========================================================
# HANDLE MISSING IMPORTS GRACEFULLY
# =========================================================
try:
    from ns3gym import ns3env
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure ns3gym is installed and in PYTHONPATH")
    sys.exit(1)

try:
    from env_config import *
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure env_config.py is in the same directory or PYTHONPATH")
    sys.exit(1)

# =========================================================
# CONSTANTS - ALIGNED WITH C++ SIMULATOR
# =========================================================
# gNB-to-MEC mapping (6 gNBs: 2 per MEC zone)
kGnbHomeMec = [0, 0, 1, 1, 2, 2]

# MEC Server frequencies (GHz)
MEC_FREQS = [30.0, 25.0, 20.0]

# MEC virtualization overhead (matches C++ kMecVmOverhead = 0.15)
# NOTE: Applied ONLY to MEC actions — RSU has no VM overhead in C++.
MEC_VM_OVERHEAD = 0.15

# MEC max VMs (matches C++ kMecMaxVms = 16.0) — used for pool saturation check
MEC_MAX_VMS = 16.0

# =========================================================
# RSU PARAMETERS - FIXED TO MATCH C++
# =========================================================
RSU_FREQ = 5.0                # GHz (physical channel; C++ uses 5.9 for V2X)
RSU_MAX_RANGE = 300.0         # meters
RSU_BANDWIDTH = 80.0          # MHz
RSU_BACKHAUL_BPS = 5.0e9      # 5 Gbps
RSU_BH_DELAY_MS = 2.0         # 2 ms
RSU_PROCESSING_EFFICIENCY = 0.7      # mirrors C++ kRsuProcessingEfficiency

# =========================================================
# SATELLITE PARAMETERS - MATCH C++
# =========================================================
SAT_ORBIT_HEIGHT    = 550.0      # km
SAT_ELEVATION_MIN   = 25.0      # degrees
SAT_BASE_DELAY_MS   = 20.0      # 20 ms floor (LEO, 550 km, physics-based)
SAT_MAX_DELAY_MS    = 60.0      # 60 ms ceiling

# =========================================================
# OBSERVATION INDEX CONSTANTS - MATCHES env_config.py
# =========================================================
# (Use these instead of magic numbers to prevent index bugs)
OBS_IDX_MEC_LOAD    = [7, 8, 9]
OBS_IDX_RSU_LOAD    = [10, 11]
OBS_IDX_MEC_QUEUE   = [31, 32, 33]
OBS_IDX_RSU_QUEUE   = [34, 35]
# RSU positions: obs[25], obs[26] = RSU0 dx, dy
#               obs[27], obs[28] = RSU1 dx, dy
# (Pre-fix version used 24/25 which is MEC2's dy and RSU0's dx - WRONG)
OBS_IDX_RSU_DX = [25, 27]
OBS_IDX_RSU_DY = [26, 28]

# =========================================================
# CONVERGENCE DETECTION - ALIGNED WITH C++
# =========================================================
CONVERGENCE_WINDOW = 50
CONVERGENCE_STDDEV_THRESHOLD = 4.0
CONVERGENCE_WARMUP_EPISODES = 50
CONVERGENCE_MIN_STABLE_WINDOWS = 10

# =========================================================
# SETTINGS
# =========================================================
GREEDY_SEEDS = MULTI_SEEDS      # [1, 7, 21, 42, 84]
BASE_PORT = 17000             # DQN=15000, Random=16000, Greedy=17000

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
def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def kill_process_on_port(port: int) -> None:
    """Kill any process using the specified port."""
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

def kill_zombie_ns3() -> None:
    """Kill any zombie ns-3 processes."""
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

def wait_port_free(port: int, timeout: int = 20) -> bool:
    """Wait for a port to become free."""
    start = time.time()
    while time.time() - start < timeout:
        if not is_port_in_use(port):
            time.sleep(2)
            return True
        time.sleep(1)
    return False

def kill_ns3(proc: Optional[subprocess.Popen]) -> None:
    """Kill an ns-3 process."""
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
# =========================================================
def launch_ns3(seed: int, port: int) -> subprocess.Popen:
    """
    Launch the ns-3 simulator in EVALUATION mode.

    Greedy is a non-learning baseline, so we always run with --evaluate=1:
      - C++ returns raw (unsmoothed) rewards
      - CSV phase column is labeled "evaluation"
      - Numbers are directly comparable to frozen DDQN/Dueling/PPO policies
    """
    cmd = [
        f"{NS3_PATH}/ns3", "run",
        f"mec_simulator_v8_robust_fixed --seed={seed} --port={port} --evaluate=1"
    ]
    print(f"Launching ns-3 v8_robust_fixed seed={seed} port={port} [EVALUATION]...")

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

# =========================================================
# HELPERS
# =========================================================
def sanitize_obs(obs: Any) -> np.ndarray:
    """Sanitize observation to ensure correct format."""
    if obs is None:
        return np.zeros(OBS_SIZE, dtype=np.float32)
    if isinstance(obs, (int, float, np.number)):
        obs = np.array([obs], dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32).flatten()
    if len(obs) < OBS_SIZE:
        obs = np.pad(obs, (0, OBS_SIZE - len(obs)), 'constant')
    return np.nan_to_num(obs[:OBS_SIZE], nan=0.0, posinf=1.0, neginf=-1.0)

def safe_index(arr: np.ndarray, idx: int, default: float = 0.0) -> float:
    """Safely get an index from an array."""
    arr = np.asarray(arr).flatten()
    if idx < 0 or idx >= len(arr):
        return default
    return float(arr[idx])

def decode_ue_type(obs: np.ndarray) -> int:
    """Decode one-hot UE type from indices 2, 3, 4."""
    if safe_index(obs, 2) > 0.5:
        return 0
    elif safe_index(obs, 3) > 0.5:
        return 1
    elif safe_index(obs, 4) > 0.5:
        return 2
    return 0

def decode_gnb_index(obs: np.ndarray) -> int:
    """
    Decode gNB index from obs[37] (normalized 0-1).
    obs[37] = serving_gnb / (kNumGnb - 1) = serving_gnb / 5
    Uses round() not int() to avoid truncation errors from float repr.
    """
    raw = safe_index(obs, 37)
    gnb = int(round(raw * 5.0))
    return max(0, min(gnb, 5))

# =========================================================
# RSU HELPER FUNCTIONS - ALIGNED WITH C++
# =========================================================
def get_rsu_dist(obs: np.ndarray, rsu_idx: int) -> float:
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
    return float(np.sqrt(dx*dx + dy*dy) * 1000.0)

def is_within_rsu_range(obs: np.ndarray, rsu_idx: int) -> bool:
    """Check if UE is within RSU range (300m)."""
    if rsu_idx < 0 or rsu_idx >= 2:
        return False
    return get_rsu_dist(obs, rsu_idx) <= RSU_MAX_RANGE

def compute_rsu_delay(obs: np.ndarray, rsu_idx: int) -> float:
    """
    Realistic RSU delay (10-80ms).
    Uses CORRECT distance from obs[25-28].

    NOTE: This is a Python-side FALLBACK estimate used only when the C++
    value (parsed from ns-3 info) is unavailable. The authoritative delay
    for thesis tables comes from the C++ simulator via 'avg_delay_ms'.
    """
    dist = get_rsu_dist(obs, rsu_idx)

    # Out of range penalty
    if dist > RSU_MAX_RANGE:
        return 0.100  # 100ms

    cpu_cycles = safe_index(obs, 0) * 5.0 * 1e9
    data_bytes = safe_index(obs, 1) * 50.0 * 1e6
    rsu_load = safe_index(obs, OBS_IDX_RSU_LOAD[rsu_idx])
    ue_type = decode_ue_type(obs)
    speed = safe_index(obs, 30)

    # 1. RSU processing (5.0 GHz × 0.7 efficiency, matches C++ ComputeRsuDelay)
    rsu_freq = 5.0e9 * RSU_PROCESSING_EFFICIENCY * (1.0 - rsu_load * 0.5)
    rsu_freq = max(rsu_freq, 1.0e9)
    comp_time = cpu_cycles / rsu_freq
    comp_time = max(0.005, min(0.200, comp_time))

    # 2. Air transmission (V2X: 5-20ms)
    sinr = safe_index(obs, 38) * 45.0 - 10.0
    sinr_linear = max(0.1, 10 ** (sinr / 10.0))
    spectral_eff = max(0.1, np.log2(1 + sinr_linear) * 0.5)
    capacity = RSU_BANDWIDTH * 1e6 * spectral_eff
    capacity = max(capacity, 1.0e6)
    air_time = (data_bytes * 8.0) / capacity
    if ue_type == 2:  # VEHICLE
        # FIX: C++ uses (1.0 + speed/kMaxSpeed * 0.2), where kMaxSpeed = 30
        mobility_factor = 1.0 + (speed / 30.0) * 0.2
        air_time *= mobility_factor
    air_time = max(0.005, min(0.020, air_time))

    # 3. RSU backhaul (2-5ms)
    bh_time = (data_bytes * 8.0) / RSU_BACKHAUL_BPS + (RSU_BH_DELAY_MS / 1000.0)
    bh_time = max(0.001, min(0.010, bh_time))

    # 4. Queueing delay (0-20ms)
    queue_delay = rsu_load * 0.020

    delay = air_time + bh_time + comp_time + queue_delay
    return delay

def compute_satellite_delay(obs: np.ndarray) -> float:
    """
    Realistic LEO satellite delay - matches C++ ComputeSatelliteDelay().
    Python-side fallback (see compute_rsu_delay note).
    """
    elevation_angle = 25.0 + random.random() * 45.0
    elevation_angle = max(20.0, min(90.0, elevation_angle))

    earth_radius = 6371.0
    orbit_radius = earth_radius + SAT_ORBIT_HEIGHT

    slant_range = np.sqrt(
        orbit_radius**2 + earth_radius**2 -
        2 * orbit_radius * earth_radius *
        np.cos((90.0 - elevation_angle) * np.pi / 180.0)
    )

    prop_delay = slant_range * 1000.0 / 3.0e8
    proc_delay = 0.015 + np.random.random() * 0.025

    speed = safe_index(obs, 30)
    doppler_delay = 0.30 * 0.010 * (speed / 30.0)

    delay = prop_delay + proc_delay + doppler_delay
    return max(0.020, min(0.060, delay))

# =========================================================
# REALISTIC DELAY ESTIMATION - ALIGNED WITH C++
# =========================================================
def estimate_realistic_delay(obs: np.ndarray, action: int) -> float:
    """
    Realistic delay estimation — aligned with C++ v8.4.0+.

    Used ONLY as a fallback when the C++ 'avg_delay_ms' is unavailable.
    The authoritative delay for thesis tables is the C++ value.
    """
    gnb = decode_gnb_index(obs)

    cpu_cycles_giga = safe_index(obs, 0) * 5.0
    data_size_mb    = safe_index(obs, 1) * 50.0
    cpu_cycles_hz   = cpu_cycles_giga * 1e9

    if action == 0:  # LOCAL
        local_freq = 8.0e9
        comp_time = cpu_cycles_hz / local_freq
        comp_time = max(0.005, min(0.200, comp_time))
        return float(comp_time)

    elif 1 <= action <= 3:  # MEC
        mec_idx = action - 1
        load = safe_index(obs, OBS_IDX_MEC_LOAD[mec_idx])
        queue_norm = safe_index(obs, OBS_IDX_MEC_QUEUE[mec_idx])

        mec_freq = MEC_FREQS[mec_idx] * 1e9
        effective_freq = mec_freq * (1.0 - load * 0.5) / (1.0 + MEC_VM_OVERHEAD)
        effective_freq = max(effective_freq, 5.0e9)

        comp_time = cpu_cycles_hz / effective_freq
        comp_time = max(comp_time, 0.005)

        vm_switch_time = 0.001 * (1.0 + load * 4.0)
        comp_time += vm_switch_time

        queue_delay = load * 0.020
        comp_time += queue_delay

        # Mirror C++ VM pool saturation (kMecMaxVms * 0.95 threshold)
        vms_in_use = load * MEC_MAX_VMS
        if vms_in_use >= MEC_MAX_VMS * 0.95:
            comp_time += 0.050

        comp_time = max(0.005, min(0.250, comp_time))

        # Air TX (matches C++ AirTxDelay)
        sinr = safe_index(obs, 38) * 45.0 - 10.0
        sinr_linear = max(0.1, 10 ** (sinr / 10.0))
        spectral_eff = max(0.1, np.log2(1 + sinr_linear) * 0.75)
        effective_cap = 1000e6 * spectral_eff
        air_time = (data_size_mb * 1e6 * 8.0) / max(effective_cap, 1e6)
        air_time *= 1.10
        air_time = max(0.0005, min(0.020, air_time))

        # Backhaul delay
        if gnb < 6 and kGnbHomeMec[gnb] == mec_idx:
            bh_delay = 0.0001
            bh_cap = 100e9
        else:
            bh_delay = 0.0005
            bh_cap = 20e9
        tx_bh = (data_size_mb * 1e6 * 8.0 * 1.1) / bh_cap
        bh_time = bh_delay + tx_bh
        bh_time = max(0.001, min(0.050, bh_time))

        queue_delay = min(0.020, queue_norm * 0.020)

        delay = air_time + bh_time + comp_time + queue_delay
        return float(max(0.005, min(0.500, delay)))

    elif 4 <= action <= 5:  # RSU
        rsu_idx = action - 4
        return compute_rsu_delay(obs, rsu_idx)

    else:  # SAT
        return compute_satellite_delay(obs)

def compute_jains_fairness(obs: np.ndarray) -> float:
    """Compute Jain's Fairness Index (matches C++)."""
    loads = [
        safe_index(obs, OBS_IDX_MEC_LOAD[0]),
        safe_index(obs, OBS_IDX_MEC_LOAD[1]),
        safe_index(obs, OBS_IDX_MEC_LOAD[2]),
        safe_index(obs, OBS_IDX_RSU_LOAD[0]) * 0.5,
        safe_index(obs, OBS_IDX_RSU_LOAD[1]) * 0.5,
    ]
    loads = [max(l, 0.01) for l in loads]

    sum_loads = sum(loads)
    if sum_loads < 1e-6:
        return 1.0

    sum_sq = sum([l * l for l in loads])
    n = len(loads)

    fairness = (sum_loads * sum_loads) / (n * sum_sq)
    return float(np.clip(fairness, 0.0, 1.0))

def get_home_mec(gnb_idx: int) -> int:
    """Get home MEC for a given gNB index."""
    if gnb_idx < 0 or gnb_idx >= len(kGnbHomeMec):
        return -1
    return kGnbHomeMec[gnb_idx]

# =========================================================
# CONVERGENCE DETECTION CLASS
# =========================================================
class ConvergenceDetector:
    """Convergence detector aligned with C++ CheckConvergence()."""
    def __init__(self, window=CONVERGENCE_WINDOW,
                 stddev_threshold=CONVERGENCE_STDDEV_THRESHOLD,
                 min_stable_windows=CONVERGENCE_MIN_STABLE_WINDOWS,
                 warmup_episodes=CONVERGENCE_WARMUP_EPISODES):
        self.window = window
        self.stddev_threshold = stddev_threshold
        self.min_stable_windows = min_stable_windows
        self.warmup_episodes = warmup_episodes
        self.history = deque(maxlen=window)
        self.converged = False
        self.convergence_episode = -1
        self.consecutive_stable = 0
        self.stability_log = []

    def update(self, dl_rate: float, episode: int) -> Tuple[bool, str]:
        self.history.append(dl_rate)

        # Guard: no convergence during curriculum ramp (< 500 episodes)
        if episode < 500:
            return False, "warmup"

        if len(self.history) < self.window:
            return self.converged, "insufficient_data"

        mean_dl = np.mean(self.history)
        stddev_dl = np.std(self.history)

        if stddev_dl < self.stddev_threshold:
            self.consecutive_stable += 1
            if self.consecutive_stable > self.min_stable_windows:
                if not self.converged:
                    self.converged = True
                    self.convergence_episode = episode
                return True, f"converged (std={stddev_dl:.2f}%)"
        else:
            if not self.converged:
                self.consecutive_stable = max(0, self.consecutive_stable - 1)

        return self.converged, f"std={stddev_dl:.2f}%"

    def get_stats(self) -> Dict[str, Any]:
        if len(self.history) < self.window:
            return {
                'converged': False,
                'convergence_episode': -1,
                'window_size': len(self.history),
                'mean_dl': np.mean(self.history) if self.history else 0.0,
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

# =========================================================
# NS-3 INFO PARSER - ALIGNED WITH FIXED SIMULATOR
# =========================================================
_info_debug_printed = False

def _parse_ns3_info(info: Any) -> Optional[Dict[str, Any]]:
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
        # NOTE: ns3gym normally hands us a JSON string (Case 2), but if the
        # binding ever changes to a dict, this branch handles it. The previous
        # version had two unreachable lines after the dict comprehension
        # return, which would have raised NameError if this branch ever ran.
        if isinstance(info, dict):
            result = {k: info.get(k, 0) for k in [
                'dl_rate', 'step', 'episode', 'phase', 'deadlines_met',
                'total_steps', 'serving_gnb', 'avg_gnb_idx', 'handover',
                'avg_delay_ms', 'avg_bh_ms', 'avg_mec_bh_ms', 'avg_rsu_bh_ms', 'avg_sat_bh_ms',
                'avg_air_ms', 'avg_comp_ms',
                'avg_queue_ms', 'home_mec_pct', 'cross_zone_pct',
                'handover_count', 'handover_rate',
                'act_local', 'act_mec0', 'act_mec1', 'act_mec2', 'act_rsu0', 'act_rsu1', 'act_sat',
                'avg_gnb_load', 'avg_mec_load', 'avg_energy_j',
                'throughput_tps', 'jains_fairness', 'converged',
                'convergence_episode', 'rsu_utilization',
                'rsu_coverage_pct', 'rsu_success_rate',
            ]}
            result['jains_fairness'] = float(info.get('jains_fairness', 1.0))
            return result

        # --- Case 2: JSON string ---
        if isinstance(info, str):
            s = info.strip()
            if s.startswith('{'):
                try:
                    d = json.loads(s)
                    dl_rate = float(d.get('dl_rate', 0.0))
                    if dl_rate > 0 and not _info_debug_printed:
                        print(f"    Parsed DL% from JSON: {dl_rate}%")

                    result = {
                        'dl_rate': dl_rate,
                        'raw_reward': float(d.get('raw_reward', 0.0)),
                        'reward': float(d.get('reward', 0.0)),
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
                        'avg_mec_bh_ms': float(d.get('avg_mec_bh_ms', 0.0)),
                        'avg_rsu_bh_ms': float(d.get('avg_rsu_bh_ms', 0.0)),
                        'avg_sat_bh_ms': float(d.get('avg_sat_bh_ms', 0.0)),
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
                        'convergence_episode': int(d.get('convergence_episode', -1)),
                        'rsu_utilization': float(d.get('rsu_utilization', 0.0)),
                        'rsu_coverage_pct': float(d.get('rsu_coverage_pct', 0.0)),
                        'rsu_success_rate': float(d.get('rsu_success_rate', 0.0)),
                        'rc_deadline':    float(d.get('rc_deadline', 0.0)),
                        'rc_delay':       float(d.get('rc_delay', 0.0)),
                        'rc_energy':      float(d.get('rc_energy', 0.0)),
                        'rc_load':        float(d.get('rc_load', 0.0)),
                        'rc_penalty':     float(d.get('rc_penalty', 0.0)),
                        'rc_backhaul':    float(d.get('rc_backhaul', 0.0)),
                        'rc_offload':     float(d.get('rc_offload', 0.0)),
                        'rc_consistency': float(d.get('rc_consistency', 0.0)),
                    }
                    return result
                except json.JSONDecodeError:
                    pass

            # --- Case 3: Regex fallback ---
            result = {k: 0.0 for k in [
                'dl_rate', 'step', 'episode', 'deadlines_met', 'total_steps',
                'raw_reward', 'reward',
                'serving_gnb', 'avg_gnb_idx', 'handover', 'avg_delay_ms',
                'avg_bh_ms', 'avg_mec_bh_ms', 'avg_rsu_bh_ms', 'avg_sat_bh_ms', 'avg_air_ms', 'avg_comp_ms', 'avg_queue_ms',
                'home_mec_pct', 'cross_zone_pct', 'handover_count',
                'handover_rate', 'act_local', 'act_mec0', 'act_mec1',
                'act_mec2', 'act_rsu0', 'act_rsu1', 'act_sat',
                'avg_gnb_load', 'avg_mec_load', 'avg_energy_j',
                'throughput_tps', 'jains_fairness', 'converged',
                'convergence_episode', 'rsu_utilization',
                'rsu_coverage_pct', 'rsu_success_rate',
                'rc_deadline', 'rc_delay', 'rc_energy', 'rc_load',
                'rc_penalty', 'rc_backhaul', 'rc_offload', 'rc_consistency',
            ]}
            result['phase'] = 'unknown'
            result['jains_fairness'] = 1.0

            patterns = {
                'dl_rate': r'"dl_rate"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'raw_reward': r'"raw_reward"\s*:\s*(-?[0-9]+\.?[0-9]*)',
                'reward': r'"reward"\s*:\s*(-?[0-9]+\.?[0-9]*)',
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
                'rsu_coverage_pct': r'"rsu_coverage_pct"\s*:\s*([0-9]+\.?[0-9]*)',
                'rsu_success_rate': r'"rsu_success_rate"\s*:\s*([0-9]+\.?[0-9]*)',
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
                        if key == 'phase':
                            result[key] = val
                        elif '.' in val or 'e' in val:
                            result[key] = float(val)
                        else:
                            result[key] = int(val)
                    except:
                        result[key] = 0.0

            return result

        # --- Case 4: Numeric ---
        if isinstance(info, (int, float)):
            return {
                'dl_rate': float(info), 'step': 0, 'episode': 0, 'phase': 'unknown',
                'deadlines_met': 0, 'total_steps': 0, 'serving_gnb': 0,
                'avg_gnb_idx': 0.0, 'handover': 0, 'avg_delay_ms': 0.0,
                'avg_bh_ms': 0.0, 'avg_mec_bh_ms': 0.0, 'avg_rsu_bh_ms': 0.0,
                'avg_sat_bh_ms': 0.0, 'avg_air_ms': 0.0, 'avg_comp_ms': 0.0,
                'avg_queue_ms': 0.0, 'home_mec_pct': 0.0, 'cross_zone_pct': 0.0,
                'handover_count': 0, 'handover_rate': 0.0, 'act_local': 0.0,
                'act_mec0': 0.0, 'act_mec1': 0.0, 'act_mec2': 0.0,
                'act_rsu0': 0.0, 'act_rsu1': 0.0, 'act_sat': 0.0,
                'avg_gnb_load': 0.0, 'avg_mec_load': 0.0, 'avg_energy_j': 0.0,
                'throughput_tps': 0.0, 'jains_fairness': 1.0, 'converged': 0,
                'convergence_episode': -1, 'rsu_utilization': 0.0,
                'rsu_coverage_pct': 0.0, 'rsu_success_rate': 0.0,
            }

        return None

    except Exception as e:
        print(f"⚠️ Error parsing ns-3 info: {e}")
        return None

def _parse_ns3_dl_rate(info: Any) -> Tuple[Optional[float], Optional[int]]:
    """Legacy: returns only dl_rate and step."""
    parsed = _parse_ns3_info(info)
    if parsed is None:
        return None, None
    return parsed.get('dl_rate', None), parsed.get('step', None)

# =========================================================
# REALISTIC GREEDY POLICIES
# =========================================================
def greedy_load_action(obs: np.ndarray) -> int:
    """
    GREEDY-LOAD policy (v8.6.13).

    Combines a load-based cost with a deadline-feasibility cost.
    The deadline cost is derived from the simulator's ratio observations
    (obs[12..18]), so the policy will not pick an action that clearly
    misses the deadline just because its server has low load.

    Weighting: 30% load, 70% deadline. A failed task is much worse than
    a slightly loaded server, so the deadline dominates.

    This makes greedy-load competitive with (but distinct from)
    greedy-ratio, which uses ratios only.
    """
    gnb = decode_gnb_index(obs)

    cpu_cycles = safe_index(obs, 0) * 5.0
    data_size = safe_index(obs, 1) * 50.0
    task_size = (cpu_cycles / 5.0 + data_size / 50.0) / 2.0
    is_heavy_task = task_size > 0.5
    is_urgent = safe_index(obs, 5) < 0.3

    ue_type = decode_ue_type(obs)
    is_vehicle = (ue_type == 2)

    # ------------------------------------------------------------
    # LOCAL cost: competitive for small tasks, expensive for heavy
    # ------------------------------------------------------------
    local_penalty = 0.30 + task_size * 0.90
    deadline_norm = safe_index(obs, 5)
    if deadline_norm < 0.3:
        local_penalty += 0.4
    elif deadline_norm > 0.6:
        local_penalty -= 0.1

    # ------------------------------------------------------------
    # MEC loads
    # ------------------------------------------------------------
    mec_loads = []
    for m in range(3):
        load = safe_index(obs, OBS_IDX_MEC_LOAD[m])
        effective_load = load * 1.15
        if kGnbHomeMec[gnb] == m:
            effective_load *= 0.85
        else:
            effective_load *= 1.1
        mec_loads.append(effective_load)

    # ------------------------------------------------------------
    # RSU loads with range check
    # ------------------------------------------------------------
    rsu_loads = []
    for r in range(2):
        load = safe_index(obs, OBS_IDX_RSU_LOAD[r])
        if get_rsu_dist(obs, r) > RSU_MAX_RANGE:
            rsu_loads.append(5.0)
        else:
            effective_load = load * 1.1
            if is_heavy_task:
                effective_load *= 1.1
            if is_vehicle:
                effective_load *= 0.7
            rsu_loads.append(effective_load)

    # ------------------------------------------------------------
    # Satellite cost
    # ------------------------------------------------------------
    sat_cost = 0.8
    if is_heavy_task or is_urgent:
        sat_cost += 0.5

    load_costs = np.array([
        local_penalty,
        mec_loads[0],
        mec_loads[1],
        mec_loads[2],
        rsu_loads[0],
        rsu_loads[1],
        sat_cost,
    ])

    # ------------------------------------------------------------
    # Deadline feasibility from the simulator's own ratios.
    # obs[i] = Clamp(ratio / 3.0, 0.0, 1.0), ratio = delay/deadline.
    # ------------------------------------------------------------
    ratios = np.array([float(obs[i]) for i in range(12, 19)]) * 3.0
    ratios = np.nan_to_num(ratios, nan=3.0, posinf=3.0, neginf=3.0)

    # Normalize load costs to [0, 1]
    load_costs_norm = np.clip(load_costs, 0.0, 1.0)

    # Deadline penalty: quadratic growth above ratio = 1
    #   ratio = 0.5 -> 0.25
    #   ratio = 1.0 -> 1.00
    #   ratio = 2.0 -> 4.00
    deadline_costs = np.where(
        ratios <= 1.0,
        ratios * 0.5,
        ratios * ratios
    )
    # v8.6.17: normalize to [0, 1] so 30/70 weighting is meaningful.
    deadline_costs = np.clip(deadline_costs / 5.0, 0.0, 1.0)

    # v8.6.14: Explicit out-of-range penalty for RSUs.
    # The simulator encodes out-of-range RSU delay as ~100 ms, which
    # for a 100 ms deadline gives ratio = 1.0 (just at the boundary).
    # That makes out-of-range RSU look attractive to a load-minimizing
    # policy. Force the ratio to a high value instead.
    for r in range(2):
        if not is_within_rsu_range(obs, r):
            deadline_costs[r + 4] = 5.0

    combined = 0.30 * load_costs_norm + 0.70 * deadline_costs

    return int(np.argmin(combined))

def greedy_ratio_action(obs: np.ndarray) -> int:
    """
    GREEDY-RATIO policy (v8.6.13).

    Picks the action with the smallest delay/deadline ratio, using the
    simulator's own ratio observations from obs[12..18].

    The simulator encodes ratios as:
        obs[i] = Clamp(ratio_i / 3.0, 0.0, 1.0)
    where
        ratio_i = delay_i / deadline
    so we recover:
        ratio_i = obs[i] * 3.0

    A ratio < 1 means the action meets the deadline for the current task.
    A ratio > 1 means it misses.

    This is the delay-optimal policy: myopic, but directly aligned with
    the DL% metric that the RL baselines are compared against.

    Out-of-range RSU actions are given a large penalty, since the
    simulator has already encoded this in the delay (the delay is set
    to a large value when out of range), but we reinforce it here for
    numerical safety.
    """
    # Recover ratios from obs[12..18]
    ratios = np.array([float(obs[i]) for i in range(12, 19)]) * 3.0
    ratios = np.nan_to_num(ratios, nan=3.0, posinf=3.0, neginf=3.0)

    # Out-of-range RSU actions get a large ratio penalty.
    for r in range(2):
        if not is_within_rsu_range(obs, r):
            ratios[r + 4] = 5.0

    # v8.6.17: SAT penalty reduced (was 0.5).
    # The simulator now models SAT congestion: delay grows with
    # aggregate SAT usage within an episode, so the raw ratio is
    # self-regulating and the explicit penalty only needs to encode
    # the backhaul cost, not act as a load-balancing proxy.
    ratios[6] += 0.2

    # Tiebreaker: small load perturbation to break ties when two
    # actions have nearly-equal ratios.
    loads = np.array([
        0.0,                                            # LOCAL: no load
        safe_index(obs, OBS_IDX_MEC_LOAD[0]),
        safe_index(obs, OBS_IDX_MEC_LOAD[1]),
        safe_index(obs, OBS_IDX_MEC_LOAD[2]),
        safe_index(obs, OBS_IDX_RSU_LOAD[0]),
        safe_index(obs, OBS_IDX_RSU_LOAD[1]),
        0.0,                                            # SAT: no load
    ])

    combined = ratios + 0.05 * loads

    return int(np.argmin(combined))

# =========================================================
# PER-EPISODE ACCUMULATOR
# =========================================================
class EpisodeStats:
    """Enhanced episode statistics with all performance metrics."""
    def __init__(self):
        self.reward: float = 0.0
        self.reward_raw: float = 0.0
        self.step_count: int = 0
        self.delays: list = []
        self.inf_times_ms: list = []
        self.action_counts: np.ndarray = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self.ue_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}

        self.handovers: int = 0
        self.home_backhaul: int = 0
        self.cross_backhaul: int = 0
        self.gnb_loads: list = []
        self.mec_loads: list = [[], [], []]
        self.jains_fairness: list = []
        self.serving_gnb: list = []
        self.deadlines_met: int = 0
        self.total_tasks: int = 0

        # RSU metrics
        self.rsu_attempts: int = 0
        self.rsu_success: int = 0
        self.rsu_in_range: int = 0

        # RSU success from C++ (authoritative)
        self.cpp_rsu_success_rate: Optional[float] = None

        # v8.6.8: per-action deadline tracking (for CSV diagnostics)
        self.action_deadline_met: np.ndarray = np.zeros(NUM_ACTIONS, dtype=np.int64)

        self.last_avg_delay_ms: Optional[float] = None
        self.last_dl_rate: Optional[float] = None

    def record_step(self, obs: np.ndarray, action: int, reward: float,
                    inf_ms: float, info: Any = None,
                    parsed_info: Optional[Dict[str, Any]] = None) -> None:
        """Record a step in the episode."""
        self.reward += reward

        if parsed_info is not None:
            rr = parsed_info.get('raw_reward')
            if rr is not None:
                self.reward_raw += float(rr)

        self.step_count += 1
        self.action_counts[action] += 1
        self.inf_times_ms.append(inf_ms)

        delay = estimate_realistic_delay(obs, action)
        self.delays.append(delay)

        # v8.6.8: per-action deadline tracking
        deadline_s = safe_index(obs, 5) * 1.0
        if delay <= deadline_s:
            self.action_deadline_met[action] += 1

        self.ue_counts[decode_ue_type(obs)] += 1

        # Handover flag
        if safe_index(obs, 39) > 0.5:
            self.handovers += 1

        gnb = decode_gnb_index(obs)
        self.serving_gnb.append(gnb)

        # Backhaul classification
        if 1 <= action <= 3:
            mec_idx = action - 1
            home_mec = kGnbHomeMec[gnb]
            if mec_idx == home_mec:
                self.home_backhaul += 1
            else:
                self.cross_backhaul += 1

        # RSU attempt tracking
        if 4 <= action <= 5:
            self.rsu_attempts += 1
            rsu_idx = action - 4
            if is_within_rsu_range(obs, rsu_idx):
                self.rsu_in_range += 1
                # C++ kMaxDeadline = 1.0, so obs[5] == deadline in seconds.
                # If kMaxDeadline changes, multiply here instead.
                deadline_s = safe_index(obs, 5) * 1.0
                self.total_tasks += 1
                if delay <= deadline_s:
                    self.deadlines_met += 1
                    self.rsu_success += 1

        # MEC loads
        for m in range(3):
            self.mec_loads[m].append(safe_index(obs, OBS_IDX_MEC_LOAD[m]))

        self.gnb_loads.append(safe_index(obs, 43))
        self.jains_fairness.append(compute_jains_fairness(obs))

        # ---- Authoritative values from C++ ----
        if info:
            parsed = parsed_info if parsed_info is not None else _parse_ns3_info(info)
            if parsed:
                step_num = parsed.get('step', 0)

                # Only accept non-zero, post-step-0 values.
                # C++ dl_rate and avg_delay_ms are cumulative across the seed,
                # which is exactly what we want for a moving average.
                if step_num > 0:
                    v = parsed.get('dl_rate')
                    if v is not None and v > 0:
                        self.last_dl_rate = v

                    v = parsed.get('avg_delay_ms')
                    if v is not None and v > 0:
                        self.last_avg_delay_ms = v

                # RSU success is also authoritative; accept whenever > 0
                cpp_rate = parsed.get('rsu_success_rate')
                if cpp_rate is not None and cpp_rate > 0:
                    self.cpp_rsu_success_rate = cpp_rate

                # NOTE: Do NOT set self.deadlines_met / total_tasks from
                # 'deadlines_met' / 'total_steps' — those are cumulative
                # counters and would make dl_rate() meaningless.

    def avg_delay(self) -> float:
        return float(np.mean(self.delays)) if self.delays else 0.0

    def max_delay(self) -> float:
        return float(np.max(self.delays)) if self.delays else 0.0

    def p95_delay(self) -> float:
        return float(np.percentile(self.delays, 95)) if self.delays else 0.0

    def avg_inference_ms(self) -> float:
        return float(np.mean(self.inf_times_ms)) if self.inf_times_ms else 0.0

    def ue_fraction(self, ue_type: int) -> Optional[float]:
        if self.step_count == 0:
            return None
        return 100.0 * self.ue_counts[ue_type] / self.step_count

    def utilization(self) -> np.ndarray:
        total = self.action_counts.sum()
        if total == 0:
            return np.zeros(NUM_ACTIONS)
        return 100.0 * self.action_counts / total

    def handover_rate(self) -> float:
        if self.step_count == 0:
            return 0.0
        return 100.0 * self.handovers / self.step_count

    def home_backhaul_pct(self) -> float:
        total = self.home_backhaul + self.cross_backhaul
        if total == 0:
            return 0.0
        return 100.0 * self.home_backhaul / total

    def cross_backhaul_pct(self) -> float:
        total = self.home_backhaul + self.cross_backhaul
        if total == 0:
            return 0.0
        return 100.0 * self.cross_backhaul / total

    def avg_gnb_load(self) -> float:
        return float(np.mean(self.gnb_loads)) if self.gnb_loads else 0.0

    def avg_mec_load(self) -> float:
        if not self.mec_loads[0]:
            return 0.0
        all_loads = []
        for m in range(3):
            all_loads.extend(self.mec_loads[m])
        return float(np.mean(all_loads)) if all_loads else 0.0

    def avg_fairness(self) -> float:
        return float(np.mean(self.jains_fairness)) if self.jains_fairness else 1.0

    def avg_gnb_idx(self) -> float:
        return float(np.mean(self.serving_gnb)) if self.serving_gnb else 0.0

    def rsu_success_rate(self) -> float:
        """
        Prefer C++'s authoritative value when available.
        Falls back to Python's estimate otherwise.
        """
        if self.cpp_rsu_success_rate is not None and self.cpp_rsu_success_rate > 0:
            return float(self.cpp_rsu_success_rate)
        if self.rsu_attempts == 0:
            return 0.0
        return 100.0 * self.rsu_success / self.rsu_attempts

    def rsu_coverage_pct(self) -> float:
        if self.rsu_attempts == 0:
            return 0.0
        return 100.0 * self.rsu_in_range / self.rsu_attempts

    def rsu_deadline_success_rate(self) -> float:
        """
        RSU-only deadline success: % of in-range RSU actions whose
        estimated delay met the task deadline.
        NOTE: total_tasks / deadlines_met are only incremented for
        in-range RSU actions in record_step(). For overall DL%, use
        the C++-reported dl_rate via EpisodeStats.dl_rate().
        """
        if self.total_tasks == 0:
            return 0.0
        return 100.0 * self.deadlines_met / self.total_tasks

    def ground_truth_avg_delay_ms(self) -> float:
        if self.last_avg_delay_ms is not None:
            return self.last_avg_delay_ms
        return self.avg_delay() * 1000.0

    def dl_rate(self) -> float:
        """
        Return the DL% reported by C++.
        C++'s dl_rate is the authoritative cumulative success rate across
        the seed, which is what we plot and compare.
        """
        if self.last_dl_rate is not None:
            return self.last_dl_rate
        return 0.0

    def rsu_attempts_this_ep(self) -> float:
        """v8.6.8: RSU attempts this episode (for CSV column)."""
        return float(self.rsu_attempts)

    def per_action_dl(self) -> Dict[str, float]:
        """
        v8.6.8: per-action deadline-success rate computed from local counters.
        Matches the schema in random_baseline_multiseed.py.
        """
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

def step_with_timeout(env: ns3env.Ns3Env,
                      action: int,
                      timeout_sec: int = STEP_TIMEOUT,
                      retries: int = 2):
    """
    Step with timeout protection and retries.

    Returns:
      (obs, reward, done, info) on success
      ("TRANSIENT_FAIL", -1.0, False, None) on exhausted retries
      (None, -1.0, True, None) on genuine simulator death
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
            # Exhausted retries — transient failure, do NOT end episode
            return "TRANSIENT_FAIL", -1.0, False, None
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ("cannot be accomplished",
                                          "closed", "timeout")):
                print("Simulator has terminated")
                return None, -1.0, True, None
            print(f"Step error: {e}")
            return None, -1.0, True, None
        finally:
            signal.alarm(0)

def close_env_safe(env: Optional[ns3env.Ns3Env], timeout_sec: int = 5) -> None:
    """Safely close the environment with timeout."""
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

# =========================================================
# SAFE RESET
# =========================================================
def safe_reset(env: ns3env.Ns3Env) -> np.ndarray:
    """Safely reset the environment."""
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
# UNIFIED AGENT RUNNER
# =========================================================
def run_agent_one_seed(seed: int, policy_fn, label: str, prefix: str) -> Dict[str, Any]:
    """Run a single seed for a greedy policy (evaluation mode only)."""
    global _info_debug_printed
    _info_debug_printed = False

    random.seed(seed)
    np.random.seed(seed)

    port = BASE_PORT + seed

    print(f"\n{'='*80}")
    print(f"{label}  SEED {seed}  (port {port})")
    print(f"Mode     : EVALUATION (no learning — Greedy policy)")
    print(f"Episodes : {NUM_EPISODES} | Steps/Episode : {MAX_STEPS}")
    print(f"{'='*80}\n")

    # Port cleanup
    kill_zombie_ns3()
    kill_process_on_port(port)
    time.sleep(5)
    wait_port_free(port, timeout=15)

    # Collectors
    episode_rewards: list = []
    episode_dl_rates: list = []
    episode_avg_delay: list = []
    episode_max_delay: list = []
    episode_p95_delay: list = []
    episode_inference: list = []
    episode_server_utilization: list = []
    episode_ue_fraction: list = []
    episode_action_counts: list = []
    episode_handover_rate: list = []
    episode_home_pct: list = []
    episode_cross_pct: list = []
    episode_gnb_load: list = []
    episode_mec_load: list = []
    episode_fairness: list = []
    episode_converged: list = []
    episode_gnb_idx: list = []
    episode_rsu_success: list = []
    episode_rsu_coverage: list = []

    ns3_dl_count: int = 0
    ns3_dl_missing: int = 0
    json_parse_failures: int = 0
    best_dl: float = 0.0
    info_available: bool = False

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if prefix == "greedy_load":
        csv_dir = GREEDY_LOAD_CSV_DIR
        eval_dir = GREEDY_LOAD_EVAL_DIR
        summary_dir = GREEDY_LOAD_SUMMARY_DIR
    else:
        csv_dir = GREEDY_RATIO_CSV_DIR
        eval_dir = GREEDY_RATIO_EVAL_DIR
        summary_dir = GREEDY_RATIO_SUMMARY_DIR

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    ns3_proc = launch_ns3(seed, port)
    env = None

    try:
        # Connect with retry
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
            raise RuntimeError(f"Could not connect to ns-3 for {label} seed={seed}")

        csv_path = os.path.join(csv_dir, f"{prefix}_seed{seed}.csv")
        convergence_detector = ConvergenceDetector()

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            # v8.6.8: 42-column schema, matching random_baseline_multiseed.py
            writer.writerow([
                "episode", "phase",
                "env_reward_raw", "env_reward_post",
                "dl_rate", "moving_dl", "best_dl",
                "avg_delay_ms", "max_delay_ms", "p95_delay_ms",
                "avg_inf_ms",
                "handover_rate", "handover_count",
                "home_mec_pct", "cross_zone_pct",
                "avg_gnb_load", "avg_mec_load", "jains_fairness",
                "avg_gnb_idx", "converged", "convergence_episode",
                "info_available",
                "ue_mobile_pct", "ue_iot_pct", "ue_vehicle_pct",
                "pct_local", "pct_mec0", "pct_mec1", "pct_mec2",
                "pct_rsu0", "pct_rsu1", "pct_sat",
                "rsu_success_rate", "rsu_coverage_pct",
                "rsu_attempts_this_ep",
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
                # Greedy is evaluation-only — always use the eval phase label
                phase = "evaluation"

                while stats.step_count < MAX_STEPS:
                    t0 = time.perf_counter()
                    action = policy_fn(obs)
                    inf_ms = (time.perf_counter() - t0) * 1000.0

                    result = step_with_timeout(env, action)

                    # Genuine simulator death
                    if result[0] is None:
                        simulator_ended = True
                        break

                    # Transient failure — end episode only, not the run
                    # NOTE: must use isinstance(str), NOT ==, because on the
                    # success path result[0] is a NumPy array and
                    # `array == "string"` triggers element-wise broadcast
                    # → ValueError on truth-value coercion.
                    if isinstance(result[0], str) and result[0] == "TRANSIENT_FAIL":
                        print(f"  EP {ep}: transient step failure at step "
                              f"{stats.step_count}, ending episode")
                        break

                    next_obs, reward, done, info = result

                    # Parse info ONCE per step — reuse everywhere
                    parsed_info = _parse_ns3_info(info) if info is not None else None

                    # Detect silent JSON parse failure
                    if (parsed_info is not None
                            and parsed_info.get('phase') == 'unknown'
                            and isinstance(info, str)
                            and info.strip().startswith('{')):
                        json_parse_failures += 1

                    stats.record_step(obs, action, reward, inf_ms, info,
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

                # Determine episode DL
                # Determine episode DL — v8.6.10
                # Compute from Python-side per-step counters, NOT from
                # extra_info['dl_rate']. The C++ dl_rate is read after
                # ResetEnvironment() fires at episode end, so it can report
                # 0.0 at the boundary. Same fix already applied to
                # random_baseline_multiseed.py.
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
                        print(f"  WARNING EP {ep}: ns-3 info missing")

                moving_dl = float(np.mean(episode_dl_rates[-20:])) if len(episode_dl_rates) >= 20 else float(np.mean(episode_dl_rates)) if episode_dl_rates else 0.0

                converged, stability_info = convergence_detector.update(moving_dl, ep)
                episode_converged.append(1 if converged else 0)

                if moving_dl > best_dl:
                    best_dl = moving_dl

                # Store data
                episode_rewards.append(stats.reward)
                episode_dl_rates.append(episode_dl)
                episode_avg_delay.append(stats.ground_truth_avg_delay_ms())
                episode_max_delay.append(stats.max_delay() * 1000.0)
                episode_p95_delay.append(stats.p95_delay() * 1000.0)
                episode_inference.append(stats.avg_inference_ms())
                episode_handover_rate.append(stats.handover_rate())
                episode_home_pct.append(stats.home_backhaul_pct())
                episode_cross_pct.append(stats.cross_backhaul_pct())
                episode_gnb_load.append(stats.avg_gnb_load())
                episode_mec_load.append(stats.avg_mec_load())
                episode_fairness.append(stats.avg_fairness())
                episode_gnb_idx.append(stats.avg_gnb_idx())
                episode_rsu_success.append(stats.rsu_success_rate())
                episode_rsu_coverage.append(stats.rsu_coverage_pct())

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

                # Write CSV — v8.6.8: 42-column row, matching random_baseline_multiseed.py
                # In eval mode, the simulator returns raw rewards only, so
                # env_reward_raw and env_reward_post are the same value.
                per_action_dl = stats.per_action_dl()
                writer.writerow([
                    ep, phase,
                    round(stats.reward_raw, 4),       # env_reward_raw
                    round(stats.reward, 4),           # env_reward_post
                    round(episode_dl, 2),
                    round(moving_dl, 2) if len(episode_dl_rates) >= 20 else "",
                    round(best_dl, 2),
                    round(stats.ground_truth_avg_delay_ms(), 2),
                    round(stats.max_delay() * 1000.0, 2),
                    round(stats.p95_delay() * 1000.0, 2),
                    round(stats.avg_inference_ms(), 4),
                    round(stats.handover_rate(), 2),
                    int(stats.handovers),
                    round(stats.home_backhaul_pct(), 2),
                    round(stats.cross_backhaul_pct(), 2),
                    round(stats.avg_gnb_load(), 4),
                    round(stats.avg_mec_load(), 4),
                    round(stats.avg_fairness(), 4),
                    round(stats.avg_gnb_idx(), 2),
                    0,                                 # converged (forced 0 in eval)
                    -1,                                # convergence_episode
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
                    round(per_action_dl.get('dl_local', 0.0), 2),
                    round(per_action_dl.get('dl_mec0', 0.0), 2),
                    round(per_action_dl.get('dl_mec1', 0.0), 2),
                    round(per_action_dl.get('dl_mec2', 0.0), 2),
                    round(per_action_dl.get('dl_rsu0', 0.0), 2),
                    round(per_action_dl.get('dl_rsu1', 0.0), 2),
                    round(per_action_dl.get('dl_sat', 0.0), 2),
                    round((parsed_info or {}).get('rc_deadline', 0.0), 4),
                    round((parsed_info or {}).get('rc_delay', 0.0), 4),
                    round((parsed_info or {}).get('rc_energy', 0.0), 4),
                    round((parsed_info or {}).get('rc_load', 0.0), 4),
                    round((parsed_info or {}).get('rc_penalty', 0.0), 4),
                    round((parsed_info or {}).get('rc_backhaul', 0.0), 4),
                    round((parsed_info or {}).get('rc_offload', 0.0), 4),
                    round((parsed_info or {}).get('rc_consistency', 0.0), 4),
                ])
                # Flush every episode — 600 rows × 30 cols is trivial I/O,
                # and this protects against losing the last 19 episodes if
                # the run crashes.
                csv_file.flush()

                # Progress
                if ep % 20 == 0 or ep >= NUM_EPISODES - 5 or ep == NUM_EPISODES:
                    status = "🎯CONVERGED" if converged else f"({stability_info})"
                    print(
                        f"EVAL | EP {ep:4d} | "
                        f"R={stats.reward:8.2f} | "
                        f"DL={episode_dl:6.2f}% | "
                        f"DL20={moving_dl:6.2f}% | "
                        f"Fair={stats.avg_fairness():.3f} | "
                        f"HO={stats.handover_rate():.2f}% | "
                        f"RSU={stats.rsu_success_rate():5.1f}% | "
                        f"Cov={stats.rsu_coverage_pct():5.1f}% | "
                        f"{status} | "
                        f"Steps={stats.step_count:3d}"
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

    # Save arrays
    reward_arr = np.asarray(episode_rewards)
    dl_arr = np.asarray(episode_dl_rates)
    avg_delay_arr = np.asarray(episode_avg_delay)
    max_delay_arr = np.asarray(episode_max_delay)
    p95_delay_arr = np.asarray(episode_p95_delay)
    inf_arr = np.asarray(episode_inference)
    server_util = np.asarray(episode_server_utilization)
    ue_fractions = np.asarray(episode_ue_fraction)
    action_counts = np.asarray(episode_action_counts)
    fairness_arr = np.asarray(episode_fairness)
    converged_arr = np.asarray(episode_converged)
    handover_arr = np.asarray(episode_handover_rate)
    home_pct_arr = np.asarray(episode_home_pct)
    cross_pct_arr = np.asarray(episode_cross_pct)
    gnb_load_arr = np.asarray(episode_gnb_load)
    mec_load_arr = np.asarray(episode_mec_load)
    gnb_idx_arr = np.asarray(episode_gnb_idx)
    rsu_success_arr = np.asarray(episode_rsu_success)
    rsu_coverage_arr = np.asarray(episode_rsu_coverage)

    seed_dir = os.path.join(eval_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    np.save(os.path.join(seed_dir, "rewards.npy"), reward_arr)
    np.save(os.path.join(seed_dir, "dl_rates.npy"), dl_arr)
    np.save(os.path.join(seed_dir, "avg_delay.npy"), avg_delay_arr)
    np.save(os.path.join(seed_dir, "max_delay.npy"), max_delay_arr)
    np.save(os.path.join(seed_dir, "p95_delay.npy"), p95_delay_arr)
    np.save(os.path.join(seed_dir, "inference.npy"), inf_arr)
    np.save(os.path.join(seed_dir, "server_util.npy"), server_util)
    np.save(os.path.join(seed_dir, "ue_fraction.npy"), ue_fractions)
    np.save(os.path.join(seed_dir, "action_counts.npy"), action_counts)
    np.save(os.path.join(seed_dir, "fairness.npy"), fairness_arr)
    np.save(os.path.join(seed_dir, "converged.npy"), converged_arr)
    np.save(os.path.join(seed_dir, "handover_rate.npy"), handover_arr)
    np.save(os.path.join(seed_dir, "home_pct.npy"), home_pct_arr)
    np.save(os.path.join(seed_dir, "cross_pct.npy"), cross_pct_arr)
    np.save(os.path.join(seed_dir, "gnb_load.npy"), gnb_load_arr)
    np.save(os.path.join(seed_dir, "mec_load.npy"), mec_load_arr)
    np.save(os.path.join(seed_dir, "gnb_idx.npy"), gnb_idx_arr)
    np.save(os.path.join(seed_dir, "rsu_success.npy"), rsu_success_arr)
    np.save(os.path.join(seed_dir, "rsu_coverage.npy"), rsu_coverage_arr)

    # ============================================================
    # RSU PERFORMANCE — filter to episodes where RSU was attempted
    # ============================================================
    # WHY: RSU success rate is bimodal — 0% for episodes with no RSU
    # action, ~90% for episodes with RSU action. Averaging over all
    # episodes gives a misleading mean (54%) with huge stddev (45%).
    # Filtering to "attempted-only" gives the true RSU performance.
    rsu_mask = (np.asarray(episode_rsu_coverage) > 0) | (np.asarray(episode_rsu_success) > 0)
    n_rsu_attempted = int(rsu_mask.sum())
    if n_rsu_attempted > 0:
        rsu_success_clean = np.asarray(episode_rsu_success)[rsu_mask]
        rsu_coverage_clean = np.asarray(episode_rsu_coverage)[rsu_mask]
        rsu_success_mean = rsu_success_clean.mean()
        rsu_success_std = rsu_success_clean.std()
        rsu_coverage_mean = rsu_coverage_clean.mean()
    else:
        rsu_success_mean = 0.0
        rsu_success_std = 0.0
        rsu_coverage_mean = 0.0

    print(f"\n{label}  SEED {seed}  FINISHED")
    print(f"  Evaluation Episodes : {len(reward_arr)} / {NUM_EPISODES}")
    print(f"  Avg Reward (Eval)   : {reward_arr.mean():.3f} ± {reward_arr.std():.3f}")
    print(f"  Avg DL (Eval)       : {dl_arr.mean():.2f}% ± {dl_arr.std():.2f}%")
    print(f"  Best DL (Eval)      : {best_dl:.2f}%")
    print(f"  Fairness (Eval)     : {fairness_arr.mean():.3f} ± {fairness_arr.std():.3f}")
    print(f"  Handover (Eval)     : {handover_arr.mean():.2f}% ± {handover_arr.std():.2f}%")
    print(f"  RSU Success (all)   : {rsu_success_arr.mean():.1f}% ± {rsu_success_arr.std():.1f}%")
    print(f"  RSU Success (used)  : {rsu_success_mean:.1f}% ± {rsu_success_std:.1f}%  "
          f"[{n_rsu_attempted}/{NUM_EPISODES} eps used RSU]")
    print(f"  RSU Coverage (all)  : {rsu_coverage_arr.mean():.1f}% ± {rsu_coverage_arr.std():.1f}%")
    print(f"  RSU Coverage (used) : {rsu_coverage_mean:.1f}%")
    print(f"  Avg Delay (Eval)    : {avg_delay_arr.mean():.2f}ms ± {avg_delay_arr.std():.2f}ms")
    print(f"  p95 Delay (Eval)    : {p95_delay_arr.mean():.2f}ms ± {p95_delay_arr.std():.2f}ms")
    print(f"  Converged           : {converged_arr.sum()}/{len(converged_arr)} episodes")
    print(f"  ns-3 OK             : {ns3_dl_count}/{NUM_EPISODES} episodes")
    print(f"  CSV                 : {csv_path}\n")

    if json_parse_failures > 0:
        print(f"  ⚠️  {json_parse_failures} JSON parse failures detected — "
              f"check ns-3 ↔ Python JSON format")

    return {
        "seed": seed, "label": label, "prefix": prefix,
        "rewards": reward_arr, "dl_rates": dl_arr, "best_dl": best_dl,
        "avg_delays": avg_delay_arr, "max_delays": max_delay_arr,
        "p95_delays": p95_delay_arr, "inf_ms": inf_arr,
        "server_util": server_util, "action_counts": action_counts,
        "ue_fractions": ue_fractions, "fairness": fairness_arr,
        "converged": converged_arr, "handover_rate": handover_arr,
        "home_pct": home_pct_arr, "cross_pct": cross_pct_arr,
        "gnb_load": gnb_load_arr, "mec_load": mec_load_arr,
        "gnb_idx": gnb_idx_arr, "rsu_success": rsu_success_arr,
        "rsu_coverage": rsu_coverage_arr,
        # attempted-only aggregates
        "rsu_success_used": rsu_success_mean,
        "rsu_coverage_used": rsu_coverage_mean,
        "rsu_episodes_used": n_rsu_attempted,
        "ns3_dl_count": ns3_dl_count, "ns3_dl_missing": ns3_dl_missing,
    }

# =========================================================
# AGGREGATE SUMMARY
# =========================================================
def print_aggregate_summary(all_results: list, prefix: str, label: str) -> None:
    """Print and save aggregate summary across all seeds."""
    if not all_results:
        print(f"No results for {label}")
        return

    rewards_per_seed = np.array([r["rewards"].mean() for r in all_results])
    dl_per_seed = np.array([np.nanmean(r["dl_rates"]) for r in all_results])
    best_dl_per_seed = np.array([r["best_dl"] for r in all_results])
    delay_per_seed = np.array([r["avg_delays"].mean() for r in all_results])
    fairness_per_seed = np.array([r["fairness"].mean() for r in all_results])
    handover_per_seed = np.array([r["handover_rate"].mean() for r in all_results])
    gnb_load_per_seed = np.array([r["gnb_load"].mean() for r in all_results])
    mec_load_per_seed = np.array([r["mec_load"].mean() for r in all_results])
    rsu_success_per_seed = np.array([r["rsu_success"].mean() for r in all_results])
    rsu_coverage_per_seed = np.array([r["rsu_coverage"].mean() for r in all_results])
    rsu_success_used_per_seed = np.array([r.get("rsu_success_used", 0.0) for r in all_results])
    rsu_coverage_used_per_seed = np.array([r.get("rsu_coverage_used", 0.0) for r in all_results])
    rsu_used_count_per_seed = np.array([r.get("rsu_episodes_used", 0) for r in all_results])

    # Correct action percentage computation
    per_seed_pct = []
    for r in all_results:
        counts = r["action_counts"]
        total = counts.sum()
        if total > 0:
            per_seed_pct.append(counts.sum(axis=0) / total * 100.0)
        else:
            per_seed_pct.append(np.zeros(NUM_ACTIONS))
    action_counts_agg = np.mean(per_seed_pct, axis=0)

    ue_stacked = np.stack([r["ue_fractions"] for r in all_results])
    ue_agg = np.mean(ue_stacked, axis=(0, 1))

    converged_per_seed = np.array([
        np.sum(r["converged"]) / len(r["converged"]) * 100
        for r in all_results
    ])

    print("\n" + "=" * 95)
    print(f"{label} — MULTI-SEED SUMMARY (EVALUATION MODE, ALIGNED WITH SIMULATOR v8.6.2)")
    print("=" * 95)
    print(f"Seeds: {[r['seed'] for r in all_results]}")
    print(f"Episodes/seed: {NUM_EPISODES} (evaluation only — Greedy does not train)")
    print(f"Convergence Threshold: {CONVERGENCE_STDDEV_THRESHOLD}%")
    print("\n" + "-" * 95)
    print(f"{'Seed':>6} {'DL%':>10} {'Best DL':>10} {'Reward':>10} {'Fairness':>10} {'Handover':>10} {'Delay ms':>10} {'RSU%':>10}")
    print("-" * 95)

    for r in all_results:
        print(
            f"{r['seed']:>6} "
            f"{np.nanmean(r['dl_rates']):>9.2f}% "
            f"{r['best_dl']:>10.2f}% "
            f"{r['rewards'].mean():>10.2f} "
            f"{np.nanmean(r['fairness']):>10.3f} "
            f"{r['handover_rate'].mean():>10.2f}% "
            f"{r['avg_delays'].mean():>10.2f} "
            f"{r['rsu_success'].mean():>9.1f}%"
        )

    print("\n" + "=" * 95)
    print("AGGREGATE STATISTICS")
    print("=" * 95)
    print(f"DL% Mean:       {np.nanmean(dl_per_seed):.2f}% ± {np.nanstd(dl_per_seed):.2f}%")
    print(f"Best DL%:       {np.mean(best_dl_per_seed):.2f}%")
    print(f"Reward:         {np.mean(rewards_per_seed):.2f} ± {np.std(rewards_per_seed):.2f}")
    print(f"Fairness:       {np.mean(fairness_per_seed):.3f} ± {np.std(fairness_per_seed):.3f}")
    print(f"Handover:       {np.mean(handover_per_seed):.2f}% ± {np.std(handover_per_seed):.2f}%")
    print(f"gNB Load:       {np.mean(gnb_load_per_seed):.3f} ± {np.std(gnb_load_per_seed):.3f}")
    print(f"MEC Load:       {np.mean(mec_load_per_seed):.3f} ± {np.std(mec_load_per_seed):.3f}")
    print(f"Avg Delay:      {np.mean(delay_per_seed):.2f}ms ± {np.std(delay_per_seed):.2f}ms")
    print(f"RSU Success:    {np.mean(rsu_success_per_seed):.1f}% ± {np.std(rsu_success_per_seed):.1f}%   [all episodes]")
    print(f"RSU Success:    {np.mean(rsu_success_used_per_seed):.1f}% ± {np.std(rsu_success_used_per_seed):.1f}%   [RSU-used only]")
    print(f"RSU Coverage:   {np.mean(rsu_coverage_per_seed):.1f}% ± {np.std(rsu_coverage_per_seed):.1f}%   [all episodes]")
    print(f"RSU Coverage:   {np.mean(rsu_coverage_used_per_seed):.1f}%   [RSU-used only]")
    print(f"RSU Used:       {np.mean(rsu_used_count_per_seed):.0f}/{NUM_EPISODES} episodes per seed")
    print(f"Convergence:    {np.mean(converged_per_seed):.1f}% ± {np.std(converged_per_seed):.1f}%")

    print("\n" + "=" * 95)
    print("ACTION DISTRIBUTION (should sum to 100%)")
    print("=" * 95)
    action_names = ["LOCAL", "MEC0", "MEC1", "MEC2", "RSU0", "RSU1", "SAT"]
    for i, name in enumerate(action_names):
        print(f"  {name:<10}: {action_counts_agg[i]:6.2f}%")
    print(f"  {'TOTAL':<10}: {action_counts_agg.sum():6.2f}%")
    if action_counts_agg[6] < 0.1:
        print(f"  NOTE: SAT is dominated by MEC/RSU under current loads "
              f"(SAT delay 20-60ms > MEC 5-200ms when MEC load < 0.7).")
        print(f"  This is EXPECTED and does not indicate a bug.")

    print("\n" + "=" * 95)
    print("UE TYPE DISTRIBUTION")
    print("=" * 95)
    ue_names = ["MOBILE", "IOT", "VEHICLE"]
    for i, name in enumerate(ue_names):
        print(f"  {name:<10}: {ue_agg[i]:.2f}%")
    print("=" * 95)

    summary_dir = GREEDY_SUMMARY_DIR
    os.makedirs(summary_dir, exist_ok=True)
    suffix = "load" if prefix == "greedy_load" else "ratio"

    np.save(os.path.join(summary_dir, f"greedy_{suffix}_rewards.npy"), rewards_per_seed)
    np.save(os.path.join(summary_dir, f"greedy_{suffix}_dl_rates.npy"), dl_per_seed)
    np.save(os.path.join(summary_dir, f"greedy_{suffix}_fairness.npy"), fairness_per_seed)
    np.save(os.path.join(summary_dir, f"greedy_{suffix}_handover.npy"), handover_per_seed)
    np.save(os.path.join(summary_dir, f"greedy_{suffix}_delay.npy"), delay_per_seed)
    np.save(os.path.join(summary_dir, f"greedy_{suffix}_rsu_success.npy"), rsu_success_per_seed)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    print("\n" + "=" * 80)
    print("GREEDY BASELINE — EVALUATION MODE (ALIGNED WITH SIMULATOR v8.6.17)")
    print("6G MEC Task Offloading with RSU/V2X and Satellite")
    print("=" * 80)
    print(f"Seeds: {GREEDY_SEEDS}")
    print(f"Mode: {mode}")
    print(f"Simulator: mec_simulator_v8_robust_fixed (--evaluate=1)")
    print(f"RSU Model: V2X-aware with range {RSU_MAX_RANGE}m")
    print(f"Satellite: LEO with elevation-aware delays (20-60ms)")
    print(f"Convergence Threshold: {CONVERGENCE_STDDEV_THRESHOLD}%")
    print(f"Fixes Applied: RSU indices, round() gNB, 100ms air cap, C++ RSU metrics,")
    print(f"               --evaluate=1 flag, dict-branch NameError fix")
    print("=" * 80)

    if mode in ("load", "both", "all"):
        print("\n" + "=" * 80)
        print("GREEDY-LOAD — MULTI-SEED EVALUATION")
        print(f"Seeds : {GREEDY_SEEDS}")
        print("=" * 80)
        load_results = []
        for seed in GREEDY_SEEDS:
            if seed in EXCLUDE_SEEDS:
                print(f"\nSkipping seed {seed} (in EXCLUDE_SEEDS)")
                continue
            try:
                print(f"\n{'='*30}\nSTARTING GREEDY-LOAD SEED {seed}\n{'='*30}")
                res = run_agent_one_seed(
                    seed, greedy_load_action, "GREEDY-LOAD", "greedy_load"
                )
                print(f"Completed seed {seed}")
                load_results.append(res)
            except Exception as e:
                print(f"\nSeed {seed} GREEDY-LOAD failed: {e}\n")
                import traceback
                traceback.print_exc()
                continue
        if load_results:
            print_aggregate_summary(load_results, "greedy_load", "GREEDY-LOAD")

    if mode in ("ratio", "both", "all"):
        print("\n" + "=" * 80)
        print("GREEDY-RATIO — MULTI-SEED EVALUATION")
        print(f"Seeds : {GREEDY_SEEDS}")
        print("=" * 80)
        ratio_results = []
        for seed in GREEDY_SEEDS:
            if seed in EXCLUDE_SEEDS:
                print(f"\nSkipping seed {seed} (in EXCLUDE_SEEDS)")
                continue
            try:
                print(f"\n{'='*30}\nSTARTING GREEDY-RATIO SEED {seed}\n{'='*30}")
                res = run_agent_one_seed(
                    seed, greedy_ratio_action, "GREEDY-RATIO", "greedy_ratio"
                )
                print(f"Completed seed {seed}")
                ratio_results.append(res)
            except Exception as e:
                print(f"\nSeed {seed} GREEDY-RATIO failed: {e}\n")
                import traceback
                traceback.print_exc()
                continue
        if ratio_results:
            print_aggregate_summary(ratio_results, "greedy_ratio", "GREEDY-RATIO")

    if mode not in ("load", "ratio", "both", "all"):
        print(f"\nUnknown mode: {mode}")
        print("Usage: python greedy_baseline_multiseed.py [load|ratio|both]")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ GREEDY BASELINE EVALUATION COMPLETE!")
    print("   Results aligned with 6G MEC simulator v8.6.17")
    print("   Evaluation mode (--evaluate=1): raw rewards, comparable to frozen RL policies")
    print("   RSU metrics included for V2X performance analysis")
    print(f"   Convergence threshold: {CONVERGENCE_STDDEV_THRESHOLD}%")
    print("=" * 80)