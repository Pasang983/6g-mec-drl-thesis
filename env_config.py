#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# env_config.py
"""
Central Configuration for 48D Observation Space (One-Hot UE Type)
Master's Thesis: Task Offloading Optimization in 6G MEC/VEC/Satellite
ALIGNED WITH: mec_simulator_v8_robust.cc (v8.6.12)

Author : Pasang Lama — IST Lisboa
"""

import os
import sys
import torch
from typing import Dict, List, Optional, Tuple, Any

# ==============================
# THESIS VERSIONING & SCENARIO - ALIGNED WITH C++
# ==============================
ENV_VERSION = "phase3_v8_robust_fixed"
EXPERIMENT_TAG = "ddqn_v8_robust_fixed"
SCENARIO_NAME = "6g_mec_v8_robust_fixed"
VERSION = "8.6.17"                    # ★ UPDATED — matches C++ v8.6.12
SIMULATOR_VERSION = "8.6.17"          # ★ UPDATED

# ==============================
# THESIS SETTINGS
# ==============================
PROJECT_NAME = "6G_MEC_RL_Offloading"
RANDOM_SEED = 42
MULTI_SEEDS = [1, 7, 21, 42, 84]
EXCLUDE_SEEDS = []

# ==============================
# PATHS - UPDATE FOR YOUR SYSTEM
# ==============================
BASE_DIR    = "/mnt/sdb1/users/plama/research/ns-3.38/scratch"
NS3_PATH    = "/mnt/sdb1/users/plama/research/ns-3.38"
RESULTS_DIR = os.path.join(BASE_DIR, "results", ENV_VERSION)

# ------------- DDQN -----------
DDQN_DIR         = os.path.join(RESULTS_DIR, "ddqn")
MODELS_DIR       = os.path.join(DDQN_DIR, "models")
DDQN_TRAIN_DIR   = os.path.join(DDQN_DIR, "train")
DDQN_EVAL_DIR    = os.path.join(DDQN_DIR, "eval")
DDQN_SUMMARY_DIR = os.path.join(DDQN_DIR, "summary")
DDQN_CSV_DIR     = os.path.join(DDQN_DIR, "csv")
DDQN_MODEL_DIR   = os.path.join(DDQN_DIR, "models")

# ---------- Dueling DDQN ----------
DUELING_DIR         = os.path.join(RESULTS_DIR, "dueling_ddqn")
DUELING_MODELS_DIR  = os.path.join(DUELING_DIR, "models")
DUELING_TRAIN_DIR   = os.path.join(DUELING_DIR, "train")
DUELING_EVAL_DIR    = os.path.join(DUELING_DIR, "eval")
DUELING_SUMMARY_DIR = os.path.join(DUELING_DIR, "summary")
DUELING_CSV_DIR     = os.path.join(DUELING_DIR, "csv")
DUELING_MODEL_DIR   = os.path.join(DUELING_DIR, "models")

# ---------- PPO ----------
PPO_DIR             = os.path.join(RESULTS_DIR, "ppo")
PPO_MODELS_DIR      = os.path.join(PPO_DIR, "models")
PPO_TRAIN_DIR       = os.path.join(PPO_DIR, "train")
PPO_EVAL_DIR        = os.path.join(PPO_DIR, "eval")
PPO_SUMMARY_DIR     = os.path.join(PPO_DIR, "summary")
PPO_CSV_DIR         = os.path.join(PPO_DIR, "csv")
PPO_MODEL_DIR       = os.path.join(PPO_DIR, "models")

# ----------- Random Baseline ----------
RANDOM_DIR          = os.path.join(RESULTS_DIR, "random")
RANDOM_EVAL_DIR     = os.path.join(RANDOM_DIR, "eval")
RANDOM_SUMMARY_DIR  = os.path.join(RANDOM_DIR, "summary")
RANDOM_CSV_DIR      = os.path.join(RANDOM_DIR, "csv")

# ------------- Greedy Baseline -----------
GREEDY_DIR              = os.path.join(RESULTS_DIR, "greedy")
GREEDY_LOAD_EVAL_DIR    = os.path.join(GREEDY_DIR, "load", "eval")
GREEDY_RATIO_EVAL_DIR   = os.path.join(GREEDY_DIR, "ratio", "eval")
GREEDY_LOAD_CSV_DIR     = os.path.join(GREEDY_DIR, "load", "csv")
GREEDY_RATIO_CSV_DIR    = os.path.join(GREEDY_DIR, "ratio", "csv")
GREEDY_LOAD_SUMMARY_DIR = os.path.join(GREEDY_DIR, "load", "summary")
GREEDY_RATIO_SUMMARY_DIR = os.path.join(GREEDY_DIR, "ratio", "summary")
GREEDY_SUMMARY_DIR      = os.path.join(GREEDY_DIR, "summary")

# -------- shared outputs --------
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR        = os.path.join(RESULTS_DIR, "logs")
COMPARISON_DIR  = os.path.join(RESULTS_DIR, "comparison")
EVALUATION_DIR  = os.path.join(RESULTS_DIR, "evaluation")

# ==============================
# EXPERIMENT CONTROLS - MATCHES C++
# ==============================
USE_MOBILITY = True
USE_WIRELESS_REALISM = True
USE_QUEUE_MODEL = True           # ✓ MATCHES C++: kEnableQueueModel = true
USE_HANDOVER = True
USE_BACKHAUL = True
USE_PRIORITY_SCHEDULING = False
USE_LOAD_BALANCING_METRICS = True
USE_RSU_MODEL = True
USE_SATELLITE_MODEL = True

# ==============================
# RSU/V2X CONFIGURATION - ALIGNED WITH C++
# ==============================
RSU_CONFIG = {
    "num_rsu": 2,
    "max_range_m": 300.0,           # ✓ MATCHES C++: kRsuMaxRange_m = 300.0
    "frequency_ghz": 5.9,           # ✓ MATCHES C++: kRsuFreq_GHz = 5.9
    "bandwidth_mhz": 80.0,          # ✓ MATCHES C++: kRsuBandwidth_MHz = 80.0
    "antenna_gain_dbi": 12.0,       # ✓ MATCHES C++: kRsuAntennaGain_dBi = 12.0
    "base_delay_ms": 10.0,
    "backhaul_bps": 5.0e9,          # ✓ MATCHES C++: kRsuBackhaul_bps = 5.0e9
    "backhaul_delay_ms": 2.0,       # ✓ MATCHES C++: kRsuBhDelay_s = 0.002
    "capacity_tasks_per_sec": 35.0,
    "processing_efficiency": 0.7,   # ✓ MATCHES C++: kRsuProcessingEfficiency = 0.7
    "mobility_bonus": 0.6,
    "path_loss_model": "v2x_3gpp_tr_37_885",
}

# ==============================
# MEC CONFIGURATION - ALIGNED WITH C++
# ==============================
MEC_CONFIG = {
    "num_mec": 3,
    "frequencies_ghz": [30.0, 25.0, 20.0],  # ✓ MATCHES C++: kMecFreq0/1/2
    "vm_overhead": 0.15,                    # ✓ MATCHES C++: kMecVmOverhead = 0.15
    "max_vms": 16.0,                        # ✓ MATCHES C++: kMecMaxVms = 16.0
    "base_delay_ms": 5.0,
    "capacity_tasks_per_sec": 100.0,
    "power_idle_w": 20.0,                   # ✓ MATCHES C++: kMecPowerIdle_W = 20.0
    "power_max_w": 80.0,                    # ✓ MATCHES C++: kMecPowerMax_W = 80.0
}

# ==============================
# SATELLITE CONFIGURATION - ALIGNED WITH C++
# ==============================
SATELLITE_CONFIG = {
    "orbit_height_km": 550.0,           # ✓ MATCHES C++: kSatOrbitHeight_km = 550.0
    "elevation_min_deg": 25.0,          # ✓ MATCHES C++: kSatElevationMin_deg = 25.0
    "base_delay_ms": 20.0,              # ✓ MATCHES C++: kSatBaseDelay_s = 0.020
    "max_delay_ms": 60.0,               # ✓ MATCHES C++: kSatMaxDelay_s = 0.060
    "propagation_delay_ms": 4.0,
    "processing_delay_ms": 5.0,
    "doppler_shift_khz": 40.0,
    "capacity_tasks_per_sec": 5.0,
    "tx_power_w": 2.0,                  # ✓ MATCHES C++: kSatTxPower_W = 2.0
}

# ==============================
# TASK PARAMETER RANGES (v8.4) — ALIGNED WITH C++ GenerateNewTask()
# ==============================
TASK_RANGES = {
    "embb": {
        "cpu_cycles_giga": (0.3, 3.0),
        "data_size_mb":    (0.5, 15.0),
        "deadline_ms":     (80, 150),    # ★ UPDATED — matches C++ Clamp(...,0.080,0.150)
    },
    "mmtc": {
        "cpu_cycles_giga": (0.01, 0.35),
        "data_size_mb":    (0.01, 1.0),
        "deadline_ms":     (200, 400),   # ★ UPDATED — matches C++ Clamp(...,0.200,0.400)
    },
    "v2x": {
        "cpu_cycles_giga": (0.3, 2.5),
        "data_size_mb":    (1.0, 18.0),
        "deadline_ms":     (60, 120),    # ★ UPDATED — matches C++ Clamp(...,0.060,0.120)
    },
}

# ==============================
# QUEUEING CONFIGURATION - ALIGNED WITH C++
# ==============================
QUEUEING_CONFIG = {
    "model": "simple",
    "priority_levels": 1,
    "urgent_threshold": 0.2,
    "normal_threshold": 0.6,
    "service_rate_base": 100.0,
    "max_utilization": 0.95,
    "priority_weights": [1.0],
    "stability_check_interval": 10,
    "decay_rate": 0.97,             # ✓ MATCHES C++: kQueueDecayRate = 0.97
}

# ==============================
# LOAD BALANCING CONFIGURATION - ALIGNED WITH C++
# ==============================
LOAD_BALANCING_CONFIG = {
    "use_jains_fairness": True,
    "use_gini_coefficient": False,
    "use_load_std_dev": False,
    "fairness_log_interval": 1,
    "target_load": 0.5,
    "load_decay_rate": 0.995,         # ★ UPDATED — C++ kLoadDecayRate = 0.995
    "load_increment": 0.015,          # ★ UPDATED — C++ kLoadIncrement = 0.015
    # v8.6.6 background traffic (models other UEs in same MEC zone)
    "background_load_prob":   0.10,   # ★ UPDATED — C++ kBackgroundLoadProb = 0.10
    "background_load_amount": 0.008,  # ★ UPDATED — C++ kBackgroundLoadAmount = 0.008
}

# ==============================
# CONVERGENCE CONFIGURATION - ALIGNED WITH C++
# ==============================
CONVERGENCE_CONFIG = {
    "window": 50,                     # ✓ MATCHES C++: kConvergenceWindow = 50
    "stddev_threshold": 4.0,          # ✓ MATCHES C++: kConvergenceThreshold = 4.0
    "min_episodes": 50,
    "consecutive_stable": 10,         # ✓ MATCHES C++: kConvergenceMinCount = 10
}

# ==============================
# REWARD CONFIGURATION - WEIGHTS SUM TO 1.0
# ALIGNED WITH C++ v8.6.12
# ==============================
REWARD_CONFIG = {
    # Primary weights - sum to 1.0 ✓ (C++ constants now authoritative)
    "weight_deadline": 0.55,          # ★ UPDATED — C++ kWeightDeadline = 0.55
    "weight_delay": 0.10,             # ★ UPDATED — C++ kWeightDelay = 0.10
    "weight_energy": 0.05,            # ★ UPDATED — C++ kWeightEnergy = 0.05
    "weight_load": 0.10,              # ✓ MATCHES C++: kWeightLoad = 0.10
    "weight_penalty": 0.05,           # ★ UPDATED — C++ kWeightPenalty = 0.05
    "weight_backhaul": 0.15,          # ✓ MATCHES C++: kWeightBackhaul = 0.15
    # Sum = 1.00 ✓

    # Bonuses
    "offload_bonus": 0.15,
    "rsu_bonus": 0.30,
    "v2x_bonus": 0.30,
    "stability_bonus": 0.03,

    # Penalties
    "satellite_penalty": 0.25,
    "local_penalty": 0.10,
    "cross_zone_penalty": 0.10,

    # Bonus configurations
    "offload_bonus_enabled": True,
    "consistency_bonus_enabled": True,
    "smooth_factor": 0.4,             # ✓ MATCHES C++: kRewardSmoothFactor = 0.4
    "reward_squash": "tanh",

    # Penalties
    "local_heavy_penalty": 0.4,
    "handover_penalty": 0.01,
    "rsu_out_of_range_penalty": 0.15,
}

# ==============================
# OBSERVATION SPACE (48 Features) - MATCHES C++ ONE-HOT ENCODING
# ==============================
OBS_SIZE = 48  # ✓ MATCHES C++: kObsSize = 48

FEATURE_NAMES = [
    # Task Features (0-1)
    "cpu_cycles_norm",      # 0
    "data_size_norm",       # 1

    # UE Type - One-Hot (2-4)
    "ue_type_mobile",       # 2
    "ue_type_iot",          # 3
    "ue_type_vehicle",      # 4

    # Deadline & Urgency (5-6)
    "deadline_norm",        # 5
    "urgency",              # 6

    # Server Loads (7-11)
    "mec_load_0",           # 7
    "mec_load_1",           # 8
    "mec_load_2",           # 9
    "rsu_load_0",           # 10
    "rsu_load_1",           # 11

    # Delay Ratios (12-18) - All 7 Actions
    "delay_ratio_local",    # 12
    "delay_ratio_mec0",     # 13
    "delay_ratio_mec1",     # 14
    "delay_ratio_mec2",     # 15
    "delay_ratio_rsu0",     # 16
    "delay_ratio_rsu1",     # 17
    "delay_ratio_sat",      # 18

    # Relative Positions (19-28) - [-1, 1] range
    "dx_mec0",              # 19
    "dy_mec0",              # 20
    "dx_mec1",              # 21
    "dy_mec1",              # 22
    "dx_mec2",              # 23
    "dy_mec2",              # 24
    "dx_rsu0",              # 25
    "dy_rsu0",              # 26
    "dx_rsu1",              # 27
    "dy_rsu1",              # 28

    # Satellite & Mobility (29-30)
    "sat_available",        # 29
    "speed_norm",           # 30

    # Queue Lengths (31-35)
    "mec_queue_0",          # 31
    "mec_queue_1",          # 32
    "mec_queue_2",          # 33
    "rsu_queue_0",          # 34
    "rsu_queue_1",          # 35

    # Channel (36)
    "fading_factor",        # 36

    # gNB + Backhaul Features (37-47)
    "serving_gnb",          # 37
    "air_sinr",             # 38
    "handover",             # 39
    "bh_delay_mec0",        # 40
    "bh_delay_mec1",        # 41
    "bh_delay_mec2",        # 42
    "gnb_load",             # 43
    "gnb_dist",             # 44
    "path_delay_mec0",      # 45
    "path_delay_mec1",      # 46
    "path_delay_mec2",      # 47
]

# ==============================
# FEATURE CATEGORIES - UPDATED FOR 48 DIMS
# ==============================
FEATURE_CATEGORIES = {
    "task": [0, 1],
    "ue": [2, 3, 4, 30],
    "deadline": [5, 6],
    "mec": [7, 8, 9, 31, 32, 33],
    "rsu": [10, 11, 34, 35],
    "delay_ratios": [12, 13, 14, 15, 16, 17, 18],
    "position": list(range(19, 29)),
    "satellite": [29],
    "channel": [36],
    "gnb": [37, 38, 39, 43, 44],
    "backhaul": [40, 41, 42],
    "path_delay": [45, 46, 47],
}

# ==============================
# FEATURE DESCRIPTIONS FOR THESIS
# ==============================
FEATURE_DESCRIPTIONS = {
    "cpu_cycles_norm": "Normalized CPU cycles requirement (0-1)",
    "data_size_norm": "Normalized data size in MB (0-1)",
    "ue_type_mobile": "Mobile UE indicator [1,0,0] (60% of UEs)",
    "ue_type_iot": "IoT UE indicator [0,1,0] (25% of UEs)",
    "ue_type_vehicle": "Vehicle UE indicator [0,0,1] (15% of UEs)",
    "deadline_norm": "Normalized task deadline (0-1)",
    "urgency": "Urgency measure: best_delay / deadline",
    "mec_load_0": "MEC0 utilization (0-1) - 30GHz server",
    "mec_load_1": "MEC1 utilization (0-1) - 25GHz server",
    "mec_load_2": "MEC2 utilization (0-1) - 20GHz server",
    "rsu_load_0": "RSU0 utilization (0-1) - V2X at 5.9GHz",
    "rsu_load_1": "RSU1 utilization (0-1) - V2X at 5.9GHz",
    "delay_ratio_local": "Local execution delay / deadline",
    "delay_ratio_mec0": "MEC0 offload delay / deadline",
    "delay_ratio_mec1": "MEC1 offload delay / deadline",
    "delay_ratio_mec2": "MEC2 offload delay / deadline",
    "delay_ratio_rsu0": "RSU0 offload delay / deadline (V2X)",
    "delay_ratio_rsu1": "RSU1 offload delay / deadline (V2X)",
    "delay_ratio_sat": "Satellite offload delay / deadline (LEO)",
    "dx_mec0": "UE-MEC0 x-distance normalized (-1 to 1)",
    "dy_mec0": "UE-MEC0 y-distance normalized (-1 to 1)",
    "dx_mec1": "UE-MEC1 x-distance normalized (-1 to 1)",
    "dy_mec1": "UE-MEC1 y-distance normalized (-1 to 1)",
    "dx_mec2": "UE-MEC2 x-distance normalized (-1 to 1)",
    "dy_mec2": "UE-MEC2 y-distance normalized (-1 to 1)",
    "dx_rsu0": "UE-RSU0 x-distance normalized (-1 to 1)",
    "dy_rsu0": "UE-RSU0 y-distance normalized (-1 to 1)",
    "dx_rsu1": "UE-RSU1 x-distance normalized (-1 to 1)",
    "dy_rsu1": "UE-RSU1 y-distance normalized (-1 to 1)",
    "sat_available": "Satellite availability (always 1)",
    "speed_norm": "UE speed normalized (0-1)",
    "mec_queue_0": "MEC0 queue length normalized",
    "mec_queue_1": "MEC1 queue length normalized",
    "mec_queue_2": "MEC2 queue length normalized",
    "rsu_queue_0": "RSU0 queue length normalized",
    "rsu_queue_1": "RSU1 queue length normalized",
    "fading_factor": "Rayleigh/Rician fading factor",
    "serving_gnb": "Serving gNB index normalized (0-5)",
    "air_sinr": "Air interface SINR normalized",
    "handover": "Handover occurrence flag (0/1)",
    "bh_delay_mec0": "gNB→MEC0 backhaul delay normalized",
    "bh_delay_mec1": "gNB→MEC1 backhaul delay normalized",
    "bh_delay_mec2": "gNB→MEC2 backhaul delay normalized",
    "gnb_load": "gNB uplink load normalized",
    "gnb_dist": "UE-gNB distance normalized",
    "path_delay_mec0": "T_air + T_bh to MEC0",
    "path_delay_mec1": "T_air + T_bh to MEC1",
    "path_delay_mec2": "T_air + T_bh to MEC2",
}

# ==============================
# OBSERVATION NORMALIZATION BOUNDS - UPDATED
# ==============================
OBS_LOW = [0.0] * OBS_SIZE
OBS_HIGH = [1.0] * OBS_SIZE
for i in range(19, 29):  # Position features (19-28) have [-1, 1] range
    OBS_LOW[i] = -1.0
    OBS_HIGH[i] = 1.0

# ==============================
# OBSERVATION VALIDATION
# ==============================
def validate_observation(obs: Any) -> bool:
    """Validate observation shape and ranges"""
    if obs is None:
        raise ValueError("Observation is None")
    if len(obs) != OBS_SIZE:
        raise ValueError(f"Expected {OBS_SIZE} features, got {len(obs)}")
    for i, val in enumerate(obs):
        low = -1.0 if 19 <= i <= 28 else 0.0
        high = 1.0
        if val < low or val > high:
            raise ValueError(f"Feature {i} ({FEATURE_NAMES[i]}) out of range: {val}")
    return True

# ==============================
# ACTION SPACE - MATCHES C++
# ==============================
NUM_ACTIONS = 7  # ✓ MATCHES C++: kNumActions = 7

ACTION_NAMES = [
    "LOCAL",      # 0
    "MEC0",       # 1
    "MEC1",       # 2
    "MEC2",       # 3
    "RSU0",       # 4
    "RSU1",       # 5
    "Satellite"   # 6
]

ACTION_LABELS = {
    0: "LOCAL (UE)",
    1: "MEC0 (30GHz, gNB→BH)",
    2: "MEC1 (25GHz, gNB→BH)",
    3: "MEC2 (20GHz, gNB→BH)",
    4: "RSU0 (V2X @ 5.9GHz, 300m range)",
    5: "RSU1 (V2X @ 5.9GHz, 300m range)",
    6: "Satellite (LEO @ 550km, 20-60ms)"
}

ACTION_CATEGORIES = {
    "local": [0],
    "mec": [1, 2, 3],
    "rsu": [4, 5],
    "satellite": [6]
}

ACTION_BACKHAUL_TYPE = {
    0: "none",
    1: "gnb",
    2: "gnb",
    3: "gnb",
    4: "rsu",
    5: "rsu",
    6: "satellite"
}

ACTION_SERVER_TYPES = {
    0: "local",
    1: "mec",
    2: "mec",
    3: "mec",
    4: "rsu",
    5: "rsu",
    6: "satellite"
}

# ==============================
# SERVER PARAMETERS - ALIGNED WITH C++
# ==============================
SERVER_PARAMETERS = {
    "local": {
        "freq_ghz": 8.0,                    # ★ UPDATED — C++ kLocalFreq = 8.0e9
        # NOTE: C++ computes LOCAL delay as cpu_cycles / kLocalFreq, clamped
        # to [5ms, 200ms]. This field is authoritative.
        "capacity_tasks_per_sec": 20.0,
        "power_idle_w": 0.05,
        "power_max_w": 0.5,
    },
    "mec": {
        "freqs_ghz": [30.0, 25.0, 20.0],    # ✓ MATCHES C++: kMecFreq0/1/2
        "base_delay_ms": 5.0,
        "capacity_tasks_per_sec": 100.0,
        "power_idle_w": 20.0,
        "power_max_w": 80.0,
        "vm_overhead": 0.15,
    },
    "rsu": {
        "freq_ghz": 5.0,                    # ★ UPDATED — C++ kRsuFreq = 5.0e9 (processing clock)
        # NOTE: kRsuFreq_GHz = 5.9 is the V2X channel carrier (in RSU_CONFIG),
        # whereas kRsuFreq = 5.0e9 is the compute clock. Do not conflate.
        "base_delay_ms": 10.0,
        "capacity_tasks_per_sec": 25.0,
        "power_idle_w": 15.0,
        "power_max_w": 35.0,
        "max_range_m": 300.0,
        "antenna_gain_dbi": 12.0,
    },
    "satellite": {
        "freq_ghz": 20.0,                   # documentation only (C++ has no kFc_SAT_GHz constant)
        "base_delay_ms": 20.0,              # ✓ MATCHES C++: kSatBaseDelay_s = 0.020
        "capacity_tasks_per_sec": 5.0,
        "orbit_height_km": 550.0,
        "elevation_min_deg": 25.0,
        "tx_power_w": 2.0,
    }
}

# ==============================
# GNB CONFIGURATION - MATCHES C++
# ==============================
GNB_LABELS = {
    0: "gNB0 (MEC0)",
    1: "gNB1 (MEC0)",
    2: "gNB2 (MEC1)",
    3: "gNB3 (MEC1)",
    4: "gNB4 (MEC2)",
    5: "gNB5 (MEC2)"
}

GNB_HOME_MEC = {
    0: 0, 1: 0,  # gNB0-1 → MEC0  ✓ MATCHES C++: kGnbHomeMec
    2: 1, 3: 1,  # gNB2-3 → MEC1
    4: 2, 5: 2   # gNB4-5 → MEC2
}

GNB_POSITIONS = {
    0: (200.0, 400.0),
    1: (400.0, 200.0),
    2: (300.0, 700.0),
    3: (600.0, 850.0),
    4: (750.0, 450.0),
    5: (900.0, 250.0)
}

MEC_POSITIONS = {
    0: (200.0, 200.0),
    1: (500.0, 800.0),
    2: (800.0, 300.0)
}

RSU_POSITIONS = {
    0: (350.0, 500.0),
    1: (700.0, 600.0)
}

# ==============================
# GNB BACKHAUL HELPERS
# ==============================
def get_home_mec(gnb_idx: int) -> int:
    """Get home MEC for a gNB index."""
    return GNB_HOME_MEC.get(gnb_idx, -1)

def is_cross_zone_backhaul(gnb_idx: int, mec_idx: int) -> bool:
    """Check if backhaul is cross-zone."""
    return get_home_mec(gnb_idx) != mec_idx

def get_backhaul_type(gnb_idx: int, mec_idx: int) -> str:
    """Get backhaul type description."""
    if is_cross_zone_backhaul(gnb_idx, mec_idx):
        return "cross-zone (2ms + 20Gbps)"
    else:
        return "home (0.1ms + 100Gbps)"

def action_uses_gnb(action: int) -> bool:
    """Check if action uses gNB backhaul."""
    return action in [1, 2, 3, 6]

def action_uses_rsu(action: int) -> bool:
    """Check if action uses RSU backhaul."""
    return action in [4, 5]

def action_uses_backhaul(action: int) -> bool:
    """Check if action uses any backhaul."""
    return action != 0

# ==============================
# TRAINING SETTINGS - MATCHES C++
# ==============================
NUM_EPISODES    = 600      # ✓ MATCHES C++: kMaxEpisodes = 600
MAX_STEPS       = 200      # ✓ MATCHES C++: kStepsPerEpisode = 200
PORT            = 5555
STEP_TIME       = 0.1      # ✓ MATCHES C++: kStepInterval_s = 0.1
STEP_TIMEOUT    = 30

# ==============================
# EVALUATION SETTINGS
# ==============================
EVAL_EPISODES       = 200
EVAL_SEEDS          = [1, 7, 21, 42, 84]
EVAL_PHASE_EPISODES = 200

# ==============================
# REPLAY / RL SETTINGS
# ==============================
GAMMA               = 0.98
LR                  = 1.2e-4
BATCH_SIZE          = 128
MEMORY_SIZE         = 100000
MIN_REPLAY_SIZE     = 5000
TAU                 = 0.005
TARGET_UPDATE_FREQ  = 1000
REWARD_CLIP         = (-1.0, 1.0)
SAVE_EVERY          = 50

EPS_START           = 1.0
EPS_END             = 0.05
EPS_DECAY           = 0.9975
EPS_DECAY_STEPS     = 1000

# ==============================
# METRICS TO LOG
# ==============================
METRICS_TO_LOG = [
    "avg_delay_ms",
    "avg_backhaul_ms",
    "home_backhaul_pct",
    "cross_backhaul_pct",
    "handover_rate",
    "serving_gnb",
    "air_sinr",
    "gnb_load",
    "mec_load_avg",
    "queue_length_avg",
    "energy_consumed_j",
    "throughput_tps",
    "action_distribution",
    "deadline_success_rate",
    "jains_fairness",
    "convergence_status",
    "convergence_episode",

    "rsu_utilization",
    "rsu_coverage_pct",
    "rsu_success_rate",
    "rsu_offload_attempts",
]

# ==============================
# ABLATION STUDIES (UPDATED)
# ==============================
ABLATION_CONFIGS = {
    "full": {
        "use_mobility": True,
        "use_wireless_realism": True,
        "use_queue_model": True,
        "use_handover": True,
        "use_backhaul": True,
        "use_priority_scheduling": False,
        "use_load_balancing_metrics": True,
        "use_rsu_model": True,
        "use_satellite_model": True,
        "description": "Full model with RSU/V2X, satellite, one-hot UE encoding"
    },
    "no_mobility": {
        "use_mobility": False,
        "use_wireless_realism": True,
        "use_queue_model": True,
        "use_handover": True,
        "use_backhaul": True,
        "use_priority_scheduling": False,
        "use_load_balancing_metrics": True,
        "use_rsu_model": True,
        "use_satellite_model": True,
        "description": "Static UE (no mobility)"
    },
    "no_backhaul": {
        "use_mobility": True,
        "use_wireless_realism": True,
        "use_queue_model": True,
        "use_handover": True,
        "use_backhaul": False,
        "use_priority_scheduling": False,
        "use_load_balancing_metrics": True,
        "use_rsu_model": True,
        "use_satellite_model": True,
        "description": "No backhaul delay modeling"
    },
    "no_rsu": {
        "use_mobility": True,
        "use_wireless_realism": True,
        "use_queue_model": True,
        "use_handover": True,
        "use_backhaul": True,
        "use_priority_scheduling": False,
        "use_load_balancing_metrics": True,
        "use_rsu_model": False,
        "use_satellite_model": True,
        "description": "No RSU/V2X model"
    },
    "no_fairness": {
        "use_mobility": True,
        "use_wireless_realism": True,
        "use_queue_model": True,
        "use_handover": True,
        "use_backhaul": True,
        "use_priority_scheduling": False,
        "use_load_balancing_metrics": False,
        "use_rsu_model": True,
        "use_satellite_model": True,
        "description": "No Jain's Fairness in reward"
    }
}

# ==============================
# HARDWARE
# ==============================
USE_CUDA            = torch.cuda.is_available()
DEVICE_NAME         = torch.device("cuda" if USE_CUDA else "cpu")
TORCH_DETERMINISTIC = True

# ==============================
# LABELS (UPDATED)
# ==============================
SERVER_LABELS = {
    0: "LOCAL", 1: "MEC0",  2: "MEC1",  3: "MEC2",
    4: "RSU0",  5: "RSU1",  6: "Satellite"
}

UE_LABEL = {
    0: "MOBILE",
    1: "IOT",
    2: "VEHICLE"
}

UE_LABEL_ONEHOT = {
    "mobile": [1, 0, 0],
    "iot": [0, 1, 0],
    "vehicle": [0, 0, 1]
}

BACKHAUL_LABELS = {
    "home": "Home (0.1ms fiber, 100Gbps)",
    "cross": "Cross-zone (2ms mmWave, 20Gbps)",
    "rsu": "RSU (2ms backhaul, 5Gbps)",
    "satellite": "Satellite LEO (20–60ms RTT, 550km)",
    "none": "No backhaul"
}

PRIORITY_LABELS = {
    0: "Urgent (dl < 200ms)",
    1: "Normal (200-600ms)",
    2: "Background (dl > 600ms)"
}

# ==============================
# COLOR SCHEMES FOR PLOTS
# ==============================
ALGORITHM_COLORS = {
    "Greedy Ratio": "#2ecc71",
    "Greedy Load": "#27ae60",
    "PPO": "#3498db",
    "Dueling DDQN": "#e67e22",
    "Random": "#95a5a6",
    "DDQN": "#e74c3c",
}

UE_TYPE_COLORS = {
    "MOBILE": "#3498db",
    "IOT": "#2ecc71",
    "VEHICLE": "#e74c3c"
}

SERVER_COLORS = {
    "LOCAL": "#95a5a6",
    "MEC0": "#3498db",
    "MEC1": "#2980b9",
    "MEC2": "#1a5276",
    "RSU0": "#2ecc71",
    "RSU1": "#27ae60",
    "Satellite": "#f39c12",
}

QUEUEING_COLORS = {
    "Urgent": "#e74c3c",
    "Normal": "#3498db",
    "Background": "#95a5a6"
}

# ==============================
# INITIALISE DIRECTORIES
# ==============================
def _init_dirs() -> None:
    """Create all required directories."""
    dirs = [
        RESULTS_DIR,
        # DDQN
        DDQN_DIR, MODELS_DIR, DDQN_TRAIN_DIR, DDQN_EVAL_DIR,
        DDQN_SUMMARY_DIR, DDQN_CSV_DIR,
        # Dueling DDQN
        DUELING_DIR, DUELING_MODELS_DIR, DUELING_TRAIN_DIR,
        DUELING_EVAL_DIR, DUELING_SUMMARY_DIR, DUELING_CSV_DIR,
        # PPO
        PPO_DIR, PPO_MODELS_DIR, PPO_TRAIN_DIR,
        PPO_EVAL_DIR, PPO_SUMMARY_DIR, PPO_CSV_DIR,
        # Random
        RANDOM_DIR, RANDOM_EVAL_DIR, RANDOM_SUMMARY_DIR, RANDOM_CSV_DIR,
        # Greedy
        GREEDY_DIR,
        GREEDY_LOAD_EVAL_DIR, GREEDY_LOAD_CSV_DIR, GREEDY_LOAD_SUMMARY_DIR,
        GREEDY_RATIO_EVAL_DIR, GREEDY_RATIO_CSV_DIR, GREEDY_RATIO_SUMMARY_DIR,
        GREEDY_SUMMARY_DIR,
        # Shared
        PLOTS_DIR, LOGS_DIR, COMPARISON_DIR, EVALUATION_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

_init_dirs()

# ==============================
# CROSS-CHECK WITH C++ GROUND TRUTH
# ==============================
# Authoritative values from mec_simulator_v8_robust.cc (v8.6.12).
# If any assertion fails at import time, env_config.py has drifted from C++.
_CPP_GROUND_TRUTH = {
    # Reward weights
    "weight_deadline":     0.55,       # ★ UPDATED
    "weight_delay":        0.10,       # ★ UPDATED
    "weight_energy":       0.05,       # ★ UPDATED
    "weight_load":         0.10,
    "weight_penalty":      0.05,       # ★ UPDATED
    "weight_backhaul":     0.15,
    "reward_smooth_factor": 0.4,
    # RSU
    "rsu_max_range_m":     300.0,
    "rsu_freq_ghz":        5.9,
    "rsu_bandwidth_mhz":   80.0,
    "rsu_backhaul_bps":    5.0e9,
    "rsu_bh_delay_s":      0.002,
    "rsu_antenna_gain_dbi": 12.0,
    "rsu_processing_efficiency": 0.7,
    # MEC
    "mec_vm_overhead":     0.15,
    "mec_freqs_ghz":       [30.0, 25.0, 20.0],
    "mec_power_idle_w":    20.0,
    "mec_power_max_w":     80.0,
    "mec_max_vms":         16.0,
    # Local
    "local_freq_ghz":      8.0,        # ★ UPDATED
    # Load & queue
    "load_increment":      0.015,      # ★ UPDATED
    "load_decay_rate":     0.995,      # ★ UPDATED
    "queue_decay_rate":    0.97,
    # Convergence
    "convergence_window":     50,
    "convergence_threshold":  4.0,
    "convergence_min_count":  10,
    "convergence_min_episodes": 50,
    # Satellite
    "sat_orbit_height_km": 550.0,
    "sat_min_elevation":   25.0,
    "sat_base_delay_s":    0.020,
    "sat_max_delay_s":     0.060,
    "sat_tx_power_w":      2.0,
    # Simulation
    "num_actions":         7,
    "obs_size":            48,
    "num_episodes":        600,
    "steps_per_episode":   200,
    "num_gnb":             6,
}

def _cross_check_with_cpp() -> None:
    """Verify env_config matches the C++ simulator ground truth."""
    checks = [
        # Reward Weights
        ("weight_deadline",       REWARD_CONFIG["weight_deadline"],         _CPP_GROUND_TRUTH["weight_deadline"]),
        ("weight_delay",          REWARD_CONFIG["weight_delay"],            _CPP_GROUND_TRUTH["weight_delay"]),
        ("weight_energy",         REWARD_CONFIG["weight_energy"],           _CPP_GROUND_TRUTH["weight_energy"]),
        ("weight_load",           REWARD_CONFIG["weight_load"],             _CPP_GROUND_TRUTH["weight_load"]),
        ("weight_penalty",        REWARD_CONFIG["weight_penalty"],          _CPP_GROUND_TRUTH["weight_penalty"]),
        ("weight_backhaul",       REWARD_CONFIG["weight_backhaul"],         _CPP_GROUND_TRUTH["weight_backhaul"]),
        ("reward_smooth_factor",  REWARD_CONFIG["smooth_factor"],           _CPP_GROUND_TRUTH["reward_smooth_factor"]),

        # RSU
        ("rsu_max_range_m",       RSU_CONFIG["max_range_m"],                _CPP_GROUND_TRUTH["rsu_max_range_m"]),
        ("rsu_freq_ghz",          RSU_CONFIG["frequency_ghz"],              _CPP_GROUND_TRUTH["rsu_freq_ghz"]),
        ("rsu_bandwidth_mhz",     RSU_CONFIG["bandwidth_mhz"],              _CPP_GROUND_TRUTH["rsu_bandwidth_mhz"]),
        ("rsu_backhaul_bps",      RSU_CONFIG["backhaul_bps"],               _CPP_GROUND_TRUTH["rsu_backhaul_bps"]),
        ("rsu_antenna_gain_dbi",  RSU_CONFIG["antenna_gain_dbi"],           _CPP_GROUND_TRUTH["rsu_antenna_gain_dbi"]),
        ("rsu_processing_efficiency", RSU_CONFIG["processing_efficiency"],  _CPP_GROUND_TRUTH["rsu_processing_efficiency"]),

        # MEC
        ("mec_vm_overhead",       MEC_CONFIG["vm_overhead"],                _CPP_GROUND_TRUTH["mec_vm_overhead"]),
        ("mec_freqs_ghz",         MEC_CONFIG["frequencies_ghz"],            _CPP_GROUND_TRUTH["mec_freqs_ghz"]),
        ("mec_power_idle_w",      MEC_CONFIG["power_idle_w"],               _CPP_GROUND_TRUTH["mec_power_idle_w"]),
        ("mec_power_max_w",       MEC_CONFIG["power_max_w"],                _CPP_GROUND_TRUTH["mec_power_max_w"]),
        ("mec_max_vms",           MEC_CONFIG["max_vms"],                    _CPP_GROUND_TRUTH["mec_max_vms"]),

        # Local
        ("local_freq_ghz",        SERVER_PARAMETERS["local"]["freq_ghz"],   _CPP_GROUND_TRUTH["local_freq_ghz"]),

        # Load & queue
        ("load_increment",        LOAD_BALANCING_CONFIG["load_increment"],  _CPP_GROUND_TRUTH["load_increment"]),
        ("load_decay_rate",       LOAD_BALANCING_CONFIG["load_decay_rate"], _CPP_GROUND_TRUTH["load_decay_rate"]),
        ("queue_decay_rate",      QUEUEING_CONFIG["decay_rate"],            _CPP_GROUND_TRUTH["queue_decay_rate"]),

        # Convergence
        ("convergence_window",    CONVERGENCE_CONFIG["window"],             _CPP_GROUND_TRUTH["convergence_window"]),
        ("convergence_threshold", CONVERGENCE_CONFIG["stddev_threshold"],   _CPP_GROUND_TRUTH["convergence_threshold"]),
        ("convergence_min_count", CONVERGENCE_CONFIG["consecutive_stable"], _CPP_GROUND_TRUTH["convergence_min_count"]),
        ("convergence_min_episodes", CONVERGENCE_CONFIG["min_episodes"],    _CPP_GROUND_TRUTH["convergence_min_episodes"]),

        # Satellite
        ("sat_orbit_height_km",   SATELLITE_CONFIG["orbit_height_km"],      _CPP_GROUND_TRUTH["sat_orbit_height_km"]),
        ("sat_min_elevation",     SATELLITE_CONFIG["elevation_min_deg"],    _CPP_GROUND_TRUTH["sat_min_elevation"]),
        ("sat_tx_power_w",        SATELLITE_CONFIG["tx_power_w"],           _CPP_GROUND_TRUTH["sat_tx_power_w"]),

        # Simulation
        ("num_actions",           NUM_ACTIONS,                              _CPP_GROUND_TRUTH["num_actions"]),
        ("obs_size",              OBS_SIZE,                                 _CPP_GROUND_TRUTH["obs_size"]),
        ("num_episodes",          NUM_EPISODES,                             _CPP_GROUND_TRUTH["num_episodes"]),
        ("steps_per_episode",     MAX_STEPS,                                _CPP_GROUND_TRUTH["steps_per_episode"]),
        ("rsu_bh_delay_ms",       RSU_CONFIG["backhaul_delay_ms"],          _CPP_GROUND_TRUTH["rsu_bh_delay_s"] * 1000.0),
        ("sat_base_delay_ms",     SATELLITE_CONFIG["base_delay_ms"],        _CPP_GROUND_TRUTH["sat_base_delay_s"] * 1000.0),
        ("sat_max_delay_ms",      SATELLITE_CONFIG["max_delay_ms"],         _CPP_GROUND_TRUTH["sat_max_delay_s"] * 1000.0),
        ("num_gnb",               len(GNB_HOME_MEC),                        _CPP_GROUND_TRUTH["num_gnb"]),
    ]

    mismatches = []
    for name, actual, expected in checks:
        if isinstance(expected, list):
            match = (actual == expected)
        else:
            try:
                match = abs(float(actual) - float(expected)) < 1e-9
            except (TypeError, ValueError):
                match = (actual == expected)
        if not match:
            mismatches.append(f"    {name}: python={actual}, cpp={expected}")

    if mismatches:
        print("\n⚠️ env_config.py MISMATCHES with C++ ground truth:")
        for m in mismatches:
            print(m)
        raise AssertionError(
            f"{len(mismatches)} env_config/C++ mismatches detected — "
            "fix before running experiments"
        )

# ==============================
# VALIDATION
# ==============================
def _validate_config() -> None:
    """Validate all configuration settings."""
    assert len(FEATURE_NAMES) == OBS_SIZE, \
        f"FEATURE_NAMES length ({len(FEATURE_NAMES)}) != OBS_SIZE ({OBS_SIZE})"

    assert len(FEATURE_CATEGORIES) > 0, "FEATURE_CATEGORIES is empty"
    all_categorized = set()
    for cat, indices in FEATURE_CATEGORIES.items():
        all_categorized.update(indices)
    for i in range(OBS_SIZE):
        assert i in all_categorized, f"Feature {i} not in any category"

    assert len(ACTION_NAMES) == NUM_ACTIONS, \
        f"ACTION_NAMES length ({len(ACTION_NAMES)}) != NUM_ACTIONS ({NUM_ACTIONS})"

    for action in range(NUM_ACTIONS):
        assert action in ACTION_LABELS, f"ACTION_LABELS missing action {action}"

    all_categorized_actions = set()
    for cat, actions in ACTION_CATEGORIES.items():
        all_categorized_actions.update(actions)
    for action in range(NUM_ACTIONS):
        assert action in all_categorized_actions, \
            f"Action {action} ({ACTION_NAMES[action]}) not in any ACTION_CATEGORY"

    for action in range(NUM_ACTIONS):
        assert action in ACTION_BACKHAUL_TYPE, \
            f"ACTION_BACKHAUL_TYPE missing action {action}"

    assert len(GNB_HOME_MEC) == 6, "GNB_HOME_MEC should have 6 entries"
    for gnb_idx in range(6):
        assert gnb_idx in GNB_HOME_MEC, f"GNB_HOME_MEC missing gNB {gnb_idx}"
        assert 0 <= GNB_HOME_MEC[gnb_idx] < 3, \
            f"GNB_HOME_MEC[{gnb_idx}] = {GNB_HOME_MEC[gnb_idx]} should be 0, 1, or 2"

    for i in range(19, 29):
        assert OBS_LOW[i] == -1.0, f"OBS_LOW[{i}] should be -1.0"
        assert OBS_HIGH[i] == 1.0, f"OBS_HIGH[{i}] should be 1.0"

    assert len(MEC_CONFIG["frequencies_ghz"]) == MEC_CONFIG["num_mec"], \
        "MEC frequencies count doesn't match num_mec"

    assert RSU_CONFIG["num_rsu"] == 2, "RSU config should have 2 RSUs"
    assert RSU_CONFIG["max_range_m"] == 300.0, \
        f"RSU max range should be 300m, got {RSU_CONFIG['max_range_m']}"
    assert RSU_CONFIG["frequency_ghz"] == 5.9, \
        f"RSU frequency should be 5.9 GHz, got {RSU_CONFIG['frequency_ghz']}"

    assert SATELLITE_CONFIG["orbit_height_km"] == 550.0, \
        f"Satellite orbit should be 550km, got {SATELLITE_CONFIG['orbit_height_km']}"

    reward_weights = [
        REWARD_CONFIG["weight_deadline"],
        REWARD_CONFIG["weight_delay"],
        REWARD_CONFIG["weight_energy"],
        REWARD_CONFIG["weight_load"],
        REWARD_CONFIG["weight_penalty"],
        REWARD_CONFIG["weight_backhaul"],
    ]
    weight_sum = sum(reward_weights)
    assert abs(weight_sum - 1.0) < 0.001, \
        f"Reward weights sum to {weight_sum}, should be 1.0"

    assert CONVERGENCE_CONFIG["stddev_threshold"] == 4.0, \
        f"Convergence threshold should be 4.0%, got {CONVERGENCE_CONFIG['stddev_threshold']}"

    assert USE_QUEUE_MODEL == True, \
        "USE_QUEUE_MODEL should be True (matches C++ kEnableQueueModel = true)"

_validate_config()
_cross_check_with_cpp()

# ==============================
# FINAL STATUS (verbose only if ENV_CONFIG_VERBOSE=1)
# ==============================
def _print_banner() -> None:
    print("=" * 70)
    print("✅ env_config.py [THESIS v8.6.12 - ALIGNED WITH C++] loaded successfully")
    print("=" * 70)
    print(f" VERSION             : {VERSION}")
    print(f" ENV_VERSION         : {ENV_VERSION}")
    print(f" EXPERIMENT_TAG      : {EXPERIMENT_TAG}")
    print(f" SCENARIO_NAME       : {SCENARIO_NAME}")
    print(f" OBS_SIZE            : {OBS_SIZE} (one-hot UE type)")
    print(f" Actions             : {NUM_ACTIONS} {ACTION_NAMES}")
    print(f" gNBs               : 6 (2 per MEC zone)")
    print()
    print(f" ✓ Load Increment    : {LOAD_BALANCING_CONFIG['load_increment']:.3f}")
    print(f" ✓ Load Decay Rate   : {LOAD_BALANCING_CONFIG['load_decay_rate']:.3f}")
    print(f" ✓ RSU Backhaul      : {RSU_CONFIG['backhaul_bps']:.1e} bps")
    print(f" ✓ RSU Range         : {RSU_CONFIG['max_range_m']}m")
    print(f" ✓ RSU Frequency     : {RSU_CONFIG['frequency_ghz']} GHz (channel)")
    print(f" ✓ Local Freq        : {SERVER_PARAMETERS['local']['freq_ghz']} GHz")
    _reward_sum = (
        REWARD_CONFIG['weight_deadline']
        + REWARD_CONFIG['weight_delay']
        + REWARD_CONFIG['weight_energy']
        + REWARD_CONFIG['weight_load']
        + REWARD_CONFIG['weight_penalty']
        + REWARD_CONFIG['weight_backhaul']
    )
    print(f" ✓ Reward Weights    : Sum = {_reward_sum:.2f} ✓")
    print(f" ✓ Convergence       : {CONVERGENCE_CONFIG['stddev_threshold']}%")
    print(f" ✓ Queue Model       : {USE_QUEUE_MODEL}")
    print()
    print("🆕 OPTIMIZED FEATURES:")
    print(f" RSU Model           : {RSU_CONFIG['num_rsu']} RSUs, "
          f"range={RSU_CONFIG['max_range_m']}m, "
          f"bandwidth={RSU_CONFIG['bandwidth_mhz']}MHz")
    print(f" Satellite Model     : LEO @ {SATELLITE_CONFIG['orbit_height_km']}km, "
          f"min elevation={SATELLITE_CONFIG['elevation_min_deg']}°")
    print(f" MEC Frequencies     : {MEC_CONFIG['frequencies_ghz']} GHz")
    print(f" VM Overhead         : {MEC_CONFIG['vm_overhead']*100}%")
    print()
    print("⚙️ REWARD CONFIGURATION:")
    print(f" Deadline Weight     : {REWARD_CONFIG['weight_deadline']:.2f}")
    print(f" Delay Weight        : {REWARD_CONFIG['weight_delay']:.2f}")
    print(f" Energy Weight       : {REWARD_CONFIG['weight_energy']:.2f}")
    print(f" Load Weight         : {REWARD_CONFIG['weight_load']:.2f}")
    print(f" Penalty Weight      : {REWARD_CONFIG['weight_penalty']:.2f}")
    print(f" Backhaul Weight     : {REWARD_CONFIG['weight_backhaul']:.2f}")
    print(f" Offload Bonus       : {REWARD_CONFIG['offload_bonus_enabled']}")
    print()
    print("📊 FEATURE CATEGORIES:")
    for cat, indices in FEATURE_CATEGORIES.items():
        print(f"  {cat:12s} : {len(indices)} features")
    print()
    print("🎯 MODE:")
    print(f"  Training Episodes  : {NUM_EPISODES}")
    print(f"  Evaluation Phases  : {EVAL_PHASE_EPISODES} (genuine held-out)")
    print(f"  Device             : {DEVICE_NAME}")
    print()
    print("✅ MATCHES C++ CODE: YES (mec_simulator_v8_robust.cc v8.6.12)")
    print("=" * 70)


if os.environ.get("ENV_CONFIG_VERBOSE", "0") == "1":
    _print_banner()

# ==============================
# EXPORT FOR USE IN OTHER MODULES
# ==============================
__all__ = [
    # Version
    'ENV_VERSION', 'EXPERIMENT_TAG', 'SCENARIO_NAME', 'VERSION', 'SIMULATOR_VERSION',
    'PROJECT_NAME', 'RANDOM_SEED', 'MULTI_SEEDS', 'EXCLUDE_SEEDS',

    # Paths
    'BASE_DIR', 'NS3_PATH', 'RESULTS_DIR',
    'DDQN_DIR', 'MODELS_DIR', 'DDQN_TRAIN_DIR', 'DDQN_EVAL_DIR',
    'DDQN_SUMMARY_DIR', 'DDQN_CSV_DIR',
    'DUELING_DIR', 'DUELING_MODELS_DIR', 'DUELING_TRAIN_DIR',
    'DUELING_EVAL_DIR', 'DUELING_SUMMARY_DIR', 'DUELING_CSV_DIR',
    'PPO_DIR', 'PPO_MODELS_DIR', 'PPO_TRAIN_DIR', 'PPO_EVAL_DIR',
    'PPO_SUMMARY_DIR', 'PPO_CSV_DIR',
    'RANDOM_DIR', 'RANDOM_EVAL_DIR', 'RANDOM_SUMMARY_DIR', 'RANDOM_CSV_DIR',
    'GREEDY_DIR', 'GREEDY_LOAD_EVAL_DIR', 'GREEDY_RATIO_EVAL_DIR',
    'GREEDY_LOAD_CSV_DIR', 'GREEDY_RATIO_CSV_DIR',
    'GREEDY_LOAD_SUMMARY_DIR', 'GREEDY_RATIO_SUMMARY_DIR', 'GREEDY_SUMMARY_DIR',
    'PLOTS_DIR', 'LOGS_DIR', 'COMPARISON_DIR', 'EVALUATION_DIR',

    # Experiment controls
    'USE_MOBILITY', 'USE_WIRELESS_REALISM', 'USE_QUEUE_MODEL',
    'USE_HANDOVER', 'USE_BACKHAUL', 'USE_PRIORITY_SCHEDULING',
    'USE_LOAD_BALANCING_METRICS', 'USE_RSU_MODEL', 'USE_SATELLITE_MODEL',

    # Configs
    'RSU_CONFIG', 'MEC_CONFIG', 'SATELLITE_CONFIG', 'TASK_RANGES',
    'QUEUEING_CONFIG', 'LOAD_BALANCING_CONFIG', 'CONVERGENCE_CONFIG',
    'REWARD_CONFIG',

    # Observation
    'OBS_SIZE', 'FEATURE_NAMES', 'FEATURE_CATEGORIES', 'FEATURE_DESCRIPTIONS',
    'OBS_LOW', 'OBS_HIGH', 'validate_observation',

    # Actions
    'NUM_ACTIONS', 'ACTION_NAMES', 'ACTION_LABELS', 'ACTION_CATEGORIES',
    'ACTION_BACKHAUL_TYPE', 'ACTION_SERVER_TYPES',

    # Servers
    'SERVER_PARAMETERS',
    'GNB_LABELS', 'GNB_HOME_MEC', 'GNB_POSITIONS',
    'MEC_POSITIONS', 'RSU_POSITIONS',
    'get_home_mec', 'is_cross_zone_backhaul', 'get_backhaul_type',
    'action_uses_gnb', 'action_uses_rsu', 'action_uses_backhaul',

    # Training
    'NUM_EPISODES', 'MAX_STEPS', 'PORT', 'STEP_TIME', 'STEP_TIMEOUT',
    'EVAL_EPISODES', 'EVAL_SEEDS', 'EVAL_PHASE_EPISODES',

    # RL
    'GAMMA', 'LR', 'BATCH_SIZE', 'MEMORY_SIZE', 'MIN_REPLAY_SIZE',
    'TAU', 'TARGET_UPDATE_FREQ', 'REWARD_CLIP', 'SAVE_EVERY',
    'EPS_START', 'EPS_END', 'EPS_DECAY', 'EPS_DECAY_STEPS',

    # Ablation
    'ABLATION_CONFIGS', 'METRICS_TO_LOG',

    # Hardware
    'USE_CUDA', 'DEVICE_NAME', 'TORCH_DETERMINISTIC',

    # Labels
    'SERVER_LABELS', 'UE_LABEL', 'UE_LABEL_ONEHOT', 'BACKHAUL_LABELS',
    'PRIORITY_LABELS',

    # Colors
    'ALGORITHM_COLORS', 'UE_TYPE_COLORS', 'SERVER_COLORS', 'QUEUEING_COLORS',
]