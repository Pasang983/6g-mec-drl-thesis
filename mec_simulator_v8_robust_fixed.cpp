/*
 * mec_simulator_v8_robust.cc
 *
 * 6G MEC Task Offloading Simulator (Thesis Version 8.6.17)
 * - v8.5.0: Fixed CPU-cycle computation bug (was 3x over-estimate for LOCAL)
 * - v8.5.0: Removed dead code (functions superseded by inline blocks)
 * - v8.6.0: Satellite action now uses gSatDelayCache_s (physical model)
 * - v8.6.0: RSU Doppler normalizer made consistent with DopplerNorm()
 * - v8.6.1: Removed dead CounterSnapshot and no-op phase-toggle placeholder
 * - v8.6.2: Fixed mobility timestep mismatch (was 0.2s at 0.1s step)
 * - v8.6.2: Fixed penalty sign in reward (was a shrinking bonus, not a penalty)
 * - v8.6.2: Force step-cache redraw on first step of each episode
 * - v8.6.2: MEC/RSU SINR fading frozen per step (was redrawn per call)
 * - v8.6.2: Observation-size assert; raw-reward CSV logging
 * - v8.6.4: Dampened load increment (0.08 → 0.03) to remove DL oscillation
 * - v8.6.4: Widened deadline bands to give a broad DL distribution
 * - v8.6.4: Suppressed C++ inline log in eval mode
 * - v8.6.4: Enabled running reward normalization
 * - v8.6.4: Reduced HO TTT (10 → 5) and bumped UE speeds
 * - v8.6.4: Python now computes episode_dl from its own per-step counters
 * - v8.6.5: Widened initial-load spread (MEC 0.15–0.70, RSU 0.10–0.55)
 * - v8.6.5: HO constants moved to file scope; banner reports actual TTT
 * - v8.6.6: Damped load further (decay 0.995, increment 0.015)
 * - v8.6.6: Wider background-load damping
 * - v8.6.7: Eval complexity drawn from wide uniform band [0.25, 0.75]
 * - v8.6.8: Bumped eMBB speed to 15..30 m/s, V2X to 40..60 m/s
 * - v8.6.8: Raised kMaxSpeed to 60 m/s
 * - v8.6.8: HO TTT reduced to 3 steps
 * - v8.6.9: regenerate the task each step (was once per episode).
 *           Fixes the bimodal DL distribution (0% or 100% per episode)
 *           that was masking true policy performance.
 * - v8.6.10: freeze reward normalization in eval mode (no drift over
 *            the course of an evaluation run).
 * - v8.6.11: narrow initial-load spread
 * - v8.6.12: redraw UE type and speed every step (was once per episode).
 *            Fixes the bimodal DL distribution caused by IoT episodes
 *            always succeeding and eMBB/V2X episodes always failing.
  * - v8.6.17: satellite load model (shared capacity). Fixes the SAT
 *            loophole where greedy-ratio got 99% DL by picking SAT 74%
 *            of the time. SAT delay now grows with usage.
 * - v8.6.17: reward weight rebalance (DL 0.70). Aligns reward with DL%
 *            so high-DL policies also get high reward.
 *  
 * Author: Pasang Lama — IST Lisboa
 */

#include "ns3/core-module.h"
#include "ns3/opengym-module.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <iomanip>
#include <fstream>
#include <memory>
#include <atomic>
#include <mutex>
#include <deque>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("MecSimulatorV8Fixed");

// ============================================================
//  CONSTANTS - ALIGNED WITH Python env_config.py
// ============================================================
static constexpr int    kObsSize          = 48;
static constexpr int    kNumActions       = 7;
static constexpr int    kMaxEpisodes      = 600;
static constexpr int    kStepsPerEpisode  = 200;
static constexpr int    kNumGnb           = 6;

// ---- Simulation timestep (single source of truth) ----
static constexpr double kStepInterval_s   = 0.1;

// Action indices
static constexpr int    kActLocal         = 0;
static constexpr int    kActMec0          = 1;
static constexpr int    kActMec1          = 2;
static constexpr int    kActMec2          = 3;
static constexpr int    kActRsu0          = 4;
static constexpr int    kActRsu1          = 5;
static constexpr int    kActSat           = 6;

// gNB-to-MEC assignment
static constexpr int    kGnbHomeMec[kNumGnb] = {0, 0, 1, 1, 2, 2};

// ---- Physical parameters ----
static constexpr double kAreaSize         = 1000.0;
static constexpr double kMaxSpeed         = 60.0;
static constexpr double kMaxQueue         = 50.0;

// ---- Handover parameters (file scope so the banner can report them) ----
static constexpr double kHandoverHysteresis_dB = 0.5;   // dB
static constexpr int    kHandoverTTTSteps      = 3;     // steps (0.5 s at 0.1 s step)

// ---- CPU frequencies ----
static constexpr double kMecFreq0         = 30.0e9;
static constexpr double kMecFreq1         = 25.0e9;
static constexpr double kMecFreq2         = 20.0e9;
static constexpr double kRsuFreq          = 5.0e9;
static constexpr double kLocalFreq        = 8.0e9;

// ---- MEC Virtualization ----
static constexpr double kMecVmOverhead    = 0.15;
static constexpr double kMecMaxVms        = 16.0;

// ---- Air interface ----
static constexpr double kAirFreq_GHz      = 28.0;
static constexpr double kAirBandwidth_MHz = 400.0;
static constexpr double kAirTxPower_dBm   = 30.0;
static constexpr double kAirNoise_dBm     = -90.0;

// ---- Backhaul ----
static constexpr double kBhFiber_s        = 0.0001;
static constexpr double kBhCross_s        = 0.0020;
static constexpr double kBhCapacity_bps   = 100.0e9;
static constexpr double kBhCrossCapacity  = 20.0e9;
static constexpr double kMaxBhDelay_norm  = 0.050;
static constexpr double kMaxPathDelay_s   = 0.100;

// ============================================================
//  RSU PARAMETERS
// ============================================================
static constexpr double kRsuMaxRange_m         = 300.0;
static constexpr double kRsuFreq_GHz           = 5.9;
static constexpr double kRsuBandwidth_MHz      = 80.0;
static constexpr double kRsuBackhaul_bps       = 5.0e9;
static constexpr double kRsuBhDelay_s          = 0.002;
static constexpr double kRsuAntennaGain_dBi    = 12.0;
static constexpr double kRsuProcessingEfficiency = 0.7;

// ---- Channel ----
static constexpr double kFc_MEC_GHz       = 3.5;
static constexpr double kFc_RSU_GHz       = 5.9;
static constexpr double kDLoS_MEC_m       = 200.0;
static constexpr double kDLoS_RSU_m       = 100.0;
static constexpr double kRicianK_dB       = 3.0;

// ---- Satellite Parameters ----
static constexpr double kSatOrbitHeight_km = 550.0;
static constexpr double kSatLightSpeed     = 3.0e8;
static constexpr double kSatBaseDelay_s    = 0.020;
static constexpr double kSatMaxDelay_s     = 0.060;

// ---- Doppler ----
static constexpr double kDopplerAlpha      = 0.30;
static constexpr double kDopplerLossMax_dB = 3.0;
static constexpr double kSpeedLight        = 3.0e8;

// ---- RF ----
static constexpr double kTxPower_dBm       = 23.0;
static constexpr double kNoisePower_dBm    = -94.0;
static constexpr int    kNumInterferers    = 4;

// ---- SINR ----
static constexpr double kSINR_Min_dB       = -10.0;
static constexpr double kSINR_Max_dB       = 35.0;

// ---- Energy ----
static constexpr double kUeTxPower_W       = 0.1;
static constexpr double kMecPowerIdle_W    = 20.0;
static constexpr double kMecPowerMax_W     = 80.0;
static constexpr double kRsuPowerIdle_W    = 15.0;
static constexpr double kRsuPowerMax_W     = 35.0;
static constexpr double kEnergyPerCycle_J  = 1.0e-12;

// ============================================================
//  REWARD WEIGHTS - SUM TO 1.0
// ============================================================
static constexpr double kWeightDeadline   = 0.55;   // was 0.40
static constexpr double kWeightDelay      = 0.10;   // was 0.15
static constexpr double kWeightEnergy     = 0.05;   // was 0.10
static constexpr double kWeightLoad       = 0.10;   // unchanged
static constexpr double kWeightPenalty    = 0.05;   // was 0.10
static constexpr double kWeightBackhaul   = 0.15;   // unchanged
// Sum = 1.00 ✓

// ============================================================
//  REWARD NORMALIZATION
// ============================================================
// Fixed normalization (v8.6.1 behaviour). Re-calibrate after reward-sign fix.
static constexpr double   kFixedRewardMean = 0.5;
static constexpr double   kFixedRewardStd  = 1.2;

// Running normalization (optional). Set to true to enable Welford-style
// running mean/var for the reward, which is more robust to the reward
// distribution change introduced by the penalty-sign fix in v8.6.2.
static constexpr bool     kUseRunningNormalization = true;   // v8.6.4: reward scale changed
static constexpr double   kRunningNormWarmup       = 500.0;  // steps
static constexpr double   kRunningNormEps          = 1e-4;

// ============================================================
//  CONVERGENCE DETECTION
// ============================================================
static constexpr int    kConvergenceWindow    = 50;
static constexpr double kConvergenceThreshold = 4.0;
static constexpr int    kConvergenceMinCount  = 10;

// ============================================================
//  QUEUE MODEL
// ============================================================
static constexpr bool   kEnableQueueModel = true;
static constexpr double kQueueDecayRate   = 0.97;
static constexpr double kLoadDecayRate    = 0.995;
static constexpr double kLoadIncrement    = 0.015;

// ---- Background traffic (other UEs in same MEC zone) ----
static constexpr double kBackgroundLoadProb   = 0.10;
static constexpr double kBackgroundLoadAmount = 0.008;

// ---- Reward Smoothing ----
static constexpr double kRewardSmoothFactor = 0.4;

// ---- Evaluation Phase ----
static bool gIsTraining = true;

// ============================================================
//  COMPILE-TIME VALIDATION
// ============================================================
static_assert(kWeightDeadline + kWeightDelay + kWeightEnergy +
              kWeightLoad + kWeightPenalty + kWeightBackhaul == 1.0,
              "Reward weights must sum to 1.0");

static_assert(kRsuMaxRange_m == 300.0, "RSU max range must be 300m");
static_assert(kRsuFreq_GHz == 5.9, "RSU frequency must be 5.9 GHz");
static_assert(kConvergenceThreshold == 4.0, "Convergence threshold must be 4.0%");
static_assert(kEnableQueueModel == true, "Queue model must be enabled");
static_assert(kObsSize == 48, "If you change kObsSize, update BuildObservation");
static_assert(kStepInterval_s == 0.1, "Step interval must match Simulator::Schedule");

// ============================================================
//  SAFE MATH FUNCTIONS
// ============================================================
inline double Clamp(double v, double lo, double hi) {
    if (!std::isfinite(v)) return lo;
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

inline double SafeDiv(double a, double b) {
    const double eps = std::numeric_limits<double>::epsilon() * 100.0;
    if (std::abs(b) < eps) {
        // Return 0 for physically-bounded quantities (doppler norm, etc.)
        // rather than sign(a), which would be a spurious maximum.
        return 0.0;
    }
    double result = a / b;
    if (!std::isfinite(result)) return 0.0;
    return result;
}

inline double SafeNorm(double val, double max_val) {
    if (!std::isfinite(val) || max_val <= 0.0) return 0.0;
    if (max_val < std::numeric_limits<double>::epsilon()) return 0.0;
    return Clamp(val / max_val, 0.0, 1.0);
}

inline double SafeLog2(double val) {
    if (val <= 0.0) return 0.0;
    double result = std::log2(val);
    if (!std::isfinite(result)) return 0.0;
    return result;
}

inline double SafeLog10(double val) {
    if (val <= 0.0) return 0.0;
    double result = std::log10(val);
    if (!std::isfinite(result)) return 0.0;
    return result;
}

// ============================================================
//  RANDOM NUMBER GENERATION
// ============================================================
class RobustRNG {
private:
    std::mt19937_64 m_generator;
    std::uniform_real_distribution<double> m_uniform;
    mutable std::mutex m_mutex;

public:
    explicit RobustRNG(uint64_t seed)
        : m_generator(seed), m_uniform(0.0, 1.0) {}

    double Uniform() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_uniform(m_generator);
    }

    void Seed(uint64_t seed) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_generator.seed(seed);
        m_uniform.reset();
    }
};

static std::unique_ptr<RobustRNG> gRng;

inline double RandUnifRobust() {
    return gRng->Uniform();
}

// ============================================================
//  CACHE MANAGEMENT
// ============================================================
struct ActionCache {
    double delays[kNumActions];
    double energies[kNumActions];
    double air[kNumActions];
    double bh[kNumActions];
    double comp[kNumActions];
    double queue[kNumActions];
    uint64_t version;
    bool valid;

    ActionCache() : version(0), valid(false) {
        for (int i = 0; i < kNumActions; i++) {
            delays[i] = 0.0;
            energies[i] = 0.0;
            air[i] = 0.0;
            bh[i] = 0.0;
            comp[i] = 0.0;
            queue[i] = 0.0;
        }
    }

    void Invalidate() {
        valid = false;
        version++;
    }
};

static ActionCache gActionCache;
static double gCachedGnbSINR[kNumGnb];
static bool gSINRCacheValid = false;
static uint64_t gSINRVersion = 0;

// ============================================================
//  REWARD STATISTICS
// ============================================================
static struct {
    uint64_t count = 0;
} gRewardStats;

// Running reward normalization state
static struct {
    double mean = 0.0;
    double m2   = 0.0;   // Welford accumulator
    uint64_t n  = 0;
} gRewardNorm;

// ============================================================
//  REWARD COMPONENT ACCUMULATORS (diagnostic)
// ============================================================
static double gRwdCompDeadline    = 0.0;
static double gRwdCompDelay       = 0.0;
static double gRwdCompEnergy      = 0.0;
static double gRwdCompLoad        = 0.0;
static double gRwdCompPenalty     = 0.0;
static double gRwdCompBackhaul    = 0.0;
static double gRwdCompOffload     = 0.0;
static double gRwdCompConsistency = 0.0;
static int    gRwdCompCount       = 0;

// ============================================================
//  STRUCTS
// ============================================================
struct MecServer {
    double x, y, freq, load, queue;
    double sinr_dB, los_prob;
    double power;
    double vm_overhead;
};

struct RsuNode {
    double x, y, freq, load, queue;
    double sinr_dB, los_prob;
    double range;
    double antenna_gain;
};

struct GnbNode {
    double x, y;
    double load;
    double bh_delay[3];
    double bh_capacity[3];
    double sinr_to_ue_dB;
};

struct Interferer { double x, y; };

struct UeState {
    double x, y, vx, vy, speed;
    int    ue_type;
    double cpu_cycles_giga;
    double data_size_mb, deadline;
    double fading, doppler_norm;
    int    serving_gnb;
    bool   handover;
    double air_sinr_dB;
    double air_tx_delay_s;
    int    stable_connection;
    double completion_reward;
};

// ============================================================
//  GLOBALS
// ============================================================
static MecServer   gMec[3];
static RsuNode     gRsu[2];
static GnbNode     gGnb[kNumGnb];
static Interferer  gInterferers[kNumInterferers];
static UeState     gUe;

static int     gEpisodeCount = 0;
static int     gStepCount    = 0;
static int     gDeadlinesMet = 0;
static int     gTotalSteps   = 0;
static float   gLastReward   = 0.0f;
static double  gLastRawReward = 0.0;
static double  gSmoothedReward = 0.0;
static int     gLastAction = -1;
static int     gActionConsistencyCount = 0;

static bool gEpisodeJustEnded = false;

// ---- MEC cold-start state ----
static int gStepsSinceMecUse[3] = {999, 999, 999};
static constexpr int kMecWarmSteps = 10;

// v8.6.17: satellite load (shared capacity model).
// Prevents the SAT loophole where SAT has a fixed 20-60ms delay
// regardless of usage, making it DL-optimal even when oversubscribed.
static double gSatLoad = 0.0;
static constexpr double kSatLoadDecayRate     = 0.98;
static constexpr double kSatLoadIncrement     = 0.03;
static constexpr double kSatCongestionMax_s   = 0.100;  // +100 ms at load=1.0
static constexpr double kSatLoadMax           = 1.0;

// Step stamp: -1 means "cache is empty, redraw on next access"
static int    gCacheStepStamp       = -1;

// MEC cold-start delay for this step
static double gMecColdStartDelay[3] = {0.0, 0.0, 0.0};

// Satellite delay realization for this step
static double gSatDelayCache_s      = 0.0;

// Air TX delay realization for this step
static double gAirTxDelayCache_s    = 0.0;

// Interference realization per MEC/RSU position for this step
static double gInterfCache_W[3]     = {0.0, 0.0, 0.0};  // indexed by MEC
static double gInterfRsuCache_W[2]  = {0.0, 0.0};       // indexed by RSU

// MEC/RSU SINR realization for this step (frozen, consistent across calls)
static double gMecSinrCache_dB[3]   = {0.0, 0.0, 0.0};
static double gMecLosProbCache[3]   = {0.0, 0.0, 0.0};
static double gRsuSinrCache_dB[2]   = {0.0, 0.0};
static double gRsuLosProbCache[2]   = {0.0, 0.0};
static bool   gSinrCachePopulated   = false;

// Helper: ensure caches are fresh for the current step
static void RefreshStepCachesIfNeeded();

// v8.6.9: task regeneration per step
static void GenerateNewTask();

// Aggregate metric counters
static double gTotalBackhaulDelay = 0.0;
static int    gHomeBackhaulCount  = 0;
static int    gCrossBackhaulCount = 0;

static double gTotalMecBackhaulDelay = 0.0;
static int    gMecBackhaulCount      = 0;
static double gTotalRsuBackhaulDelay = 0.0;
static int    gRsuBackhaulCount      = 0;
static double gTotalSatBackhaulDelay = 0.0;
static int    gSatBackhaulCount      = 0;

static int    gHandoverCount = 0;
static int    gActionCount[7] = {0, 0, 0, 0, 0, 0, 0};
static int    gActionAttempts[7]     = {0, 0, 0, 0, 0, 0, 0};
static int    gActionDeadlinesMet[7] = {0, 0, 0, 0, 0, 0, 0};
static double gTotalGnbLoad = 0.0;
static double gTotalMecUtil = 0.0;
static double gTotalDelay = 0.0;
static double gTotalEnergy = 0.0;
static double gTotalAirDelay = 0.0;
static double gTotalCompDelay = 0.0;
static double gTotalQueueDelay = 0.0;
static double gTotalGnbServing = 0.0;
static double gTotalJainsFairness = 0.0;

// RSU-specific metrics
static struct RsuMetrics {
    double total_rsu_utilization = 0.0;
    double total_rsu_coverage_time = 0.0;
    double total_rsu_handovers = 0.0;
    double total_rsu_offload_success = 0.0;
    double rsu_offload_attempts = 0.0;
    double rsu_avg_sinr = 0.0;
    int rsu_sinr_samples = 0;
} gRsuMetrics;

static uint32_t gSeed = 1;
static uint16_t gPort = 5555;

// ============================================================
//  CONVERGENCE TRACKING
// ============================================================
static double gDLHistory[50];
static int gDLHistoryIdx = 0;
static int gDLHistoryCount = 0;
static int gConvergenceCount = 0;
static bool gConverged = false;
static bool gConvergenceLogged = false;
static int gConvergenceEpisode = -1;

// ============================================================
//  UTILITY FUNCTIONS
// ============================================================
inline double EuclidDist(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    double result = std::sqrt(dx * dx + dy * dy);
    if (!std::isfinite(result)) return 0.0;
    return result;
}

inline double dBm_to_W(double dBm) {
    if (!std::isfinite(dBm) || dBm < -100.0) return 1e-12;
    if (dBm > 100.0) return 1e10;
    double result = std::pow(10.0, (dBm - 30.0) / 10.0);
    if (!std::isfinite(result)) return 1e-12;
    return result;
}

static double NormSINR(double sinr_dB, double lo = kSINR_Min_dB, double hi = kSINR_Max_dB) {
    if (!std::isfinite(sinr_dB)) return 0.5;
    sinr_dB = Clamp(sinr_dB, lo - 10.0, hi + 10.0);
    double result = (sinr_dB - lo) / (hi - lo);
    if (!std::isfinite(result)) return 0.5;
    return Clamp(result, 0.0, 1.0);
}

// ============================================================
//  LOAD BALANCING
// ============================================================
static double ComputeJainsFairnessIndex() {
    double loads[5] = {
        gMec[0].load,
        gMec[1].load,
        gMec[2].load,
        gRsu[0].load * 0.5,
        gRsu[1].load * 0.5
    };

    double sum = 0.0, sum_sq = 0.0;
    int n = 5;
    for (int i = 0; i < n; i++) {
        double load = Clamp(loads[i], 0.01, 1.0);
        sum += load;
        sum_sq += load * load;
    }

    if (sum_sq < 1e-12 || sum < 1e-12) return 0.0;

    double result = (sum * sum) / (n * sum_sq);
    return Clamp(result, 0.0, 1.0);
}

// ============================================================
//  FADING FUNCTIONS
// ============================================================
static double SampleRayleighFading() {
    double exp_val = Clamp(RandUnifRobust(), 1e-10, 0.9999999999);
    double result = 1.0 - std::exp(-exp_val);
    if (!std::isfinite(result)) return 0.5;
    return Clamp(result, 0.0, 1.0);
}

static double SampleRicianFading(double K_linear) {
    if (!std::isfinite(K_linear) || K_linear < 0.0) K_linear = 1.0;
    K_linear = Clamp(K_linear, 0.0, 1000.0);
    double los = K_linear / (K_linear + 1.0);
    double sc = 1.0 / (K_linear + 1.0);
    double exp_val = Clamp(RandUnifRobust(), 1e-10, 0.9999999999);
    double result = (los + sc * std::exp(-exp_val)) / (1.0 + los);
    if (!std::isfinite(result)) return 0.5;
    return Clamp(result, 0.0, 1.0);
}

// ============================================================
//  PATH LOSS MODELS
// ============================================================
static double PathLoss_UMa(double d_m, double fc_GHz, bool los) {
    d_m = std::max(d_m, 1.0);
    if (!std::isfinite(d_m) || !std::isfinite(fc_GHz)) return 100.0;
    fc_GHz = Clamp(fc_GHz, 0.1, 100.0);
    double result;
    if (los) {
        result = 28.0 + 22.0 * SafeLog10(d_m) + 20.0 * SafeLog10(fc_GHz);
    } else {
        result = 13.54 + 39.08 * SafeLog10(d_m) + 20.0 * SafeLog10(fc_GHz);
    }
    if (!std::isfinite(result)) return 100.0;
    return Clamp(result, 10.0, 200.0);
}

static double PathLoss_V2X(double d_m, double fc_GHz, bool los) {
    d_m = std::max(d_m, 1.0);
    fc_GHz = Clamp(fc_GHz, 0.1, 100.0);

    double result;
    if (los) {
        result = 32.4 + 21.0 * SafeLog10(d_m) + 20.0 * SafeLog10(fc_GHz);
        result += 0.5 * (1.5 - 1.0);
    } else {
        result = 35.3 + 37.6 * SafeLog10(d_m) + 21.3 * SafeLog10(fc_GHz);
        result += 12.0;
    }

    if (!std::isfinite(result)) return 100.0;
    return Clamp(result, 10.0, 200.0);
}

static double PathLoss_mmWave(double d_m) {
    d_m = std::max(d_m, 1.0);
    if (!std::isfinite(d_m)) return 100.0;
    const double lambda = kSpeedLight / (kAirFreq_GHz * 1.0e9);
    const double fspl0 = 20.0 * SafeLog10(4.0 * M_PI / lambda);
    double result = fspl0 + 21.0 * SafeLog10(d_m);
    if (!std::isfinite(result)) return 100.0;
    return Clamp(result, 10.0, 200.0);
}

static double LoSProb(double d_m, double d0) {
    if (d_m < 0.0) return 1.0;
    if (!std::isfinite(d_m) || !std::isfinite(d0) || d0 <= 0.0) return 1.0;
    double result = std::exp(-d_m / d0);
    if (!std::isfinite(result)) return 1.0;
    return Clamp(result, 0.0, 1.0);
}

// ============================================================
//  INTERFERENCE AND SINR
// ============================================================
static double Interference_W(double rx_x, double rx_y, double fc_GHz) {
    double I_W = 0.0;
    for (int i = 0; i < kNumInterferers; i++) {
        double d = EuclidDist(rx_x, rx_y, gInterferers[i].x, gInterferers[i].y);
        double pl = PathLoss_UMa(std::max(d, 10.0), fc_GHz, false);
        double fad = SampleRayleighFading();
        double power = dBm_to_W(46.0 - pl + 10.0 * SafeLog10(fad + 1e-9));
        if (std::isfinite(power)) I_W += power;
    }
    if (!std::isfinite(I_W)) return 1e-12;
    return Clamp(I_W, 0.0, 1e-3);
}

// NOTE: ComputeSINR_MEC / ComputeSINR_RSU are now pure cache readers.
// Their realizations are populated once per step in RefreshStepCachesIfNeeded.
static double ComputeSINR_MEC(int mec_idx, double& out_los_prob) {
    if (gCacheStepStamp != gStepCount) RefreshStepCachesIfNeeded();
    if (mec_idx < 0 || mec_idx >= 3) {
        out_los_prob = 0.0;
        return 0.0;
    }
    out_los_prob = gMecLosProbCache[mec_idx];
    return gMecSinrCache_dB[mec_idx];
}

static double ComputeSINR_RSU(int rsu_idx, double* out_los_prob = nullptr) {
    if (gCacheStepStamp != gStepCount) RefreshStepCachesIfNeeded();
    if (rsu_idx < 0 || rsu_idx >= 2) {
        if (out_los_prob) *out_los_prob = 0.0;
        return -30.0;
    }
    if (out_los_prob) *out_los_prob = gRsuLosProbCache[rsu_idx];
    return gRsuSinrCache_dB[rsu_idx];
}

// ============================================================
//  CAPACITY FUNCTION
// ============================================================
static double Capacity_bps(double sinr_dB, double bw_MHz) {
    double sinr_linear = std::pow(10.0, sinr_dB / 10.0);
    if (!std::isfinite(sinr_linear) || sinr_linear < 1e-6) {
        sinr_linear = 1e-6;
    }
    sinr_linear = Clamp(sinr_linear, 1e-6, 1e9);
    double result = (bw_MHz * 1.0e6) * SafeLog2(1.0 + sinr_linear);
    if (!std::isfinite(result) || result < 0.0) return 1.0;
    return result;
}

static double DopplerNorm(double fc_GHz) {
    if (!std::isfinite(fc_GHz) || fc_GHz <= 0.0) return 0.0;
    const double fd = (gUe.speed / kSpeedLight) * fc_GHz * 1.0e9;
    const double fd_max = (kMaxSpeed / kSpeedLight) * fc_GHz * 1.0e9;
    double result = SafeDiv(fd, fd_max);
    if (!std::isfinite(result)) return 0.0;
    return Clamp(result, 0.0, 1.0);
}

// ============================================================
//  COMPUTE ALL gNB SINR
// ============================================================
static void ComputeAllGnbSINR() {
    for (int k = 0; k < kNumGnb; k++) {
        const double d = EuclidDist(gUe.x, gUe.y, gGnb[k].x, gGnb[k].y);
        const double pl = PathLoss_mmWave(d);
        const double K = std::pow(10.0, kRicianK_dB / 10.0);
        const double fad = SampleRicianFading(K);
        const double rx = kAirTxPower_dBm - pl + 10.0 * SafeLog10(fad + 1e-9);
        const double I_W = dBm_to_W(-90.0);
        const double N_W = dBm_to_W(kAirNoise_dBm);
        double sinr = 10.0 * SafeLog10(dBm_to_W(rx) / std::max(I_W + N_W, 1e-20));

        if (std::isfinite(sinr) && std::isfinite(gUe.doppler_norm)) {
            sinr -= kDopplerAlpha * gUe.doppler_norm * kDopplerLossMax_dB;
        }

        gCachedGnbSINR[k] = Clamp(sinr, -20.0, 50.0);
    }
    gSINRCacheValid = true;
    gSINRVersion++;
}

static int SelectServingGnb() {
    if (!gSINRCacheValid) ComputeAllGnbSINR();

    int best_gnb = 0;
    double best_sinr = -1000.0;
    for (int k = 0; k < kNumGnb; k++) {
        if (gCachedGnbSINR[k] > best_sinr) {
            best_sinr = gCachedGnbSINR[k];
            best_gnb = k;
        }
    }
    return best_gnb;
}

static double ComputeAirSINR(int gnb_idx) {
    if (gnb_idx >= 0 && gnb_idx < kNumGnb) {
        if (!gSINRCacheValid) ComputeAllGnbSINR();
        return gCachedGnbSINR[gnb_idx];
    }
    return 0.0;
}

// ============================================================
//  STEP CACHE REFRESH
// ============================================================
static void RefreshStepCachesIfNeeded() {
    if (gCacheStepStamp == gStepCount) return;

    NS_ASSERT_MSG(std::isfinite(gUe.x) && std::isfinite(gUe.y),
                  "RefreshStepCachesIfNeeded called before UE position set");

    // --- 1. MEC cold-start decisions (one draw per MEC per step) ---
    for (int m = 0; m < 3; m++) {
        const bool vm_warm = (gStepsSinceMecUse[m] < kMecWarmSteps);
        if (!vm_warm && RandUnifRobust() < 0.15) {
            gMecColdStartDelay[m] = 0.080 + RandUnifRobust() * 0.070;
        } else {
            gMecColdStartDelay[m] = 0.0;
        }
    }

    // --- 2. Satellite delay (elevation + processing + doppler jitter) ---
    {
        double elevation_angle = 25.0 + RandUnifRobust() * 45.0;
        elevation_angle = std::max(elevation_angle, 20.0);

        double earth_radius = 6371.0;
        double orbit_radius = earth_radius + kSatOrbitHeight_km;
        double slant_range = std::sqrt(
            orbit_radius * orbit_radius +
            earth_radius * earth_radius -
            2 * orbit_radius * earth_radius *
            std::cos((90.0 - elevation_angle) * M_PI / 180.0));

        double prop_delay    = slant_range * 1000.0 / kSatLightSpeed;
        double proc_delay    = 0.015 + RandUnifRobust() * 0.025;
        double doppler_delay = kDopplerAlpha * 0.010 * (gUe.speed / kMaxSpeed);

        // v8.6.17: satellite congestion delay (shared capacity model).
        // As gSatLoad grows, SAT becomes slower, mimicking the fact
        // that a single LEO satellite serves thousands of UEs.
        double congestion_delay = gSatLoad * kSatCongestionMax_s;

        gSatDelayCache_s = Clamp(prop_delay + proc_delay + doppler_delay
                                 + congestion_delay,
                                 kSatBaseDelay_s,
                                 kSatMaxDelay_s + kSatCongestionMax_s);
    }

    // --- 3. Interference realization (frozen per step) ---
    for (int m = 0; m < 3; m++) {
        gInterfCache_W[m] = Interference_W(gMec[m].x, gMec[m].y, kFc_MEC_GHz);
    }
    for (int r = 0; r < 2; r++) {
        gInterfRsuCache_W[r] = Interference_W(gRsu[r].x, gRsu[r].y, kFc_RSU_GHz);
    }

    // --- 4. MEC and RSU SINR realizations (frozen per step) ---
    //    Fading and LoS draws are performed ONCE here so that any later
    //    call to ComputeSINR_MEC/RSU within the same step returns the
    //    same value (prevents intra-step SINR flicker).
    for (int m = 0; m < 3; m++) {
        const double d      = EuclidDist(gUe.x, gUe.y, gMec[m].x, gMec[m].y);
        const double los_p  = LoSProb(d, kDLoS_MEC_m);
        const bool   is_LoS = (RandUnifRobust() < los_p);
        const double K_lin  = std::pow(10.0, kRicianK_dB / 10.0);

        const double pl  = PathLoss_UMa(d, kFc_MEC_GHz, is_LoS);
        const double fad = is_LoS ? SampleRicianFading(K_lin)
                                  : SampleRayleighFading();
        const double rx_dBm = kTxPower_dBm - pl + 10.0 * SafeLog10(fad + 1e-9);

        const double I_W = gInterfCache_W[m];
        const double N_W = dBm_to_W(kNoisePower_dBm);
        double sinr = 10.0 * SafeLog10(dBm_to_W(rx_dBm) /
                                       std::max(I_W + N_W, 1e-20));
        gMecSinrCache_dB[m] = Clamp(sinr, -20.0, 50.0);
        gMecLosProbCache[m] = los_p;
    }

    for (int r = 0; r < 2; r++) {
        const double d = EuclidDist(gUe.x, gUe.y, gRsu[r].x, gRsu[r].y);
        if (d > gRsu[r].range) {
            gRsuSinrCache_dB[r] = -30.0;
            gRsuLosProbCache[r] = 0.0;
            continue;
        }
        const double los_p = LoSProb(d, kDLoS_RSU_m);
        const bool   los   = (RandUnifRobust() < los_p);
        const double pl    = PathLoss_V2X(d, kFc_RSU_GHz, los);

        double fad;
        if (los) {
            const double K_lin = std::pow(10.0, 6.0 / 10.0);
            fad = SampleRicianFading(K_lin);
        } else {
            fad = SampleRayleighFading();
        }

        const double antenna_gain = gRsu[r].antenna_gain;
        const double rx_power = kTxPower_dBm + antenna_gain - pl
                              + 10.0 * SafeLog10(fad + 1e-9);
        const double rx_W = dBm_to_W(rx_power);

        double interf_W = 1e-12;
        for (int q = 0; q < 2; q++) {
            if (q == r) continue;
            double d_i = EuclidDist(gUe.x, gUe.y, gRsu[q].x, gRsu[q].y);
            if (d_i < gRsu[q].range) {
                interf_W += gInterfRsuCache_W[q];
            }
        }

        const double N_W = dBm_to_W(kNoisePower_dBm);
        double sinr = 10.0 * SafeLog10(rx_W / std::max(N_W + interf_W, 1e-20));
        gRsuSinrCache_dB[r] = Clamp(sinr, -20.0, 50.0);
        gRsuLosProbCache[r] = los_p;
    }
    gSinrCachePopulated = true;

    gCacheStepStamp = gStepCount;
}

// ============================================================
//  BACKHAUL DELAY
// ============================================================
static double BackhaulDelay(int mec_idx) {
    const int k = gUe.serving_gnb;
    if (k < 0 || k >= kNumGnb) return 0.005;

    const double prop = gGnb[k].bh_delay[mec_idx];
    const double data_bytes = gUe.data_size_mb * 1e6;
    const double cap = std::max(gGnb[k].bh_capacity[mec_idx], 1.0e9);

    const double tx_bh = (data_bytes * 8.0 * 1.1) / cap;
    const double q_bh = gGnb[k].load * 0.005;

    double result = std::max(prop + tx_bh + q_bh, 0.001);
    return Clamp(result, 0.001, 0.050);
}

// ============================================================
//  COMPUTATION DELAY (raw, no VM overhead)
// ============================================================
static double CompDelayRaw(double cpu_raw, double freq, double load) {
    double avail_freq = freq * (1.0 - Clamp(load, 0.0, 0.95) * 0.4);
    avail_freq = std::max(avail_freq, 1.0e6);

    double result = cpu_raw / avail_freq;
    result = std::max(result, 0.005);

    double queue_delay = Clamp(load / (1.0 - load + 0.01), 0.0, 0.050);
    result += queue_delay;

    return Clamp(result, 0.005, 0.200);
}

// ============================================================
//  MEC DELAY - REALISTIC 5-250ms
// ============================================================
static double ComputeMecDelay(int mec_idx) {
    RefreshStepCachesIfNeeded();

    if (mec_idx < 0 || mec_idx >= 3) return 0.200;

    const double cpu_cycles = gUe.cpu_cycles_giga * 1e9;
    const double freq_hz    = gMec[mec_idx].freq;
    const double load       = gMec[mec_idx].load;

    const double vms_in_use     = load * kMecMaxVms;
    const bool   pool_saturated = (vms_in_use >= kMecMaxVms * 0.95);

    const double effective_freq = freq_hz * (1.0 - load * 0.3) / (1.0 + kMecVmOverhead);
    const double min_freq       = 5.0e9;
    const double safe_freq      = std::max(effective_freq, min_freq);

    double comp_time = cpu_cycles / safe_freq;
    comp_time = std::max(comp_time, 0.005);

    const double vm_switch_time = 0.001 * (1.0 + load * 4.0);
    comp_time += vm_switch_time;

    comp_time += gMecColdStartDelay[mec_idx];

    const double queue_delay = load * 0.020;
    comp_time += queue_delay;

    if (pool_saturated) {
        comp_time += 0.050;
    }

    return Clamp(comp_time, 0.005, 0.250);
}

// ============================================================
//  ENERGY CALCULATION
// ============================================================
static double ComputeEnergyAction(int action) {
    if (!std::isfinite(gUe.air_sinr_dB)) gUe.air_sinr_dB = 0.0;
    for (int r = 0; r < 2; r++) {
        if (!std::isfinite(gRsu[r].sinr_dB)) gRsu[r].sinr_dB = 0.0;
    }

    const double cpu_raw    = gUe.cpu_cycles_giga * 1.0e9;
    const double data_bytes = gUe.data_size_mb * 1e6;

    auto cached_or_compute_comp = [&](int act, double fallback_freq, double fallback_load) -> double {
        if (gActionCache.valid && act >= 0 && act < kNumActions) {
            return gActionCache.comp[act];
        }
        return CompDelayRaw(cpu_raw, fallback_freq, fallback_load);
    };

    try {
        switch (action) {
            case kActLocal: {
                double comp_time = cached_or_compute_comp(action, kLocalFreq, 0.0);
                double cpu_energy = cpu_raw * kEnergyPerCycle_J;
                double active_power = 0.5 * Clamp(cpu_raw / 1e9, 0.0, 1.0);
                double energy = (active_power * comp_time) + cpu_energy;
                if (!std::isfinite(energy)) return 0.1;
                return Clamp(energy, 0.0, 2.0);
            }
            case kActMec0: case kActMec1: case kActMec2: {
                int m = action - 1;
                double tx_power = 0.05 + 0.15 * (1.0 - NormSINR(gUe.air_sinr_dB));
                double tx_time  = (gActionCache.valid) ? gActionCache.air[action]
                                                       : gUe.air_tx_delay_s;
                double tx_energy = tx_power * tx_time;

                double load_factor = std::pow(Clamp(gMec[m].load, 0.0, 1.0), 1.2);
                double mec_power = kMecPowerIdle_W + (kMecPowerMax_W - kMecPowerIdle_W) * load_factor;
                double comp_time = cached_or_compute_comp(action, gMec[m].freq, gMec[m].load);
                double mec_energy = mec_power * comp_time * 0.3;

                double total_energy = tx_energy + mec_energy;
                if (!std::isfinite(total_energy)) return 0.5;
                return Clamp(total_energy, 0.0, 5.0);
            }
            case kActRsu0: case kActRsu1: {
                int r = action - 4;
                double sinr = gRsu[r].sinr_dB;
                double capacity = Capacity_bps(sinr, kRsuBandwidth_MHz);
                double tx_time = (data_bytes * 8.0) / std::max(capacity, 1.0);
                double tx_energy = tx_time * kUeTxPower_W;

                double load_factor = std::pow(Clamp(gRsu[r].load, 0.0, 1.0), 1.2);
                double rsu_power = kRsuPowerIdle_W + (kRsuPowerMax_W - kRsuPowerIdle_W) * load_factor;
                double comp_time = cached_or_compute_comp(action, gRsu[r].freq, gRsu[r].load);
                double rsu_energy = rsu_power * comp_time * 0.25;

                double total_energy = tx_energy + rsu_energy;
                if (!std::isfinite(total_energy)) return 0.5;
                return Clamp(total_energy, 0.0, 5.0);
            }
            case kActSat: {
                RefreshStepCachesIfNeeded();
                double tx_time = (gActionCache.valid) ? gActionCache.air[action]
                                                      : gUe.air_tx_delay_s;
                double tx_energy = tx_time * 0.5;
                double processing = 0.3;
                double total_energy = tx_energy + processing;
                if (!std::isfinite(total_energy)) return 0.5;
                return Clamp(total_energy, 0.0, 5.0);
            }
            default:
                return 0.0;
        }
    } catch (...) {
        return 0.5;
    }
}

// ============================================================
//  RSU HELPER
// ============================================================
static bool IsWithinRsuRange(int rsu_idx) {
    if (rsu_idx < 0 || rsu_idx >= 2) return false;
    double distance = EuclidDist(gUe.x, gUe.y, gRsu[rsu_idx].x, gRsu[rsu_idx].y);
    return distance <= gRsu[rsu_idx].range;
}

// ============================================================
//  CACHE ALL DELAYS
// ============================================================
static void ComputeAllDelaysAndEnergy() {
    RefreshStepCachesIfNeeded();

    // --- Compute air TX delay using the CURRENT step's SINR ---
    {
        const double sinr = gUe.air_sinr_dB;
        double sinr_lin = std::pow(10.0, sinr / 10.0);
        sinr_lin = std::max(sinr_lin, 1e-3);

        double spectral_eff = SafeLog2(1 + sinr_lin) * 0.75;
        double capacity = kAirBandwidth_MHz * 1e6 * spectral_eff;
        capacity = std::max(capacity, 1.0e6);

        double tx_time = (gUe.data_size_mb * 1e6 * 8.0) / capacity;
        tx_time *= 1.10;
        gAirTxDelayCache_s = Clamp(tx_time, 0.0005, 0.020);
    }
    gUe.air_tx_delay_s = gAirTxDelayCache_s;

    gActionCache.Invalidate();

    const double cpu_raw    = gUe.cpu_cycles_giga * 1.0e9;
    const double data_bytes = gUe.data_size_mb * 1e6;

    // --- PASS 1: delays only ---
    for (int action = 0; action < kNumActions; action++) {
        double t_air = 0.0, t_bh = 0.0, t_comp = 0.0, t_q = 0.0;

        switch (action) {
            case kActLocal: {
                t_comp = CompDelayRaw(cpu_raw, kLocalFreq, 0.0);
                break;
            }
            case kActMec0: case kActMec1: case kActMec2: {
                int m = action - 1;
                t_air  = gUe.air_tx_delay_s;
                t_bh   = BackhaulDelay(m);
                t_comp = ComputeMecDelay(m);
                double queue_ratio = Clamp(gMec[m].queue / kMaxQueue, 0.0, 1.0);
                t_q    = queue_ratio * 0.020;
                break;
            }
            case kActRsu0: case kActRsu1: {
                int r = action - 4;
                if (!IsWithinRsuRange(r)) {
                    t_q = 0.100;
                    break;
                }
                const double sinr = gRsu[r].sinr_dB;

                // --- Air time ---
                double sinr_linear = std::pow(10.0, sinr / 10.0);
                sinr_linear = std::max(sinr_linear, 0.1);
                double spectral_eff = SafeLog2(1.0 + sinr_linear) * 0.5;
                double capacity = kRsuBandwidth_MHz * 1e6 * spectral_eff;
                capacity = std::max(capacity, 1.0e6);
                t_air = (data_bytes * 8.0) / capacity;
                if (gUe.ue_type == 2) {
                    t_air *= (1.0 + gUe.speed / kMaxSpeed * 0.2);
                }
                t_air = Clamp(t_air, 0.005, 0.020);

                // --- Backhaul ---
                t_bh = (data_bytes * 8.0) / kRsuBackhaul_bps + kRsuBhDelay_s;
                t_bh = Clamp(t_bh, 0.001, 0.010);

                // --- Compute ---
                const double cpu_raw_cycles = gUe.cpu_cycles_giga * 1e9;
                const double rsu_freq = 5.0e9 * kRsuProcessingEfficiency
                                        * (1.0 - gRsu[r].load * 0.5);
                t_comp = cpu_raw_cycles / std::max(rsu_freq, 1.0e9);
                t_comp = Clamp(t_comp, 0.005, 0.200);

                // --- Queue ---
                double queue_ratio = Clamp(gRsu[r].queue / kMaxQueue, 0.0, 1.0);
                t_q = queue_ratio * 0.020;
                break;
            }
            // Satellite path uses the physically-modeled one-way delay
            // (propagation + processing + Doppler jitter) cached in
            // gSatDelayCache_s by RefreshStepCachesIfNeeded().
            case kActSat: {
                RefreshStepCachesIfNeeded();
                t_air  = gUe.air_tx_delay_s;
                t_bh   = gSatDelayCache_s;          // physical one-way propagation
                t_comp = 0.0;                       // no separate on-board compute modeled
                t_q    = 0.0;
                break;
            }
            default: {
                t_comp = 0.100;
                break;
            }
        }

        const double c_air  = Clamp(t_air,  0.0, 0.150);
        const double c_bh   = Clamp(t_bh,   0.0, 0.300);
        const double c_comp = Clamp(t_comp, 0.0, 0.200);
        const double c_q    = Clamp(t_q,    0.0, 0.030);
        const double c_delay = c_air + c_bh + c_comp + c_q;

        gActionCache.delays[action] = c_delay;
        gActionCache.air[action]    = c_air;
        gActionCache.bh[action]     = c_bh;
        gActionCache.comp[action]   = c_comp;
        gActionCache.queue[action]  = c_q;
    }

    gActionCache.valid = true;

    // --- PASS 2: energies ---
    for (int action = 0; action < kNumActions; action++) {
        gActionCache.energies[action] = ComputeEnergyAction(action);
    }
}

// ============================================================
//  CURRICULUM LEARNING
// ============================================================
static double GetComplexityRange() {
    // v8.6.7: In eval mode, draw complexity from a WIDE uniform band
    // so tasks span the full difficulty range. Freezing it at 0.7 was
    // producing a narrow task distribution that allowed runs of easy
    // tasks, causing 100%-DL episodes.
    if (!gIsTraining) {
        return 0.25 + RandUnifRobust() * 0.50;   // uniform in [0.25, 0.75]
    }
    const int plateau_episodes = 400;
    double progress = std::min(1.0, (double)gEpisodeCount / plateau_episodes);
    double complexity = 0.1 + 0.6 * (1.0 - std::exp(-progress * 2.5));
    double variation = (RandUnifRobust() - 0.5) * 0.12;
    return Clamp(complexity + variation, 0.1, 0.75);
}

// ============================================================
//  TASK GENERATION (v8.4 difficulty retune)
// ============================================================
static void GenerateNewTask() {
    // v8.6.12: redraw UE type and speed every step. Previously the UE
    // type was set once per episode in ResetUe(), which caused the DL
    // distribution to be bimodal: IoT episodes always succeeded, eMBB
    // and V2X episodes always failed. Redrawing per step mixes UE types
    // within each episode, giving a smooth DL distribution that reflects
    // the average task mix.
    double type_rand = RandUnifRobust();
    if (type_rand < 0.60) {
        gUe.ue_type = 0;
        gUe.speed = 15.0 + RandUnifRobust() * 15.0;
    } else if (type_rand < 0.85) {
        gUe.ue_type = 1;
        gUe.speed = RandUnifRobust() * 0.2;
    } else {
        gUe.ue_type = 2;
        gUe.speed = 40.0 + RandUnifRobust() * 20.0;
    }

    double complexity = GetComplexityRange();
    switch (gUe.ue_type) {
        case 0: { // MOBILE — eMBB
            gUe.cpu_cycles_giga = Clamp(0.5 + complexity * 2.5, 0.3, 3.0);
            gUe.data_size_mb    = Clamp(0.8 + complexity * 10.0, 0.5, 15.0);
            // v8.6.4: widened to give a broad DL distribution.
            // At complexity=0.7: deadline ∈ [0.086, 0.150]
            gUe.deadline = Clamp(0.15 - complexity * 0.09, 0.080, 0.150);
            break;
        }
        case 1: { // IOT — mMTC
            gUe.cpu_cycles_giga = Clamp(0.03 + complexity * 0.25, 0.01, 0.35);
            gUe.data_size_mb    = Clamp(0.03 + complexity * 0.6, 0.01, 1.0);
            // v8.6.4: widened. At complexity=0.7: deadline ∈ [0.200, 0.400]
            gUe.deadline = Clamp(0.40 - complexity * 0.28, 0.200, 0.400);
            break;
        }
        case 2: { // VEHICLE — V2X
            gUe.cpu_cycles_giga = Clamp(0.5 + complexity * 2.0, 0.3, 2.5);
            gUe.data_size_mb    = Clamp(1.5 + complexity * 12.0, 1.0, 18.0);
            // v8.6.4: widened. At complexity=0.7: deadline ∈ [0.060, 0.120]
            gUe.deadline = Clamp(0.12 - complexity * 0.09, 0.060, 0.120);
            break;
        }
    }
}

// ============================================================
//  RESET FUNCTIONS
// ============================================================
static void ResetUe() {
    gUe.x = RandUnifRobust() * kAreaSize;
    gUe.y = RandUnifRobust() * kAreaSize;

    double type_rand = RandUnifRobust();
    if (type_rand < 0.60) {
        gUe.ue_type = 0;
        // v8.6.8: bumped to 15..30 m/s so HO events actually occur
        gUe.speed = 15.0 + RandUnifRobust() * 15.0;
    } else if (type_rand < 0.85) {
        gUe.ue_type = 1;
        gUe.speed = RandUnifRobust() * 0.2;
    } else {
        gUe.ue_type = 2;
        // v8.6.8: bumped to 40..60 m/s (highway speeds)
        gUe.speed = 40.0 + RandUnifRobust() * 20.0;
    }

    const double angle = RandUnifRobust() * 2.0 * M_PI;
    gUe.vx = gUe.speed * std::cos(angle);
    gUe.vy = gUe.speed * std::sin(angle);

    GenerateNewTask();
    gUe.fading = SampleRayleighFading();
    gUe.doppler_norm = DopplerNorm(kFc_MEC_GHz);
    gUe.handover = false;
    gUe.stable_connection = 0;
    gUe.completion_reward = 0.0;

    gSINRCacheValid = false;
    ComputeAllGnbSINR();
    gUe.serving_gnb = SelectServingGnb();
    gUe.air_sinr_dB = ComputeAirSINR(gUe.serving_gnb);

    // Force step-cache refresh at the newly-set UE position. This will also
    // populate gMecSinrCache_dB and gRsuSinrCache_dB in one shot.
    gCacheStepStamp = -1;
    RefreshStepCachesIfNeeded();

    for (int m = 0; m < 3; m++) {
        gMec[m].sinr_dB  = gMecSinrCache_dB[m];
        gMec[m].los_prob = gMecLosProbCache[m];
    }
    for (int r = 0; r < 2; r++) {
        gRsu[r].sinr_dB  = gRsuSinrCache_dB[r];
        gRsu[r].los_prob = gRsuLosProbCache[r];
        gRsuMetrics.rsu_avg_sinr += gRsu[r].sinr_dB;
        gRsuMetrics.rsu_sinr_samples++;
    }

    gActionCache.Invalidate();
    gSINRCacheValid = false;
}

static void InitServers() {
    gMec[0] = {200.0, 200.0, kMecFreq0, 0.0, 0.0, 0.0, 0.0, 150.0, kMecVmOverhead};
    gMec[1] = {500.0, 800.0, kMecFreq1, 0.0, 0.0, 0.0, 0.0, 90.0, kMecVmOverhead};
    gMec[2] = {800.0, 300.0, kMecFreq2, 0.0, 0.0, 0.0, 0.0, 60.0, kMecVmOverhead};
    gRsu[0] = {350.0, 500.0, kRsuFreq, 0.0, 0.0, 10.0, 0.9, kRsuMaxRange_m, kRsuAntennaGain_dBi};
    gRsu[1] = {700.0, 600.0, kRsuFreq, 0.0, 0.0, 10.0, 0.9, kRsuMaxRange_m, kRsuAntennaGain_dBi};
}

static void InitGnbs() {
    const double pos[kNumGnb][2] = {
        {200.0, 400.0}, {400.0, 200.0},
        {300.0, 700.0}, {600.0, 850.0},
        {750.0, 450.0}, {900.0, 250.0}
    };

    for (int k = 0; k < kNumGnb; k++) {
        gGnb[k].x = pos[k][0];
        gGnb[k].y = pos[k][1];
        gGnb[k].load = 0.0;
        gGnb[k].sinr_to_ue_dB = 20.0;

        for (int m = 0; m < 3; m++) {
            if (kGnbHomeMec[k] == m) {
                gGnb[k].bh_delay[m] = kBhFiber_s;
                gGnb[k].bh_capacity[m] = kBhCapacity_bps;
            } else {
                gGnb[k].bh_delay[m] = kBhCross_s + (RandUnifRobust() - 0.5) * 0.001;
                gGnb[k].bh_capacity[m] = kBhCrossCapacity;
            }
        }
    }
}

static void InitInterferers() {
    for (int i = 0; i < kNumInterferers; i++) {
        gInterferers[i].x = 100.0 + RandUnifRobust() * (kAreaSize - 200.0);
        gInterferers[i].y = 100.0 + RandUnifRobust() * (kAreaSize - 200.0);
    }
}

static void ResetEnvironment() {
    InitServers();
    InitGnbs();
    InitInterferers();

    gStepCount = 0;
    gCacheStepStamp = -1;

    for (int m = 0; m < 3; m++) gStepsSinceMecUse[m] = 999;

    // v8.6.17: reset satellite load
    gSatLoad = 0.0;

    // v8.6.11: narrow initial-load spread.
    // The previous 0.15..0.70 range produced bimodal episode outcomes:
    // episodes starting at low load succeeded throughout, episodes
    // starting at high load failed throughout. Since load evolves too
    // slowly to change within an episode (decay 0.995), the initial
    // draw determined the whole episode's DL. Narrowing to 0.25..0.40
    // keeps load near the mid-range and eliminates the bimodality.
    for (int m = 0; m < 3; m++) {
        gMec[m].queue = RandUnifRobust() * 20.0;
        gMec[m].load  = 0.25 + RandUnifRobust() * 0.15;   // 0.25..0.40
    }
    for (int r = 0; r < 2; r++) {
        gRsu[r].queue = RandUnifRobust() * 10.0;
        gRsu[r].load  = 0.20 + RandUnifRobust() * 0.15;   // 0.20..0.35
    }

    ResetUe();

    gDeadlinesMet = 0;
    gTotalSteps = 0;
    gLastReward = 0.0f;
    gLastRawReward = 0.0;
    gSmoothedReward = 0.0;
    gLastAction = -1;
    gActionConsistencyCount = 0;

    gTotalBackhaulDelay = 0.0;
    gHomeBackhaulCount = 0;
    gCrossBackhaulCount = 0;
    gTotalMecBackhaulDelay = 0.0;
    gMecBackhaulCount = 0;
    gTotalRsuBackhaulDelay = 0.0;
    gRsuBackhaulCount = 0;
    gTotalSatBackhaulDelay = 0.0;
    gSatBackhaulCount = 0;

    gHandoverCount = 0;
    for (int a = 0; a < 7; a++) {
        gActionCount[a] = 0;
        gActionAttempts[a] = 0;
        gActionDeadlinesMet[a] = 0;
    }
    gTotalGnbLoad = 0.0;
    gTotalMecUtil = 0.0;
    gTotalDelay = 0.0;
    gTotalEnergy = 0.0;
    gTotalAirDelay = 0.0;
    gTotalCompDelay = 0.0;
    gTotalQueueDelay = 0.0;
    gTotalGnbServing = 0.0;
    gTotalJainsFairness = 0.0;

    gRwdCompDeadline    = 0.0;
    gRwdCompDelay       = 0.0;
    gRwdCompEnergy      = 0.0;
    gRwdCompLoad        = 0.0;
    gRwdCompPenalty     = 0.0;
    gRwdCompBackhaul    = 0.0;
    gRwdCompOffload     = 0.0;
    gRwdCompConsistency = 0.0;
    gRwdCompCount       = 0;

    gRsuMetrics.total_rsu_utilization = 0.0;
    gRsuMetrics.total_rsu_coverage_time = 0.0;
    gRsuMetrics.total_rsu_handovers = 0.0;
    gRsuMetrics.total_rsu_offload_success = 0.0;
    gRsuMetrics.rsu_offload_attempts = 0.0;

    gActionCache.Invalidate();
    gSINRCacheValid = false;

    // Force the next RefreshStepCachesIfNeeded() to redraw cold-start,
    // satellite delay, and interference for the first step of the new episode.
    gCacheStepStamp = -1;
    gSinrCachePopulated = false;
}

// ============================================================
//  REFRESH WITH REALISTIC HANDOVER
// ============================================================
static void RefreshChannelState() {
    gUe.doppler_norm = DopplerNorm(kFc_MEC_GHz);
    const int prev_gnb = gUe.serving_gnb;

    gSINRCacheValid = false;
    ComputeAllGnbSINR();
    const int new_gnb = SelectServingGnb();

    static int s_pending_target = -1;
    static int s_pending_count  = 0;

    if (new_gnb != prev_gnb) {
        double new_sinr  = ComputeAirSINR(new_gnb);
        double prev_sinr = ComputeAirSINR(prev_gnb);

        if (new_sinr > prev_sinr + kHandoverHysteresis_dB) {
            if (s_pending_target == new_gnb) {
                s_pending_count++;
            } else {
                s_pending_target = new_gnb;
                s_pending_count  = 1;
            }

            if (s_pending_count >= kHandoverTTTSteps) {
                gUe.serving_gnb       = new_gnb;
                gUe.handover          = true;
                gUe.stable_connection = 0;
                s_pending_target      = -1;
                s_pending_count       = 0;
            } else {
                gUe.serving_gnb       = prev_gnb;
                gUe.handover          = false;
                gUe.stable_connection++;
            }
        } else {
            s_pending_target      = -1;
            s_pending_count       = 0;
            gUe.serving_gnb       = prev_gnb;
            gUe.handover          = false;
            gUe.stable_connection++;
        }
    } else {
        s_pending_target      = -1;
        s_pending_count       = 0;
        gUe.handover          = false;
        gUe.stable_connection++;
    }

    gUe.air_sinr_dB = ComputeAirSINR(gUe.serving_gnb);

    // Force a refresh of the step caches with the new UE position, so
    // MEC/RSU SINR, cold-start, sat delay, and interference all use the
    // same geometry for this step.
    gCacheStepStamp = -1;
    RefreshStepCachesIfNeeded();

    for (int m = 0; m < 3; m++) {
        double sinr = gMecSinrCache_dB[m];
        if (gUe.ue_type == 2 && std::isfinite(gUe.doppler_norm)) {
            sinr -= kDopplerAlpha * gUe.doppler_norm * kDopplerLossMax_dB;
        }
        gMec[m].sinr_dB = Clamp(sinr, -20.0, 50.0);
        gMec[m].los_prob = gMecLosProbCache[m];
    }

    for (int r = 0; r < 2; r++) {
        double sinr = gRsuSinrCache_dB[r];
        // RSU Doppler penalty: normalized max Doppler shift at the RSU
        // carrier, mirroring DopplerNorm() but for an arbitrary fc.
        if (gUe.ue_type == 2) {
            auto doppler_norm_at_fc = [](double fc_GHz) -> double {
                if (!std::isfinite(fc_GHz) || fc_GHz <= 0.0) return 0.0;
                const double fd     = (gUe.speed  / kSpeedLight) * fc_GHz * 1.0e9;
                const double fd_max = (kMaxSpeed  / kSpeedLight) * fc_GHz * 1.0e9;
                double n = SafeDiv(fd, fd_max);
                if (!std::isfinite(n)) return 0.0;
                return Clamp(n, 0.0, 1.0);
            };
            double doppler_rsu = doppler_norm_at_fc(kFc_RSU_GHz);
            sinr -= kDopplerAlpha * doppler_rsu * kDopplerLossMax_dB;
        }
        gRsu[r].sinr_dB = Clamp(sinr, -20.0, 50.0);
        gRsu[r].los_prob = gRsuLosProbCache[r];
        gRsuMetrics.rsu_avg_sinr += gRsu[r].sinr_dB;
        gRsuMetrics.rsu_sinr_samples++;
    }

    if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb) {
        gGnb[gUe.serving_gnb].sinr_to_ue_dB = gUe.air_sinr_dB;
    }
    gUe.fading = SampleRayleighFading();
    gActionCache.Invalidate();
}

// ============================================================
//  MOBILITY (v8.6.2: integrate at the actual step interval)
// ============================================================
static void StepMobility() {
    gUe.x += gUe.vx * kStepInterval_s;
    gUe.y += gUe.vy * kStepInterval_s;

    if (gUe.x <= 0.0 || gUe.x >= kAreaSize) {
        gUe.vx = -gUe.vx;
        gUe.x = Clamp(gUe.x, 0.0, kAreaSize);
    }
    if (gUe.y <= 0.0 || gUe.y >= kAreaSize) {
        gUe.vy = -gUe.vy;
        gUe.y = Clamp(gUe.y, 0.0, kAreaSize);
    }

    // v8.6.9: regenerate the task each step.
    // Previously the task was drawn only at episode reset, which meant
    // every step in an episode faced the SAME task. Easy tasks produced
    // DL=100% for the whole episode, hard tasks produced DL=0%. That
    // produced the bimodal DL distribution observed in the baselines.
    GenerateNewTask();

    RefreshChannelState();
}

// ============================================================
//  UPDATE LOADS
// ============================================================
static void UpdateLoads(int action) {
    for (int m = 0; m < 3; m++) {
        gMec[m].load = Clamp(gMec[m].load * kLoadDecayRate, 0.0, 1.0);
        gMec[m].queue = Clamp(gMec[m].queue * kQueueDecayRate, 0.0, kMaxQueue);
    }
    for (int r = 0; r < 2; r++) {
        gRsu[r].load = Clamp(gRsu[r].load * kLoadDecayRate, 0.0, 1.0);
        gRsu[r].queue = Clamp(gRsu[r].queue * kQueueDecayRate, 0.0, kMaxQueue);
        gRsuMetrics.total_rsu_utilization += gRsu[r].load;
    }

    // v8.6.17: satellite load decays like MEC/RSU
    gSatLoad = Clamp(gSatLoad * kSatLoadDecayRate, 0.0, kSatLoadMax);

    // Background traffic
    for (int m = 0; m < 3; m++) {
        if (RandUnifRobust() < kBackgroundLoadProb) {
            gMec[m].load  = Clamp(gMec[m].load + kBackgroundLoadAmount, 0.0, 1.0);
            gMec[m].queue = Clamp(gMec[m].queue + 1.0, 0.0, kMaxQueue);
        }
    }
    for (int r = 0; r < 2; r++) {
        if (RandUnifRobust() < kBackgroundLoadProb * 0.5) {
            gRsu[r].load  = Clamp(gRsu[r].load + kBackgroundLoadAmount * 0.5, 0.0, 1.0);
            gRsu[r].queue = Clamp(gRsu[r].queue + 0.5, 0.0, kMaxQueue);
        }
    }

    for (int k = 0; k < kNumGnb; k++)
        gGnb[k].load = Clamp(gGnb[k].load * 0.97, 0.0, 1.0);

    switch (action) {
        case kActMec0:
            gMec[0].load += kLoadIncrement;
            gMec[0].queue += 2.0;
            if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb)
                gGnb[gUe.serving_gnb].load += 0.02;
            break;
        case kActMec1:
            gMec[1].load += kLoadIncrement;
            gMec[1].queue += 2.0;
            if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb)
                gGnb[gUe.serving_gnb].load += 0.02;
            break;
        case kActMec2:
            gMec[2].load += kLoadIncrement;
            gMec[2].queue += 2.0;
            if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb)
                gGnb[gUe.serving_gnb].load += 0.02;
            break;
        case kActRsu0:
            gRsu[0].load += kLoadIncrement * 1.2;
            gRsu[0].queue += 2.5;
            if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb)
                gGnb[gUe.serving_gnb].load += 0.025;
            break;
        case kActRsu1:
            gRsu[1].load += kLoadIncrement * 1.2;
            gRsu[1].queue += 2.5;
            if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb)
                gGnb[gUe.serving_gnb].load += 0.025;
            break;
        case kActSat:
            if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb)
                gGnb[gUe.serving_gnb].load += 0.015;
            for (int m = 0; m < 3; m++) {
                gMec[m].load = Clamp(gMec[m].load - 0.005, 0.0, 1.0);
            }
            // v8.6.17: increment satellite load on SAT use
            gSatLoad = Clamp(gSatLoad + kSatLoadIncrement, 0.0, kSatLoadMax);
            break;
        default:
            break;
    }

    for (int m = 0; m < 3; m++) {
        gMec[m].load = Clamp(gMec[m].load, 0.0, 1.0);
        gMec[m].queue = Clamp(gMec[m].queue, 0.0, kMaxQueue);
    }
    for (int r = 0; r < 2; r++) {
        gRsu[r].load = Clamp(gRsu[r].load, 0.0, 1.0);
        gRsu[r].queue = Clamp(gRsu[r].queue, 0.0, kMaxQueue);
    }
    for (int k = 0; k < kNumGnb; k++) {
        gGnb[k].load = Clamp(gGnb[k].load, 0.0, 1.0);
    }

    for (int m = 0; m < 3; m++) {
        if (action == (m + 1)) {
            gStepsSinceMecUse[m] = 0;
        } else {
            gStepsSinceMecUse[m]++;
        }
    }
}

// ============================================================
//  CONVERGENCE DETECTION
// ============================================================
static bool CheckConvergence(double dl_rate) {
    int idx = gDLHistoryIdx % kConvergenceWindow;
    gDLHistory[idx] = dl_rate;
    gDLHistoryIdx = (gDLHistoryIdx + 1) % kConvergenceWindow;
    gDLHistoryCount = std::min(gDLHistoryCount + 1, kConvergenceWindow);

    if (gDLHistoryCount < kConvergenceWindow) return gConverged;

    double mean = 0.0;
    for (int i = 0; i < kConvergenceWindow; i++) {
        mean += gDLHistory[i];
    }
    mean /= kConvergenceWindow;

    double variance = 0.0;
    for (int i = 0; i < kConvergenceWindow; i++) {
        variance += (gDLHistory[i] - mean) * (gDLHistory[i] - mean);
    }
    variance /= kConvergenceWindow;
    double stddev = std::sqrt(std::max(variance, 0.0));

    if (stddev < kConvergenceThreshold) {
        gConvergenceCount++;
        if (gConvergenceCount > kConvergenceMinCount) {
            gConverged = true;
            if (gConvergenceEpisode == -1) {
                gConvergenceEpisode = gEpisodeCount;
            }
        }
    } else {
        // Reset stability counter regardless of whether we've already
        // converged (allows de-convergence detection).
        gConvergenceCount = 0;
        if (gConverged) {
            gConverged = false;
            gConvergenceLogged = false;
            gConvergenceEpisode = -1;
        }
    }

    return gConverged;
}

// ============================================================
//  REWARD FUNCTION
// ============================================================
static double ComputeReward(int action, double delay, double energy) {
    const bool met = (delay <= gUe.deadline);
    if (met) gDeadlinesMet++;

    // 1. DEADLINE COMPONENT (55%)
    double deadline_component = 0.0;
    if (met) {
        double margin = 1.0 - (delay / gUe.deadline);
        deadline_component = 0.5 + 0.5 * Clamp(margin, 0.0, 1.0);
    } else {
        double violation = (delay - gUe.deadline) / gUe.deadline;
        deadline_component = -Clamp(violation, 0.0, 2.0);
    }

    // 2. DELAY COMPONENT (10%)
    double norm_delay = Clamp(delay / 0.5, 0.0, 1.0);
    double delay_component = 1.0 - norm_delay;

    // 3. ENERGY COMPONENT (5%)
    double norm_energy = Clamp(energy / 2.0, 0.0, 1.0);
    double energy_component = 1.0 - norm_energy;

    // 4. LOAD BALANCING COMPONENT (10%)
    double load_penalty = 0.0;
    if (action >= kActMec0 && action <= kActMec2) {
        int m = action - 1;
        load_penalty = gMec[m].load;
    } else if (action >= kActRsu0 && action <= kActRsu1) {
        int r = action - 4;
        load_penalty = gRsu[r].load * 0.5;
    }
    double load_component = 1.0 - Clamp(load_penalty, 0.0, 1.0);

    // 5. PENALTY COMPONENT (5%)
    //    penalty is in [-0.5, 0]; penalty_component is in [-1, 0].
    //    Weighted sum therefore SUBTRACTS for bad actions.
    double penalty = 0.0;

    if (action == kActLocal) {
        double task_size = (gUe.cpu_cycles_giga / 5.0 + gUe.data_size_mb / 50.0) / 2.0;
        if (task_size > 0.5) {
            penalty -= 0.2 * (task_size - 0.5) / 0.5;
        }
    }

    if (action == kActSat) {
        penalty -= 0.15;   // was 0.25 — reduce, backhaul_cost already penalizes
    }

    if (action >= kActMec0 && action <= kActMec2) {
        int m = action - 1;
        if (kGnbHomeMec[gUe.serving_gnb] != m) {
            penalty -= 0.03;
        }
    }

    if (action >= kActRsu0 && action <= kActRsu1) {
        int r = action - 4;
        if (!IsWithinRsuRange(r)) {
            penalty -= 0.15;
        }
    }

    double penalty_component = Clamp(penalty, -1.0, 0.0);

    // 6. BACKHAUL COMPONENT (15%)
    double backhaul_cost = 0.0;
    if (action >= kActMec0 && action <= kActMec2) {
        int m = action - 1;
        const int k = gUe.serving_gnb;
        if (k >= 0 && k < kNumGnb) {
            double bh_delay = gGnb[k].bh_delay[m];
            backhaul_cost = Clamp(bh_delay / kMaxBhDelay_norm, 0.0, 1.0);
            if (kGnbHomeMec[k] != m) {
                backhaul_cost += 0.1;
            }
        }
    } else if (action >= kActRsu0 && action <= kActRsu1) {
        backhaul_cost = 0.05;
    } else if (action == kActSat) {
        backhaul_cost = 0.8;
    }
    double backhaul_component = 1.0 - Clamp(backhaul_cost, 0.0, 1.0);

    // 7. OFFLOAD INCENTIVE
    double offload_bonus = 0.0;
    if (action != kActLocal) {
        double task_size = (gUe.cpu_cycles_giga / 5.0 + gUe.data_size_mb / 50.0) / 2.0;
        double urgency = Clamp(gUe.deadline / 0.15, 0.0, 1.0);

        if (task_size > 0.3) {
            offload_bonus += 0.15 * (task_size - 0.3) / 0.7;
        }
        if (urgency > 0.6) {
            offload_bonus += 0.15 * (urgency - 0.6) / 0.4;
        }

        if (action >= kActRsu0 && action <= kActRsu1 && gUe.ue_type == 2) {
            int r = action - 4;
            if (IsWithinRsuRange(r)) {
                offload_bonus += 0.25;
                if (met) {
                    offload_bonus += 0.15;
                }
            }
        }
    }
    offload_bonus = Clamp(offload_bonus, 0.0, 0.4);

    // 8. ACTION CONSISTENCY BONUS
    double consistency_bonus = 0.0;
    if (action == gLastAction) {
        gActionConsistencyCount++;
        if (gActionConsistencyCount > 3) {
            consistency_bonus = 0.03 * std::min(gActionConsistencyCount / 10.0, 1.0);
        }
    } else {
        gActionConsistencyCount = 0;
        gLastAction = action;
    }

    // 9. COMBINE
    double raw_reward =
        kWeightDeadline * deadline_component +
        kWeightDelay * delay_component +
        kWeightEnergy * energy_component +
        kWeightLoad * load_component +
        kWeightPenalty * penalty_component +   // now actually subtracts
        kWeightBackhaul * backhaul_component +
        offload_bonus * 0.05 +
        consistency_bonus * 0.02;

    raw_reward = Clamp(raw_reward, -3.0, 4.0);

    // --- Diagnostic accumulators (per-step additive view of reward) ---
    gRwdCompDeadline    += kWeightDeadline * deadline_component;
    gRwdCompDelay       += kWeightDelay * delay_component;
    gRwdCompEnergy      += kWeightEnergy * energy_component;
    gRwdCompLoad        += kWeightLoad * load_component;
    gRwdCompPenalty     += kWeightPenalty * penalty_component;
    gRwdCompBackhaul    += kWeightBackhaul * backhaul_component;
    gRwdCompOffload     += offload_bonus * 0.05;
    gRwdCompConsistency += consistency_bonus * 0.02;
    gRwdCompCount       += 1;

    if (gEpisodeCount == 0 && gStepCount == 0) {
        gSmoothedReward = raw_reward;
    } else {
        gSmoothedReward = (1.0 - kRewardSmoothFactor) * gSmoothedReward +
                          kRewardSmoothFactor * raw_reward;
    }

    gRewardStats.count++;

    double value = gIsTraining ? gSmoothedReward : raw_reward;
    gLastRawReward = value;

    // v8.6.10: freeze normalization in eval mode so the reward signal
    // does not drift over the course of an evaluation run.
    double normalized;
    if (gIsTraining) {
        if (kUseRunningNormalization) {
            // Welford running mean/var over the raw (pre-tanh) value
            gRewardNorm.n++;
            double delta = value - gRewardNorm.mean;
            gRewardNorm.mean += delta / gRewardNorm.n;
            double delta2 = value - gRewardNorm.mean;
            gRewardNorm.m2 += delta * delta2;

            if (gRewardNorm.n < kRunningNormWarmup) {
                normalized = 0.0;   // no normalization during warmup
            } else {
                double var = gRewardNorm.m2 / std::max<double>(gRewardNorm.n - 1.0, 1.0);
                double std_dev = std::sqrt(std::max(var, kRunningNormEps));
                normalized = (value - gRewardNorm.mean) / std_dev;
            }
        } else {
            double denom = std::max(kFixedRewardStd, 1e-6);
            normalized = (value - kFixedRewardMean) / denom;
        }
    } else {
        // Eval mode: use fixed normalization so the reward is stationary
        double denom = std::max(kFixedRewardStd, 1e-6);
        normalized = (value - kFixedRewardMean) / denom;
    }

    return Clamp(std::tanh(normalized), -1.0, 1.0);
}

// ============================================================
//  ACTION DELAY ACCESSOR
// ============================================================
static double ActionDelay(int action, double* p_air = nullptr,
                         double* p_bh = nullptr, double* p_comp = nullptr,
                         double* p_queue = nullptr) {
    if (action < 0 || action >= kNumActions) action = kActLocal;

    if (!gActionCache.valid) ComputeAllDelaysAndEnergy();

    if (p_air)   *p_air   = gActionCache.air[action];
    if (p_bh)    *p_bh    = gActionCache.bh[action];
    if (p_comp)  *p_comp  = gActionCache.comp[action];
    if (p_queue) *p_queue = gActionCache.queue[action];
    return gActionCache.delays[action];
}

static double ComputeEnergy(int action) {
    if (gActionCache.valid && action >= 0 && action < kNumActions) {
        return gActionCache.energies[action];
    }
    return ComputeEnergyAction(action);
}

// ============================================================
//  CSV LOGGING
// ============================================================
static void WriteStatsToFile() {
    static bool header_written = false;
    static std::string filename = "episode_stats_v8_fixed_"
                            + std::to_string(gSeed) + ".csv";
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;

    if (!header_written) {
        file << "episode,phase,reward,raw_reward,dl_rate_pct,avg_delay_ms,avg_bh_ms,"
             << "avg_mec_bh_ms,avg_rsu_bh_ms,avg_sat_bh_ms,"
             << "avg_air_ms,avg_comp_ms,avg_queue_ms,avg_energy_j,"
             << "throughput_tps,handover_pct,home_mec_pct,cross_zone_pct,"
             << "avg_gnb_load,avg_mec_load,pct_local,pct_mec0,pct_mec1,pct_mec2,"
             << "pct_rsu0,pct_rsu1,pct_sat,avg_gnb_idx,jains_fairness,"
             << "rsu_utilization,rsu_coverage_pct,rsu_offload_success_rate,"
             << "rc_deadline,rc_delay,rc_energy,rc_load,rc_penalty,"
             << "rc_backhaul,rc_offload,rc_consistency,"
             << "converged,convergence_episode\n";
        header_written = true;
    }

    const double dl = gTotalSteps > 0 ? 100.0 * gDeadlinesMet / gTotalSteps : 0.0;
    const double avg_delay = gTotalSteps > 0 ? (gTotalDelay / gTotalSteps) * 1000.0 : 0.0;
    const double avg_bh = gTotalSteps > 0 ? (gTotalBackhaulDelay / gTotalSteps) * 1000.0 : 0.0;
    const double avg_air = gTotalSteps > 0 ? (gTotalAirDelay / gTotalSteps) * 1000.0 : 0.0;
    const double avg_comp = gTotalSteps > 0 ? (gTotalCompDelay / gTotalSteps) * 1000.0 : 0.0;
    const double avg_queue = gTotalSteps > 0 ? (gTotalQueueDelay / gTotalSteps) * 1000.0 : 0.0;
    const double avg_energy = gTotalSteps > 0 ? gTotalEnergy / gTotalSteps : 0.0;
    const double throughput = (gTotalSteps > 0 && gTotalSteps * 0.1 > 0) ?
        gDeadlinesMet / (gTotalSteps * 0.1) : 0.0;

    int total_actions = 0;
    for (int a = 0; a < 7; a++) total_actions += gActionCount[a];
    double act_pct[7] = {0.0};
    if (total_actions > 0) {
        for (int a = 0; a < 7; a++) {
            act_pct[a] = 100.0 * gActionCount[a] / total_actions;
        }
    }

    const double ho_rate = gTotalSteps > 0 ? 100.0 * gHandoverCount / gTotalSteps : 0.0;
    const double avg_gnb_idx = gTotalSteps > 0 ? gTotalGnbServing / gTotalSteps : 0.0;

    int total_bh = gHomeBackhaulCount + gCrossBackhaulCount;
    double home_pct  = total_bh > 0 ? 100.0 * gHomeBackhaulCount / total_bh : 0.0;
    double cross_pct = total_bh > 0 ? 100.0 * gCrossBackhaulCount / total_bh : 0.0;

    double avg_gnb = gTotalSteps > 0 ? gTotalGnbLoad / gTotalSteps : 0.0;
    double avg_mec = gTotalSteps > 0 ? gTotalMecUtil / gTotalSteps : 0.0;

    double avg_mec_bh = gMecBackhaulCount > 0
        ? (gTotalMecBackhaulDelay / gMecBackhaulCount) * 1000.0 : 0.0;
    double avg_rsu_bh = gRsuBackhaulCount > 0
        ? (gTotalRsuBackhaulDelay / gRsuBackhaulCount) * 1000.0 : 0.0;
    double avg_sat_bh = gSatBackhaulCount > 0
        ? (gTotalSatBackhaulDelay / gSatBackhaulCount) * 1000.0 : 0.0;

    double fairness = gTotalSteps > 0 ? gTotalJainsFairness / gTotalSteps : 1.0;

    double rsu_util = gTotalSteps > 0 ? gRsuMetrics.total_rsu_utilization / (2.0 * gTotalSteps) : 0.0;

    double rsu_coverage = gRsuMetrics.rsu_offload_attempts > 0
        ? 100.0 * gRsuMetrics.total_rsu_coverage_time / gRsuMetrics.rsu_offload_attempts
        : 0.0;

    double rsu_success_rate = gRsuMetrics.rsu_offload_attempts > 0
        ? 100.0 * gRsuMetrics.total_rsu_offload_success / gRsuMetrics.rsu_offload_attempts : 0.0;

    std::string phase = gIsTraining ? "training" : "evaluation";

    // Episode index convention: CSV records the *completed* episode index.
    file << gEpisodeCount << ","
         << phase << ","
         << gLastReward << ","
         << gLastRawReward << ","
         << dl << ","
         << avg_delay << ","
         << avg_bh << ","
         << avg_mec_bh << ","
         << avg_rsu_bh << ","
         << avg_sat_bh << ","
         << avg_air << ","
         << avg_comp << ","
         << avg_queue << ","
         << avg_energy << ","
         << throughput << ","
         << ho_rate << ","
         << home_pct << ","
         << cross_pct << ","
         << avg_gnb << ","
         << avg_mec << ","
         << act_pct[0] << ","
         << act_pct[1] << ","
         << act_pct[2] << ","
         << act_pct[3] << ","
         << act_pct[4] << ","
         << act_pct[5] << ","
         << act_pct[6] << ","
         << avg_gnb_idx << ","
         << fairness << ","
         << rsu_util << ","
         << rsu_coverage << ","
         << rsu_success_rate << ","
         << (gRwdCompDeadline    / std::max(1, gRwdCompCount)) << ","
         << (gRwdCompDelay       / std::max(1, gRwdCompCount)) << ","
         << (gRwdCompEnergy      / std::max(1, gRwdCompCount)) << ","
         << (gRwdCompLoad        / std::max(1, gRwdCompCount)) << ","
         << (gRwdCompPenalty     / std::max(1, gRwdCompCount)) << ","
         << (gRwdCompBackhaul    / std::max(1, gRwdCompCount)) << ","
         << (gRwdCompOffload     / std::max(1, gRwdCompCount)) << ","
         << (gRwdCompConsistency / std::max(1, gRwdCompCount)) << ","
         << (gConverged ? 1 : 0) << ","
         << gConvergenceEpisode << "\n";

    file.close();
}

// ============================================================
//  EXTRA INFO (JSON for Python)
// ============================================================
static std::string BuildExtraInfo() {
    const double dl = gTotalSteps > 0 ? 100.0 * gDeadlinesMet / gTotalSteps : 0.0;

    // Per-step average reward components (for diagnostic)
    const double rcn = (gRwdCompCount > 0) ? double(gRwdCompCount) : 1.0;
    const double rc_deadline    = gRwdCompDeadline    / rcn;
    const double rc_delay       = gRwdCompDelay       / rcn;
    const double rc_energy      = gRwdCompEnergy      / rcn;
    const double rc_load        = gRwdCompLoad        / rcn;
    const double rc_penalty     = gRwdCompPenalty     / rcn;
    const double rc_backhaul    = gRwdCompBackhaul    / rcn;
    const double rc_offload     = gRwdCompOffload     / rcn;
    const double rc_consistency = gRwdCompConsistency / rcn;

    double avg_delay = gTotalSteps > 0 ? (gTotalDelay / gTotalSteps) * 1000.0 : 0.0;
    avg_delay = std::min(avg_delay, 5000.0);
    double avg_bh = gTotalSteps > 0 ? (gTotalBackhaulDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_air = gTotalSteps > 0 ? (gTotalAirDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_comp = gTotalSteps > 0 ? (gTotalCompDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_queue = gTotalSteps > 0 ? (gTotalQueueDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_gnb_idx = gTotalSteps > 0 ? gTotalGnbServing / gTotalSteps : 0.0;
    double avg_energy = gTotalSteps > 0 ? gTotalEnergy / gTotalSteps : 0.0;
    double fairness = gTotalSteps > 0 ? gTotalJainsFairness / gTotalSteps : 1.0;

    double rsu_util = gTotalSteps > 0 ? gRsuMetrics.total_rsu_utilization / (2.0 * gTotalSteps) : 0.0;
    double rsu_success_rate = gRsuMetrics.rsu_offload_attempts > 0
        ? 100.0 * gRsuMetrics.total_rsu_offload_success / gRsuMetrics.rsu_offload_attempts : 0.0;
    double rsu_coverage_pct = gRsuMetrics.rsu_offload_attempts > 0
        ? 100.0 * gRsuMetrics.total_rsu_coverage_time / gRsuMetrics.rsu_offload_attempts : 0.0;

    int total_bh = gHomeBackhaulCount + gCrossBackhaulCount;
    double home_pct = total_bh > 0 ? 100.0 * gHomeBackhaulCount / total_bh : 0.0;
    double cross_pct = total_bh > 0 ? 100.0 * gCrossBackhaulCount / total_bh : 0.0;
    double ho_rate = gTotalSteps > 0 ? 100.0 * gHandoverCount / gTotalSteps : 0.0;
    double avg_gnb = gTotalSteps > 0 ? gTotalGnbLoad / gTotalSteps : 0.0;
    double avg_mec = gTotalSteps > 0 ? gTotalMecUtil / gTotalSteps : 0.0;
    double throughput = (gTotalSteps > 0) ? gDeadlinesMet / (gTotalSteps * 0.1) : 0.0;

    int total_actions = 0;
    for (int a = 0; a < 7; a++) total_actions += gActionCount[a];
    double act_pct[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (total_actions > 0) {
        for (int a = 0; a < 7; a++) {
            act_pct[a] = 100.0 * gActionCount[a] / total_actions;
        }
    }

    // v8.6.3: per-action deadline-success helper
    auto pct = [](int met, int att) -> double {
        return att > 0 ? 100.0 * met / att : 0.0;
    };

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{"
       << "\"reward\":" << gLastReward << ","
       << "\"raw_reward\":" << gLastRawReward << ","
       << "\"dl_rate\":" << dl << ","
       << "\"step\":" << gStepCount << ","
       << "\"episode\":" << gEpisodeCount << ","
       << "\"phase\":\"" << (gIsTraining ? "training" : "evaluation") << "\","
       << "\"deadlines_met\":" << gDeadlinesMet << ","
       << "\"total_steps\":" << gTotalSteps << ","
       << "\"serving_gnb\":" << gUe.serving_gnb << ","
       << "\"avg_gnb_idx\":" << avg_gnb_idx << ","
       << "\"handover\":" << (gUe.handover ? 1 : 0) << ","
       << "\"handover_count\":" << gHandoverCount << ","
       << "\"handover_rate\":" << ho_rate << ","
       << "\"avg_delay_ms\":" << avg_delay << ","
       << "\"avg_bh_ms\":" << avg_bh << ","
       << "\"avg_air_ms\":" << avg_air << ","
       << "\"avg_comp_ms\":" << avg_comp << ","
       << "\"avg_queue_ms\":" << avg_queue << ","
       << "\"avg_energy_j\":" << avg_energy << ","
       << "\"throughput_tps\":" << throughput << ","
       << "\"jains_fairness\":" << fairness << ","

       << "\"rc_deadline\":"    << rc_deadline << ","
       << "\"rc_delay\":"       << rc_delay << ","
       << "\"rc_energy\":"      << rc_energy << ","
       << "\"rc_load\":"        << rc_load << ","
       << "\"rc_penalty\":"     << rc_penalty << ","
       << "\"rc_backhaul\":"    << rc_backhaul << ","
       << "\"rc_offload\":"     << rc_offload << ","
       << "\"rc_consistency\":" << rc_consistency << ","
       << "\"rc_count\":"       << gRwdCompCount << ","

       << "\"avg_gnb_load\":" << avg_gnb << ","
       << "\"avg_mec_load\":" << avg_mec << ","
       << "\"home_mec_pct\":" << home_pct << ","
       << "\"cross_zone_pct\":" << cross_pct << ","
       << "\"avg_mec_bh_ms\":" << (gMecBackhaulCount > 0 ? (gTotalMecBackhaulDelay / gMecBackhaulCount) * 1000.0 : 0.0) << ","
       << "\"avg_rsu_bh_ms\":" << (gRsuBackhaulCount > 0 ? (gTotalRsuBackhaulDelay / gRsuBackhaulCount) * 1000.0 : 0.0) << ","
       << "\"avg_sat_bh_ms\":" << (gSatBackhaulCount > 0 ? (gTotalSatBackhaulDelay / gSatBackhaulCount) * 1000.0 : 0.0) << ","
       << "\"rsu_utilization\":" << rsu_util << ","
       << "\"rsu_success_rate\":" << rsu_success_rate << ","
       << "\"rsu_coverage_pct\":" << rsu_coverage_pct << ","
       << "\"rsu_offload_attempts\":" << gRsuMetrics.rsu_offload_attempts << ","
       << "\"dl_local\":" << pct(gActionDeadlinesMet[0], gActionAttempts[0]) << ","
       << "\"dl_mec0\":"  << pct(gActionDeadlinesMet[1], gActionAttempts[1]) << ","
       << "\"dl_mec1\":"  << pct(gActionDeadlinesMet[2], gActionAttempts[2]) << ","
       << "\"dl_mec2\":"  << pct(gActionDeadlinesMet[3], gActionAttempts[3]) << ","
       << "\"dl_rsu0\":"  << pct(gActionDeadlinesMet[4], gActionAttempts[4]) << ","
       << "\"dl_rsu1\":"  << pct(gActionDeadlinesMet[5], gActionAttempts[5]) << ","
       << "\"dl_sat\":"   << pct(gActionDeadlinesMet[6], gActionAttempts[6]) << ","
       << "\"act_local\":" << act_pct[0] << ","
       << "\"act_mec0\":"  << act_pct[1] << ","
       << "\"act_mec1\":"  << act_pct[2] << ","
       << "\"act_mec2\":"  << act_pct[3] << ","
       << "\"act_rsu0\":"  << act_pct[4] << ","
       << "\"act_rsu1\":"  << act_pct[5] << ","
       << "\"act_sat\":"   << act_pct[6] << ","
       << "\"converged\":" << (gConverged ? 1 : 0) << ","
       << "\"convergence_episode\":" << gConvergenceEpisode
       << "}";

    return ss.str();
}

// ============================================================
//  OBSERVATION BUILDING
// ============================================================
static Ptr<OpenGymBoxContainer<float>> BuildObservation() {
    auto obs = CreateObject<OpenGymBoxContainer<float>>();

    ComputeAllDelaysAndEnergy();

    const double MAX_CPU = 5.0;
    const double MAX_DATA = 50.0;
    const double MAX_DEADLINE = 1.0;

    // 0-1: task features
    obs->AddValue(static_cast<float>(SafeNorm(gUe.cpu_cycles_giga, MAX_CPU)));
    obs->AddValue(static_cast<float>(SafeNorm(gUe.data_size_mb, MAX_DATA)));

    // 2-4: ONE-HOT UE type encoding
    obs->AddValue(gUe.ue_type == 0 ? 1.0f : 0.0f);
    obs->AddValue(gUe.ue_type == 1 ? 1.0f : 0.0f);
    obs->AddValue(gUe.ue_type == 2 ? 1.0f : 0.0f);

    // 5: deadline norm
    obs->AddValue(static_cast<float>(SafeNorm(gUe.deadline, MAX_DEADLINE)));

    // 6: urgency
    double best_delay = 1.0e9;
    for (int a = 0; a < kNumActions; a++) {
        double d = gActionCache.delays[a];
        if (std::isfinite(d) && d > 0 && d < best_delay) {
            best_delay = d;
        }
    }
    double urgency = (std::isfinite(best_delay) && gUe.deadline > 0) ?
        Clamp(best_delay / gUe.deadline, 0.0, 1.0) : 0.5;
    obs->AddValue(static_cast<float>(urgency));

    // 7-9: MEC loads
    for (int m = 0; m < 3; m++) {
        obs->AddValue(static_cast<float>(Clamp(gMec[m].load, 0.0, 1.0)));
    }

    // 10-11: RSU loads
    for (int r = 0; r < 2; r++) {
        obs->AddValue(static_cast<float>(Clamp(gRsu[r].load, 0.0, 1.0)));
    }

    // 12-18: delay ratios (7 actions)
    for (int a = 0; a < kNumActions; a++) {
        double d = gActionCache.delays[a];
        double deadline = std::max(gUe.deadline, 0.001);
        double ratio = (std::isfinite(d) && d > 0) ? d / deadline : 1.0;
        obs->AddValue(static_cast<float>(Clamp(ratio / 3.0, 0.0, 1.0)));
    }

    // 19-24: distances to MEC
    for (int m = 0; m < 3; m++) {
        double dx = (gUe.x - gMec[m].x) / kAreaSize;
        double dy = (gUe.y - gMec[m].y) / kAreaSize;
        obs->AddValue(static_cast<float>(Clamp(dx, -1.0, 1.0)));
        obs->AddValue(static_cast<float>(Clamp(dy, -1.0, 1.0)));
    }

    // 25-28: distances to RSU
    for (int r = 0; r < 2; r++) {
        double dx = (gUe.x - gRsu[r].x) / kAreaSize;
        double dy = (gUe.y - gRsu[r].y) / kAreaSize;
        obs->AddValue(static_cast<float>(Clamp(dx, -1.0, 1.0)));
        obs->AddValue(static_cast<float>(Clamp(dy, -1.0, 1.0)));
    }

    // 29: sat available
    double sat_available = (gUe.speed < kMaxSpeed) ? 1.0 : 0.0;
    obs->AddValue(static_cast<float>(sat_available));

    // 30: speed
    obs->AddValue(static_cast<float>(SafeNorm(gUe.speed, kMaxSpeed)));

    // 31-33: MEC queues
    for (int m = 0; m < 3; m++) {
        obs->AddValue(static_cast<float>(SafeNorm(gMec[m].queue, kMaxQueue)));
    }

    // 34-35: RSU queues
    for (int r = 0; r < 2; r++) {
        obs->AddValue(static_cast<float>(SafeNorm(gRsu[r].queue, kMaxQueue)));
    }

    // 36: fading
    obs->AddValue(static_cast<float>(Clamp(gUe.fading, 0.0, 1.0)));

    // 37: serving gNB
    obs->AddValue(static_cast<float>(Clamp(gUe.serving_gnb / (kNumGnb - 1.0), 0.0, 1.0)));

    // 38: air SINR
    obs->AddValue(static_cast<float>(NormSINR(gUe.air_sinr_dB)));

    // 39: handover
    obs->AddValue(gUe.handover ? 1.0f : 0.0f);

    // 40-42: backhaul delays
    const int k = gUe.serving_gnb;
    if (k >= 0 && k < kNumGnb) {
        for (int m = 0; m < 3; m++) {
            obs->AddValue(static_cast<float>(SafeNorm(gGnb[k].bh_delay[m], kMaxBhDelay_norm)));
        }
    } else {
        for (int m = 0; m < 3; m++) obs->AddValue(0.0f);
    }

    // 43: gNB load
    double gnb_load = (k >= 0 && k < kNumGnb) ? gGnb[k].load : 0.0;
    obs->AddValue(static_cast<float>(Clamp(gnb_load, 0.0, 1.0)));

    // 44: distance to gNB
    double d_gnb = (k >= 0 && k < kNumGnb) ?
        EuclidDist(gUe.x, gUe.y, gGnb[k].x, gGnb[k].y) : 0.0;
    obs->AddValue(static_cast<float>(SafeNorm(d_gnb, 500.0)));

    // 45-47: path delays
    if (k >= 0 && k < kNumGnb) {
        for (int m = 0; m < 3; m++) {
            double path = gUe.air_tx_delay_s + gGnb[k].bh_delay[m];
            obs->AddValue(static_cast<float>(SafeNorm(path, kMaxPathDelay_s)));
        }
    } else {
        for (int m = 0; m < 3; m++) obs->AddValue(0.0f);
    }

    return obs;
}

// ============================================================
//  OPENGYM CALLBACKS
// ============================================================
Ptr<OpenGymSpace> GetObservationSpace() {
    std::vector<float> low(kObsSize, 0.0f);
    std::vector<float> high(kObsSize, 1.0f);
    for (int i = 19; i <= 28; i++) {
        low[i] = -1.0f;
        high[i] = 1.0f;
    }
    std::vector<uint32_t> shape = {kObsSize};
    return CreateObject<OpenGymBoxSpace>(low, high, shape, "float32");
}

Ptr<OpenGymSpace> GetActionSpace() {
    return CreateObject<OpenGymDiscreteSpace>(kNumActions);
}

bool GetGameOver() {
    return (gEpisodeCount >= kMaxEpisodes);
}

Ptr<OpenGymDataContainer> GetObservation() {
    return BuildObservation();
}

float GetReward() {
    return gLastReward;
}

std::string GetExtraInfo() {
    return BuildExtraInfo();
}

// ============================================================
//  PRINT EPISODE SUMMARY
// ============================================================
static void PrintEpisodeSummary(bool detailed = false) {
    const double dl = gTotalSteps > 0 ? 100.0 * gDeadlinesMet / gTotalSteps : 0.0;
    double avg_delay = gTotalSteps > 0 ? (gTotalDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_air = gTotalSteps > 0 ? (gTotalAirDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_comp = gTotalSteps > 0 ? (gTotalCompDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_queue = gTotalSteps > 0 ? (gTotalQueueDelay / gTotalSteps) * 1000.0 : 0.0;
    double avg_gnb_idx = gTotalSteps > 0 ? gTotalGnbServing / gTotalSteps : 0.0;

    int total_bh = gHomeBackhaulCount + gCrossBackhaulCount;
    double home_pct = total_bh > 0 ? 100.0 * gHomeBackhaulCount / total_bh : 0.0;
    double cross_pct = total_bh > 0 ? 100.0 * gCrossBackhaulCount / total_bh : 0.0;

    double ho_rate = gTotalSteps > 0 ? 100.0 * gHandoverCount / gTotalSteps : 0.0;
    double fairness = gTotalSteps > 0 ? gTotalJainsFairness / gTotalSteps : 1.0;

    int total_actions = 0;
    for (int a = 0; a < 7; a++) total_actions += gActionCount[a];
    double act_pct[7] = {0.0};
    if (total_actions > 0) {
        for (int a = 0; a < 7; a++) {
            act_pct[a] = 100.0 * gActionCount[a] / total_actions;
        }
    }

    double avg_gnb = gTotalSteps > 0 ? gTotalGnbLoad / gTotalSteps : 0.0;
    double avg_mec = gTotalSteps > 0 ? gTotalMecUtil / gTotalSteps : 0.0;
    double avg_energy = gTotalSteps > 0 ? gTotalEnergy / gTotalSteps : 0.0;
    double throughput = (gTotalSteps > 0) ? gDeadlinesMet / (gTotalSteps * 0.1) : 0.0;

    double avg_mec_bh = gMecBackhaulCount > 0
        ? (gTotalMecBackhaulDelay / gMecBackhaulCount) * 1000.0 : 0.0;
    double avg_rsu_bh = gRsuBackhaulCount > 0
        ? (gTotalRsuBackhaulDelay / gRsuBackhaulCount) * 1000.0 : 0.0;
    double avg_sat_bh = gSatBackhaulCount > 0
        ? (gTotalSatBackhaulDelay / gSatBackhaulCount) * 1000.0 : 0.0;

    double rsu_coverage = gRsuMetrics.rsu_offload_attempts > 0
        ? 100.0 * gRsuMetrics.total_rsu_coverage_time / gRsuMetrics.rsu_offload_attempts
        : 0.0;

    double rsu_success_rate = gRsuMetrics.rsu_offload_attempts > 0
        ? 100.0 * gRsuMetrics.total_rsu_offload_success / gRsuMetrics.rsu_offload_attempts : 0.0;

    std::string phase = gIsTraining ? "TRAINING" : "EVALUATION";

    if (detailed || (gEpisodeCount + 1) % 20 == 0 ||
        (gEpisodeCount + 1) >= kMaxEpisodes - 10) {

        NS_LOG_UNCOND("\n" << std::string(70, '='));
        NS_LOG_UNCOND("📊 " << phase << " EPISODE " << gEpisodeCount + 1 << " / " << kMaxEpisodes);
        NS_LOG_UNCOND(std::string(70, '='));

        if (gConverged && gConvergenceLogged) {
            NS_LOG_UNCOND("🎯 CONVERGED! Policy is stable.");
        }

        NS_LOG_UNCOND("\n📈 Performance Metrics:");
        NS_LOG_UNCOND("  Reward              : " << std::fixed << std::setprecision(2) << gLastReward);
        NS_LOG_UNCOND("  Raw Reward          : " << std::fixed << std::setprecision(2) << gLastRawReward);
        NS_LOG_UNCOND("  Deadline Success    : " << std::fixed << std::setprecision(1) << dl << "%");
        NS_LOG_UNCOND("  Throughput          : " << std::fixed << std::setprecision(2) << throughput << " tasks/s");
        NS_LOG_UNCOND("  Avg gNB Index       : " << std::fixed << std::setprecision(1) << avg_gnb_idx);
        NS_LOG_UNCOND("  Jain's Fairness     : " << std::fixed << std::setprecision(3) << fairness);

        NS_LOG_UNCOND("\n⏱️  Delay Breakdown (ms):");
        NS_LOG_UNCOND("  Air Interface    : " << std::fixed << std::setprecision(2) << avg_air);
        NS_LOG_UNCOND("  Backhaul (MEC)   : " << std::fixed << std::setprecision(2) << avg_mec_bh);
        NS_LOG_UNCOND("  Backhaul (RSU)   : " << std::fixed << std::setprecision(2) << avg_rsu_bh);
        NS_LOG_UNCOND("  Backhaul (SAT)   : " << std::fixed << std::setprecision(2) << avg_sat_bh);
        NS_LOG_UNCOND("  Processing       : " << std::fixed << std::setprecision(2) << avg_comp);
        NS_LOG_UNCOND("  Queuing          : " << std::fixed << std::setprecision(2) << avg_queue);
        NS_LOG_UNCOND("  ─────────────────────────");
        NS_LOG_UNCOND("  Total            : " << std::fixed << std::setprecision(2) << avg_delay);

        NS_LOG_UNCOND("\n⚡ Energy & Load:");
        NS_LOG_UNCOND("  Energy           : " << std::fixed << std::setprecision(2) << avg_energy << " J");
        NS_LOG_UNCOND("  gNB Load         : " << std::fixed << std::setprecision(2) << avg_gnb);
        NS_LOG_UNCOND("  MEC Utilization  : " << std::fixed << std::setprecision(2) << avg_mec);
        NS_LOG_UNCOND("  Handover Rate    : " << std::fixed << std::setprecision(1) << ho_rate << "%");
        NS_LOG_UNCOND("  Home MEC BH      : " << std::fixed << std::setprecision(1) << home_pct << "%");
        NS_LOG_UNCOND("  Cross-zone BH    : " << std::fixed << std::setprecision(1) << cross_pct << "%");

        NS_LOG_UNCOND("\n🚗 RSU Metrics:");
        NS_LOG_UNCOND("  RSU Success Rate : " << std::fixed << std::setprecision(1) << rsu_success_rate << "%");
        NS_LOG_UNCOND("  RSU Attempts     : " << std::fixed << std::setprecision(0) << gRsuMetrics.rsu_offload_attempts);
        NS_LOG_UNCOND("  RSU Coverage     : " << std::fixed << std::setprecision(1) << rsu_coverage << "%");

        NS_LOG_UNCOND("\n🎯 Action Distribution:");
        NS_LOG_UNCOND("  LOCAL    : " << std::fixed << std::setprecision(1) << act_pct[0] << "%");
        NS_LOG_UNCOND("  MEC0     : " << std::fixed << std::setprecision(1) << act_pct[1] << "%");
        NS_LOG_UNCOND("  MEC1     : " << std::fixed << std::setprecision(1) << act_pct[2] << "%");
        NS_LOG_UNCOND("  MEC2     : " << std::fixed << std::setprecision(1) << act_pct[3] << "%");
        NS_LOG_UNCOND("  RSU0     : " << std::fixed << std::setprecision(1) << act_pct[4] << "%");
        NS_LOG_UNCOND("  RSU1     : " << std::fixed << std::setprecision(1) << act_pct[5] << "%");
        NS_LOG_UNCOND("  SAT      : " << std::fixed << std::setprecision(1) << act_pct[6] << "%");

        NS_LOG_UNCOND("\n" << std::string(70, '=') << "\n");
    }
}

// ============================================================
//  EXECUTE ACTIONS
// ============================================================
bool ExecuteActions(Ptr<OpenGymDataContainer> action) {
    try {
        gEpisodeJustEnded = false;

        auto disc = DynamicCast<OpenGymDiscreteContainer>(action);
        if (!disc) {
            NS_LOG_ERROR("Invalid action container");
            return false;
        }

        int act = static_cast<int>(disc->GetValue());
        act = Clamp(act, 0, kNumActions - 1);

        double t_air = 0, t_bh = 0, t_comp = 0, t_queue = 0;
        double delay = ActionDelay(act, &t_air, &t_bh, &t_comp, &t_queue);
        double energy = ComputeEnergy(act);
        double reward = ComputeReward(act, delay, energy);

        if (!std::isfinite(reward) || std::isnan(reward)) {
            reward = 0.0;
        }

        gLastReward = static_cast<float>(reward);
        gTotalSteps++;

        gActionCount[act]++;
        gActionAttempts[act]++;
        if (delay <= gUe.deadline) gActionDeadlinesMet[act]++;
        if (gUe.handover) gHandoverCount++;

        // RSU metric counting
        if (act >= kActRsu0 && act <= kActRsu1) {
            int r = act - 4;
            gRsuMetrics.rsu_offload_attempts += 1.0;
            if (IsWithinRsuRange(r)) {
                gRsuMetrics.total_rsu_coverage_time += 1.0;
                if (delay <= gUe.deadline) {
                    gRsuMetrics.total_rsu_offload_success += 1.0;
                }
            }
        }

        gTotalDelay += delay;
        gTotalAirDelay += t_air;

        if (act >= kActMec0 && act <= kActMec2) {
            gTotalMecBackhaulDelay += t_bh;
            gMecBackhaulCount++;
        } else if (act >= kActRsu0 && act <= kActRsu1) {
            gTotalRsuBackhaulDelay += t_bh;
            gRsuBackhaulCount++;
        } else if (act == kActSat) {
            gTotalSatBackhaulDelay += t_bh;
            gSatBackhaulCount++;
        }
        gTotalBackhaulDelay += t_bh;

        gTotalCompDelay += t_comp;
        gTotalQueueDelay += t_queue;
        gTotalEnergy += energy;

        if (act >= kActMec0 && act <= kActMec2) {
            int m = act - 1;
            int home_mec = kGnbHomeMec[gUe.serving_gnb];
            if (m == home_mec) {
                gHomeBackhaulCount++;
            } else {
                gCrossBackhaulCount++;
            }
        }

        if (gUe.serving_gnb >= 0 && gUe.serving_gnb < kNumGnb) {
            gTotalGnbLoad += gGnb[gUe.serving_gnb].load;
        }
        gTotalGnbServing += gUe.serving_gnb;

        double mec_util_sum = 0.0;
        for (int m = 0; m < 3; m++) {
            mec_util_sum += gMec[m].load;
        }
        gTotalMecUtil += mec_util_sum / 3.0;

        gTotalJainsFairness += ComputeJainsFairnessIndex();

        UpdateLoads(act);
        gStepCount++;

        if (gStepCount >= kStepsPerEpisode) {
            const double ep_dl = gTotalSteps > 0 ? 100.0 * gDeadlinesMet / gTotalSteps : 0.0;
            const double avg_air = gTotalSteps > 0 ? (gTotalAirDelay / gTotalSteps) * 1000.0 : 0.0;
            const double avg_comp = gTotalSteps > 0 ? (gTotalCompDelay / gTotalSteps) * 1000.0 : 0.0;
            const double avg_queue = gTotalSteps > 0 ? (gTotalQueueDelay / gTotalSteps) * 1000.0 : 0.0;
            const double avg_gnb = gTotalSteps > 0 ? gTotalGnbLoad / gTotalSteps : 0.0;

            // Feed raw episode DL to convergence check (no double-smoothing).
            bool converged = CheckConvergence(ep_dl);

            if (converged && !gConvergenceLogged) {
                gConvergenceLogged = true;
                gConverged = true;
                if (gConvergenceEpisode == -1) {
                    gConvergenceEpisode = gEpisodeCount;
                }
            }

            std::string phase = gIsTraining ? "TRAIN" : "EVAL";

            // v8.6.4: suppress the inline log in eval mode; the Python
            // harness prints an equivalent compact line.
            if (gIsTraining && (gEpisodeCount + 1) % 20 == 0) {
                NS_LOG_UNCOND("[" << phase << " Ep " << gEpisodeCount + 1 << "/" << kMaxEpisodes
                            << "]  DL%=" << ep_dl
                            << "  Air=" << avg_air << "ms"
                            << "  Comp=" << avg_comp << "ms"
                            << "  Q=" << avg_queue << "ms"
                            << "  gNB=" << gUe.serving_gnb
                            << "  gNBload=" << avg_gnb
                            << "  R=" << reward
                            << (converged ? " 🎯CONVERGED" : ""));
            }

            // v8.6.3: suppress the full block in evaluation mode, since
            // the Python harness prints an equivalent compact line.
            const bool print_full_summary =
                gIsTraining && (
                    (gEpisodeCount < 3) ||
                    ((gEpisodeCount + 1) % 50 == 0) ||
                    ((gEpisodeCount + 1) >= kMaxEpisodes - 5)
                );

            if (print_full_summary) {
                PrintEpisodeSummary(true);
            }

            if (gIsTraining) {
                WriteStatsToFile();
            }

            gEpisodeCount++;

            if (gEpisodeCount >= kMaxEpisodes) {
                std::string done_phase = gIsTraining ? "TRAINING" : "EVALUATION";
                const double final_fairness = gTotalSteps > 0 ? gTotalJainsFairness / gTotalSteps : 1.0;

                NS_LOG_UNCOND("\n" << std::string(70, '='));
                NS_LOG_UNCOND("✅ " << done_phase << " COMPLETE!");
                NS_LOG_UNCOND("  Total episodes       : " << kMaxEpisodes);
                NS_LOG_UNCOND("  Final DL Rate        : " << std::fixed << std::setprecision(1) << ep_dl << "%");
                NS_LOG_UNCOND("  Final Reward         : " << std::fixed << std::setprecision(2) << gLastReward);
                NS_LOG_UNCOND("  Final Raw Reward     : " << std::fixed << std::setprecision(2) << gLastRawReward);
                NS_LOG_UNCOND("  Final Fairness       : " << std::fixed << std::setprecision(3) << final_fairness);
                NS_LOG_UNCOND("  RSU Success Rate     : " << std::fixed << std::setprecision(1) <<
                    (gRsuMetrics.rsu_offload_attempts > 0 ? 100.0 * gRsuMetrics.total_rsu_offload_success / gRsuMetrics.rsu_offload_attempts : 0.0) << "%");

                if (gIsTraining) {
                    if (gConverged) {
                        NS_LOG_UNCOND("🎯 Policy converged at episode " << gConvergenceEpisode);
                    } else {
                        NS_LOG_UNCOND("⚠️  Policy did not converge within " << kMaxEpisodes << " episodes");
                    }
                } else {
                    NS_LOG_UNCOND("✅ Evaluation complete. No convergence check in eval mode.");
                }
                NS_LOG_UNCOND(std::string(70, '=') << "\n");

                Simulator::Stop();
                return true;
            }

            gEpisodeJustEnded = true;
            ResetEnvironment();
            return true;
        }

        StepMobility();
        return true;

    } catch (const std::exception& e) {
        NS_LOG_ERROR("Exception in ExecuteActions (env reset, step treated as ok): "
                     << e.what());
        gEpisodeJustEnded = true;
        ResetEnvironment();
        return true;
    }
}

// ============================================================
//  STEP FUNCTION
// ============================================================
void Step(Ptr<OpenGymInterface> gym) {
    gym->NotifyCurrentState();
    Simulator::Schedule(Seconds(kStepInterval_s), &Step, gym);
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char* argv[]) {
    bool evaluate_flag = false;

    CommandLine cmd;
    cmd.AddValue("seed", "RNG seed", gSeed);
    cmd.AddValue("port", "OpenGym ZMQ port", gPort);
    cmd.AddValue("evaluate", "Run evaluation phase", evaluate_flag);
    cmd.Parse(argc, argv);

    gIsTraining = !evaluate_flag;

    gRng = std::make_unique<RobustRNG>(gSeed);

    NS_LOG_UNCOND("==============================================");
    NS_LOG_UNCOND("  MEC Simulator v8 - FULLY FIXED (v8.6.17)");
    NS_LOG_UNCOND("  THESIS VERSION - REALISTIC DELAYS");
    NS_LOG_UNCOND("==============================================");
    NS_LOG_UNCOND("  Seed             = " << gSeed);
    NS_LOG_UNCOND("  Port             = " << gPort);
    NS_LOG_UNCOND("  Phase            = " << (gIsTraining ? "TRAINING" : "EVALUATION"));
    NS_LOG_UNCOND("  Observation Size = " << kObsSize);
    NS_LOG_UNCOND("  Actions          = " << kNumActions);
    NS_LOG_UNCOND("  Step Interval    = " << kStepInterval_s << " s");
    NS_LOG_UNCOND("  RSU Max Range    = " << kRsuMaxRange_m << "m");
    NS_LOG_UNCOND("  RSU Freq         = " << kRsuFreq_GHz << " GHz");
    NS_LOG_UNCOND("  Reward Weights   = DL:" << kWeightDeadline << ", Delay:" << kWeightDelay
                << ", Energy:" << kWeightEnergy << ", Load:" << kWeightLoad
                << ", Pen:" << kWeightPenalty << ", BH:" << kWeightBackhaul
                << " (Sum=" << (kWeightDeadline + kWeightDelay + kWeightEnergy +
                                kWeightLoad + kWeightPenalty + kWeightBackhaul) << ") ✓");
    NS_LOG_UNCOND("  Reward Norm      = " << (kUseRunningNormalization ? "running (Welford)" : "fixed"));
    NS_LOG_UNCOND("  Load Increment   = " << kLoadIncrement);
    NS_LOG_UNCOND("  Handover Hyst    = " << kHandoverHysteresis_dB
                << "dB (TTT=" << kHandoverTTTSteps << " steps)");
    NS_LOG_UNCOND("==============================================");

    InitServers();
    InitGnbs();
    InitInterferers();
    ResetEnvironment();

    Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface>(gPort);

    openGym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
    openGym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
    openGym->SetGetObservationCb(MakeCallback(&GetObservation));
    openGym->SetGetGameOverCb(MakeCallback(&GetGameOver));
    openGym->SetGetRewardCb(MakeCallback(&GetReward));
    openGym->SetGetExtraInfoCb(MakeCallback(&GetExtraInfo));
    openGym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));

    NS_LOG_UNCOND("Waiting for Python agent on port tcp://localhost:" << gPort);
    Simulator::Schedule(Seconds(0.1), &Step, openGym);
    Simulator::Stop(Seconds(86400.0));

    try {
        Simulator::Run();
    } catch (const std::exception& e) {
        NS_LOG_ERROR("Simulator crashed: " << e.what());
        return 1;
    }

    Simulator::Destroy();
    openGym->NotifySimulationEnd();

    NS_LOG_UNCOND("✅ Simulator finished successfully");

    return 0;
}