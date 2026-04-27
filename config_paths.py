"""
config_paths.py — SINGLE SOURCE OF TRUTH
=========================================
Supersedes 00_config.py. Do not edit 00_config.py.

All scripts import from this file only.
Update PROJECT_ROOT to match your machine; everything else is auto-derived.
"""
import os
from pathlib import Path

# ── THE ONLY LINE YOU EVER CHANGE ─────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\Projects\diabetes_prediction_project\federated")
# ─────────────────────────────────────────────────────────────────────────────

# Legacy string form kept for scripts that use os.path.join
PROJECT_ROOT_STR = str(PROJECT_ROOT)

# Derived directories
DATA_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR   = PROJECT_ROOT / "results"
MODELS_DIR    = PROJECT_ROOT / "models"
PLOTS_DIR     = PROJECT_ROOT / "plots"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"   # NEW: canonical artefacts dir

for _d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, ARTEFACTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── DATA PATHS ────────────────────────────────────────────────────────────────
NHANES_PATH      = DATA_DIR / "centralised_full.csv"      # NHANES processed data
BRFSS_PATH       = Path(r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv")
CENTRALISED_PATH = DATA_DIR / "centralised_full.csv"

# Global scaler — fitted ONCE on NHANES train split, loaded everywhere
GLOBAL_SCALER_PATH = ARTEFACTS_DIR / "global_nhanes_scaler.joblib"

# Node data paths
NODE_A_PATH  = DATA_DIR / "node_a_young_urban.csv"
NODE_B_PATH  = DATA_DIR / "node_b_elderly_rural.csv"
NODE_C_PATH  = DATA_DIR / "node_c_mixed_metro.csv"
NODE_PATHS   = [NODE_A_PATH, NODE_B_PATH, NODE_C_PATH]
NODE_NAMES   = ["Node A — Young Urban", "Node B — Elderly Rural", "Node C — Mixed Metro"]

# ── FEATURES ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'RIDAGEYR',      # Age (continuous)
    'RIAGENDR',      # Sex (1=Male, 2=Female)
    'RIDRETH3',      # Race/ethnicity
    'BMXBMI',        # BMI (continuous)
    'SMOKING',       # 0=never, 1=former, 2=current
    'PHYS_ACTIVITY', # 0=inactive, 1=active
    'HEART_ATTACK',  # binary
    'STROKE',        # binary
]
TARGET_COL = 'DIABETES'
LABEL_COL  = TARGET_COL   # alias used by new scripts
N_FEATURES = len(FEATURE_COLS)

# ── PUBLISHED BASELINES ───────────────────────────────────────────────────────
PUBLISHED_INTERNAL_AUC  = 0.794
PUBLISHED_EXTERNAL_AUC  = 0.717
PUBLISHED_ELDERLY_AUC   = 0.607   # age >= 60
PUBLISHED_YOUNG_AUC     = 0.742   # age 18-39
PUBLISHED_ELDERLY_GAP   = 0.135   # young - elderly
PUBLISHED_BRIER         = 0.123   # original key kept for backward compat
PUBLISHED_BRIER_SCORE   = 0.123   # consistent naming — new scripts use this
PUBLISHED_AUC           = 0.794   # alias for PUBLISHED_INTERNAL_AUC
PUBLISHED_F1            = 0.518
PUBLISHED_SENSITIVITY   = 0.762
PUBLISHED_SPECIFICITY   = 0.695

# ── XGBOOST ──────────────────────────────────────────────────────────────────
# use_label_encoder REMOVED — deprecated since XGBoost >= 1.6
XGB_PARAMS = dict(
    learning_rate    = 0.11,
    max_depth        = 6,
    n_estimators     = 240,
    subsample        = 0.85,
    colsample_bytree = 0.80,
    min_child_weight = 3,
    eval_metric      = 'logloss',
    random_state     = 42,
    # 'use_label_encoder': False  <-- REMOVED, deprecated since XGBoost 1.6
)

# ── NEURAL NETWORK ────────────────────────────────────────────────────────────
NN_HIDDEN_DIMS   = [64, 32, 16]
NN_DROPOUT       = 0.3
NN_LR            = 1e-3
NN_LOCAL_EPOCHS  = 5
<<<<<<< HEAD
NN_BATCH_SIZE    = 64
=======
NN_BATCH_SIZE    = 256   # 256 saturates RTX 4060 Tensor Cores; use 64 for CPU-only
>>>>>>> 435718c297f04a6b74b12d2ac00504407237e06b
NN_WEIGHT_DECAY  = 1e-4

# Aliases for new scripts
NET_HIDDEN_LAYERS   = NN_HIDDEN_DIMS
NET_DROPOUT_RATE    = NN_DROPOUT
FL_LEARNING_RATE    = NN_LR
FL_LOCAL_EPOCHS_DEFAULT = NN_LOCAL_EPOCHS
FL_BATCH_SIZE       = NN_BATCH_SIZE
FL_WEIGHT_DECAY     = NN_WEIGHT_DECAY

<<<<<<< HEAD
=======
# ── HARDWARE ACCELERATION ─────────────────────────────────────────────────────
# Tuned for i7-13650HX + RTX 4060 (8 GB VRAM). All flags are checked at
# runtime — safe to leave as-is on a CPU-only machine.
USE_AMP      = True   # FP16 mixed-precision via torch.autocast; ~1.5x on Ampere+
NUM_WORKERS  = 0      # 0 = safest on Windows (no subprocess spawning issues);
                      # raise to 4 on Linux/WSL for larger datasets

>>>>>>> 435718c297f04a6b74b12d2ac00504407237e06b
# ── FEDERATED LEARNING ────────────────────────────────────────────────────────
FL_NUM_ROUNDS    = 50
FL_NUM_CLIENTS   = 3
FEDPROX_MU       = 0.1

# FedNova — heterogeneous local epochs (Wang et al. NeurIPS 2020)
# Nodes with greater distribution shift use FEWER local steps to prevent drift.
<<<<<<< HEAD
# Node B (elderly-rural, high shift) → fewest steps; Node A (urban, low shift) → most
=======
# Node B (elderly-rural, high shift) -> fewest steps; Node A (urban, low shift) -> most
>>>>>>> 435718c297f04a6b74b12d2ac00504407237e06b
NODE_LOCAL_EPOCHS_FEDNOVA = {
    0: 5,   # Node A — Young Urban:    low distribution shift, safe to do more local work
    1: 3,   # Node B — Elderly Rural:  HIGH distribution shift, FEWER steps to reduce drift
    2: 4,   # Node C — Mixed Metro:    intermediate heterogeneity
}
# WRONG (original — collapses FedNova to FedAvg): {0: 5, 1: 5, 2: 5}
# WRONG (previous docstring error — backwards): {0: 3, 1: 7, 2: 5}

# ── DIFFERENTIAL PRIVACY ─────────────────────────────────────────────────────
DP_TARGET_EPSILON   = 1.0
DP_TARGET_DELTA     = 1e-5
DP_MAX_GRAD_NORM    = 1.0
DP_EPSILON_LEVELS   = [0.5, 1.0, 2.0, 5.0, float('inf')]
# Aliases
DP_EPSILON = DP_TARGET_EPSILON
DP_DELTA   = DP_TARGET_DELTA

# ── FAIRNESS SUBGROUPS ────────────────────────────────────────────────────────
AGE_GROUPS = {'18-39': (18, 39), '40-59': (40, 59), '60+': (60, 130)}
BMI_GROUPS = {'Normal': (0, 24.9), 'Overweight': (25, 29.9), 'Obese': (30, 999)}
SEX_GROUPS = {'Male': 1.0, 'Female': 2.0}

# ── STATISTICAL ANALYSIS ─────────────────────────────────────────────────────
CI_ALPHA    = 0.05
N_BOOTSTRAP = 2000
TEST_SIZE   = 0.20
RANDOM_SEED = 42   # alias
SEED        = 42   # original key kept for backward compat
