"""
SCRIPT 10 — STRATIFIED CENTRALISED EXPERIMENT
==============================================
Isolates the federation effect from the data-composition effect.

RESEARCH QUESTION:
  Is the FedAvg performance difference vs centralised due to the federation
  mechanism itself, or due to the different data composition each approach sees?

CONDITIONS:
  A. Standard centralised DiabetesNet — trained on full NHANES centralised set
  B. Node-B-weighted centralised — 80% of training data sampled from Node B
     (elderly-rural, highest distribution shift). Simulates composition effect.
  C. FedAvg — load pre-trained fedavg_weights.pt (no retraining)

HONEST REPORTING:
  If condition B ≈ condition C (FedAvg), that clarifies the mechanism:
  federation overhead is minimal, and the composition of elderly-skewed data
  drives performance differences. This is a publishable finding.

INPUTS:
  data/centralised_full.csv
  data/node_b_elderly_rural.csv
  models/fedavg_weights.pt
  artefacts/global_nhanes_scaler.joblib
  results/y_true_brfss.npy         (for external comparison)
  results/pred_fedavg_external.npy

OUTPUTS:
  results/stratified_centralised_results.json

RUNTIME: ~10 minutes
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    CENTRALISED_PATH, NODE_PATHS, NODE_NAMES,
    RESULTS_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_COL, GLOBAL_SCALER_PATH,
    NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY, NN_LOCAL_EPOCHS,
    FL_NUM_ROUNDS, SEED,
)
from nn_model   import DiabetesNet, train_one_epoch, get_device
from data_utils import DiabetesDataset, compute_class_weight, get_dataloaders

DEVICE = get_device()

print("=" * 65)
print("  10_stratified_centralised_experiment.py")
print("  Federation Effect vs Data Composition Effect")
print("=" * 65)


# ── Load global scaler and eval data ─────────────────────────────────────────
_scaler  = joblib.load(GLOBAL_SCALER_PATH)
df_c     = pd.read_csv(CENTRALISED_PATH)
X_all    = _scaler.transform(df_c[FEATURE_COLS].values.astype(np.float32))
y_all    = df_c[TARGET_COL].values.astype(np.float32)

# Internal train/test split (consistent with 02_centralised_baseline.py)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=SEED
)

# Load Node B data for composition experiment
NODE_B_PATH = NODE_PATHS[1]
df_b   = pd.read_csv(NODE_B_PATH)
X_b    = _scaler.transform(df_b[FEATURE_COLS].values.astype(np.float32))
y_b    = df_b[TARGET_COL].values.astype(np.float32)

print(f"\n  Centralised n_train={len(X_tr_c):,}  n_test={len(X_te_c):,}")
print(f"  Node B (elderly-rural) n={len(X_b):,}")


def train_nn(X_train, y_train, X_test, y_test, label):
    """Train DiabetesNet for FL_NUM_ROUNDS epochs, return test AUC."""
    pos_w     = compute_class_weight(y_train)
    model     = DiabetesNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY
    )

    from torch.utils.data import DataLoader
    ds    = DiabetesDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=NN_BATCH_SIZE, shuffle=True,
                        pin_memory=(DEVICE.type == 'cuda'))
    scaler_amp = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    t0 = time.time()
    for epoch in range(FL_NUM_ROUNDS):
        train_one_epoch(model, loader, optimizer, criterion, DEVICE,
                        proximal_mu=0.0, global_params=None, scaler=scaler_amp)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                X_t   = torch.FloatTensor(X_test).to(DEVICE)
                probs = torch.sigmoid(model(X_t)).cpu().numpy()
            auc = roc_auc_score(y_test, probs)
            print(f"    [{label}] Epoch {epoch+1}/{FL_NUM_ROUNDS}  AUC={auc:.4f}  "
                  f"({time.time()-t0:.0f}s)")
            model.train()

    model.eval()
    with torch.no_grad():
        X_t   = torch.FloatTensor(X_test).to(DEVICE)
        probs = torch.sigmoid(model(X_t)).cpu().numpy()
    return float(roc_auc_score(y_test, probs)), model


# ── Condition A: Standard centralised ─────────────────────────────────────────
print("\n  [A] Standard centralised DiabetesNet...")
auc_a, model_a = train_nn(X_tr_c, y_tr_c, X_te_c, y_te_c, 'Centralised')
print(f"      Final AUC (internal): {auc_a:.4f}")


# ── Condition B: Node-B-weighted centralised ──────────────────────────────────
print("\n  [B] Node-B-weighted centralised (80% elderly-rural)...")
# Sample 80% from Node B, 20% from standard centralised training set
n_total = len(X_tr_c)
n_b     = int(0.80 * n_total)
n_c     = n_total - n_b

rng   = np.random.default_rng(SEED)
idx_b = rng.choice(len(X_b), size=min(n_b, len(X_b)), replace=len(X_b) < n_b)
idx_c = rng.choice(len(X_tr_c), size=n_c, replace=False)

X_tr_b = np.concatenate([X_b[idx_b], X_tr_c[idx_c]])
y_tr_b = np.concatenate([y_b[idx_b], y_tr_c[idx_c]])

auc_b, model_b = train_nn(X_tr_b, y_tr_b, X_te_c, y_te_c, 'NodeB-Weighted')
print(f"      Final AUC (internal): {auc_b:.4f}")


# ── Condition C: FedAvg (pre-trained) ─────────────────────────────────────────
print("\n  [C] FedAvg (loading pre-trained weights)...")
fedavg_path = os.path.join(MODELS_DIR, 'fedavg_weights.pt')
auc_c       = None
if os.path.exists(fedavg_path):
    model_c = DiabetesNet().to(DEVICE)
    model_c.load_state_dict(torch.load(fedavg_path, map_location=DEVICE))
    model_c.eval()
    with torch.no_grad():
        X_t   = torch.FloatTensor(X_te_c).to(DEVICE)
        probs = torch.sigmoid(model_c(X_t)).cpu().numpy()
    auc_c = float(roc_auc_score(y_te_c, probs))
    print(f"      FedAvg AUC (internal): {auc_c:.4f}")
else:
    print(f"      fedavg_weights.pt not found — run 03_federated_simulation.py first")


# ── External comparison (if BRFSS preds available) ───────────────────────────
ext_results = {}
y_brfss     = None
p_fedavg_x  = None
ext_path    = os.path.join(RESULTS_DIR, 'y_true_brfss.npy')
fedavg_x    = os.path.join(RESULTS_DIR, 'pred_fedavg_external.npy')

if os.path.exists(ext_path) and os.path.exists(fedavg_x):
    y_brfss    = np.load(ext_path)
    p_fedavg_x = np.load(fedavg_x)
    ext_results['FedAvg_external_AUC'] = float(roc_auc_score(y_brfss, p_fedavg_x))
    print(f"\n  FedAvg external AUC (BRFSS): {ext_results['FedAvg_external_AUC']:.4f}")


# ── Mechanism interpretation ──────────────────────────────────────────────────
print("\n  === Mechanism Analysis ===")
if auc_c is not None:
    diff_ab = abs(auc_a - auc_b)
    diff_ac = abs(auc_a - auc_c)
    diff_bc = abs(auc_b - auc_c)
    print(f"  |A - B| (composition gap): {diff_ab:.4f}")
    print(f"  |A - C| (federation gap) : {diff_ac:.4f}")
    print(f"  |B - C| (B vs FedAvg)    : {diff_bc:.4f}")
    if diff_bc < 0.005:
        print("  FINDING: B ≈ FedAvg → performance difference driven by DATA COMPOSITION")
        finding = 'composition_dominant'
    elif diff_bc < diff_ac:
        print("  FINDING: federation adds modest overhead beyond composition effect")
        finding = 'composition_dominant_with_federation_overhead'
    else:
        print("  FINDING: federation overhead exceeds composition effect")
        finding = 'federation_dominant'
else:
    finding = 'fedavg_weights_not_found'


# ── Save results ──────────────────────────────────────────────────────────────
results = {
    'conditions': {
        'A_standard_centralised': {
            'description': 'Standard centralised DiabetesNet on full NHANES',
            'n_train': len(X_tr_c),
            'auc_internal': auc_a,
        },
        'B_node_b_weighted': {
            'description': '80% Node B (elderly-rural) + 20% centralised training data',
            'n_train': len(X_tr_b),
            'node_b_fraction': 0.80,
            'auc_internal': auc_b,
        },
        'C_fedavg_pretrained': {
            'description': 'FedAvg weights from 03_federated_simulation.py',
            'auc_internal': auc_c,
        },
    },
    'external': ext_results,
    'mechanism_finding': finding,
}
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, 'stratified_centralised_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved -> results/stratified_centralised_results.json")

print("\n" + "=" * 65)
print("  CONDITION COMPARISON TABLE")
print("=" * 65)
print(f"  {'Condition':<40}  {'AUC (internal)':>15}")
print(f"  {'-'*57}")
print(f"  {'A — Standard centralised':<40}  {auc_a:>15.4f}")
print(f"  {'B — Node-B-weighted (80% elderly)':<40}  {auc_b:>15.4f}")
if auc_c is not None:
    print(f"  {'C — FedAvg (pre-trained)':<40}  {auc_c:>15.4f}")
print("=" * 65)
