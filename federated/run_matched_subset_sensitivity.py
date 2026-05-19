"""
MATCHED SUBSET SENSITIVITY: train centralised DiabetesNet on federated subset
(n=11,906, all three nodes combined) and evaluate externally on BRFSS.
Directly tests: does FedAvg generalisation advantage persist when centralised
model sees the same n as FedAvg?
"""
import os, sys, json, warnings
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
    NODE_PATHS, RESULTS_DIR, FEATURE_COLS, TARGET_COL,
    GLOBAL_SCALER_PATH, NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    FL_NUM_ROUNDS, SEED,
)
from nn_model import DiabetesNet, train_one_epoch, get_device
from data_utils import DiabetesDataset, compute_class_weight

DEVICE = get_device()
print("MATCHED SUBSET SENSITIVITY ANALYSIS")
print(f"Device: {DEVICE}")

scaler = joblib.load(GLOBAL_SCALER_PATH)

# Load all three nodes → combine = federated training subset (n=11,906)
frames = []
for p in NODE_PATHS:
    df = pd.read_csv(p)
    frames.append(df)
df_fed = pd.concat(frames, ignore_index=True)
print(f"Combined federated subset: n={len(df_fed):,}")

X = scaler.transform(df_fed[FEATURE_COLS].values.astype(np.float32))
y = df_fed[TARGET_COL].values.astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")

# Train centralised DiabetesNet on federated subset
pos_w = compute_class_weight(y_tr)
model = DiabetesNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)

from torch.utils.data import DataLoader
ds = DiabetesDataset(X_tr, y_tr)
loader = DataLoader(ds, batch_size=NN_BATCH_SIZE, shuffle=True)
scaler_amp = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

for epoch in range(FL_NUM_ROUNDS):
    train_one_epoch(model, loader, optimizer, criterion, DEVICE,
                    proximal_mu=0.0, global_params=None, scaler=scaler_amp)

model.eval()
with torch.no_grad():
    probs_int = torch.sigmoid(model(torch.FloatTensor(X_te).to(DEVICE))).cpu().numpy()
auc_internal = float(roc_auc_score(y_te, probs_int))
print(f"Internal AUC (fed subset test): {auc_internal:.4f}")

# External BRFSS evaluation
y_brfss = np.load(os.path.join(RESULTS_DIR, 'y_true_brfss.npy'))
print(f"BRFSS n={len(y_brfss):,}")

# We need BRFSS features — use pred_fedavg_external.npy as shape check
# Must re-run external validation; check if BRFSS data available
brfss_path = os.environ.get('BRFSS_PATH', '')
if os.path.exists(brfss_path):
    df_brfss = pd.read_csv(brfss_path)
    # Map BRFSS features — same as 07_external_validation.py
    from data_utils import harmonise_brfss
    X_brfss = harmonise_brfss(df_brfss, scaler, FEATURE_COLS)
    y_brfss_labels = (df_brfss['DIABETE4'].isin([1, 2])).astype(float).values
    with torch.no_grad():
        probs_ext = torch.sigmoid(model(torch.FloatTensor(X_brfss).to(DEVICE))).cpu().numpy()
    auc_external = float(roc_auc_score(y_brfss_labels, probs_ext))
    print(f"External BRFSS AUC: {auc_external:.4f}")
else:
    # BRFSS path not set — use stored y_true and generate predictions via
    # the same feature preprocessing that produced the stored predictions
    print("BRFSS_PATH not set — computing external AUC from stored features")
    # Load BRFSS X features from pred arrays (can't recover X from preds alone)
    # Fall back: use the exp2 result as proxy (not ideal)
    print("  Cannot compute external AUC without BRFSS CSV")
    auc_external = None

gap = (auc_internal - auc_external) if auc_external else None
result = {
    'description': 'Centralised DiabetesNet trained on federated subset (n=11906)',
    'n_train': len(X_tr),
    'n_test': len(X_te),
    'auc_internal': auc_internal,
    'auc_external': auc_external,
    'generalisation_gap': gap,
    'note': 'If auc_external is null, BRFSS_PATH env var was not set'
}
print(f"\nSummary: internal={auc_internal:.4f}, external={auc_external}, gap={gap}")

with open(os.path.join(RESULTS_DIR, 'matched_subset_sensitivity.json'), 'w') as f:
    json.dump(result, f, indent=2)
print("Saved -> results/matched_subset_sensitivity.json")
