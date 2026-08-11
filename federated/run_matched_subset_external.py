"""
Run external BRFSS validation for the matched-subset centralised DiabetesNet.
Trains on combined federated nodes (n=11,906) → evaluate on BRFSS.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
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
from nn_model   import DiabetesNet, train_one_epoch, get_device
from data_utils import DiabetesDataset, compute_class_weight, load_brfss_clean

DEVICE = get_device()

# Reproducibility: this control feeds the paper's headline matched comparison.
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

scaler = joblib.load(GLOBAL_SCALER_PATH)

# ── Train on federated subset ─────────────────────────────────────────────────
frames = [pd.read_csv(p) for p in NODE_PATHS]
df_fed = pd.concat(frames, ignore_index=True)
print(f"Federated subset: n={len(df_fed):,}")

X = scaler.transform(df_fed[FEATURE_COLS].values.astype(np.float32))
y = df_fed[TARGET_COL].values.astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Train={len(X_tr):,}  Test={len(X_te):,}")

pos_w = compute_class_weight(y_tr)
model = DiabetesNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
from torch.utils.data import DataLoader
loader = DataLoader(DiabetesDataset(X_tr, y_tr), batch_size=NN_BATCH_SIZE, shuffle=True)
amp = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

for epoch in range(FL_NUM_ROUNDS):
    train_one_epoch(model, loader, optimizer, criterion, DEVICE,
                    proximal_mu=0.0, global_params=None, scaler=amp)

model.eval()
with torch.no_grad():
    probs_int = torch.sigmoid(model(torch.FloatTensor(X_te).to(DEVICE))).cpu().numpy()
auc_int = float(roc_auc_score(y_te, probs_int))
print(f"Internal AUC: {auc_int:.4f}")

# ── External BRFSS ────────────────────────────────────────────────────────────
# Canonical loader — identical population and preprocessing to every other model
# (07_external_validation.py). Previously this script rolled its own load and
# diverged: complete-case deletion (720,161 rows vs 1,282,897) and no BRFSS->NHANES
# sex remap. Both made this control non-comparable to the models it anchors.
df_b, X_brfss_raw, y_brfss = load_brfss_clean()
X_brfss = scaler.transform(X_brfss_raw)

BS = 65536
all_probs = []
with torch.no_grad():
    for i in range(0, len(X_brfss), BS):
        batch = torch.FloatTensor(X_brfss[i:i+BS]).to(DEVICE)
        all_probs.append(torch.sigmoid(model(batch)).cpu().numpy())
probs_ext = np.concatenate(all_probs)
auc_ext = float(roc_auc_score(y_brfss, probs_ext))
gap = auc_int - auc_ext
print(f"External AUC: {auc_ext:.4f}")
print(f"Generalisation gap: {gap:.4f}")

result = {
    'description': 'Centralised DiabetesNet on federated subset (n=11906)',
    'n_train': int(len(X_tr)),
    'n_test_internal': int(len(X_te)),
    'n_brfss': int(len(y_brfss)),
    'auc_internal': round(auc_int, 4),
    'auc_external': round(auc_ext, 4),
    'generalisation_gap': round(gap, 4),
}
print(json.dumps(result, indent=2))
with open(os.path.join(RESULTS_DIR, 'matched_subset_sensitivity.json'), 'w') as f:
    json.dump(result, f, indent=2)
print("Saved -> results/matched_subset_sensitivity.json")
