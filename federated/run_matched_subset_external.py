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
from data_utils import DiabetesDataset, compute_class_weight

DEVICE = get_device()
BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
BRFSS_COL_MAP = {
    'Age'                 : 'RIDAGEYR',
    'Gender'              : 'RIAGENDR',
    'Race_Ethnicity'      : 'RIDRETH3',
    'BMI'                 : 'BMXBMI',
    'Smoking_Status'      : 'SMOKING',
    'Physical_Activity'   : 'PHYS_ACTIVITY',
    'History_Heart_Attack': 'HEART_ATTACK',
    'History_Stroke'      : 'STROKE',
    'Diabetes_Outcome'    : 'DIABETES',
}

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
print(f"Loading BRFSS from {BRFSS_PATH} ...")
df_b = pd.read_csv(BRFSS_PATH)
print(f"BRFSS raw n={len(df_b):,}")

# Rename columns to NHANES names
df_b = df_b.rename(columns=BRFSS_COL_MAP)
# Keep only needed cols
needed = FEATURE_COLS + [TARGET_COL]
available = [c for c in needed if c in df_b.columns]
df_b = df_b[available].dropna()
print(f"BRFSS after QC: n={len(df_b):,}")

X_brfss = scaler.transform(df_b[FEATURE_COLS].values.astype(np.float32))
y_brfss = df_b[TARGET_COL].values.astype(np.float32)

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
