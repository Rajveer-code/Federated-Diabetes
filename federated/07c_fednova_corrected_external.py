"""
07c_fednova_corrected_external.py
==================================
Targeted gap-fix: evaluates the corrected FedNova model (fednova_corrected_weights.pt,
tau_B=3 per Wang NeurIPS 2020 Theorem 2) on the BRFSS external dataset.

Saves:
  results/pred_fednova_corrected_external.npy
  (also patches external_validation.json with corrected FedNova AUC + fairness)

Run ONCE after 07_external_validation.py has already been run.
Expected runtime: ~90 seconds on CPU.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR, MODELS_DIR, FEATURE_COLS, TARGET_COL,
    GLOBAL_SCALER_PATH, AGE_GROUPS, BMI_GROUPS, SEX_GROUPS, SEED,
)
from nn_model import DiabetesNet, get_device

BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
DEVICE = get_device()

# ── Column remapping (BRFSS -> NHANES naming) ─────────────────────────────────
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

print("=" * 65)
print("  07c — Corrected FedNova External Validation (BRFSS)")
print("=" * 65)

# ── 1. Load and preprocess BRFSS ──────────────────────────────────────────────
print(f"\n[1/4] Loading BRFSS: {BRFSS_PATH}")
df = pd.read_csv(BRFSS_PATH)
df = df.rename(columns=BRFSS_COL_MAP)

# Gender remap: BRFSS 0/1 -> NHANES 1/2
if df['RIAGENDR'].max() <= 1.0:
    df['RIAGENDR'] = df['RIAGENDR'].map({1.0: 1.0, 0.0: 2.0})

df = df.dropna(subset=['DIABETES'])
df['DIABETES'] = df['DIABETES'].astype(int)

for col in ['BMXBMI', 'RIDAGEYR']:
    df[col] = df[col].fillna(df[col].median())
for col in ['RIAGENDR', 'RIDRETH3', 'SMOKING', 'PHYS_ACTIVITY', 'HEART_ATTACK', 'STROKE']:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.reset_index(drop=True)
X = df[FEATURE_COLS].values.astype(np.float32)
y = df['DIABETES'].values.astype(np.float32)
print(f"      {len(df):,} rows | prevalence: {y.mean():.1%}")

# ── 2. Scale ──────────────────────────────────────────────────────────────────
if os.path.exists(GLOBAL_SCALER_PATH):
    scaler = joblib.load(GLOBAL_SCALER_PATH)
    print(f"      Loaded global scaler: {GLOBAL_SCALER_PATH}")
else:
    _fb = os.path.join(MODELS_DIR, 'scaler.pkl')
    scaler = joblib.load(_fb)
    print(f"      Loaded fallback scaler: {_fb}")
X_sc = scaler.transform(X).astype(np.float32)

# ── 3. Evaluate corrected FedNova ─────────────────────────────────────────────
weights_path = os.path.join(MODELS_DIR, 'fednova_corrected_weights.pt')
print(f"\n[2/4] Evaluating corrected FedNova: {weights_path}")

model = DiabetesNet().to(DEVICE)
model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()

chunk = 50000
probs_list = []
with torch.no_grad():
    for i in range(0, len(X_sc), chunk):
        X_chunk = torch.FloatTensor(X_sc[i:i+chunk]).to(DEVICE)
        p = torch.sigmoid(model(X_chunk)).cpu().numpy()
        probs_list.append(p)
        if (i // chunk) % 5 == 0:
            print(f"      processed {min(i+chunk, len(X_sc)):,}/{len(X_sc):,} rows...")

y_prob = np.concatenate(probs_list)
auc = float(roc_auc_score(y, y_prob))
print(f"      AUC = {auc:.4f}")

# Compute full metrics
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_prob)
j = tpr - fpr
thresh = float(thresholds[np.argmax(j)])
y_pred = (y_prob >= thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
metrics = {
    'auc'        : auc,
    'brier'      : float(brier_score_loss(y, y_prob)),
    'f1'         : float(f1_score(y, y_pred, zero_division=0)),
    'sensitivity': float(tp/(tp+fn)) if (tp+fn)>0 else 0,
    'specificity': float(tn/(tn+fp)) if (tn+fp)>0 else 0,
    'ppv'        : float(precision_score(y, y_pred, zero_division=0)),
    'npv'        : float(tn/(tn+fn)) if (tn+fn)>0 else 0,
    'n'          : int(len(y)),
    'n_pos'      : int(y.sum()),
    'prevalence' : float(y.mean()),
}

# Fairness
fairness = {}
for label, (lo, hi) in AGE_GROUPS.items():
    mask = (df['RIDAGEYR'] >= lo) & (df['RIDAGEYR'] <= hi)
    if mask.sum() >= 100 and y[mask].sum() >= 20:
        fairness[f'age_{label}'] = float(roc_auc_score(y[mask], y_prob[mask]))
for label, (lo, hi) in BMI_GROUPS.items():
    mask = (df['BMXBMI'] >= lo) & (df['BMXBMI'] <= hi)
    if mask.sum() >= 100 and y[mask].sum() >= 20:
        fairness[f'bmi_{label}'] = float(roc_auc_score(y[mask], y_prob[mask]))
for label, val in SEX_GROUPS.items():
    mask = df['RIAGENDR'] == val
    if mask.sum() >= 100 and y[mask].sum() >= 20:
        fairness[f'sex_{label}'] = float(roc_auc_score(y[mask], y_prob[mask]))
if 'age_18-39' in fairness and 'age_60+' in fairness:
    fairness['elderly_gap'] = round(fairness['age_18-39'] - fairness['age_60+'], 5)

print(f"      Elderly gap = {fairness.get('elderly_gap', 'N/A')}")

# ── 4. Save ───────────────────────────────────────────────────────────────────
print("\n[3/4] Saving predictions...")

# Save prediction array
out_npy = os.path.join(RESULTS_DIR, 'pred_fednova_corrected_external.npy')
np.save(out_npy, y_prob)
print(f"      Saved: {out_npy}")

# Also overwrite pred_fednova_external.npy — corrected IS the definitive FedNova
out_npy2 = os.path.join(RESULTS_DIR, 'pred_fednova_external.npy')
np.save(out_npy2, y_prob)
print(f"      Updated: {out_npy2}  (corrected FedNova is the definitive version)")

# Patch external_validation.json
print("\n[4/4] Patching external_validation.json...")
ext_json_path = os.path.join(RESULTS_DIR, 'external_validation.json')
with open(ext_json_path, 'r') as f:
    ext_results = json.load(f)

# Add or update FedNova entry with corrected results
ext_results['federated']['FedNova'] = {'metrics': metrics, 'fairness': fairness}
ext_results['note_fednova'] = (
    'FedNova uses heterogeneous tau per Wang et al. NeurIPS 2020 Theorem 2: '
    'tau_A=5 (Node A, low shift), tau_B=3 (Node B, high shift), tau_C=4 (Node C, medium). '
    'Weights: fednova_corrected_weights.pt'
)

with open(ext_json_path, 'w') as f:
    json.dump(ext_results, f, indent=2)
print(f"      Updated: {ext_json_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  GAP 1 FIXED — Corrected FedNova external AUC")
print("=" * 65)
print(f"\n  FedNova (corrected) external AUC : {auc:.4f}")
print(f"  Elderly fairness gap             : {fairness.get('elderly_gap','N/A')}")
print(f"\n  Saved pred_fednova_corrected_external.npy")
print(f"  Updated external_validation.json (FedNova entry now = corrected)")
print(f"\n  Next: run 07_statistical_analysis.py to get DeLong CIs for all models")
print("=" * 65)
