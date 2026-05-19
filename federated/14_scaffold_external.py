"""
14_scaffold_external.py
========================
Evaluates SCAFFOLD model on BRFSS external dataset.
Fills the '—' gap in Table 1 for SCAFFOLD external AUC.

Modeled on 07c_fednova_corrected_external.py.

Saves:
  results/scaffold_external_results.json
  results/pred_scaffold_external.npy
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, confusion_matrix, precision_score
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR, MODELS_DIR, FEATURE_COLS, TARGET_COL,
    GLOBAL_SCALER_PATH, AGE_GROUPS, BMI_GROUPS, SEX_GROUPS, SEED, CI_ALPHA,
)
from nn_model import DiabetesNet, get_device

BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
DEVICE = get_device()

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


def delong_ci(y_true, y_score, alpha=CI_ALPHA):
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)

    neg_sorted = np.sort(neg_scores)
    n_below = np.searchsorted(neg_sorted, pos_scores, side='left')
    n_equal = np.searchsorted(neg_sorted, pos_scores, side='right') - n_below
    V10 = (n_below.astype(np.float64) + 0.5 * n_equal) / n_neg

    pos_sorted = np.sort(pos_scores)
    n_above    = n_pos - np.searchsorted(pos_sorted, neg_scores, side='right')
    n_equal_01 = (np.searchsorted(pos_sorted, neg_scores, side='right') -
                  np.searchsorted(pos_sorted, neg_scores, side='left'))
    V01 = (n_above.astype(np.float64) + 0.5 * n_equal_01) / n_pos

    auc = float(V10.mean())
    s10 = np.var(V10, ddof=1) / n_pos
    s01 = np.var(V01, ddof=1) / n_neg
    se  = float(np.sqrt(s10 + s01))
    z   = stats.norm.ppf(1.0 - alpha / 2.0)
    return auc, float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se)), se


print("=" * 65)
print("  14 — SCAFFOLD External Validation (BRFSS)")
print("=" * 65)

# 1. Load BRFSS
print(f"\n[1/4] Loading BRFSS: {BRFSS_PATH}")
df = pd.read_csv(BRFSS_PATH)
df = df.rename(columns=BRFSS_COL_MAP)
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

# 2. Scale
if os.path.exists(GLOBAL_SCALER_PATH):
    scaler = joblib.load(GLOBAL_SCALER_PATH)
else:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
X_sc = scaler.transform(X).astype(np.float32)

# 3. Load SCAFFOLD model and run inference
weights_path = os.path.join(MODELS_DIR, 'scaffold_weights.pt')
print(f"\n[2/4] Loading SCAFFOLD weights: {weights_path}")
model = DiabetesNet().to(DEVICE)
model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
model.eval()

chunk = 50000
probs_list = []
with torch.no_grad():
    for i in range(0, len(X_sc), chunk):
        X_chunk = torch.FloatTensor(X_sc[i:i+chunk]).to(DEVICE)
        p = torch.sigmoid(model(X_chunk)).cpu().numpy()
        probs_list.append(p)
        if (i // chunk) % 5 == 0:
            print(f"      processed {min(i+chunk, len(X_sc)):,}/{len(X_sc):,}")

y_prob = np.concatenate(probs_list)

# 4. Compute metrics
print("\n[3/4] Computing metrics...")
auc_val, ci_lo, ci_hi, se = delong_ci(y, y_prob)
brier = float(brier_score_loss(y, y_prob))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_prob)
thresh = float(thresholds[np.argmax(tpr - fpr)])
y_pred = (y_prob >= thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

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

print(f"      AUC = {auc_val:.4f} [{ci_lo:.4f}–{ci_hi:.4f}]")
print(f"      Brier = {brier:.4f}")
print(f"      Elderly gap = {fairness.get('elderly_gap', 'N/A')}")

# 5. Save
print("\n[4/4] Saving...")
np.save(os.path.join(RESULTS_DIR, 'pred_scaffold_external.npy'), y_prob)

out = {
    'model': 'SCAFFOLD (Option II)',
    'dataset': 'BRFSS 2020-2022',
    'n': int(len(y)),
    'external_auc': auc_val,
    'external_ci_lower': ci_lo,
    'external_ci_upper': ci_hi,
    'external_se': se,
    'external_brier': brier,
    'sensitivity': float(tp/(tp+fn)) if (tp+fn) > 0 else 0,
    'specificity': float(tn/(tn+fp)) if (tn+fp) > 0 else 0,
    'threshold': thresh,
    'fairness': fairness,
    'elderly_gap': fairness.get('elderly_gap'),
}
with open(os.path.join(RESULTS_DIR, 'scaffold_external_results.json'), 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"  SCAFFOLD External AUC: {auc_val:.4f} [{ci_lo:.4f}–{ci_hi:.4f}]")
print(f"  Elderly gap: {fairness.get('elderly_gap', 'N/A')}")
print(f"  Brier: {brier:.4f}")
print(f"  Saved: scaffold_external_results.json, pred_scaffold_external.npy")
print(f"{'='*65}")
