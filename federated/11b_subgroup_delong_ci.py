"""
SCRIPT 11b — SUBGROUP DeLong CIs (BRFSS External)
===================================================
Computes DeLong 95% CIs for subgroup AUC (age 18-39 and age 60+) on
the BRFSS external set. Uses same data loading QC as 07_external_validation.py.

Outputs:
  results/subgroup_ci_results.json
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR, FEATURE_COLS, TARGET_COL,
    GLOBAL_SCALER_PATH, AGE_GROUPS, CI_ALPHA,
)

BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
Z_ALPHA = 1.959964  # z for 95% CI

print("=" * 65)
print("  11b_subgroup_delong_ci.py")
print("  DeLong CIs — Subgroup AUC (BRFSS External)")
print("=" * 65)

# ── DeLong variance ──────────────────────────────────────────────────────────
def delong_auc_ci(y_true, y_score):
    """Efficient DeLong via sort-based placement values (O(n log n), O(n) memory)."""
    y_true  = np.asarray(y_true,  dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos < 2 or n_neg < 2:
        return None
    auc = float(roc_auc_score(y_true, y_score))

    X = np.sort(y_score[pos_idx])   # sorted positive scores
    Y = np.sort(y_score[neg_idx])   # sorted negative scores

    # V10[i] = fraction of negatives with score < X[i] + 0.5 * tied
    lo_neg = np.searchsorted(Y, X, side='left')   # count Y < X[i]
    hi_neg = np.searchsorted(Y, X, side='right')  # count Y <= X[i]
    V10 = (lo_neg + 0.5 * (hi_neg - lo_neg)) / n_neg

    # V01[j] = fraction of positives with score > Y[j] + 0.5 * tied
    lo_pos = np.searchsorted(X, Y, side='left')
    hi_pos = np.searchsorted(X, Y, side='right')
    V01 = ((n_pos - hi_pos) + 0.5 * (hi_pos - lo_pos)) / n_pos

    var = (np.var(V10, ddof=1) / n_pos) + (np.var(V01, ddof=1) / n_neg)
    se  = float(np.sqrt(var))
    lo  = float(auc - Z_ALPHA * se)
    hi  = float(auc + Z_ALPHA * se)
    return {
        'auc'  : round(auc, 4),
        'lower': round(max(lo, 0.0), 4),
        'upper': round(min(hi, 1.0), 4),
        'se'   : round(se, 5),
        'n'    : int(len(y_true)),
        'n_pos': int(n_pos),
    }


# ── Load BRFSS ───────────────────────────────────────────────────────────────
print(f"\n[1/4] Loading BRFSS: {BRFSS_PATH}")
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

y_true = df['DIABETES'].values.astype(np.float32)
n_total = len(y_true)
print(f"      n={n_total:,}  prevalence={y_true.mean()*100:.2f}%")

# ── Age masks ────────────────────────────────────────────────────────────────
mask_young   = (df['RIDAGEYR'] >= 18) & (df['RIDAGEYR'] <= 39)
mask_elderly = (df['RIDAGEYR'] >= 60)
print(f"      Young 18-39: n={mask_young.sum():,}")
print(f"      Elderly 60+: n={mask_elderly.sum():,}")

# Save masks for 11_subgroup_confidence_intervals.py if needed later
np.save(os.path.join(RESULTS_DIR, 'brfss_age_young.npy'),   mask_young.values)
np.save(os.path.join(RESULTS_DIR, 'brfss_age_elderly.npy'), mask_elderly.values)
print("      Age mask .npy files saved.")


# ── Load predictions ─────────────────────────────────────────────────────────
print(f"\n[2/4] Loading predictions...")
def load_pred(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        return np.load(path)
    print(f"      WARNING: {fname} not found")
    return None

preds = {
    'XGBoost'              : load_pred('pred_xgb_external.npy'),
    'Centralised_NN'       : load_pred('pred_central_nn_external.npy'),
    'FedAvg'               : load_pred('pred_fedavg_external.npy'),
    'FedProx'              : load_pred('pred_fedprox_external.npy'),
    'FedNova'              : load_pred('pred_fednova_external.npy'),
    'SCAFFOLD'             : load_pred('pred_scaffold_external.npy'),
}

# Check dimensions match
for name, p in preds.items():
    if p is not None:
        if len(p) != n_total:
            print(f"      WARNING: {name} pred len={len(p)} != n_brfss={n_total}")
        else:
            print(f"      {name:<10}: n={len(p):,}  OK")


# ── Compute DeLong CIs ───────────────────────────────────────────────────────
print(f"\n[3/4] Computing DeLong CIs (subgroups)...")
results = {}

for model_name, pred in preds.items():
    if pred is None:
        continue
    if len(pred) != n_total:
        print(f"  {model_name}: dimension mismatch — skipped")
        continue

    r_overall = delong_auc_ci(y_true, pred)
    r_young   = delong_auc_ci(y_true[mask_young], pred[mask_young])
    r_elderly = delong_auc_ci(y_true[mask_elderly], pred[mask_elderly])

    gap_point = None
    gap_lo    = None
    gap_hi    = None
    if r_young and r_elderly:
        gap_point = round(r_young['auc'] - r_elderly['auc'], 4)
        # Gap CI via error propagation (independent subgroups)
        se_gap = float(np.sqrt(r_young['se']**2 + r_elderly['se']**2))
        gap_lo = round(gap_point - Z_ALPHA * se_gap, 4)
        gap_hi = round(gap_point + Z_ALPHA * se_gap, 4)

    results[model_name] = {
        'overall'  : r_overall,
        'young_18_39'  : r_young,
        'elderly_60plus': r_elderly,
        'fairness_gap_ci': {
            'gap'  : gap_point,
            'lower': gap_lo,
            'upper': gap_hi,
        } if gap_point is not None else None,
    }

    print(f"\n  {model_name}:")
    if r_overall:
        print(f"    Overall : {r_overall['auc']:.4f} [{r_overall['lower']:.4f}–{r_overall['upper']:.4f}]  n={r_overall['n']:,}")
    if r_young:
        print(f"    Young   : {r_young['auc']:.4f} [{r_young['lower']:.4f}–{r_young['upper']:.4f}]  n={r_young['n']:,}")
    if r_elderly:
        print(f"    Elderly : {r_elderly['auc']:.4f} [{r_elderly['lower']:.4f}–{r_elderly['upper']:.4f}]  n={r_elderly['n']:,}")
    if gap_point is not None:
        print(f"    Gap     : {gap_point:.4f} [{gap_lo:.4f}–{gap_hi:.4f}]")


# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'method': 'DeLong structural components (Hanley-McNeil 1988), 95% CI',
    'n_brfss': int(n_total),
    'models' : results,
}
out_path = os.path.join(RESULTS_DIR, 'subgroup_ci_results.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n[4/4] Saved -> {out_path}")


# ── Manuscript table ──────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  TABLE S6 — SUBGROUP AUC WITH 95% CI (DeLong)")
print("=" * 75)
print(f"  {'Model':<10}  {'AUC 18-39 [95% CI]':<26}  {'AUC ≥60 [95% CI]':<26}  {'Gap [95% CI]'}")
print("  " + "-" * 72)
for nm, r in results.items():
    y  = r['young_18_39']
    e  = r['elderly_60plus']
    g  = r['fairness_gap_ci']
    ys = f"{y['auc']:.3f} [{y['lower']:.3f}–{y['upper']:.3f}]" if y else "N/A"
    es = f"{e['auc']:.3f} [{e['lower']:.3f}–{e['upper']:.3f}]" if e else "N/A"
    gs = f"{g['gap']:.3f} [{g['lower']:.3f}–{g['upper']:.3f}]" if g else "N/A"
    print(f"  {nm:<10}  {ys:<26}  {es:<26}  {gs}")
print("=" * 75)
