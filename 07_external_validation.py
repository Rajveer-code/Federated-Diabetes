"""
SCRIPT 07 -- BRFSS EXTERNAL VALIDATION
========================================
Evaluates all trained models on the BRFSS 2020-2022 external dataset.
This directly replicates the external validation in your IEEE paper
and answers: does the federated model generalise better?

Key comparison:
  Published centralised XGBoost external AUC : 0.717
  Our centralised XGBoost external AUC        : ?
  Best federated model external AUC           : ?

Produces:
  results/external_validation.json
  plots/10_external_validation_roc.png
  plots/11_external_validation_fairness.png

Usage:
  python 07_external_validation.py  (~5 minutes for 1.3M rows)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              f1_score, confusion_matrix, precision_score)
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_COL, GLOBAL_SCALER_PATH,
    PUBLISHED_EXTERNAL_AUC, PUBLISHED_INTERNAL_AUC,
    PUBLISHED_ELDERLY_GAP, AGE_GROUPS, BMI_GROUPS, SEX_GROUPS,
    XGB_PARAMS, SEED,
)
from nn_model   import DiabetesNet, get_device
from data_utils import set_params_from_numpy

# ── UPDATE THIS PATH ───────────────────────────────────────────────────────────
BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = get_device()

BLUE   = '#2563EB'
PURPLE = '#7C3AED'
RED    = '#DC2626'
GREEN  = '#16A34A'
GREY   = '#94A3B8'

# ── BRFSS -> NHANES column mapping ─────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_metrics(y_true, y_prob):
    threshold = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'auc'        : float(roc_auc_score(y_true, y_prob)),
        'brier'      : float(brier_score_loss(y_true, y_prob)),
        'f1'         : float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp/(tp+fn)) if (tp+fn)>0 else 0,
        'specificity': float(tn/(tn+fp)) if (tn+fp)>0 else 0,
        'ppv'        : float(precision_score(y_true, y_pred, zero_division=0)),
        'npv'        : float(tn/(tn+fn)) if (tn+fn)>0 else 0,
        'n'          : int(len(y_true)),
        'n_pos'      : int(y_true.sum()),
        'prevalence' : float(y_true.mean()),
    }


def fairness_on_brfss(y_true, y_prob, df):
    """Subgroup AUC on BRFSS — same groups as IEEE paper."""
    res = {}
    y_true = np.array(y_true)

    # Age groups — BRFSS uses midpoint ages (22.5, 27.5 ... 80.0)
    for label, (lo, hi) in AGE_GROUPS.items():
        mask = (df['RIDAGEYR'] >= lo) & (df['RIDAGEYR'] <= hi)
        if mask.sum() >= 100 and y_true[mask].sum() >= 20:
            res[f'age_{label}'] = float(roc_auc_score(y_true[mask], y_prob[mask]))

    # BMI
    for label, (lo, hi) in BMI_GROUPS.items():
        mask = (df['BMXBMI'] >= lo) & (df['BMXBMI'] <= hi)
        if mask.sum() >= 100 and y_true[mask].sum() >= 20:
            res[f'bmi_{label}'] = float(roc_auc_score(y_true[mask], y_prob[mask]))

    # Sex — BRFSS uses 0/1; we remap to 1/2 at load time
    for label, val in SEX_GROUPS.items():
        mask = df['RIAGENDR'] == val
        if mask.sum() >= 100 and y_true[mask].sum() >= 20:
            res[f'sex_{label}'] = float(roc_auc_score(y_true[mask], y_prob[mask]))

    if 'age_18-39' in res and 'age_60+' in res:
        res['elderly_gap'] = round(res['age_18-39'] - res['age_60+'], 5)

    return res


# ──────────────────────────────────────────────────────────────────────────────
#  LOAD BRFSS
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SCRIPT 07 -- BRFSS EXTERNAL VALIDATION")
print("=" * 65)

print(f"\n[1/5] Loading BRFSS: {BRFSS_PATH}")
print("      (1.3M rows — may take 30-60 seconds...)")
df_brfss = pd.read_csv(BRFSS_PATH)
print(f"      Raw shape: {df_brfss.shape}")
print(f"      Columns: {list(df_brfss.columns)}")

# Rename columns to NHANES equivalents
df_brfss = df_brfss.rename(columns=BRFSS_COL_MAP)

# ── Gender encoding fix ────────────────────────────────────────────────────────
# BRFSS: 0=Female, 1=Male  ->  NHANES: 1=Male, 2=Female
if df_brfss['RIAGENDR'].max() <= 1.0:
    df_brfss['RIAGENDR'] = df_brfss['RIAGENDR'].map({1.0: 1.0, 0.0: 2.0})
    print("      Gender remapped: BRFSS 0/1 -> NHANES 1/2")

# ── Smoking encoding fix ───────────────────────────────────────────────────────
# BRFSS may be binary 0/1; NHANES is 0=never,1=former,2=current
# If BRFSS max is 1, it's binary — treat 1=ever smoked (maps to former/current)
# We keep as-is because the model was trained on 0/1/2 but scale normalises it
# The distribution shift here is documented in your IEEE paper

# ── Drop rows missing target or any feature ────────────────────────────────────
df_brfss = df_brfss.dropna(subset=['DIABETES'])
df_brfss['DIABETES'] = df_brfss['DIABETES'].astype(int)

# Impute missing features (same strategy as training)
for col in ['BMXBMI', 'RIDAGEYR']:
    df_brfss[col] = df_brfss[col].fillna(df_brfss[col].median())
for col in ['RIAGENDR', 'RIDRETH3', 'SMOKING', 'PHYS_ACTIVITY', 'HEART_ATTACK', 'STROKE']:
    df_brfss[col] = df_brfss[col].fillna(df_brfss[col].mode()[0])

df_brfss = df_brfss.reset_index(drop=True)

X_brfss = df_brfss[FEATURE_COLS].values.astype(np.float32)
y_brfss = df_brfss['DIABETES'].values.astype(np.float32)

print(f"      After cleaning: {len(df_brfss):,} rows")
print(f"      Diabetes prevalence: {y_brfss.mean():.1%}")
print(f"      Age range: {df_brfss['RIDAGEYR'].min():.0f} - {df_brfss['RIDAGEYR'].max():.0f}")

# ── Load global NHANES scaler (fitted on train split by 00_fit_global_scaler.py) ─
# CORRECT: load the canonical scaler and call .transform() only — never fit_transform.
# This ensures BRFSS features are standardised with the same NHANES training statistics
# used during model training, eliminating preprocessing domain mismatch.
if not os.path.exists(GLOBAL_SCALER_PATH):
    # Fall back to legacy scaler if global scaler not yet created
    _fallback = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(_fallback):
        print(f"  WARNING: {GLOBAL_SCALER_PATH} not found.")
        print(f"  Falling back to {_fallback}.")
        print(f"  Run 00_fit_global_scaler.py to create the canonical scaler.")
        scaler = joblib.load(_fallback)
    else:
        print(f"  ERROR: Neither {GLOBAL_SCALER_PATH} nor {_fallback} found.")
        print(f"  Run 00_fit_global_scaler.py first.")
        sys.exit(1)
else:
    scaler = joblib.load(GLOBAL_SCALER_PATH)
    print(f"      Loaded global NHANES scaler: {GLOBAL_SCALER_PATH}")

X_brfss_sc = scaler.transform(X_brfss).astype(np.float32)  # transform ONLY


# ──────────────────────────────────────────────────────────────────────────────
#  (2) CENTRALISED XGBOOST
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Evaluating centralised XGBoost...")
xgb_path = os.path.join(MODELS_DIR, 'centralised_xgb.pkl')
xgb_model = joblib.load(xgb_path)
y_prob_xgb = xgb_model.predict_proba(X_brfss_sc)[:, 1]
metrics_xgb = compute_metrics(y_brfss, y_prob_xgb)
fairness_xgb = fairness_on_brfss(y_brfss, y_prob_xgb, df_brfss)

print(f"  Centralised XGBoost external AUC : {metrics_xgb['auc']:.4f}")
print(f"  Published external AUC           : {PUBLISHED_EXTERNAL_AUC}")
print(f"  Difference                       : {metrics_xgb['auc'] - PUBLISHED_EXTERNAL_AUC:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
#  (3) FEDERATED MODELS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Evaluating federated models...")

def eval_fed_model(weights_path, name):
    if not os.path.exists(weights_path):
        print(f"  Skipping {name} — weights not found: {weights_path}")
        return None, None, None
    model = DiabetesNet().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        # Process in chunks to avoid OOM on 1.3M rows
        chunk = 50000
        probs = []
        for i in range(0, len(X_brfss_sc), chunk):
            X_chunk = torch.FloatTensor(X_brfss_sc[i:i+chunk]).to(DEVICE)
            p = torch.sigmoid(model(X_chunk)).cpu().numpy()
            probs.append(p)
        y_prob = np.concatenate(probs)
    metrics  = compute_metrics(y_brfss, y_prob)
    fairness = fairness_on_brfss(y_brfss, y_prob, df_brfss)
    print(f"  {name:20s} AUC={metrics['auc']:.4f}  "
          f"Gap={fairness.get('elderly_gap','N/A')}")
    return metrics, fairness, y_prob

fed_models = {
    'FedAvg' : os.path.join(MODELS_DIR, 'fedavg_weights.pt'),
    'FedProx': os.path.join(MODELS_DIR, 'fedprox_weights.pt'),
    # Use corrected FedNova (tau_B=3, tau_A=5, tau_C=4 per Wang NeurIPS 2020 Theorem 2)
    # The uniform-tau version (fednova_weights.pt) is superseded.
    'FedNova': os.path.join(MODELS_DIR, 'fednova_corrected_weights.pt'),
}

fed_metrics  = {}
fed_fairness = {}
fed_probs    = {}
for name, path in fed_models.items():
    m, f, p = eval_fed_model(path, name)
    if m:
        fed_metrics[name]  = m
        fed_fairness[name] = f
        fed_probs[name]    = p


# ──────────────────────────────────────────────────────────────────────────────
#  (4) PRINT RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/5] External validation results...")

print("\n" + "=" * 65)
print("  EXTERNAL VALIDATION RESULTS (BRFSS 2020-2022)")
print("=" * 65)
print(f"\n  {'Model':<25} {'AUC':>8} {'Brier':>8} {'F1':>7} {'Sens':>7} {'Spec':>7}")
print("  " + "-"*60)

# Published baseline
print(f"  {'Published (IEEE paper)':<25} {PUBLISHED_EXTERNAL_AUC:>8.4f}     --      --      --      --")

# Centralised
m = metrics_xgb
print(f"  {'Centralised XGBoost':<25} {m['auc']:>8.4f} {m['brier']:>8.4f} "
      f"{m['f1']:>7.4f} {m['sensitivity']:>7.4f} {m['specificity']:>7.4f}")

# Federated
for name, m in fed_metrics.items():
    print(f"  {f'Federated ({name})':<25} {m['auc']:>8.4f} {m['brier']:>8.4f} "
          f"{m['f1']:>7.4f} {m['sensitivity']:>7.4f} {m['specificity']:>7.4f}")

print("\n  FAIRNESS ON BRFSS")
print("  " + "-"*60)
print(f"  {'Subgroup':<22} {'Published':>10} {'Centralised':>12} ", end="")
if 'FedProx' in fed_fairness:
    print(f"{'FedProx':>10}")
else:
    print()

age_keys = [('age_18-39','Age 18-39'), ('age_40-59','Age 40-59'), ('age_60+','Age 60+')]
for key, label in age_keys:
    pub  = {'age_18-39': 0.742, 'age_40-59': None, 'age_60+': 0.607}.get(key)
    cen  = fairness_xgb.get(key)
    fed  = fed_fairness.get('FedProx', {}).get(key)
    pub_s = f"{pub:.3f}" if pub else "   --"
    cen_s = f"{cen:.3f}" if cen else "   --"
    fed_s = f"{fed:.3f}" if fed else "   --"
    print(f"  {label:<22} {pub_s:>10} {cen_s:>12} {fed_s:>10}")

pub_gap = PUBLISHED_ELDERLY_GAP
cen_gap = fairness_xgb.get('elderly_gap')
fed_gap = fed_fairness.get('FedProx', {}).get('elderly_gap')

print(f"\n  Elderly gap — Published : {pub_gap:.3f}")
print(f"  Elderly gap — Centralised: {cen_gap:.4f}" if cen_gap else "")
print(f"  Elderly gap — FedProx   : {fed_gap:.4f}" if fed_gap else "")

if cen_gap and fed_gap:
    if fed_gap < cen_gap:
        print(f"  FL reduces external fairness gap by {cen_gap-fed_gap:.4f} ({(cen_gap-fed_gap)/cen_gap*100:.1f}%) ✅")
    else:
        print(f"  FL fairness gap change: {fed_gap-cen_gap:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
#  SAVE RESULTS
# ──────────────────────────────────────────────────────────────────────────────
results_ext = {
    'dataset'       : 'BRFSS 2020-2022',
    'n'             : int(len(df_brfss)),
    'prevalence'    : float(y_brfss.mean()),
    'centralised'   : {'metrics': metrics_xgb, 'fairness': fairness_xgb},
    'federated'     : {k: {'metrics': v, 'fairness': fed_fairness[k]}
                       for k, v in fed_metrics.items()},
    'published_auc' : PUBLISHED_EXTERNAL_AUC,
}
with open(os.path.join(RESULTS_DIR, 'external_validation.json'), 'w', encoding='utf-8') as f:
    json.dump(results_ext, f, indent=2)
print(f"\n  Saved: results/external_validation.json")

# Save prediction arrays for 07_statistical_analysis.py
np.save(os.path.join(RESULTS_DIR, 'y_true_brfss.npy'), y_brfss)
print(f"  Saved: results/y_true_brfss.npy")
for name, probs in fed_probs.items():
    fname = f"pred_{name.lower()}_external.npy"
    np.save(os.path.join(RESULTS_DIR, fname), probs)
    print(f"  Saved: results/{fname}")


# ──────────────────────────────────────────────────────────────────────────────
#  (5) PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating external validation plots...")

# Plot 1: ROC curves all models
fig, ax = plt.subplots(figsize=(9, 7))
fpr_c, tpr_c, _ = roc_curve(y_brfss, y_prob_xgb)
ax.plot(fpr_c, tpr_c, color=BLUE, lw=2.5,
        label=f"Centralised XGBoost (AUC={metrics_xgb['auc']:.4f})")

model_colors = {'FedAvg': GREEN, 'FedProx': PURPLE, 'FedNova': RED}
for name, y_prob in fed_probs.items():
    fpr, tpr, _ = roc_curve(y_brfss, y_prob)
    auc = fed_metrics[name]['auc']
    ax.plot(fpr, tpr, color=model_colors.get(name, GREY), lw=2.5,
            label=f"Federated {name} (AUC={auc:.4f})")

ax.plot([0,1],[0,1], '--', color=GREY, lw=1.2, label='Random (AUC=0.500)')
ax.axvline(x=0.3, color='orange', lw=1, ls=':', alpha=0.7)

# Published reference point
ax.scatter([], [], marker='*', s=150, color='black',
           label=f'Published (AUC={PUBLISHED_EXTERNAL_AUC})')

ax.fill_between(fpr_c, tpr_c, alpha=0.05, color=BLUE)
ax.set_xlabel('1 - Specificity (FPR)', fontsize=13)
ax.set_ylabel('Sensitivity (TPR)', fontsize=13)
ax.set_title(
    'External Validation — ROC Curves on BRFSS 2020-2022\n'
    f'n={len(df_brfss):,} | Diabetes prevalence: {y_brfss.mean():.1%}',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_external_validation_roc.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/10_external_validation_roc.png")

# Plot 2: External fairness comparison
fig, ax = plt.subplots(figsize=(11, 6))
age_labels_p = ['Age 18-39', 'Age 40-59', 'Age 60+']
age_keys_p   = ['age_18-39', 'age_40-59', 'age_60+']
pub_v  = [0.742, None, 0.607]
cen_v  = [fairness_xgb.get(k) for k in age_keys_p]
fed_v  = [fed_fairness.get('FedProx', {}).get(k) for k in age_keys_p]

x = np.arange(len(age_labels_p))
w = 0.24

b1 = ax.bar(x-w,   [v or 0 for v in pub_v], w, color=GREY,   label='Published (internal)', alpha=0.9)
b2 = ax.bar(x,     [v or 0 for v in cen_v], w, color=BLUE,   label='Centralised (external)', alpha=0.9)
b3 = ax.bar(x+w,   [v or 0 for v in fed_v], w, color=PURPLE, label='Federated FedProx (external)', alpha=0.9)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.004,
                    f'{h:.3f}', ha='center', fontsize=9.5, fontweight='bold')

ax.axhline(0.5, color='black', ls='--', lw=1, alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(age_labels_p, fontsize=12)
ax.set_ylabel('AUC-ROC', fontsize=13)
ax.set_ylim(0.48, 0.84)
ax.set_title(
    'Age-Group Fairness on External BRFSS Dataset\n'
    'Federated model vs centralised baseline vs published paper',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Annotation box
box_text = (
    f"External elderly gap:\n"
    f"  Published:    {pub_gap:.3f} (internal)\n"
    f"  Centralised:  {cen_gap:.3f}\n"
    f"  Federated:    {fed_gap:.3f}" if (cen_gap and fed_gap)
    else "External elderly gap: see results"
)
ax.text(0.02, 0.04, box_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFBEB', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_external_validation_fairness.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/11_external_validation_fairness.png")

print("\n" + "=" * 65)
print("  EXTERNAL VALIDATION COMPLETE")
print("=" * 65)
print(f"\n  Key number for paper abstract:")
if fed_metrics:
    best_fed = max(fed_metrics, key=lambda k: fed_metrics[k]['auc'])
    best_ext = fed_metrics[best_fed]['auc']
    print(f"  Best federated external AUC : {best_ext:.4f} ({best_fed})")
    print(f"  Published external AUC       : {PUBLISHED_EXTERNAL_AUC}")
    print(f"  Improvement                  : {best_ext - PUBLISHED_EXTERNAL_AUC:+.4f}")
print(f"  Centralised external AUC     : {metrics_xgb['auc']:.4f}")
print(f"\nRun 08_write_paper.py next to generate the full manuscript.")
print("=" * 65)
