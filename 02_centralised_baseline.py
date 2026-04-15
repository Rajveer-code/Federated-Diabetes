"""
SCRIPT 02 — CENTRALISED BASELINE
===================================
Replicates your IEEE paper XGBoost model exactly.
This is the gold standard every federated model is compared against.

Produces:
  results/centralised_metrics.json
  results/centralised_fairness.json
  plots/01_centralised_roc.png
  plots/02_centralised_fairness_age.png
  models/centralised_xgb.pkl
  models/scaler.pkl

Usage:
  cd D:\Projects\diabetes_prediction_project\federated
  python 02_centralised_baseline.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
warnings.filterwarnings('ignore')

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    CENTRALISED_PATH, RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_COL,
    PUBLISHED_INTERNAL_AUC, PUBLISHED_EXTERNAL_AUC, PUBLISHED_ELDERLY_GAP,
    PUBLISHED_BRIER, PUBLISHED_F1, PUBLISHED_SENSITIVITY, PUBLISHED_SPECIFICITY,
    XGB_PARAMS, AGE_GROUPS, BMI_GROUPS, SEX_GROUPS, SEED,
)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score,
                              roc_curve, confusion_matrix,
                              precision_score, recall_score)
import xgboost as xgb
import joblib


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_metrics(y_true, y_prob, threshold=None):
    """Full metric set matching your IEEE paper Table II."""
    if threshold is None:
        threshold = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'auc'        : float(roc_auc_score(y_true, y_prob)),
        'brier'      : float(brier_score_loss(y_true, y_prob)),
        'f1'         : float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp / (tp+fn)) if (tp+fn) > 0 else 0.0,
        'specificity': float(tn / (tn+fp)) if (tn+fp) > 0 else 0.0,
        'ppv'        : float(precision_score(y_true, y_pred, zero_division=0)),
        'npv'        : float(tn / (tn+fn)) if (tn+fn) > 0 else 0.0,
        'threshold'  : float(threshold),
        'n'          : int(len(y_true)),
        'n_pos'      : int(y_true.sum()),
    }


def fairness_analysis(y_true, y_prob, df):
    """
    Subgroup AUC analysis — exact replication of your IEEE paper Table III.
    Returns dict with per-subgroup AUC + fairness gaps.
    """
    res = {}
    y_true = np.array(y_true)

    for label, (lo, hi) in AGE_GROUPS.items():
        mask = (df['RIDAGEYR'] >= lo) & (df['RIDAGEYR'] <= hi)
        if mask.sum() >= 30 and y_true[mask].sum() >= 5 and (y_true[mask]==0).sum() >= 5:
            res[f'age_{label}'] = {
                'auc' : float(roc_auc_score(y_true[mask], y_prob[mask])),
                'n'   : int(mask.sum()),
                'prev': float(y_true[mask].mean()),
            }

    for label, (lo, hi) in BMI_GROUPS.items():
        mask = (df['BMXBMI'] >= lo) & (df['BMXBMI'] <= hi)
        if mask.sum() >= 30 and y_true[mask].sum() >= 5 and (y_true[mask]==0).sum() >= 5:
            res[f'bmi_{label}'] = {
                'auc' : float(roc_auc_score(y_true[mask], y_prob[mask])),
                'n'   : int(mask.sum()),
                'prev': float(y_true[mask].mean()),
            }

    for label, val in SEX_GROUPS.items():
        mask = df['RIAGENDR'] == val
        if mask.sum() >= 30 and y_true[mask].sum() >= 5 and (y_true[mask]==0).sum() >= 5:
            res[f'sex_{label}'] = {
                'auc' : float(roc_auc_score(y_true[mask], y_prob[mask])),
                'n'   : int(mask.sum()),
                'prev': float(y_true[mask].mean()),
            }

    if 'age_18-39' in res and 'age_60+' in res:
        res['elderly_gap'] = float(res['age_18-39']['auc'] - res['age_60+']['auc'])
    if 'bmi_Normal' in res and 'bmi_Obese' in res:
        res['bmi_gap'] = float(res['bmi_Normal']['auc'] - res['bmi_Obese']['auc'])
    if 'sex_Male' in res and 'sex_Female' in res:
        res['sex_gap'] = float(abs(res['sex_Male']['auc'] - res['sex_Female']['auc']))

    return res


# ──────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

BLUE   = '#2563EB'
GREEN  = '#16A34A'
RED    = '#DC2626'
PURPLE = '#7C3AED'
GREY   = '#6B7280'

def plot_roc(y_true, y_prob, label, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color=BLUE,
            label=f'{label}\nAUC = {auc:.3f}')
    ax.plot([0,1],[0,1], '--', color=GREY, lw=1.2, label='Random (AUC = 0.500)')
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
    ax.set_xlabel('1 − Specificity (FPR)', fontsize=13)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=13)
    ax.set_title('ROC Curve — Centralised XGBoost\n'
                 '5-fold Cross-Validation on NHANES 2015–2020',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fairness_age(fairness_central, save_path):
    age_keys   = ['18-39', '40-59', '60+']
    age_labels = ['Age 18–39\n(Young)', 'Age 40–59\n(Middle-aged)', 'Age ≥60\n(Elderly)  ★']

    pub_vals  = [0.742, None,  0.607]
    cent_vals = [fairness_central.get(f'age_{k}', {}).get('auc', None) for k in age_keys]

    x = np.arange(len(age_labels))
    w = 0.32
    fig, ax = plt.subplots(figsize=(10, 6))

    b_pub  = ax.bar(x-w/2, [v or 0 for v in pub_vals],  w,
                    label='Published IEEE paper',  color='#94A3B8', alpha=0.9)
    b_cent = ax.bar(x+w/2, [v or 0 for v in cent_vals], w,
                    label='Our centralised baseline', color=BLUE, alpha=0.9)

    for bars in [b_pub, b_cent]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.004,
                        f'{h:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Gap annotation
    if cent_vals[0] and cent_vals[2]:
        gap = cent_vals[0] - cent_vals[2]
        ax.annotate(
            f'Fairness gap = {gap:.3f}\n(Federated learning targets this)',
            xy=(x[2]+w/2, cent_vals[2]),
            xytext=(x[2]-0.6, cent_vals[2]+0.05),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8),
            fontsize=10, color=RED, fontweight='bold',
        )

    ax.axhline(0.5, color='black', ls='--', lw=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels, fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=13)
    ax.set_ylim(0.45, 0.85)
    ax.set_title(
        'Age-Subgroup Fairness — Centralised Model\n'
        '★ The elderly performance gap is the key problem FL addresses',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SCRIPT 02 — CENTRALISED BASELINE")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading: {CENTRALISED_PATH}")
df  = pd.read_csv(CENTRALISED_PATH)
X   = df[FEATURE_COLS].values.astype(np.float32)
y   = df[TARGET_COL].values.astype(np.float32)
print(f"      n = {len(df):,}  |  Diabetes prevalence: {y.mean():.1%}")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X).astype(np.float32)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print(f"      Scaler saved → models/scaler.pkl")

# ── 5-fold CV XGBoost ─────────────────────────────────────────────────────────
print("\n[2/5] Running 5-fold CV — XGBoost (paper hyperparameters)...")
scale_pos = float((y==0).sum() / (y==1).sum())
xgb_cv    = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_prob_xgb = cross_val_predict(xgb_cv, X_sc, y, cv=cv, method='predict_proba')[:,1]

# ── 5-fold CV Logistic Regression baseline ───────────────────────────────────
print("       Running logistic regression baseline...")
lr_cv      = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)
y_prob_lr  = cross_val_predict(lr_cv, X_sc, y, cv=cv, method='predict_proba')[:,1]

# ── Train final model on full data ────────────────────────────────────────────
print("\n[3/5] Training final model on full dataset...")
xgb_final = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
xgb_final.fit(X_sc, y)
joblib.dump(xgb_final, os.path.join(MODELS_DIR, 'centralised_xgb.pkl'))
np.save(os.path.join(MODELS_DIR, 'centralised_probs.npy'), y_prob_xgb)
print(f"      Model saved → models/centralised_xgb.pkl")

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n[4/5] Computing metrics and fairness analysis...")
metrics_xgb = compute_metrics(y, y_prob_xgb)
metrics_lr  = compute_metrics(y, y_prob_lr)
fairness    = fairness_analysis(y, y_prob_xgb, df)

# Print results table
print("\n" + "─" * 65)
print(f"  {'Metric':<16} {'XGBoost':>10} {'LogReg':>10} {'Published':>10}")
print("─" * 65)
rows = [
    ('AUC',          metrics_xgb['auc'],         metrics_lr['auc'],         PUBLISHED_INTERNAL_AUC),
    ('Brier score',  metrics_xgb['brier'],        metrics_lr['brier'],       PUBLISHED_BRIER),
    ('F1',           metrics_xgb['f1'],           metrics_lr['f1'],          PUBLISHED_F1),
    ('Sensitivity',  metrics_xgb['sensitivity'],  metrics_lr['sensitivity'], PUBLISHED_SENSITIVITY),
    ('Specificity',  metrics_xgb['specificity'],  metrics_lr['specificity'], PUBLISHED_SPECIFICITY),
    ('PPV',          metrics_xgb['ppv'],          metrics_lr['ppv'],         0.438),
    ('NPV',          metrics_xgb['npv'],          metrics_lr['npv'],         0.901),
]
for name, xg, lr_, pub in rows:
    diff = xg - pub
    flag = '✅' if abs(diff) < 0.03 else '⚠️'
    print(f"  {name:<16} {xg:>10.3f} {lr_:>10.3f} {pub:>10.3f}  {flag}")

print("\n  FAIRNESS — Age subgroups")
print("─" * 65)
for key in ['age_18-39', 'age_40-59', 'age_60+']:
    v = fairness.get(key, {})
    if v:
        print(f"  {key:<16} AUC={v['auc']:.3f}  n={v['n']:,}  prev={v['prev']:.1%}")

gap = fairness.get('elderly_gap')
if gap:
    ref_gap = PUBLISHED_ELDERLY_GAP
    print(f"\n  Elderly fairness gap : {gap:.3f}  (published: {ref_gap})")
    if gap < ref_gap:
        print(f"  ✅ Our replication gap is {ref_gap-gap:.3f} smaller than paper")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, 'centralised_metrics.json'), 'w') as f:
    json.dump({'xgboost': metrics_xgb, 'logistic_reg': metrics_lr,
               'fairness': fairness}, f, indent=2)
print(f"\n  Saved → results/centralised_metrics.json")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating plots...")
plot_roc(y, y_prob_xgb, 'Centralised XGBoost (5-fold CV)',
         os.path.join(PLOTS_DIR, '01_centralised_roc.png'))
plot_fairness_age(fairness,
                  os.path.join(PLOTS_DIR, '02_centralised_fairness_age.png'))

print("\n" + "=" * 65)
print("✅  Script 02 complete — run 03_federated_simulation.py next")
print("=" * 65)
