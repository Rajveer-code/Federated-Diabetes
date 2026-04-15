"""
SCRIPT 05 — FAIRNESS ANALYSIS
================================
The core scientific contribution of this paper.

Compares subgroup AUC across ALL demographic groups for:
  - Published IEEE model (your centralised XGBoost)
  - Our centralised replication
  - Best federated model (FedProx)

Key question: Does federated training on demographically heterogeneous
nodes REDUCE the elderly AUC gap of 0.135 from your IEEE paper?

Produces:
  results/fairness_comparison.json
  plots/06_fairness_age_comparison.png
  plots/07_fairness_full_profile.png
  plots/08_node_b_analysis.png  (the elderly node deep-dive)

Usage:
  cd D:\Projects\diabetes_prediction_project\federated
  python 05_fairness_analysis.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, NODE_NAMES, CENTRALISED_PATH,
    RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_COL, GLOBAL_SCALER_PATH,
    XGB_PARAMS, AGE_GROUPS, BMI_GROUPS, SEX_GROUPS,
    PUBLISHED_INTERNAL_AUC, PUBLISHED_ELDERLY_GAP,
    PUBLISHED_ELDERLY_AUC, PUBLISHED_YOUNG_AUC,
    NN_LR, NN_LOCAL_EPOCHS, NN_BATCH_SIZE, NN_WEIGHT_DECAY,
    FL_NUM_ROUNDS, FEDPROX_MU, SEED,
)
from nn_model   import DiabetesNet, train_one_epoch, get_device
from data_utils import (DiabetesDataset, load_node_data, get_dataloaders,
                         compute_class_weight, set_params_from_numpy,
                         get_params_as_numpy)

DEVICE = get_device()
BLUE   = '#2563EB'
GREEN  = '#16A34A'
PURPLE = '#7C3AED'
RED    = '#DC2626'
GREY   = '#94A3B8'


# ──────────────────────────────────────────────────────────────────────────────
#  FAIRNESS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_prob):
    try:
        if y_true.sum() < 5 or (y_true==0).sum() < 5:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def full_fairness(y_true: np.ndarray, y_prob: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Compute all subgroup AUCs from your IEEE paper Table III.
    Returns flat dict: {'age_18-39': 0.742, 'age_60+': 0.607, 'elderly_gap': 0.135, ...}
    """
    res = {}
    y_true = np.array(y_true)

    for label, (lo, hi) in AGE_GROUPS.items():
        mask = (df['RIDAGEYR'].values >= lo) & (df['RIDAGEYR'].values <= hi)
        auc  = safe_auc(y_true[mask], y_prob[mask])
        if auc:
            res[f'age_{label}'] = auc

    for label, (lo, hi) in BMI_GROUPS.items():
        mask = (df['BMXBMI'].values >= lo) & (df['BMXBMI'].values <= hi)
        auc  = safe_auc(y_true[mask], y_prob[mask])
        if auc:
            res[f'bmi_{label}'] = auc

    for label, val in SEX_GROUPS.items():
        mask = df['RIAGENDR'].values == val
        auc  = safe_auc(y_true[mask], y_prob[mask])
        if auc:
            res[f'sex_{label}'] = auc

    if 'age_18-39' in res and 'age_60+' in res:
        res['elderly_gap'] = round(res['age_18-39'] - res['age_60+'], 5)
    if 'bmi_Normal' in res and 'bmi_Obese' in res:
        res['bmi_gap'] = round(res['bmi_Normal'] - res['bmi_Obese'], 5)
    if 'sex_Male' in res and 'sex_Female' in res:
        res['sex_gap'] = round(abs(res['sex_Male'] - res['sex_Female']), 5)

    return res


def equalized_odds_diff(y_true, y_score, sensitive_attr, threshold=None):
    """
    Equalized Odds Difference = max(|DELTA_TPR|, |DELTA_FPR|) across groups.

    Threshold: if None, uses Youden's J on the full population.
    Groups:    sensitive_attr == 0  vs  sensitive_attr == 1.

    Returns dict with per-group TPR/FPR and the EOD scalar.
    Reference: Hardt, Price & Srebro (NeurIPS 2016).
    """
    y_true         = np.asarray(y_true)
    y_score        = np.asarray(y_score)
    sensitive_attr = np.asarray(sensitive_attr)

    if threshold is None:
        fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_score)
        threshold = float(thresholds_all[np.argmax(tpr_all - fpr_all)])

    y_pred  = (y_score >= threshold).astype(int)
    results = {}
    for g in [0, 1]:
        m  = sensitive_attr == g
        tp = np.sum((y_pred[m] == 1) & (y_true[m] == 1))
        fn = np.sum((y_pred[m] == 0) & (y_true[m] == 1))
        fp = np.sum((y_pred[m] == 1) & (y_true[m] == 0))
        tn = np.sum((y_pred[m] == 0) & (y_true[m] == 0))
        results[g] = {
            'tpr': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'n'  : int(m.sum()),
        }
    eod = max(
        abs(results[0]['tpr'] - results[1]['tpr']),
        abs(results[0]['fpr'] - results[1]['fpr']),
    )
    results['eod']       = float(eod)
    results['threshold'] = float(threshold)
    return results


def youden_j_per_subgroup(y_true, y_score, sensitive_attr, subgroup_names=None):
    """
    Youden's J = sensitivity + specificity - 1 per subgroup.

    Uses the globally optimal Youden threshold applied consistently to all
    subgroups (not a separate threshold per subgroup).

    Returns dict with global threshold, global J, and per-subgroup stats.
    """
    y_true         = np.asarray(y_true)
    y_score        = np.asarray(y_score)
    sensitive_attr = np.asarray(sensitive_attr)

    # Global optimal threshold
    fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_score)
    best_idx  = int(np.argmax(tpr_all - fpr_all))
    threshold = float(thresholds_all[best_idx])
    global_j  = float((tpr_all - fpr_all)[best_idx])

    y_pred = (y_score >= threshold).astype(int)
    groups = np.unique(sensitive_attr)
    names  = subgroup_names or {g: f'Group_{g}' for g in groups}

    results = {
        'global_threshold': threshold,
        'global_j'        : global_j,
        'subgroups'       : {},
    }
    for g in groups:
        m    = sensitive_attr == g
        tp   = np.sum((y_pred[m] == 1) & (y_true[m] == 1))
        fn   = np.sum((y_pred[m] == 0) & (y_true[m] == 1))
        fp   = np.sum((y_pred[m] == 1) & (y_true[m] == 0))
        tn   = np.sum((y_pred[m] == 0) & (y_true[m] == 0))
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        results['subgroups'][names[g]] = {
            'n'          : int(m.sum()),
            'n_pos'      : int(y_true[m].sum()),
            'sensitivity': sens,
            'specificity': spec,
            'youden_j'   : sens + spec - 1.0,
        }
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  (1) CENTRALISED XGBOOST
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SCRIPT 05 — FAIRNESS ANALYSIS")
print("=" * 65)

print("\n[1/4] Centralised XGBoost (5-fold CV)...")
df_c  = pd.read_csv(CENTRALISED_PATH)
X_c   = df_c[FEATURE_COLS].values.astype(np.float32)
y_c   = df_c[TARGET_COL].values.astype(np.float32)

scaler   = StandardScaler()
X_c_sc   = scaler.fit_transform(X_c).astype(np.float32)
scale_pos = float((y_c==0).sum() / (y_c==1).sum())

xgb_clf  = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_prob_c  = cross_val_predict(xgb_clf, X_c_sc, y_c, cv=cv, method='predict_proba')[:,1]

fair_central = full_fairness(y_c, y_prob_c, df_c)
print(f"  Overall AUC     : {roc_auc_score(y_c, y_prob_c):.4f}")
print(f"  Elderly gap     : {fair_central.get('elderly_gap', 'N/A'):.4f}  "
      f"(published: {PUBLISHED_ELDERLY_GAP})")


# ──────────────────────────────────────────────────────────────────────────────
#  (2) FEDERATED MODEL (FedProx) — train or load
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Federated model (FedProx)...")
weights_path = os.path.join(MODELS_DIR, 'fedprox_weights.pt')

def train_fedprox(num_rounds=FL_NUM_ROUNDS):
    """Train FedProx directly (no Flower dependency for this script)."""
    # Load all nodes
    all_loaders, all_n, all_y = [], [], []
    for path in NODE_PATHS:
        X_tr, y_tr, X_val, y_val, _ = load_node_data(
            path, val_size=0.2, seed=SEED
        )
        tr_dl, _ = get_dataloaders(X_tr, y_tr, X_val, y_val, NN_BATCH_SIZE)
        all_loaders.append(tr_dl)
        all_n.append(len(X_tr))
        all_y.append(y_tr)

    total = sum(all_n)

    # Initialise global model
    global_params = get_params_as_numpy(DiabetesNet())

    for rnd in range(1, num_rounds + 1):
        client_updates = []
        for i, (dl, y_tr) in enumerate(zip(all_loaders, all_y)):
            pos_w  = compute_class_weight(y_tr)
            model  = DiabetesNet().to(DEVICE)
            set_params_from_numpy(model, [p.copy() for p in global_params])

            crit = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_w], device=DEVICE)
            )
            opt = torch.optim.AdamW(
                model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY
            )
            # Global params for proximal term
            g_tensors = [p.detach().clone() for p in model.parameters()]

            for _ in range(NN_LOCAL_EPOCHS):
                train_one_epoch(model, dl, opt, crit, DEVICE, FEDPROX_MU, g_tensors)

            client_updates.append(get_params_as_numpy(model))

        # FedAvg aggregation
        global_params = [
            sum((all_n[c]/total) * client_updates[c][l]
                for c in range(len(client_updates)))
            for l in range(len(global_params))
        ]

        if rnd % 10 == 0 or rnd == 1:
            # Quick eval on centralised set
            m = DiabetesNet().to(DEVICE)
            set_params_from_numpy(m, global_params)
            m.eval()
            with torch.no_grad():
                probs = torch.sigmoid(
                    m(torch.FloatTensor(X_c_sc).to(DEVICE))
                ).cpu().numpy()
            auc = roc_auc_score(y_c, probs)
            print(f"    Round {rnd:3d}/{num_rounds}  AUC={auc:.4f}")

    # Build final model
    final = DiabetesNet()
    set_params_from_numpy(final, global_params)
    return final


if os.path.exists(weights_path):
    print(f"  Loading saved weights: {weights_path}")
    fed_model = DiabetesNet()
    fed_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
else:
    print(f"  Training FedProx ({FL_NUM_ROUNDS} rounds)...")
    fed_model = train_fedprox()
    torch.save(fed_model.state_dict(), weights_path)
    print(f"  Saved: {weights_path}")

fed_model = fed_model.to(DEVICE)
fed_model.eval()
with torch.no_grad():
    y_prob_fed = torch.sigmoid(
        fed_model(torch.FloatTensor(X_c_sc).to(DEVICE))
    ).cpu().numpy()

fair_fed = full_fairness(y_c, y_prob_fed, df_c)
print(f"  Overall AUC     : {roc_auc_score(y_c, y_prob_fed):.4f}")
print(f"  Elderly gap     : {fair_fed.get('elderly_gap', 'N/A'):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
#  (3) PRINT FULL COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Full fairness comparison table...")

PUBLISHED = {
    'age_18-39'      : PUBLISHED_YOUNG_AUC,
    'age_60+'        : PUBLISHED_ELDERLY_AUC,
    'elderly_gap'    : PUBLISHED_ELDERLY_GAP,
    'bmi_Normal'     : 0.735,
    'bmi_Obese'      : 0.698,
    'sex_Male'       : 0.723,
    'sex_Female'     : 0.712,
}

subgroups = [
    ('age_18-39',     'Age 18–39'),
    ('age_40-59',     'Age 40–59'),
    ('age_60+',       'Age 60+   ★'),
    ('bmi_Normal',    'BMI Normal'),
    ('bmi_Overweight','BMI Overweight'),
    ('bmi_Obese',     'BMI Obese'),
    ('sex_Male',      'Male'),
    ('sex_Female',    'Female'),
]

print("\n" + "─"*70)
print(f"  {'Subgroup':<22} {'Published':>10} {'Centralised':>12} "
      f"{'Federated':>11} {'Δ(Fed-Cen)':>11}")
print("─"*70)

table = {}
for key, label in subgroups:
    pub  = PUBLISHED.get(key)
    cen  = fair_central.get(key)
    fed  = fair_fed.get(key)
    delta = (fed - cen) if (fed and cen) else None
    flag  = ('✅' if delta and delta > 0 else '⬇') if delta else ''

    print(f"  {label:<22} "
          f"{f'{pub:.3f}' if pub else '  —  ':>10} "
          f"{f'{cen:.3f}' if cen else '  —  ':>12} "
          f"{f'{fed:.3f}' if fed else '  —  ':>11} "
          f"{f'{delta:+.3f}' if delta else '  —  ':>11}  {flag}")

    table[key] = {'published': pub, 'centralised': cen, 'federated': fed}

print("─"*70)

cen_gap = fair_central.get('elderly_gap', 0)
fed_gap = fair_fed.get('elderly_gap', 0)
if cen_gap and fed_gap:
    abs_red  = cen_gap - fed_gap
    rel_red  = (abs_red / cen_gap) * 100
    print(f"\n  ★ ELDERLY FAIRNESS GAP REDUCTION")
    print(f"    Published baseline  : {PUBLISHED_ELDERLY_GAP:.3f}")
    print(f"    Centralised (ours)  : {cen_gap:.3f}")
    print(f"    Federated (FedProx) : {fed_gap:.3f}")
    print(f"    Absolute reduction  : {abs_red:.3f}")
    print(f"    Relative reduction  : {rel_red:.1f}%")
    if fed_gap < cen_gap:
        print(f"    ✅ Federated training REDUCES the elderly fairness gap")
    else:
        print(f"    ⚠️  Federated training does not reduce the elderly gap")
        print(f"       (This is itself a meaningful finding for the paper)")

# ── Equalized Odds Difference and Youden's J ──────────────────────────────────
print("\n[3b/4] Equalized Odds Difference and Youden's J (JBI/npj required)...")

# Binary sensitive attributes: age (<60=0, >=60=1) and race (non-White=0, White=1)
age_binary  = (df_c['RIDAGEYR'].values >= 60).astype(int)   # 0=young, 1=elderly
race_binary = (df_c['RIDRETH3'].values == 3).astype(int)    # 1=White (RIDRETH3==3), 0=non-White

age_names  = {0: 'Young (<60)', 1: 'Elderly (>=60)'}
race_names = {0: 'Non-White', 1: 'White (NH)'}

# Centralised EOD
eod_age_cen  = equalized_odds_diff(y_c, y_prob_c,   age_binary)
eod_race_cen = equalized_odds_diff(y_c, y_prob_c,   race_binary)
# Federated EOD
eod_age_fed  = equalized_odds_diff(y_c, y_prob_fed, age_binary)
eod_race_fed = equalized_odds_diff(y_c, y_prob_fed, race_binary)

# Youden's J
yj_age_cen   = youden_j_per_subgroup(y_c, y_prob_c,   age_binary,  age_names)
yj_race_cen  = youden_j_per_subgroup(y_c, y_prob_c,   race_binary, race_names)
yj_age_fed   = youden_j_per_subgroup(y_c, y_prob_fed, age_binary,  age_names)
yj_race_fed  = youden_j_per_subgroup(y_c, y_prob_fed, race_binary, race_names)

print("\n  Equalized Odds Difference (lower = fairer):")
print(f"    Age  — Centralised: {eod_age_cen['eod']:.4f}   FedProx: {eod_age_fed['eod']:.4f}")
print(f"    Race — Centralised: {eod_race_cen['eod']:.4f}   FedProx: {eod_race_fed['eod']:.4f}")
print("\n  Youden's J per subgroup (global threshold applied consistently):")
for name, val in yj_age_cen['subgroups'].items():
    fed_j = yj_age_fed['subgroups'].get(name, {}).get('youden_j', float('nan'))
    print(f"    {name:<20}  Centralised J={val['youden_j']:.4f}   FedProx J={fed_j:.4f}")
for name, val in yj_race_cen['subgroups'].items():
    fed_j = yj_race_fed['subgroups'].get(name, {}).get('youden_j', float('nan'))
    print(f"    {name:<20}  Centralised J={val['youden_j']:.4f}   FedProx J={fed_j:.4f}")

fairness_metrics = {
    'eod': {
        'age': {
            'centralised': eod_age_cen,
            'federated'  : eod_age_fed,
        },
        'race': {
            'centralised': eod_race_cen,
            'federated'  : eod_race_fed,
        },
    },
    'youden_j': {
        'age': {
            'centralised': yj_age_cen,
            'federated'  : yj_age_fed,
        },
        'race': {
            'centralised': yj_race_cen,
            'federated'  : yj_race_fed,
        },
    },
}
with open(os.path.join(RESULTS_DIR, 'fairness_metrics.json'), 'w') as f:
    json.dump(fairness_metrics, f, indent=2)
print(f"  Saved → results/fairness_metrics.json")

# Save
all_fairness = {
    'centralised'    : fair_central,
    'federated'      : fair_fed,
    'published'      : PUBLISHED,
    'table'          : table,
    'fairness_metrics': fairness_metrics,
}
with open(os.path.join(RESULTS_DIR, 'fairness_comparison.json'), 'w') as f:
    json.dump(all_fairness, f, indent=2)
print(f"  Saved → results/fairness_comparison.json")


# ──────────────────────────────────────────────────────────────────────────────
#  (4) PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Generating fairness plots...")

# ─ Plot 1: Age subgroup bar chart ─────────────────────────────────────────────
age_keys   = ['18-39', '40-59', '60+']
age_labels = ['Age 18–39\n(Young)', 'Age 40–59\n(Middle)', 'Age ≥60\n(Elderly)  ★']
pub_vals  = [PUBLISHED.get(f'age_{k}') for k in age_keys]
cen_vals  = [fair_central.get(f'age_{k}') for k in age_keys]
fed_vals  = [fair_fed.get(f'age_{k}')     for k in age_keys]

x = np.arange(len(age_labels))
w = 0.24
fig, ax = plt.subplots(figsize=(11, 7))

b1 = ax.bar(x-w,   [v or 0 for v in pub_vals], w, label='Published IEEE paper',
            color=GREY,   alpha=0.9)
b2 = ax.bar(x,     [v or 0 for v in cen_vals], w, label='Our centralised baseline',
            color=BLUE,   alpha=0.9)
b3 = ax.bar(x+w,   [v or 0 for v in fed_vals], w, label='Federated FedProx',
            color=PURPLE, alpha=0.9)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.004, f'{h:.3f}',
                    ha='center', fontsize=9.5, fontweight='bold')

# Annotate gap improvement on elderly bar
if cen_vals[2] and fed_vals[2]:
    delta = fed_vals[2] - cen_vals[2]
    col   = GREEN if delta > 0 else RED
    sym   = '+' if delta > 0 else ''
    ax.annotate(
        f'Δ = {sym}{delta:.3f}',
        xy=(x[2]+w, fed_vals[2]+0.01),
        ha='center', fontsize=11, fontweight='bold', color=col,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )

ax.axhline(0.5, color='black', ls='--', lw=1, alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(age_labels, fontsize=12)
ax.set_ylabel('AUC-ROC', fontsize=13)
ax.set_ylim(0.48, 0.86)
ax.set_title(
    'Age-Group Fairness: Published vs Centralised vs Federated\n'
    'Core finding: Does federated training reduce the elderly performance gap?',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

if cen_gap and fed_gap:
    ax.text(
        0.02, 0.04,
        f"★ Elderly fairness gap:\n"
        f"   Published: {PUBLISHED_ELDERLY_GAP:.3f}\n"
        f"   Centralised: {cen_gap:.3f}\n"
        f"   Federated: {fed_gap:.3f}\n"
        f"   Reduction: {(cen_gap-fed_gap)/cen_gap*100:.1f}%",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFBEB', alpha=0.9)
    )

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_fairness_age_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/06_fairness_age_comparison.png")


# ─ Plot 2: Full subgroup line profile ─────────────────────────────────────────
plot_pairs = [
    ('age_18-39','Age 18–39'), ('age_40-59','Age 40–59'), ('age_60+','Age 60+'),
    ('bmi_Normal','BMI Normal'), ('bmi_Overweight','BMI Overweight'),
    ('bmi_Obese','BMI Obese'), ('sex_Male','Male'), ('sex_Female','Female'),
]
keys_p   = [k for k,_ in plot_pairs]
labels_p = [l for _,l in plot_pairs]

pub_v  = [PUBLISHED.get(k) for k in keys_p]
cen_v  = [fair_central.get(k) for k in keys_p]
fed_v  = [fair_fed.get(k)     for k in keys_p]
x_p    = np.arange(len(keys_p))

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(x_p, pub_v, 'o--', color=GREY,   lw=2,   ms=8, label='Published IEEE paper')
ax.plot(x_p, cen_v, 's-',  color=BLUE,   lw=2.5, ms=9, label='Centralised (our replication)')
ax.plot(x_p, fed_v, '^-',  color=PURPLE, lw=2.5, ms=9, label='Federated (FedProx)')

# Shade improvements
for i, (c, f) in enumerate(zip(cen_v, fed_v)):
    if c and f:
        col = '#16A34A' if f >= c else '#DC2626'
        ax.fill_between([i-0.1, i+0.1], [c,c], [f,f], color=col, alpha=0.25)

ax.axhline(0.5, color='black', ls=':', lw=1, alpha=0.4)
ax.set_xticks(x_p)
ax.set_xticklabels(labels_p, rotation=20, ha='right', fontsize=11)
ax.set_ylabel('AUC-ROC', fontsize=13)
ax.set_ylim(0.45, 0.87)
ax.set_title(
    'Full Subgroup Fairness Profile — All Demographic Groups\n'
    'Green shading = federated improves over centralised  |  '
    'Red = worsens',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_fairness_full_profile.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/07_fairness_full_profile.png")


# ─ Plot 3: Node B deep-dive (the elderly node) ─────────────────────────────────
df_b  = pd.read_csv(NODE_PATHS[1])   # Node B
X_b   = df_b[FEATURE_COLS].values.astype(np.float32)
y_b   = df_b[TARGET_COL].values.astype(np.float32)

# CORRECT — load the global NHANES-fitted scaler, never refit on node data
# (a fresh fit_transform on X_b would contaminate evaluation with node-local
#  statistics instead of the NHANES training statistics — data leakage bug)
_global_scaler = joblib.load(GLOBAL_SCALER_PATH)
X_b_sc = _global_scaler.transform(X_b).astype(np.float32)

# XGBoost predictions on Node B
xgb_b = xgb.XGBClassifier(
    **XGB_PARAMS,
    scale_pos_weight=float((y_b==0).sum()/(y_b==1).sum())
)
from sklearn.model_selection import cross_val_predict as cvp
y_prob_b_xgb = cvp(
    xgb_b, X_b_sc, y_b,
    cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    method='predict_proba'
)[:,1]

# Federated model predictions on Node B
fed_model.eval()
with torch.no_grad():
    y_prob_b_fed = torch.sigmoid(
        fed_model(torch.FloatTensor(X_b_sc).to(DEVICE))
    ).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, (probs, title, color) in zip(axes, [
    (y_prob_b_xgb, 'Centralised XGBoost\non Node B (Elderly Rural)', BLUE),
    (y_prob_b_fed, 'Federated FedProx\non Node B (Elderly Rural)', PURPLE),
]):
    fpr, tpr, _ = roc_curve(y_b, probs)
    auc = roc_auc_score(y_b, probs)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'AUC = {auc:.3f}')
    ax.plot([0,1],[0,1], '--', color=GREY, lw=1.2)
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)
    ax.set_xlabel('1 − Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity',     fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

fig.suptitle(
    'Node B Deep-Dive: Elderly Rural Population (82.4% age ≥60, 28.5% diabetes)\n'
    'This is the population your IEEE model failed on most severely',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_node_b_elderly_analysis.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/08_node_b_elderly_analysis.png")

print("\n✅  Script 05 complete — run 06_results_summary.py next")
print("=" * 65)
