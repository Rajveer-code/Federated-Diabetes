"""
exp3_fedprox_sensitivity.py — FedProx μ sensitivity (μ=0.01, 0.05)
====================================================================
Validates that μ=0.1 suppresses Node B's learning signal by comparing
lower μ values. Supplements the existing μ=0.1 result (AUC=0.752, gap=0.066).

Produces:
  results/exp3_fedprox_sensitivity.json
"""

import os, sys, json, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, NODE_NAMES, RESULTS_DIR, MODELS_DIR, GLOBAL_SCALER_PATH,
    FEATURE_COLS, TARGET_COL, AGE_GROUPS,
    NN_LOCAL_EPOCHS, NN_LR, NN_WEIGHT_DECAY,
    FL_NUM_ROUNDS, CI_ALPHA, RANDOM_SEED,
)
from nn_model import DiabetesNet, train_one_epoch, get_device
from data_utils import (load_node_data, get_dataloaders, compute_class_weight,
                        get_params_as_numpy, set_params_from_numpy)

BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
BRFSS_COL_MAP = {
    'Age': 'RIDAGEYR', 'Gender': 'RIAGENDR', 'Race_Ethnicity': 'RIDRETH3',
    'BMI': 'BMXBMI', 'Smoking_Status': 'SMOKING', 'Physical_Activity': 'PHYS_ACTIVITY',
    'History_Heart_Attack': 'HEART_ATTACK', 'History_Stroke': 'STROKE',
    'Diabetes_Outcome': 'DIABETES',
}

MU_VALUES  = [0.01, 0.05]
BATCH_SIZE = 64
DEVICE     = get_device()

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("=" * 65)
print("  EXP 3 — FEDPROX μ SENSITIVITY")
print(f"  μ values: {MU_VALUES}  (μ=0.10 already in paper)")
print(f"  Rounds={FL_NUM_ROUNDS}  LocalEpochs={NN_LOCAL_EPOCHS}")
print("=" * 65)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── CI helpers ────────────────────────────────────────────────────────────────

def delong_ci_fast(y_true, y_score, alpha=CI_ALPHA):
    """O(n log n) DeLong CI via searchsorted."""
    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    pos_sorted = np.sort(y_score[y_true == 1])
    neg_sorted = np.sort(y_score[y_true == 0])
    n_pos, n_neg = len(pos_sorted), len(neg_sorted)
    lft = np.searchsorted(neg_sorted, pos_sorted, side='left')
    rgt = np.searchsorted(neg_sorted, pos_sorted, side='right')
    V10 = (lft + 0.5 * (rgt - lft)) / n_neg
    lft2 = np.searchsorted(pos_sorted, neg_sorted, side='left')
    rgt2 = np.searchsorted(pos_sorted, neg_sorted, side='right')
    V01  = (lft2 + 0.5 * (rgt2 - lft2)) / n_pos
    auc  = float(V10.mean())
    se   = float(np.sqrt(np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg))
    z    = stats.norm.ppf(1 - alpha / 2)
    return auc, float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


# ── Load all 3 nodes ──────────────────────────────────────────────────────────
print("\n[1/3] Loading node data (all 3 nodes)...")
loaders, n_samples = [], []
for path, name in zip(NODE_PATHS, NODE_NAMES):
    X_tr, y_tr, X_val, y_val, sc = load_node_data(path, val_size=0.2, seed=RANDOM_SEED)
    tr_dl, val_dl = get_dataloaders(X_tr, y_tr, X_val, y_val, BATCH_SIZE)
    loaders.append((tr_dl, val_dl, y_tr, y_val))
    n_samples.append(len(X_tr))
    print(f"  {name}: train={len(X_tr):,}  prevalence={y_tr.mean():.1%}")


# ── BRFSS preprocessing (once) ────────────────────────────────────────────────
print("\n[2/3] Loading BRFSS data...")
df_brfss = pd.read_csv(BRFSS_PATH)
df_brfss = df_brfss.rename(columns=BRFSS_COL_MAP)
if df_brfss['RIAGENDR'].max() <= 1.0:
    df_brfss['RIAGENDR'] = df_brfss['RIAGENDR'].map({1.0: 1.0, 0.0: 2.0})
df_brfss = df_brfss.dropna(subset=['DIABETES'])
df_brfss['DIABETES'] = df_brfss['DIABETES'].astype(int)
for col in ['BMXBMI', 'RIDAGEYR']:
    df_brfss[col] = df_brfss[col].fillna(df_brfss[col].median())
for col in ['RIAGENDR', 'RIDRETH3', 'SMOKING', 'PHYS_ACTIVITY', 'HEART_ATTACK', 'STROKE']:
    df_brfss[col] = df_brfss[col].fillna(df_brfss[col].mode()[0])
df_brfss = df_brfss.reset_index(drop=True)
X_brfss    = df_brfss[FEATURE_COLS].values.astype(np.float32)
y_brfss    = df_brfss['DIABETES'].values.astype(np.float32)
global_scaler = joblib.load(GLOBAL_SCALER_PATH)
X_brfss_sc = global_scaler.transform(X_brfss).astype(np.float32)
print(f"  BRFSS n={len(df_brfss):,}  prevalence={y_brfss.mean():.1%}")


# ── FL helpers ────────────────────────────────────────────────────────────────

def local_train(global_params, train_dl, y_train, proximal_mu):
    pos_weight   = compute_class_weight(y_train)
    model        = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, global_params)
    criterion    = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    optimizer    = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
    global_tensors = [p.detach().clone() for p in model.parameters()]
    for _ in range(NN_LOCAL_EPOCHS):
        train_one_epoch(model, train_dl, optimizer, criterion,
                        DEVICE, proximal_mu, global_tensors)
    return get_params_as_numpy(model)


def fedavg_aggregate(updates, n_samples):
    total = sum(n_samples)
    return [sum((n / total) * updates[c][i]
                for c, n in enumerate(n_samples))
            for i in range(len(updates[0]))]


def eval_on_brfss(model):
    model.eval()
    chunk = 50_000
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_brfss_sc), chunk):
            X_c = torch.FloatTensor(X_brfss_sc[i:i+chunk]).to(DEVICE)
            probs.append(torch.sigmoid(model(X_c)).cpu().numpy())
    return np.concatenate(probs)


def subgroup_aucs(y_prob):
    subs = {}
    for label, (lo, hi) in AGE_GROUPS.items():
        mask = (df_brfss['RIDAGEYR'] >= lo) & (df_brfss['RIDAGEYR'] <= hi)
        if mask.sum() >= 100 and y_brfss[mask].sum() >= 20:
            key = f'age_{label.replace("-","_").replace("+","_plus")}'
            subs[key] = float(roc_auc_score(y_brfss[mask], y_prob[mask]))
    elderly_key = 'age_60_plus'
    young_key   = 'age_18_39'
    gap = float(subs[young_key] - subs[elderly_key]) \
          if elderly_key in subs and young_key in subs else None
    return subs, gap


# ── Run FedProx for each μ ────────────────────────────────────────────────────
print(f"\n[3/3] Running FedProx for μ ∈ {MU_VALUES}...")
results = {}

for mu in MU_VALUES:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print(f"\n  ── FedProx μ={mu} ──")
    init_model    = DiabetesNet()
    global_params = get_params_as_numpy(init_model)
    t0 = time.time()

    for rnd in range(1, FL_NUM_ROUNDS + 1):
        client_updates = []
        for tr_dl, _, y_tr, _ in loaders:
            updated = local_train([p.copy() for p in global_params], tr_dl, y_tr, mu)
            client_updates.append(updated)
        global_params = fedavg_aggregate(client_updates, n_samples)
        if rnd % 10 == 0 or rnd == 1:
            print(f"    Round {rnd:3d}/{FL_NUM_ROUNDS}  elapsed={time.time()-t0:.0f}s")

    final_model = DiabetesNet()
    set_params_from_numpy(final_model, global_params)
    final_model.to(DEVICE)
    weights_path = os.path.join(MODELS_DIR, f'fedprox_mu{str(mu).replace(".","")}_weights.pt')
    torch.save(final_model.state_dict(), weights_path)

    y_prob_ext   = eval_on_brfss(final_model)
    auc_ext, ci_lo, ci_hi = delong_ci_fast(y_brfss, y_prob_ext)
    subs, gap    = subgroup_aucs(y_prob_ext)
    young_auc    = subs.get('age_18_39', None)
    elderly_auc  = subs.get('age_60_plus', None)

    mu_key = f'mu_{str(mu).replace(".","_")}'
    results[mu_key] = {
        'external_auc': auc_ext,
        'ci_low':       ci_lo,
        'ci_high':      ci_hi,
        'elderly_gap':  gap,
        'elderly_auc':  elderly_auc,
        'young_auc':    young_auc,
    }
    print(f"    External AUC={auc_ext:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]  "
          f"Elderly gap={gap:.3f}" if gap else f"    External AUC={auc_ext:.3f}")
    print(f"    Runtime: {time.time()-t0:.0f}s")

# Add existing μ=0.10 result
results['mu_0_10_existing'] = {
    'external_auc': 0.752,
    'elderly_gap':  0.066,
}

out_path = os.path.join(RESULTS_DIR, 'exp3_fedprox_sensitivity.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*65}")
print("  FedProx μ Sensitivity Results:")
for k, v in results.items():
    if k == 'mu_0_10_existing':
        print(f"  μ=0.10 (existing): AUC={v['external_auc']}  Gap={v['elderly_gap']}")
    else:
        mu_label = k.replace('mu_', 'μ=').replace('_', '.')
        print(f"  {mu_label}: AUC={v['external_auc']:.3f}  Gap={v['elderly_gap']:.3f}"
              if v['elderly_gap'] else f"  {mu_label}: AUC={v['external_auc']:.3f}")
print(f"{'='*65}")
print(f"\n✓ DONE: exp3_fedprox_sensitivity.py — results saved to {out_path}")
