"""
exp2_node_b_ablation.py — FedAvg with Node B removed (A + C only)
==================================================================
Proves that the elderly fairness improvement is causally due to Node B's
demographic distribution, not just the FL algorithm.

Produces:
  results/exp2_node_b_ablation.json
"""

import os, sys, json, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score,
                              roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_A_PATH, NODE_C_PATH, CENTRALISED_PATH,
    RESULTS_DIR, MODELS_DIR, GLOBAL_SCALER_PATH,
    FEATURE_COLS, TARGET_COL, AGE_GROUPS,
    NN_LR, NN_WEIGHT_DECAY,
    FL_NUM_ROUNDS, N_BOOTSTRAP, CI_ALPHA, RANDOM_SEED, TEST_SIZE,
)
from nn_model import DiabetesNet, train_one_epoch, get_device
from data_utils import (DiabetesDataset, load_node_data, get_dataloaders,
                        compute_class_weight, get_params_as_numpy, set_params_from_numpy)

BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
BRFSS_COL_MAP = {
    'Age': 'RIDAGEYR', 'Gender': 'RIAGENDR', 'Race_Ethnicity': 'RIDRETH3',
    'BMI': 'BMXBMI', 'Smoking_Status': 'SMOKING', 'Physical_Activity': 'PHYS_ACTIVITY',
    'History_Heart_Attack': 'HEART_ATTACK', 'History_Stroke': 'STROKE',
    'Diabetes_Outcome': 'DIABETES',
}

BATCH_SIZE = 64
# Per-node local epochs for ablation (consistent with node design)
NODE_LOCAL_EPOCHS = {0: 5, 1: 4}  # Node A=5, Node C=4

DEVICE = get_device()
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("=" * 65)
print("  EXP 2 — NODE B ABLATION (FedAvg: Node A + Node C only)")
print(f"  Rounds={FL_NUM_ROUNDS}  Nodes: A({NODE_LOCAL_EPOCHS[0]} epochs) + C({NODE_LOCAL_EPOCHS[1]} epochs)")
print("=" * 65)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── CI helpers ────────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP,
                     alpha=CI_ALPHA, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n_bootstrap):
        bi = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        if len(np.unique(y_true[bi])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[bi], y_score[bi])))
    aucs = np.array(aucs)
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))))


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


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def compute_metrics(y_true, y_prob):
    thr = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'auc'        : float(roc_auc_score(y_true, y_prob)),
        'brier'      : float(brier_score_loss(y_true, y_prob)),
        'f1'         : float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


# ── Load Node A and Node C only ───────────────────────────────────────────────
print("\n[1/5] Loading Node A and Node C data (Node B excluded)...")
node_paths  = [NODE_A_PATH, NODE_C_PATH]
node_names  = ['Node A — Young Urban', 'Node C — Mixed Metro']
loaders, n_samples = [], []

for path, name in zip(node_paths, node_names):
    X_tr, y_tr, X_val, y_val, sc = load_node_data(path, val_size=0.2, seed=RANDOM_SEED)
    tr_dl, val_dl = get_dataloaders(X_tr, y_tr, X_val, y_val, BATCH_SIZE)
    loaders.append((tr_dl, val_dl, y_tr, y_val))
    n_samples.append(len(X_tr))
    print(f"  {name}: train={len(X_tr):,}  val={len(X_val):,}  "
          f"prevalence={y_tr.mean():.1%}")


# ── Local training function ───────────────────────────────────────────────────

def local_train(global_params, train_dl, y_train, local_epochs, proximal_mu=0.0):
    pos_weight = compute_class_weight(y_train)
    model      = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, global_params)
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    optimizer  = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
    for _ in range(local_epochs):
        train_one_epoch(model, train_dl, optimizer, criterion, DEVICE, proximal_mu)
    return get_params_as_numpy(model)


def fedavg_aggregate(updates, n_samples):
    total = sum(n_samples)
    return [sum((n / total) * updates[c][i]
                for c, n in enumerate(n_samples))
            for i in range(len(updates[0]))]


# ── Run FedAvg (A + C only) ───────────────────────────────────────────────────
print(f"\n[2/5] Running FedAvg (A+C only) for {FL_NUM_ROUNDS} rounds...")
init_model    = DiabetesNet()
global_params = get_params_as_numpy(init_model)
t0 = time.time()

for rnd in range(1, FL_NUM_ROUNDS + 1):
    client_updates = []
    for node_idx, (tr_dl, _, y_tr, _) in enumerate(loaders):
        updated = local_train([p.copy() for p in global_params],
                              tr_dl, y_tr, NODE_LOCAL_EPOCHS[node_idx])
        client_updates.append(updated)
    global_params = fedavg_aggregate(client_updates, n_samples)
    if rnd % 10 == 0 or rnd == 1:
        print(f"  Round {rnd:3d}/{FL_NUM_ROUNDS}  elapsed={time.time()-t0:.0f}s")

ablation_model = DiabetesNet()
set_params_from_numpy(ablation_model, global_params)
ablation_model.to(DEVICE)
torch.save(ablation_model.state_dict(), os.path.join(MODELS_DIR, 'fedavg_ablation_ac.pt'))
print(f"  Runtime: {time.time()-t0:.0f}s")


# ── Internal evaluation ───────────────────────────────────────────────────────
print("\n[3/5] Internal evaluation (NHANES held-out test)...")
df_nhanes = pd.read_csv(CENTRALISED_PATH)
X_nh = df_nhanes[FEATURE_COLS].values.astype(np.float32)
y_nh = df_nhanes[TARGET_COL].values.astype(np.float32)
_, X_test_nh, _, y_test_nh = train_test_split(
    X_nh, y_nh, test_size=TEST_SIZE, stratify=y_nh, random_state=RANDOM_SEED
)
global_scaler = joblib.load(GLOBAL_SCALER_PATH)
X_test_sc = global_scaler.transform(X_test_nh).astype(np.float32)

ablation_model.eval()
with torch.no_grad():
    y_prob_int = torch.sigmoid(
        ablation_model(torch.FloatTensor(X_test_sc).to(DEVICE))
    ).cpu().numpy()

m_int = compute_metrics(y_test_nh, y_prob_int)
ci_lo, ci_hi = bootstrap_auc_ci(y_test_nh, y_prob_int)
m_int['ci_low']  = ci_lo
m_int['ci_high'] = ci_hi
print(f"  Internal AUC: {m_int['auc']:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]")


# ── BRFSS external evaluation ─────────────────────────────────────────────────
print("\n[4/5] BRFSS external evaluation...")
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
X_brfss_sc = global_scaler.transform(X_brfss).astype(np.float32)
print(f"  BRFSS n={len(df_brfss):,}  prevalence={y_brfss.mean():.1%}")

ablation_model.eval()
chunk = 50_000
probs_ext = []
with torch.no_grad():
    for i in range(0, len(X_brfss_sc), chunk):
        X_c = torch.FloatTensor(X_brfss_sc[i:i+chunk]).to(DEVICE)
        probs_ext.append(torch.sigmoid(ablation_model(X_c)).cpu().numpy())
y_prob_ext = np.concatenate(probs_ext)

m_ext = compute_metrics(y_brfss, y_prob_ext)
auc_ext, ci_lo_ext, ci_hi_ext = delong_ci_fast(y_brfss, y_prob_ext)
m_ext['ci_low']  = ci_lo_ext
m_ext['ci_high'] = ci_hi_ext

subgroup_auc = {}
for label, (lo, hi) in AGE_GROUPS.items():
    mask = (df_brfss['RIDAGEYR'] >= lo) & (df_brfss['RIDAGEYR'] <= hi)
    if mask.sum() >= 100 and y_brfss[mask].sum() >= 20:
        key = f'age_{label.replace("-","_").replace("+","_plus")}'
        subgroup_auc[key] = float(roc_auc_score(y_brfss[mask], y_prob_ext[mask]))

elderly_key = 'age_60_plus'
young_key   = 'age_18_39'
elderly_gap = None
if elderly_key in subgroup_auc and young_key in subgroup_auc:
    elderly_gap = float(subgroup_auc[young_key] - subgroup_auc[elderly_key])

m_ext['subgroup_auc'] = subgroup_auc
m_ext['elderly_gap']  = elderly_gap

print(f"  External AUC: {auc_ext:.3f} [{ci_lo_ext:.3f}–{ci_hi_ext:.3f}]")
print(f"  Elderly gap: {elderly_gap:.3f}" if elderly_gap else "  Elderly gap: N/A")


# ── Save results ──────────────────────────────────────────────────────────────
print("\n[5/5] Saving results...")
full_fl_ext_auc   = 0.757
full_fl_eld_gap   = 0.054
gap_increase_pct  = float((elderly_gap - full_fl_eld_gap) / full_fl_eld_gap * 100) \
                    if elderly_gap is not None else None

out = {
    'internal': {
        'auc':         m_int['auc'],
        'ci_low':      m_int['ci_low'],
        'ci_high':     m_int['ci_high'],
        'brier':       m_int['brier'],
        'f1':          m_int['f1'],
        'sensitivity': m_int['sensitivity'],
        'specificity': m_int['specificity'],
    },
    'external': {
        'auc':         m_ext['auc'],
        'ci_low':      m_ext['ci_low'],
        'ci_high':     m_ext['ci_high'],
        'brier':       m_ext['brier'],
        'f1':          m_ext['f1'],
        'sensitivity': m_ext['sensitivity'],
        'specificity': m_ext['specificity'],
        'subgroup_auc': subgroup_auc,
        'elderly_gap':  elderly_gap,
    },
    'comparison': {
        'full_fl_elderly_gap':    full_fl_eld_gap,
        'ablation_elderly_gap':   elderly_gap,
        'gap_increase_pct':       gap_increase_pct,
        'full_fl_external_auc':   full_fl_ext_auc,
        'ablation_external_auc':  m_ext['auc'],
    },
}
out_path = os.path.join(RESULTS_DIR, 'exp2_node_b_ablation.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"  Ablation (A+C) External AUC: {m_ext['auc']:.3f}  "
      f"(Full FL: {full_fl_ext_auc})")
print(f"  Ablation Elderly Gap: {elderly_gap:.3f}  "
      f"(Full FL: {full_fl_eld_gap})")
print(f"  Gap increase: {gap_increase_pct:+.1f}%" if gap_increase_pct else "")
print(f"{'='*65}")
print(f"\n✓ DONE: exp2_node_b_ablation.py — results saved to {out_path}")
