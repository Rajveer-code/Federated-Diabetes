"""
exp1_central_nn_baseline.py — Centralised DiabetesNet (no federation)
=======================================================================
Trains DiabetesNet on the full NHANES training split (no federation).
Separates the effect of federated learning from the neural-network architecture.

Produces:
  results/exp1_central_nn.json
  models/central_nn.pt
"""

import os, sys, json, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score,
                              roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    CENTRALISED_PATH, RESULTS_DIR, MODELS_DIR, GLOBAL_SCALER_PATH,
    FEATURE_COLS, TARGET_COL, AGE_GROUPS,
    NN_HIDDEN_DIMS, NN_DROPOUT, NN_LR, NN_WEIGHT_DECAY,
    N_BOOTSTRAP, CI_ALPHA, RANDOM_SEED, TEST_SIZE,
)
from nn_model import DiabetesNet, get_device
from data_utils import DiabetesDataset, compute_class_weight

BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
BRFSS_COL_MAP = {
    'Age': 'RIDAGEYR', 'Gender': 'RIAGENDR', 'Race_Ethnicity': 'RIDRETH3',
    'BMI': 'BMXBMI', 'Smoking_Status': 'SMOKING', 'Physical_Activity': 'PHYS_ACTIVITY',
    'History_Heart_Attack': 'HEART_ATTACK', 'History_Stroke': 'STROKE',
    'Diabetes_Outcome': 'DIABETES',
}

CENTRAL_EPOCHS = 250
BATCH_SIZE     = 64
DEVICE         = get_device()

# ── Seeds ────────────────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 65)
print("  EXP 1 — CENTRALISED DiabetesNet BASELINE")
print(f"  Epochs={CENTRAL_EPOCHS} | Batch={BATCH_SIZE} | lr={NN_LR}")
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
    """O(n log n) DeLong CI via searchsorted — safe for n=1.28M."""
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

    auc = float(V10.mean())
    se  = float(np.sqrt(np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg))
    z   = stats.norm.ppf(1 - alpha / 2)
    return auc, float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def compute_metrics(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'auc'        : float(roc_auc_score(y_true, y_prob)),
        'brier'      : float(brier_score_loss(y_true, y_prob)),
        'f1'         : float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'threshold'  : float(threshold),
    }


# ── Load & split NHANES ───────────────────────────────────────────────────────
print("\n[1/6] Loading NHANES data...")
df = pd.read_csv(CENTRALISED_PATH)
X  = df[FEATURE_COLS].values.astype(np.float32)
y  = df[TARGET_COL].values.astype(np.float32)
print(f"  Total n={len(df):,}  prevalence={y.mean():.1%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
print(f"  Train={len(X_train):,}  Test={len(X_test):,}")

scaler = joblib.load(GLOBAL_SCALER_PATH)
X_train_sc = scaler.transform(X_train).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)
print(f"  Global scaler loaded from {GLOBAL_SCALER_PATH}")


# ── Build DataLoaders ─────────────────────────────────────────────────────────
from torch.utils.data import DataLoader
train_dl = DataLoader(DiabetesDataset(X_train_sc, y_train),
                      batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


# ── Train centralised DiabetesNet ─────────────────────────────────────────────
print(f"\n[2/6] Training centralised DiabetesNet for {CENTRAL_EPOCHS} epochs...")
model     = DiabetesNet(input_dim=len(FEATURE_COLS),
                        hidden_dims=NN_HIDDEN_DIMS,
                        dropout=NN_DROPOUT).to(DEVICE)
pos_wt    = compute_class_weight(y_train)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_wt], device=DEVICE)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)

t0 = time.time()
for epoch in range(1, CENTRAL_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_b, y_b in train_dl:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 50 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{CENTRAL_EPOCHS}  loss={epoch_loss/len(train_dl):.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'central_nn.pt'))
print(f"  Model saved → models/central_nn.pt")


# ── Internal evaluation (NHANES held-out) ─────────────────────────────────────
print("\n[3/6] Internal evaluation (NHANES held-out test)...")
model.eval()
with torch.no_grad():
    X_ts = torch.FloatTensor(X_test_sc).to(DEVICE)
    y_prob_int = torch.sigmoid(model(X_ts)).cpu().numpy()

thr_int   = youden_threshold(y_test, y_prob_int)
m_int     = compute_metrics(y_test, y_prob_int, thr_int)
ci_lo, ci_hi = bootstrap_auc_ci(y_test, y_prob_int)
m_int['ci_low']  = ci_lo
m_int['ci_high'] = ci_hi
print(f"  Internal AUC: {m_int['auc']:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]  "
      f"Brier={m_int['brier']:.3f}  F1={m_int['f1']:.3f}")


# ── BRFSS external evaluation ─────────────────────────────────────────────────
print("\n[4/6] Loading BRFSS external data...")
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
X_brfss   = df_brfss[FEATURE_COLS].values.astype(np.float32)
y_brfss   = df_brfss['DIABETES'].values.astype(np.float32)
X_brfss_sc = scaler.transform(X_brfss).astype(np.float32)
print(f"  BRFSS n={len(df_brfss):,}  prevalence={y_brfss.mean():.1%}")

print("\n[5/6] BRFSS inference (batched)...")
model.eval()
chunk = 50_000
probs_ext = []
with torch.no_grad():
    for i in range(0, len(X_brfss_sc), chunk):
        X_c = torch.FloatTensor(X_brfss_sc[i:i+chunk]).to(DEVICE)
        probs_ext.append(torch.sigmoid(model(X_c)).cpu().numpy())
y_prob_ext = np.concatenate(probs_ext)

thr_ext = youden_threshold(y_brfss, y_prob_ext)
m_ext   = compute_metrics(y_brfss, y_prob_ext, thr_ext)
auc_ext, ci_lo_ext, ci_hi_ext = delong_ci_fast(y_brfss, y_prob_ext)
m_ext['ci_low']  = ci_lo_ext
m_ext['ci_high'] = ci_hi_ext

# Subgroup AUC on BRFSS
subgroup_auc = {}
for label, (lo, hi) in AGE_GROUPS.items():
    mask = (df_brfss['RIDAGEYR'] >= lo) & (df_brfss['RIDAGEYR'] <= hi)
    if mask.sum() >= 100 and y_brfss[mask].sum() >= 20:
        subgroup_auc[f'age_{label.replace("-","_").replace("+","_plus")}'] = \
            float(roc_auc_score(y_brfss[mask], y_prob_ext[mask]))

elderly_key = 'age_60_plus'
young_key   = 'age_18_39'
if elderly_key in subgroup_auc and young_key in subgroup_auc:
    elderly_gap = float(subgroup_auc[young_key] - subgroup_auc[elderly_key])
else:
    elderly_gap = None

m_ext['subgroup_auc'] = subgroup_auc
m_ext['elderly_gap']  = elderly_gap

delta_int_to_ext = float(m_int['auc'] - m_ext['auc'])

print(f"  External AUC: {auc_ext:.3f} [{ci_lo_ext:.3f}–{ci_hi_ext:.3f}]  "
      f"Brier={m_ext['brier']:.3f}")
print(f"  Elderly gap: {elderly_gap:.3f}" if elderly_gap else "  Elderly gap: N/A")
print(f"  Delta Int→Ext: {delta_int_to_ext:+.3f}")


# ── Save results ──────────────────────────────────────────────────────────────
print("\n[6/6] Saving results...")
out = {
    'internal': {
        'auc':         m_int['auc'],
        'ci_low':      m_int['ci_low'],
        'ci_high':     m_int['ci_high'],
        'brier':       m_int['brier'],
        'f1':          m_int['f1'],
        'sensitivity': m_int['sensitivity'],
        'specificity': m_int['specificity'],
        'threshold':   m_int['threshold'],
    },
    'external': {
        'auc':         m_ext['auc'],
        'ci_low':      m_ext['ci_low'],
        'ci_high':     m_ext['ci_high'],
        'brier':       m_ext['brier'],
        'f1':          m_ext['f1'],
        'sensitivity': m_ext['sensitivity'],
        'specificity': m_ext['specificity'],
        'threshold':   m_ext['threshold'],
        'subgroup_auc': subgroup_auc,
        'elderly_gap':  elderly_gap,
    },
    'delta_int_to_ext': delta_int_to_ext,
}
out_path = os.path.join(RESULTS_DIR, 'exp1_central_nn.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"  Centralised NN — Internal AUC: {m_int['auc']:.3f} [{m_int['ci_low']:.3f}–{m_int['ci_high']:.3f}]")
print(f"  Centralised NN — External AUC: {auc_ext:.3f} [{ci_lo_ext:.3f}–{ci_hi_ext:.3f}]")
print(f"  Delta Int→Ext: {delta_int_to_ext:+.3f}")
print(f"{'='*65}")
print(f"\n✓ DONE: exp1_central_nn_baseline.py — results saved to {out_path}")
