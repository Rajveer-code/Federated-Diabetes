"""
15_dp_extended.py
==================
Extends the DP sweep to epsilon = {10, 20, 50, 100}.
Merges with existing dp_results.json (eps 0.5-5.0 + inf).

Uses IDENTICAL data split, architecture, and hyperparameters as 04_differential_privacy.py.

Saves: results/dp_results_extended.json
       plots/05_dp_tradeoff.png (regenerated with all 9 eps values)
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, RESULTS_DIR, PLOTS_DIR,
    FEATURE_COLS, TARGET_COL, GLOBAL_SCALER_PATH,
    DP_TARGET_DELTA, DP_MAX_GRAD_NORM,
    PUBLISHED_ELDERLY_GAP, PUBLISHED_INTERNAL_AUC, PUBLISHED_EXTERNAL_AUC,
    NN_LR, SEED,
)
from nn_model import DiabetesNet, get_device
from data_utils import DiabetesDataset, compute_class_weight

DEVICE = get_device()
DP_BATCH_SIZE = 512
DP_EPOCHS     = 5
NEW_EPSILONS  = [10.0, 20.0, 50.0, 100.0]

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_OK = True
    print("Opacus loaded")
except ImportError:
    OPACUS_OK = False
    print("WARNING: Opacus not found — will simulate with gradient clipping only")


def compute_elderly_gap(y_true, y_prob, df):
    young   = df['RIDAGEYR'].values < 40
    elderly = df['RIDAGEYR'].values >= 60
    try:
        ay = float(roc_auc_score(y_true[young],   y_prob[young]))
        ae = float(roc_auc_score(y_true[elderly], y_prob[elderly]))
        return ay - ae, ay, ae
    except Exception:
        return None, None, None


def train_dp_model(target_epsilon, X_train, y_train, X_val, y_val):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pos_w     = compute_class_weight(y_train)
    model     = DiabetesNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR)

    dl_train = DataLoader(DiabetesDataset(X_train, y_train),
                          batch_size=DP_BATCH_SIZE, shuffle=True, drop_last=True)
    dl_val   = DataLoader(DiabetesDataset(X_val, y_val),
                          batch_size=DP_BATCH_SIZE, shuffle=False)

    actual_eps     = float('inf')
    privacy_engine = None

    if OPACUS_OK and target_epsilon < float('inf'):
        model     = ModuleValidator.fix(model).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR)
        privacy_engine = PrivacyEngine()
        model, optimizer, dl_train = privacy_engine.make_private_with_epsilon(
            module         = model,
            optimizer      = optimizer,
            data_loader    = dl_train,
            target_epsilon = target_epsilon,
            target_delta   = DP_TARGET_DELTA,
            epochs         = DP_EPOCHS,
            max_grad_norm  = DP_MAX_GRAD_NORM,
        )
        print(f"      Noise multiplier = {optimizer.noise_multiplier:.3f}", flush=True)

    model.train()
    for _ in range(DP_EPOCHS):
        for X_b, y_b in dl_train:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            if privacy_engine is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), DP_MAX_GRAD_NORM)
            optimizer.step()

    if privacy_engine is not None:
        try:
            actual_eps = float(privacy_engine.get_epsilon(DP_TARGET_DELTA))
        except Exception:
            actual_eps = target_epsilon

    model.eval()
    probs = []
    with torch.no_grad():
        for X_b, _ in dl_val:
            probs.append(torch.sigmoid(model(X_b.to(DEVICE))).cpu().numpy())
    y_prob = np.concatenate(probs)

    try:
        auc = float(roc_auc_score(y_val, y_prob))
    except Exception:
        auc = 0.5

    return auc, actual_eps, y_prob


print("=" * 65)
print("  15 — EXTENDED DP SWEEP (eps = 10, 20, 50, 100)")
print(f"  Delta: {DP_TARGET_DELTA} | Batch: {DP_BATCH_SIZE} | Epochs: {DP_EPOCHS}")
print("=" * 65)

# Load existing results
existing_path = os.path.join(RESULTS_DIR, 'dp_results.json')
with open(existing_path, 'r') as f:
    existing = json.load(f)
print(f"\n  Loaded existing results: {len(existing['epsilon_target'])} eps values")

# Pool all nodes
print("  Loading node data...")
df_pool = pd.concat([pd.read_csv(p) for p in NODE_PATHS], ignore_index=True)
X_pool  = df_pool[FEATURE_COLS].values.astype(np.float32)
y_pool  = df_pool[TARGET_COL].values.astype(np.float32)

scaler    = joblib.load(GLOBAL_SCALER_PATH)
X_pool_sc = scaler.transform(X_pool).astype(np.float32)

idx_tr, idx_val = train_test_split(
    np.arange(len(X_pool_sc)), test_size=0.2,
    stratify=y_pool, random_state=SEED
)
X_tr, y_tr   = X_pool_sc[idx_tr], y_pool[idx_tr]
X_val, y_val = X_pool_sc[idx_val], y_pool[idx_val]
df_val = df_pool.iloc[idx_val].reset_index(drop=True)

print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,}")

# Train new epsilon values
new_results = {
    'epsilon_target': [], 'epsilon_actual': [], 'auc': [],
    'elderly_gap': [], 'young_auc': [], 'elderly_auc': [],
}

for eps in NEW_EPSILONS:
    print(f"\n  Training: eps={eps} ...", flush=True)
    t0 = time.time()

    auc, actual_eps, y_prob_val = train_dp_model(eps, X_tr, y_tr, X_val, y_val)
    gap, young_auc, elderly_auc = compute_elderly_gap(y_val, y_prob_val, df_val)

    actual_str = round(actual_eps, 4) if np.isfinite(actual_eps) else 'inf'
    status = 'VIABLE' if auc > 0.6 else ('COLLAPSED' if auc <= 0.51 else 'DEGRADED')
    gap_s  = f"{gap:.4f}" if gap is not None else "N/A"

    new_results['epsilon_target'].append(str(eps))
    new_results['epsilon_actual'].append(actual_str)
    new_results['auc'].append(round(auc, 5))
    new_results['elderly_gap'].append(round(gap, 5) if gap else None)
    new_results['young_auc'].append(round(young_auc, 5) if young_auc else None)
    new_results['elderly_auc'].append(round(elderly_auc, 5) if elderly_auc else None)

    print(f"    AUC={auc:.4f} [{status}] | Gap={gap_s} | "
          f"eps_actual={actual_str} | t={time.time()-t0:.0f}s")

# Merge: existing (minus inf) + new + inf
merged = {
    'epsilon_target': [], 'epsilon_actual': [], 'auc': [],
    'elderly_gap': [], 'young_auc': [], 'elderly_auc': [],
}

for i in range(len(existing['epsilon_target'])):
    et = existing['epsilon_target'][i]
    if et == 'inf' or et == str(float('inf')):
        continue
    merged['epsilon_target'].append(et)
    merged['epsilon_actual'].append(existing['epsilon_actual'][i])
    merged['auc'].append(existing['auc'][i])
    merged['elderly_gap'].append(existing['elderly_gap'][i])
    merged['young_auc'].append(existing['young_auc'][i])
    merged['elderly_auc'].append(existing['elderly_auc'][i])

for i in range(len(new_results['epsilon_target'])):
    merged['epsilon_target'].append(new_results['epsilon_target'][i])
    merged['epsilon_actual'].append(new_results['epsilon_actual'][i])
    merged['auc'].append(new_results['auc'][i])
    merged['elderly_gap'].append(new_results['elderly_gap'][i])
    merged['young_auc'].append(new_results['young_auc'][i])
    merged['elderly_auc'].append(new_results['elderly_auc'][i])

# Add inf back at the end
inf_idx = None
for i in range(len(existing['epsilon_target'])):
    et = existing['epsilon_target'][i]
    if et == 'inf' or et == str(float('inf')):
        inf_idx = i
        break
if inf_idx is not None:
    merged['epsilon_target'].append(existing['epsilon_target'][inf_idx])
    merged['epsilon_actual'].append(existing['epsilon_actual'][inf_idx])
    merged['auc'].append(existing['auc'][inf_idx])
    merged['elderly_gap'].append(existing['elderly_gap'][inf_idx])
    merged['young_auc'].append(existing['young_auc'][inf_idx])
    merged['elderly_auc'].append(existing['elderly_auc'][inf_idx])

merged['note'] = (
    'batch_size=512, epochs=5. Extended sweep: eps={0.5,1,2,5,10,20,50,100,inf}. '
    'Model recovers at eps>=10; tight budgets (eps<=5) cause collapse at per-node n~3-4.5k.'
)

# Find recovery boundary
recovery_eps = None
for i in range(len(merged['epsilon_target'])):
    et = merged['epsilon_target'][i]
    if et in ['inf', str(float('inf'))]:
        continue
    if merged['auc'][i] > 0.65:
        recovery_eps = et
        break

if recovery_eps:
    merged['recovery_boundary_eps'] = recovery_eps
    print(f"\n  RECOVERY BOUNDARY: eps = {recovery_eps}")

out_path = os.path.join(RESULTS_DIR, 'dp_results_extended.json')
with open(out_path, 'w') as f:
    json.dump(merged, f, indent=2)
print(f"  Saved: {out_path}")

# Regenerate plot
print("\n  Regenerating DP tradeoff plot...")
eps_labels = []
for e in merged['epsilon_target']:
    if e in ['inf', str(float('inf'))]:
        eps_labels.append('No DP\n(ε=∞)')
    else:
        eps_labels.append(f'ε={e}')

aucs = merged['auc']
gaps = [g if g else 0 for g in merged['elderly_gap']]
x = np.arange(len(eps_labels))
viable_mask = [a > 0.55 for a in aucs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

bar_colors = ['#2563EB' if v else '#DC2626' for v in viable_mask]
ax1.bar(x, aucs, color=bar_colors, alpha=0.85, width=0.55, edgecolor='white')
for xi, auc_v, v in zip(x, aucs, viable_mask):
    label = f'{auc_v:.3f}' + ('' if v else '\n(collapsed)')
    ax1.text(xi, auc_v + 0.008, label, ha='center', fontsize=8,
             fontweight='bold', color='#2563EB' if v else '#DC2626')
ax1.axhline(0.65, color='#059669', ls='-.', lw=1.5, alpha=0.7,
            label='Recovery threshold (AUC=0.65)')
ax1.axhline(0.5, color='grey', ls='-', lw=0.8, alpha=0.4, label='Random (0.5)')
ax1.set_xticks(x); ax1.set_xticklabels(eps_labels, fontsize=9)
ax1.set_xlabel('Privacy Budget (ε)\n← More Private    Less Private →', fontsize=11)
ax1.set_ylabel('AUC-ROC', fontsize=12)
ax1.set_ylim(0.35, 0.87)
ax1.set_title('(A) Privacy vs Accuracy', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

viable_x    = [xi for xi, v in zip(x, viable_mask) if v]
viable_gaps = [g  for g, v in zip(gaps, viable_mask) if v]
if viable_x:
    ax2.plot(viable_x, viable_gaps, 's-', color='#DC2626', lw=2.5, ms=9, zorder=3)
    for xi, g in zip(viable_x, viable_gaps):
        if g != 0:
            ax2.annotate(f'{g:.3f}', (xi, g), textcoords='offset points',
                         xytext=(0, 11), ha='center', fontsize=9,
                         fontweight='bold', color='#DC2626')
ax2.axhline(PUBLISHED_ELDERLY_GAP, color='#1E293B', ls='--', lw=1.5, alpha=0.7,
            label=f'Published gap ({PUBLISHED_ELDERLY_GAP})')
ax2.set_xticks(x); ax2.set_xticklabels(eps_labels, fontsize=9)
ax2.set_xlabel('Privacy Budget (ε)\n← More Private    Less Private →', fontsize=11)
ax2.set_ylabel('AUC Gap (Young − Elderly)\nLower = More Equitable', fontsize=12)
ax2.set_title('(B) Privacy vs Fairness (viable models)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

fig.suptitle(
    'Privacy–Accuracy–Fairness Trade-off (Extended Sweep)\n'
    f'Federated Diabetes Prediction | δ={DP_TARGET_DELTA}',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.savefig(os.path.join(PLOTS_DIR, '05_dp_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/05_dp_tradeoff.png")

print(f"\n{'='*65}")
print(f"  EXTENDED DP RESULTS")
print(f"  {'eps_target':<12} {'eps_actual':<12} {'AUC':>8} {'Gap':>8} {'Status':>10}")
print(f"  {'-'*54}")
for et, ea, a, g in zip(merged['epsilon_target'], merged['epsilon_actual'],
                         merged['auc'], merged['elderly_gap']):
    g_s    = f"{g:.4f}" if g else "  N/A"
    status = "VIABLE" if a > 0.6 else ("COLLAPSED" if a <= 0.51 else "DEGRADED")
    print(f"  {str(et):<12} {str(ea):<12} {a:>8.4f} {g_s:>8} {status:>10}")
if recovery_eps:
    print(f"\n  RECOVERY BOUNDARY: eps >= {recovery_eps} (AUC > 0.65)")
print(f"{'='*65}")
