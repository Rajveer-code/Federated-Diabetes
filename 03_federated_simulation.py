"""
SCRIPT 03 — FEDERATED SIMULATION
====================================
Runs 50-round federated training across 3 hospital nodes.
Compares three aggregation strategies:
  (A) FedAvg   — McMahan et al. 2017
  (B) FedProx  — Li et al. 2020  (handles non-IID data)
  (C) FedNova  — Wang et al. 2020 (normalised local updates)

Produces:
  results/federated_convergence.json
  plots/03_fl_convergence.png
  plots/04_fl_strategy_comparison.png
  models/fedavg_weights.pt
  models/fedprox_weights.pt
  models/fednova_weights.pt

Usage:
  cd D:\Projects\diabetes_prediction_project\federated
  python 03_federated_simulation.py
  Runtime: ~20-40 minutes on RTX 4060
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, NODE_NAMES, CENTRALISED_PATH,
    RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_COL,
    FL_NUM_ROUNDS, FL_NUM_CLIENTS, FEDPROX_MU,
    NN_LOCAL_EPOCHS, NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    PUBLISHED_INTERNAL_AUC, PUBLISHED_EXTERNAL_AUC, SEED,
)
from nn_model   import DiabetesNet, train_one_epoch
from data_utils import (DiabetesDataset, load_node_data, get_dataloaders,
                         compute_class_weight, get_device,
                         get_params_as_numpy, set_params_from_numpy)

DEVICE = get_device()

# ── Load centralised eval set once (used to track global AUC every round) ─────
df_eval  = pd.read_csv(CENTRALISED_PATH)
X_eval   = df_eval[FEATURE_COLS].values.astype(np.float32)
y_eval   = df_eval[TARGET_COL].values.astype(np.float32)
_scaler  = StandardScaler()
X_eval_sc = _scaler.fit_transform(X_eval).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  EVALUATE GLOBAL MODEL ON CENTRALISED SET
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_global(params) -> float:
    """Quick AUC evaluation of current global weights on centralised set."""
    model = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, params)
    model.eval()
    X_t = torch.FloatTensor(X_eval_sc).to(DEVICE)
    probs = torch.sigmoid(model(X_t)).cpu().numpy()
    try:
        return float(roc_auc_score(y_eval, probs))
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  PER-NODE DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────
def build_node_loaders():
    """Load all 3 nodes, fit scaler on each node independently (real FL)."""
    loaders, n_samples, scalers = [], [], []
    for path in NODE_PATHS:
        X_tr, y_tr, X_val, y_val, sc = load_node_data(path, val_size=0.2, seed=SEED)
        tr_dl, val_dl = get_dataloaders(X_tr, y_tr, X_val, y_val, NN_BATCH_SIZE)
        loaders.append((tr_dl, val_dl, y_tr, y_val))
        n_samples.append(len(X_tr))
        scalers.append(sc)
    return loaders, n_samples, scalers


# ──────────────────────────────────────────────────────────────────────────────
#  FEDERATED AGGREGATION STRATEGIES
# ──────────────────────────────────────────────────────────────────────────────

def fedavg_aggregate(updates, n_samples):
    """
    FedAvg — McMahan et al. 2017.
    Weighted average of client model updates proportional to dataset size.
    """
    total = sum(n_samples)
    agg   = []
    for layer_idx in range(len(updates[0])):
        layer = sum(
            (n / total) * updates[c][layer_idx]
            for c, n in enumerate(n_samples)
        )
        agg.append(layer)
    return agg


def fedprox_aggregate(updates, n_samples):
    """
    FedProx — Li et al. 2020.
    Same aggregation as FedAvg; the proximal term is applied CLIENT-SIDE
    during local training (handled in train_one_epoch with mu>0).
    """
    return fedavg_aggregate(updates, n_samples)


def fednova_aggregate(updates, n_samples, local_steps):
    """
    FedNova — Wang et al. 2020.
    Normalises each client's update by its local gradient step count
    before aggregation, correcting objective inconsistency under non-IID.

    Formula: w_new = w_old + τ_eff * Σ_i (n_i/N) * (Δw_i / τ_i)
    where τ_i = local_steps (same for all clients here),
    τ_eff = harmonic mean of local step counts.
    """
    total = sum(n_samples)
    # τ_eff (effective local steps) — equal local_steps -> simplifies to local_steps
    tau_eff = local_steps

    agg = []
    for layer_idx in range(len(updates[0])):
        layer = sum(
            (n / total) * (updates[c][layer_idx] / local_steps) * tau_eff
            for c, n in enumerate(n_samples)
        )
        agg.append(layer)
    return agg


# ──────────────────────────────────────────────────────────────────────────────
#  LOCAL TRAINING FOR ONE CLIENT ONE ROUND
# ──────────────────────────────────────────────────────────────────────────────

def local_train(global_params, train_dl, y_train, proximal_mu=0.0):
    """
    One client's local training for one FL round.
    Returns updated parameters as list of numpy arrays.
    """
    pos_weight = compute_class_weight(y_train)
    model      = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, global_params)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY
    )

    global_tensors = None
    if proximal_mu > 0.0:
        global_tensors = [p.detach().clone() for p in model.parameters()]

    for _ in range(NN_LOCAL_EPOCHS):
        train_one_epoch(model, train_dl, optimizer, criterion,
                        DEVICE, proximal_mu, global_tensors)

    return get_params_as_numpy(model)


# ──────────────────────────────────────────────────────────────────────────────
#  RUN ONE STRATEGY
# ──────────────────────────────────────────────────────────────────────────────

def run_federated(
    strategy_name: str,
    aggregator,
    loaders,
    n_samples,
    proximal_mu: float = 0.0,
):
    """
    Run full federated training for a given strategy.
    Returns dict with round-by-round AUC history and final weights.
    """
    print(f"\n  {'─'*55}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Rounds: {FL_NUM_ROUNDS}  |  "
          f"Local epochs: {NN_LOCAL_EPOCHS}  |  mu={proximal_mu}")
    print(f"  {'─'*55}")

    # Initialise global model
    init_model    = DiabetesNet()
    global_params = get_params_as_numpy(init_model)

    round_aucs  = []
    round_nums  = []
    t0 = time.time()

    for rnd in range(1, FL_NUM_ROUNDS + 1):
        # ── Each client trains locally ───────────────────────────────────────
        client_updates = []
        for node_idx, (tr_dl, _, y_tr, _) in enumerate(loaders):
            updated = local_train(
                [p.copy() for p in global_params],
                tr_dl, y_tr, proximal_mu,
            )
            client_updates.append(updated)

        # ── Server aggregates ────────────────────────────────────────────────
        if 'nova' in strategy_name.lower():
            global_params = aggregator(
                client_updates, n_samples,
                local_steps=NN_LOCAL_EPOCHS
            )
        else:
            global_params = aggregator(client_updates, n_samples)

        # ── Track centralised AUC every round ───────────────────────────────
        auc = eval_global(global_params)
        round_nums.append(rnd)
        round_aucs.append(auc)

        if rnd % 10 == 0 or rnd == 1:
            elapsed = time.time() - t0
            eta     = (elapsed / rnd) * (FL_NUM_ROUNDS - rnd)
            print(f"    Round {rnd:3d}/{FL_NUM_ROUNDS}  "
                  f"AUC={auc:.4f}  "
                  f"Elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Save final weights
    final_model = DiabetesNet()
    set_params_from_numpy(final_model, global_params)
    save_name = strategy_name.lower().split(' ')[0].replace('(', '').replace(')', '')
    weights_path = os.path.join(MODELS_DIR, f'{save_name}_weights.pt')
    torch.save(final_model.state_dict(), weights_path)

    elapsed = time.time() - t0
    print(f"\n  Final AUC : {round_aucs[-1]:.4f}")
    print(f"  Best AUC  : {max(round_aucs):.4f}  (round {round_aucs.index(max(round_aucs))+1})")
    print(f"  Runtime   : {elapsed:.0f}s")
    print(f"  Saved     : {weights_path}")

    return {
        'strategy'     : strategy_name,
        'rounds'       : round_nums,
        'aucs'         : round_aucs,
        'final_auc'    : round_aucs[-1],
        'best_auc'     : max(round_aucs),
        'best_round'   : int(round_aucs.index(max(round_aucs)) + 1),
        'runtime_s'    : elapsed,
        'weights_path' : weights_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SCRIPT 03 — FEDERATED SIMULATION")
print(f"  Rounds={FL_NUM_ROUNDS} | LocalEpochs={NN_LOCAL_EPOCHS} | "
      f"BatchSize={NN_BATCH_SIZE}")
print("=" * 65)

print("\n  Loading node data...")
loaders, n_samples, _ = build_node_loaders()
for i, (name, n) in enumerate(zip(NODE_NAMES, n_samples)):
    print(f"    {name}: {n} training samples")

all_results = {}

# ── (A) FedAvg ─────────────────────────────────────────────────────────────
all_results['FedAvg'] = run_federated(
    'FedAvg', fedavg_aggregate, loaders, n_samples, proximal_mu=0.0
)

# ── (B) FedProx ───────────────────────────────────────────────────────────
all_results['FedProx'] = run_federated(
    f'FedProx (μ={FEDPROX_MU})', fedprox_aggregate,
    loaders, n_samples, proximal_mu=FEDPROX_MU
)

# ── (C) FedNova ───────────────────────────────────────────────────────────
all_results['FedNova'] = run_federated(
    'FedNova', fednova_aggregate, loaders, n_samples, proximal_mu=0.0
)

# ── Save results ─────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, 'federated_convergence.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved -> results/federated_convergence.json")

# ──────────────────────────────────────────────────────────────────────────────
#  PLOT 1 — CONVERGENCE CURVES
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    'FedAvg' : '#2563EB',
    'FedProx': '#7C3AED',
    'FedNova': '#DC2626',
}

fig, ax = plt.subplots(figsize=(11, 6))
for name, res in all_results.items():
    color = COLORS.get(name.split(' ')[0], 'grey')
    ax.plot(res['rounds'], res['aucs'], lw=2.5, color=color,
            label=f"{name}  (final={res['final_auc']:.4f}, best={res['best_auc']:.4f})")
    # Mark best round
    best_idx = res['aucs'].index(max(res['aucs']))
    ax.scatter(res['rounds'][best_idx], res['aucs'][best_idx],
               s=80, color=color, zorder=5, marker='*')

ax.axhline(PUBLISHED_INTERNAL_AUC, color='#1E293B', ls='--', lw=1.8, alpha=0.7,
           label=f'Centralised internal AUC ({PUBLISHED_INTERNAL_AUC})')
ax.axhline(PUBLISHED_EXTERNAL_AUC, color='#94A3B8', ls=':', lw=1.8, alpha=0.7,
           label=f'Centralised external AUC ({PUBLISHED_EXTERNAL_AUC})')

ax.set_xlabel('Communication Round', fontsize=13)
ax.set_ylabel('AUC-ROC (evaluated on full centralised set)', fontsize=13)
ax.set_title(
    'Federated Learning Convergence — FedAvg vs FedProx vs FedNova\n'
    'NHANES 2015–2020 | 3 Hospital Nodes | No Patient Data Sharing',
    fontsize=13, fontweight='bold'
)
ax.set_xlim(1, FL_NUM_ROUNDS)
ax.set_ylim(0.55, 0.82)
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_fl_convergence.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/03_fl_convergence.png")

# ──────────────────────────────────────────────────────────────────────────────
#  PLOT 2 — FINAL STRATEGY COMPARISON BAR CHART
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
names   = list(all_results.keys())
f_aucs  = [all_results[k]['final_auc'] for k in names]
b_aucs  = [all_results[k]['best_auc']  for k in names]
colors  = [COLORS.get(k.split(' ')[0], 'grey') for k in names]

x = np.arange(len(names))
bars_f = ax.bar(x-0.18, f_aucs, 0.32, label='Final round AUC',
                color=colors, alpha=0.85, edgecolor='white')
bars_b = ax.bar(x+0.18, b_aucs, 0.32, label='Best round AUC',
                color=colors, alpha=0.50, edgecolor=colors, linewidth=1.5)

for bars in [bars_f, bars_b]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.002,
                f'{h:.4f}', ha='center', fontsize=9, fontweight='bold')

ax.axhline(PUBLISHED_INTERNAL_AUC, color='#1E293B', ls='--', lw=1.5, alpha=0.7,
           label=f'Centralised internal ({PUBLISHED_INTERNAL_AUC})')
ax.axhline(PUBLISHED_EXTERNAL_AUC, color='#94A3B8', ls=':', lw=1.5, alpha=0.7,
           label=f'Centralised external ({PUBLISHED_EXTERNAL_AUC})')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=12)
ax.set_ylabel('AUC-ROC', fontsize=13)
ax.set_ylim(0.60, 0.82)
ax.set_title('Aggregation Strategy Comparison — Final & Best AUC',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_fl_strategy_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/04_fl_strategy_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
#  SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FEDERATED RESULTS SUMMARY")
print("=" * 65)
print(f"\n  {'Strategy':<25} {'Final AUC':>10} {'Best AUC':>10} "
      f"{'Best Rnd':>9} {'Runtime':>9}")
print("  " + "─"*60)
for name, res in all_results.items():
    print(f"  {name:<25} {res['final_auc']:>10.4f} "
          f"{res['best_auc']:>10.4f} {res['best_round']:>9d} "
          f"{res['runtime_s']:>8.0f}s")
print("  " + "─"*60)
print(f"  {'Centralised (internal)':<25} {PUBLISHED_INTERNAL_AUC:>10.4f}")
print(f"  {'Centralised (external)':<25} {PUBLISHED_EXTERNAL_AUC:>10.4f}")

print("\n✅  Script 03 complete — run 04_differential_privacy.py next")
print("=" * 65)
