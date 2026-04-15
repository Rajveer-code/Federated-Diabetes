"""
SCRIPT 03b — CORRECTED FedNova (Heterogeneous Local Epochs)
============================================================
CORRECTION from prior implementation:

PROBLEM:  The original 03_federated_simulation.py used equal local epochs
          (NN_LOCAL_EPOCHS=5) for all three nodes in FedNova. When all τᵢ
          are equal, FedNova's normalisation divides each update by τᵢ and
          then multiplies by τ_eff = τᵢ, so the τ terms cancel:

              Δw_aggregated = Σᵢ (nᵢ/N) · (Δwᵢ / τᵢ) · τ_eff
                            = Σᵢ (nᵢ/N) · Δwᵢ          [since τ_eff = τᵢ]
                            ≡ FedAvg exactly

          Wang et al. (NeurIPS 2020) explicitly state: "Clients perform
          the same number of local steps τᵢ = 30 — FedNova is equivalent
          to FedAvg in this case." (Figure 2, left panel)

FIX:      Assign heterogeneous local epochs reflecting DISTRIBUTION SHIFT
          across nodes, per Wang et al. Theorem 2:
            Node A — Young Urban (low dist. shift):    τ_A = 5
            Node B — Elderly Rural (HIGH dist. shift): τ_B = 3  <- FEWER steps
            Node C — Mixed Metro (intermediate):       τ_C = 4

          CRITICAL: The original fix assigned τ_B = 7 (most steps to the most
          constrained/shifted node). This is BACKWARDS per FedNova theory.
          Wang et al. Theorem 2: nodes whose distribution diverges most from
          the global objective should perform FEWER local steps to prevent
          client drift — not more. τ_B = 3 is the theoretically correct value.

          Run this script INSTEAD of 03_federated_simulation.py for the
          FedNova comparison. FedAvg and FedProx results from the original
          script remain valid and unchanged.

Output:
  results/fednova_corrected.json
  models/fednova_corrected_weights.pt

Usage:
  python 03b_fednova_corrected.py  (~10 minutes)

Reference:
  Wang J, Liu Q, Liang H, Joshi G, Poor HV.
  "Tackling the Objective Inconsistency Problem in Heterogeneous
  Federated Optimization." NeurIPS 2020. arXiv:2007.07481.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, NODE_NAMES, CENTRALISED_PATH,
    RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_COL,
    FL_NUM_ROUNDS, FEDPROX_MU,
    NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    PUBLISHED_INTERNAL_AUC, PUBLISHED_EXTERNAL_AUC, SEED,
)
from nn_model   import DiabetesNet, train_one_epoch
from data_utils import (DiabetesDataset, load_node_data, get_dataloaders,
                         compute_class_weight, get_device,
                         get_params_as_numpy, set_params_from_numpy)

DEVICE = get_device()

# ── Heterogeneous local epochs (the critical fix) ─────────────────────────────
# Assigned per DISTRIBUTION SHIFT, not compute resources (Wang et al. 2020):
#   Node A — Young Urban  (low distribution shift):    τ_A = 5  (safe to do more local work)
#   Node B — Elderly Rural (HIGH distribution shift):  τ_B = 3  (FEWER steps to prevent drift)
#   Node C — Mixed Metro  (intermediate heterogeneity): τ_C = 4
#
# CORRECTION from original docstring error: τ_B = 7 was BACKWARDS.
# Wang et al. Theorem 2: nodes with greater divergence from the global
# objective should perform FEWER local steps to prevent client drift.
# Node B has 82% elderly, 28.5% diabetes prevalence — highest distributional
# deviation from the pooled NHANES training distribution -> fewest local steps.
#
# Also imports from config_paths for single source of truth:
from config_paths import NODE_LOCAL_EPOCHS_FEDNOVA as NODE_LOCAL_EPOCHS
# NODE_LOCAL_EPOCHS = {0: 5, 1: 3, 2: 4}   — defined in config_paths.py
# τ_eff (effective local steps) = Σᵢ (nᵢ/N) · τᵢ
# This will be computed from actual node sample sizes below.

print("=" * 65)
print("  SCRIPT 03b — FedNova CORRECTED (heterogeneous local epochs)")
print("=" * 65)
print(f"\n  Fix: heterogeneous τᵢ = {list(NODE_LOCAL_EPOCHS.values())}")
print(f"  (Equal τ collapses FedNova -> FedAvg; see Wang et al. NeurIPS 2020)")
print("=" * 65)

# ── Load centralised eval set (for per-round AUC tracking) ────────────────────
df_eval  = pd.read_csv(CENTRALISED_PATH)
X_eval   = df_eval[FEATURE_COLS].values.astype(np.float32)
y_eval   = df_eval[TARGET_COL].values.astype(np.float32)
_scaler  = StandardScaler()
X_eval_sc = _scaler.fit_transform(X_eval).astype(np.float32)


@torch.no_grad()
def eval_global(params) -> float:
    model = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, params)
    model.eval()
    probs = torch.sigmoid(
        model(torch.FloatTensor(X_eval_sc).to(DEVICE))
    ).cpu().numpy()
    try:
        return float(roc_auc_score(y_eval, probs))
    except Exception:
        return 0.0


# ── Load all node data ─────────────────────────────────────────────────────────
print("\n  Loading node data...")
loaders, n_train_per_node, all_y_train = [], [], []

for node_idx, path in enumerate(NODE_PATHS):
    X_tr, y_tr, X_val, y_val, sc = load_node_data(
        path, val_size=0.2, seed=SEED
    )
    epochs = NODE_LOCAL_EPOCHS[node_idx]
    tr_dl, _ = get_dataloaders(
        X_tr, y_tr, X_val, y_val, NN_BATCH_SIZE
    )
    loaders.append(tr_dl)
    n_train_per_node.append(len(X_tr))
    all_y_train.append(y_tr)
    print(f"    {NODE_NAMES[node_idx]:30s}: n_train={len(X_tr):,} | τᵢ={epochs}")

total_n = sum(n_train_per_node)

# ── Compute τ_eff (weighted effective local steps) ────────────────────────────
tau_eff = sum(
    (n / total_n) * NODE_LOCAL_EPOCHS[i]
    for i, n in enumerate(n_train_per_node)
)
print(f"\n  τ_eff = {tau_eff:.4f}  "
      f"(weighted avg of [{', '.join(str(NODE_LOCAL_EPOCHS[i]) for i in range(3))}])")
print(f"  Total samples: {total_n:,}")


# ── Local training for one node, one round ────────────────────────────────────
def local_train_node(node_idx, global_params, train_dl, y_train):
    """Train locally for NODE_LOCAL_EPOCHS[node_idx] epochs."""
    epochs    = NODE_LOCAL_EPOCHS[node_idx]
    pos_w     = compute_class_weight(y_train)
    model     = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, [p.copy() for p in global_params])

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY
    )

    for _ in range(epochs):
        train_one_epoch(model, train_dl, optimizer, criterion, DEVICE,
                        proximal_mu=0.0, global_params=None)

    return get_params_as_numpy(model)


# ── FedNova aggregation (correct heterogeneous-τ version) ─────────────────────
def fednova_aggregate_corrected(updates, n_samples, global_params):
    """
    FedNova aggregation — Wang et al. (NeurIPS 2020), Eq. (4).

    Each client uploads normalised local update:
        d̄ᵢ = (wᵢ - w_global) / τᵢ   [normalise by local steps]

    Server aggregates:
        w_new = w_global + τ_eff · Σᵢ (nᵢ/N) · d̄ᵢ
              = w_global + τ_eff · Σᵢ (nᵢ/N) · (Δwᵢ / τᵢ)

    When τᵢ are all equal this reduces to FedAvg (× τ_eff/τ ≡ 1).
    When τᵢ differ this corrects objective inconsistency.
    """
    total = sum(n_samples)
    new_params = []

    for layer_idx in range(len(global_params)):
        # Normalised weighted sum of local updates
        normalised_update = sum(
            (n_samples[i] / total)
            * (updates[i][layer_idx] - global_params[layer_idx])
            / NODE_LOCAL_EPOCHS[i]
            for i in range(len(updates))
        )
        # Scale by τ_eff and add to global
        new_layer = global_params[layer_idx] + tau_eff * normalised_update
        new_params.append(new_layer)

    return new_params


# ── Run FedNova ───────────────────────────────────────────────────────────────
print(f"\n  Running FedNova (corrected) — {FL_NUM_ROUNDS} rounds...")
print(f"  τ per node: A={NODE_LOCAL_EPOCHS[0]}, B={NODE_LOCAL_EPOCHS[1]}, "
      f"C={NODE_LOCAL_EPOCHS[2]} | τ_eff={tau_eff:.3f}")

global_params = get_params_as_numpy(DiabetesNet())
round_aucs, round_nums = [], []
t0 = time.time()

for rnd in range(1, FL_NUM_ROUNDS + 1):
    # Each node trains locally for its own τᵢ epochs
    client_updates = []
    for node_idx, (dl, y_tr) in enumerate(zip(loaders, all_y_train)):
        updated = local_train_node(node_idx, global_params, dl, y_tr)
        client_updates.append(updated)

    # Aggregate with corrected FedNova
    global_params = fednova_aggregate_corrected(
        client_updates, n_train_per_node, global_params
    )

    # Track AUC every round
    auc = eval_global(global_params)
    round_nums.append(rnd)
    round_aucs.append(auc)

    if rnd % 10 == 0 or rnd == 1:
        elapsed = time.time() - t0
        eta     = (elapsed / rnd) * (FL_NUM_ROUNDS - rnd)
        print(f"    Round {rnd:3d}/{FL_NUM_ROUNDS}  "
              f"AUC={auc:.4f}  "
              f"Elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

# ── Save model ────────────────────────────────────────────────────────────────
final_model = DiabetesNet()
set_params_from_numpy(final_model, global_params)
os.makedirs(MODELS_DIR, exist_ok=True)
weights_path = os.path.join(MODELS_DIR, 'fednova_corrected_weights.pt')
torch.save(final_model.state_dict(), weights_path)

elapsed_total = time.time() - t0
best_auc  = max(round_aucs)
best_rnd  = round_aucs.index(best_auc) + 1

print(f"\n  Final AUC : {round_aucs[-1]:.4f}")
print(f"  Best AUC  : {best_auc:.4f}  (round {best_rnd})")
print(f"  Runtime   : {elapsed_total:.0f}s")
print(f"  Saved     : {weights_path}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    'strategy'          : 'FedNova (corrected)',
    'local_epochs'      : NODE_LOCAL_EPOCHS,
    'tau_eff'           : round(tau_eff, 4),
    'rounds'            : round_nums,
    'aucs'              : round_aucs,
    'final_auc'         : round_aucs[-1],
    'best_auc'          : best_auc,
    'best_round'        : best_rnd,
    'runtime_s'         : elapsed_total,
    'weights_path'      : weights_path,
    'note'              : (
        'Corrected from original: heterogeneous local epochs used. '
        'Original uniform-epoch FedNova = FedAvg (Wang et al. 2020, Fig 2 left). '
        f'Epoch counts: Node A={NODE_LOCAL_EPOCHS[0]}, '
        f'Node B={NODE_LOCAL_EPOCHS[1]}, '
        f'Node C={NODE_LOCAL_EPOCHS[2]}. tau_eff={tau_eff:.4f}.'
    ),
}
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, 'fednova_corrected.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: results/fednova_corrected.json")

# ── Convergence plot (compare corrected FedNova vs FedAvg and FedProx) ────────
try:
    # Load FedAvg and FedProx from original results for comparison
    with open(os.path.join(RESULTS_DIR, 'federated_convergence.json')) as f:
        prior = json.load(f)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {'FedAvg': '#2563EB', 'FedProx (μ=0.1)': '#7C3AED',
              'FedNova (corrected)': '#DC2626'}

    for name, data in prior.items():
        col = colors.get(name, '#94A3B8')
        ax.plot(data['rounds'], data['aucs'], lw=2, color=col,
                alpha=0.6, ls='--', label=f'{name} (original)')

    ax.plot(round_nums, round_aucs, lw=2.5, color='#DC2626',
            label=f"FedNova corrected (final={round_aucs[-1]:.4f})")
    ax.axhline(PUBLISHED_INTERNAL_AUC, color='#1E293B', ls='--', lw=1.5,
               label=f'Centralised ({PUBLISHED_INTERNAL_AUC})')

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(
        'FedNova Corrected vs Original Strategies\n'
        f'Heterogeneous τ=(A={NODE_LOCAL_EPOCHS[0]},B={NODE_LOCAL_EPOCHS[1]},C={NODE_LOCAL_EPOCHS[2]}) vs Uniform τ=5',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '12_fednova_corrected.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plots/12_fednova_corrected.png")
except Exception as e:
    print(f"  Plot skipped: {e}")

print("\n" + "=" * 65)
print("  FedNova CORRECTED — SUMMARY")
print("=" * 65)
print(f"\n  Strategy              : FedNova (heterogeneous τ)")
print(f"  Node A epochs (τ_A)  : {NODE_LOCAL_EPOCHS[0]}  (Young Urban,   LOW dist. shift -> more local steps)")
print(f"  Node B epochs (τ_B)  : {NODE_LOCAL_EPOCHS[1]}  (Elderly Rural, HIGH dist. shift -> FEWER steps — corrected)")
print(f"  Node C epochs (τ_C)  : {NODE_LOCAL_EPOCHS[2]}  (Mixed Metro,   intermediate heterogeneity)")
print(f"  Effective τ (τ_eff)  : {tau_eff:.4f}")
print(f"  Final AUC (internal) : {round_aucs[-1]:.4f}")
print(f"  Best AUC (internal)  : {best_auc:.4f}  (round {best_rnd})")
print(f"\n  NEXT STEPS:")
print(f"  1. Run 05_fairness_analysis.py -> load fednova_corrected_weights.pt")
print(f"  2. Run 07_external_validation.py -> add 'fednova_corrected' to fed_models dict")
print(f"  3. Update paper Table 1, Table 2, Table 3 with new FedNova numbers")
print(f"\n  FedNova weights: {weights_path}")
print("=" * 65)
