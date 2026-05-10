"""
SCRIPT 08 — SCAFFOLD BASELINE
==============================
SCAFFOLD Option II (Karimireddy et al., ICML 2020, arXiv:2003.00295).

Corrects client drift via per-client control variates c_i and global c.
Unlike FedProx (adds proximal term), SCAFFOLD modifies the gradient
direction itself, which gives tighter convergence under heterogeneous data.

Algorithm (Option II):
  Server init: w_0, c = 0
  Per round:
    Broadcast w_t, c to each client i
    Client i runs K local steps:
      x_{ik+1} = x_ik - eta_l * (grad f_i(x_ik) - c_i + c)
    Client i computes CV update:
      c_i+ = c_i - c + (w_t - x_iK) / (K * eta_l)
      delta_c_i = c_i+ - c_i
    Server aggregates:
      w_{t+1} = w_t + (eta_g/N) * sum_i (x_iK - w_t)
      c+      = c  + (1/N)    * sum_i delta_c_i

INPUTS:
  data/node_{a,b,c}_*.csv    (partitioned node data)
  artefacts/global_nhanes_scaler.joblib

OUTPUTS:
  results/scaffold_results.json
  models/scaffold_weights.pt
  results/pred_scaffold_internal.npy

RUNTIME: ~15-30 minutes on RTX 4060

REFERENCE:
  Karimireddy SP, Kale S, Mohri M, Reddi SJ, Stich SU, Suresh AT.
  "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning."
  ICML 2020. arXiv:2003.00295.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, NODE_NAMES, CENTRALISED_PATH,
    RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_COL, GLOBAL_SCALER_PATH,
    FL_NUM_ROUNDS, NN_LOCAL_EPOCHS, NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    PUBLISHED_INTERNAL_AUC, SEED,
)
from nn_model   import DiabetesNet, get_device
from data_utils import (DiabetesDataset, load_node_data, get_dataloaders,
                         compute_class_weight,
                         get_params_as_numpy, set_params_from_numpy)

DEVICE     = get_device()
ETA_LOCAL  = NN_LR        # client learning rate
ETA_GLOBAL = 1.0          # server learning rate (Option II standard)
K          = NN_LOCAL_EPOCHS  # local steps per round

print("=" * 65)
print("  08_scaffold_baseline.py")
print("  SCAFFOLD Option II — Karimireddy et al. ICML 2020")
print("=" * 65)
print(f"  Device    : {DEVICE}")
print(f"  Rounds    : {FL_NUM_ROUNDS}")
print(f"  Local K   : {K}")
print(f"  eta_local : {ETA_LOCAL}")
print(f"  eta_global: {ETA_GLOBAL}")


# ── Load global scaler + centralised eval set ─────────────────────────────────
_scaler   = joblib.load(GLOBAL_SCALER_PATH)
df_eval   = pd.read_csv(CENTRALISED_PATH)
X_eval    = _scaler.transform(df_eval[FEATURE_COLS].values.astype(np.float32))
y_eval    = df_eval[TARGET_COL].values.astype(np.float32)


# ── Load node data ────────────────────────────────────────────────────────────
print("\n  Loading node data...")
loaders, n_samples, y_trains = [], [], []
use_pin = DEVICE.type == 'cuda'

for path in NODE_PATHS:
    X_tr, y_tr, X_val, y_val, _ = load_node_data(
        path, val_size=0.2, seed=SEED, scaler=_scaler, fit_scaler=False
    )
    tr_dl, _ = get_dataloaders(
        X_tr, y_tr, X_val, y_val, NN_BATCH_SIZE, pin_memory=use_pin
    )
    loaders.append(tr_dl)
    n_samples.append(len(X_tr))
    y_trains.append(y_tr)
    print(f"    {NODE_NAMES[NODE_PATHS.index(path)]:30s}: n_train={len(X_tr):,}")

N_CLIENTS = len(NODE_PATHS)
total_n   = sum(n_samples)


# ── Model helpers ─────────────────────────────────────────────────────────────
def zero_like_params(params):
    return [np.zeros_like(p) for p in params]


@torch.no_grad()
def eval_global(params) -> float:
    model = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, params)
    model.eval()
    X_t   = torch.FloatTensor(X_eval).to(DEVICE)
    probs = torch.sigmoid(model(X_t)).cpu().numpy()
    return float(roc_auc_score(y_eval, probs))


# ── SCAFFOLD client step ──────────────────────────────────────────────────────
def scaffold_client_update(node_idx, global_params, c_i, c_global, train_dl):
    """
    Run K local SCAFFOLD steps for client i.
    Returns (local_params, delta_c_i).
    """
    y_tr  = y_trains[node_idx]
    pos_w = compute_class_weight(y_tr)

    model = DiabetesNet().to(DEVICE)
    set_params_from_numpy(model, [p.copy() for p in global_params])

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=DEVICE)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=ETA_LOCAL)

    # Convert control variates to tensors (one per parameter)
    c_i_t = [torch.tensor(v, device=DEVICE, dtype=torch.float32) for v in c_i]
    c_g_t = [torch.tensor(v, device=DEVICE, dtype=torch.float32) for v in c_global]

    scaler_amp = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    step = 0
    while step < K:
        for X_batch, y_batch in train_dl:
            if step >= K:
                break
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            if scaler_amp is not None:
                with torch.autocast('cuda'):
                    loss = criterion(model(X_batch), y_batch)
                scaler_amp.scale(loss).backward()
                # Apply SCAFFOLD correction: grad = grad - c_i + c_global
                scaler_amp.unscale_(optimizer)
                for param, ci_v, cg_v in zip(model.parameters(), c_i_t, c_g_t):
                    if param.grad is not None:
                        param.grad.add_(-ci_v + cg_v)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                for param, ci_v, cg_v in zip(model.parameters(), c_i_t, c_g_t):
                    if param.grad is not None:
                        param.grad.add_(-ci_v + cg_v)
                optimizer.step()
            step += 1

    local_params = get_params_as_numpy(model)

    # CV update (Option II):
    # c_i+ = c_i - c + (w_global - x_iK) / (K * eta_l)
    new_c_i  = [
        ci - cg + (wp - lp) / (K * ETA_LOCAL)
        for ci, cg, wp, lp in zip(c_i, c_global, global_params, local_params)
    ]
    delta_c_i = [nc - oc for nc, oc in zip(new_c_i, c_i)]

    return local_params, delta_c_i, new_c_i


# ── Initialise global model + control variates ────────────────────────────────
global_params = get_params_as_numpy(DiabetesNet())
c_global      = zero_like_params(global_params)
c_clients     = [zero_like_params(global_params) for _ in range(N_CLIENTS)]

round_nums, round_aucs = [], []
t0 = time.time()

print(f"\n  Starting SCAFFOLD training ({FL_NUM_ROUNDS} rounds)...")

for rnd in range(1, FL_NUM_ROUNDS + 1):
    local_updates  = []
    delta_c_sum    = zero_like_params(global_params)

    for i, (tr_dl, n_i) in enumerate(zip(loaders, n_samples)):
        lp, dc_i, new_ci = scaffold_client_update(
            i, global_params, c_clients[i], c_global, tr_dl
        )
        local_updates.append((lp, n_i))
        c_clients[i] = new_ci
        for l_idx, dc in enumerate(dc_i):
            delta_c_sum[l_idx] += dc

    # Server aggregation
    new_params = [p.copy() for p in global_params]
    for layer_idx in range(len(global_params)):
        weighted = sum(
            (n_i / total_n) * (lp[layer_idx] - global_params[layer_idx])
            for lp, n_i in local_updates
        )
        new_params[layer_idx] = global_params[layer_idx] + ETA_GLOBAL * weighted

    # Global CV update: c+ = c + (1/N) * sum_i delta_c_i
    for layer_idx in range(len(c_global)):
        c_global[layer_idx] = c_global[layer_idx] + delta_c_sum[layer_idx] / N_CLIENTS

    global_params = new_params

    auc = eval_global(global_params)
    round_nums.append(rnd)
    round_aucs.append(auc)

    if rnd % 10 == 0 or rnd == 1:
        elapsed = time.time() - t0
        eta     = (elapsed / rnd) * (FL_NUM_ROUNDS - rnd)
        print(f"    Round {rnd:3d}/{FL_NUM_ROUNDS}  AUC={auc:.4f}  "
              f"Elapsed={elapsed:.0f}s  ETA={eta:.0f}s")


# ── Save model ────────────────────────────────────────────────────────────────
final_model = DiabetesNet()
set_params_from_numpy(final_model, global_params)
os.makedirs(MODELS_DIR, exist_ok=True)
weights_path = os.path.join(MODELS_DIR, 'scaffold_weights.pt')
torch.save(final_model.state_dict(), weights_path)
print(f"\n  Saved model -> {weights_path}")


# ── Save internal predictions ─────────────────────────────────────────────────
final_model.to(DEVICE)
final_model.eval()
with torch.no_grad():
    X_t = torch.FloatTensor(X_eval).to(DEVICE)
    scaffold_probs = torch.sigmoid(final_model(X_t)).cpu().numpy()
os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(os.path.join(RESULTS_DIR, 'pred_scaffold_internal.npy'), scaffold_probs)
print(f"  Saved preds -> results/pred_scaffold_internal.npy")


# ── Save results JSON ─────────────────────────────────────────────────────────
best_auc  = max(round_aucs)
best_rnd  = round_aucs.index(best_auc) + 1
elapsed_total = time.time() - t0

results = {
    'strategy'   : 'SCAFFOLD (Option II)',
    'reference'  : 'Karimireddy et al. ICML 2020, arXiv:2003.00295',
    'K_local'    : K,
    'eta_local'  : ETA_LOCAL,
    'eta_global' : ETA_GLOBAL,
    'rounds'     : round_nums,
    'aucs'       : round_aucs,
    'final_auc'  : round_aucs[-1],
    'best_auc'   : best_auc,
    'best_round' : best_rnd,
    'runtime_s'  : elapsed_total,
}
with open(os.path.join(RESULTS_DIR, 'scaffold_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved -> results/scaffold_results.json")


# ── Comparison plot ───────────────────────────────────────────────────────────
try:
    conv_path = os.path.join(RESULTS_DIR, 'federated_convergence.json')
    with open(conv_path) as f:
        prior = json.load(f)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        'FedAvg': '#2563EB', 'FedProx (μ=0.1)': '#7C3AED',
        'FedNova': '#F59E0B', 'SCAFFOLD': '#16A34A',
    }
    for name, data in prior.items():
        col = colors.get(name, '#94A3B8')
        ax.plot(data['rounds'], data['aucs'], lw=1.8, color=col,
                alpha=0.7, ls='--', label=f'{name}')
    ax.plot(round_nums, round_aucs, lw=2.5, color='#16A34A',
            label=f'SCAFFOLD (final={round_aucs[-1]:.4f})')
    ax.axhline(PUBLISHED_INTERNAL_AUC, color='#1E293B', ls=':', lw=1.5,
               label=f'Centralised ({PUBLISHED_INTERNAL_AUC})')
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('SCAFFOLD vs FL Baselines — Internal AUC', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_scaffold_convergence.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> plots/08_scaffold_convergence.png")
except Exception as e:
    print(f"  Plot skipped: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SCAFFOLD — SUMMARY")
print("=" * 65)
print(f"\n  Strategy              : SCAFFOLD Option II")
print(f"  Local steps (K)       : {K}")
print(f"  Final AUC (internal)  : {round_aucs[-1]:.4f}")
print(f"  Best  AUC (internal)  : {best_auc:.4f}  (round {best_rnd})")
print(f"  Runtime               : {elapsed_total:.0f}s")
print("=" * 65)
