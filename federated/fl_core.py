"""
fl_core.py — canonical federated-learning core
==============================================
Single source of truth for FL training. Both 03_federated_simulation.py (the
original single-run pipeline) and 33_multiseed_all_models.py import from here,
so there is exactly one implementation of FedAvg/FedProx/FedNova in the project.

Extracted from 03_federated_simulation.py. Behaviour is unchanged except that
every entry point now accepts an explicit `seed`. Previously nothing in the FL
pipeline was seeded — `new_net()` was initialised randomly on every run — so
every reported federated number came from one uncontrolled draw.

Seeding covers: node train/val split, global-model initialisation, dropout, and
DataLoader shuffle order. Pass the same seed to build_node_loaders() and
run_federated() to make a run reproducible end to end.
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_auc_score

from config_paths import (
    NODE_PATHS, NODE_NAMES, CENTRALISED_PATH, MODELS_DIR,
    FEATURE_COLS_NORACE as FEATURE_COLS, TARGET_COL,
    GLOBAL_SCALER_NORACE_PATH as GLOBAL_SCALER_PATH, N_FEATURES_NORACE,
    FL_NUM_ROUNDS, FEDPROX_MU,
    NN_LOCAL_EPOCHS, NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    SEED,
)
from nn_model   import DiabetesNet, train_one_epoch
from data_utils import (load_node_data, get_dataloaders, compute_class_weight,
                        get_device, get_params_as_numpy, set_params_from_numpy)

DEVICE = get_device()


def new_net():
    """
    Every model in this pipeline is race-free: RIDRETH3 is dropped as an input
    and used only to stratify evaluation. See config_paths.FEATURE_COLS_NORACE
    for why. input_dim must therefore be 7, not DiabetesNet's default of 8.
    """
    return DiabetesNet(input_dim=N_FEATURES_NORACE)


def seed_everything(seed: int):
    """Seed torch/numpy/CUDA. Call before building loaders and before training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── BatchNorm buffers ─────────────────────────────────────────────────────────
# DiabetesNet uses nn.BatchNorm1d. Its weight/bias are in model.parameters() and
# so ride along with every aggregation, but running_mean/running_var are BUFFERS:
# not in parameters(), therefore never sent, aggregated, or restored. Rebuilding
# a model from parameters alone leaves them at their init values (mean 0, var 1),
# so every federated model used to be scored in eval() mode with untrained BN
# statistics while the centralised controls kept their learned ones. Measured
# cost at seed 42: FedAvg external AUC 0.7406 -> 0.7457 (+0.0051), the same size
# as the federated-vs-centralised difference being tested. See
# 36_bn_buffer_check.py and results/bn_buffer_check.json.

def get_buffers_as_numpy(model):
    return {k: v.detach().cpu().numpy() for k, v in model.named_buffers()}


def set_buffers_from_numpy(model, buffers):
    if not buffers:
        return
    with torch.no_grad():
        for k, b in model.named_buffers():
            if k in buffers:
                b.copy_(torch.tensor(buffers[k], dtype=b.dtype, device=b.device))


def aggregate_buffers(client_buffers, n_samples):
    """
    Size-weighted pooling of BatchNorm buffers, matching FedAvg's n_k/N weighting.

    running_mean pools as a weighted mean. running_var uses the exact pooled
    variance identity Var = E[x^2] - (E[x])^2 rather than a weighted mean of the
    client variances, which would discard the between-client mean shift — and
    these clients are deliberately non-IID, so that shift is not negligible.
    """
    w = np.asarray(n_samples, dtype=np.float64)
    w = w / w.sum()
    names = list(client_buffers[0].keys())
    out = {}
    for name in names:
        if name.endswith('num_batches_tracked'):
            # Unused in eval() while momentum is a fixed float (default 0.1).
            out[name] = client_buffers[0][name]
        elif name.endswith('running_var'):
            pass                                    # handled with its mean
        elif name.endswith('running_mean'):
            v_name = name[:-len('running_mean')] + 'running_var'
            means = np.stack([cb[name]   for cb in client_buffers])
            varis = np.stack([cb[v_name] for cb in client_buffers])
            wc = w.reshape(-1, *([1] * (means.ndim - 1)))
            m  = (wc * means).sum(0)
            second = (wc * (varis + means ** 2)).sum(0)
            out[name]   = m
            out[v_name] = np.maximum(second - m ** 2, 0.0)
        else:
            arr = np.stack([cb[name] for cb in client_buffers])
            wc  = w.reshape(-1, *([1] * (arr.ndim - 1)))
            out[name] = (wc * arr).sum(0)
    return out


# ── Centralised NHANES eval set (lazy; used to track global AUC each round) ────
_EVAL_CACHE = None


def get_eval_set():
    """(X_eval_scaled, y_eval) on the centralised NHANES pool. Cached."""
    global _EVAL_CACHE
    if _EVAL_CACHE is None:
        df = pd.read_csv(CENTRALISED_PATH)
        X  = df[FEATURE_COLS].values.astype(np.float32)
        y  = df[TARGET_COL].values.astype(np.float32)
        # Global NHANES-fitted scaler only — never fit on node data (leakage).
        sc = joblib.load(GLOBAL_SCALER_PATH)
        _EVAL_CACHE = (sc.transform(X).astype(np.float32), y)
    return _EVAL_CACHE


@torch.no_grad()
def eval_global(params, buffers=None) -> float:
    """AUC of the current global weights on the centralised NHANES set."""
    X_eval_sc, y_eval = get_eval_set()
    model = new_net().to(DEVICE)
    set_params_from_numpy(model, params)
    set_buffers_from_numpy(model, buffers)
    model.eval()
    probs = torch.sigmoid(model(torch.FloatTensor(X_eval_sc).to(DEVICE))).cpu().numpy()
    try:
        return float(roc_auc_score(y_eval, probs))
    except Exception:
        return 0.0


def build_node_loaders(seed: int = SEED):
    """Load all 3 nodes with the global NHANES scaler (no per-node fitting)."""
    global_scaler = joblib.load(GLOBAL_SCALER_PATH)
    use_pin = DEVICE.type == 'cuda'
    loaders, n_samples = [], []
    for path in NODE_PATHS:
        X_tr, y_tr, X_val, y_val, _ = load_node_data(
            path, val_size=0.2, seed=seed, scaler=global_scaler, fit_scaler=False,
            feature_cols=FEATURE_COLS,      # race-free; must match the scaler
        )
        tr_dl, val_dl = get_dataloaders(
            X_tr, y_tr, X_val, y_val, NN_BATCH_SIZE, pin_memory=use_pin
        )
        loaders.append((tr_dl, val_dl, y_tr, y_val))
        n_samples.append(len(X_tr))
    return loaders, n_samples


# ── Aggregation strategies ────────────────────────────────────────────────────

def fedavg_aggregate(updates, n_samples):
    """FedAvg (McMahan et al. 2017): size-weighted average of client updates."""
    total = sum(n_samples)
    return [
        sum((n / total) * updates[c][i] for c, n in enumerate(n_samples))
        for i in range(len(updates[0]))
    ]


def fedprox_aggregate(updates, n_samples):
    """FedProx (Li et al. 2020): aggregation identical to FedAvg; the proximal
    term is applied client-side in train_one_epoch with mu>0."""
    return fedavg_aggregate(updates, n_samples)


def fednova_aggregate(updates, n_samples, local_steps):
    """FedNova (Wang et al. 2020): normalise each client update by its local
    step count before aggregation, correcting objective inconsistency."""
    total   = sum(n_samples)
    tau_eff = local_steps
    return [
        sum((n / total) * (updates[c][i] / local_steps) * tau_eff
            for c, n in enumerate(n_samples))
        for i in range(len(updates[0]))
    ]


def equal_client_aggregate(updates, n_samples):
    """
    Equal-client-weight aggregation: every client contributes 1/K regardless of
    its sample count, instead of FedAvg's n_k/N.

    Used by the client-weighting attribution experiment (T1.3) to separate
    "federation helps" from "the implicit reweighting of demographic groups that
    client-level aggregation induces".
    """
    K = len(updates)
    return [
        sum((1.0 / K) * updates[c][i] for c in range(K))
        for i in range(len(updates[0]))
    ]


# ── Local client training ─────────────────────────────────────────────────────

def local_train(global_params, train_dl, y_train, proximal_mu=0.0, global_buffers=None):
    """One client's local training for one FL round -> (params, buffers) as numpy."""
    pos_weight = compute_class_weight(y_train)
    model      = new_net().to(DEVICE)
    set_params_from_numpy(model, global_params)
    set_buffers_from_numpy(model, global_buffers)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    global_tensors = None
    if proximal_mu > 0.0:
        global_tensors = [p.detach().clone() for p in model.parameters()]

    for _ in range(NN_LOCAL_EPOCHS):
        train_one_epoch(model, train_dl, optimizer, criterion,
                        DEVICE, proximal_mu, global_tensors, scaler=scaler)

    return get_params_as_numpy(model), get_buffers_as_numpy(model)


# ── One full strategy run ─────────────────────────────────────────────────────

def run_federated(
    strategy_name: str,
    aggregator,
    loaders,
    n_samples,
    proximal_mu: float = 0.0,
    seed: int = None,
    save_weights: bool = True,
    weights_suffix: str = '',
    verbose: bool = True,
):
    """
    Full federated training for one strategy.

    seed           — seeds global-model init (and dropout/shuffle). None keeps the
                     legacy unseeded behaviour.
    save_weights   — False for sweep runs that should not clobber canonical weights.
    weights_suffix — e.g. '_seed43', appended before '.pt'.

    Returns dict with round-by-round AUC history, final params, and weights path.
    """
    if seed is not None:
        seed_everything(seed)

    if verbose:
        print(f"\n  {'─'*55}")
        print(f"  Strategy: {strategy_name}"
              + (f"   seed={seed}" if seed is not None else ""))
        print(f"  {'─'*55}")

    _init = new_net()
    global_params  = get_params_as_numpy(_init)
    global_buffers = get_buffers_as_numpy(_init)
    round_aucs, round_nums = [], []
    t0 = time.time()

    for rnd in range(1, FL_NUM_ROUNDS + 1):
        results = [
            local_train([p.copy() for p in global_params], tr_dl, y_tr, proximal_mu,
                        global_buffers=global_buffers)
            for (tr_dl, _, y_tr, _) in loaders
        ]
        client_updates = [r[0] for r in results]
        client_buffers = [r[1] for r in results]

        if 'nova' in strategy_name.lower():
            global_params = aggregator(client_updates, n_samples,
                                       local_steps=NN_LOCAL_EPOCHS)
        else:
            global_params = aggregator(client_updates, n_samples)
        # BN statistics are pooled by client size regardless of the parameter
        # aggregator: they are descriptive statistics of the clients' data, not
        # optimiser state, so size weighting is the correct pooling either way.
        global_buffers = aggregate_buffers(client_buffers, n_samples)

        auc = eval_global(global_params, global_buffers)
        round_nums.append(rnd)
        round_aucs.append(auc)

        if verbose and (rnd % 10 == 0 or rnd == 1):
            elapsed = time.time() - t0
            eta     = (elapsed / rnd) * (FL_NUM_ROUNDS - rnd)
            print(f"    Round {rnd:3d}/{FL_NUM_ROUNDS}  AUC={auc:.4f}  "
                  f"Elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    weights_path = None
    if save_weights:
        final_model = new_net()
        set_params_from_numpy(final_model, global_params)
        set_buffers_from_numpy(final_model, global_buffers)
        save_name = (strategy_name.lower().split(' ')[0]
                     .replace('(', '').replace(')', ''))
        weights_path = os.path.join(MODELS_DIR,
                                    f'{save_name}{weights_suffix}_weights.pt')
        torch.save(final_model.state_dict(), weights_path)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Final AUC : {round_aucs[-1]:.4f}   Runtime: {elapsed:.0f}s")

    return {
        'strategy'      : strategy_name,
        'seed'          : seed,
        'rounds'        : round_nums,
        'aucs'          : round_aucs,
        'final_auc'     : round_aucs[-1],
        'best_auc'      : max(round_aucs),
        'best_round'    : int(round_aucs.index(max(round_aucs)) + 1),
        'runtime_s'     : elapsed,
        'weights_path'  : weights_path,
        'global_params' : global_params,
        'global_buffers': global_buffers,
    }


# ── SCAFFOLD ──────────────────────────────────────────────────────────────────
# Kept separate from run_federated() because SCAFFOLD is not just an aggregation
# rule: it carries per-client control variates and modifies the local gradient,
# so it does not fit the (updates, n_samples) -> params signature.
#
# This is a reimplementation, not a port. 08_scaffold_baseline.py ran `while
# step < K` with K = NN_LOCAL_EPOCHS = 5 and incremented `step` once per BATCH,
# so it performed 5 mini-batch steps per round where every other strategy
# performed 5 full epochs (~90 batches) — roughly 1/18th the local budget. Its
# resulting underperformance was therefore a property of the training budget,
# not of SCAFFOLD, and the manuscript's "instructive negative result" paragraph
# rests on it. Here the local budget matches the other strategies exactly.
#
# The local optimiser remains SGD: SCAFFOLD's control-variate correction is
# derived for SGD, and applying it to AdamW's adaptively rescaled step has no
# such guarantee. That leaves an optimiser difference against the AdamW-based
# strategies which must be stated when the result is reported; eta_local is
# therefore selected on held-out node validation AUC (never on BRFSS) rather
# than inherited from the AdamW learning rate, which 08_scaffold_baseline.py
# did (SGD at lr=1e-3).

def scaffold_local_train(global_params, global_buffers, c_i, c_global,
                         train_dl, y_train, eta_local):
    """One client's SCAFFOLD round -> (params, buffers, new_c_i, n_steps)."""
    pos_weight = compute_class_weight(y_train)
    model      = new_net().to(DEVICE)
    set_params_from_numpy(model, global_params)
    set_buffers_from_numpy(model, global_buffers)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=eta_local)

    c_i_t = [torch.tensor(v, device=DEVICE, dtype=torch.float32) for v in c_i]
    c_g_t = [torch.tensor(v, device=DEVICE, dtype=torch.float32) for v in c_global]

    # No AMP here: the correction is applied directly to .grad, and interleaving
    # that with GradScaler's unscale/skip logic is a silent-corruption risk for
    # no meaningful speedup on a 4,993-parameter model.
    steps = 0
    model.train()
    for _ in range(NN_LOCAL_EPOCHS):
        for X_batch, y_batch in train_dl:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            # SCAFFOLD: g <- g - c_i + c
            for p, ci_v, cg_v in zip(model.parameters(), c_i_t, c_g_t):
                if p.grad is not None:
                    p.grad.add_(cg_v - ci_v)
            optimizer.step()
            steps += 1

    local_params = get_params_as_numpy(model)
    # Option II control variate: c_i+ = c_i - c + (x_global - x_local)/(K*eta_l)
    new_c_i = [ci - cg + (wp - lp) / (steps * eta_local)
               for ci, cg, wp, lp in zip(c_i, c_global, global_params, local_params)]
    return local_params, get_buffers_as_numpy(model), new_c_i, steps


def run_scaffold(loaders, n_samples, eta_local, eta_global=1.0,
                 seed=None, verbose=True):
    """SCAFFOLD (Karimireddy et al. 2020), Option II control variates."""
    if seed is not None:
        seed_everything(seed)

    _init          = new_net()
    global_params  = get_params_as_numpy(_init)
    global_buffers = get_buffers_as_numpy(_init)
    c_global       = [np.zeros_like(p) for p in global_params]
    c_clients      = [[np.zeros_like(p) for p in global_params] for _ in loaders]

    n_clients  = len(loaders)
    total      = sum(n_samples)
    round_aucs, round_nums = [], []
    t0 = time.time()

    for rnd in range(1, FL_NUM_ROUNDS + 1):
        local_params_all, buffers_all = [], []
        delta_c_sum = [np.zeros_like(p) for p in global_params]

        for i, (tr_dl, _, y_tr, _) in enumerate(loaders):
            lp, buf, new_ci, _ = scaffold_local_train(
                [p.copy() for p in global_params], global_buffers,
                c_clients[i], c_global, tr_dl, y_tr, eta_local)
            for l, (nc, oc) in enumerate(zip(new_ci, c_clients[i])):
                delta_c_sum[l] += nc - oc
            c_clients[i] = new_ci
            local_params_all.append(lp)
            buffers_all.append(buf)

        global_params = [
            gp + eta_global * sum((n / total) * (lp[l] - gp)
                                  for lp, n in zip(local_params_all, n_samples))
            for l, gp in enumerate(global_params)
        ]
        c_global = [cg + dc / n_clients for cg, dc in zip(c_global, delta_c_sum)]
        global_buffers = aggregate_buffers(buffers_all, n_samples)

        auc = eval_global(global_params, global_buffers)
        round_nums.append(rnd)
        round_aucs.append(auc)
        if verbose and (rnd % 10 == 0 or rnd == 1):
            print(f"    Round {rnd:3d}/{FL_NUM_ROUNDS}  AUC={auc:.4f}")

    return {
        'strategy'      : 'SCAFFOLD',
        'seed'          : seed,
        'eta_local'     : eta_local,
        'eta_global'    : eta_global,
        'rounds'        : round_nums,
        'aucs'          : round_aucs,
        'final_auc'     : round_aucs[-1],
        'best_auc'      : max(round_aucs),
        'runtime_s'     : time.time() - t0,
        'global_params' : global_params,
        'global_buffers': global_buffers,
    }
