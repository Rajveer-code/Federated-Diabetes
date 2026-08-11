"""
37_scaffold_lr_sweep.py — choose SCAFFOLD's local learning rate honestly
=======================================================================
SCAFFOLD trains locally with SGD (its control-variate correction is derived for
SGD). 08_scaffold_baseline.py reused NN_LR = 1e-3, which is an AdamW learning
rate; plain SGD at 1e-3 on this network barely moves. That, plus a local budget
of 5 mini-batches instead of 5 epochs, is why the old SCAFFOLD run looked like a
negative result.

eta_local has to come from somewhere, and guessing it would be inventing a
statistical parameter. It is selected here on the concatenated held-out node
VALIDATION splits — the same internal set used for the generalisation gap. BRFSS
is never touched during selection, so the external estimate stays clean.

Output: results/scaffold_lr_sweep.json

Usage:  python 37_scaffold_lr_sweep.py [--seed 42]
"""
import os, sys, json, argparse, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, SEED
from nn_model     import DiabetesNet
from data_utils   import set_params_from_numpy
from fl_core      import (DEVICE, seed_everything, build_node_loaders,
                          run_scaffold, set_buffers_from_numpy)

GRID = [1e-3, 1e-2, 3e-2, 1e-1]


@torch.no_grad()
def auc_on(params, buffers, X, y):
    m = DiabetesNet().to(DEVICE)
    set_params_from_numpy(m, params)
    set_buffers_from_numpy(m, buffers)
    m.eval()
    p = torch.sigmoid(m(torch.FloatTensor(X).to(DEVICE))).cpu().numpy()
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--grid', type=float, nargs='+', default=None,
                    help='override the eta_local grid (e.g. --grid 0.1 0.3 1.0)')
    args = ap.parse_args()
    grid = args.grid if args.grid else GRID

    seed_everything(args.seed)
    loaders, n_samples = build_node_loaders(seed=args.seed)
    X_val = np.concatenate([dl_v.dataset.X.numpy() for (_, dl_v, _, _) in loaders])
    y_val = np.concatenate([dl_v.dataset.y.numpy() for (_, dl_v, _, _) in loaders])
    print(f"Selecting SCAFFOLD eta_local on {len(y_val):,} held-out node-validation rows")

    runs = []
    for eta in grid:
        seed_everything(args.seed)
        loaders, n_samples = build_node_loaders(seed=args.seed)
        r = run_scaffold(loaders, n_samples, eta_local=eta,
                         seed=args.seed, verbose=False)
        a = auc_on(r['global_params'], r['global_buffers'], X_val, y_val)
        runs.append({'eta_local': eta, 'internal_val_auc': round(a, 5),
                     'final_pool_auc': round(r['final_auc'], 5)})
        print(f"  eta_local={eta:<8g} internal val AUC={a:.5f}")

    best = max(runs, key=lambda r: r['internal_val_auc'])
    print(f"\n  Selected eta_local = {best['eta_local']}  "
          f"(internal val AUC {best['internal_val_auc']:.5f})")
    if best['eta_local'] == grid[-1]:
        print("  NOTE: the best value is at the top of the grid — extend it before trusting.")

    out = os.path.join(RESULTS_DIR, 'scaffold_lr_sweep.json')
    with open(out, 'w') as f:
        json.dump({'seed': args.seed, 'grid': grid,
                   'selected_on': 'held-out node validation AUC (BRFSS untouched)',
                   'n_val': int(len(y_val)), 'runs': runs,
                   'selected_eta_local': best['eta_local']}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
