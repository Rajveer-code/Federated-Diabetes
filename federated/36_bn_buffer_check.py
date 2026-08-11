"""
36_bn_buffer_check.py — is the federated/centralised comparison confounded by BatchNorm?
========================================================================================
DiabetesNet uses nn.BatchNorm1d. Its learnable weight/bias are in
model.parameters() and so are aggregated by every FL strategy here. Its BUFFERS
(running_mean, running_var) are not in model.parameters(), so they are never
sent, never aggregated, and never restored: params_to_model() builds a fresh
DiabetesNet (running_mean=0, running_var=1) and overwrites only the parameters.

Every federated model is therefore scored in eval() mode using untrained BN
statistics, while train_centralised() returns the actual trained module and keeps
its learned ones. If that costs the federated models any AUC, it is a systematic
handicap pointing in exactly the direction of the headline result
("FedAvg is significantly worse than the matched centralised control").

This script measures the size of the effect at one seed:
  A. as currently scored — default BN buffers
  B. same weights, BN statistics recomputed on the federated training data
     (a cumulative average over one pass, which is what the buffers would hold
      had they been tracked and aggregated)

Output: results/bn_buffer_check.json

Usage:  python 36_bn_buffer_check.py [--seed 42]
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
import joblib
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (NODE_PATHS, RESULTS_DIR, FEATURE_COLS, TARGET_COL,
                          GLOBAL_SCALER_PATH, SEED, FEDPROX_MU)
from nn_model   import DiabetesNet
from data_utils import load_brfss_clean, set_params_from_numpy
from fl_core    import (DEVICE, seed_everything, build_node_loaders, run_federated,
                        fedavg_aggregate, fedprox_aggregate)

CHUNK = 65536


@torch.no_grad()
def score(model, X):
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(torch.FloatTensor(X[i:i + CHUNK]).to(DEVICE))).cpu().numpy()
        for i in range(0, len(X), CHUNK)])


@torch.no_grad()
def refresh_bn(model, X, batch=1024):
    """Recompute BN running stats on X with a cumulative average (momentum=None)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    for i in range(0, len(X), batch):
        xb = torch.FloatTensor(X[i:i + batch]).to(DEVICE)
        if len(xb) > 1:                      # BN needs >1 row in train mode
            model(xb)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    scaler = joblib.load(GLOBAL_SCALER_PATH)
    df_b, X_b_raw, y_b = load_brfss_clean(verbose=False)
    X_b = scaler.transform(X_b_raw)

    df_fed = pd.concat([pd.read_csv(p) for p in NODE_PATHS], ignore_index=True)
    X_fed  = scaler.transform(df_fed[FEATURE_COLS].values.astype(np.float32))

    seed_everything(args.seed)
    loaders, n_samples = build_node_loaders(seed=args.seed)
    X_val = np.concatenate([dl_v.dataset.X.numpy() for (_, dl_v, _, _) in loaders])
    y_val = np.concatenate([dl_v.dataset.y.numpy() for (_, dl_v, _, _) in loaders])

    out = {'seed': args.seed, 'n_brfss': int(len(y_b)), 'strategies': {}}

    for name, agg, mu in [('FedAvg', fedavg_aggregate, 0.0),
                          ('FedProx', fedprox_aggregate, FEDPROX_MU)]:
        r = run_federated(name, agg, loaders, n_samples, proximal_mu=mu,
                          seed=args.seed, save_weights=False, verbose=False)

        m_default = DiabetesNet().to(DEVICE)
        set_params_from_numpy(m_default, r['global_params'])
        ext_a = float(roc_auc_score(y_b,   score(m_default, X_b)))
        val_a = float(roc_auc_score(y_val, score(m_default, X_val)))

        m_fixed = DiabetesNet().to(DEVICE)
        set_params_from_numpy(m_fixed, r['global_params'])
        refresh_bn(m_fixed, X_fed)
        ext_b = float(roc_auc_score(y_b,   score(m_fixed, X_b)))
        val_b = float(roc_auc_score(y_val, score(m_fixed, X_val)))

        out['strategies'][name] = {
            'external_default_bn': round(ext_a, 5),
            'external_refreshed_bn': round(ext_b, 5),
            'external_delta': round(ext_b - ext_a, 5),
            'internal_default_bn': round(val_a, 5),
            'internal_refreshed_bn': round(val_b, 5),
            'internal_delta': round(val_b - val_a, 5),
        }
        print(f"  {name:<9} external {ext_a:.5f} -> {ext_b:.5f}  ({ext_b - ext_a:+.5f})"
              f"   internal {val_a:.5f} -> {val_b:.5f}  ({val_b - val_a:+.5f})")

    path = os.path.join(RESULTS_DIR, 'bn_buffer_check.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")

    worst = max(abs(v['external_delta']) for v in out['strategies'].values())
    print(f"\n  Largest external-AUC shift: {worst:+.5f}")
    print("  The headline FedAvg-vs-matched-control delta is -0.0049 [-0.0098, -0.0001].")
    if worst > 0.001:
        print("  -> MATERIAL. BN buffers must be aggregated before the comparison stands.")
    else:
        print("  -> negligible; the comparison is unaffected.")


if __name__ == '__main__':
    main()
