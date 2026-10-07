"""
62_layernorm_arm.py — the headline comparison with LayerNorm in place of BatchNorm
=================================================================================
Reviewer QUmG asked whether alternative normalisation changes the result. The
no-BatchNorm arm (56) tests whether pooled BN statistics are needed; it is not a
LayerNorm or GroupNorm model. This arm swaps nn.BatchNorm1d for nn.LayerNorm in
both FedAvg and the composition-matched centralised control, everything else
unchanged. LayerNorm normalises within each sample, so no running statistics
exist and nothing but weights is averaged across clients.

FedBN (local BN statistics) is not run: it yields one model per site, and the
BRFSS external cohort has no site to pick statistics from.

Output: <RESULTS_DIR>/layernorm_arm.json  (+ cached preds FedAvg-LN / Centralised-matched-LN)
Usage:  python 62_layernorm_arm.py [--seeds 10]
"""
import os, sys, json, argparse, importlib
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn
from typing import List
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (NODE_PATHS, RESULTS_DIR, TARGET_COL,
                          FEATURE_COLS_NORACE as FEATURE_COLS,
                          GLOBAL_SCALER_NORACE_PATH as GLOBAL_SCALER_PATH,
                          N_FEATURES_NORACE, FL_NUM_ROUNDS, SEED)
from data_utils import load_brfss_clean, set_params_from_numpy
import fl_core
from fl_core import DEVICE, seed_everything, build_node_loaders, run_federated, fedavg_aggregate

b56 = importlib.import_module('56_budget_and_bn_controls')   # score, cache, train_centralised
PRED_DIR = b56.PRED_DIR
RACE_STRATA = {'White': [1.0], 'Black': [2.0], 'Hispanic': [8.0], 'Asian': [4.0],
               'Other': [3.0, 5.0, 6.0, 7.0]}
Q = 0.20


class DiabetesNetLN(nn.Module):
    """DiabetesNet with nn.BatchNorm1d -> nn.LayerNorm; widths, dropout, init unchanged."""

    def __init__(self, input_dim: int = N_FEATURES_NORACE,
                 hidden_dims: List[int] = [64, 32, 16], dropout: float = 0.3):
        super().__init__()
        layers, in_dim = [], input_dim
        for i, h in enumerate(hidden_dims):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=False),
                       nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.6)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {'mean': float(x.mean()), 'lo': float(x.mean() - h), 'hi': float(x.mean() + h),
            'n_pos': int((x > 0).sum()), 'n': int(len(x))}


def metrics(p, y, masks):
    thr = np.sort(p)[int(len(p) * (1 - Q))]
    flag, wn = p >= thr, masks['White'] & (y == 0)
    out = {}
    for g, mk in masks.items():
        pos = mk & (y == 1)
        out[g] = {'auc': roc_auc_score(y[mk], p[mk]), 'sel': flag[mk].mean(), 'sens': flag[pos].mean(),
                  'xauc': (roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(wn.sum())], np.r_[p[pos], p[wn]])
                           if g != 'White' else np.nan)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()
    seeds = [SEED + i for i in range(args.seeds)]
    os.makedirs(PRED_DIR, exist_ok=True)

    scaler = joblib.load(GLOBAL_SCALER_PATH)
    df_b, X_b_raw, y_b = load_brfss_clean(feature_cols=FEATURE_COLS)
    X_b = scaler.transform(X_b_raw)
    df_fed = pd.concat([pd.read_csv(p) for p in NODE_PATHS], ignore_index=True)
    X_fed = scaler.transform(df_fed[FEATURE_COLS].values.astype(np.float32))
    y_fed = df_fed[TARGET_COL].values.astype(np.float32)

    for seed in seeds:
        Xtr, _, ytr, _ = train_test_split(X_fed, y_fed, test_size=0.20, stratify=y_fed, random_state=seed)
        m = b56.train_centralised(Xtr, ytr, seed, FL_NUM_ROUNDS, DiabetesNetLN)
        b56.cache(f'Centralised-matched-LN_seed{seed}_external', b56.score(m, X_b))

        seed_everything(seed)
        loaders, n_samples = build_node_loaders(seed=seed)
        orig = fl_core.new_net
        fl_core.new_net = lambda: DiabetesNetLN()
        try:
            r = run_federated('FedAvg-LN', fedavg_aggregate, loaders, n_samples, proximal_mu=0.0,
                              seed=seed, save_weights=False, verbose=False)
            m = DiabetesNetLN().to(DEVICE)
            set_params_from_numpy(m, r['global_params'])
        finally:
            fl_core.new_net = orig
        p = b56.score(m, X_b)
        b56.cache(f'FedAvg-LN_seed{seed}_external', p)
        print(f'  seed {seed}: FedAvg-LN ext AUC {roc_auc_score(y_b, p):.4f}', flush=True)

    y = y_b.astype(int)
    race = df_b['RIDRETH3_RAW'].values
    masks = {g: np.isin(race, v) for g, v in RACE_STRATA.items()}
    per = [(metrics(np.load(os.path.join(PRED_DIR, f'FedAvg-LN_seed{s}_external.npy')), y, masks),
            metrics(np.load(os.path.join(PRED_DIR, f'Centralised-matched-LN_seed{s}_external.npy')), y, masks))
           for s in seeds]
    res = {}
    for g in masks:
        for k in ('auc', 'xauc', 'sel', 'sens'):
            if not (g == 'White' and k == 'xauc'):
                res[f'{k}_{g}'] = ci([f[g][k] - c[g][k] for f, c in per])
    res['auc_gap_WB'] = ci([(f['White']['auc'] - f['Black']['auc']) - (c['White']['auc'] - c['Black']['auc']) for f, c in per])
    res['sens_gap_WB'] = ci([(f['White']['sens'] - f['Black']['sens']) - (c['White']['sens'] - c['Black']['sens']) for f, c in per])
    out = {'comparison': 'FedAvg-LN minus Centralised-matched-LN', 'capacity': Q, 'seeds': seeds, 'deltas': res}
    with open(os.path.join(RESULTS_DIR, 'layernorm_arm.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for k in ['auc_Black', 'auc_gap_WB', 'xauc_Black', 'sel_Black', 'sens_Black', 'sens_gap_WB',
              'sens_Hispanic', 'sens_Other', 'sens_Asian']:
        d = res[k]
        print(f"  {k:14s} {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  {d['n_pos']}/10 positive")


if __name__ == '__main__':
    main()
