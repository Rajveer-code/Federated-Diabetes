"""
43_partition_sweep.py — does the subgroup effect scale with node heterogeneity?
==============================================================================
The proposed mechanism is that averaging node-specific optima stops the pooled
majority stratum from dominating the shared representation, and that this only
bites when the nodes are racially separated. Eliminating two confounds (data
volume, client weighting) supports that indirectly. A dose-response test is
stronger: if the mechanism is right, the effect must vanish when the nodes are
racially homogeneous and grow when they are racially separated.

Same 11,906 federated cohort in every arm -- only the partition changes -- so the
composition-matched centralised control from 33_multiseed_all_models.py remains
the correct reference for all of them.

Partitions, spanning low to high racial separation:
  random   3 random equal nodes                 -> near-zero separation
  bmi      BMI tertiles                         -> real covariate shift, little racial
  age_urb  the original age/urbanicity split    -> incidental racial separation
  by_race  explicit racial grouping             -> maximal separation

Prediction: |Delta White-Black gap| increases with cross-node racial heterogeneity,
and is ~0 for `random`.

Output: results/partition_sweep.json

Usage:  python 43_partition_sweep.py [--seeds 5]
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
import torch
warnings.filterwarnings('ignore')
import joblib
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (NODE_PATHS, RESULTS_DIR, TARGET_COL, RACE_COL, SEED,
                          FEATURE_COLS_NORACE as FEATURE_COLS,
                          GLOBAL_SCALER_NORACE_PATH as SCALER_PATH,
                          NN_BATCH_SIZE)
from data_utils import DiabetesDataset, get_dataloaders, load_brfss_clean
import fl_core
from fl_core import DEVICE, seed_everything, run_federated, fedavg_aggregate

CHUNK = 65536
RACE_STRATA = {'White': [1.0], 'Black': [2.0], 'Hispanic': [8.0],
               'Asian': [4.0], 'Other': [3.0, 5.0, 6.0, 7.0]}
# NHANES RIDRETH3 buckets used only to define the `by_race` partition.
RACE_BUCKETS = [[3.0], [4.0, 7.0], [1.0, 2.0, 6.0]]


@torch.no_grad()
def score(model, X):
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(torch.FloatTensor(X[i:i + CHUNK]).to(DEVICE))).cpu().numpy()
        for i in range(0, len(X), CHUNK)])


def make_partitions(df, rng):
    """Return {name: [idx_arrays]} — four 3-node partitions of the same cohort."""
    n = len(df)
    parts = {}

    # Original: node membership is the file each row came from.
    parts['age_urb'] = [np.flatnonzero(df['_node'].values == i) for i in range(3)]

    perm = rng.permutation(n)
    parts['random'] = [np.sort(a) for a in np.array_split(perm, 3)]

    bmi = df['BMXBMI'].values
    cuts = np.quantile(bmi, [1 / 3, 2 / 3])
    parts['bmi'] = [np.flatnonzero(bmi <= cuts[0]),
                    np.flatnonzero((bmi > cuts[0]) & (bmi <= cuts[1])),
                    np.flatnonzero(bmi > cuts[1])]

    race = df[RACE_COL].values
    parts['by_race'] = [np.flatnonzero(np.isin(race, b)) for b in RACE_BUCKETS]
    return parts


def heterogeneity(df, idx_list):
    """Mean total-variation distance between each node's race mix and the pooled mix."""
    codes = sorted(df[RACE_COL].dropna().unique())
    pooled = np.array([(df[RACE_COL].values == c).mean() for c in codes])
    tvds = []
    for idx in idx_list:
        sub = df[RACE_COL].values[idx]
        p = np.array([(sub == c).mean() for c in codes])
        tvds.append(0.5 * np.abs(p - pooled).sum())
    return float(np.mean(tvds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    scaler = joblib.load(SCALER_PATH)

    frames = []
    for i, p in enumerate(NODE_PATHS):
        d = pd.read_csv(p); d['_node'] = i; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    X = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    y = df[TARGET_COL].values.astype(np.float32)
    print(f"Federated cohort: n={len(df):,}")

    df_b, X_b_raw, y_b = load_brfss_clean(verbose=False, feature_cols=FEATURE_COLS)
    X_b = scaler.transform(X_b_raw)
    race_b = df_b['RIDRETH3_RAW'].values
    masks = {k: np.isin(race_b, v) for k, v in RACE_STRATA.items()}

    parts = make_partitions(df, np.random.default_rng(SEED))
    het = {k: heterogeneity(df, v) for k, v in parts.items()}
    print("\ncross-node racial heterogeneity (mean TVD vs pooled):")
    for k in sorted(het, key=het.get):
        print(f"  {k:<9} {het[k]:.4f}   node sizes {[len(a) for a in parts[k]]}")

    records = []
    for pname, idx_list in parts.items():
        for i in range(args.seeds):
            seed = SEED + i
            seed_everything(seed)
            loaders, n_samples = [], []
            for idx in idx_list:
                Xi, yi = X[idx], y[idx]
                cut = int(0.8 * len(idx))
                perm = np.random.default_rng(seed).permutation(len(idx))
                tr, va = perm[:cut], perm[cut:]
                tr_dl, va_dl = get_dataloaders(Xi[tr], yi[tr], Xi[va], yi[va],
                                               NN_BATCH_SIZE,
                                               pin_memory=DEVICE.type == 'cuda')
                loaders.append((tr_dl, va_dl, yi[tr], yi[va]))
                n_samples.append(len(tr))

            r = run_federated('FedAvg', fedavg_aggregate, loaders, n_samples,
                              seed=seed, save_weights=False, verbose=False)
            m = fl_core.new_net().to(DEVICE)
            from data_utils import set_params_from_numpy
            set_params_from_numpy(m, r['global_params'])
            fl_core.set_buffers_from_numpy(m, r['global_buffers'])
            p = score(m, X_b)

            aucs = {k: float(roc_auc_score(y_b[mk], p[mk])) for k, mk in masks.items()}
            rec = {'partition': pname, 'seed': seed, 'heterogeneity': het[pname],
                   'auc_external': float(roc_auc_score(y_b, p)),
                   **{f'auc_{k}': round(v, 5) for k, v in aucs.items()},
                   'race_gap': round(aucs['White'] - aucs['Black'], 5),
                   'race_spread': round(max(aucs.values()) - min(aucs.values()), 5)}
            records.append(rec)
            print(f"  {pname:<9} seed {seed}: ext={rec['auc_external']:.4f} "
                  f"W-B={rec['race_gap']:+.4f} spread={rec['race_spread']:.4f}")

    # Reference: the composition-matched centralised control, already computed.
    dd = json.load(open(os.path.join(RESULTS_DIR, 'delong_deltas.json')))
    ref = [r for r in dd['absolute_per_seed'] if r['model'] == 'Centralised-matched']
    ref_gap = float(np.mean([r['race_gap'] for r in ref]))

    print(f"\n{'='*72}\n  matched centralised reference W-B gap = {ref_gap:+.4f}\n{'='*72}")
    print(f"  {'partition':<10}{'heterog.':>10}{'W-B gap':>10}{'d vs ref':>10}{'spread':>10}")
    summary = {}
    for pname in sorted(het, key=het.get):
        rs = [r for r in records if r['partition'] == pname]
        g = float(np.mean([r['race_gap'] for r in rs]))
        sp = float(np.mean([r['race_spread'] for r in rs]))
        summary[pname] = {'heterogeneity': round(het[pname], 4),
                          'race_gap_mean': round(g, 5),
                          'race_gap_sd': round(float(np.std([r['race_gap'] for r in rs], ddof=1)), 5),
                          'delta_vs_matched': round(g - ref_gap, 5),
                          'race_spread_mean': round(sp, 5),
                          'n_seeds': len(rs)}
        print(f"  {pname:<10}{het[pname]:>10.4f}{g:>+10.4f}{g - ref_gap:>+10.4f}{sp:>10.4f}")

    hs = [summary[p]['heterogeneity'] for p in summary]
    ds = [summary[p]['delta_vs_matched'] for p in summary]
    corr = float(np.corrcoef(hs, ds)[0, 1])
    print(f"\n  Pearson r(heterogeneity, delta W-B gap) = {corr:+.3f}")
    print("  Mechanism predicts a strong NEGATIVE r: more racial separation -> more gap closure.")

    out = os.path.join(RESULTS_DIR, 'partition_sweep.json')
    with open(out, 'w') as f:
        json.dump({'n_seeds': args.seeds, 'cohort_n': int(len(df)),
                   'reference_race_gap_matched_centralised': round(ref_gap, 5),
                   'heterogeneity_metric': 'mean total-variation distance between '
                                           'each node race distribution and pooled',
                   'summary': summary, 'runs': records,
                   'pearson_r_heterogeneity_vs_gap_delta': round(corr, 4)}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
