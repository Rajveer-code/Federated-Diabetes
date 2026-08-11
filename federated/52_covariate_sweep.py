"""
52_covariate_sweep.py — the complement: covariate separation at fixed prevalence
===============================================================================
49_prevalence_sweep.py held P(X|Y) identical and varied P(Y). The effect did not
appear: across six arms spanning a prevalence spread of 0 to 0.35, the change in
the White-Black sensitivity gap stayed within [-0.008, -0.003] against the +0.027
the real partition produces, and xAUC(Black+ > White-) never moved off 0.747.
Pure label shift is not the driver, so the r=-0.95 across the four natural
partitions of 46_partition_mechanism.py was confounded -- BMI tertiles separate
the nodes on covariates and on prevalence at the same time.

This is the other half of the design. Nodes are separated on COVARIATES by a
tunable amount while every node is forced to the pooled outcome prevalence, so

    P(Y) is identical at every node,
    only P(X) differs.

Dose. Each row gets a mixture score u = (1-alpha)*Uniform + alpha*rank(BMI),
and nodes are the tertiles of u. alpha=0 is a random partition; alpha=1 is the
BMI-tertile partition that produced the largest natural effect. Intermediate
alpha interpolates, giving a graded dose rather than two points.

Equal prevalence is enforced by subsampling, not by moving rows between nodes:
each node contributes exactly the same number of positives and negatives, drawn
at random within its own label pools. Moving rows would have degraded the very
covariate separation being dosed. Subsampling costs cohort size, so the reference
here is the alpha=0 arm of this same experiment at the same size rather than the
full-cohort centralised control -- the comparison stays within the experiment.

Prediction, if covariate heterogeneity is the driver: the White-Black sensitivity
gap widens and xAUC(Black+ > White-) falls monotonically in alpha.

Output: results/covariate_sweep.json

Usage:  python 52_covariate_sweep.py [--seeds 10] [--alphas 0 0.25 0.5 0.75 1.0]
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
import torch
warnings.filterwarnings('ignore')
import joblib
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (NODE_PATHS, RESULTS_DIR, TARGET_COL, SEED,
                          FEATURE_COLS_NORACE as FEATURE_COLS,
                          GLOBAL_SCALER_NORACE_PATH as SCALER_PATH,
                          NN_BATCH_SIZE)
from data_utils import get_dataloaders, load_brfss_clean, set_params_from_numpy
import fl_core
from fl_core import DEVICE, seed_everything, run_federated, fedavg_aggregate

CHUNK = 65536
FLAG_RATE = 0.20
RACE_STRATA = {'White': [1.0], 'Black': [2.0], 'Hispanic': [8.0],
               'Asian': [4.0], 'Other': [3.0, 5.0, 6.0, 7.0]}


@torch.no_grad()
def score(model, X):
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(torch.FloatTensor(X[i:i + CHUNK]).to(DEVICE))).cpu().numpy()
        for i in range(0, len(X), CHUNK)])


def covariate_partition(cov, y, alpha, per_node, rng, n_nodes=3):
    """
    Nodes are tertiles of (1-alpha)*Uniform + alpha*rank(cov), then subsampled so
    each holds `per_node` rows at exactly the pooled prevalence. Returns None if
    a node cannot supply its quota, which is how infeasible `per_node` is caught.
    """
    n = len(y)
    rank = np.argsort(np.argsort(cov)) / max(n - 1, 1)
    u = (1.0 - alpha) * rng.random(n) + alpha * rank
    node = np.clip((np.argsort(np.argsort(u)) / max(n - 1, 1) * n_nodes).astype(int),
                   0, n_nodes - 1)

    p_bar = float(y.mean())
    n_pos = int(round(per_node * p_bar))
    n_neg = per_node - n_pos

    idx = []
    for k in range(n_nodes):
        pos = np.flatnonzero((node == k) & (y == 1))
        neg = np.flatnonzero((node == k) & (y == 0))
        if len(pos) < n_pos or len(neg) < n_neg:
            return None
        idx.append(rng.permutation(np.concatenate([
            rng.permutation(pos)[:n_pos], rng.permutation(neg)[:n_neg]])))
    return idx


def cov_separation(Xz, idx_list):
    """Mean |node mean - pooled mean| over standardised features: the achieved dose."""
    return float(np.mean([np.abs(Xz[i].mean(0)).mean() for i in idx_list]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--alphas', type=float, nargs='+',
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--per-node', type=int, default=0,
                    help='rows per node; 0 = pick the largest feasible value')
    args = ap.parse_args()

    scaler = joblib.load(SCALER_PATH)
    frames = []
    for i, p in enumerate(NODE_PATHS):
        d = pd.read_csv(p); d['_node'] = i; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    X = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    y = df[TARGET_COL].values.astype(np.float32)
    cov = df['BMXBMI'].values.astype(float)
    Xraw = df[FEATURE_COLS].values.astype(float)
    Xz = (Xraw - Xraw.mean(0)) / Xraw.std(0)
    print(f"Federated cohort: n={len(df):,}  pooled prevalence={y.mean():.4f}")

    # Largest per-node size feasible at the most separated arm (alpha=1), where
    # the low-BMI node is poorest in positives.
    per = args.per_node
    if per == 0:
        per = max(args.alphas and 200, 200)
        probe = np.random.default_rng(SEED)
        hi = len(y) // 3
        lo, best = 200, 200
        while lo <= hi:
            mid = (lo + hi) // 2
            ok = all(covariate_partition(cov, y, a, mid, np.random.default_rng(SEED))
                     is not None for a in args.alphas)
            if ok:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        per = int(best * 0.97)          # small margin so other seeds also fit
    print(f"per-node size: {per}  (cohort {3 * per:,} in every arm)")

    df_b, X_b_raw, y_b = load_brfss_clean(verbose=False, feature_cols=FEATURE_COLS)
    X_b = scaler.transform(X_b_raw)
    race_b = df_b['RIDRETH3_RAW'].values
    masks = {k: np.isin(race_b, v) for k, v in RACE_STRATA.items()}
    bw_idx = np.concatenate([np.flatnonzero(masks['Black'] & (y_b == 1)),
                             np.flatnonzero(masks['White'] & (y_b == 0))])
    bw_lab = np.concatenate([np.ones(int((masks['Black'] & (y_b == 1)).sum())),
                             np.zeros(int((masks['White'] & (y_b == 0)).sum()))])

    records = []
    for alpha in args.alphas:
        for i in range(args.seeds):
            seed = SEED + i
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            idx_list = covariate_partition(cov, y, alpha, per, rng)
            if idx_list is None:
                print(f"  alpha {alpha} seed {seed}: INFEASIBLE, skipped")
                continue
            sep = cov_separation(Xz, idx_list)
            prevs = [float(y[j].mean()) for j in idx_list]

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
            set_params_from_numpy(m, r['global_params'])
            fl_core.set_buffers_from_numpy(m, r['global_buffers'])
            p = score(m, X_b)

            aucs = {k: float(roc_auc_score(y_b[mk], p[mk])) for k, mk in masks.items()}
            thr = np.quantile(p, 1 - FLAG_RATE)
            flag = p >= thr
            sens = {k: float(flag[mk & (y_b == 1)].mean()) for k, mk in masks.items()}

            rec = {'alpha': alpha, 'seed': seed,
                   'cov_separation': round(sep, 4),
                   'node_prevalence': [round(v, 4) for v in prevs],
                   'auc_external': round(float(roc_auc_score(y_b, p)), 5),
                   **{f'auc_{k}': round(v, 5) for k, v in aucs.items()},
                   **{f'sens_{k}': round(v, 5) for k, v in sens.items()},
                   'race_gap': round(aucs['White'] - aucs['Black'], 5),
                   'sens_gap': round(sens['White'] - sens['Black'], 5),
                   'xauc_black_white': round(float(roc_auc_score(bw_lab, p[bw_idx])), 5)}
            records.append(rec)
            print(f"  alpha {alpha:.2f} seed {seed}: sep={sep:.4f} "
                  f"prev={[f'{v:.3f}' for v in prevs]} "
                  f"W-B AUC={rec['race_gap']:+.4f} W-B sens={rec['sens_gap']:+.4f} "
                  f"xAUC(B>W)={rec['xauc_black_white']:.4f}")

    # Reference is the alpha=0 arm of this same experiment (same cohort size).
    base = [r for r in records if r['alpha'] == args.alphas[0]]
    b_gap = float(np.mean([r['race_gap'] for r in base]))
    b_sens = float(np.mean([r['sens_gap'] for r in base]))
    b_xauc = float(np.mean([r['xauc_black_white'] for r in base]))

    summary = {}
    for alpha in args.alphas:
        rs = [r for r in records if r['alpha'] == alpha]
        if not rs:
            continue
        def m(key):
            v = [r[key] for r in rs]
            return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        g, g_sd = m('race_gap'); s, s_sd = m('sens_gap'); x, x_sd = m('xauc_black_white')
        summary[f'{alpha:.2f}'] = {
            'alpha': alpha, 'n_seeds': len(rs),
            'cov_separation': round(float(np.mean([r['cov_separation'] for r in rs])), 4),
            'race_gap_mean': round(g, 5), 'race_gap_sd': round(g_sd, 5),
            'sens_gap_mean': round(s, 5), 'sens_gap_sd': round(s_sd, 5),
            'xauc_black_white_mean': round(x, 5), 'xauc_black_white_sd': round(x_sd, 5),
            'delta_race_gap_vs_alpha0': round(g - b_gap, 5),
            'delta_sens_gap_vs_alpha0': round(s - b_sens, 5),
            'delta_xauc_vs_alpha0': round(x - b_xauc, 5),
        }

    ks = list(summary)
    xs = [summary[k]['cov_separation'] for k in ks]
    def corr(field):
        return round(float(np.corrcoef(xs, [summary[k][field] for k in ks])[0, 1]), 4)
    r_sens, r_auc, r_x = (corr('delta_sens_gap_vs_alpha0'),
                          corr('delta_race_gap_vs_alpha0'),
                          corr('delta_xauc_vs_alpha0'))

    print(f"\n{'='*96}\n  Covariate separation at FIXED prevalence "
          f"(reference = alpha 0 arm)\n{'='*96}")
    print(f"  {'alpha':>6}{'cov sep':>10}{'W-B AUC':>10}{'d vs a0':>10}"
          f"{'W-B sens':>10}{'d vs a0':>10}{'xAUC B>W':>11}{'d vs a0':>10}")
    for k in ks:
        s = summary[k]
        print(f"  {s['alpha']:>6.2f}{s['cov_separation']:>10.4f}"
              f"{s['race_gap_mean']:>+10.4f}{s['delta_race_gap_vs_alpha0']:>+10.4f}"
              f"{s['sens_gap_mean']:>+10.4f}{s['delta_sens_gap_vs_alpha0']:>+10.4f}"
              f"{s['xauc_black_white_mean']:>11.4f}{s['delta_xauc_vs_alpha0']:>+10.4f}")
    print(f"\n  Pearson r vs achieved covariate separation ({len(ks)} arms):")
    print(f"    delta W-B sensitivity gap  r = {r_sens:+.3f}  (positive = gap widens)")
    print(f"    delta W-B AUC gap          r = {r_auc:+.3f}  (negative = AUC gap closes)")
    print(f"    delta xAUC(Black+>White-)  r = {r_x:+.3f}  (negative = cases pushed down)")

    out = os.path.join(RESULTS_DIR, 'covariate_sweep.json')
    with open(out, 'w') as f:
        json.dump({
            'design': 'Nodes are tertiles of (1-alpha)*Uniform + alpha*rank(BMI), '
                      'then subsampled so every node holds the same number of '
                      'positives and negatives. P(Y) is therefore identical across '
                      'nodes and only P(X) varies. Reference is the alpha=0 arm of '
                      'this experiment, at the same cohort size.',
            'motivation': 'Complement to 49_prevalence_sweep.py, which held P(X|Y) '
                          'fixed and varied P(Y) and found no effect.',
            'controls_held_fixed': ['P(Y) per node', 'cohort size', 'n_clients=3',
                                    'equal node sizes', 'model', 'optimiser',
                                    'rounds', 'local epochs', 'external test set'],
            'flag_rate': FLAG_RATE, 'n_seeds': args.seeds, 'per_node': per,
            'cohort_n': int(3 * per),
            'pooled_prevalence': round(float(y.mean()), 5),
            'pearson_r_sep_vs_delta_sens_gap': r_sens,
            'pearson_r_sep_vs_delta_auc_gap': r_auc,
            'pearson_r_sep_vs_delta_xauc': r_x,
            'summary': summary, 'runs': records}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
