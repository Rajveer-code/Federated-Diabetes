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


def real_partition(node_id, y, per_node, rng, n_nodes=3):
    """
    The actual age/urbanicity partition, subsampled to `per_node` rows per node
    while preserving each node's OWN prevalence.

    This is the size control. The covariate sweep runs on a cohort roughly half
    the size of the main experiment, because forcing equal prevalence costs rows.
    If the detection reversal also disappears for the real partition at this size,
    the sweep's null on detection is a power artefact rather than evidence that
    covariate heterogeneity fails to propagate to the threshold. Run as its own
    arm so the comparison is against the same alpha=0 reference at the same size.
    """
    idx = []
    for k in range(n_nodes):
        rows = np.flatnonzero(node_id == k)
        p_k = float(y[rows].mean())
        n_p = int(round(per_node * p_k))
        pos = rng.permutation(rows[y[rows] == 1])
        neg = rng.permutation(rows[y[rows] == 0])
        if len(pos) < n_p or len(neg) < per_node - n_p:
            return None
        idx.append(rng.permutation(np.concatenate([pos[:n_p], neg[:per_node - n_p]])))
    return idx


def cov_separation(Xz, idx_list):
    """Mean |node mean - pooled mean| over standardised features: the achieved dose."""
    return float(np.mean([np.abs(Xz[i].mean(0)).mean() for i in idx_list]))


def allocation(p, y_b, masks, rate):
    """Selection rate, sensitivity and PPV per stratum at one global threshold."""
    thr = np.quantile(p, 1 - rate)
    flag = p >= thr
    out = {}
    for k, m in masks.items():
        yk, fk = y_b[m], flag[m]
        n_flag = int(fk.sum())
        out[k] = {'sel': float(fk.mean()),
                  'sens': float(fk[yk == 1].mean()),
                  'ppv': float(yk[fk].mean()) if n_flag else float('nan')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--alphas', type=float, nargs='+',
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--per-node', type=int, default=0,
                    help='rows per node; 0 = pick the largest feasible value')
    ap.add_argument('--no-size-control', action='store_true',
                    help='skip the real-partition arm (alpha=-1), the size control')
    ap.add_argument('--covariate', default='BMXBMI',
                    help='feature whose rank defines the separation direction. '
                         'The real partition is defined on age and urbanicity, so '
                         'RIDAGEYR tests whether the direction of separation matters '
                         'and not only its magnitude.')
    ap.add_argument('--out', default='covariate_sweep.json')
    args = ap.parse_args()
    REAL = -1.0
    arms = list(args.alphas) + ([] if args.no_size_control else [REAL])

    scaler = joblib.load(SCALER_PATH)
    frames = []
    for i, p in enumerate(NODE_PATHS):
        d = pd.read_csv(p); d['_node'] = i; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    X = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    y = df[TARGET_COL].values.astype(np.float32)
    cov = df[args.covariate].values.astype(float)
    node_id = df['_node'].values
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
            # Every arm must fit, including the real-partition size control --
            # its smallest node (elderly-rural, n=3,406) binds before the dose
            # arms do at large sizes.
            ok = all((real_partition(node_id, y, mid, np.random.default_rng(SEED))
                      if a == REAL
                      else covariate_partition(cov, y, a, mid, np.random.default_rng(SEED)))
                     is not None for a in arms)
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
    for alpha in arms:
        for i in range(args.seeds):
            seed = SEED + i
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            idx_list = (real_partition(node_id, y, per, rng) if alpha == REAL
                        else covariate_partition(cov, y, alpha, per, rng))
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
            alloc = allocation(p, y_b, masks, FLAG_RATE)
            sens = {k: v['sens'] for k, v in alloc.items()}

            rec = {'alpha': alpha, 'seed': seed,
                   **{f'sel_{k}': round(v['sel'], 5) for k, v in alloc.items()},
                   **{f'ppv_{k}': round(v['ppv'], 5) for k, v in alloc.items()},
                   'sel_gap': round(alloc['White']['sel'] - alloc['Black']['sel'], 5),
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

    b_sel = float(np.mean([r['sel_gap'] for r in base]))

    summary = {}
    for alpha in arms:
        rs = [r for r in records if r['alpha'] == alpha]
        if not rs:
            continue
        def m(key):
            v = [r[key] for r in rs]
            return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        g, g_sd = m('race_gap'); s, s_sd = m('sens_gap')
        x, x_sd = m('xauc_black_white'); q, q_sd = m('sel_gap')
        key = 'real' if alpha == REAL else f'{alpha:.2f}'
        summary[key] = {
            'alpha': alpha, 'n_seeds': len(rs),
            'is_size_control': alpha == REAL,
            'cov_separation': round(float(np.mean([r['cov_separation'] for r in rs])), 4),
            'race_gap_mean': round(g, 5), 'race_gap_sd': round(g_sd, 5),
            'sens_gap_mean': round(s, 5), 'sens_gap_sd': round(s_sd, 5),
            'sel_gap_mean': round(q, 5), 'sel_gap_sd': round(q_sd, 5),
            'xauc_black_white_mean': round(x, 5), 'xauc_black_white_sd': round(x_sd, 5),
            'delta_race_gap_vs_alpha0': round(g - b_gap, 5),
            'delta_sens_gap_vs_alpha0': round(s - b_sens, 5),
            'delta_sel_gap_vs_alpha0': round(q - b_sel, 5),
            'delta_xauc_vs_alpha0': round(x - b_xauc, 5),
        }

    # Correlations are over the DOSE arms only; the real-partition arm is a
    # separate control and is not a point on the covariate-separation scale.
    ks = [k for k in summary if not summary[k]['is_size_control']]
    xs = [summary[k]['cov_separation'] for k in ks]
    def corr(field):
        return round(float(np.corrcoef(xs, [summary[k][field] for k in ks])[0, 1]), 4)
    r_sens, r_auc, r_x = (corr('delta_sens_gap_vs_alpha0'),
                          corr('delta_race_gap_vs_alpha0'),
                          corr('delta_xauc_vs_alpha0'))

    print(f"\n{'='*96}\n  Covariate separation at FIXED prevalence "
          f"(reference = alpha 0 arm)\n{'='*96}")
    print(f"  {'arm':>6}{'cov sep':>10}{'W-B AUC':>10}{'d vs a0':>10}"
          f"{'W-B sens':>10}{'d vs a0':>10}{'W-B sel':>10}{'d vs a0':>10}"
          f"{'xAUC B>W':>11}{'d vs a0':>10}")
    for k in list(summary):
        s = summary[k]
        lab = 'real' if s['is_size_control'] else f"{s['alpha']:.2f}"
        print(f"  {lab:>6}{s['cov_separation']:>10.4f}"
              f"{s['race_gap_mean']:>+10.4f}{s['delta_race_gap_vs_alpha0']:>+10.4f}"
              f"{s['sens_gap_mean']:>+10.4f}{s['delta_sens_gap_vs_alpha0']:>+10.4f}"
              f"{s['sel_gap_mean']:>+10.4f}{s['delta_sel_gap_vs_alpha0']:>+10.4f}"
              f"{s['xauc_black_white_mean']:>11.4f}{s['delta_xauc_vs_alpha0']:>+10.4f}")
    print(f"\n  Pearson r vs achieved covariate separation ({len(ks)} dose arms):")
    print(f"    delta W-B sensitivity gap  r = {r_sens:+.3f}  (positive = gap widens)")
    print(f"    delta W-B AUC gap          r = {r_auc:+.3f}  (negative = AUC gap closes)")
    print(f"    delta xAUC(Black+>White-)  r = {r_x:+.3f}  (negative = cases pushed down)")
    if 'real' in summary:
        rr = summary['real']
        print(f"\n  SIZE CONTROL — the real age/urbanicity partition at this same "
              f"cohort size ({3 * per:,}):")
        print(f"    d W-B sensitivity gap vs alpha0 = "
              f"{rr['delta_sens_gap_vs_alpha0']:+.4f}   "
              f"(full cohort, main experiment: +0.0246)")
        print("    If this is also near zero, the dose sweep's null on DETECTION is a")
        print("    power artefact of the reduced cohort, not a failure of the mechanism")
        print("    to propagate to the threshold.")

    out = os.path.join(RESULTS_DIR, args.out)
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
            'covariate': args.covariate,
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
