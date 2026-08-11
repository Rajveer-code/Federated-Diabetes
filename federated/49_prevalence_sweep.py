"""
49_prevalence_sweep.py — manipulate the proposed driver directly
================================================================
46_partition_mechanism.py found that the reversal tracks how much the nodes
differ in outcome prevalence (r = -0.95) rather than how racially separated they
are (r = +0.29, wrong sign). With four naturally occurring partitions that is a
descriptive correlation, and a reviewer is right to say so: BMI tertiles differ
in prevalence *because* BMI predicts diabetes, so prevalence and covariate shift
move together and neither can be credited.

This script removes the confound by construction. Nodes are built by drawing
positives and negatives independently from the pooled positive and negative
pools, so

    P(X | Y=1)  and  P(X | Y=0)  are identical at every node,
    only P(Y) differs.

That is prior (label) shift in the exact technical sense, with covariate shift
held at zero by design rather than measured and hoped small. The dose is the
spread pi_max - pi_min; everything else is fixed:

  * the same 11,906 rows in every arm -- every positive and every negative is
    allocated exactly once, so the pooled cohort, its size, and its overall
    prevalence are invariant, and the composition-matched centralised control
    from 33_multiseed_all_models.py remains the correct reference for all arms;
  * three clients, equal node sizes, same model, same optimiser, same rounds,
    same local epochs, same external test set.

The zero-spread arm is a second control worth naming: three i.i.d. clients with
identical P(Y) and identical P(X|Y). It runs the full federated procedure --
same local SGD, same number of aggregation rounds -- with the heterogeneity
removed. If the effect is present there it is a property of local optimisation;
if it appears only as the spread grows it is a property of heterogeneity.

Prediction: |Delta White-Black gap| and the widening of the White-Black
sensitivity gap both increase monotonically with the prevalence spread, and are
~0 at spread 0.

Output: results/prevalence_sweep.json

Usage:  python 49_prevalence_sweep.py [--seeds 5] [--spreads 0 0.07 0.14 0.21 0.28 0.35]
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
from data_utils import DiabetesDataset, get_dataloaders, load_brfss_clean, set_params_from_numpy
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


def prevalence_partition(y, spread, rng, n_nodes=3):
    """
    Allocate every row to one of `n_nodes` equal-sized nodes so that node
    prevalences are centred on the pooled prevalence and span `spread`.

    Positives and negatives are drawn from their own shuffled pools, which is
    what makes P(X|Y) identical across nodes: within a label, node membership is
    independent of the features. Node sizes are equal; the last node absorbs the
    rounding remainder, so the achieved spread is reported rather than assumed.
    """
    pos = rng.permutation(np.flatnonzero(y == 1))
    neg = rng.permutation(np.flatnonzero(y == 0))
    n = len(y)
    base = n // n_nodes
    sizes = [base] * n_nodes
    sizes[-1] += n - base * n_nodes

    p_bar = len(pos) / n
    # A symmetric spread is only feasible while both ends stay in [0, 1]; asking
    # for more would clip one end, pushing the mean off p_bar and silently
    # changing the pooled prevalence the arms are supposed to share.
    spread = min(spread, 2.0 * min(p_bar, 1.0 - p_bar))
    targets = p_bar + np.linspace(-spread / 2, spread / 2, n_nodes)

    want = [int(np.clip(round(sizes[k] * targets[k]), 0, sizes[k]))
            for k in range(n_nodes)]
    # Make the allocation exact: every positive and every negative is placed once.
    # Rounding residual goes to the highest-target nodes (or is taken from the
    # lowest-target ones), which preserves the intended ordering of prevalences.
    diff = len(pos) - sum(want)
    order = np.argsort(-np.asarray(targets)) if diff > 0 else np.argsort(targets)
    for k in order:
        if diff == 0:
            break
        room = (sizes[k] - want[k]) if diff > 0 else want[k]
        step = int(np.sign(diff)) * int(min(abs(diff), room))
        want[k] += step
        diff -= step
    assert diff == 0 and sum(want) == len(pos), (want, len(pos))

    idx, pi, pc = [], [], 0
    nc = 0
    for k in range(n_nodes):
        n_p, n_n = want[k], sizes[k] - want[k]
        node = np.concatenate([pos[pc:pc + n_p], neg[nc:nc + n_n]])
        pc += n_p; nc += n_n
        idx.append(rng.permutation(node))
        pi.append(n_p / sizes[k])
    assert pc == len(pos) and nc == len(neg), (pc, len(pos), nc, len(neg))
    return idx, pi


def selfcheck():
    """Partition must be a true partition and must hit the requested spread."""
    rng = np.random.default_rng(0)
    for n, p_bar in ((11906, 0.19), (999, 0.05), (1000, 0.5)):
        y = (rng.random(n) < p_bar).astype(float)
        for spread in (0.0, 0.1, 0.25, 2 * y.mean()):
            idx, pi = prevalence_partition(y, spread, np.random.default_rng(1))
            flat = np.concatenate(idx)
            assert len(flat) == n and len(np.unique(flat)) == n, 'not a partition'
            assert abs(np.concatenate([y[i] for i in idx]).mean() - y.mean()) < 1e-12
            feasible = min(spread, 2 * min(y.mean(), 1 - y.mean()))
            got = max(pi) - min(pi)
            assert abs(got - feasible) < 0.02, (spread, feasible, got, pi)
    print('selfcheck OK: exact partition, pooled prevalence invariant, spread hit')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--spreads', type=float, nargs='+',
                    default=[0.0, 0.07, 0.14, 0.21, 0.28, 0.35])
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()

    scaler = joblib.load(SCALER_PATH)
    frames = []
    for i, p in enumerate(NODE_PATHS):
        d = pd.read_csv(p); d['_node'] = i; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    X = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    y = df[TARGET_COL].values.astype(np.float32)
    print(f"Federated cohort: n={len(df):,}  pooled prevalence={y.mean():.4f}")

    df_b, X_b_raw, y_b = load_brfss_clean(verbose=False, feature_cols=FEATURE_COLS)
    X_b = scaler.transform(X_b_raw)
    race_b = df_b['RIDRETH3_RAW'].values
    masks = {k: np.isin(race_b, v) for k, v in RACE_STRATA.items()}
    # Cross-group ranking cell: Black cases against White non-cases (see 47_xauc.py).
    bw_idx = np.concatenate([np.flatnonzero(masks['Black'] & (y_b == 1)),
                             np.flatnonzero(masks['White'] & (y_b == 0))])
    bw_lab = np.concatenate([np.ones(int((masks['Black'] & (y_b == 1)).sum())),
                             np.zeros(int((masks['White'] & (y_b == 0)).sum()))])

    records = []
    for spread in args.spreads:
        for i in range(args.seeds):
            seed = SEED + i
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            idx_list, pis = prevalence_partition(y, spread, rng)
            achieved = float(max(pis) - min(pis))

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

            rec = {'spread_target': spread, 'spread_achieved': round(achieved, 4),
                   'node_prevalence': [round(v, 4) for v in pis],
                   'node_sizes': [int(len(a)) for a in idx_list],
                   'seed': seed,
                   'auc_external': round(float(roc_auc_score(y_b, p)), 5),
                   **{f'auc_{k}': round(v, 5) for k, v in aucs.items()},
                   **{f'sens_{k}': round(v, 5) for k, v in sens.items()},
                   'race_gap': round(aucs['White'] - aucs['Black'], 5),
                   'sens_gap': round(sens['White'] - sens['Black'], 5),
                   'xauc_black_white': round(float(roc_auc_score(bw_lab, p[bw_idx])), 5)}
            records.append(rec)
            print(f"  spread {spread:.2f} (got {achieved:.3f}) seed {seed}: "
                  f"pi={[f'{v:.3f}' for v in pis]} ext={rec['auc_external']:.4f} "
                  f"W-B AUC={rec['race_gap']:+.4f} W-B sens={rec['sens_gap']:+.4f} "
                  f"xAUC(B>W)={rec['xauc_black_white']:.4f}")

    # ── reference: the composition-matched centralised control ────────────────
    dd = json.load(open(os.path.join(RESULTS_DIR, 'delong_deltas.json')))
    ref_rows = [r for r in dd['absolute_per_seed'] if r['model'] == 'Centralised-matched']
    ref_gap = float(np.mean([r['race_gap'] for r in ref_rows]))
    ct = json.load(open(os.path.join(RESULTS_DIR, 'clinical_translation.json')))
    ref_sens_gap = ct['by_rate'][f'{FLAG_RATE:.2f}']['_white_black_tpr_gap']['reference']

    summary = {}
    for spread in args.spreads:
        rs = [r for r in records if r['spread_target'] == spread]
        def agg(key):
            v = [r[key] for r in rs]
            return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        g, g_sd = agg('race_gap')
        s, s_sd = agg('sens_gap')
        x, x_sd = agg('xauc_black_white')
        summary[f'{spread:.2f}'] = {
            'spread_target': spread,
            'spread_achieved': round(float(np.mean([r['spread_achieved'] for r in rs])), 4),
            'n_seeds': len(rs),
            'race_gap_mean': round(g, 5), 'race_gap_sd': round(g_sd, 5),
            'delta_race_gap_vs_matched': round(g - ref_gap, 5),
            'sens_gap_mean': round(s, 5), 'sens_gap_sd': round(s_sd, 5),
            'delta_sens_gap_vs_matched': round(s - ref_sens_gap, 5),
            'xauc_black_white_mean': round(x, 5), 'xauc_black_white_sd': round(x_sd, 5),
        }

    xs = [summary[k]['spread_achieved'] for k in summary]
    r_auc = float(np.corrcoef(xs, [summary[k]['delta_race_gap_vs_matched'] for k in summary])[0, 1])
    r_sens = float(np.corrcoef(xs, [summary[k]['delta_sens_gap_vs_matched'] for k in summary])[0, 1])
    r_x = float(np.corrcoef(xs, [summary[k]['xauc_black_white_mean'] for k in summary])[0, 1])

    print(f"\n{'='*94}\n  Controlled prevalence sweep — P(X|Y) identical across nodes "
          f"by construction\n  reference (matched centralised): W-B AUC gap "
          f"{ref_gap:+.4f}, W-B sensitivity gap {ref_sens_gap:+.4f}\n{'='*94}")
    print(f"  {'spread':>8}{'W-B AUC':>10}{'d vs ref':>10}"
          f"{'W-B sens':>10}{'d vs ref':>10}{'xAUC B>W':>11}")
    for k in summary:
        s = summary[k]
        print(f"  {s['spread_achieved']:>8.3f}{s['race_gap_mean']:>+10.4f}"
              f"{s['delta_race_gap_vs_matched']:>+10.4f}"
              f"{s['sens_gap_mean']:>+10.4f}{s['delta_sens_gap_vs_matched']:>+10.4f}"
              f"{s['xauc_black_white_mean']:>11.4f}")
    print(f"\n  Pearson r vs achieved prevalence spread ({len(summary)} arms):")
    print(f"    delta W-B AUC gap         r = {r_auc:+.3f}   (negative = AUC gap closes more)")
    print(f"    delta W-B sensitivity gap r = {r_sens:+.3f}   (positive = detection gap widens more)")
    print(f"    xAUC(Black+ > White-)     r = {r_x:+.3f}   (negative = Black cases pushed down)")

    out = os.path.join(RESULTS_DIR, 'prevalence_sweep.json')
    with open(out, 'w') as f:
        json.dump({
            'design': 'Nodes drawn from the pooled positive and negative pools '
                      'separately, so P(X|Y=1) and P(X|Y=0) are identical across '
                      'nodes and only P(Y) varies. Every row is used exactly once '
                      'in every arm, so cohort size and pooled prevalence are '
                      'invariant and the composition-matched centralised control '
                      'is the correct reference throughout.',
            'controls_held_fixed': ['cohort rows', 'cohort size', 'pooled prevalence',
                                    'n_clients=3', 'equal node sizes', 'model',
                                    'optimiser', 'rounds', 'local epochs',
                                    'external test set', 'P(X|Y)'],
            'flag_rate': FLAG_RATE, 'n_seeds': args.seeds,
            'cohort_n': int(len(df)), 'pooled_prevalence': round(float(y.mean()), 5),
            'reference_race_gap_matched_centralised': round(ref_gap, 5),
            'reference_sens_gap_matched_centralised': round(ref_sens_gap, 5),
            'pearson_r_spread_vs_delta_auc_gap': round(r_auc, 4),
            'pearson_r_spread_vs_delta_sens_gap': round(r_sens, 4),
            'pearson_r_spread_vs_xauc_black_white': round(r_x, 4),
            'summary': summary, 'runs': records}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
