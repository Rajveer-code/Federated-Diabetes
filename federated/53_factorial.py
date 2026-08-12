"""
53_factorial.py — 2x2: does the detection reversal need BOTH heterogeneities?
============================================================================
Two controlled sweeps have each answered half a question and left the important
half open.

  49_prevalence_sweep.py   label shift alone, P(X|Y) fixed  -> nothing moves.
  52_covariate_sweep.py    covariate separation alone, P(Y) fixed ->
                           AUC gap closes (r=-0.96), xAUC(Black+ > White-) falls
                           0.024, the White-Black SELECTION gap widens 0.019 --
                           but the SENSITIVITY gap does not move (+0.002).

The obvious explanation was that the covariate sweep runs on a cohort roughly
half the size of the main experiment and simply lacks power at the threshold.
The size-control arm added to 52 rejects that: the real age/urbanicity partition,
subsampled to the same 5,916 rows, produces a sensitivity-gap change of +0.034 --
larger than the +0.025 it produces at full size. The reduced cohort has power to
spare. Covariate separation on its own genuinely does not reproduce the detection
reversal.

What distinguishes the real partition from the alpha=1 arm is that the real one
carries BOTH kinds of heterogeneity at once, while each controlled sweep isolates
one. Neither main effect explains the detection reversal, which points at an
interaction. A null main effect says nothing about an interaction, so this has to
be measured rather than argued.

Design: fully crossed, all four cells run through the same code path so the
comparison is internally consistent.

                     label spread 0        label spread s
  covariate a=0      cell(0,0) control     cell(0,1) label only
  covariate a=1      cell(1,0) covar only  cell(1,1) both

Reported per outcome, on the cell means:
  main effect of covariate  = mean(a=1) - mean(a=0)
  main effect of label      = mean(s>0) - mean(s=0)
  interaction               = [c(1,1)-c(1,0)] - [c(0,1)-c(0,0)]

Node prevalences are ordered WITH the covariate (higher-BMI node, higher
prevalence), which is the configuration that arises naturally and the one the
real partition has.

Output: results/factorial.json

Usage:  python 53_factorial.py [--seeds 10] [--spread 0.15]
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

_cs = __import__('52_covariate_sweep')
score, cov_separation, allocation = _cs.score, _cs.cov_separation, _cs.allocation
RACE_STRATA, FLAG_RATE = _cs.RACE_STRATA, _cs.FLAG_RATE


def factorial_partition(cov, y, alpha, spread, per_node, rng, n_nodes=3):
    """
    Nodes are tertiles of (1-alpha)*Uniform + alpha*rank(cov); node k is then
    subsampled to `per_node` rows at target prevalence p_bar + offset_k, where the
    offsets span `spread` and increase with the node's covariate rank.

    alpha controls how separated the nodes are in X; spread controls how separated
    they are in P(Y). Setting either to zero recovers one of the single-factor
    sweeps, which is what makes the four cells commensurable.
    """
    n = len(y)
    rank = np.argsort(np.argsort(cov)) / max(n - 1, 1)
    u = (1.0 - alpha) * rng.random(n) + alpha * rank
    node = np.clip((np.argsort(np.argsort(u)) / max(n - 1, 1) * n_nodes).astype(int),
                   0, n_nodes - 1)

    p_bar = float(y.mean())
    spread = min(spread, 2.0 * min(p_bar, 1.0 - p_bar))
    targets = p_bar + np.linspace(-spread / 2, spread / 2, n_nodes)

    idx, prevs = [], []
    for k in range(n_nodes):
        n_p = int(round(per_node * targets[k]))
        n_n = per_node - n_p
        pos = np.flatnonzero((node == k) & (y == 1))
        neg = np.flatnonzero((node == k) & (y == 0))
        if len(pos) < n_p or len(neg) < n_n:
            return None, None
        idx.append(rng.permutation(np.concatenate([
            rng.permutation(pos)[:n_p], rng.permutation(neg)[:n_n]])))
        prevs.append(n_p / per_node)
    return idx, prevs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--spread', type=float, default=0.15)
    ap.add_argument('--per-node', type=int, default=0)
    args = ap.parse_args()

    cells = [(0.0, 0.0), (0.0, args.spread), (1.0, 0.0), (1.0, args.spread)]

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

    per = args.per_node
    if per == 0:
        lo, hi, best = 200, len(y) // 3, 200
        while lo <= hi:
            mid = (lo + hi) // 2
            ok = all(factorial_partition(cov, y, a, s, mid,
                                         np.random.default_rng(SEED))[0] is not None
                     for a, s in cells)
            if ok:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        per = int(best * 0.97)
    print(f"Cohort {len(df):,}  prevalence {y.mean():.4f}  |  "
          f"per-node {per}  (cohort {3 * per:,} in every cell)")

    df_b, X_b_raw, y_b = load_brfss_clean(verbose=False, feature_cols=FEATURE_COLS)
    X_b = scaler.transform(X_b_raw)
    race_b = df_b['RIDRETH3_RAW'].values
    masks = {k: np.isin(race_b, v) for k, v in RACE_STRATA.items()}
    bw_idx = np.concatenate([np.flatnonzero(masks['Black'] & (y_b == 1)),
                             np.flatnonzero(masks['White'] & (y_b == 0))])
    bw_lab = np.concatenate([np.ones(int((masks['Black'] & (y_b == 1)).sum())),
                             np.zeros(int((masks['White'] & (y_b == 0)).sum()))])

    records = []
    for alpha, spread in cells:
        for i in range(args.seeds):
            seed = SEED + i
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            idx_list, prevs = factorial_partition(cov, y, alpha, spread, per, rng)

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
            al = allocation(p, y_b, masks, FLAG_RATE)
            rec = {'alpha': alpha, 'spread': spread, 'seed': seed,
                   'cov_separation': round(cov_separation(Xz, idx_list), 4),
                   'node_prevalence': [round(v, 4) for v in prevs],
                   'race_gap': round(aucs['White'] - aucs['Black'], 5),
                   'sens_gap': round(al['White']['sens'] - al['Black']['sens'], 5),
                   'sel_gap': round(al['White']['sel'] - al['Black']['sel'], 5),
                   'xauc_black_white': round(float(roc_auc_score(bw_lab, p[bw_idx])), 5)}
            records.append(rec)
            print(f"  a={alpha:.0f} s={spread:.2f} seed {seed}: "
                  f"sep={rec['cov_separation']:.4f} "
                  f"prev={[f'{v:.3f}' for v in prevs]} "
                  f"W-B AUC={rec['race_gap']:+.4f} sens={rec['sens_gap']:+.4f} "
                  f"sel={rec['sel_gap']:+.4f} xAUC={rec['xauc_black_white']:.4f}")

    # ── cell means, main effects, interaction ─────────────────────────────────
    OUT = ['race_gap', 'sens_gap', 'sel_gap', 'xauc_black_white']

    def cell(a, s, key):
        v = [r[key] for r in records if r['alpha'] == a and r['spread'] == s]
        return float(np.mean(v)), float(np.std(v, ddof=1)), len(v)

    s1 = args.spread
    effects, table = {}, {}
    for key in OUT:
        c00, sd00, n = cell(0.0, 0.0, key)
        c01, sd01, _ = cell(0.0, s1, key)
        c10, sd10, _ = cell(1.0, 0.0, key)
        c11, sd11, _ = cell(1.0, s1, key)
        # Each cell mean has SE = sd/sqrt(n); effects are contrasts of cell means,
        # so their variances add (the four cells use disjoint training runs).
        se = np.sqrt(sum((x / np.sqrt(n)) ** 2 for x in (sd00, sd01, sd10, sd11)))
        table[key] = {'cell_00': round(c00, 5), 'cell_01': round(c01, 5),
                      'cell_10': round(c10, 5), 'cell_11': round(c11, 5),
                      'sd': [round(x, 5) for x in (sd00, sd01, sd10, sd11)]}
        effects[key] = {
            'main_covariate': round(((c10 + c11) - (c00 + c01)) / 2, 5),
            'main_label': round(((c01 + c11) - (c00 + c10)) / 2, 5),
            'interaction': round((c11 - c10) - (c01 - c00), 5),
            'se_interaction': round(float(se), 5),
            'both_vs_neither': round(c11 - c00, 5),
        }

    print(f"\n{'='*94}\n  2x2 cell means (reference cell is a=0,s=0)\n{'='*94}")
    print(f"  {'outcome':<20}{'neither':>11}{'label':>11}{'covar':>11}"
          f"{'both':>11}{'main cov':>11}{'main lab':>11}{'inter':>10}")
    for key in OUT:
        t, e = table[key], effects[key]
        print(f"  {key:<20}{t['cell_00']:>+11.4f}{t['cell_01']:>+11.4f}"
              f"{t['cell_10']:>+11.4f}{t['cell_11']:>+11.4f}"
              f"{e['main_covariate']:>+11.4f}{e['main_label']:>+11.4f}"
              f"{e['interaction']:>+10.4f}")

    e = effects['sens_gap']
    print(f"\n  DETECTION (White-Black sensitivity gap), the quantity neither single")
    print(f"  manipulation reproduced:")
    print(f"    covariate alone      {table['sens_gap']['cell_10'] - table['sens_gap']['cell_00']:+.4f}")
    print(f"    label alone          {table['sens_gap']['cell_01'] - table['sens_gap']['cell_00']:+.4f}")
    print(f"    both                 {e['both_vs_neither']:+.4f}")
    print(f"    interaction          {e['interaction']:+.4f}  +/- {1.96 * e['se_interaction']:.4f} (95%)")

    out = os.path.join(RESULTS_DIR, 'factorial.json')
    with open(out, 'w') as f:
        json.dump({
            'design': '2x2 fully crossed: covariate separation (alpha in {0,1}) x '
                      'label spread (0 vs %.2f). Node prevalences ordered with the '
                      'covariate. Every cell subsampled to the same cohort size, '
                      'same model, optimiser, rounds, local epochs and external '
                      'test set.' % args.spread,
            'motivation': 'Neither single-factor sweep reproduced the fixed-capacity '
                          'detection reversal, and the size-control arm of '
                          '52_covariate_sweep.py ruled out lack of power. A null '
                          'main effect does not constrain an interaction, so the '
                          'interaction is measured here rather than argued.',
            'n_seeds': args.seeds, 'per_node': per, 'cohort_n': int(3 * per),
            'spread': args.spread, 'flag_rate': FLAG_RATE,
            'cells': table, 'effects': effects, 'runs': records}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
