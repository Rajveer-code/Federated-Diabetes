"""
59_rebuttal_threshold.py — threshold-level analyses requested in the ML4H reviews
================================================================================
Reads the cached per-seed BRFSS scores written by 33_multiseed_all_models.py and
answers five reviewer questions on the same 1,282,897-row external cohort:

  A  boot    Paired bootstrap for the allocation deltas. Each replicate draws
             Poisson(1) row weights shared by every model and seed (paired), RE-
             ESTIMATES each model-seed's pooled top-q cutoff under those weights,
             and resamples the 10 seeds -> intervals that carry test-sampling,
             threshold-estimation and optimisation noise together. (QUmG, mKuf, WxxW)
  B  xroc    Exact link between cross-group ranking and sensitivity at capacity:
             a single pooled cutoff t gives sens_g(t) = xROC_{g->W}(FPR_W(t)), the
             cross-group ROC of g-positives against White non-cases read at the
             operating point. Delta sens_g splits exactly into a curve shift at the
             operating point plus an operating-point shift. (WxxW W1, mKuf W2)
  C  local   Site-local capacity: every US state (and every BRFSS urban/rural
             stratum) flags its own top q instead of one national cutoff. (QUmG)
  D  wtd     Survey-weighted readout with BRFSS _LLCPWT: weighted cutoff, weighted
             sensitivity/selection, weighted within-stratum AUC. (QUmG, WxxW)
  E  ties    Exact top-k with random tie-breaking instead of flag-all-ties. (WxxW Q1)

Output: <RESULTS_DIR>/rebuttal_threshold.json
Usage:  python 59_rebuttal_threshold.py [--boot 2000]
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, BRFSS_PATH

PRED_DIR = os.path.join(RESULTS_DIR, 'preds_multiseed')
REF = 'Centralised-matched'
STRATEGIES = ['FedAvg', 'FedProx', 'FedNova', 'FedAvg-EqualW', 'SCAFFOLD']
SEEDS = list(range(42, 52))
RACE_STRATA = {'White': [1.0], 'Black': [2.0], 'Hispanic': [8.0],
               'Asian': [4.0], 'Other': [3.0, 5.0, 6.0, 7.0]}
G = list(RACE_STRATA)
QS = [0.10, 0.20, 0.30]
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load(name):
    return np.load(os.path.join(PRED_DIR, f'{name}.npy'))


def seed_ci(x):
    x = np.asarray(x, float)
    m, h = x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {'mean': float(m), 'lo': float(m - h), 'hi': float(m + h),
            'n_pos': int((x > 0).sum()), 'n': int(len(x))}


def alloc(p, y, masks, q, w=None, flag=None):
    """Per-stratum selection rate and sensitivity under one pooled top-q rule."""
    w = np.ones_like(p) if w is None else w
    if flag is None:
        o = np.argsort(-p, kind='stable')
        cw = np.cumsum(w[o])
        thr = p[o][min(np.searchsorted(cw, q * cw[-1]), len(p) - 1)]
        flag = p >= thr
    out = {}
    for g, mk in masks.items():
        wp, wa = w * (mk & (y == 1)), w * mk
        out[g] = {'sel': float((wa * flag).sum() / wa.sum()),
                  'sens': float((wp * flag).sum() / wp.sum())}
    return out


def summarise(per_seed):
    """per_seed: list over seeds of {model: alloc-dict}; Delta = strategy - REF."""
    res = {}
    for m in STRATEGIES:
        d = {f'{k}_{g}': [] for g in G for k in ('sel', 'sens')}
        d['sens_gap_WB'] = []
        for r in per_seed:
            for g in G:
                for k in ('sel', 'sens'):
                    d[f'{k}_{g}'].append(r[m][g][k] - r[REF][g][k])
            d['sens_gap_WB'].append((r[m]['White']['sens'] - r[m]['Black']['sens'])
                                    - (r[REF]['White']['sens'] - r[REF]['Black']['sens']))
        res[m] = {k: seed_ci(v) for k, v in d.items()}
    return res


# ── A: paired bootstrap with cutoff re-estimation ────────────────────────────
def boot(scores, y, masks, n_boot, q, rng_seed=0):
    gen = torch.Generator(device=DEV).manual_seed(rng_seed)
    yt = torch.tensor(y, device=DEV)
    ind = torch.stack([torch.tensor(masks[g], device=DEV) & (yt == 1) for g in G]
                      + [torch.tensor(masks[g], device=DEV) for g in G]).float()
    models = STRATEGIES + [REF]
    order = {(m, s): torch.argsort(torch.tensor(scores[(m, s)], device=DEV), descending=True)
             for m in models for s in SEEDS}
    n, S = len(y), len(SEEDS)
    keys = ['sens_Black', 'sel_Black', 'sens_gap_WB', 'cases100k_Black']
    draws = {m: {k: [] for k in keys} for m in STRATEGIES}
    prev_b = y[masks['Black']].mean()
    for b in range(n_boot):
        w = torch.poisson(torch.ones(n, device=DEV), generator=gen)
        seed_draw = torch.randint(0, S, (S,), generator=gen, device=DEV).cpu().numpy()
        val = {}
        for (m, s), o in order.items():
            wo = w[o]
            cw = torch.cumsum(wo, 0)
            k = int(torch.searchsorted(cw, q * cw[-1]))
            num = (ind[:, o][:, :k + 1] * wo[:k + 1]).sum(1)
            den = (ind * w).sum(1)
            r = (num / den).cpu().numpy()
            val[(m, s)] = {'sens': dict(zip(G, r[:5])), 'sel': dict(zip(G, r[5:]))}
        for m in STRATEGIES:
            ds = [val[(m, SEEDS[i])]['sens']['Black'] - val[(REF, SEEDS[i])]['sens']['Black'] for i in seed_draw]
            dl = [val[(m, SEEDS[i])]['sel']['Black'] - val[(REF, SEEDS[i])]['sel']['Black'] for i in seed_draw]
            dg = [(val[(m, SEEDS[i])]['sens']['White'] - val[(m, SEEDS[i])]['sens']['Black'])
                  - (val[(REF, SEEDS[i])]['sens']['White'] - val[(REF, SEEDS[i])]['sens']['Black'])
                  for i in seed_draw]
            draws[m]['sens_Black'].append(np.mean(ds))
            draws[m]['sel_Black'].append(np.mean(dl))
            draws[m]['sens_gap_WB'].append(np.mean(dg))
            draws[m]['cases100k_Black'].append(np.mean(ds) * prev_b * 1e5)
        if (b + 1) % 200 == 0:
            print(f'    boot {b + 1}/{n_boot}', flush=True)
    return {m: {k: {'lo': float(np.percentile(v, 2.5)), 'hi': float(np.percentile(v, 97.5)),
                    'p_two_sided': float(min(1.0, 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))))}
                for k, v in d.items()} for m, d in draws.items()}


# ── B: exact xROC decomposition ──────────────────────────────────────────────
def xroc_point(p, y, masks, g, f):
    """Sensitivity of g-positives at the cutoff where White non-cases' FPR = f."""
    wn = np.sort(p[masks['White'] & (y == 0)])
    t = wn[min(int(np.floor(len(wn) * (1 - f))), len(wn) - 1)]
    return float((p[masks[g] & (y == 1)] >= t).mean())


def xroc(scores, y, masks, q):
    out = {m: {g: {'total': [], 'curve_shift': [], 'op_shift': []} for g in G if g != 'White'}
           for m in STRATEGIES}
    for s in SEEDS:
        fpr = {}
        for m in STRATEGIES + [REF]:
            p = scores[(m, s)]
            thr = np.sort(p)[int(len(p) * (1 - q))]
            fpr[m] = float((p[masks['White'] & (y == 0)] >= thr).mean())
        for m in STRATEGIES:
            for g in out[m]:
                a = xroc_point(scores[(m, s)], y, masks, g, fpr[m])
                c_at_f = xroc_point(scores[(REF, s)], y, masks, g, fpr[m])
                c = xroc_point(scores[(REF, s)], y, masks, g, fpr[REF])
                out[m][g]['total'].append(a - c)
                out[m][g]['curve_shift'].append(a - c_at_f)
                out[m][g]['op_shift'].append(c_at_f - c)
    return {m: {g: {k: seed_ci(v) for k, v in d.items()} for g, d in mm.items()}
            for m, mm in out.items()}


def weighted_auc(p, y, w):
    pos, neg = y == 1, y == 0
    o = np.argsort(p[neg])
    sn, cwn = p[neg][o], np.cumsum(w[neg][o])
    idx = np.searchsorted(sn, p[pos], side='left')
    below = np.where(idx > 0, cwn[np.maximum(idx - 1, 0)], 0.0)
    return float((w[pos] * below).sum() / (w[pos].sum() * cwn[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    y = load('_y_brfss').astype(int)
    race = load('_race_brfss')
    masks = {g: np.isin(race, v) for g, v in RACE_STRATA.items()}
    design = pd.read_csv(BRFSS_PATH, usecols=['Diabetes_Outcome', '_LLCPWT', '_STATE', '_URBSTAT'])
    design = design.dropna(subset=['Diabetes_Outcome']).reset_index(drop=True)
    assert len(design) == len(y) and (design['Diabetes_Outcome'].values == y).all()
    scores = {(m, s): load(f'{m}_seed{s}_external') for m in STRATEGIES + [REF] for s in SEEDS}
    out = {'n': int(len(y)), 'seeds': SEEDS, 'reference': REF,
           'race_known_n': int(sum(m.sum() for m in masks.values())),
           'strata_n': {g: int(m.sum()) for g, m in masks.items()},
           'strata_prev': {g: float(y[m].mean()) for g, m in masks.items()}}

    print('  [B] xROC decomposition', flush=True)
    out['xroc'] = {str(q): xroc(scores, y, masks, q) for q in QS}

    print('  [C] local thresholds', flush=True)
    out['local'] = {}
    for name, col in [('state', '_STATE'), ('urbanicity', '_URBSTAT')]:
        grp = design[col].fillna(-1).values
        per_q = {}
        for q in QS:
            per_seed = []
            for s in SEEDS:
                r = {}
                for m in STRATEGIES + [REF]:
                    p = scores[(m, s)]
                    flag = np.zeros(len(p), bool)
                    for v in np.unique(grp):
                        ix = np.flatnonzero(grp == v)
                        thr = np.sort(p[ix])[int(len(ix) * (1 - q))]
                        flag[ix] = p[ix] >= thr
                    r[m] = alloc(p, y, masks, q, flag=flag)
                per_seed.append(r)
            per_q[str(q)] = summarise(per_seed)
        out['local'][name] = {'n_groups': int(len(np.unique(grp))), 'by_q': per_q}

    print('  [D] survey-weighted', flush=True)
    w = design['_LLCPWT'].values / 3.0          # three pooled survey years
    out['weighted'] = {'by_q': {}, 'auc': {}}
    for q in QS:
        out['weighted']['by_q'][str(q)] = summarise(
            [{m: alloc(scores[(m, s)], y, masks, q, w=w) for m in STRATEGIES + [REF]} for s in SEEDS])
    for m in STRATEGIES:
        out['weighted']['auc'][m] = {g: seed_ci([
            weighted_auc(scores[(m, s)][masks[g]], y[masks[g]], w[masks[g]])
            - weighted_auc(scores[(REF, s)][masks[g]], y[masks[g]], w[masks[g]]) for s in SEEDS]) for g in G}
        out['weighted']['auc'][m]['gap_WB'] = seed_ci([
            (weighted_auc(scores[(m, s)][masks['White']], y[masks['White']], w[masks['White']])
             - weighted_auc(scores[(m, s)][masks['Black']], y[masks['Black']], w[masks['Black']]))
            - (weighted_auc(scores[(REF, s)][masks['White']], y[masks['White']], w[masks['White']])
               - weighted_auc(scores[(REF, s)][masks['Black']], y[masks['Black']], w[masks['Black']]))
            for s in SEEDS])

    print('  [E] random tie-break, exact top-k', flush=True)
    rng = np.random.default_rng(0)
    out['ties'] = {}
    for q in QS:
        per_seed = []
        for s in SEEDS:
            r = {}
            for m in STRATEGIES + [REF]:
                p = scores[(m, s)]
                o = np.lexsort((rng.random(len(p)), -p))
                flag = np.zeros(len(p), bool)
                flag[o[:int(q * len(p))]] = True
                r[m] = alloc(p, y, masks, q, flag=flag)
            per_seed.append(r)
        out['ties'][str(q)] = summarise(per_seed)
    out['global_flag_ties'] = {str(q): summarise(
        [{m: alloc(scores[(m, s)], y, masks, q) for m in STRATEGIES + [REF]} for s in SEEDS]) for q in QS}

    print(f'  [A] paired bootstrap, B={args.boot}', flush=True)
    out['boot'] = {str(q): boot(scores, y, masks, args.boot, q) for q in [0.20]}

    path = os.path.join(RESULTS_DIR, 'rebuttal_threshold.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'  Saved {path}')


if __name__ == '__main__':
    main()
