"""
test_delong.py — correctness check for the DeLong implementation in 35_delong_deltas.py

Three assertions, in increasing strength:
  1. the AUC DeLong computes equals sklearn's roc_auc_score
  2. the DeLong variance of a single AUC matches a real bootstrap SD
  3. for two correlated models on shared rows, the paired DeLong variance of the
     difference is smaller than the sum of the individual variances (i.e. the
     pairing is actually being credited) and matches a paired bootstrap

Run:  python test_delong.py
"""
import importlib.util, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'delong_mod', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '35_delong_deltas.py'))
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

RNG = np.random.default_rng(0)


def synth(n=4000, rho=0.9):
    """Two correlated scorers on the same rows, with a real signal."""
    y = (RNG.random(n) < 0.25).astype(np.float64)
    signal = y * 1.1 + RNG.normal(0, 1, n)
    a = signal + RNG.normal(0, 0.45, n)
    b = rho * signal + np.sqrt(1 - rho ** 2) * RNG.normal(0, 1, n) + RNG.normal(0, 0.45, n)
    return y, a, b


def bootstrap(y, preds, n_boot=4000):
    """Stratified paired bootstrap -> SDs of each AUC and of their difference."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    out = np.empty((n_boot, len(preds)))
    for i in range(n_boot):
        idx = np.concatenate([RNG.choice(pos, len(pos), replace=True),
                              RNG.choice(neg, len(neg), replace=True)])
        ys = y[idx]
        for j, p in enumerate(preds):
            out[i, j] = roc_auc_score(ys, p[idx])
    return out.std(axis=0, ddof=1), (out[:, 0] - out[:, 1]).std(ddof=1)


def main():
    y, a, b = synth()

    # 1 — AUC agrees with sklearn
    auc_d, var_d = dl.delong_single(y, a)
    auc_sk = roc_auc_score(y, a)
    assert abs(auc_d - auc_sk) < 1e-9, f"AUC mismatch: {auc_d} vs {auc_sk}"
    print(f"1. AUC        DeLong={auc_d:.10f}  sklearn={auc_sk:.10f}  OK")

    # 2 + 3 — variances agree with a real bootstrap
    (sd_a, sd_b), sd_diff = bootstrap(y, [a, b])
    se_d = np.sqrt(var_d)
    assert abs(se_d - sd_a) / sd_a < 0.10, f"SE mismatch: DeLong={se_d} boot={sd_a}"
    print(f"2. SE(AUC_a)  DeLong={se_d:.5f}  bootstrap={sd_a:.5f}  "
          f"rel.diff={abs(se_d - sd_a) / sd_a:.1%}  OK")

    _, _, delta, var_delta = dl.paired_delta(y, a, b)
    se_delta = np.sqrt(var_delta)
    _, var_b = dl.delong_single(y, b)
    se_unpaired = np.sqrt(var_d + var_b)

    assert se_delta < se_unpaired, (
        f"pairing not credited: paired SE {se_delta} >= unpaired {se_unpaired}")
    assert abs(se_delta - sd_diff) / sd_diff < 0.10, (
        f"paired SE mismatch: DeLong={se_delta} boot={sd_diff}")
    print(f"3. SE(delta)  DeLong={se_delta:.5f}  bootstrap={sd_diff:.5f}  "
          f"rel.diff={abs(se_delta - sd_diff) / sd_diff:.1%}  OK")
    print(f"   (unpaired-if-independent would be {se_unpaired:.5f}, "
          f"{se_unpaired / se_delta:.2f}x wider -> pairing credited)")

    # ties must not break midranks
    coarse = np.round(a, 1)
    auc_t, _ = dl.delong_single(y, coarse)
    assert abs(auc_t - roc_auc_score(y, coarse)) < 1e-9, "tie handling broken"
    print(f"4. ties       DeLong={auc_t:.10f}  sklearn={roc_auc_score(y, coarse):.10f}  OK")

    print("\nAll DeLong checks passed.")


if __name__ == '__main__':
    main()
