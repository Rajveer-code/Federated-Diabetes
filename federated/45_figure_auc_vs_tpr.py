"""
45_figure_auc_vs_tpr.py — the paper's headline figure
=====================================================
Two panels, same strata, opposite conclusions:

  left  — change in within-stratum AUC vs the composition-matched centralised
          control. Minority strata gain. By the standard AUC-gap fairness metric,
          federation looks equity-improving.
  right — change in sensitivity at a fixed 20% screening capacity, the operating
          point a programme would actually deploy. The same minority strata lose.

Reads results/delong_deltas.json (AUC) and results/clinical_translation.json
(sensitivity); no hardcoded numbers.

Output: results/figures/fig_auc_vs_tpr.{pdf,png}
Usage:  python 45_figure_auc_vs_tpr.py [--rate 0.20]
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

STRATA = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
GAIN, LOSS, FLAT = '#0072B2', '#D55E00', '#999999'   # Okabe-Ito, colourblind-safe

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=float, default=0.20)
    ap.add_argument('--model', default='FedAvg')
    args = ap.parse_args()

    dd = json.load(open(os.path.join(RESULTS_DIR, 'delong_deltas.json')))
    ct = json.load(open(os.path.join(RESULTS_DIR, 'clinical_translation.json')))
    rate_key = f'{args.rate:.2f}'
    ctm = ct['by_rate'][rate_key][args.model]
    n_by = dd['subgroup_n']

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    ys = np.arange(len(STRATA))[::-1]

    # ---- left: AUC ----
    ax = axes[0]
    for y, s in zip(ys, STRATA):
        c = dd['across_seeds'][args.model][f'd_race_{s}']['combined']
        col = FLAT if not c['excludes_zero'] else (GAIN if c['delta'] > 0 else LOSS)
        ax.plot([c['ci_low'], c['ci_high']], [y, y], color=col, lw=2.2,
                solid_capstyle='round', zorder=2)
        ax.plot([c['delta']], [y], 'o', color=col, ms=5.5, mec='white', mew=0.8, zorder=3)
    ax.set_title('Within-stratum AUC', pad=6)
    ax.set_xlabel('$\\Delta$ AUC')

    # ---- right: sensitivity at fixed capacity ----
    ax = axes[1]
    for y, s in zip(ys, STRATA):
        d  = ctm[s]['delta_tpr']
        se = ctm[s]['tpr_sd'] / np.sqrt(len(ct['seeds']))
        lo, hi = d - 1.96 * se, d + 1.96 * se
        col = FLAT if lo <= 0 <= hi else (GAIN if d > 0 else LOSS)
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle='round', zorder=2)
        ax.plot([d], [y], 'o', color=col, ms=5.5, mec='white', mew=0.8, zorder=3)
    ax.set_title(f'Sensitivity at {args.rate:.0%} screening capacity', pad=6)
    ax.set_xlabel('$\\Delta$ sensitivity')

    for ax in axes:
        ax.axvline(0, color='black', lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.margins(y=0.14)
        # The two panels differ ~5x in scale; let each pick few, readable ticks
        # rather than inheriting a dense shared locator that overlaps.
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune=None))
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f"{s}\n$n$={n_by['race_' + s]:,}" for s in STRATA])

    fig.suptitle('The same federated model, judged two ways '
                 f'({args.model} vs. composition-matched centralised, '
                 f'{len(ct["seeds"])} seeds)', fontsize=8.8, y=1.02)
    fig.tight_layout()

    out_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(out_dir, f'fig_auc_vs_tpr.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  saved -> {p}")

    print(f"\n  {'stratum':<10}{'dAUC':>10}{'dSens':>10}{'cases/100k':>13}")
    for s in STRATA:
        c = dd['across_seeds'][args.model][f'd_race_{s}']['combined']
        print(f"  {s:<10}{c['delta']:>+10.4f}{ctm[s]['delta_tpr']:>+10.4f}"
              f"{ctm[s]['delta_cases_per_100k']:>+13.1f}")


if __name__ == '__main__':
    main()
