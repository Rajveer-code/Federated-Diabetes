"""
41_figure_race_strata.py — the paper's headline figure
======================================================
Two panels, same axes, contrasting how two ways of getting a better model change
per-stratum discrimination relative to the composition-matched centralised control:

  left  — federated training (FedAvg): gains concentrated in the minority strata,
          majority stratum unchanged. Levelling up.
  right — more centralised data (Centralised-full): every stratum improves by a
          similar amount, so the White-Black gap does not close.

That contrast is the argument: the effect is not "a better model", it is
specifically federated optimisation.

Reads results/delong_deltas.json only. Regenerates from the results file, never
from hardcoded numbers.

Output: results/figures/fig_race_strata.{pdf,png}

Usage:  python 41_figure_race_strata.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

STRATA = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
PANELS = [('FedAvg', 'Federated (FedAvg)'),
          ('Centralised-full', 'More centralised data')]

# Colourblind-safe (Okabe-Ito): blue for significant, grey for not.
SIG, NS = '#0072B2', '#999999'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def main():
    path = os.path.join(RESULTS_DIR, 'delong_deltas.json')
    with open(path) as f:
        d = json.load(f)
    across = d['across_seeds']
    n_by = d['subgroup_n']
    n_seeds = len(d['seeds'])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True, sharey=True)

    for ax, (model, title) in zip(axes, PANELS):
        ys = np.arange(len(STRATA))[::-1]
        for y, s in zip(ys, STRATA):
            c = across[model][f'd_race_{s}']['combined']
            colour = SIG if c['excludes_zero'] else NS
            ax.plot([c['ci_low'], c['ci_high']], [y, y], color=colour, lw=2.2,
                    solid_capstyle='round', zorder=2)
            ax.plot([c['delta']], [y], 'o', color=colour, ms=5.5,
                    mec='white', mew=0.8, zorder=3)
        ax.axvline(0, color='black', lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{s}\n$n$={n_by['race_' + s]:,}" for s in STRATA])
        ax.set_title(title, pad=6)
        ax.set_xlabel('$\\Delta$ AUC vs. composition-matched centralised')
        ax.margins(y=0.14)

    # One annotation, on the panel it refers to.
    gap = across['FedAvg']['d_race_gap']['combined']
    axes[0].annotate(
        f"White$-$Black gap\n$\\Delta$={gap['delta']:+.4f}, $p$={gap['p']:.1e}",
        # Top-right: the only quadrant of this panel with no interval in it.
        xy=(0.97, 0.95), xycoords='axes fraction', fontsize=7.6,
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cccccc', lw=0.6))

    fig.suptitle(
        f'Per-stratum external AUC change, {n_seeds} seeds '
        f'(BRFSS $n$={d["n_brfss"]:,}); filled = 95% CI excludes zero',
        fontsize=8.6, y=1.005)
    fig.tight_layout()

    out_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(out_dir, f'fig_race_strata.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  saved -> {p}")

    print("\n  values plotted:")
    for model, _ in PANELS:
        print(f"  {model}")
        for s in STRATA:
            c = across[model][f'd_race_{s}']['combined']
            print(f"    {s:<9} {c['delta']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
                  f"p={c['p']:<10.3g} {'*' if c['excludes_zero'] else ''}")


if __name__ == '__main__':
    main()
