"""
50_figure_mechanism.py — the causal chain in two panels, plus its linking statistic
==================================================================================
The paper's argument is a chain:

    node prevalence spread  ->  cross-group ranking (xAUC)  ->  detection at
                                                                fixed capacity

Panel (a) is the first link, and it is the only experimental one: the controlled
sweep of 49_prevalence_sweep.py holds P(X|Y) identical across nodes and varies
only P(Y), so the dose is manipulated rather than observed.

Panel (b) is the robustness of the last link: the White-Black sensitivity gap
across every screening capacity from 1% to 50%, so the headline 20% is visibly
one point on a curve rather than a chosen operating point.

The middle link is a number, not a panel: across the five strata, the change in
xAUC_{g->White} (how often g's cases are ranked below White non-cases) is
compared against the change in that stratum's sensitivity at fixed capacity. If
the mechanism is right these must move together, and the one stratum that escapes
the cross-group penalty must be the one that escapes the detection loss.

Reads results/prevalence_sweep.json, results/capacity_curve.json,
results/xauc.json, results/clinical_translation.json. No hardcoded numbers.

Output: results/figures/fig_mechanism.{pdf,png}
        results/mechanism_chain.json

Usage:  python 50_figure_mechanism.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

STRATA = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
MODEL, RATE = 'FedAvg', '0.20'
GAIN, LOSS, REF = '#0072B2', '#D55E00', '#999999'      # Okabe-Ito

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def main():
    ps = load('prevalence_sweep.json')
    cc = load('capacity_curve.json')
    xa = load('xauc.json')
    ct = load('clinical_translation.json')

    # ── middle link: does cross-group ranking predict detection? ─────────────
    d_x, d_s = [], []
    for g in STRATA:
        cell = f'{g}->White' if g != 'White' else 'White->White'
        d_x.append(xa['across_seeds'][MODEL][cell]['mean'])
        d_s.append(ct['by_rate'][RATE][MODEL][g]['delta_tpr'])
    r_chain = float(np.corrcoef(d_x, d_s)[0, 1])

    chain = {
        'link_metric': 'Pearson r across the five strata between the change in '
                       'xAUC_{g->White} and the change in sensitivity at 20% '
                       'capacity, both FedAvg minus the composition-matched control.',
        'note': 'For White the diagonal cell White->White is used, since there is '
                'no cross-group cell against itself.',
        'r': round(r_chain, 4),
        'per_stratum': {g: {'delta_xauc_vs_white': round(x, 5),
                            'delta_sensitivity': round(s, 5)}
                        for g, x, s in zip(STRATA, d_x, d_s)},
    }

    print(f"  {'stratum':<10}{'d xAUC(g>White)':>18}{'d sens@20%':>13}")
    for g, x, s in zip(STRATA, d_x, d_s):
        print(f"  {g:<10}{x:>+18.4f}{s:>+13.4f}")
    print(f"\n  Pearson r(cross-group ranking, detection) = {r_chain:+.3f} "
          f"across {len(STRATA)} strata")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

    # ── (a) dose-response ────────────────────────────────────────────────────
    ax = axes[0]
    keys = sorted(ps['summary'], key=lambda k: ps['summary'][k]['spread_achieved'])
    x = [ps['summary'][k]['spread_achieved'] for k in keys]
    y = [ps['summary'][k]['delta_sens_gap_vs_matched'] for k in keys]
    sd = [ps['summary'][k]['sens_gap_sd'] for k in keys]
    n = ps['summary'][keys[0]]['n_seeds']
    se = [s / np.sqrt(n) for s in sd]
    ax.errorbar(x, y, yerr=[1.96 * s for s in se], fmt='o-', color=LOSS,
                ms=4.5, lw=1.5, capsize=2.5, mec='white', mew=0.7, zorder=3)
    ax.axhline(0, color='black', lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlabel('inter-node prevalence spread $\\pi_{\\max}-\\pi_{\\min}$')
    ax.set_ylabel('$\\Delta$ White$-$Black\nsensitivity gap')
    ax.set_title('(a) Only $P(Y)$ varies across nodes', pad=5)
    ax.margins(x=0.08)

    # ── (b) capacity curve ───────────────────────────────────────────────────
    ax = axes[1]
    rates = np.asarray(cc['rates'])
    for name, col, lab in ((cc['reference'], REF, 'centralised (matched)'),
                           (MODEL, LOSS, MODEL)):
        g = np.asarray(cc['by_model'][name]['_wb_sens_gap'])
        s = np.asarray(cc['by_model'][name]['_wb_sens_gap_sd']) / np.sqrt(len(cc['seeds']))
        ax.plot(rates, g, color=col, lw=1.6, label=lab, zorder=3)
        ax.fill_between(rates, g - 1.96 * s, g + 1.96 * s, color=col,
                        alpha=0.18, lw=0, zorder=2)
    ax.axhline(0, color='black', lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axvline(0.20, color='black', lw=0.7, ls=':', zorder=1)
    ax.set_xlabel('screening capacity (fraction flagged)')
    ax.set_ylabel('White$-$Black sensitivity gap')
    ax.set_title('(b) Not an artefact of one threshold', pad=5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.legend(frameon=False, loc='upper left')

    fig.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(out_dir, f'fig_mechanism.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  saved -> {p}")

    path = os.path.join(RESULTS_DIR, 'mechanism_chain.json')
    with open(path, 'w') as f:
        json.dump(chain, f, indent=2)
    print(f"Saved -> {path}")


if __name__ == '__main__':
    main()
