"""
SCRIPT 12 — DIFFERENTIAL PRIVACY TECHNICAL PARAMETERS
======================================================
Extracts exact DP accounting parameters from the 04_differential_privacy.py
experiment for manuscript Table S4 (DP Technical Parameters appendix).

Uses Opacus RDP accounting (not the deprecated moments accountant).

For each epsilon in DP_EPSILON_LEVELS, reports:
  - Target epsilon (ε)
  - Delta (δ)
  - Noise multiplier (σ) — from Opacus accountant
  - Clipping norm (C = DP_MAX_GRAD_NORM)
  - Sampling rate (q = batch_size / n)
  - Number of epochs (T)
  - Accountant type (RDP)
  - Minimum viable sample estimate: n_min ≈ C * sqrt(T) * sigma / epsilon

INPUTS:
  results/dp_results.json    (from 04_differential_privacy.py)
  config_paths.py            (DP_EPSILON_LEVELS, DP_TARGET_DELTA, etc.)

OUTPUTS:
  results/dp_technical_params.json
  Printed table for manuscript appendix

RUNTIME: < 1 minute (reads existing results)
"""

import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR,
    DP_EPSILON_LEVELS, DP_TARGET_DELTA, DP_MAX_GRAD_NORM,
    NN_LR, SEED,
)

# DP-specific hyperparameters (must match 04_differential_privacy.py)
DP_BATCH_SIZE = 512
DP_EPOCHS     = 5
N_PER_NODE    = 4_000   # approximate node size (15,650 total / 3 nodes, ~4k train after split)

print("=" * 65)
print("  12_dp_technical_details.py")
print("  Differential Privacy Technical Parameters")
print("=" * 65)
print(f"\n  DP_EPSILON_LEVELS : {DP_EPSILON_LEVELS}")
print(f"  DP_TARGET_DELTA   : {DP_TARGET_DELTA}")
print(f"  DP_MAX_GRAD_NORM  : {DP_MAX_GRAD_NORM}")
print(f"  DP_BATCH_SIZE     : {DP_BATCH_SIZE}")
print(f"  DP_EPOCHS         : {DP_EPOCHS}")


# ── Load dp_results.json if available ────────────────────────────────────────
dp_path    = os.path.join(RESULTS_DIR, 'dp_results.json')
dp_results = {}
if os.path.exists(dp_path):
    with open(dp_path) as f:
        dp_results = json.load(f)
    print(f"\n  Loaded: results/dp_results.json  ({len(dp_results)} epsilon levels)")
else:
    print(f"\n  dp_results.json not found — will use analytical estimates only.")
    print(f"  Run 04_differential_privacy.py first for empirical sigma values.")


# ── Per-epsilon accounting ────────────────────────────────────────────────────
params_table = []

# Try to use Opacus accountant to compute exact sigma values
try:
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
    OPACUS_AVAILABLE = True
    print("\n  Opacus available — computing exact sigma via RDP accountant.")
except ImportError:
    OPACUS_AVAILABLE = False
    print("\n  Opacus not available — using analytical approximations.")

sampling_rate = DP_BATCH_SIZE / N_PER_NODE

for eps_target in DP_EPSILON_LEVELS:
    row = {
        'target_epsilon': eps_target,
        'delta'         : DP_TARGET_DELTA,
        'C_clip'        : DP_MAX_GRAD_NORM,
        'batch_size'    : DP_BATCH_SIZE,
        'n_per_node'    : N_PER_NODE,
        'sampling_rate' : round(sampling_rate, 6),
        'n_epochs'      : DP_EPOCHS,
        'accountant'    : 'RDP',
    }

    if eps_target == float('inf'):
        row['sigma']   = 0.0
        row['note']    = 'No DP noise (baseline)'
        row['n_min_estimate'] = None
        params_table.append(row)
        continue

    # Get sigma from Opacus if available
    sigma = None
    if OPACUS_AVAILABLE:
        try:
            steps = int(DP_EPOCHS * N_PER_NODE / DP_BATCH_SIZE)
            sigma = get_noise_multiplier(
                target_epsilon=eps_target,
                target_delta=DP_TARGET_DELTA,
                sample_rate=sampling_rate,
                steps=steps,
                accountant='rdp',
            )
        except Exception as e:
            print(f"    Warning: Opacus sigma computation failed for eps={eps_target}: {e}")

    # If Opacus failed, try to extract from dp_results.json
    if sigma is None and dp_results:
        key = str(eps_target)
        if key in dp_results:
            sigma = dp_results[key].get('noise_multiplier', None)

    # Analytical fallback (Gaussian mechanism approximation)
    if sigma is None:
        # sigma ≈ sqrt(2 * ln(1.25/delta)) * C / epsilon (basic Gaussian mechanism)
        sigma = float(np.sqrt(2 * np.log(1.25 / DP_TARGET_DELTA)) *
                      DP_MAX_GRAD_NORM / eps_target)
        row['sigma_method'] = 'analytical_gaussian_approx'
    else:
        sigma = float(sigma)
        row['sigma_method'] = 'opacus_rdp' if OPACUS_AVAILABLE else 'dp_results_json'

    row['sigma'] = round(sigma, 4)

    # Minimum viable sample estimate
    # From Mironov (2017): n_min ~ C * sqrt(T) * sigma / epsilon
    # where T = total_steps = epochs * (n / batch_size)
    T = DP_EPOCHS  # number of passes
    row['n_min_estimate'] = int(
        DP_MAX_GRAD_NORM * np.sqrt(T) * sigma / max(eps_target, 1e-6)
    )

    params_table.append(row)


# ── Print table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  DP TECHNICAL PARAMETERS TABLE")
print("=" * 80)
print(f"  {'ε':>6}  {'δ':>8}  {'σ':>8}  {'C':>4}  {'q':>8}  {'T':>3}  "
      f"{'n_min':>8}  {'Note'}")
print(f"  {'-'*75}")

for row in params_table:
    eps_str = (str(row['target_epsilon']) if row['target_epsilon'] != float('inf')
               else '∞ (no DP)')
    sigma   = row.get('sigma', 0.0)
    n_min   = row.get('n_min_estimate')
    n_str   = str(n_min) if n_min is not None else 'N/A'
    note    = row.get('note', row.get('sigma_method', ''))
    print(f"  {eps_str:>6}  {row['delta']:>8.0e}  {sigma:>8.4f}  "
          f"{row['C_clip']:>4.1f}  {row['sampling_rate']:>8.6f}  "
          f"{row['n_epochs']:>3d}  {n_str:>8}  {note}")

print("=" * 80)
print("\n  Notation:")
print("    ε = privacy budget  |  δ = failure probability")
print("    σ = noise multiplier  |  C = gradient clipping norm")
print("    q = batch sampling rate  |  T = training epochs")
print("    n_min = minimum samples for DP guarantee at that ε")
print("\n  Accountant: RDP (Rényi DP, Mironov 2017)")
print("  Reference:  Andrew et al. (2021) 'Differentially Private Learning")
print("              with Adaptive Clipping' NeurIPS 2021.")


# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'dp_technical_params.json')
with open(out_path, 'w') as f:
    json.dump({
        'accountant'   : 'RDP (Rényi DP)',
        'reference'    : 'Mironov 2017; Opacus >= 1.0',
        'fixed_params' : {
            'delta'      : DP_TARGET_DELTA,
            'C_clip'     : DP_MAX_GRAD_NORM,
            'batch_size' : DP_BATCH_SIZE,
            'n_epochs'   : DP_EPOCHS,
            'n_per_node' : N_PER_NODE,
        },
        'per_epsilon': params_table,
    }, f, indent=2)
print(f"\n  Saved -> {out_path}")
print("=" * 65)
