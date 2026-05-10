# Privacy-Preserving Federated Learning for Diabetes Risk Prediction

> Multi-site machine learning with demographic fairness analysis, differential privacy, and post-hoc calibration — validated on 1.28 million patients.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org)
[![Flower FL](https://img.shields.io/badge/Flower-1.5%2B-blueviolet.svg)](https://flower.dev)
[![Opacus DP](https://img.shields.io/badge/Opacus-1.4%2B-ff69b4.svg)](https://opacus.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Target: JBI Q1](https://img.shields.io/badge/Target-JBI%20Q1%20Elsevier-orange.svg)](https://www.journals.elsevier.com/journal-of-biomedical-informatics)

**Author:** Rajveer Singh Pall  
**Affiliation:** Gyan Ganga Institute of Technology and Sciences, Jabalpur, India  
**Submitted to:** *Journal of Biomedical Informatics* (JBI/Elsevier, Q1)

---

## Overview

Centralised machine learning for healthcare requires pooling raw patient records across institutions — a practice that conflicts with HIPAA, GDPR, and the practical realities of hospital data governance. This project implements and evaluates a **privacy-preserving federated learning (FL)** framework for type-2 diabetes risk prediction that keeps patient data on-site while still enabling multi-institutional collaboration.

Four FL aggregation strategies — **FedAvg**, **FedProx** (μ=0.1), **FedNova** (τ={5,3,4}), and **SCAFFOLD** (Option II) — are trained on demographically partitioned NHANES data and validated externally on **1,282,897 BRFSS respondents**, the largest independent validation reported for this prediction task. Beyond discrimination (AUC), the study rigorously evaluates **demographic fairness** (elderly–young AUC gap), **calibration** (Platt, isotonic, temperature scaling), and the **differential privacy–utility tradeoff** (DP-SGD, ε ∈ {0.5, 1.0, 2.0, 5.0, ∞}).

FedAvg achieves an external AUC of **0.757 [0.756–0.758]**, outperforming the centralised XGBoost baseline (0.700) by 8.2 percentage points. The federated framework reduces the elderly–young fairness gap from 0.069 (centralised) to **0.054**, and isotonic recalibration reduces the Expected Calibration Error from 0.319 to **< 0.002**. Tight differential privacy (ε ≤ 5) causes model collapse at the per-node sample sizes typical in healthcare, surfacing a fundamental privacy–utility tension relevant to real-world deployment.

---

## Results at a Glance

### Discrimination — Internal (NHANES) and External (BRFSS) Validation

| Model | Internal AUC [95% CI] | External AUC [95% CI] | Elderly Gap (Δ) |
|:---|:---:|:---:|:---:|
| Centralised XGBoost (baseline) | 0.769 [0.760–0.777] | 0.700 [0.698–0.701] | 0.069 |
| **FedAvg** ⭐ | **0.788 [0.779–0.796]** | **0.757 [0.756–0.758]** | **0.054** |
| FedProx (μ=0.1) | 0.785 [0.776–0.793] | 0.752 [0.751–0.753] | 0.066 |
| FedNova (τ={5,3,4}) | 0.786 [0.778–0.794] | 0.744 [0.743–0.745] | 0.064 |
| SCAFFOLD (SGD, 50 rounds) | 0.642 (internal only) | — | — |
| Published benchmark [Ahsan et al. 2022] | 0.742 (young subgroup) | — | 0.135 |

Internal CIs: stratified bootstrap (N=2,000, n=15,650). External CIs: DeLong structural components estimator (n=1,282,897).

### Calibration — FedProx on BRFSS External Set

| Method | ECE | AUC | Notes |
|:---|:---:|:---:|:---|
| Uncalibrated | 0.319 | 0.752 | Severe overconfidence |
| Platt scaling | 0.016 | 0.752 | Two-parameter fix; robust to small cal sets |
| **Isotonic regression** ⭐ | **0.001** | 0.752 | Near-perfect calibration |
| Temperature scaling (T=2.25) | 0.311 | 0.752 | Limited benefit; non-monotone curve |

### Differential Privacy Tradeoff

| ε (target) | AUC | Outcome |
|:---:|:---:|:---|
| 0.5 | 0.500 | Model collapse |
| 1.0 | 0.498 | Model collapse |
| 2.0 | 0.516 | Model collapse |
| 5.0 | 0.500 | Model collapse |
| ∞ (no DP) | 0.766 | Full recovery |

All collapse at ε ≤ 5 reflects the high noise multiplier required at healthcare-scale sampling rates (~5–15%).

---

## Research Questions

| | Question | Finding |
|:---|:---|:---|
| **RQ1 · Performance** | Can FL match or exceed a centralised model on external data? | ✅ FedAvg +8.2 pp external AUC vs. centralised XGBoost |
| **RQ2 · Fairness** | Does federated training reduce demographic AUC disparity? | ✅ Elderly gap: 0.069 → 0.054 (21.7% within-study improvement) |
| **RQ3 · Privacy** | What is the accuracy cost of differential privacy? | ⚠️ Model collapse at ε ≤ 5 for per-node sample sizes of 3–4.5k |
| **RQ4 · Calibration** | Are federated risk scores clinically trustworthy? | ✅ ECE 0.319 → 0.001 after isotonic recalibration |

---

## Repository Structure

```
Federated-Diabetes/
│
├── federated/                              # Core FL pipeline
│   │
│   ├── ── Core modules ──
│   ├── config_paths.py                     # All hyperparameters and paths (single source of truth)
│   ├── nn_model.py                         # DiabetesNet: 4-layer MLP, BatchNorm, AdamW
│   ├── fl_client.py                        # Flower FL client (DiabetesClient)
│   ├── data_utils.py                       # Data loading, preprocessing, node splits
│   │
│   ├── ── Pipeline (run in order) ──
│   ├── 00_fit_global_scaler.py             # Fit StandardScaler on NHANES training split only
│   ├── 01_partition_data.py                # Partition NHANES into 3 demographically distinct nodes
│   ├── 02_centralised_baseline.py          # XGBoost + centralised DiabetesNet baselines
│   ├── 03_federated_simulation.py          # FedAvg + FedProx (50 rounds, 3 nodes)
│   ├── 03b_fednova_corrected.py            # FedNova with corrected gradient normalisation
│   ├── 04_differential_privacy.py          # DP-SGD experiments (ε ∈ {0.5, 1, 2, 5, ∞})
│   ├── 05_fairness_analysis.py             # Subgroup AUC, fairness gap, equalised odds
│   ├── 06_results_summary.py               # Consolidated results table
│   ├── 07_external_validation.py           # BRFSS 2020–2022 external validation (n=1.28M)
│   ├── 07_statistical_analysis.py          # Stratified bootstrap + DeLong CIs
│   ├── 08_scaffold_baseline.py             # SCAFFOLD Option II (Karimireddy et al. ICML 2020)
│   ├── 09_calibration_analysis.py          # Platt / isotonic / temperature scaling + ECE
│   ├── 10_stratified_centralised_experiment.py  # Mechanism analysis (federation vs. composition)
│   ├── 11_subgroup_confidence_intervals.py      # Bootstrap CIs for fairness gap
│   ├── 12_dp_technical_details.py               # RDP accounting tables
│   │
│   ├── ── Outputs ──
│   ├── generate_publication_figures.py     # All 8 publication figures (300 dpi)
│   ├── write_manuscript_v5.py              # Generates complete JBI manuscript (DOCX)
│   │
│   └── results/                            # All committed result artefacts
│       ├── auc_confidence_intervals.json   # Bootstrap + DeLong CIs for all models
│       ├── external_validation.json        # BRFSS AUC + subgroup fairness
│       ├── calibration_results.json        # ECE before/after calibration
│       ├── federated_convergence.json      # Round-by-round AUC for FedAvg/FedProx
│       ├── fednova_corrected.json          # FedNova results
│       ├── scaffold_results.json           # SCAFFOLD convergence (50 rounds)
│       ├── dp_results.json                 # DP-SGD results per ε level
│       ├── fairness_metrics.json           # Subgroup AUC + fairness gaps
│       └── figures/                        # 8 × 300 dpi publication figures
│           ├── fig1_architecture.png       # System architecture + DiabetesNet diagram
│           ├── fig2_convergence.png        # Round-by-round AUC convergence
│           ├── fig3_roc_curves.png         # ROC curves (internal + external)
│           ├── fig4_fairness.png           # Age-stratified AUC + fairness gap
│           ├── fig5_dp_tradeoff.png        # Privacy–utility tradeoff
│           ├── fig6_calibration.png        # Reliability diagrams (4 methods)
│           ├── fig7_generalisation_gap.png # Internal vs. external AUC gap
│           └── fig8_summary_comparison.png # Overall model comparison
│
├── README.md
├── requirements.txt
├── LICENSE                                 # MIT
└── .gitignore
```

> **Not tracked** (reproducible or sensitive): raw data (`data/`), model weights (`models/`), scaler artefacts (`artefacts/`), prediction arrays (`*.npy`), manuscript DOCX (under review).

---

## Setup

**Prerequisites:** Python 3.10+, CUDA-capable GPU recommended (CPU works but is slower).

```bash
git clone https://github.com/Rajveer-code/Federated-Diabetes.git
cd Federated-Diabetes
pip install -r requirements.txt
```

**Data acquisition:**

| Dataset | Source | Purpose |
|:---|:---|:---|
| NHANES 2013–2020 | [cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/) | Training (n=15,650) |
| BRFSS 2020–2022 | [cdc.gov/brfss](https://www.cdc.gov/brfss/annual_data/annual_data.htm) | External validation (n=1,282,897) |

Download NHANES XPT files for cycles 2013-14, 2015-16, 2017-18, 2019-20. Features used: `RIDAGEYR`, `BMXBMI`, `BPXOSY3`, `LBXGH`, `LBXGLU`, `LBXTC`, `PAQ650`, `SMQ020`, `RIAGENDR`.

For BRFSS external validation, set the path via environment variable:
```bash
# Linux / macOS
export BRFSS_PATH="/path/to/brfss_final.csv"

# Windows (PowerShell)
$env:BRFSS_PATH = "C:\path\to\brfss_final.csv"
```

---

## Reproducing Results

Run scripts in order from the `federated/` directory. Total expected runtime on an RTX 3060+: **~25 minutes**.

```bash
cd federated/

# Step 1 — Fit global scaler (MUST run first; prevents data leakage)
python 00_fit_global_scaler.py
# → artefacts/global_nhanes_scaler.joblib

# Step 2 — Partition NHANES into 3 demographically stratified nodes
python 01_partition_data.py
# → data/node_{A,B,C}_{train,val}.csv

# Step 3 — Centralised baselines (~2 min)
python 02_centralised_baseline.py
# → results/centralised_metrics.json

# Step 4 — FedAvg + FedProx simulation (~5 min)
python 03_federated_simulation.py
# → results/federated_convergence.json

# Step 5 — FedNova with corrected gradient normalisation (~5 min)
python 03b_fednova_corrected.py
# → results/fednova_corrected.json

# Step 6 — Differential privacy experiments (~3 min)
python 04_differential_privacy.py
# → results/dp_results.json

# Step 7 — Fairness analysis: subgroup AUC + equalised odds (~2 min)
python 05_fairness_analysis.py
# → results/fairness_metrics.json

# Step 8 — Results summary
python 06_results_summary.py

# Step 9 — External validation on BRFSS 2020–2022 (~4 min)
python 07_external_validation.py
# → results/external_validation.json

# Step 10 — Statistical CIs: stratified bootstrap + DeLong (~3 min)
python 07_statistical_analysis.py
# → results/auc_confidence_intervals.json

# Step 11 — SCAFFOLD Option II (~1 min)
python 08_scaffold_baseline.py
# → results/scaffold_results.json

# Step 12 — Post-hoc calibration analysis (~2 min)
python 09_calibration_analysis.py
# → results/calibration_results.json

# Step 13 — Generate all 8 publication figures (300 dpi)
python generate_publication_figures.py
# → results/figures/fig{1..8}_*.png

# Step 14 — Generate complete manuscript
python write_manuscript_v5.py
# → ../FL_Diabetes_Manuscript_v5_Submission.docx
```

All pre-computed results are committed under `federated/results/` so you can inspect metrics and figures immediately without running the full pipeline.

---

## Hyperparameters

All hyperparameters are centralised in `federated/config_paths.py`.

| Parameter | Value | Rationale |
|:---|:---:|:---|
| `FL_NUM_ROUNDS` | 50 | Convergence criterion ΔAUC < 0.001 met by round 38 |
| `NN_LOCAL_EPOCHS` | 5 | Balances convergence speed and communication cost |
| `NN_BATCH_SIZE` | 256 | GPU-optimised; fits 4 GB VRAM for 8-feature input |
| `FedProx μ` | 0.1 | Grid search over {0.01, 0.1, 1.0}; best external AUC |
| `FedNova τ` | {5, 3, 4} | Node-specific local update counts (high-shift nodes use fewer steps) |
| `AdamW lr` | 0.001 | CosineAnnealingLR schedule; η_min = 10⁻⁶ |
| `RANDOM_SEED` | 42 | Fixed across all experiments for reproducibility |
| `DP δ` | 10⁻⁵ | Standard choice for datasets with n < 10⁶ |
| `DP clipping C` | 1.0 | Gradient clipping norm for DP-SGD |

---

## Key Technical Decisions

### Global Scaler — Preventing Data Leakage
A single `StandardScaler` is fitted **once** on the NHANES training split (`00_fit_global_scaler.py`) and saved as a shared artefact. Every node calls `.transform()` only — never `.fit_transform()`. Fitting separate scalers per node would contaminate evaluation with node-local statistics, a subtle but critical form of data leakage that invalidates cross-node comparisons.

### DeLong Estimator for Large-Scale CIs
Stratified bootstrap on 1.28 million BRFSS records is computationally intractable (kernel matrix ≈ 708 GB). We use the DeLong structural components estimator — an O(n log n) algorithm implemented via `numpy.searchsorted` — which produces mathematically equivalent 95% CIs in seconds and is used throughout the external validation analysis.

### SCAFFOLD with SGD
SCAFFOLD's convergence guarantee (Karimireddy et al., ICML 2020) requires SGD. Our implementation respects this, which means SCAFFOLD is compared against AdamW-based strategies under an inherent optimizer mismatch. The lower AUC (0.642 vs. 0.788) reflects this combined effect and is reported transparently as an honest finding. A matched AdamW-SCAFFOLD ablation is a natural extension.

---

## Limitations

- Node partitioning is **simulated** from a single national cohort (NHANES). Real deployments involve genuine between-site batch effects, divergent measurement protocols, and independent patient populations not captured here.
- The BRFSS smoking variable mapping (cigarette-days-per-year → binary) introduces measurement error relative to the clinical NHANES measure.
- SCAFFOLD was evaluated with SGD (required by theory); performance relative to AdamW-based methods may improve with an adaptive optimiser.
- Tight differential privacy (ε ≤ 5) causes model collapse at per-node sample sizes of ~3,000–4,500. Realistic deployment at these sizes requires either larger cohorts or weaker DP notions (local DP, shuffling).
- All nodes share the same DiabetesNet architecture; personalised FL with heterogeneous local models was not evaluated.

---

## Citation

If this work is useful to you, please cite:

```bibtex
@article{pall2025fl_diabetes,
  title   = {Privacy-Preserving Federated Learning for Diabetes Risk Prediction
             Across Demographically Heterogeneous Hospital Nodes},
  author  = {Pall, Rajveer Singh and Yadav, Sameer},
  journal = {Journal of Biomedical Informatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## Ethics Statement

- **NHANES** (training data): publicly available, de-identified survey data collected under US federal ethics oversight (NCHS IRB protocol).
- **BRFSS** (external validation): publicly available, de-identified, state-administered telephone survey with CDC oversight.
- No new patient data were collected for this study. Secondary analysis of publicly released, de-identified datasets does not require IRB approval under US federal regulations (45 CFR 46.104).
- All model outputs are for **research purposes only**. This codebase does not constitute a validated clinical diagnostic tool and should not be used for patient-level clinical decisions.

---

## License

MIT License — see [LICENSE](LICENSE). Free to use, modify, and distribute with attribution.
