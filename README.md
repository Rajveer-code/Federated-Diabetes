# Federated Learning for Diabetes Prediction

**Privacy-Preserving Multi-Site Machine Learning with Fairness and Differential Privacy Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Author:** Rajveer Singh Pall  
**Affiliation:** Gyan Ganga Institute of Technology and Sciences, Jabalpur  
**Target venue:** Journal of Biomedical Informatics (JBI)

---

## Abstract

We extend a published centralised XGBoost diabetes-prediction model (internal AUC = 0.794,
external AUC = 0.717 on BRFSS) with a federated learning framework that trains across three
demographically heterogeneous simulated hospital nodes without sharing raw patient data.
We evaluate FedAvg, FedProx (μ=0.1), FedNova (τ={5,3,4}), and SCAFFOLD against centralised
baselines, and assess fairness (age-subgroup AUC gap), distributional robustness
(internal-to-external AUC degradation), post-hoc calibration, and privacy
(differential privacy ε ∈ {0.5, 1.0, 2.0, 5.0}).

FedAvg is associated with improved external AUC (0.757 vs 0.717 centralised) while
reducing the elderly–young fairness gap from 0.135 to 0.069. Tight DP budgets
(ε < 2) cause model collapse at our per-node sample sizes (~3,200–4,500), consistent
with theoretical minimum viable sample estimates.

---

## Key Results

| Model | Internal AUC [95% CI] | External AUC [95% CI] |
|---|---|---|
| Published XGBoost | 0.794 | 0.717 |
| Centralised XGBoost (replicated) | 0.769 [0.760–0.777] | 0.700 [0.698–0.701] |
| Centralised NN (DiabetesNet) | 0.801 [0.782–0.819] | 0.749 [0.748–0.750] |
| FedAvg | 0.788 [0.779–0.796] | 0.757 [0.756–0.758] |
| FedProx μ=0.1 | 0.785 [0.776–0.793] | 0.752 [0.751–0.753] |
| FedNova τ={5,3,4} | 0.786 [0.778–0.794] | 0.744 [0.743–0.745] |
| SCAFFOLD (Option II) | run 08_scaffold_baseline.py | — |

Internal: stratified bootstrap CIs (N=2,000, NHANES test set n≈3,130).
External: DeLong CIs (structural components; BRFSS n=1,282,897).

---

## Research Contributions

1. **RQ1 — Performance**: Can federated training match or exceed a centralised
   model trained on pooled data? (FedAvg yes on external; NN centralised marginally
   better on internal.)

2. **RQ2 — Fairness**: Does federation reduce demographic disparity? (Yes: elderly
   gap 0.135→0.069 under FedAvg.)

3. **RQ3 — Privacy**: What is the accuracy cost of differential privacy at clinically
   meaningful ε levels? (Model collapse at ε<2; ε=5 loses <0.01 AUC.)

4. **RQ4 — Calibration**: Are federated model probabilities well-calibrated? (Moderate
   miscalibration; Platt scaling reduces ECE by ~40%.)

---

## Repository Structure

```
diabetes_prediction_project/
├── federated/                          # All source code (Python)
│   ├── config_paths.py                 # Single source of truth — all paths/hyperparameters
│   ├── nn_model.py                     # DiabetesNet architecture + train_one_epoch
│   ├── fl_client.py                    # Federated client (FedAvg/FedProx)
│   ├── data_utils.py                   # Data loading, scaling, class weight utilities
│   ├── 00_fit_global_scaler.py         # Fit NHANES scaler once (MUST run first)
│   ├── 01_partition_data.py            # Partition NHANES into 3 hospital nodes
│   ├── 02_centralised_baseline.py      # XGBoost + NN centralised baselines
│   ├── 03_federated_simulation.py      # FedAvg + FedProx + FedNova (50 rounds)
│   ├── 03b_fednova_corrected.py        # FedNova with correct heterogeneous τ
│   ├── 04_differential_privacy.py      # DP-SGD epsilon sweep (Opacus)
│   ├── 05_fairness_analysis.py         # Subgroup AUC fairness metrics
│   ├── 06_results_summary.py           # Compile master results table
│   ├── 07_external_validation.py       # External validation on BRFSS
│   ├── 07_statistical_analysis.py      # DeLong CIs + paired tests
│   ├── 07c_fednova_corrected_external.py  # FedNova corrected external eval
│   ├── 08_scaffold_baseline.py         # SCAFFOLD Option II (Karimireddy 2020)
│   ├── 09_calibration_analysis.py      # Platt/isotonic/temperature scaling + ECE
│   ├── 10_stratified_centralised_experiment.py  # Federation vs composition effect
│   ├── 11_subgroup_confidence_intervals.py  # Bootstrap CIs for subgroup AUC
│   ├── 12_dp_technical_details.py      # DP RDP accounting parameters table
│   ├── 13_apply_manuscript_edits.py    # Apply 16 edits -> v5 submission docx
│   ├── create_supplementary.py         # Generate Supplementary_Material.docx
│   ├── data/                           # CSV inputs (not committed; see Data section)
│   ├── models/                         # Model weights .pt (not committed)
│   ├── results/                        # JSON result files (committed)
│   └── artefacts/                      # Fitted scalers (not committed)
├── FL_Diabetes_Manuscript_v4_Final.docx  # v4 manuscript
├── FL_Diabetes_Manuscript_v5_Submission.docx  # generated by 13_apply_manuscript_edits.py
├── Supplementary_Material.docx         # Tables S1-S4 (generated by create_supplementary.py)
└── README.md
```

---

## Requirements

```
python      >= 3.10
torch       >= 2.0
xgboost     >= 1.7
scikit-learn >= 1.3
numpy       >= 1.24
pandas      >= 2.0
matplotlib  >= 3.7
scipy       >= 1.11
joblib      >= 1.3
python-docx >= 1.2.0
opacus      >= 1.4
```

Install (conda recommended):

```bash
conda create -n fl-diabetes python=3.10
conda activate fl-diabetes
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xgboost scikit-learn numpy pandas matplotlib scipy joblib python-docx opacus
```

For CPU-only machines, omit the `--index-url` flag.

See `Supplementary_Material.docx` Table S2 for full hyperparameter specification.

---

## Data Access

Data files are **not committed** — they are large and publicly available from the CDC.

### NHANES 2015–2020 (internal training and test)

Download from: https://www.cdc.gov/nchs/nhanes/index.htm

Relevant survey years: 2015-2016, 2017-2018, 2019-2020 (pre-pandemic).
Required modules: Demographics (DEMO), BMI (BMX), Smoking (SMQ), Physical Activity (PAQ),
Medical Conditions (MCQ), Diabetes (DIQ).

Processed file expected at: `federated/data/centralised_full.csv`

Required columns: `RIDAGEYR RIAGENDR RIDRETH3 BMXBMI SMOKING PHYS_ACTIVITY HEART_ATTACK STROKE DIABETES`

### BRFSS 2021 (external validation)

Download from: https://www.cdc.gov/brfss/annual_data/annual_data.htm (2021 annual survey)

Processed file is configured via environment variable (no hardcoded paths):

```powershell
# PowerShell
$env:BRFSS_PATH = "C:\your\path\to\brfss_final.csv"
python 07_external_validation.py
```

```bash
# bash / WSL
export BRFSS_PATH=/your/path/to/brfss_final.csv
python 07_external_validation.py
```

Column mapping is applied automatically in `07_external_validation.py`.
See `Supplementary_Material.docx` Table S1 for the full feature harmonisation specification.

---

## Reproducing Results

All scripts run from `federated/`. Expected runtimes on RTX 4060 (8 GB VRAM):

```bash
cd D:/Projects/diabetes_prediction_project/federated
```

**Step 0 — Fit global scaler** (must run first; ~1 min):
```bash
python 00_fit_global_scaler.py
# Output: artefacts/global_nhanes_scaler.joblib
```

**Step 1 — Partition data** (~1 min):
```bash
python 01_partition_data.py
# Output: data/node_{a,b,c}_*.csv
```

**Step 2 — Centralised baselines** (~10 min):
```bash
python 02_centralised_baseline.py
# Output: results/centralised_results.json, models/xgb_model.pkl
#         results/y_true_internal.npy, results/pred_xgb_internal.npy
```

**Step 3 — Federated simulation** (~25–40 min):
```bash
python 03_federated_simulation.py   # FedAvg + FedProx + FedNova
python 03b_fednova_corrected.py     # FedNova with heterogeneous tau
# Output: models/{fedavg,fedprox,fednova_corrected}_weights.pt
#         results/federated_convergence.json
#         results/pred_{fedavg,fedprox,fednova}_internal.npy
```

**Step 4 — Differential privacy** (~10 min; requires Opacus):
```bash
python 04_differential_privacy.py
# Output: results/dp_results.json
```

**Step 5 — Fairness analysis** (~5 min):
```bash
python 05_fairness_analysis.py
# Output: results/fairness_metrics.json
```

**Step 6 — Results summary** (~1 min):
```bash
python 06_results_summary.py
# Output: results/master_results_table.json
```

**Step 7 — External validation** (~15 min; requires BRFSS_PATH env var):
```bash
python 07_external_validation.py
# Output: results/external_validation.json
#         results/y_true_brfss.npy, results/pred_{fedavg,fedprox,fednova}_external.npy

python 07_statistical_analysis.py
# Output: results/auc_confidence_intervals.json
```

**Step 8 — SCAFFOLD baseline** (~20 min):
```bash
python 08_scaffold_baseline.py
# Output: models/scaffold_weights.pt, results/scaffold_results.json
#         results/pred_scaffold_internal.npy
```

**Step 9 — Calibration analysis** (~5 min; requires BRFSS preds):
```bash
python 09_calibration_analysis.py
# Output: results/calibration_results.json
#         plots/calibration_reliability_diagram.png
```

**Step 10–12 — Additional experiments** (~10 min each):
```bash
python 10_stratified_centralised_experiment.py
python 11_subgroup_confidence_intervals.py
python 12_dp_technical_details.py
```

**Step 13 — Generate submission manuscript** (~1 min):
```bash
python 13_apply_manuscript_edits.py
# Output: ../FL_Diabetes_Manuscript_v5_Submission.docx
```

---

## Hyperparameters

See `Supplementary_Material.docx` Table S2 for the complete hyperparameter table
with search ranges and rationale.

Key values (all in `federated/config_paths.py`):

| Parameter | Value | Notes |
|---|---|---|
| Architecture | [64, 32, 16] | Input(8)→Dense(64,BN,ReLU,Drop)→… |
| Dropout | 0.3 | 5-fold CV tuned |
| Learning rate | 1e-3 | AdamW; grid over {1e-4, 1e-3, 5e-3} |
| FL rounds | 50 | AUC plateau <0.001 in last 15 rounds |
| FedProx μ | 0.1 | Grid over {0.01, 0.1, 0.5} |
| FedNova τ | {5, 3, 4} | Per Wang et al. Theorem 2 |
| Random seed | 42 | All stochastic operations |

---

## Known Limitations

- **Simulated federation**: All three nodes are derived from NHANES (a single national
  survey); true inter-institutional heterogeneity is not captured.
- **External shift**: NHANES (lab-confirmed diabetes) → BRFSS (self-reported diagnosis)
  creates label noise not quantified in the current analysis.
- **DP at small n**: Per-node samples (~3,200–4,500) fall below the ~40,000 minimum
  for ε=1.0 DP (see `12_dp_technical_details.py`). Results at tight ε reflect this.
- **Semi-honest adversary only**: DP guarantees do not extend to a malicious server
  or collusion. Secure aggregation is not implemented.
- **Smoking harmonisation**: NHANES lab-verified vs BRFSS self-reported smoking
  represents the highest feature harmonisation risk (Table S1).

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{pall2025feddiabetes,
  title   = {Federated Learning for Diabetes Prediction: Privacy-Preserving
             Multi-Site Machine Learning with Fairness Analysis},
  author  = {Pall, Rajveer Singh},
  journal = {Journal of Biomedical Informatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Ethics Statement

NHANES and BRFSS are publicly available de-identified datasets released by the US
Centers for Disease Control and Prevention (CDC). No IRB approval is required for
secondary analysis of these public datasets. The federated learning framework is
designed to demonstrate privacy-preserving methodology and has not been evaluated
for clinical deployment.
