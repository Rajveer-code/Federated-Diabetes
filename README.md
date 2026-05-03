# Federated Learning for Diabetes Prediction
### Privacy-Preserving ML Across Demographically Heterogeneous Hospital Nodes

**Author:** Rajveer Singh Pall  
**Affiliation:** Gyan Ganga Institute of Technology and Sciences, Jabalpur  
**Target venue:** IEEE Journal of Biomedical and Health Informatics (JBHI)

---

## Abstract

This project extends a published centralised XGBoost diabetes-prediction model (internal AUC = 0.794, external AUC = 0.717 on BRFSS) with a federated learning framework that trains across three demographically heterogeneous simulated hospital nodes without sharing raw patient data. We evaluate FedAvg, FedProx (mu = 0.1), and FedNova (tau = {5, 3, 4}) against centralised baselines and assess fairness (age-subgroup AUC gap), distributional robustness (internal-to-external AUC degradation), and privacy (differential privacy epsilon in {0.5, 1.0, 2.0, 5.0}). FedAvg achieves the best external AUC (0.757), surpassing the published centralised model by 0.040 while reducing the elderly fairness gap from 0.135 to 0.069.

---

## Key Results

| Model | Internal AUC [95% CI] | External AUC [95% CI] |
|---|---|---|
| Published XGBoost [2] | 0.794 | 0.717 |
| Centralised XGBoost (rep.) | 0.769 [0.760-0.777] | 0.700 [0.698-0.701] |
| FedAvg | 0.788 [0.779-0.796] | 0.757 [0.756-0.758] |
| FedProx mu=0.1 | 0.785 [0.776-0.793] | 0.752 [0.751-0.753] |
| FedNova tau={5,3,4} | 0.786 [0.778-0.794] | 0.744 [0.743-0.745] |
| Centralised NN | 0.801 [0.782-0.819] | 0.749 [0.748-0.750] |

All CIs are DeLong 95% bootstrap (N=2,000). External validation on BRFSS 2020-2022 (n=1,282,897).

---

## Repository Structure

```
diabetes_prediction_project/
|-- federated/                  # All source code (Python)
|   |-- config_paths.py         # Single source of truth for all paths and hyperparameters
|   |-- 00_fit_global_scaler.py # Step 0: fit NHANES scaler
|   |-- 01_partition_data.py    # Step 1: partition NHANES into 3 nodes
|   |-- 02_centralised_baseline.py  # Step 2: centralised XGBoost baseline
|   |-- 03_federated_simulation.py  # Step 3: FedAvg + FedProx
|   |-- 03b_fednova_corrected.py    # Step 3b: FedNova with corrected normalization
|   |-- 04_differential_privacy.py  # Step 4: DP-SGD epsilon sweep
|   |-- 05_fairness_analysis.py     # Step 5: subgroup fairness metrics
|   |-- 06_results_summary.py       # Step 6: compile master results table
|   |-- 07_external_validation.py   # Step 7: evaluate on BRFSS
|   |-- 07_statistical_analysis.py  # Step 7b: DeLong CIs + McNemar tests
|   |-- 07c_fednova_corrected_external.py  # Step 7c: FedNova external eval
|   |-- 08_compute_eod_table6.py    # Step 8: EOD table + calibration comparison
|   |-- 09_delong_xgboost_external.py  # Step 9: XGBoost external DeLong CI
|   |-- 10_regenerate_all_figures.py   # Step 10: regenerate 13 publication figures
|   |-- 11_patch_manuscript.py         # Step 11: apply all edits to DOCX
|   |-- nn_model.py             # DiabetesNet architecture
|   |-- fl_client.py            # Federated client (FedAvg/FedProx/FedNova)
|   |-- data_utils.py           # Data loading and preprocessing utilities
|   |-- data/                   # CSV/parquet inputs (not committed; see Data section)
|   |-- models/                 # Saved model weights (not committed; regenerate from code)
|   |-- results/                # JSON result files (committed)
|   |-- plots/                  # Figure PNGs (not committed; regenerate via step 10)
|   `-- artefacts/              # Fitted scalers and encoders (not committed)
|-- FL_Diabetes_Manuscript_v4_Final.docx  # Submission-ready manuscript
|-- docs/superpowers/plans/     # Implementation plans
`-- README.md
```

---

## Requirements

```
python >= 3.10
torch >= 2.0
xgboost >= 1.7
scikit-learn >= 1.3
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
scipy >= 1.11
joblib >= 1.3
python-docx >= 1.2.0
lxml >= 4.9
opacus >= 1.4
```

Install via conda (recommended):

```bash
conda create -n fl-diabetes python=3.10
conda activate fl-diabetes
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xgboost scikit-learn numpy pandas matplotlib scipy joblib python-docx lxml opacus
```

---

## Data

Data files are **not committed** to this repository because they are large and publicly available.

### NHANES 2015-2020 (internal training/test)

Download from the CDC NHANES website: https://www.cdc.gov/nchs/nhanes/index.htm

The processed file is expected at: `federated/data/centralised_full.csv`

Columns required: `RIDAGEYR`, `RIAGENDR`, `RIDRETH3`, `BMXBMI`, `SMOKING`, `PHYS_ACTIVITY`, `HEART_ATTACK`, `STROKE`, `DIABETES`

### BRFSS 2020-2022 (external validation)

Download from the CDC BRFSS website: https://www.cdc.gov/brfss/annual_data/annual_data.htm

The processed file is expected at: `C:\diabetes_prediction_project\data\03_processed\brfss_final.csv`

Column mapping (applied automatically in scripts):
- `Age -> RIDAGEYR`, `Gender -> RIAGENDR`, `Race_Ethnicity -> RIDRETH3`, `BMI -> BMXBMI`
- `Smoking_Status -> SMOKING`, `Physical_Activity -> PHYS_ACTIVITY`
- `History_Heart_Attack -> HEART_ATTACK`, `History_Stroke -> STROKE`, `Diabetes_Outcome -> DIABETES`

---

## Reproducing the Results

All scripts run from `federated/` (or set `PROJECT_ROOT` in `config_paths.py` to your machine's path).

```bash
cd D:/Projects/diabetes_prediction_project/federated

# Step 0: Fit the global NHANES scaler (must run first)
python 00_fit_global_scaler.py

# Step 1: Partition NHANES into 3 demographically heterogeneous nodes
python 01_partition_data.py

# Step 2: Train centralised XGBoost baseline
python 02_centralised_baseline.py

# Step 3: Federated simulation (FedAvg + FedProx, 50 rounds)
python 03_federated_simulation.py

# Step 3b: FedNova with corrected normalization
python 03b_fednova_corrected.py

# Step 4: Differential privacy epsilon sweep
python 04_differential_privacy.py

# Step 5: Fairness analysis (age, sex, BMI subgroups)
python 05_fairness_analysis.py

# Step 6: Compile master results table
python 06_results_summary.py

# Step 7: External validation on BRFSS
python 07_external_validation.py
python 07_statistical_analysis.py
python 07c_fednova_corrected_external.py

# Step 8: Verify EOD table and calibration comparison
python 08_compute_eod_table6.py

# Step 9: Verify XGBoost external DeLong CI
python 09_delong_xgboost_external.py

# Step 10: Regenerate all 13 publication figures
python 10_regenerate_all_figures.py

# Step 11: Apply all manuscript edits (run from project root)
cd D:/Projects/diabetes_prediction_project
python federated/11_patch_manuscript.py
```

Expected total runtime: ~2-4 hours on GPU (RTX 4060 or equivalent); ~8-12 hours CPU-only.

---

## Hyperparameters

All hyperparameters are defined in `federated/config_paths.py`:

| Parameter | Value | Notes |
|---|---|---|
| FL rounds | 50 | |
| FL clients | 3 | Node A (young urban), Node B (elderly rural), Node C (mixed metro) |
| Local epochs | 5 (FedAvg/FedProx) | FedNova: {5, 3, 4} per node |
| Batch size | 256 | Use 64 for CPU-only |
| Learning rate | 1e-3 | Adam optimizer |
| NN architecture | [64, 32, 16] + dropout=0.3 | DiabetesNet |
| FedProx mu | 0.1 | Sensitivity tested: {0.01, 0.05, 0.1, 0.5} |
| FedNova tau | {5, 3, 4} | Node A, B, C (inversely proportional to distribution shift) |
| DP epsilon | {0.5, 1.0, 2.0, 5.0, inf} | Opacus DP-SGD, delta=1e-5 |
| XGBoost | lr=0.11, depth=6, n=240 | Replicated from published baseline |

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{pall2026federated,
  title   = {Privacy-Preserving Federated Learning for Diabetes Prediction
             Across Demographically Heterogeneous Hospital Nodes},
  author  = {Pall, Rajveer Singh},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Ethics and Data Usage

- **NHANES** and **BRFSS** are publicly available de-identified survey datasets released by the US Centers for Disease Control and Prevention (CDC) under open data policies. No IRB approval is required for secondary analysis of these datasets.
- No individual-level data is committed to this repository.
- The federated learning simulation uses partitioned node data stored locally and is intended only to demonstrate privacy-preserving ML methodology on public health data.

---

## Licence

MIT License. See `LICENSE` for details.
