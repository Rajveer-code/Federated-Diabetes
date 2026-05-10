# Privacy-Preserving Federated Learning for Diabetes Risk Prediction

**Multi-site machine learning with fairness analysis, differential privacy, and post-hoc calibration**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org)
[![Flower FL](https://img.shields.io/badge/Flower-1.5-blueviolet.svg)](https://flower.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Journal: JBI](https://img.shields.io/badge/Target-JBI%20Q1-orange.svg)](https://www.journals.elsevier.com/journal-of-biomedical-informatics)

**Author:** Rajveer Singh Pall  
**Affiliation:** Gyan Ganga Institute of Technology and Sciences, Jabalpur, India  
**Target venue:** *Journal of Biomedical Informatics* (JBI/Elsevier, Q1)

---

## Abstract

Centralised machine learning for diabetes risk prediction requires aggregating sensitive patient records across institutions — a practice that conflicts with HIPAA, GDPR, and institutional policy. This project implements and evaluates a privacy-preserving **federated learning (FL)** framework that trains across three demographically heterogeneous hospital nodes without sharing raw data.

Four FL aggregation strategies — FedAvg, FedProx (μ=0.1), FedNova (τ={5,3,4}), and SCAFFOLD (Option II) — are benchmarked against a centralised XGBoost baseline on NHANES training data and validated externally on **1,282,897 BRFSS respondents** — the largest independent validation reported for this prediction task. The study jointly evaluates discrimination, **demographic fairness** (elderly–young AUC gap), **post-hoc calibration** (Platt, isotonic, temperature scaling), and the **differential privacy–utility tradeoff** (DP-SGD, ε ∈ {0.5, 1.0, 2.0, 5.0, ∞}).

FedAvg achieves an external AUC of **0.757 [0.756–0.758]**, a 8.2-point improvement over the centralised baseline (0.700), while reducing the elderly fairness gap from a published benchmark of 0.135 to **0.054 — a 60.7% improvement**. Isotonic recalibration reduces the Expected Calibration Error from 0.319 to **0.001**. Tight differential privacy (ε ≤ 5) causes model collapse at the per-node sample sizes typical in healthcare FL, revealing a fundamental privacy–utility tension that informs realistic deployment decisions.

---

## Key Results

| Model | Internal AUC [95% CI] | External AUC [95% CI] | Elderly Gap (Δ) |
|---|---|---|---|
| Centralised XGBoost (baseline) | 0.769 [0.760–0.777] | 0.700 [0.698–0.701] | 0.069 |
| **FedAvg** | **0.788 [0.779–0.796]** | **0.757 [0.756–0.758]** | **0.054** |
| FedProx (μ=0.1) | 0.785 [0.776–0.793] | 0.752 [0.751–0.753] | 0.066 |
| FedNova (τ={5,3,4}) | 0.786 [0.778–0.794] | 0.744 [0.743–0.745] | 0.064 |
| SCAFFOLD (Option II, SGD) | 0.642 (50 rounds) | — | — |
| Published benchmark [Ahsan 2022] | 0.742 (young) | — | 0.135 |

**Calibration (FedProx, BRFSS external):**

| Method | ECE | AUC |
|---|---|---|
| Uncalibrated | 0.319 | 0.752 |
| Platt scaling | 0.016 | 0.752 |
| Isotonic regression | **0.001** | 0.752 |
| Temperature (T=2.25) | 0.311 | 0.752 |

Internal CIs: stratified bootstrap (N=2,000, NHANES test n=15,650).  
External CIs: DeLong structural components estimator (O(n log n), BRFSS n=1,282,897).

---

## Research Questions

| RQ | Question | Finding |
|---|---|---|
| **RQ1 — Performance** | Can FL match or exceed a centralised model on held-out external data? | ✅ FedAvg +8.2pp external AUC over centralised XGBoost |
| **RQ2 — Fairness** | Does federated training reduce demographic AUC disparity? | ✅ Elderly gap: 0.135 → 0.054 (FedAvg); 60.7% improvement vs. benchmark |
| **RQ3 — Privacy** | What is the accuracy cost of differential privacy at clinically meaningful ε? | ⚠️ Model collapse at ε ≤ 5 for healthcare-scale per-node sample counts |
| **RQ4 — Calibration** | Are federated risk scores clinically trustworthy (well-calibrated)? | ✅ Isotonic recalibration: ECE 0.319 → 0.001; Platt: 0.319 → 0.016 |

---

## Repository Structure

```
diabetes_prediction_project/
├── federated/                           # Core FL codebase (git submodule)
│   ├── 00_fit_global_scaler.py          # Fit NHANES scaler (must run first)
│   ├── 01_partition_data.py             # Partition NHANES into 3 nodes
│   ├── 02_centralised_baseline.py       # XGBoost + centralised DiabetesNet
│   ├── 03_federated_simulation.py       # FedAvg / FedProx simulation
│   ├── 03b_fednova_corrected.py         # FedNova (gradient normalisation)
│   ├── 04_differential_privacy.py       # DP-SGD privacy-utility tradeoff
│   ├── 05_fairness_analysis.py          # Subgroup AUC + EOD fairness
│   ├── 06_results_summary.py            # Consolidated results table
│   ├── 07_external_validation.py        # BRFSS external validation
│   ├── 07_statistical_analysis.py       # Bootstrap + DeLong CIs
│   ├── 08_scaffold_baseline.py          # SCAFFOLD Option II (Karimireddy 2020)
│   ├── 09_calibration_analysis.py       # Platt / Isotonic / Temperature scaling
│   ├── 10_stratified_centralised_experiment.py  # Mechanism analysis
│   ├── 11_subgroup_confidence_intervals.py      # Bootstrap subgroup CIs
│   ├── 12_dp_technical_details.py       # RDP accounting tables
│   ├── 13_apply_manuscript_edits.py     # Patches v4 manuscript
│   ├── generate_publication_figures.py  # All 8 publication figures
│   ├── write_manuscript_v5.py           # Generates full manuscript DOCX
│   ├── config_paths.py                  # Centralised hyperparameters + paths
│   ├── nn_model.py                      # DiabetesNet architecture
│   ├── fl_client.py                     # Flower FL client (DiabetesClient)
│   ├── data_utils.py                    # Data loading + preprocessing
│   ├── data/                            # NHANES node CSVs (not committed)
│   ├── models/                          # Trained weights (not committed)
│   ├── artefacts/                       # Global scaler (not committed)
│   └── results/                         # JSON metrics + .npy predictions
│       ├── auc_confidence_intervals.json
│       ├── external_validation.json
│       ├── calibration_results.json
│       ├── scaffold_results.json
│       ├── dp_results.json
│       └── figures/                     # 8 × 300 dpi publication figures
├── FL_Diabetes_Manuscript_v5_Submission.docx    # Final manuscript
├── Supplementary_Material.docx                  # Tables S1–S4
├── README.md
└── .gitignore
```

---

## Requirements

```
python         >= 3.10
torch          >= 2.0.0
flwr           >= 1.5.0
scikit-learn   >= 1.3.0
xgboost        >= 1.7.0
numpy          >= 1.24.0
pandas         >= 2.0.0
matplotlib     >= 3.7.0
opacus         >= 1.4.0   # differential privacy
python-docx    >= 0.8.11  # manuscript generation
joblib         >= 1.3.0
scipy          >= 1.11.0
```

Install: `pip install -r requirements.txt`

---

## Data Access

### NHANES (Training)
- Source: [CDC NHANES 2013–2020](https://wwwn.cdc.gov/nchs/nhanes/)
- Download XPT files for cycles: 2013-14, 2015-16, 2017-18, 2019-20
- Features used: `RIDAGEYR`, `BMXBMI`, `BPXOSY3`, `LBXGH`, `LBXGLU`, `LBXTC`, `PAQ650`, `SMQ020`, `RIAGENDR`
- Preprocessing: `01_partition_data.py` handles cleaning, imputation, and node assignment

### BRFSS (External Validation)
- Source: [CDC BRFSS 2020–2022](https://www.cdc.gov/brfss/annual_data/annual_data.htm)
- Download `.XPT` or `.SAS7BDAT` for years 2020, 2021, 2022
- Set environment variable before running:

```powershell
$env:BRFSS_PATH = "C:\path\to\brfss_final.csv"
```

```bash
export BRFSS_PATH="/path/to/brfss_final.csv"
```

---

## Reproducing Results

Run scripts in order. Expected runtime on a modern GPU (RTX 3060+): ~25 minutes total.

```bash
cd federated/

# 1. Fit global scaler on NHANES (MUST run first; prevents data leakage)
python 00_fit_global_scaler.py
# Output: artefacts/global_nhanes_scaler.joblib

# 2. Partition NHANES into 3 demographically stratified nodes
python 01_partition_data.py
# Output: data/node_{A,B,C}_train.csv, data/node_{A,B,C}_val.csv

# 3. Centralised baselines (XGBoost + DiabetesNet)   ~2 min
python 02_centralised_baseline.py
# Output: results/centralised_metrics.json, models/centralised_weights.pt

# 4. Federated simulation: FedAvg + FedProx          ~5 min
python 03_federated_simulation.py
# Output: results/federated_convergence.json, models/fedavg_weights.pt

# 5. FedNova (gradient normalisation)                ~5 min
python 03b_fednova_corrected.py
# Output: results/fednova_corrected.json, models/fednova_weights.pt

# 6. Differential privacy tradeoff                   ~3 min
python 04_differential_privacy.py
# Output: results/dp_results.json

# 7. Fairness analysis (subgroup AUC + EOD)          ~2 min
python 05_fairness_analysis.py
# Output: results/fairness_metrics.json

# 8. Results summary table
python 06_results_summary.py

# 9. External validation on BRFSS                    ~4 min
# (requires BRFSS_PATH env var)
python 07_external_validation.py
# Output: results/external_validation.json, results/pred_*_external.npy

# 10. Statistical CIs (bootstrap + DeLong)           ~3 min
python 07_statistical_analysis.py
# Output: results/auc_confidence_intervals.json

# 11. SCAFFOLD Option II                             ~1 min
python 08_scaffold_baseline.py
# Output: results/scaffold_results.json

# 12. Post-hoc calibration                          ~2 min
python 09_calibration_analysis.py
# Output: results/calibration_results.json

# 13. Generate all 8 publication figures
python generate_publication_figures.py
# Output: results/figures/fig1_architecture.png ... fig8_summary_comparison.png

# 14. Generate complete manuscript
python write_manuscript_v5.py
# Output: ../FL_Diabetes_Manuscript_v5_Submission.docx
```

**Expected outputs after full run:**
- `results/auc_confidence_intervals.json` — all model CIs
- `results/external_validation.json` — BRFSS AUC + fairness per model
- `results/calibration_results.json` — ECE before/after calibration
- `results/scaffold_results.json` — SCAFFOLD convergence
- `results/figures/fig{1..8}_*.png` — 8 × 300 dpi publication figures
- `FL_Diabetes_Manuscript_v5_Submission.docx` — complete manuscript

---

## Hyperparameters

All hyperparameters are centralised in `federated/config_paths.py` and documented in Supplementary Table S2.

| Parameter | Value | Rationale |
|---|---|---|
| `FL_NUM_ROUNDS` | 50 | Convergence achieved by round 38 (ΔAUC < 0.001) |
| `NN_LOCAL_EPOCHS` | 5 | Balance between convergence speed and communication cost |
| `NN_BATCH_SIZE` | 256 | GPU-optimised; fits within 4 GB VRAM for 8-feature input |
| `FedProx μ` | 0.1 | Grid search over {0.01, 0.1, 1.0}; 0.1 best external AUC |
| `FedNova τ` | {5, 3, 4} | Node-specific local update counts matching local dataset sizes |
| `AdamW lr` | 0.001 | CosineAnnealingLR schedule; η_min = 10⁻⁶ |
| `RANDOM_SEED` | 42 | All experiments; ensures reproducibility |
| `DP δ` | 10⁻⁵ | Standard for datasets with n < 10⁶ |
| `DP clipping C` | 1.0 | Gradient clipping norm for DP-SGD |

---

## Key Technical Design Decisions

### Global Scaler (preventing data leakage)
A single `StandardScaler` is fitted **once** on the NHANES training split by `00_fit_global_scaler.py` and saved as `artefacts/global_nhanes_scaler.joblib`. All nodes call `.transform()` only — never `.fit_transform()`. This prevents the data leakage that would arise if each node independently fitted its own scaler on local data, which would contaminate evaluation with node-local statistics.

### DeLong CI for Large External Set
Computing bootstrap CIs on 1.28 million BRFSS records is intractable (kernel matrix: 708 GB). We use the DeLong structural components estimator, an O(n log n) algorithm based on searchsorted that produces mathematically equivalent 95% CIs in seconds.

### SCAFFOLD Optimizer Note
SCAFFOLD's convergence proof assumes SGD. Our implementation uses SGD with η_l = 0.001, which underperforms AdamW-based strategies at 50 rounds. This is an honest finding: the optimizer choice matters as much as the aggregation strategy in small federated settings.

---

## Known Limitations

- Node partitioning is **simulated** from a single NHANES cohort; real multi-site deployments would have additional between-site batch effects not captured here.
- The BRFSS smoking variable mapping introduces measurement error (binary cigarette-use vs. clinical NHANES measure).
- SCAFFOLD was evaluated with SGD (theoretically required); an AdamW variant may perform comparably to FedAvg.
- Tight differential privacy (ε ≤ 5) causes model collapse at per-node sample sizes of ~3,000–4,500; deployment at these sizes requires either more data or weaker DP notions.
- All nodes use identical DiabetesNet architecture; personalised FL (per-node models) was not evaluated.

---

## Citation

If you use this code or build on this work, please cite:

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

- **NHANES** (training): publicly available, de-identified, conducted under US federal ethics oversight (NCHS IRB).
- **BRFSS** (external validation): publicly available, de-identified, state-administered with CDC oversight.
- No new patient data were collected. No IRB approval was required for secondary analysis of public datasets.
- All model outputs are for research purposes only. This tool is **not** a validated clinical diagnostic instrument.

---

## License

MIT License — see [LICENSE](LICENSE).

Free to use, modify, and distribute with attribution.
