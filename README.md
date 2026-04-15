# Federated Learning for Diabetes Prediction
## Privacy-Preserving ML Across Demographically Heterogeneous Hospital Nodes

**Author:** Rajveer Singh Pall  
**Affiliation:** Gyan Ganga Institute of Technology and Sciences, Jabalpur  
**Extends:** [IEEE paper — XGBoost diabetes prediction, AUC=0.794, NHANES 2015–2020]  
**Target venues:** CHIL 2026 · ML4H @ NeurIPS 2026 · JAMIA · npj Digital Medicine

---

## What This Project Does

Your IEEE paper trained a centralised XGBoost model on NHANES data and found:
- Strong internal AUC (0.794) but significant drop on external validation (0.717)
- Severe fairness gap: elderly patients (≥60) AUC = 0.607 vs young adults = 0.742 (gap = **0.135**)
- Patient data cannot be shared across institutions

This project builds a **federated learning system** that trains across 3 simulated hospital nodes without sharing raw patient data, and directly answers:
> *Does federated training on demographically heterogeneous nodes reduce the elderly fairness gap?*

---

## Setup

### 1. Create Environment
```bash
conda create -n federated_diabetes python=3.10 -y
conda activate federated_diabetes
```

### 2. Install Dependencies
```bash
# PyTorch with CUDA (RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# All other packages
pip install flwr==1.8.0 xgboost lightgbm scikit-learn pandas numpy \
            matplotlib seaborn shap opacus joblib
```

### 3. Verify Installation
```bash
python -c "import torch, flwr, xgboost, opacus; print('CUDA:', torch.cuda.is_available())"
```

---

## Project Structure

```
federated/
├── config_paths.py          ← UPDATE PATHS HERE (just 2 lines)
├── data_utils.py            ← Shared data loading + PyTorch Dataset
├── nn_model.py              ← 3-layer PyTorch DiabetesNet
├── fl_client.py             ← Flower NumPyClient (NN + LogReg)
│
├── 01_partition_data.py     ← Run FIRST — creates 3 node CSVs
├── 02_centralised_baseline.py ← Replicates your IEEE paper results
├── 03_federated_simulation.py ← FedAvg, FedProx, FedNova (50 rounds)
├── 04_differential_privacy.py ← Privacy-accuracy-fairness trade-off
├── 05_fairness_analysis.py  ← Full subgroup AUC analysis
├── 06_results_summary.py    ← Master table + publication figures
│
├── data/                    ← Created by script 01
│   ├── node_a_young_urban.csv      (n≈4,500 | young + minority)
│   ├── node_b_elderly_rural.csv    (n≈3,400 | elderly + White)
│   ├── node_c_mixed_metro.csv      (n≈4,000 | mixed)
│   └── centralised_full.csv        (n=15,650 | full dataset)
│
├── results/                 ← Created by scripts 02–06
│   ├── centralised_metrics.json
│   ├── federated_convergence.json
│   ├── dp_results.json
│   ├── fairness_comparison.json
│   ├── master_results_table.csv
│   └── table_latex.tex
│
├── plots/                   ← All publication figures
│   ├── centralised_roc.png
│   ├── fl_convergence.png
│   ├── fairness_age_comparison.png
│   ├── fairness_all_subgroups.png
│   ├── dp_tradeoff.png
│   └── publication_summary_2x2.png
│
└── models/                  ← Saved model weights
    ├── centralised_xgb.pkl
    ├── scaler.pkl
    └── fedprox_final_weights.pt
```

---

## Run Order

```bash
cd C:\diabetes_prediction_project\federated
conda activate federated_diabetes

python 01_partition_data.py       # ~30 seconds
python 02_centralised_baseline.py # ~2 minutes
python 03_federated_simulation.py # ~20–40 minutes (50 rounds × 3 clients)
python 04_differential_privacy.py # ~15 minutes
python 05_fairness_analysis.py    # ~5 minutes
python 06_results_summary.py      # ~1 minute
```

---

## Hospital Node Design

| Node | Population | Age | Diabetes % | Simulates |
|------|-----------|-----|-----------|-----------|
| **A — Young Urban** | Minority-heavy | Mean 40.8 | 13.8% | Community Health Clinic |
| **B — Elderly Rural** | White/Hispanic | Mean 69.1 | **28.5%** | Rural Critical Access Hospital |
| **C — Mixed Metro** | Mixed | Mean 45.0 | 16.7% | Academic Medical Center |

Node B is the critical node — 82.4% elderly patients, highest diabetes prevalence, the population your IEEE paper failed on.

---

## Federated Architecture

```
Server (FedAvg / FedProx / FedNova)
    │
    ├─── Node A (Young Urban)    ─── Local train (5 epochs) ─── Return Δw
    ├─── Node B (Elderly Rural)  ─── Local train (5 epochs) ─── Return Δw  
    └─── Node C (Mixed Metro)    ─── Local train (5 epochs) ─── Return Δw
    
Raw patient data NEVER leaves each node.
Only model weight updates are transmitted.
```

---

## Key Research Questions

1. **Accuracy**: Does a federated model (trained without sharing data) match the centralised AUC of 0.717 on BRFSS external validation?
2. **Fairness**: Does federated training reduce the elderly AUC gap of 0.135 documented in the IEEE paper?
3. **Privacy**: How does adding (ε, δ)-DP affect the accuracy-fairness trade-off?
4. **Aggregation**: Which strategy (FedAvg / FedProx / FedNova) performs best on non-IID clinical data?

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{pall2026federated,
  title     = {Federated Learning for Privacy-Preserving Diabetes Prediction: 
               Addressing Fairness in Elderly Populations},
  author    = {Pall, Rajveer Singh and Yadav, Sameer and others},
  booktitle = {ACM Conference on Health, Inference, and Learning (CHIL)},
  year      = {2026}
}
```

And your IEEE paper:
```bibtex
@article{pall2025diabetes,
  title   = {Machine Learning-Based Type 2 Diabetes Risk Prediction with 
             External Validation and Fairness Assessment},
  author  = {Pall, Rajveer Singh and Yadav, Sameer and others},
  journal = {IEEE ...},
  year    = {2025}
}
```
