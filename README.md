# Federated Learning for Diabetes Prediction
## Privacy-Preserving ML Across Demographically Heterogeneous Hospital Nodes

**Author:** Rajveer Singh Pall  
**Affiliation:** Gyan Ganga Institute of Technology and Sciences, Jabalpur  
**Extends:** IEEE paper — XGBoost diabetes prediction, AUC = 0.794, NHANES 2015-2020  
**Target venue:** Journal of Biomedical Informatics (JBI, Q1 Elsevier)

---

## What This Project Does

The IEEE paper trained a centralised XGBoost model on NHANES data and found:
- Strong internal AUC (0.794) but significant drop on external validation (0.717)
- Severe fairness gap: elderly patients (>=60) AUC = 0.607 vs young adults = 0.742 (gap = **0.135**)
- Raw patient data cannot be shared across institutions

This project builds a **federated learning system** that trains across 3 simulated hospital nodes without sharing raw patient data, and directly answers:

> *Does federated training on demographically heterogeneous nodes reduce the elderly fairness gap?*

---

## Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| GPU | RTX 4060 8 GB (or better) | CPU-only (3-5x slower) |
| RAM | 16 GB | 8 GB |
| Storage | 5 GB free | 3 GB |
| OS | Windows 10/11, Linux, macOS | - |

Scripts auto-detect CUDA and fall back gracefully to CPU.

---

## Setup

### 1. Create Virtual Environment

```bash
cd D:/Projects/diabetes_prediction_project/federated
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# source venv/bin/activate       # Linux / macOS
```

### 2. Install CUDA PyTorch (RTX 4060 — do this FIRST)

**This is the most important step. The CPU-only torch build wastes your GPU entirely.**

```bash
# Remove CPU-only torch if already installed
pip uninstall torch torchvision torchaudio -y

# Install CUDA 12.4 build (compatible with CUDA 13.x drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: CUDA: True  NVIDIA GeForce RTX 4060 Laptop GPU
```

### 3. Install All Other Dependencies

```bash
pip install flwr opacus xgboost scikit-learn pandas numpy scipy matplotlib joblib
```

### 4. Verify

```bash
python -c "import torch, flwr, xgboost, opacus; print('All OK. CUDA:', torch.cuda.is_available())"
```

---

## Project Structure

```
federated/
|
|-- config_paths.py           <- SINGLE CONFIG: update PROJECT_ROOT only
|-- data_utils.py             <- DataLoader, pin_memory, cuDNN benchmark
|-- nn_model.py               <- DiabetesNet + AMP mixed-precision training
|-- fl_client.py              <- Flower NumPyClient (FedAvg / FedProx)
|
|-- 00_fit_global_scaler.py   <- Run FIRST after data partition
|-- 01_partition_data.py      <- Creates 3 node CSVs from centralised data
|-- 02_centralised_baseline.py  <- XGBoost baseline (GPU-accelerated)
|-- 03_federated_simulation.py  <- FedAvg, FedProx, FedNova (50 rounds)
|-- 03b_fednova_corrected.py    <- FedNova with corrected heterogeneous epochs
|-- 04_differential_privacy.py  <- DP-SGD via Opacus (epsilon = 1.0)
|-- 05_fairness_analysis.py     <- Subgroup AUC, EOD, Youden's J
|-- 07_external_validation.py   <- BRFSS external validation (1.3M rows)
|-- 07_statistical_analysis.py  <- DeLong CIs + bootstrap CIs + paired tests
|
|-- data/                     <- Created by 01_partition_data.py
|   |-- centralised_full.csv         (n~15,650)
|   |-- node_a_young_urban.csv       (n~4,500)
|   |-- node_b_elderly_rural.csv     (n~3,400)
|   `-- node_c_mixed_metro.csv       (n~4,000)
|
|-- artefacts/                <- Created by 00_fit_global_scaler.py
|   `-- global_nhanes_scaler.joblib  (canonical scaler — all scripts load this)
|
|-- models/                   <- Saved weights
|   |-- centralised_xgb.pkl
|   |-- fedavg_weights.pt
|   |-- fedprox_weights.pt
|   `-- fednova_corrected_weights.pt
|
|-- results/                  <- JSON outputs
|   |-- centralised_metrics.json
|   |-- federated_convergence.json
|   |-- fednova_corrected.json
|   |-- fairness_metrics.json
|   |-- external_validation.json
|   |-- auc_confidence_intervals.json  <- manuscript numbers
|   |-- y_true_internal.npy            <- for CI analysis
|   |-- pred_*.npy                     <- model prediction arrays
|   `-- ...
|
`-- plots/                    <- Publication figures (300 dpi PNG)
```

---

## Run Order

```bash
cd D:/Projects/diabetes_prediction_project/federated
source venv/Scripts/activate

# Step 0 — partition raw data into node CSVs (run once)
python 01_partition_data.py           # ~30 sec

# Step 1 — fit global NHANES scaler (must run before anything else)
python 00_fit_global_scaler.py        # ~10 sec

# Step 2 — centralised XGBoost baseline (GPU-accelerated)
python 02_centralised_baseline.py     # ~2 min

# Step 3 — FedAvg, FedProx, FedNova simulation
python 03_federated_simulation.py     # ~8-15 min (GPU) / 30-40 min (CPU)

# Step 4 — corrected FedNova with heterogeneous local epochs
python 03b_fednova_corrected.py       # ~4-8 min (GPU) / 10-15 min (CPU)

# Step 5 — differential privacy
python 04_differential_privacy.py     # ~10-20 min

# Step 6 — fairness analysis (EOD + Youden's J per subgroup)
python 05_fairness_analysis.py        # ~2 min

# Step 7 — external BRFSS validation (1.3M rows)
python 07_external_validation.py      # ~5 min

# Step 8 — statistical CIs (DeLong + bootstrap) — prints manuscript numbers
python 07_statistical_analysis.py     # ~5 min
```

After Step 8, copy the numbers printed under **MANUSCRIPT NUMBER SUMMARY** into the paper
to replace all `X.XXX` placeholders.

---

## Hospital Node Design

| Node | Population | Mean Age | Diabetes % | Simulates |
|------|-----------|----------|-----------|-----------|
| **A - Young Urban** | Minority-heavy | 40.8 | 13.8% | Community Health Clinic |
| **B - Elderly Rural** | White/Hispanic | 69.1 | **28.5%** | Rural Critical Access Hospital |
| **C - Mixed Metro** | Mixed | 45.0 | 16.7% | Academic Medical Center |

Node B is the critical node: 82.4% elderly patients, highest diabetes prevalence,
the population the IEEE paper performed worst on (AUC = 0.607).

---

## Federated Architecture

```
Server (FedAvg / FedProx / FedNova)
    |
    |--- Node A (Young Urban)    --- Local train (tau_A = 5 epochs) --- Return delta_w
    |--- Node B (Elderly Rural)  --- Local train (tau_B = 3 epochs) --- Return delta_w
    `--- Node C (Mixed Metro)    --- Local train (tau_C = 4 epochs) --- Return delta_w

Raw patient data NEVER leaves each node.
Only model weight updates are transmitted.
```

**FedNova epoch assignment** (Wang et al. NeurIPS 2020, Theorem 2):  
Nodes with greater distribution shift use *fewer* local steps to prevent client drift.
- Node B has the highest shift -> tau_B = 3 (fewest)
- Node A has the lowest shift  -> tau_A = 5 (most)

Equal epochs (tau = 5 for all) collapses FedNova to FedAvg algebraically.

---

## Key Phase 2 Bug Fixes (vs original code)

| Fix | File | Impact |
|-----|------|--------|
| Data leakage: Node B scaler | `05_fairness_analysis.py` | Was fit_transform, now transform-only with global scaler |
| Global NHANES scaler | `00_fit_global_scaler.py` | New script — single source of truth for scaling |
| BRFSS scaler | `07_external_validation.py` | Loads global scaler instead of hardcoded path |
| FedNova epoch correction | `03b_fednova_corrected.py` | tau_B corrected from 7 to 3 (backwards in original) |
| Statistical CIs | `07_statistical_analysis.py` | New script — DeLong + bootstrap CIs required by JBI |
| Fairness metrics | `05_fairness_analysis.py` | Added EOD + Youden's J (required by JBI, npj Digital Medicine) |
| Config merge | `config_paths.py` | Single source of truth; removed deprecated XGBoost flag |

---

## GPU Performance Notes (i7-13650HX + RTX 4060)

The following optimisations are automatically active when CUDA PyTorch is installed:

| Optimisation | Where | Effect |
|---|---|---|
| `torch.backends.cudnn.benchmark = True` | `data_utils.get_device()` | Auto-tunes cuDNN kernels |
| FP16 AMP (`torch.autocast`) | `nn_model.train_one_epoch()` | ~1.5x NN throughput |
| `GradScaler` per client | `03_federated_simulation.py`, `03b`, `fl_client.py` | Stable FP16 gradient scaling |
| `pin_memory=True` + `non_blocking=True` | DataLoaders + `.to(device)` | Overlaps H->D transfer with compute |
| `device='cuda'` XGBoost | `02_centralised_baseline.py` | GPU-accelerated tree building |
| Batch size 256 | `config_paths.py` | Saturates RTX 4060 Tensor Cores (vs 64) |

Expected runtimes with CUDA PyTorch vs CPU-only:

| Script | CPU (i7-13650HX) | GPU (RTX 4060) |
|--------|-----------------|----------------|
| `03_federated_simulation.py` | ~35 min | ~10 min |
| `03b_fednova_corrected.py` | ~12 min | ~4 min |
| `04_differential_privacy.py` | ~20 min | ~8 min |
| `02_centralised_baseline.py` | ~2 min | ~40 sec |

---

## Key Research Questions

1. **Accuracy**: Does a federated model match the centralised AUC of 0.717 on BRFSS external validation?
2. **Fairness**: Does federated training reduce the elderly AUC gap of 0.135?
3. **Privacy**: How does (epsilon, delta)-DP affect the accuracy-fairness trade-off?
4. **Aggregation**: Which strategy (FedAvg / FedProx / FedNova) performs best on non-IID clinical data?

---

## Citation

```bibtex
@article{pall2026federated,
  title   = {Federated Learning for Privacy-Preserving Diabetes Prediction:
             Addressing Fairness in Elderly Populations},
  author  = {Pall, Rajveer Singh and Yadav, Sameer and others},
  journal = {Journal of Biomedical Informatics},
  year    = {2026}
}
```

```bibtex
@article{pall2025diabetes,
  title   = {Machine Learning-Based Type 2 Diabetes Risk Prediction with
             External Validation and Fairness Assessment},
  author  = {Pall, Rajveer Singh and Yadav, Sameer and others},
  journal = {IEEE Access},
  year    = {2025}
}
```
