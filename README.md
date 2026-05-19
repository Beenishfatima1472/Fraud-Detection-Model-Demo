# Lightweight Model Monitoring Framework for Production Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-FF6600?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)
![Status](https://img.shields.io/badge/Paper-Under%20Review%20%40%20Wiley-blueviolet?style=flat-square)

**End-to-end implementation of the monitoring framework described in:**
*"Lightweight Model Monitoring Framework for Production Fraud Detection Systems"*
— Under review, Wiley Journal

[Overview](#overview) · [Architecture](#architecture) · [Notebook Walkthrough](#notebook-walkthrough) · [Results](#results) · [Run It](#run-it) · [Citation](#citation)

</div>

---

## Overview

Most fraud detection models are deployed and then forgotten. Infrastructure stays green. Dashboards stay clean. **Losses silently grow.**

This repository implements a production-grade monitoring framework that detects and partially recovers from silent model failure — without retraining — by combining statistical drift detection with cost-sensitive threshold optimization.

The companion notebook `fraud_model_drift_demo.ipynb` demonstrates the full monitoring loop on synthetic data: from a healthy baseline, through five periods of increasing drift, to silent recall collapse, to partial recovery via threshold recalibration.

**Key result:** In the simulated scenario, XGBoost recall drops from **91% → 0%** across five production periods while AUC remains deceptively stable. The monitoring framework detects deterioration at Period 2 via PSI and KS tests, and recovers recall to **47%** without any retraining.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION INFERENCE PIPELINE                     │
│                                                                      │
│  Incoming          Feature          Prediction         Business      │
│  Transactions ──► Extraction ──────► Model ──────────► Decision     │
│                       │                │                             │
└───────────────────────┼────────────────┼─────────────────────────────┘
                        │                │
                ┌───────▼────────────────▼───────┐
                │     MONITORING LAYER           │
                │                                │
                │  ┌──────────────┐              │
                │  │  PSI Monitor │◄─ Feature    │
                │  │  (per-feat)  │   Batches    │
                │  └──────┬───────┘              │
                │         │  PSI ≥ 0.25?         │
                │  ┌──────▼───────┐              │
                │  │  KS Test     │  Drift        │
                │  │  Validator   │  Confirmed    │
                │  └──────┬───────┘              │
                │         │                      │
                │  ┌──────▼───────┐              │
                │  │  Threshold   │  θ* updated  │
                │  │  Optimizer   │  (no retrain)│
                │  └──────┬───────┘              │
                │         │                      │
                │  ┌──────▼───────┐              │
                │  │  MLflow      │  Full audit  │
                │  │  Logging     │  trail       │
                │  └──────────────┘              │
                └────────────────────────────────┘
```

### Components

| Component | Method | Trigger |
|-----------|--------|---------|
| **Feature Drift Detection** | Population Stability Index (PSI) | PSI ≥ 0.25 per feature |
| **Distribution Shift Validation** | Kolmogorov-Smirnov test | p-value ≤ 0.05 |
| **Performance Monitoring** | Recall, Precision, F1, AUC per period | Recall drop ≥ 15% from baseline |
| **Threshold Recalibration** | Cost-sensitive optimization: `θ* = argmax[Recall − λ·FPR]` | Triggered on confirmed drift |
| **Experiment Tracking** | MLflow (local or remote) | Every production period |

### PSI Thresholds (Industry Standard)
```
PSI < 0.10   →  Stable          No action required
PSI 0.10–0.25 →  Moderate drift  Monitor closely; prepare fallback
PSI ≥ 0.25   →  Significant drift  Trigger recalibration or retraining
```

---

## Notebook Walkthrough

The notebook `fraud_model_drift_demo.ipynb` is structured as a reproducible end-to-end experiment:

### Section 1 — Synthetic Data Generation
Generates a credit card fraud dataset with realistic properties:
- 50,000 transactions per period, 30 anonymized features (V1–V28 + Time, Amount)
- Extreme class imbalance: ~0.17% fraud rate (matching real-world benchmarks)
- Reproducible with `RANDOM_SEED = 42`

### Section 2 — Baseline Model Training
Trains both **Logistic Regression** and **XGBoost** on Period 0 data. Both models remain frozen across all production periods — this simulates the real-world scenario where models are not continuously retrained.

### Section 3 — Production Drift Injection
Injects three simultaneous drift mechanisms across five periods:

| Drift Type | Mechanism | Real-World Analogue |
|-----------|-----------|---------------------|
| **Feature drift** | Gaussian noise added to V1–V5 | New POS terminal rollout |
| **Concept drift** | Fraud pattern shifts over time | Fraud ring adapts tactics |
| **Covariate shift** | Amount distribution changes | Inflation, seasonal spend |

Drift severity scales linearly with period number.

### Section 4 — Drift Detection (PSI + KS)
Computes PSI for each feature against the baseline reference window. Validates flagged features using KS test. Outputs a PSI heatmap across all five periods showing which features drift first and fastest.

### Section 5 — Silent Failure Demonstration
The critical result: **AUC remains stable (looks healthy) while Recall collapses to 0%**. This is the silent failure pattern that standard infrastructure monitoring (latency, error rates) completely misses.

### Section 6 — Threshold Recalibration
Cost-sensitive threshold optimization recovers partial performance without retraining. The optimizer sweeps threshold candidates and selects:
```
θ* = argmax[Recall(θ) − λ · FPR(θ)]
```
where `λ = 0.2` weights the false-positive cost relative to recall recovery.

### Section 7 — MLflow Tracking
Logs all drift metrics, model performance, and recalibrated thresholds to MLflow for full auditability. Every production period generates a logged run with PSI scores, KS p-values, recall/precision/F1/AUC, and the applied threshold.

### Section 8 — Production Recommendations
Summarizes the full monitoring loop and maps findings to production action triggers.

---

## Results

### Silent Failure — XGBoost (Fixed Threshold = 0.5)

| Period | AUC | Recall | Precision | F1 | Status |
|--------|-----|--------|-----------|----|--------|
| 0 (Baseline) | 0.96 | 0.91 | 0.83 | 0.87 | ✅ Healthy |
| 1 | 0.94 | 0.74 | 0.79 | 0.76 | ⚠️ Degrading |
| 2 | 0.91 | 0.41 | 0.76 | 0.53 | 🔴 Alert |
| 3 | 0.87 | 0.12 | 0.71 | 0.20 | 🔴 Critical |
| 4 | 0.84 | 0.03 | 0.67 | 0.06 | 🔴 Failed |
| 5 | 0.82 | **0.00** | — | — | 🔴 Silent failure |

*AUC deceives. Recall tells the truth.*

### After Threshold Recalibration (Period 5, λ = 0.2)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Recall | 0.00 | 0.47 | **+47pp** |
| FPR | 0.01 | 0.08 | +7pp |
| F1 | 0.00 | 0.54 | +54pp |

Partial recovery with no retraining, no model changes, no infrastructure deployment.

---

## Run It

### Prerequisites

```bash
pip install scikit-learn xgboost mlflow scipy matplotlib seaborn pandas numpy
```

Python 3.9+ required.

### Quickstart

```bash
git clone https://github.com/Beenishfatima1472/fraud-model-drift-demo
cd fraud-model-drift-demo
jupyter notebook fraud_model_drift_demo.ipynb
```

### MLflow UI (optional)

After running the notebook, view the full experiment log:

```bash
mlflow ui
# Open http://localhost:5000
```

All drift metrics, performance curves, and threshold decisions are logged per period under the `fraud_drift_monitoring` experiment.

### No external data required

The notebook generates all data synthetically. Nothing to download.

---

## Repository Structure

```
fraud-model-drift-demo/
├── fraud_model_drift_demo.ipynb   # Full end-to-end notebook
├── README.md
├── requirements.txt
└── mlruns/                        # Generated by MLflow on first run
```

---

## Related Work

This notebook accompanies a broader production MLOps audit framework. For the full system — including Platt calibration, stratified validation, real-time PSI pipelines, and enterprise integration guides — see:

- **MaqasidAI MLOps Audit Service** → [maqasidai.org/mlops-audit](https://maqasidai.org/mlops-audit)
- **Halal-AI-Auditor (MACI Framework)** → [github.com/Beenishfatima1472/Halal-AI-Auditor](https://github.com/Beenishfatima1472/Halal-AI-Auditor)

---

## Citation

If you use this framework in research or production, please cite:

```bibtex
@article{fatima2025lightweightmonitoring,
  title   = {Lightweight Model Monitoring Framework for Production Fraud Detection Systems},
  author  = {Fatima, Syeda Beenish},
  journal = {Wiley Journal},
  year    = {2025},
  note    = {Under review}
}
```

#Contact

**GitHub:** https://github.com/Beenishfatima1472
LinkedIn: https://www.linkedin.com/in/syeda-beenish-fatima-395bb2263/

---


---
## License

Apache 2.0 — open for academic and commercial use with attribution.

---

<div align="center">

Built by [Syeda Beenish Fatima](https://maqasidai.org) · AI Governance Researcher · Founder, MaqasidAI

</div>
