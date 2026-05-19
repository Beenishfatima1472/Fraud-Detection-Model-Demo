# Production Fraud Model — Drift Detection Pipeline

**Beenish Fatima** · Production ML Specialist · [maqasidai.org](https://maqasidai.org)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-FF6600?style=flat)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat&logo=mlflow)
![Status](https://img.shields.io/badge/Status-Production--Ready-0D9488?style=flat)

---

## Overview

A complete, runnable monitoring pipeline demonstrating how production fraud detection models fail silently — and how to detect, measure, and partially recover from that failure without retraining.

This repository accompanies published research on lightweight model monitoring frameworks for production fraud systems. The notebook uses **synthetic data only**. No proprietary datasets or internal systems are referenced.

---

## The problem this solves

Most fraud detection models are deployed and then left unmonitored. Infrastructure metrics stay green while model recall quietly collapses as data distributions shift and fraud patterns evolve. This is known as **silent model failure** — and it is the most common, least monitored failure mode in production fintech systems.

---

## What the notebook demonstrates

| Section | Content |
|---|---|
| Data generation | Synthetic credit card fraud dataset · 0.17% fraud rate · realistic class imbalance |
| Baseline training | Logistic Regression and XGBoost trained on Period 0 |
| Drift simulation | 5 production periods with increasing data and concept drift |
| Drift detection | Population Stability Index (PSI) + Kolmogorov-Smirnov test |
| Performance monitoring | Recall, AUC, FPR tracking across periods |
| Threshold optimization | Cost-sensitive recovery without retraining |
| Experiment tracking | MLflow logging for all metrics and alert levels |

---

## Key results

```
Period 0  →  Recall: 84%   PSI: stable
Period 1-4 →  Recall:  0%   PSI: 0.67–0.75 (CRITICAL)
Period 5  →  Recall: 33%   after threshold optimization, no retraining
```

Drift was detectable **1–2 periods before** performance collapse.
Threshold optimization recovered **33% of fraud recall** with no model changes.

---

## Stack

```
scikit-learn    model training, metrics
xgboost         gradient boosting classifier
scipy           KS test for distribution comparison
mlflow          experiment tracking and audit trail
matplotlib      visualizations
seaborn         heatmaps
pandas/numpy    data manipulation
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/Beenishfatima1472/Fraud-Detection-Model-Demo.git
cd Fraud-Detection-Model-Demo

# Install
pip install -r requirements.txt

# Run
jupyter notebook fraud_model_drift_demo.ipynb

# View MLflow logs
mlflow ui
# → http://localhost:5000
```

---

## File structure

```
├── fraud_model_drift_demo.ipynb   # Main notebook
├── outputs/                       # Generated charts saved here
│   ├── psi_heatmap.png
│   └── silent_failure.png
├── requirements.txt
└── README.md
```

---

## Related work

This demo is part of a broader research program at **Maqasid AI** on production ML reliability, explainability, and AI governance for fintech and GovTech environments.

- **Research paper:** *Lightweight Model Monitoring Framework for Production Fraud Detection Systems* — under journal review
- **MLOps audit service:** [maqasidai.org/mlops-audit](https://maqasidai.org/mlops-audit)
- **LinkedIn:** [Beenish Fatima](https://linkedin.com/in/syeda-beenish-fatima-395bb2263)

---

*Synthetic data only. Methodology reflects production patterns observed in real fintech deployments.*
