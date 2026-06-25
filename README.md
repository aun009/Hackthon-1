# 🏀 GuardianAI — NBA Injury Risk & Load Management

> **AQX Sports Analytics Hackathon 2026**  
> *Protect your roster, not just track it.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-green)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 The Problem

NBA teams lose **tens of millions of dollars annually** to preventable soft-tissue injuries. Most load-management decisions are still based on gut feel or generic "days rest" rules. **GuardianAI changes that.**

## 💡 The Solution

GuardianAI predicts whether a player is at elevated injury risk **before each game** — giving coaches a plain-English recommendation backed by the exact workload metrics driving that call.

## 🔑 Features

| | Feature | Description |
|--|---------|-------------|
| 🃏 | **Risk Cards** | Per-player injury probability with top 3 SHAP factors |
| 📈 | **Team Workload Map** | ACWR scatter plot with danger-zone overlay |
| 🧠 | **SHAP Explainability** | Which features drive each prediction, and by how much |
| 📐 | **Model Validation** | AUC, precision, recall — with temporal split proof |

## 🏋️ Core Analytics: ACWR

The **Acute:Chronic Workload Ratio** is the gold standard in sports science for injury prediction:

```
ACWR = (7-day avg minutes) ÷ (28-day avg minutes)
```

| ACWR | Zone | Risk Level |
|------|------|-----------|
| < 0.8 | Under-trained | Moderate ↑ |
| 0.8 – 1.3 | ✅ Sweet Spot | Low |
| 1.3 – 1.5 | ⚠️ Caution | Elevated |
| **> 1.5** | **🔴 Danger Zone** | **High** |

*Source: Hulin et al. (2016), British Journal of Sports Medicine*

## 🛠️ Tech Stack

```
Python 3.10+
pandas · numpy            — Data processing
nba_api                   — Live NBA game logs (2018–2024)
scikit-learn              — ML pipeline utilities
imbalanced-learn          — SMOTE for class imbalance
xgboost                   — Gradient boosted tree classifier
shap                      — Shapley value explainability
streamlit · plotly        — Interactive dashboard
```

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/guardianai
cd guardianai
pip install -r requirements.txt

# Step 1 — Download NBA game logs (~15 min, API rate-limited)
python src/data/download.py

# Step 2 — Engineer ACWR + rolling features
python src/data/features.py

# Step 3 — Train XGBoost model (~5 min)
python src/model/train.py

# Step 4 — Launch dashboard
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

## 🧠 Model Design

### Temporal Split (no data leakage)
Train on 2018–2022 seasons, test on 2022–2024. Random splits let future games inform past predictions — that's leakage. We simulate real deployment conditions.

### Class Imbalance (SMOTE + scale_pos_weight)
Injuries occur in ~5–8% of games. Without correction, the model learns to predict "never injured" and achieves 95% accuracy while being useless. SMOTE oversamples the minority class; XGBoost's `scale_pos_weight` reinforces this.

### SHAP Explainability
A risk score without a reason isn't actionable. SHAP tells coaches: *"Player X is flagged because his 7-day workload spiked 40% above his 28-day baseline."* That's a decision, not just a number.

## 📊 Dataset

- **Source:** `nba_api` (official NBA Stats API Python wrapper)
- **Players:** 20 high-minutes players across 6 seasons (2018–2024)
- **Features:** 14 engineered (ACWR, rolling load metrics, schedule density, fatigue score)
- **Target:** Proxy injury label — player missed 8+ days after a game

## 📁 Project Structure

```
guardianai/
├── src/
│   ├── data/
│   │   ├── download.py      # NBA API data acquisition
│   │   └── features.py      # ACWR + rolling feature engineering
│   ├── model/
│   │   ├── train.py         # XGBoost training pipeline
│   │   └── explain.py       # SHAP prediction API
│   └── dashboard/
│       ├── app.py           # Main Streamlit application
│       └── components.py    # Reusable UI components
├── data/
│   ├── raw/                 # Downloaded CSVs (gitignored)
│   └── processed/           # ML-ready dataset
├── models/                  # Trained model artifacts
├── .streamlit/config.toml   # Dark theme config
└── requirements.txt
```

## 🏆 Hackathon

**Event:** AQX Sports Analytics Hackathon 2026  
**Category:** Web Application + Predictive Model  
**Impact:** Prevents injuries before they happen — gives coaches a data-driven "who to rest" decision before every game.

---

*Built with ❤️ for the AQX Sports Analytics Hackathon 2026*
