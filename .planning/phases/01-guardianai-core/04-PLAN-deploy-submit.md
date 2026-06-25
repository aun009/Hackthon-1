---
phase: 4
plan: 04-deploy-submit
wave: 1
type: execute
depends_on: [03-dashboard]
files_modified:
  - README.md
  - requirements.txt
  - .streamlit/secrets.toml.example
autonomous: true
---

<objective>
Write the project README, polish the submission description, deploy to Streamlit Cloud,
and push to a public GitHub repo. All hackathon submission requirements met.
</objective>

<tasks>

## Task 1 — README.md (Judge-facing document)

<read_first>
- .planning/PROJECT.md
</read_first>

<action>
Overwrite `README.md` with a polished, judge-facing README:

```markdown
# 🏀 GuardianAI — NBA Injury Risk & Load Management

> **AQX Sports Analytics Hackathon 2026**  
> *Protect your roster, not just track it.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-green)](https://shap.readthedocs.io)

## 🎯 The Problem

NBA teams lose **tens of millions of dollars annually** to preventable soft-tissue injuries — hamstrings, ankles, load-related fatigue tears. Most load-management decisions are still based on gut feel or generic "days rest" rules. **GuardianAI changes that.**

## 💡 The Solution

GuardianAI uses **XGBoost + SHAP explainability** to predict whether a player is at elevated injury risk before each game. It gives coaches a plain-English recommendation — "REST" or "CLEAR" — backed by the exact workload metrics driving that call.

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| 🔴 **Risk Cards** | Per-player injury probability with top 3 SHAP factors |
| 📈 **ACWR Scatter** | Team workload map with danger-zone overlay |
| 🧠 **SHAP Explainability** | Which features drive each prediction |
| 📐 **Model Validation** | AUC, precision, recall on held-out seasons |

## 🏋️ Core Analytics: ACWR

The **Acute:Chronic Workload Ratio** is the gold standard in sports science for injury prediction:

```
ACWR = (7-day avg minutes) ÷ (28-day avg minutes)
```

| ACWR | Zone | Risk |
|------|------|------|
| < 0.8 | Under-trained | Moderate |
| 0.8 – 1.3 | ✅ Sweet Spot | Low |
| 1.3 – 1.5 | ⚠️ Caution | Elevated |
| > 1.5 | 🔴 Danger Zone | **High** |

## 🛠️ Tech Stack

```
Python 3.10+
pandas · numpy               # Data processing
nba_api                      # Live NBA game logs
scikit-learn · imbalanced-learn  # ML utilities
xgboost                      # Gradient boosted trees
shap                         # Explainability
streamlit · plotly           # Dashboard
```

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/guardianai
cd guardianai
pip install -r requirements.txt

# Step 1: Download NBA data (15–20 min, API rate-limited)
python src/data/download.py

# Step 2: Engineer features
python src/data/features.py

# Step 3: Train model (~5 min)
python src/model/train.py

# Step 4: Launch dashboard
streamlit run src/dashboard/app.py
```

## 🧪 Model Design Decisions

### Why Temporal Split?
Random splits cause **data leakage** — future games inform past predictions. We train on 2018–2022 seasons, test on 2022–2024. This simulates real deployment conditions.

### Why SMOTE?
Injuries are rare (~5–8% of games). Without handling class imbalance, the model learns to predict "no injury" 100% of the time and achieve 95% accuracy while being useless. SMOTE + `scale_pos_weight` balances this.

### Why SHAP?
A model that says "75% injury risk" is useless without knowing *why*. SHAP tells coaches: "Player X is flagged because his ACWR spiked 40% above his 28-day baseline." That's actionable.

## 📊 Dataset

- **Source:** `nba_api` (official NBA Stats API wrapper)
- **Players:** 20 high-minutes stars across 6 seasons (2018–2024)
- **Target:** Proxy injury label (player missed 8+ days after a game)
- **Features:** 14 engineered (ACWR, rolling minutes, fatigue score, schedule density, etc.)

## 🏆 Hackathon Submission

- **Event:** AQX Sports Analytics Hackathon 2026
- **Category:** Web Application + Predictive Model
- **Impact:** Provides coaches with a decision-support tool to reduce injury incidence — not just track statistics, but prevent outcomes.

---

*Built with ❤️ for the AQX Sports Analytics Hackathon 2026*
```
</action>

<acceptance_criteria>
- `README.md` contains `GuardianAI`
- `README.md` contains `ACWR`
- `README.md` contains `XGBoost`
- `README.md` contains `SHAP`
- `README.md` contains `## 🚀 Run Locally`
- `README.md` contains `python src/data/download.py`
- `README.md` contains `streamlit run src/dashboard/app.py`
- `README.md` contains `temporal`
</acceptance_criteria>

---

## Task 2 — Streamlit Cloud Deployment

<read_first>
- README.md
- requirements.txt
</read_first>

<action>
Deploy to Streamlit Cloud:

1. Push all code to a **public GitHub repository**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/guardianai.git
git add .
git commit -m "feat: GuardianAI — NBA injury risk dashboard (AQX hackathon)"
git push -u origin master
```

2. Go to https://share.streamlit.io
   - Click "New app"
   - Select your GitHub repo
   - Set main file path: `src/dashboard/app.py`
   - Click "Deploy"

3. **IMPORTANT:** The NBA data download and model training need to be pre-run. For Streamlit Cloud deployment, commit `data/processed/ml_dataset.csv` and `models/` directory (remove them from `.gitignore` for this).

Alternative for quick deploy:
```bash
# Temporarily allow data files in git
echo "!data/processed/*.csv" >> .gitignore
echo "!models/*.pkl" >> .gitignore
echo "!models/*.json" >> .gitignore
echo "!models/*.csv" >> .gitignore

git add data/processed/ml_dataset.csv models/
git commit -m "chore: add precomputed model artifacts for deployment"
git push
```

4. Update README badge with actual Streamlit URL once deployed.
</action>

<acceptance_criteria>
- GitHub repository is public
- All source files committed
- `README.md` has actual Streamlit URL (not placeholder)
- Streamlit app accessible at the shared URL
</acceptance_criteria>

---

## Task 3 — Hackathon Submission Description

<read_first>
- README.md
- .planning/PROJECT.md
</read_first>

<action>
Write the following as the Devpost/hackathon submission description:

---

**Project Name:** GuardianAI

**Short Description (under 200 chars):**
> GuardianAI predicts NBA player injury risk using ACWR workload analytics + XGBoost ML. Gives coaches a "rest or play" recommendation backed by SHAP explainability — before each game.

**Full Description:**

**The Problem**
NBA teams lose tens of millions annually to preventable soft-tissue injuries. Most load-management decisions rely on gut feel. GuardianAI fixes that with data.

**The Solution**
GuardianAI is a full-stack sports analytics application combining:
- **XGBoost** injury prediction model trained on 6 seasons of NBA game logs
- **ACWR (Acute:Chronic Workload Ratio)** — the gold standard sports-science injury metric
- **SHAP explainability** — so coaches know *why* a player is flagged, not just *that* they are
- **Streamlit dashboard** — dark-themed, interactive, deployable in one click

**Key Analytical Insight**
The ACWR (7-day avg ÷ 28-day avg minutes) is a proven injury predictor from sports science literature. Research shows athletes with ACWR > 1.5 have significantly elevated soft-tissue injury risk. We extend this with 14 additional features: back-to-back games, schedule density, rolling efficiency metrics, and a composite fatigue score.

**Actionable Impact**
A coach opens GuardianAI before a game and sees:
- 🔴 Player A: 74% risk — "REST: ACWR=1.67 + back-to-back"
- 🟡 Player B: 41% risk — "MONITOR: high games density this month"  
- 🟢 Player C: 12% risk — "CLEAR: normal load"

That's not just analytics — that's a decision. That's GuardianAI.

**Technical Highlights**
- Temporal train/test split (2018–2022 train, 2022–2024 test) — no data leakage
- SMOTE + `scale_pos_weight` for class imbalance (injuries are ~5% of games)
- SHAP TreeExplainer for per-player factor attribution
- Live NBA data via `nba_api`

**GitHub:** [link]  
**Live App:** [link]

---

Copy-paste this into the hackathon submission form.
</action>

<acceptance_criteria>
- Submission description written and ready to paste
- GitHub repo is public with complete source code
- Live Streamlit URL works
</acceptance_criteria>

</tasks>

<verification>
```bash
# Final checklist
echo "=== Submission Checklist ==="
test -f README.md && echo "✅ README.md"
test -f requirements.txt && echo "✅ requirements.txt"
test -f src/dashboard/app.py && echo "✅ Dashboard"
test -f src/model/train.py && echo "✅ Training script"
test -f src/data/download.py && echo "✅ Data download"
test -f src/data/features.py && echo "✅ Feature engineering"
test -f models/xgb_injury_model.pkl && echo "✅ Trained model"
test -f data/processed/ml_dataset.csv && echo "✅ Dataset"

git log --oneline -5
echo "=== Repo status ==="
git remote -v
```
</verification>

<success_criteria>
- Public GitHub repo with all source code
- Streamlit app deployed and accessible
- README explains the problem, solution, ACWR, and run instructions
- Hackathon submission text ready
- All checklist items pass
</success_criteria>
