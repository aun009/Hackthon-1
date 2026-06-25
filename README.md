# 🏀 GuardianAI — NBA Injury Risk Prediction & Workload Management

GuardianAI is a predictive analytics system designed for professional basketball teams to forecast player injury risks and manage workloads using explainable AI. By pairing the gold-standard sports science metric—the **Acute:Chronic Workload Ratio (ACWR)**—with Gradient Boosted Trees (XGBoost) and SHAP-based feature attribution, GuardianAI helps coaches and training staff make transparent, data-driven decisions on when to rest players.

---

## 🎯 The Core Concept: Acute-to-Chronic Workload Ratio (ACWR)

Injury forecasting in professional sports relies heavily on tracking fatigue. The **Acute:Chronic Workload Ratio (ACWR)** is widely regarded as a key indicator of soft-tissue injury risk by sports scientists. It compares a player's short-term physical exertion against their historical baseline:

$$ACWR = \frac{\text{Acute Workload (7-day rolling average minutes)}}{\text{Chronic Workload (28-day rolling average minutes)}}$$

The ratio identifies four distinct physiological states:
* **Under-trained ($ACWR < 0.8$)**: The player is not conditioned well enough, increasing risk during high-intensity games.
* **Sweet Spot ($0.8 \le ACWR \le 1.3$)**: Optimal workload and conditioning; injury risk is minimized.
* **Caution Zone ($1.3 < ACWR \le 1.5$)**: Workload is spiking. Caution is recommended.
* **Danger Zone ($ACWR > 1.5$)**: The player's workload has exceeded safe capacity. Soft-tissue injury risk increases significantly.

---

## 🛠️ Built With

GuardianAI is built entirely with production-grade Python tools and web technologies:

* **Language**: Python 3.11 (optimized with a pinned environment for Streamlit deployment)
* **Data Retrieval**: `nba_api` (the official client wrapper for NBA.com Stats endpoints)
* **Data Pipelines & Manipulation**: `pandas` and `numpy`
* **Machine Learning**: `xgboost` (extreme gradient boosting framework) and `scikit-learn` (pipeline utilities, data splits, and scoring)
* **Class Balancing**: `imbalanced-learn` (implementing SMOTE to address dataset sparsity)
* **Model Interpretability (XAI)**: `shap` (SHAP TreeExplainer for game-by-game feature attribution)
* **Visualization**: `plotly` (interactive team scatter maps and feature curves)
* **Web Interface**: `streamlit` (interactive, reactive dashboard application)
* **Deployment Platform**: Streamlit Community Cloud

---

## 📐 Machine Learning Pipeline & Methodology

### 1. Feature Engineering
We construct 18 distinct features across three main categories:
* **Workload Metrics**: Rolling minute averages (7-day and 28-day windows) and the raw ACWR score.
* **Schedule & Fatigue**: Recovery days (`days_rest`), density metrics (number of games played in the last 30 days), back-to-back flags, and a composite `fatigue_score` combining ACWR with scheduling density.
* **Contextual Performance**: Efficiency metrics (points per minute), matchup location (home/away flags), and game number in the current season.

### 2. Strict Temporal Splitting
In time-series sports data, standard random train-test splits lead to critical data leakage. Shuffling games allows future performance to inform past predictions. To prevent this:
* **Training Set**: Game logs from the 2018–19 to 2021–22 NBA seasons.
* **Testing Set**: Game logs from the 2022–23 and 2023–24 NBA seasons.

### 3. Resolving Severe Class Imbalance
Injuries are rare events, appearing in only $\approx 4.5\%$ of our historical player-game entries.
* **Target Label Proxy**: An injury event is flagged if a player experiences a gap of $8+$ consecutive days between consecutive games (used to represent game-terminating soft-tissue injuries or forced rest).
* **Treatment**: We apply **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set to construct synthetic minority samples. We back this up with XGBoost's `scale_pos_weight` parameter to ensure the model penalizes missed injury flags heavily.

### 4. Decision Threshold Calibration
Because missing a real injury (False Negative) is far more costly to an NBA franchise than resting a player unnecessarily (False Positive), we tune the classification threshold to **$0.15$** to prioritize recall.

---

## 📈 Model Validation & Performance

The pipeline generates the following metrics on the out-of-sample temporal test set:

* **ROC-AUC**: `0.5867` (demonstrating predictive signal significantly above random chance in a highly volatile domain)
* **Average Precision (PR-AUC)**: `0.0633`
* **Test Injury Rate**: `4.53%`
* **Recall**: `0.2447` (the model successfully flags approximately a quarter of all incoming high-risk injury events)
* **Decision Threshold**: `0.15`

---

## 📂 Project Structure

```directory
guardianai/
├── .streamlit/
│   └── config.toml       # Dark mode theme and UI configuration
├── src/
│   ├── data/
│   │   ├── download.py   # Raw data ingestion from the NBA Stats API
│   │   └── features.py   # Rolling workload calculations and ACWR feature engineering
│   ├── model/
│   │   ├── train.py      # XGBoost model training and SHAP calibration pipeline
│   │   └── explain.py    # Predictor interface for SHAP explanations
│   └── dashboard/
│       ├── app.py        # Streamlit dashboard layout and controller
│       └── components.py # Reusable custom visualization modules
├── data/
│   ├── raw/              # Raw data CSV storage (gitignored)
│   └── processed/        # Processed and engineered ML-ready dataset
├── models/
│   ├── xgb_injury_model.pkl   # Serialized XGBoost model binary
│   ├── shap_explainer.pkl     # Pre-calculated SHAP TreeExplainer object
│   ├── feature_cols.json      # Mapping of feature order used during training
│   ├── model_metrics.json     # Saved evaluation metrics
│   └── shap_importance.csv    # Overall global feature importance ranks
├── .python-version       # Pinned environment version (3.11)
├── requirements.txt      # Dependency manifest
└── README.md
```

---

## 🚀 Installation & Local Execution

Follow these steps to run the data pipeline, train the model, and launch the interactive dashboard locally.

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/aun009/Hackthon-1.git
cd Hackthon-1

# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execute Data Pipeline & Ingestion
```bash
# Fetch raw player statistics from the NBA API (takes ~10-15 mins due to rate-limiting)
python src/data/download.py

# Run rolling workload calculations and feature engineering
python src/data/features.py
```

### 3. Model Training & Diagnostics
```bash
# Train the XGBoost model, evaluate metrics, and serialize artifacts
python src/model/train.py
```

### 4. Run the Streamlit Dashboard
```bash
# Launch the local dashboard
streamlit run src/dashboard/app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🃏 Dashboard Features

* **Risk Cards**: Select any individual player to view their computed risk probability. Includes a detailed panel showing the top 3 contributing factors to their risk score generated dynamically via local SHAP attribution.
* **Workload Map**: Shows a team-wide overview plotting players along the ACWR scale. Highlights players currently entering the Caution and Danger zones.
* **Model Validation Tab**: Live performance metrics displaying ROC-AUC, Recall, and the Precision-Recall curves. Allows coaches to see the transparency and reliability of the model.

---

*Built with ❤️ for the AQX Sports Analytics Hackathon 2026*
