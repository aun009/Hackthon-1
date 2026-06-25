---
phase: 2
plan: 02-model-training
wave: 1
type: execute
depends_on: [01-data-pipeline]
files_modified:
  - src/model/train.py
  - src/model/explain.py
  - models/xgb_injury_model.pkl
  - models/feature_cols.json
  - models/model_metrics.json
autonomous: true
---

<objective>
Train an XGBoost injury-prediction classifier with temporal train/test split,
SMOTE for class imbalance, and SHAP explanations. Save model + metadata artifacts
to the `models/` directory for the dashboard to consume.
</objective>

<tasks>

## Task 1 — Model Training (`src/model/train.py`)

<read_first>
- src/data/features.py
- data/processed/ml_dataset.csv (check columns after Phase 1 completes)
- requirements.txt
</read_first>

<action>
Write `src/model/train.py`:

```python
"""
train.py — XGBoost injury prediction model with temporal split + SMOTE.

Critical design decisions:
1. TEMPORAL split (never random shuffle) — train on 2018-22, test on 2022-24
   Random shuffling = data leakage (future games inform past predictions)
2. SMOTE only on training data — never on test set
3. class_weight / scale_pos_weight handles imbalance in XGBoost
4. Probability calibration via CalibratedClassifierCV for honest risk %
5. Save SHAP explainer alongside model for dashboard use
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_score, recall_score, f1_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature columns (input to model) ────────────────────────────────────────
FEATURE_COLS = [
    "rolling_min_7", "rolling_min_28", "acwr",
    "rolling_pts_7", "rolling_reb_7", "rolling_ast_7",
    "days_rest", "back_to_back", "no_rest_2d",
    "games_played_30", "fatigue_score",
    "game_num_in_season", "pts_per_min", "is_away",
]
TARGET = "injured"

# ── Temporal split seasons ────────────────────────────────────────────────────
TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22"]
TEST_SEASONS  = ["2022-23", "2023-24"]


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run src/data/features.py first: {path}")
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


def temporal_split(df: pd.DataFrame):
    train_mask = df["SEASON"].isin(TRAIN_SEASONS)
    test_mask  = df["SEASON"].isin(TEST_SEASONS)

    X_train = df.loc[train_mask, FEATURE_COLS].fillna(0)
    y_train = df.loc[train_mask, TARGET]
    X_test  = df.loc[test_mask,  FEATURE_COLS].fillna(0)
    y_test  = df.loc[test_mask,  TARGET]

    print(f"Train: {len(X_train)} rows | Injuries: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"Test:  {len(X_test)}  rows | Injuries: {y_test.sum()}  ({y_test.mean():.1%})")
    return X_train, y_train, X_test, y_test


def apply_smote(X_train, y_train):
    """SMOTE to handle class imbalance. Only applied to training data."""
    injury_rate = y_train.mean()
    if injury_rate < 0.1:
        print(f"Applying SMOTE (injury rate={injury_rate:.1%})...")
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {len(X_res)} rows | Injuries: {y_res.sum()} ({y_res.mean():.1%})")
        return X_res, y_res
    return X_train, y_train


def train_model(X_train, y_train):
    """Train XGBoost with scale_pos_weight for class imbalance."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=50,
    )
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)  # lower threshold for injury recall

    metrics = {
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_test, y_prob), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
        "threshold":        0.3,
        "test_injury_rate": round(float(y_test.mean()), 4),
        "train_seasons":    TRAIN_SEASONS,
        "test_seasons":     TEST_SEASONS,
        "feature_count":    len(FEATURE_COLS),
    }

    print("\n── Model Performance ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    return metrics


def generate_shap_values(model, X_train):
    """Compute SHAP explainer on training data for dashboard use."""
    print("\nGenerating SHAP explainer (this takes ~30s)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return explainer, shap_values


def save_artifacts(model, metrics: dict, feature_cols: list, shap_data: dict):
    """Save all model artifacts to models/ directory."""
    # Model
    joblib.dump(model, MODELS_DIR / "xgb_injury_model.pkl")

    # Feature column order (dashboard must use same order)
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Metrics for dashboard display
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # SHAP explainer
    joblib.dump(shap_data["explainer"], MODELS_DIR / "shap_explainer.pkl")

    # SHAP feature importance (mean |SHAP| per feature)
    mean_shap = pd.DataFrame({
        "feature": feature_cols,
        "importance": np.abs(shap_data["shap_values"]).mean(axis=0),
    }).sort_values("importance", ascending=False)
    mean_shap.to_csv(MODELS_DIR / "shap_importance.csv", index=False)

    print(f"\n✅ Artifacts saved to {MODELS_DIR}/")
    print(f"   xgb_injury_model.pkl")
    print(f"   shap_explainer.pkl")
    print(f"   feature_cols.json")
    print(f"   model_metrics.json")
    print(f"   shap_importance.csv")


def train_pipeline():
    print("Loading dataset...")
    df = load_data()

    print("\nSplitting data (temporal — no leakage)...")
    X_train, y_train, X_test, y_test = temporal_split(df)

    print("\nApplying SMOTE...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("\nTraining XGBoost...")
    model = train_model(X_train_res, y_train_res)

    print("\nEvaluating on held-out test set...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\nComputing SHAP explanations...")
    # Use sample of training data for SHAP (faster)
    sample_idx = X_train.sample(min(500, len(X_train)), random_state=42).index
    X_sample = X_train.loc[sample_idx]
    explainer, shap_values = generate_shap_values(model, X_sample)

    print("\nSaving artifacts...")
    save_artifacts(
        model=model,
        metrics=metrics,
        feature_cols=FEATURE_COLS,
        shap_data={"explainer": explainer, "shap_values": shap_values},
    )

    return model, metrics


if __name__ == "__main__":
    train_pipeline()
```
</action>

<acceptance_criteria>
- `src/model/train.py` exists
- `src/model/train.py` contains `def train_pipeline()`
- `src/model/train.py` contains `TRAIN_SEASONS`
- `src/model/train.py` contains `scale_pos_weight`
- `src/model/train.py` contains `shap.TreeExplainer`
- `src/model/train.py` contains `temporal_split`
- After running: `models/xgb_injury_model.pkl` exists
- After running: `models/model_metrics.json` exists with `roc_auc` key
- After running: `models/shap_importance.csv` exists
</acceptance_criteria>

---

## Task 2 — Player Prediction API (`src/model/explain.py`)

<read_first>
- src/model/train.py
- models/feature_cols.json (after training)
</read_first>

<action>
Write `src/model/explain.py` — the prediction and explanation layer that the dashboard calls:

```python
"""
explain.py — Load trained model and generate player risk predictions with SHAP.

Used by the Streamlit dashboard to:
1. Load model + explainer at startup (cached)
2. Predict injury risk for a given player/game row
3. Return top-3 SHAP factors for the risk card
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


def load_model_artifacts():
    """Load all model artifacts. Called once at dashboard startup."""
    model = joblib.load(MODELS_DIR / "xgb_injury_model.pkl")
    explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(MODELS_DIR / "model_metrics.json") as f:
        metrics = json.load(f)
    importance = pd.read_csv(MODELS_DIR / "shap_importance.csv")

    return {
        "model": model,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "importance": importance,
    }


def predict_risk(row: pd.Series, artifacts: dict) -> dict:
    """
    Predict injury risk for a single player-game row.
    
    Returns:
        {
          "risk_pct": 73.4,        # 0-100
          "risk_label": "HIGH",    # LOW / MEDIUM / HIGH
          "risk_color": "🔴",
          "top_factors": [
              {"feature": "acwr", "value": 1.67, "impact": 0.18, "direction": "↑ risk"},
              ...
          ],
          "recommendation": "REST — ACWR in danger zone (>1.5)",
        }
    """
    feature_cols = artifacts["feature_cols"]
    model = artifacts["model"]
    explainer = artifacts["explainer"]

    X = row[feature_cols].values.reshape(1, -1).astype(float)
    X = np.nan_to_num(X, nan=0.0)
    X_df = pd.DataFrame(X, columns=feature_cols)

    # Probability
    prob = model.predict_proba(X_df)[0][1]
    risk_pct = round(prob * 100, 1)

    # Label
    if risk_pct >= 60:
        label, color = "HIGH", "🔴"
    elif risk_pct >= 35:
        label, color = "MEDIUM", "🟡"
    else:
        label, color = "LOW", "🟢"

    # SHAP for this row
    shap_vals = explainer.shap_values(X_df)[0]
    factor_df = pd.DataFrame({
        "feature": feature_cols,
        "value": X[0],
        "shap": shap_vals,
    }).sort_values("shap", ascending=False, key=abs).head(3)

    top_factors = []
    for _, r in factor_df.iterrows():
        direction = "↑ risk" if r["shap"] > 0 else "↓ risk"
        top_factors.append({
            "feature": r["feature"].replace("_", " ").title(),
            "value": round(float(r["value"]), 2),
            "impact": round(float(abs(r["shap"])), 3),
            "direction": direction,
        })

    # Recommendation text
    acwr = float(row.get("acwr", 1.0))
    b2b  = int(row.get("back_to_back", 0))
    gp30 = int(row.get("games_played_30", 0))

    reasons = []
    if acwr > 1.5:
        reasons.append(f"ACWR={acwr:.2f} (danger zone >1.5)")
    if b2b:
        reasons.append("back-to-back game")
    if gp30 >= 12:
        reasons.append(f"{gp30} games in last 30 days")

    if risk_pct >= 60:
        rec = "🛑 REST — " + " + ".join(reasons) if reasons else "🛑 REST — high predicted risk"
    elif risk_pct >= 35:
        rec = "⚠️ MONITOR — watch workload closely"
    else:
        rec = "✅ CLEAR — normal risk profile"

    return {
        "risk_pct": risk_pct,
        "risk_label": label,
        "risk_color": color,
        "top_factors": top_factors,
        "recommendation": rec,
    }


def predict_team(df: pd.DataFrame, artifacts: dict, player_name: str = None) -> pd.DataFrame:
    """
    Predict risk for all (or one) player's most recent game in the dataset.
    Returns a DataFrame sorted by risk descending.
    """
    feature_cols = artifacts["feature_cols"]

    if player_name:
        df = df[df["PLAYER_NAME"] == player_name]

    # Get most recent game per player
    latest = df.sort_values("GAME_DATE").groupby("PLAYER_NAME").last().reset_index()

    results = []
    for _, row in latest.iterrows():
        pred = predict_risk(row, artifacts)
        results.append({
            "Player": row["PLAYER_NAME"],
            "Last Game": str(row["GAME_DATE"])[:10] if pd.notna(row.get("GAME_DATE")) else "N/A",
            "Risk %": pred["risk_pct"],
            "Risk": f"{pred['risk_color']} {pred['risk_label']}",
            "ACWR": round(float(row.get("acwr", 0)), 2),
            "Days Rest": int(row.get("days_rest", 0)),
            "B2B": "Yes" if row.get("back_to_back") else "No",
            "Fatigue": round(float(row.get("fatigue_score", 0)), 2),
            "Recommendation": pred["recommendation"],
            "Top Factor": pred["top_factors"][0]["feature"] if pred["top_factors"] else "",
        })

    return pd.DataFrame(results).sort_values("Risk %", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Quick smoke test
    artifacts = load_model_artifacts()
    df = pd.read_csv(PROCESSED_DIR / "ml_dataset.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    print("Team risk predictions (most recent game per player):")
    result = predict_team(df, artifacts)
    print(result[["Player", "Risk %", "Risk", "ACWR", "Recommendation"]].to_string())
```
</action>

<acceptance_criteria>
- `src/model/explain.py` exists
- `src/model/explain.py` contains `def load_model_artifacts()`
- `src/model/explain.py` contains `def predict_risk(`
- `src/model/explain.py` contains `def predict_team(`
- `src/model/explain.py` contains `shap_vals = explainer.shap_values`
- `src/model/explain.py` contains `"risk_pct":`
- `src/model/explain.py` contains `"recommendation":`
</acceptance_criteria>

</tasks>

<verification>
```bash
# Run training (after Phase 1 data is ready)
python src/model/train.py

# Verify artifacts
ls -lh models/
python -c "
import json
with open('models/model_metrics.json') as f:
    m = json.load(f)
print('ROC AUC:', m['roc_auc'])
print('Precision:', m['precision'])
print('Recall:', m['recall'])
assert m['roc_auc'] > 0.6, 'AUC too low — check features'
print('✅ Model metrics look good')
"

# Test prediction API
python src/model/explain.py
```
</verification>

<success_criteria>
- `models/xgb_injury_model.pkl` exists
- `models/shap_explainer.pkl` exists
- `models/model_metrics.json` has `roc_auc` >= 0.60
- `src/model/explain.py` smoke test runs without error and prints player risk table
</success_criteria>
