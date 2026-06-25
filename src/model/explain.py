"""
explain.py — Load trained model and generate per-player injury risk predictions.

Used by the Streamlit dashboard to:
  - Predict injury probability for any player-game row
  - Return top-3 SHAP factors driving the risk score
  - Generate team-level risk table for the overview tab
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")

RISK_THRESHOLDS = {"HIGH": 60, "MEDIUM": 35}  # %


def load_model_artifacts() -> dict:
    """Load all model artifacts once at dashboard startup."""
    model     = joblib.load(MODELS_DIR / "xgb_injury_model.pkl")
    explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(MODELS_DIR / "model_metrics.json") as f:
        metrics = json.load(f)

    importance = pd.read_csv(MODELS_DIR / "shap_importance.csv")

    return {
        "model":        model,
        "explainer":    explainer,
        "feature_cols": feature_cols,
        "metrics":      metrics,
        "importance":   importance,
    }


def predict_risk(row: pd.Series, artifacts: dict) -> dict:
    """
    Predict injury risk for a single player-game row.

    Returns a dict with:
      risk_pct    : 0-100 float
      risk_label  : "HIGH" | "MEDIUM" | "LOW"
      risk_color  : emoji indicator
      top_factors : list of {feature, value, impact, direction}
      recommendation : plain-English coaching action
    """
    feature_cols = artifacts["feature_cols"]
    model        = artifacts["model"]
    explainer    = artifacts["explainer"]

    X     = np.array([float(row.get(c, 0) or 0) for c in feature_cols]).reshape(1, -1)
    X_df  = pd.DataFrame(X, columns=feature_cols)

    prob     = float(model.predict_proba(X_df)[0][1])
    risk_pct = round(prob * 100, 1)

    if risk_pct >= RISK_THRESHOLDS["HIGH"]:
        label, color = "HIGH",   "🔴"
    elif risk_pct >= RISK_THRESHOLDS["MEDIUM"]:
        label, color = "MEDIUM", "🟡"
    else:
        label, color = "LOW",    "🟢"

    # SHAP for this row
    shap_vals = explainer.shap_values(X_df)
    if isinstance(shap_vals, list):          # some versions return list
        shap_vals = shap_vals[1]
    shap_vals = np.asarray(shap_vals).flatten()

    factor_df = pd.DataFrame({
        "feature": feature_cols,
        "value":   X.flatten(),
        "shap":    shap_vals,
    }).sort_values("shap", key=abs, ascending=False).head(3)

    top_factors = [
        {
            "feature":   r["feature"].replace("_", " ").title(),
            "value":     round(float(r["value"]), 2),
            "impact":    round(float(abs(r["shap"])), 3),
            "direction": "↑ risk" if r["shap"] > 0 else "↓ risk",
        }
        for _, r in factor_df.iterrows()
    ]

    # Coaching recommendation text
    acwr = float(row.get("acwr", 1.0) or 1.0)
    b2b  = int(row.get("back_to_back", 0) or 0)
    gp30 = int(row.get("games_played_30", 0) or 0)

    reasons = []
    if acwr > 1.5:
        reasons.append(f"ACWR {acwr:.2f} > 1.5 danger zone")
    if b2b:
        reasons.append("back-to-back game")
    if gp30 >= 12:
        reasons.append(f"{gp30} games in 30 days")

    if risk_pct >= RISK_THRESHOLDS["HIGH"]:
        rec = "🛑 REST — " + (", ".join(reasons) if reasons else "high predicted risk")
    elif risk_pct >= RISK_THRESHOLDS["MEDIUM"]:
        rec = "⚠️ MONITOR — watch workload closely"
    else:
        rec = "✅ CLEAR — normal risk profile"

    return {
        "risk_pct":      risk_pct,
        "risk_label":    label,
        "risk_color":    color,
        "top_factors":   top_factors,
        "recommendation": rec,
    }


def predict_team(df: pd.DataFrame, artifacts: dict, player_name: str = None) -> pd.DataFrame:
    """
    Generate risk predictions for all (or one) player's most recent game.
    Returns DataFrame sorted by risk descending.
    """
    if player_name and player_name != "All Players":
        df = df[df["PLAYER_NAME"] == player_name]

    latest = df.sort_values("GAME_DATE").groupby("PLAYER_NAME").last().reset_index()

    rows = []
    for _, row in latest.iterrows():
        pred = predict_risk(row, artifacts)
        rows.append({
            "Player":         row["PLAYER_NAME"],
            "Last Game":      str(row["GAME_DATE"])[:10] if pd.notna(row.get("GAME_DATE")) else "N/A",
            "Risk %":         pred["risk_pct"],
            "Risk":           f"{pred['risk_color']} {pred['risk_label']}",
            "ACWR":           round(float(row.get("acwr", 0) or 0), 2),
            "Days Rest":      int(row.get("days_rest", 0) or 0),
            "B2B":            "Yes" if row.get("back_to_back") else "No",
            "Fatigue":        round(float(row.get("fatigue_score", 0) or 0), 2),
            "Recommendation": pred["recommendation"],
            "Top Factor":     pred["top_factors"][0]["feature"] if pred["top_factors"] else "",
            "_risk_label":    pred["risk_label"],  # for filtering
        })

    return pd.DataFrame(rows).sort_values("Risk %", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    artifacts = load_model_artifacts()
    df = pd.read_csv(PROCESSED_DIR / "ml_dataset.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    result = predict_team(df, artifacts)
    print(result[["Player", "Risk %", "Risk", "ACWR", "Recommendation"]].to_string())
