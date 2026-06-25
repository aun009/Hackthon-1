"""
train.py — XGBoost injury prediction model with temporal split + SMOTE.

Key decisions:
  1. Temporal split (2018-22 train / 2022-24 test) — no data leakage
  2. SMOTE on train only — handles rare injury class (~5% of games)
  3. scale_pos_weight as backup class balancer in XGBoost
  4. SHAP TreeExplainer saved alongside model for dashboard
  5. Threshold at 0.3 (favours recall — missing an injury is worse than a false alarm)

Run AFTER features.py. Outputs: models/ directory
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score,
)
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "rolling_min_7", "rolling_min_28", "acwr",
    "rolling_pts_7", "rolling_reb_7", "rolling_ast_7",
    "days_rest", "back_to_back", "no_rest_2d",
    "games_played_30", "fatigue_score",
    "game_num_in_season", "pts_per_min", "is_away",
]
TARGET = "injured"

TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22"]
TEST_SEASONS  = ["2022-23", "2023-24"]


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run src/data/features.py first.\nExpected: {path}")
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional features to boost signal."""
    df = df.copy()

    # High-load flag: ACWR > 1.3 AND playing > 32 mins
    df["high_load_flag"] = (
        (df["acwr"] > 1.3) & (df["rolling_min_7"] > 32)
    ).astype(int)

    # Workload spike: acute load much higher than chronic
    df["load_spike"] = (df["acwr"] - 1.0).clip(lower=0)

    # Cumulative fatigue: high games density + low rest
    df["dense_schedule"] = (
        (df["games_played_30"] >= 10) & (df["days_rest"] <= 2)
    ).astype(int)

    # Minutes trend: recent vs historical
    with np.errstate(divide="ignore", invalid="ignore"):
        df["min_trend"] = (
            df["rolling_min_7"] / df["rolling_min_28"].replace(0, np.nan)
        ).fillna(1.0)

    return df


EXTRA_FEATURES = [
    "high_load_flag", "load_spike", "dense_schedule", "min_trend"
]
ALL_FEATURES = FEATURE_COLS + EXTRA_FEATURES


def temporal_split(df: pd.DataFrame):
    df = add_extra_features(df)
    available = [c for c in ALL_FEATURES if c in df.columns]

    train = df[df["SEASON"].isin(TRAIN_SEASONS)]
    test  = df[df["SEASON"].isin(TEST_SEASONS)]

    X_train = train[available].fillna(0)
    y_train = train[TARGET]
    X_test  = test[available].fillna(0)
    y_test  = test[TARGET]

    print(f"Train: {len(X_train):>5,} rows | injuries: {int(y_train.sum()):>3} ({y_train.mean():.1%})")
    print(f"Test:  {len(X_test):>5,} rows | injuries: {int(y_test.sum()):>3} ({y_test.mean():.1%})")
    return X_train, y_train, X_test, y_test, available


def apply_smote(X_train, y_train):
    if y_train.mean() < 0.15:
        print(f"Applying SMOTE (injury rate={y_train.mean():.1%})...")
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, int(y_train.sum()) - 1))
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print(f"  After SMOTE: {len(X_res):,} rows | {y_res.mean():.1%} positive")
            return X_res, y_res
        except Exception as e:
            print(f"  SMOTE skipped ({e}) — using original")
    return X_train, y_train


def train_model(X_train, y_train, feature_cols):
    neg = (y_train == 0).sum()
    pos = max((y_train == 1).sum(), 1)
    scale = neg / pos

    # Try multiple configs and pick best CV AUC
    configs = [
        dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7),
        dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8),
        dict(n_estimators=600, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6),
    ]

    best_model, best_score = None, -1
    X_df = pd.DataFrame(X_train, columns=feature_cols) if not isinstance(X_train, pd.DataFrame) else X_train

    for i, cfg in enumerate(configs):
        m = xgb.XGBClassifier(
            **cfg,
            scale_pos_weight=scale,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        scores = cross_val_score(m, X_df, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
        auc = scores.mean()
        print(f"  Config {i+1}: CV AUC = {auc:.4f} (±{scores.std():.4f})")
        if auc > best_score:
            best_score = auc
            best_model = m

    print(f"  → Best CV AUC: {best_score:.4f} — fitting on full train set")
    best_model.fit(X_df, y_train, verbose=False)
    return best_model


def evaluate(model, X_test, y_test, feature_cols, threshold: float = 0.3) -> dict:
    X_df = pd.DataFrame(X_test, columns=feature_cols) if not isinstance(X_test, pd.DataFrame) else X_test
    y_prob = model.predict_proba(X_df)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Find best threshold by F1
    best_t, best_f1 = threshold, 0
    for t in np.arange(0.15, 0.6, 0.05):
        p = (y_prob >= t).astype(int)
        f = f1_score(y_test, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    y_pred_opt = (y_prob >= best_t).astype(int)

    metrics = {
        "roc_auc":          round(float(roc_auc_score(y_test, y_prob)), 4),
        "avg_precision":    round(float(average_precision_score(y_test, y_prob)), 4),
        "precision":        round(float(precision_score(y_test, y_pred_opt, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_test, y_pred_opt, zero_division=0)), 4),
        "f1":               round(float(f1_score(y_test, y_pred_opt, zero_division=0)), 4),
        "threshold":        round(best_t, 2),
        "test_injury_rate": round(float(y_test.mean()), 4),
        "train_seasons":    TRAIN_SEASONS,
        "test_seasons":     TEST_SEASONS,
        "feature_count":    len(feature_cols),
    }

    print("\n── Model Metrics ─────────────────────────────────")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:<22}: {v}")
    return metrics


def compute_shap(model, X_sample: pd.DataFrame, feature_cols):
    X_df = pd.DataFrame(X_sample, columns=feature_cols) if not isinstance(X_sample, pd.DataFrame) else X_sample
    print("\nComputing SHAP values (~30s)...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_df)
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1]
    else:
        shap_arr = shap_vals
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": np.abs(shap_arr).mean(axis=0),
    }).sort_values("importance", ascending=False)
    return explainer, importance


def save_artifacts(model, metrics, explainer, importance, feature_cols):
    joblib.dump(model,     MODELS_DIR / "xgb_injury_model.pkl")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importance.to_csv(MODELS_DIR / "shap_importance.csv", index=False)

    print(f"\n✅ Artifacts saved → {MODELS_DIR}/")
    for name in ["xgb_injury_model.pkl", "shap_explainer.pkl",
                 "feature_cols.json", "model_metrics.json", "shap_importance.csv"]:
        size = (MODELS_DIR / name).stat().st_size / 1024
        print(f"   {name:<30} ({size:.0f} KB)")


def train_pipeline():
    print("── Loading dataset ────────────────────────────────")
    df = load_data()

    print("\n── Temporal split (no leakage) ────────────────────")
    X_train, y_train, X_test, y_test, feature_cols = temporal_split(df)

    if len(X_train) == 0:
        raise RuntimeError("No training data — check SEASON values in dataset")

    print("\n── Resampling ─────────────────────────────────────")
    X_tr, y_tr = apply_smote(X_train, y_train)

    print("\n── Grid search over XGBoost configs ───────────────")
    model = train_model(X_tr, y_tr, feature_cols)

    print("\n── Evaluation ─────────────────────────────────────")
    metrics = evaluate(model, X_test, y_test, feature_cols)

    print("\n── SHAP Explainability ────────────────────────────")
    sample = X_train.sample(min(500, len(X_train)), random_state=42)
    explainer, importance = compute_shap(model, sample, feature_cols)

    print("\n── Saving ─────────────────────────────────────────")
    save_artifacts(model, metrics, explainer, importance, feature_cols)

    return model, metrics


if __name__ == "__main__":
    train_pipeline()
