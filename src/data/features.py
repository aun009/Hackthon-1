"""
features.py — Engineer ACWR + rolling workload features for injury prediction.

ACWR = Acute:Chronic Workload Ratio (gold standard sports-science injury metric)
  ACWR > 1.5 = danger zone (significantly elevated soft-tissue injury risk)

Run AFTER download.py. Outputs: data/processed/ml_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_game_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # nba_api returns dates as 'MMM DD, YYYY'
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    # fallback for other formats
    mask = df["GAME_DATE"].isna()
    if mask.any():
        df.loc[mask, "GAME_DATE"] = pd.to_datetime(
            df.loc[mask, "GAME_DATE"], infer_datetime_format=True, errors="coerce"
        )
    return df


def safe_float(series: pd.Series) -> pd.Series:
    """Convert column to float, handling ':' in MIN column (nba_api quirk)."""
    if series.dtype == object:
        series = series.astype(str).str.split(":").str[0]
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player rolling features. Computed within each player group,
    sorted chronologically — no leakage across players.
    """
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    for player_id, grp in df.groupby("PLAYER_ID", sort=False):
        idx  = grp.index
        mins = safe_float(grp["MIN"])
        pts  = safe_float(grp["PTS"])
        reb  = safe_float(grp["REB"])
        ast  = safe_float(grp["AST"])

        # ── Rolling workload windows ────────────────────────────────────
        df.loc[idx, "rolling_min_7"]  = mins.rolling(7,  min_periods=1).mean().values
        df.loc[idx, "rolling_min_28"] = mins.rolling(28, min_periods=1).mean().values
        df.loc[idx, "rolling_pts_7"]  = pts.rolling(7,  min_periods=1).mean().values
        df.loc[idx, "rolling_reb_7"]  = reb.rolling(7,  min_periods=1).mean().values
        df.loc[idx, "rolling_ast_7"]  = ast.rolling(7,  min_periods=1).mean().values

        # ── ACWR (core metric: acute / chronic load) ───────────────────
        r7  = df.loc[idx, "rolling_min_7"].values
        r28 = df.loc[idx, "rolling_min_28"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            acwr = np.where(r28 > 0, r7 / r28, 1.0)
        df.loc[idx, "acwr"] = acwr

        # ── Rest & scheduling features ─────────────────────────────────
        dates = grp["GAME_DATE"].reset_index(drop=True)
        days_rest = dates.diff().dt.days.fillna(3.0).values
        df.loc[idx, "days_rest"]    = days_rest
        df.loc[idx, "back_to_back"] = (days_rest == 1).astype(int)
        df.loc[idx, "no_rest_2d"]   = (days_rest <= 2).astype(int)

        # ── Games played in rolling 30-day window ──────────────────────
        date_list = dates.tolist()
        gp30 = []
        for i, d in enumerate(date_list):
            cutoff = d - pd.Timedelta(days=30)
            gp30.append(sum(1 for dd in date_list[:i] if dd >= cutoff))
        df.loc[idx, "games_played_30"] = gp30

        # ── Composite fatigue score ────────────────────────────────────
        gp30_norm = np.array(gp30) / 15.0
        no_rest   = df.loc[idx, "no_rest_2d"].values
        df.loc[idx, "fatigue_score"] = (
            acwr * 0.5 + gp30_norm * 0.3 + no_rest * 0.2
        )

    return df


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_num_in_season"] = df.groupby(["PLAYER_ID", "SEASON"]).cumcount() + 1

    mins = safe_float(df["MIN"])
    pts  = safe_float(df["PTS"])
    df["pts_per_min"] = (pts / mins.replace(0, np.nan)).fillna(0)

    if "MATCHUP" in df.columns:
        df["is_away"] = df["MATCHUP"].astype(str).str.contains("@").astype(int)
    else:
        df["is_away"] = 0

    return df


def add_injury_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy label: injured=1 if player's *next* appearance is 8+ days later
    (implying missed games — either injury or load management).

    Limitation: also catches suspension, trade, etc. — acceptable proxy for hackathon.
    """
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    df["injured"] = 0

    for _, grp in df.groupby("PLAYER_ID"):
        idx_list = grp.index.tolist()
        for i in range(len(idx_list) - 1):
            gap = (
                df.loc[idx_list[i + 1], "GAME_DATE"]
                - df.loc[idx_list[i],     "GAME_DATE"]
            ).days
            if gap >= 8:
                df.loc[idx_list[i], "injured"] = 1

    return df


FEATURE_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "SEASON", "GAME_DATE",
    "MIN", "PTS", "REB", "AST",
    "rolling_min_7", "rolling_min_28", "acwr",
    "rolling_pts_7", "rolling_reb_7", "rolling_ast_7",
    "days_rest", "back_to_back", "no_rest_2d",
    "games_played_30", "fatigue_score",
    "game_num_in_season", "pts_per_min", "is_away",
    "injured",
]


def build_ml_dataset() -> pd.DataFrame:
    raw = RAW_DIR / "nba_game_logs.csv"
    if not raw.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw}\nRun:  python src/data/download.py"
        )

    print("Loading raw data...")
    df = pd.read_csv(raw)
    print(f"  Raw rows: {len(df):,}")

    print("Parsing dates...")
    df = parse_game_date(df)

    print("Engineering rolling + ACWR features...")
    df = add_rolling_features(df)

    print("Engineering player features...")
    df = add_player_features(df)

    print("Adding injury target label...")
    df = add_injury_target(df)

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_final = df[available].dropna(subset=["rolling_min_7", "acwr", "injured"])

    out = PROCESSED_DIR / "ml_dataset.csv"
    df_final.to_csv(out, index=False)

    n_inj = int(df_final["injured"].sum())
    print(f"\n✅ Saved {len(df_final):,} rows → {out}")
    print(f"   Injury events : {n_inj} ({n_inj / len(df_final) * 100:.1f}%)")
    print(f"   Feature cols  : {len(available) - 4} (excl. ID/meta)")
    return df_final


if __name__ == "__main__":
    build_ml_dataset()
