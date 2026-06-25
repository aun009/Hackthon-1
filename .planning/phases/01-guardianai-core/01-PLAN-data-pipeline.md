---
phase: 1
plan: 01-data-pipeline
wave: 1
type: execute
files_modified:
  - src/data/download.py
  - src/data/features.py
  - data/raw/.gitkeep
  - data/processed/.gitkeep
autonomous: true
---

<objective>
Download NBA player game logs and injury data, engineer ACWR + rolling workload features,
and produce a clean `data/processed/ml_dataset.csv` ready for model training.
</objective>

<tasks>

## Task 1 — Project Scaffolding

<read_first>
- .planning/PROJECT.md
</read_first>

<action>
Create the following directory structure and files:

```
GuardianAI/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── features.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── explain.py
│   └── dashboard/
│       ├── __init__.py
│       └── app.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

Write `requirements.txt`:
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.0.3
shap==0.45.1
streamlit==1.35.0
plotly==5.22.0
nba_api==1.4.1
kaggle==1.6.14
imbalanced-learn==0.12.3
matplotlib==3.9.0
joblib==1.4.2
```

Write `.gitignore`:
```
data/raw/*.csv
data/processed/*.csv
models/*.pkl
models/*.json
__pycache__/
*.pyc
.env
.DS_Store
```
</action>

<acceptance_criteria>
- `src/data/__init__.py` exists
- `src/model/__init__.py` exists
- `src/dashboard/__init__.py` exists
- `requirements.txt` contains `xgboost==2.0.3`
- `requirements.txt` contains `shap==0.45.1`
- `requirements.txt` contains `nba_api==1.4.1`
- `.gitignore` contains `models/*.pkl`
</acceptance_criteria>

---

## Task 2 — Data Download (`src/data/download.py`)

<read_first>
- requirements.txt
</read_first>

<action>
Write `src/data/download.py` with the following content. This uses `nba_api` to pull real player game logs and a curated injury list from Basketball-Reference compatible CSV:

```python
"""
download.py — Fetch NBA player game logs + injury data.

Strategy:
1. Use nba_api to get per-player game logs for seasons 2018-19 to 2023-24
2. Target 20 high-minutes players to keep download manageable in hackathon time
3. Save raw CSVs to data/raw/
"""

import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Target seasons ─────────────────────────────────────────────────────────
SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24",
]

# ── Target players (high-minutes stars with injury history) ─────────────────
TARGET_PLAYER_NAMES = [
    "LeBron James", "Kevin Durant", "Stephen Curry",
    "Kawhi Leonard", "Anthony Davis", "Ja Morant",
    "Zion Williamson", "Joel Embiid", "Jayson Tatum",
    "Luka Doncic", "Damian Lillard", "Devin Booker",
    "Giannis Antetokounmpo", "Nikola Jokic", "Paul George",
    "Jimmy Butler", "Kyrie Irving", "Chris Paul",
    "Donovan Mitchell", "De'Aaron Fox",
]

# ── Known injury games (manually curated — game_id, player, date) ──────────
# This is a simplified ground truth. In production you'd scrape prosportstransactions.com
KNOWN_INJURIES = {
    # player_name: list of (season, approx game number when injured)
    # We synthesize: if a player missed 3+ consecutive games after playing, mark prior game as injured=1
}


def get_player_id(name: str) -> int | None:
    all_players = players.get_active_players()
    inactive = players.get_inactive_players()
    for p in (all_players + inactive):
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None


def fetch_player_logs(player_id: int, player_name: str) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        try:
            logs = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                timeout=60,
            )
            df = logs.get_data_frames()[0]
            df["PLAYER_ID"] = player_id
            df["PLAYER_NAME"] = player_name
            df["SEASON"] = season
            frames.append(df)
            time.sleep(0.7)  # respect NBA API rate limit
        except Exception as e:
            print(f"  ⚠ Could not fetch {player_name} {season}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_all() -> pd.DataFrame:
    all_logs = []

    for name in TARGET_PLAYER_NAMES:
        print(f"Fetching {name}...")
        pid = get_player_id(name)
        if pid is None:
            print(f"  ✗ Player not found: {name}")
            continue
        df = fetch_player_logs(pid, name)
        if not df.empty:
            all_logs.append(df)
            print(f"  ✓ {len(df)} game logs")

    if not all_logs:
        raise RuntimeError("No data fetched. Check nba_api installation.")

    combined = pd.concat(all_logs, ignore_index=True)
    out = RAW_DIR / "nba_game_logs.csv"
    combined.to_csv(out, index=False)
    print(f"\n✅ Saved {len(combined)} rows → {out}")
    return combined


if __name__ == "__main__":
    download_all()
```
</action>

<acceptance_criteria>
- `src/data/download.py` exists
- `src/data/download.py` contains `def download_all()`
- `src/data/download.py` contains `playergamelog`
- `src/data/download.py` contains `SEASONS = [`
- `src/data/download.py` contains `time.sleep(0.7)`
</acceptance_criteria>

---

## Task 3 — Feature Engineering (`src/data/features.py`)

<read_first>
- src/data/download.py
</read_first>

<action>
Write `src/data/features.py`. This is the **analytical core** — the ACWR and rolling features are what judges will evaluate:

```python
"""
features.py — Engineer ACWR + rolling workload features for injury prediction.

Key insight: The Acute:Chronic Workload Ratio (ACWR) is the gold standard
sports-science metric for injury prediction. ACWR > 1.5 = danger zone.

Features engineered per player, per game (sorted chronologically):
  - rolling_min_7:   avg minutes over last 7 games
  - rolling_min_28:  avg minutes over last 28 games  
  - acwr:            rolling_min_7 / rolling_min_28
  - back_to_back:    1 if previous game was yesterday
  - days_rest:       days since last game
  - rolling_pts_7:   avg points over last 7 games
  - rolling_reb_7:   avg rebounds over last 7 games
  - games_played_30: games played in last 30 days
  - fatigue_score:   composite (acwr * 0.5 + games_played_30_norm * 0.3 + no_rest * 0.2)
  - injured:         TARGET — did player miss the NEXT game? (proxy for injury)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_game_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert GAME_DATE string to datetime. NBA API returns 'MMM DD, YYYY'."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player rolling features. MUST be computed within each player group,
    sorted chronologically, to avoid data leakage across players/seasons.
    """
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    df = df.reset_index(drop=True)

    feature_cols = []

    for player_id, group in df.groupby("PLAYER_ID", sort=False):
        idx = group.index
        mins = group["MIN"].astype(float)

        # ── Rolling windows ──────────────────────────────────────────────
        df.loc[idx, "rolling_min_7"]  = mins.rolling(7,  min_periods=1).mean()
        df.loc[idx, "rolling_min_28"] = mins.rolling(28, min_periods=1).mean()
        df.loc[idx, "rolling_pts_7"]  = group["PTS"].astype(float).rolling(7, min_periods=1).mean()
        df.loc[idx, "rolling_reb_7"]  = group["REB"].astype(float).rolling(7, min_periods=1).mean()
        df.loc[idx, "rolling_ast_7"]  = group["AST"].astype(float).rolling(7, min_periods=1).mean()

        # ── ACWR (core metric) ───────────────────────────────────────────
        r7  = df.loc[idx, "rolling_min_7"]
        r28 = df.loc[idx, "rolling_min_28"]
        df.loc[idx, "acwr"] = (r7 / r28.replace(0, np.nan)).fillna(1.0)

        # ── Rest & scheduling ────────────────────────────────────────────
        dates = group["GAME_DATE"]
        days_rest = dates.diff().dt.days.fillna(3.0)
        df.loc[idx, "days_rest"] = days_rest
        df.loc[idx, "back_to_back"] = (days_rest == 1).astype(int)
        df.loc[idx, "no_rest_2d"]  = (days_rest <= 2).astype(int)

        # ── Games played in rolling 30-day window ────────────────────────
        game_count_30 = []
        date_list = dates.tolist()
        for i, d in enumerate(date_list):
            cutoff = d - pd.Timedelta(days=30)
            count = sum(1 for dd in date_list[:i] if dd >= cutoff)
            game_count_30.append(count)
        df.loc[idx, "games_played_30"] = game_count_30

        # ── Composite fatigue score ──────────────────────────────────────
        gp30_norm = np.array(game_count_30) / 15.0  # normalize to [0,2]
        acwr_vals = df.loc[idx, "acwr"].values
        no_rest   = df.loc[idx, "no_rest_2d"].values
        df.loc[idx, "fatigue_score"] = (
            acwr_vals * 0.5 +
            gp30_norm * 0.3 +
            no_rest   * 0.2
        )

    return df


def add_injury_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy injury label: a player is marked injured=1 if they MISS the next game.
    (i.e., next row for that player has days_rest >= 5, implying a missed game).
    
    This is a simplified proxy — in production you'd scrape prosportstransactions.com.
    For the hackathon, this gives a real binary classification target.
    """
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    df["injured"] = 0

    for player_id, group in df.groupby("PLAYER_ID"):
        idx = group.index.tolist()
        for i in range(len(idx) - 1):
            curr_i = idx[i]
            next_i = idx[i + 1]
            days_gap = (
                df.loc[next_i, "GAME_DATE"] - df.loc[curr_i, "GAME_DATE"]
            ).days
            # If next appearance is 8+ days later, player likely missed games
            if days_gap >= 8:
                df.loc[curr_i, "injured"] = 1

    return df


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demographic / career features."""
    df = df.copy()

    # Game number in season (proxy for accumulated fatigue)
    df["game_num_in_season"] = df.groupby(["PLAYER_ID", "SEASON"]).cumcount() + 1

    # Points per minute (efficiency — high value + high minutes = risk)
    df["pts_per_min"] = df["PTS"].astype(float) / df["MIN"].astype(float).replace(0, np.nan)
    df["pts_per_min"] = df["pts_per_min"].fillna(0)

    # Win/loss (road trips often correlate with fatigue)
    df["is_away"] = df["MATCHUP"].str.contains("@").astype(int) if "MATCHUP" in df.columns else 0

    return df


def build_ml_dataset() -> pd.DataFrame:
    raw_path = RAW_DIR / "nba_game_logs.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run src/data/download.py first."
        )

    print("Loading raw data...")
    df = pd.read_csv(raw_path)
    print(f"  Raw rows: {len(df)}")

    print("Parsing dates...")
    df = parse_game_date(df)

    print("Engineering rolling features...")
    df = add_rolling_features(df)

    print("Engineering player features...")
    df = add_player_features(df)

    print("Adding injury target label...")
    df = add_injury_target(df)

    # ── Select final feature set ─────────────────────────────────────────
    FEATURE_COLS = [
        "PLAYER_ID", "PLAYER_NAME", "SEASON", "GAME_DATE",
        "MIN", "PTS", "REB", "AST",
        "rolling_min_7", "rolling_min_28", "acwr",
        "rolling_pts_7", "rolling_reb_7", "rolling_ast_7",
        "days_rest", "back_to_back", "no_rest_2d",
        "games_played_30", "fatigue_score",
        "game_num_in_season", "pts_per_min", "is_away",
        "injured",  # TARGET
    ]
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_final = df[available].copy()

    # ── Drop rows with NaN in key features ──────────────────────────────
    df_final = df_final.dropna(subset=["rolling_min_7", "acwr", "injured"])

    out_path = PROCESSED_DIR / "ml_dataset.csv"
    df_final.to_csv(out_path, index=False)

    injury_count = df_final["injured"].sum()
    print(f"\n✅ ML dataset saved → {out_path}")
    print(f"   Total rows:    {len(df_final)}")
    print(f"   Injury events: {int(injury_count)} ({injury_count/len(df_final)*100:.1f}%)")
    print(f"   Features:      {len(available) - 4} (excluding ID/meta cols)")

    return df_final


if __name__ == "__main__":
    build_ml_dataset()
```
</action>

<acceptance_criteria>
- `src/data/features.py` exists
- `src/data/features.py` contains `def build_ml_dataset()`
- `src/data/features.py` contains `acwr`
- `src/data/features.py` contains `days_rest`
- `src/data/features.py` contains `fatigue_score`
- `src/data/features.py` contains `def add_injury_target(`
- `src/data/features.py` contains `def add_rolling_features(`
</acceptance_criteria>

</tasks>

<verification>
```bash
# Install dependencies
pip install -r requirements.txt

# Run download (takes 10-15 min due to API rate limiting)
python src/data/download.py

# Run feature engineering
python src/data/features.py

# Verify output
test -f data/processed/ml_dataset.csv && echo "✅ Dataset ready" || echo "❌ Missing"
python -c "
import pandas as pd
df = pd.read_csv('data/processed/ml_dataset.csv')
print(f'Rows: {len(df)}')
print(f'Injured rate: {df[\"injured\"].mean():.2%}')
print(f'Columns: {list(df.columns)}')
print(f'ACWR stats: min={df[\"acwr\"].min():.2f} max={df[\"acwr\"].max():.2f}')
"
```
</verification>

<success_criteria>
- `data/processed/ml_dataset.csv` exists with >500 rows
- Dataset has `acwr`, `fatigue_score`, `injured` columns
- `injured` column has values in {0, 1} with 2-15% positive rate
- No NaN values in FEATURE_COLS after preprocessing
</success_criteria>
