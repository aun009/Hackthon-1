"""
download.py — Fetch NBA player game logs via nba_api.

Pulls ~20 high-minutes players across 6 seasons (2018-19 to 2023-24).
Saves combined logs to data/raw/nba_game_logs.csv.

Run time: 10-15 minutes (API rate-limited at 0.7s/call)
"""

import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24",
]

TARGET_PLAYER_NAMES = [
    "LeBron James", "Kevin Durant", "Stephen Curry",
    "Kawhi Leonard", "Anthony Davis", "Ja Morant",
    "Zion Williamson", "Joel Embiid", "Jayson Tatum",
    "Luka Doncic", "Damian Lillard", "Devin Booker",
    "Giannis Antetokounmpo", "Nikola Jokic", "Paul George",
    "Jimmy Butler", "Kyrie Irving", "Chris Paul",
    "Donovan Mitchell", "De'Aaron Fox",
]


def get_player_id(name: str):
    all_players = players.get_active_players() + players.get_inactive_players()
    for p in all_players:
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
            print(f"  ⚠ {player_name} {season}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_all() -> pd.DataFrame:
    out = RAW_DIR / "nba_game_logs.csv"
    if out.exists():
        print(f"✅ Data already downloaded: {out} — delete to re-fetch")
        return pd.read_csv(out)

    all_logs = []
    for i, name in enumerate(TARGET_PLAYER_NAMES, 1):
        print(f"[{i}/{len(TARGET_PLAYER_NAMES)}] Fetching {name}...")
        pid = get_player_id(name)
        if pid is None:
            print(f"  ✗ Not found: {name}")
            continue
        df = fetch_player_logs(pid, name)
        if not df.empty:
            all_logs.append(df)
            print(f"  ✓ {len(df)} games")

    if not all_logs:
        raise RuntimeError("No data fetched. Check nba_api installation.")

    combined = pd.concat(all_logs, ignore_index=True)
    combined.to_csv(out, index=False)
    print(f"\n✅ Saved {len(combined)} rows → {out}")
    return combined


if __name__ == "__main__":
    download_all()
