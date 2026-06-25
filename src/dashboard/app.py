"""
app.py — GuardianAI Streamlit Dashboard

Run: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.model.explain import load_model_artifacts, predict_team, predict_risk
from src.dashboard.components import (
    risk_card_html, team_scatter_chart, shap_bar_chart, metrics_panel
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuardianAI — NBA Injury Risk",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0E1117; }
[data-testid="stSidebar"]          { background: #0A0D14; }
[data-testid="metric-container"]   {
    background: #1C2333;
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid #2A3347;
}
h1, h2, h3 { color: #FAFAFA !important; }
.guardian-header {
    background: linear-gradient(135deg, #1a2035 0%, #0E1117 100%);
    border-left: 4px solid #FF4B4B;
    padding: 18px 24px;
    border-radius: 0 12px 12px 0;
    margin-bottom: 28px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading GuardianAI model...")
def _load_artifacts():
    return load_model_artifacts()


@st.cache_data(show_spinner="⏳ Loading player data...")
def _load_df():
    path = Path("data/processed/ml_dataset.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 GuardianAI")
    st.caption("NBA Injury Risk & Load Management")
    st.divider()

    df_all    = _load_df()
    artifacts = None

    if df_all is not None:
        try:
            artifacts = _load_artifacts()
        except Exception as e:
            st.error(f"Model load error: {e}")

    st.markdown("### 🔎 Filters")
    player_options = ["All Players"] + (
        sorted(df_all["PLAYER_NAME"].unique().tolist()) if df_all is not None else []
    )
    season_options = ["All Seasons"] + (
        sorted(df_all["SEASON"].unique().tolist(), reverse=True) if df_all is not None else []
    )

    selected_player = st.selectbox("Player", player_options)
    selected_season = st.selectbox("Season",  season_options)

    st.divider()
    st.markdown("### ⚙️ Risk Thresholds")
    high_thresh   = st.slider("🔴 HIGH risk (%)",   40, 90, 60, 5)
    medium_thresh = st.slider("🟡 MEDIUM risk (%)", 15, 59, 35, 5)

    st.divider()
    st.caption("**AQX Sports Analytics Hackathon 2026**")
    st.caption("Model: XGBoost + SHAP  \nData: nba_api (2018–2024)")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="guardian-header">
  <h1 style="margin:0;font-size:2rem;">🏀 GuardianAI</h1>
  <p style="margin:6px 0 0;color:#AAA;font-size:15px;">
    NBA Injury Risk &amp; Load Management — 
    <em>Protect your roster, not just track it.</em>
  </p>
</div>
""", unsafe_allow_html=True)


# ── Error / setup state ───────────────────────────────────────────────────────
if df_all is None or artifacts is None:
    st.error("### 🚧 Data or Model Not Found")
    st.markdown("""
Run the setup pipeline first:
```bash
# Step 1 — Download NBA data (~15 min, API rate-limited)
python src/data/download.py

# Step 2 — Engineer features
python src/data/features.py

# Step 3 — Train model (~5 min)
python src/model/train.py

# Step 4 — Launch dashboard
streamlit run src/dashboard/app.py
```
""")
    st.stop()


# ── Filter data ───────────────────────────────────────────────────────────────
df = df_all.copy()
if selected_season != "All Seasons":
    df = df[df["SEASON"] == selected_season]
if selected_player != "All Players":
    df = df[df["PLAYER_NAME"] == selected_player]

if df.empty:
    st.warning("No data for current filters. Adjust the sidebar selections.")
    st.stop()


# ── Compute team risk ─────────────────────────────────────────────────────────
with st.spinner("Computing injury risk predictions..."):
    pname = None if selected_player == "All Players" else selected_player
    team_risk = predict_team(df, artifacts, player_name=pname)


def relabel_risk(row):
    p = row["Risk %"]
    if p >= high_thresh:   return f"🔴 HIGH"
    if p >= medium_thresh: return f"🟡 MEDIUM"
    return "🟢 LOW"

team_risk["Risk"] = team_risk.apply(relabel_risk, axis=1)

high_players   = team_risk[team_risk["Risk"].str.contains("HIGH")]
medium_players = team_risk[team_risk["Risk"].str.contains("MEDIUM")]


# ── Executive Summary ─────────────────────────────────────────────────────────
st.markdown("## 📊 Executive Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Players Analyzed", len(team_risk))
c2.metric("🔴 High Risk",     len(high_players),
          delta=f"{len(high_players)/max(len(team_risk),1)*100:.0f}% of roster")
c3.metric("🟡 Medium Risk",   len(medium_players))
c4.metric("Avg ACWR",         f"{team_risk['ACWR'].mean():.2f}",
          delta="↑ above 1.0 = elevated" if team_risk["ACWR"].mean() > 1.0 else None)

if not high_players.empty:
    names = ", ".join(high_players["Player"].tolist()[:4])
    st.warning(f"⚠️ **{len(high_players)} player(s) flagged HIGH risk:** {names}")
elif team_risk["ACWR"].mean() > 1.3:
    st.info("📊 No HIGH risk, but team average ACWR is elevated — monitor workload.")
else:
    st.success("✅ All players at LOW or MEDIUM risk.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🃏 Risk Cards", "📈 Team Overview", "🧠 SHAP", "📐 Model Metrics"
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Risk Cards
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Player Risk Cards")
    st.caption("Sorted highest → lowest risk. Each card shows the top 3 SHAP factors driving the prediction.")

    latest = df.sort_values("GAME_DATE").groupby("PLAYER_NAME").last().reset_index()

    if team_risk.empty:
        st.info("No players to display.")
    else:
        col_a, col_b = st.columns(2)
        for i, (_, row) in enumerate(team_risk.iterrows()):
            prow = latest[latest["PLAYER_NAME"] == row["Player"]]
            if prow.empty:
                continue
            pred  = predict_risk(prow.iloc[0], artifacts)
            html  = risk_card_html(
                player       = row["Player"],
                risk_pct     = row["Risk %"],
                risk_label   = pred["risk_label"],
                top_factors  = pred["top_factors"],
                recommendation = row["Recommendation"],
                acwr         = row["ACWR"],
                days_rest    = row["Days Rest"],
            )
            (col_a if i % 2 == 0 else col_b).markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Team Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Workload vs. Injury Risk")
    st.caption("Point size ∝ Risk %. Zone shaded red = ACWR danger zone (> 1.5).")

    # Merge fatigue
    fat_map = df.groupby("PLAYER_NAME")["fatigue_score"].last()
    team_risk["Fatigue"] = team_risk["Player"].map(fat_map).fillna(0)

    st.plotly_chart(team_scatter_chart(team_risk), use_container_width=True)

    st.markdown("### Risk Table")
    display_cols = ["Player", "Risk %", "Risk", "ACWR", "Days Rest", "B2B", "Fatigue", "Recommendation"]
    avail = [c for c in display_cols if c in team_risk.columns]
    styled = (
        team_risk[avail]
        .style
        .background_gradient(subset=["Risk %"], cmap="RdYlGn_r")
        .format({"Risk %": "{:.1f}", "ACWR": "{:.2f}", "Fatigue": "{:.2f}"})
    )
    st.dataframe(styled, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Feature Importance (Mean |SHAP|)")
    st.caption(
        "Red bars = injury-risk drivers. Blue bars = neutral/protective. "
        "SHAP measures each feature's average contribution across all predictions."
    )

    imp = artifacts.get("importance", pd.DataFrame())
    if not imp.empty:
        st.plotly_chart(shap_bar_chart(imp), use_container_width=True)
    else:
        st.info("SHAP importance not found — re-run src/model/train.py")

    st.markdown("### Understanding ACWR")
    st.markdown("""
| ACWR | Zone | Injury Risk |
|------|------|-------------|
| < 0.8 | Under-trained | Moderate ↑ |
| 0.8 – 1.3 | ✅ Sweet Spot | Low |
| 1.3 – 1.5 | ⚠️ Caution | Elevated |
| **> 1.5** | **🔴 Danger Zone** | **High** |

`ACWR = (7-day avg minutes) ÷ (28-day avg minutes)`

*Source: Hulin et al. (2016), British Journal of Sports Medicine*
""")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    m = artifacts["metrics"]
    train_s = ", ".join(m.get("train_seasons", []))
    test_s  = ", ".join(m.get("test_seasons",  []))

    st.markdown("### Model Validation")
    st.caption(f"Train: **{train_s}** | Test: **{test_s}** | Temporal split — no data leakage")

    metrics_panel(m)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Why Temporal Split?")
        st.info("""
**Random splits cause data leakage** in sports prediction.

A random split lets a game from 2024 inform predictions about
2019 games — the future leaking into the past. We always train
on older seasons and test on newer ones to simulate real deployment.
""")

    with col_r:
        st.markdown("#### Feature Summary")
        st.markdown(f"""
| Feature | Role |
|---------|------|
| **ACWR** | Core injury predictor (sports-science gold standard) |
| **Fatigue Score** | Composite load metric |
| **Back-to-Back** | Recovery deprivation flag |
| **Days Rest** | Recovery window |
| **Rolling Min 7d** | Acute workload |
| **Rolling Min 28d** | Chronic workload baseline |
| **Games/30d** | Schedule density |
""")

    st.markdown("---")
    st.caption(
        f"Model: XGBoost | {m.get('feature_count', len(artifacts['feature_cols']))} features | "
        f"Threshold: {m.get('threshold', 0.3):.0%} | "
        f"Built for AQX Sports Analytics Hackathon 2026"
    )
