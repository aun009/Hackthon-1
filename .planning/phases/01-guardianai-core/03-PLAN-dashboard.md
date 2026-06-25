---
phase: 3
plan: 03-streamlit-dashboard
wave: 1
type: execute
depends_on: [02-model-training]
files_modified:
  - src/dashboard/app.py
  - src/dashboard/components.py
  - .streamlit/config.toml
autonomous: true
---

<objective>
Build the dark-theme Streamlit dashboard: executive summary, player risk cards,
team scatter plot, SHAP bar chart, and model performance metrics panel.
This is the primary judge-facing deliverable.
</objective>

<tasks>

## Task 1 — Streamlit Config (Dark Theme)

<read_first>
- requirements.txt
</read_first>

<action>
Create `.streamlit/config.toml` with:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1C2333"
textColor = "#FAFAFA"
font = "sans serif"

[server]
headless = true
port = 8501
```

Create `.streamlit/` directory first with `mkdir -p .streamlit`.
</action>

<acceptance_criteria>
- `.streamlit/config.toml` exists
- `.streamlit/config.toml` contains `backgroundColor = "#0E1117"`
- `.streamlit/config.toml` contains `primaryColor = "#FF4B4B"`
</acceptance_criteria>

---

## Task 2 — UI Components (`src/dashboard/components.py`)

<read_first>
- src/model/explain.py
</read_first>

<action>
Write `src/dashboard/components.py` with reusable card and chart functions:

```python
"""
components.py — Reusable Streamlit UI components for GuardianAI.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st


RISK_COLORS = {
    "HIGH":   "#FF4B4B",
    "MEDIUM": "#FFA500",
    "LOW":    "#00CC66",
}

CARD_BG = {
    "HIGH":   "rgba(255,75,75,0.12)",
    "MEDIUM": "rgba(255,165,0,0.12)",
    "LOW":    "rgba(0,204,102,0.10)",
}


def risk_card_html(player: str, risk_pct: float, risk_label: str,
                   top_factors: list, recommendation: str,
                   acwr: float, days_rest: int) -> str:
    """Generate a risk card as an HTML string for st.markdown."""
    color = RISK_COLORS[risk_label]
    bg    = CARD_BG[risk_label]
    icon  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk_label]

    factors_html = "".join(
        f'<div style="font-size:12px;color:#AAA;margin:2px 0;">'
        f'• <b>{f["feature"]}</b>: {f["value"]} ({f["direction"]})</div>'
        for f in top_factors[:3]
    )

    return f"""
    <div style="
        background:{bg};
        border:1.5px solid {color};
        border-radius:12px;
        padding:16px 20px;
        margin:8px 0;
        font-family:'Segoe UI',sans-serif;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:18px;font-weight:700;color:#FFF;">{player}</span>
                <span style="margin-left:10px;font-size:13px;color:#AAA;">
                    ACWR {acwr:.2f} | {days_rest}d rest
                </span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:26px;font-weight:800;color:{color};">{risk_pct:.0f}%</span>
                <span style="margin-left:6px;font-size:14px;color:{color};font-weight:600;">
                    {icon} {risk_label}
                </span>
            </div>
        </div>
        <div style="margin-top:8px;font-size:13px;color:#DDD;font-weight:600;">
            {recommendation}
        </div>
        <div style="margin-top:6px;">{factors_html}</div>
    </div>
    """


def team_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: ACWR (x) vs Fatigue Score (y), sized by Risk %,
    colored by risk label. Danger zone shaded.
    """
    df = df.copy()
    df["color"] = df["Risk"].apply(
        lambda x: RISK_COLORS.get(x.split()[-1].upper(), "#888")
    )

    fig = go.Figure()

    # Danger zone rectangle (ACWR > 1.5)
    fig.add_vrect(
        x0=1.5, x1=df["ACWR"].max() * 1.1 if len(df) else 2.5,
        fillcolor="rgba(255,75,75,0.08)",
        line_width=0,
        annotation_text="⚠ Danger Zone (ACWR > 1.5)",
        annotation_position="top left",
        annotation_font_color="#FF4B4B",
        annotation_font_size=11,
    )

    # Player points
    for label, color in RISK_COLORS.items():
        sub = df[df["Risk"].str.contains(label, na=False)]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["ACWR"],
            y=sub["Fatigue"],
            mode="markers+text",
            marker=dict(
                size=sub["Risk %"] / 4 + 10,
                color=color,
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            text=sub["Player"].str.split().str[-1],  # last name
            textposition="top center",
            textfont=dict(size=10, color="#DDD"),
            name=label,
        ))

    fig.update_layout(
        title="Team Workload vs. Injury Risk",
        xaxis_title="ACWR (Acute:Chronic Workload Ratio)",
        yaxis_title="Fatigue Score",
        plot_bgcolor="#1C2333",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        height=420,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#555", annotation_text="Baseline")
    fig.add_vline(x=1.5, line_dash="dash", line_color="#FF4B4B", opacity=0.5)

    return fig


def shap_bar_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of mean |SHAP| importance."""
    top = importance_df.head(10).sort_values("importance")
    colors = [
        "#FF4B4B" if "acwr" in f or "fatigue" in f or "back_to_back" in f
        else "#4B9BFF"
        for f in top["feature"]
    ]

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"].str.replace("_", " ").str.title(),
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        plot_bgcolor="#1C2333",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        height=380,
    )
    return fig


def metrics_panel(metrics: dict):
    """Display model performance KPIs in Streamlit columns."""
    cols = st.columns(4)
    kpis = [
        ("ROC AUC",    f"{metrics.get('roc_auc', 0):.3f}",    "Separability"),
        ("Precision",  f"{metrics.get('precision', 0):.3f}",   "When flagged = injury"),
        ("Recall",     f"{metrics.get('recall', 0):.3f}",      "Injuries caught"),
        ("Avg Prec.",  f"{metrics.get('avg_precision', 0):.3f}", "Area under PR curve"),
    ]
    for col, (label, value, desc) in zip(cols, kpis):
        col.metric(label, value, help=desc)
```
</action>

<acceptance_criteria>
- `src/dashboard/components.py` exists
- `src/dashboard/components.py` contains `def risk_card_html(`
- `src/dashboard/components.py` contains `def team_scatter_chart(`
- `src/dashboard/components.py` contains `def shap_bar_chart(`
- `src/dashboard/components.py` contains `def metrics_panel(`
- `src/dashboard/components.py` contains `ACWR > 1.5`
</acceptance_criteria>

---

## Task 3 — Main App (`src/dashboard/app.py`)

<read_first>
- src/dashboard/components.py
- src/model/explain.py
- .streamlit/config.toml
</read_first>

<action>
Write `src/dashboard/app.py` — the main Streamlit application:

```python
"""
app.py — GuardianAI Streamlit Dashboard

Run with:  streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.model.explain import load_model_artifacts, predict_team, predict_risk
from src.dashboard.components import (
    risk_card_html, team_scatter_chart, shap_bar_chart, metrics_panel
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuardianAI — NBA Injury Risk Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS injection ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0E1117; }
    .stMetric { background: #1C2333; border-radius: 10px; padding: 12px; }
    .stSidebar { background: #0E1117; }
    h1 { color: #FF4B4B !important; font-weight: 800 !important; }
    h2, h3 { color: #FAFAFA !important; }
    .guardian-header {
        background: linear-gradient(135deg, #1C2333 0%, #0E1117 100%);
        border-left: 4px solid #FF4B4B;
        padding: 20px 24px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading GuardianAI model...")
def load_artifacts():
    return load_model_artifacts()


@st.cache_data(show_spinner="Loading player data...")
def load_player_data():
    path = Path("data/processed/ml_dataset.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 GuardianAI")
    st.markdown("**NBA Injury Risk & Load Management**")
    st.divider()

    st.markdown("### Filters")
    artifacts = load_artifacts()
    df_all = load_player_data()

    if df_all is not None:
        players = ["All Players"] + sorted(df_all["PLAYER_NAME"].unique().tolist())
        selected_player = st.selectbox("Select Player", players)

        seasons = ["All Seasons"] + sorted(df_all["SEASON"].unique().tolist(), reverse=True)
        selected_season = st.selectbox("Season", seasons)
    else:
        selected_player = "All Players"
        selected_season = "All Seasons"

    st.divider()
    st.markdown("### Risk Threshold")
    high_thresh   = st.slider("HIGH risk (%)",   50, 80, 60)
    medium_thresh = st.slider("MEDIUM risk (%)", 20, 50, 35)

    st.divider()
    st.caption("Built for AQX Sports Analytics Hackathon 2026")
    st.caption("Model: XGBoost + SHAP | Data: nba_api")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="guardian-header">
    <h1 style="margin:0;">🏀 GuardianAI</h1>
    <p style="margin:4px 0 0;color:#AAA;font-size:16px;">
        NBA Injury Risk & Load Management Dashboard — <em>Protect your roster, not just track it.</em>
    </p>
</div>
""", unsafe_allow_html=True)


# ── Error state if no data ───────────────────────────────────────────────────
if df_all is None or artifacts is None:
    st.error("""
    **Model or data not found.** Please run the setup pipeline first:
    ```bash
    python src/data/download.py    # ~15 min
    python src/data/features.py
    python src/model/train.py      # ~5 min
    ```
    """)
    st.stop()


# ── Apply filters ─────────────────────────────────────────────────────────────
df_filtered = df_all.copy()
if selected_season != "All Seasons":
    df_filtered = df_filtered[df_filtered["SEASON"] == selected_season]
if selected_player != "All Players":
    df_filtered = df_filtered[df_filtered["PLAYER_NAME"] == selected_player]


# ── Compute team risk ─────────────────────────────────────────────────────────
team_risk = predict_team(df_filtered, artifacts,
                         player_name=None if selected_player == "All Players" else selected_player)

# Override thresholds from sidebar
def relabel(row):
    p = row["Risk %"]
    if p >= high_thresh:
        return f"🔴 HIGH"
    elif p >= medium_thresh:
        return f"🟡 MEDIUM"
    return f"🟢 LOW"
team_risk["Risk"] = team_risk.apply(relabel, axis=1)

high_risk   = team_risk[team_risk["Risk"].str.contains("HIGH")]
medium_risk = team_risk[team_risk["Risk"].str.contains("MEDIUM")]


# ── Executive Summary ──────────────────────────────────────────────────────────
st.markdown("## 📊 Executive Summary")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Players Analyzed", len(team_risk))
c2.metric("🔴 High Risk",   len(high_risk),   delta=f"{len(high_risk)/max(len(team_risk),1)*100:.0f}%")
c3.metric("🟡 Medium Risk", len(medium_risk), delta=None)
c4.metric("Avg ACWR",       f"{team_risk['ACWR'].mean():.2f}")

if len(high_risk) > 0:
    players_str = ", ".join(high_risk["Player"].tolist()[:3])
    st.warning(f"⚠️ **{len(high_risk)} player(s) at HIGH injury risk tonight:** {players_str}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🃏 Risk Cards", "📈 Team Overview", "🧠 SHAP Explainability", "📐 Model Performance"
])


# ────────────────────────────────────────────────────────────────────────────
# Tab 1: Risk Cards
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Player Risk Cards")
    st.caption("Sorted by injury risk (highest first). Cards show top SHAP factors driving the prediction.")

    # Get most recent game per player for SHAP
    latest = df_filtered.sort_values("GAME_DATE").groupby("PLAYER_NAME").last().reset_index()

    if len(team_risk) == 0:
        st.info("No players match current filters.")
    else:
        col_a, col_b = st.columns(2)
        for i, (_, row) in enumerate(team_risk.iterrows()):
            player_row = latest[latest["PLAYER_NAME"] == row["Player"]]
            if player_row.empty:
                continue
            pred = predict_risk(player_row.iloc[0], artifacts)

            card_html = risk_card_html(
                player=row["Player"],
                risk_pct=row["Risk %"],
                risk_label=pred["risk_label"],
                top_factors=pred["top_factors"],
                recommendation=row["Recommendation"],
                acwr=row["ACWR"],
                days_rest=row["Days Rest"],
            )
            if i % 2 == 0:
                col_a.markdown(card_html, unsafe_allow_html=True)
            else:
                col_b.markdown(card_html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Tab 2: Team Overview
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Workload vs. Injury Risk")
    st.caption("Point size = Risk %. Points in the red shaded zone (ACWR > 1.5) are in the danger zone.")

    if "Fatigue" not in team_risk.columns:
        # Merge fatigue from df
        fatigue_map = df_filtered.groupby("PLAYER_NAME")["fatigue_score"].last()
        team_risk["Fatigue"] = team_risk["Player"].map(fatigue_map).fillna(0)

    fig_scatter = team_scatter_chart(team_risk)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Full Risk Table")
    display_cols = ["Player", "Risk %", "Risk", "ACWR", "Days Rest", "B2B", "Fatigue", "Recommendation"]
    available_display = [c for c in display_cols if c in team_risk.columns]
    st.dataframe(
        team_risk[available_display].style.background_gradient(
            subset=["Risk %"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Tab 3: SHAP Explainability
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### SHAP Feature Importance")
    st.caption("""
    SHAP (SHapley Additive exPlanations) measures each feature's contribution to every prediction.
    Red bars = injury-risk drivers (ACWR, fatigue, back-to-back). Blue bars = protective factors.
    """)

    importance_df = artifacts.get("importance", pd.DataFrame())
    if not importance_df.empty:
        fig_shap = shap_bar_chart(importance_df)
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("### How ACWR Works")
    st.markdown("""
    | ACWR Range | Zone | Injury Risk |
    |------------|------|------------|
    | < 0.8 | Under-trained | Moderate ↑ |
    | 0.8 – 1.3 | Sweet Spot | **Low ✅** |
    | 1.3 – 1.5 | Caution | Elevated ⚠️ |
    | > 1.5 | Danger Zone | **High 🔴** |

    *ACWR = (7-day avg minutes) ÷ (28-day avg minutes)*
    """)


# ────────────────────────────────────────────────────────────────────────────
# Tab 4: Model Performance
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Model Validation")
    st.caption(f"""
    Trained on seasons: {', '.join(artifacts['metrics'].get('train_seasons', []))} |
    Tested on: {', '.join(artifacts['metrics'].get('test_seasons', []))} |
    **Temporal split — no data leakage**
    """)

    metrics_panel(artifacts["metrics"])

    st.markdown("### Why Temporal Split Matters")
    st.info("""
    **Random splits cause data leakage in sports prediction.**
    
    If we randomly split game-by-game, a 2024 game can inform predictions about 2019 games —
    that's the future leaking into the past. We always train on older seasons (2018–2022)
    and test on newer ones (2022–2024) to simulate real-world deployment.
    """)

    st.markdown("### Feature Engineering Summary")
    st.markdown(f"""
    | Feature | Description | Why It Matters |
    |---------|-------------|---------------|
    | **ACWR** | Acute:Chronic Workload Ratio | Gold standard sports-science injury predictor |
    | **Fatigue Score** | Composite workload metric | Holistic load signal |
    | **Back-to-Back** | Played yesterday? | Recovery deprivation |
    | **Days Rest** | Days since last game | Recovery time |
    | **Rolling Min (7d)** | Avg minutes, last 7 games | Acute workload |
    | **Rolling Min (28d)** | Avg minutes, last 28 games | Chronic workload |
    | **Games (30d)** | Games played in 30 days | Schedule density |
    """)
```
</action>

<acceptance_criteria>
- `src/dashboard/app.py` exists
- `src/dashboard/app.py` contains `st.set_page_config(`
- `src/dashboard/app.py` contains `GuardianAI`
- `src/dashboard/app.py` contains `tab1, tab2, tab3, tab4`
- `src/dashboard/app.py` contains `def load_artifacts()`
- `src/dashboard/app.py` contains `risk_card_html`
- `src/dashboard/app.py` contains `team_scatter_chart`
- `src/dashboard/app.py` contains `shap_bar_chart`
- `src/dashboard/app.py` contains `metrics_panel`
</acceptance_criteria>

</tasks>

<verification>
```bash
mkdir -p .streamlit

# Launch dashboard
streamlit run src/dashboard/app.py

# Should open at http://localhost:8501
# Verify:
# 1. Dark theme loads
# 2. All 4 tabs visible
# 3. Executive summary shows player count
# 4. Risk cards appear in two-column layout
# 5. SHAP chart renders
```
</verification>

<success_criteria>
- Dashboard loads at localhost:8501 without errors
- All 4 tabs render: Risk Cards, Team Overview, SHAP, Model Performance
- Risk cards display in High/Medium/Low color coding
- Team scatter plot shows ACWR danger zone annotation
- SHAP bar chart shows top 10 features
- Model metrics (AUC, precision, recall) display in KPI panel
</success_criteria>
