"""
components.py — Reusable Streamlit UI components for GuardianAI.
"""

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

RISK_COLORS = {"HIGH": "#FF4B4B", "MEDIUM": "#FFA500", "LOW": "#00CC66"}
CARD_BG     = {"HIGH": "rgba(255,75,75,0.12)", "MEDIUM": "rgba(255,165,0,0.12)", "LOW": "rgba(0,204,102,0.10)"}
RISK_ICON   = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


def risk_card_html(player: str, risk_pct: float, risk_label: str,
                   top_factors: list, recommendation: str,
                   acwr: float, days_rest: int) -> str:
    """HTML risk card for one player."""
    color = RISK_COLORS.get(risk_label, "#888")
    bg    = CARD_BG.get(risk_label, "rgba(0,0,0,0.1)")
    icon  = RISK_ICON.get(risk_label, "⚪")

    factors_html = "".join(
        f'<div style="font-size:12px;color:#BBB;margin:2px 0;">'
        f'&bull; <b>{f["feature"]}</b>: {f["value"]} <span style="color:#888">({f["direction"]})</span></div>'
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
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <span style="font-size:17px;font-weight:700;color:#FFF;">{player}</span><br>
                <span style="font-size:12px;color:#999;">
                    ACWR&nbsp;<b style="color:{color}">{acwr:.2f}</b>
                    &nbsp;|&nbsp;{days_rest}d rest
                </span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:28px;font-weight:800;color:{color};">{risk_pct:.0f}%</span><br>
                <span style="font-size:13px;color:{color};font-weight:600;">{icon} {risk_label}</span>
            </div>
        </div>
        <div style="margin:10px 0 4px;font-size:13px;color:#DDD;font-weight:600;">{recommendation}</div>
        <div>{factors_html}</div>
    </div>
    """


def team_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter: ACWR (x) vs Fatigue Score (y), sized by Risk %, coloured by label."""
    fig = go.Figure()

    # Danger zone shading
    x_max = max(df["ACWR"].max() * 1.1, 2.0) if len(df) else 2.5
    fig.add_vrect(
        x0=1.5, x1=x_max,
        fillcolor="rgba(255,75,75,0.07)",
        line_width=0,
        annotation_text="⚠ Danger Zone (ACWR > 1.5)",
        annotation_position="top left",
        annotation_font_color="#FF4B4B",
        annotation_font_size=11,
    )

    for label in ["HIGH", "MEDIUM", "LOW"]:
        sub = df[df["Risk"].str.contains(label, na=False)]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["ACWR"],
            y=sub["Fatigue"],
            mode="markers+text",
            marker=dict(
                size=(sub["Risk %"] / 4 + 10).clip(10, 35),
                color=RISK_COLORS[label],
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            text=sub["Player"].str.split().str[-1],
            textposition="top center",
            textfont=dict(size=10, color="#DDD"),
            name=label,
        ))

    fig.update_layout(
        title="Team Workload Map",
        xaxis_title="ACWR (Acute:Chronic Workload Ratio)",
        yaxis_title="Composite Fatigue Score",
        plot_bgcolor="#1C2333",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        height=430,
        margin=dict(t=50, b=40),
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#555",
                  annotation_text="Baseline (1.0)", annotation_font_size=10)
    fig.add_vline(x=1.5, line_dash="dash", line_color="#FF4B4B", opacity=0.6)
    return fig


def shap_bar_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar — mean |SHAP| per feature (top 10)."""
    top = importance_df.head(10).sort_values("importance")
    injury_features = {"acwr", "fatigue_score", "back_to_back", "no_rest_2d", "rolling_min_7"}
    colors = [
        "#FF6B6B" if f in injury_features else "#4B9BFF"
        for f in top["feature"]
    ]

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"].str.replace("_", " ").str.title(),
        orientation="h",
        marker_color=colors,
        text=top["importance"].round(4),
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Feature Importance — Mean |SHAP Value|",
        xaxis_title="Mean |SHAP|",
        plot_bgcolor="#1C2333",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        height=400,
        margin=dict(l=160, r=60, t=50, b=40),
    )
    return fig


def metrics_panel(metrics: dict):
    """Four KPI columns: AUC, Precision, Recall, Avg Precision."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC",         f"{metrics.get('roc_auc', 0):.3f}",       help="Separability score (0.5=random, 1.0=perfect)")
    c2.metric("Precision",       f"{metrics.get('precision', 0):.3f}",     help="When model flags risk — how often is it right?")
    c3.metric("Recall",          f"{metrics.get('recall', 0):.3f}",        help="Of all actual injuries, what % are caught?")
    c4.metric("Avg Precision",   f"{metrics.get('avg_precision', 0):.3f}", help="Area under precision-recall curve")
