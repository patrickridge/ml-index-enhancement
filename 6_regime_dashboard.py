"""
6_regime_dashboard.py — ML Index Enhancement: Regime Backtesting Dashboard
===========================================================================
Interactive Streamlit application for regime-conditional performance analysis
of the CS-Transformer, FT-Transformer, LGBM, and Factor-Combo IE strategies.

Tabs:
  1. Regime Breakdown       — IR, cumulative alpha, hit rate by regime
  2. Bootstrap Analysis     — Block bootstrap confidence intervals on IR
  3. Monte Carlo Projection — Forward simulation of cumulative active return
  4. Statistics             — Full metrics table with IR heatmap
  5. Volatility Surface     — Interactive 3-D cross-sectional vol surface

HOW TO RUN:
  streamlit run 6_regime_dashboard.py
  Opens at: http://localhost:8501

HOW TO STOP:
  Press Ctrl+C in the terminal where it is running.

HOW STREAMLIT WORKS:
  Runs a local web server on port 8501. Every time you interact with a widget
  (slider, dropdown, checkbox) the entire script reruns from top to bottom
  with the updated widget values.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import streamlit as st

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

# ── Regime definitions (no emoji — clean labels) ────────────────────────────
ALL_REGIMES = {
    "GFC Crash (2008–2009)":         ("2008-01-01", "2009-12-31", "#7B241C"),
    "Post-GFC Recovery (2010–2012)": ("2010-01-01", "2012-12-31", "#1A5276"),
    "QE Bull (2013–2019)":           ("2013-01-01", "2019-12-31", "#1E5631"),
    "COVID Crash (Q1 2020)":         ("2020-01-01", "2020-05-31", "#1C2833"),
    "COVID Recovery (2020–2021)":    ("2020-06-01", "2021-12-31", "#117A65"),
    "Rate Hike Bear (2022)":         ("2022-01-01", "2022-12-31", "#784212"),
    "AI Bull (2023–present)":        ("2023-01-01", "2099-12-31", "#4A235A"),
}

MODEL_FILES = {
    "CS-Transformer": "bt_ie_cs_transformer.csv",
    "FT-Transformer": "bt_ie_transformer.csv",
    "LGBM":           "bt_ie_lgbm.csv",
    "Factor-Combo":   "bt_ie_factor_combo.csv",
}

# Professional, academically muted model colours
MODEL_COLORS = {
    "CS-Transformer": "#1565C0",   # navy blue
    "FT-Transformer": "#B71C1C",   # deep red
    "LGBM":           "#2E7D32",   # forest green
    "Factor-Combo":   "#546E7A",   # slate grey
}

# Shared Plotly layout defaults
_LAYOUT = dict(
    plot_bgcolor="#1A1D27",
    paper_bgcolor="#1A1D27",
    font=dict(family="Arial, sans-serif", size=11, color="#E8EDF2"),
    xaxis=dict(showgrid=True, gridcolor="#2A2D3A", linecolor="#3A3D4A",
               linewidth=1, zeroline=False, tickfont=dict(color="#B0B8C4")),
    yaxis=dict(showgrid=True, gridcolor="#2A2D3A", linecolor="#3A3D4A",
               linewidth=1, zeroline=False, tickfont=dict(color="#B0B8C4")),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#E8EDF2")),
    margin=dict(t=60, b=40, l=50, r=30),
)

# ── Custom CSS — academic, clean ────────────────────────────────────────────
_CSS = """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

/* Headings */
h1 { font-size: 1.35rem !important; font-weight: 700; color: #E8EDF2;
     letter-spacing: -0.01em; border-bottom: 2px solid #4A9EE0;
     padding-bottom: 0.35rem; margin-bottom: 0.8rem; }
h2 { font-size: 1.0rem !important; font-weight: 600; color: #D0D8E4;
     border-bottom: 1px solid #2A2D3A; padding-bottom: 0.25rem; margin-top: 1rem; }
h3 { font-size: 0.9rem !important; font-weight: 600; color: #C0C8D4; }
p  { font-size: 0.86rem; color: #A8B4C0; line-height: 1.55; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1A1D27;
    border: 1px solid #2A2D3A;
    border-radius: 5px;
    padding: 0.7rem 0.9rem;
}
[data-testid="metric-container"] label { font-size: 0.78rem !important;
    color: #8A96A8; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.2rem !important; font-weight: 600; color: #E8EDF2; }

/* Tabs */
[data-baseweb="tab-list"] { border-bottom: 2px solid #2A2D3A; gap: 0; }
[data-baseweb="tab"] { font-size: 0.83rem !important; font-weight: 500;
    color: #8A96A8; padding: 0.5rem 1rem; }
[data-baseweb="tab"][aria-selected="true"] { color: #4A9EE0 !important;
    border-bottom: 2px solid #4A9EE0; font-weight: 600; }

/* Sidebar */
[data-testid="stSidebar"] { background: #1A1D27; border-right: 1px solid #2A2D3A; }
[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }
section[data-testid="stSidebar"] h2 { font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 0.06em; color: #6A7A8A;
    border-bottom: 1px solid #2A2D3A; padding-bottom: 0.2rem; }

/* Captions / subtext — bright enough to read on dark bg */
.stCaption, [data-testid="stCaptionContainer"] { color: #9AAABB !important;
    font-size: 0.8rem; font-style: normal; }

/* Info / warning boxes */
.stAlert { border-radius: 4px; font-size: 0.84rem; }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: #1A1D27; border: 1px solid #3A4A5A; color: #C0C8D4;
    font-size: 0.83rem; font-weight: 500; padding: 0.35rem 0.9rem;
    border-radius: 4px; }
[data-testid="stDownloadButton"] > button:hover { background: #2A2D3A; }

/* Dividers */
hr { border: none; border-top: 1px solid #2A2D3A; margin: 1rem 0; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #2A2D3A; border-radius: 4px; }

/* Spinner */
.stSpinner { color: #4A9EE0; }
</style>
"""

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IE Regime Dashboard",
    page_icon=None,
    layout="wide",
)
st.markdown(_CSS, unsafe_allow_html=True)

st.title("ML S&P 500 Index Enhancement — Regime Analysis Dashboard")
st.caption(
    "Out-of-sample performance (Jan 2023 – Nov 2025) decomposed by market regime. "
    "Models: CS-Transformer · FT-Transformer · LGBM · Factor-Combo baseline."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Models")
    selected_models = st.multiselect(
        "Select models:",
        options=list(MODEL_FILES.keys()),
        default=["CS-Transformer", "FT-Transformer", "LGBM"],
    )

    st.markdown("## Regimes")
    selected_regimes = st.multiselect(
        "Select regimes:",
        options=list(ALL_REGIMES.keys()),
        default=[k for k in ALL_REGIMES if "GFC Crash" not in k],
    )

    st.markdown("## Stochastic Engine")
    n_bootstrap = st.slider("Bootstrap iterations", 100, 2000, 1000, step=100)
    block_size  = st.slider("Block size (months)", 1, 6, 3)
    n_mc_months = st.slider("Monte Carlo horizon (months)", 12, 60, 36)
    n_mc_sims   = st.slider("Monte Carlo paths", 100, 2000, 1000, step=100)

    st.markdown("## Display")
    show_ci = st.checkbox("Show 90% CI on IR chart", value=True)

    st.markdown("---")
    st.caption("CS-Transformer IR 1.874 · Test period 2023–2025")

show_gfc_note = "GFC Crash (2008–2009)" in selected_regimes


# ── Helper functions ─────────────────────────────────────────────────────────

def load_backtest(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / MODEL_FILES[name]
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def tag_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = "Unclassified"
    for name, (start, end, _) in ALL_REGIMES.items():
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        df.loc[mask, "regime"] = name
    return df


def regime_stats(returns: pd.Series) -> dict:
    r = returns.dropna()
    n = len(r)
    if n < 3:
        return dict(n=n, ann_alpha=np.nan, track_err=np.nan,
                    ir=np.nan, hit_rate=np.nan, max_dd=np.nan)
    ann_alpha = r.mean() * 12
    track_err = r.std() * np.sqrt(12)
    ir        = ann_alpha / (track_err + 1e-8)
    hit_rate  = (r > 0).mean() * 100
    cum       = (1 + r).cumprod()
    max_dd    = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return dict(n=n,
                ann_alpha = round(ann_alpha * 100, 2),
                track_err = round(track_err * 100, 2),
                ir        = round(ir, 3),
                hit_rate  = round(hit_rate, 1),
                max_dd    = round(max_dd, 2))


def block_bootstrap(returns: pd.Series, n_boot=1000, block_size=3) -> np.ndarray:
    r = returns.dropna().values
    n = len(r)
    if n < 6:
        return np.full(n_boot, np.nan)
    irs = []
    for _ in range(n_boot):
        n_blocks  = max(1, n // block_size)
        starts    = np.random.randint(0, max(1, n - block_size + 1), n_blocks)
        resampled = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        a = resampled.mean() * 12
        v = resampled.std() * np.sqrt(12) + 1e-8
        irs.append(a / v)
    return np.array(irs)


def monte_carlo_projection(ir: float, te: float, n_months: int = 36,
                            n_sims: int = 1000) -> pd.DataFrame:
    monthly_alpha = ir * te / 12
    monthly_vol   = te / np.sqrt(12)
    paths    = np.random.normal(monthly_alpha, monthly_vol, (n_sims, n_months))
    cum_path = np.cumprod(1 + paths, axis=1) - 1
    months   = np.arange(1, n_months + 1)
    return pd.DataFrame({
        "month": months,
        "p5":  np.percentile(cum_path,  5, axis=0) * 100,
        "p25": np.percentile(cum_path, 25, axis=0) * 100,
        "p50": np.percentile(cum_path, 50, axis=0) * 100,
        "p75": np.percentile(cum_path, 75, axis=0) * 100,
        "p95": np.percentile(cum_path, 95, axis=0) * 100,
    })


def apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    cfg = {**_LAYOUT, **kwargs}
    fig.update_layout(**cfg)
    return fig


# ── Load backtest data ───────────────────────────────────────────────────────
@st.cache_data
def load_all_data() -> dict:
    return {
        name: tag_regime(df)
        for name in MODEL_FILES
        if (df := load_backtest(name)) is not None
    }

all_data = load_all_data()

if not all_data:
    st.error("No backtest CSV files found in data/. Run 4b_index_enhancement.py first.")
    st.stop()

available_models = [m for m in selected_models if m in all_data]
if not available_models:
    st.warning("No selected models have backtest data available.")
    st.stop()

if show_gfc_note:
    st.info(
        "GFC period (2008–2009): ML model scores are not available for this window. "
        "Models were trained on data from 2010 onwards. The Factor-Combo baseline "
        "can be extended if price data prior to 2010 is loaded."
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Regime Breakdown",
    "Bootstrap Analysis",
    "Monte Carlo Projection",
    "Statistics",
    "Volatility Surface",
    "Walk-Forward RL",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Regime Breakdown
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    # Build regime stats table
    rows = []
    for model in available_models:
        df = all_data[model]
        rows.append({"Model": model, "Regime": "Full Period", **regime_stats(df["active_ret"])})
        for reg in selected_regimes:
            sub = df[df["regime"] == reg]["active_ret"]
            rows.append({"Model": model, "Regime": reg, **regime_stats(sub)})
    stats_df = pd.DataFrame(rows)

    # ── IR bar chart
    st.markdown("## Information Ratio by Regime")
    st.caption("IR = annualised alpha / tracking error. IR > 0.5: institutional-grade. IR > 1.0: top-quartile.")

    fig_ir = go.Figure()
    all_regime_labels = ["Full Period"] + selected_regimes
    for model in available_models:
        sub = stats_df[(stats_df["Model"] == model) &
                       (stats_df["Regime"].isin(all_regime_labels))]
        fig_ir.add_trace(go.Bar(
            name=model,
            x=sub["Regime"].str[:35],
            y=sub["ir"],
            marker_color=MODEL_COLORS.get(model, "#607D8B"),
            marker_line_width=0,
            opacity=0.88,
        ))

    fig_ir.add_hline(y=0.5, line_dash="dot", line_color="#2E7D32", line_width=1.2,
                     annotation_text="0.5", annotation_position="right",
                     annotation_font_size=9)
    fig_ir.add_hline(y=1.0, line_dash="dot", line_color="#F57F17", line_width=1.2,
                     annotation_text="1.0", annotation_position="right",
                     annotation_font_size=9)
    fig_ir.add_hline(y=0, line_color="#AAAAAA", line_width=0.8)
    apply_layout(fig_ir, height=400, barmode="group",
                 xaxis=dict(tickangle=-25, tickfont=dict(size=10)),
                 yaxis_title="Information Ratio")
    st.plotly_chart(fig_ir, use_container_width=True)

    st.divider()

    # ── Cumulative active return
    st.markdown("## Cumulative Active Return")
    st.caption("Portfolio cumulative alpha relative to the S&P 500 benchmark.")

    fig_cum = go.Figure()
    for model in available_models:
        df_s = all_data[model].sort_values("date")
        cum  = (1 + df_s["active_ret"].fillna(0)).cumprod() - 1
        fig_cum.add_trace(go.Scatter(
            x=df_s["date"], y=cum * 100,
            name=model, mode="lines",
            line=dict(color=MODEL_COLORS.get(model, "#607D8B"), width=1.8),
        ))

    for reg in selected_regimes:
        start_str, end_str, color = ALL_REGIMES[reg]
        short = reg.split(" (")[0]
        fig_cum.add_vrect(
            x0=start_str, x1=end_str,
            fillcolor=color, opacity=0.06,
            annotation_text=short[:14],
            annotation_position="top left",
            annotation_font_size=8,
            annotation_font_color="#555",
        )

    fig_cum.add_hline(y=0, line_color="#AAAAAA", line_width=0.8)
    apply_layout(fig_cum, height=380,
                 xaxis_title="Date",
                 yaxis_title="Cumulative Active Return (%)")
    st.plotly_chart(fig_cum, use_container_width=True)

    st.divider()

    # ── Hit rate & ann alpha side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Hit Rate by Regime (%)")
        fig_hr = go.Figure()
        for model in available_models:
            sub = stats_df[(stats_df["Model"] == model) &
                           (stats_df["Regime"].isin(selected_regimes))]
            fig_hr.add_trace(go.Bar(
                name=model, x=sub["Regime"].str[:22], y=sub["hit_rate"],
                marker_color=MODEL_COLORS.get(model, "#607D8B"),
                marker_line_width=0, opacity=0.88,
            ))
        fig_hr.add_hline(y=50, line_dash="dot", line_color="#888",
                         line_width=1.0, annotation_text="50%",
                         annotation_font_size=9)
        apply_layout(fig_hr, height=300, barmode="group", showlegend=False,
                     xaxis=dict(tickangle=-25, tickfont=dict(size=9)),
                     yaxis_title="Hit Rate (%)")
        st.plotly_chart(fig_hr, use_container_width=True)

    with col2:
        st.markdown("## Annualised Alpha by Regime (%)")
        fig_alpha = go.Figure()
        for model in available_models:
            sub = stats_df[(stats_df["Model"] == model) &
                           (stats_df["Regime"].isin(selected_regimes))]
            fig_alpha.add_trace(go.Bar(
                name=model, x=sub["Regime"].str[:22], y=sub["ann_alpha"],
                marker_color=MODEL_COLORS.get(model, "#607D8B"),
                marker_line_width=0, opacity=0.88,
            ))
        fig_alpha.add_hline(y=0, line_color="#AAAAAA", line_width=0.8)
        apply_layout(fig_alpha, height=300, barmode="group", showlegend=False,
                     xaxis=dict(tickangle=-25, tickfont=dict(size=9)),
                     yaxis_title="Ann. Alpha (%)")
        st.plotly_chart(fig_alpha, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bootstrap Analysis
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Block Bootstrap — IR Confidence Intervals")
    st.caption(
        f"Block bootstrap ({n_bootstrap:,} iterations, block size = {block_size} months). "
        "Consecutive blocks preserve temporal autocorrelation. "
        "Vertical bars span the 5th–95th percentile range."
    )

    with st.spinner(f"Running {n_bootstrap:,} bootstrap iterations ..."):
        boot_rows = []
        for model in available_models:
            df = all_data[model]
            for reg in ["Full Period"] + selected_regimes:
                sub = (df["active_ret"] if reg == "Full Period"
                       else df[df["regime"] == reg]["active_ret"])
                if sub.dropna().shape[0] < 6:
                    continue
                boot_irs = block_bootstrap(sub, n_boot=n_bootstrap, block_size=block_size)
                boot_rows.append({
                    "Model":  model,
                    "Regime": reg,
                    "point":  regime_stats(sub)["ir"],
                    "p5":     np.nanpercentile(boot_irs, 5),
                    "p50":    np.nanpercentile(boot_irs, 50),
                    "p95":    np.nanpercentile(boot_irs, 95),
                    "n":      sub.dropna().shape[0],
                })

    boot_df = pd.DataFrame(boot_rows)

    # IR with CI
    regime_labels = ["Full Period"] + selected_regimes
    x_positions   = list(range(len(regime_labels)))
    offsets       = np.linspace(-0.28, 0.28, max(len(available_models), 1))

    fig_boot = go.Figure()
    for i, model in enumerate(available_models):
        sub   = (boot_df[boot_df["Model"] == model]
                 .set_index("Regime").reindex(regime_labels).reset_index())
        x_jit = [x + offsets[i] for x in x_positions]
        color = MODEL_COLORS.get(model, "#607D8B")

        if show_ci:
            for j, row in sub.iterrows():
                if pd.isna(row["p5"]):
                    continue
                fig_boot.add_shape(
                    type="line",
                    x0=x_jit[j], x1=x_jit[j],
                    y0=row["p5"], y1=row["p95"],
                    line=dict(color=color, width=1.8),
                )
                # tick marks
                for yval in [row["p5"], row["p95"]]:
                    fig_boot.add_shape(
                        type="line",
                        x0=x_jit[j] - 0.05, x1=x_jit[j] + 0.05,
                        y0=yval, y1=yval,
                        line=dict(color=color, width=1.8),
                    )

        fig_boot.add_trace(go.Scatter(
            x=x_jit, y=sub["point"],
            mode="markers", name=model,
            marker=dict(color=color, size=9, symbol="circle",
                        line=dict(color="white", width=1.5)),
        ))

    fig_boot.add_hline(y=0,   line_color="#BBBBBB", line_width=0.8)
    fig_boot.add_hline(y=0.5, line_dash="dot", line_color="#2E7D32", line_width=1.0,
                       annotation_text="0.5", annotation_position="right",
                       annotation_font_size=9)
    fig_boot.add_hline(y=1.0, line_dash="dot", line_color="#F57F17", line_width=1.0,
                       annotation_text="1.0", annotation_position="right",
                       annotation_font_size=9)
    apply_layout(fig_boot, height=430,
                 xaxis=dict(tickmode="array", tickvals=x_positions,
                            ticktext=[r[:28] for r in regime_labels],
                            tickangle=-25, tickfont=dict(size=10)),
                 yaxis_title="Information Ratio",
                 title=dict(text="IR Point Estimates with 90% Bootstrap CI",
                            font=dict(size=12), x=0))
    st.plotly_chart(fig_boot, use_container_width=True)

    st.divider()

    # Bootstrap distribution for selected model/regime
    st.markdown("## Bootstrap IR Distribution")
    col_m, col_r = st.columns(2)
    with col_m:
        boot_model  = st.selectbox("Model:", available_models, key="boot_model")
    with col_r:
        boot_regime = st.selectbox("Regime:", ["Full Period"] + selected_regimes,
                                   key="boot_regime")

    df_sel  = all_data[boot_model]
    sub_sel = (df_sel["active_ret"] if boot_regime == "Full Period"
               else df_sel[df_sel["regime"] == boot_regime]["active_ret"])

    if sub_sel.dropna().shape[0] >= 6:
        boot_irs_sel = block_bootstrap(sub_sel, n_boot=n_bootstrap,
                                       block_size=block_size)
        point_ir_sel = regime_stats(sub_sel)["ir"]
        p5, p50, p95 = (np.nanpercentile(boot_irs_sel, q) for q in [5, 50, 95])

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=boot_irs_sel, nbinsx=50,
            marker_color=MODEL_COLORS.get(boot_model, "#607D8B"),
            marker_line_width=0, opacity=0.72, name="Bootstrap IR",
        ))
        for xval, label, dash in [
            (point_ir_sel, f"Point IR = {point_ir_sel:.3f}", "solid"),
            (p5,  f"5th pct = {p5:.3f}",  "dot"),
            (p95, f"95th pct = {p95:.3f}", "dot"),
        ]:
            fig_dist.add_vline(x=xval, line_dash=dash, line_width=1.6,
                               line_color="#1565C0" if dash == "solid" else "#888",
                               annotation_text=label,
                               annotation_font_size=9,
                               annotation_position="top right" if xval == point_ir_sel else "top left")
        fig_dist.add_vline(x=0, line_color="#CCCCCC", line_width=1.0)
        apply_layout(fig_dist, height=300, showlegend=False,
                     xaxis_title="IR",
                     yaxis_title="Count",
                     title=dict(
                         text=f"{boot_model} — {boot_regime}  (n = {sub_sel.dropna().shape[0]} months)",
                         font=dict(size=11), x=0))
        st.plotly_chart(fig_dist, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Point IR",  f"{point_ir_sel:.3f}")
        c2.metric("Median IR", f"{p50:.3f}")
        c3.metric("5th pct",   f"{p5:.3f}")
        c4.metric("95th pct",  f"{p95:.3f}")
    else:
        st.warning(f"Insufficient data for {boot_model} / {boot_regime} "
                   f"({sub_sel.dropna().shape[0]} months — minimum 6 required).")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Monte Carlo Projection
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Monte Carlo Forward Projection")
    st.caption(
        f"Simulates {n_mc_sims:,} forward paths of cumulative active return over "
        f"{n_mc_months} months. Monthly active returns are drawn from "
        "N(IR × TE / 12, TE / √12). Shaded bands show the 5th–95th and "
        "25th–75th percentile ranges."
    )

    col_m, col_r = st.columns(2)
    with col_m:
        mc_model  = st.selectbox("Model:", available_models, key="mc_model")
    with col_r:
        mc_regime = st.selectbox("Base regime:", ["Full Period"] + selected_regimes,
                                 key="mc_regime")

    df_mc  = all_data[mc_model]
    sub_mc = (df_mc["active_ret"] if mc_regime == "Full Period"
              else df_mc[df_mc["regime"] == mc_regime]["active_ret"])
    s_mc   = regime_stats(sub_mc)
    ir_mc  = s_mc["ir"]   if not pd.isna(s_mc["ir"])       else 0.0
    te_mc  = s_mc["track_err"] / 100 if not pd.isna(s_mc["track_err"]) else 0.02

    col_ir, col_te = st.columns(2)
    with col_ir:
        ir_override = st.number_input("Override IR (0 = use historical):", 0.0, 3.0,
                                      value=round(float(ir_mc), 3), step=0.05)
    with col_te:
        te_override = st.number_input("Override TE % (0 = use historical):", 0.0, 20.0,
                                      value=round(float(te_mc * 100), 2), step=0.1)

    ir_use = ir_override if ir_override > 0 else ir_mc
    te_use = te_override / 100 if te_override > 0 else te_mc

    mc_df = monte_carlo_projection(ir_use, te_use, n_months=n_mc_months, n_sims=n_mc_sims)
    color = MODEL_COLORS.get(mc_model, "#1565C0")

    fig_mc = go.Figure()
    # Outer band: 5–95
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p95"], mode="lines",
        line=dict(width=0), showlegend=False, name="95th pct",
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
        name="5th–95th pct",
    ))
    # Inner band: 25–75
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p75"], mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p25"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.22)",
        name="25th–75th pct",
    ))
    # Median
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p50"], mode="lines",
        line=dict(color=color, width=2.2), name="Median",
    ))
    fig_mc.add_hline(y=0, line_color="#AAAAAA", line_width=0.8)
    apply_layout(
        fig_mc, height=420,
        xaxis_title="Months Forward",
        yaxis_title="Cumulative Active Return (%)",
        title=dict(text=(f"{mc_model}  |  IR = {ir_use:.3f}  |  "
                         f"TE = {te_use*100:.1f}%  |  "
                         f"{n_mc_sims:,} paths"),
                   font=dict(size=11), x=0),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    last = mc_df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Median ({n_mc_months}m)", f"{last['p50']:.1f}%")
    c2.metric("5th pct",  f"{last['p5']:.1f}%")
    c3.metric("25th pct", f"{last['p25']:.1f}%")
    c4.metric("75th pct", f"{last['p75']:.1f}%")
    c5.metric("95th pct", f"{last['p95']:.1f}%")

    prob_pos = (mc_df["p50"] > 0).all()
    te_display = te_use * 100
    st.caption(
        f"Median cumulative active return at {n_mc_months} months: {last['p50']:.1f}%. "
        f"Annualised alpha assumption: {ir_use * te_display:.2f}% "
        f"(IR × TE = {ir_use:.3f} × {te_display:.1f}%)."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Statistics
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Performance Statistics — All Models × Regimes")
    st.caption("IR > 0.5 = good (institutional grade). IR > 1.0 = excellent (top-quartile).")

    disp_rows = []
    for model in available_models:
        df = all_data[model]
        for reg in ["Full Period"] + selected_regimes:
            sub = (df["active_ret"] if reg == "Full Period"
                   else df[df["regime"] == reg]["active_ret"])
            s   = regime_stats(sub)
            disp_rows.append({
                "Model":      model,
                "Regime":     reg[:38],
                "N":          s["n"],
                "Ann α (%)":  s["ann_alpha"],
                "TE (%)":     s["track_err"],
                "IR":         s["ir"],
                "Hit Rate %": s["hit_rate"],
                "Max DD %":   s["max_dd"],
            })

    disp_df = pd.DataFrame(disp_rows)

    def colour_ir(val):
        if pd.isna(val):
            return ""
        if val >= 1.0:
            return "background-color: #D5E8D4; color: #1E5631"
        if val >= 0.5:
            return "background-color: #FFF2CC; color: #7D6608"
        if val < 0:
            return "background-color: #F8D7DA; color: #721C24"
        return "color: #555"

    styled = (disp_df.style
              .applymap(colour_ir, subset=["IR"])
              .format({"Ann α (%)": "{:.2f}", "TE (%)": "{:.2f}",
                       "IR": "{:.3f}", "Hit Rate %": "{:.1f}",
                       "Max DD %": "{:.2f}"}, na_rep="—")
              .set_table_styles([
                  {"selector": "th",
                   "props": [("font-size", "0.82rem"), ("font-weight", "600"),
                             ("color", "#2c3e50"), ("border-bottom", "2px solid #dde3ea")]},
                  {"selector": "td",
                   "props": [("font-size", "0.83rem"), ("padding", "5px 10px")]},
              ])
              .hide(axis="index"))

    st.dataframe(styled, use_container_width=True, height=480)

    st.download_button(
        "Download as CSV",
        data=disp_df.to_csv(index=False),
        file_name="regime_statistics.csv",
        mime="text/csv",
    )

    st.divider()

    # IR heatmap
    st.markdown("## IR Heatmap — Models × Regimes")
    st.caption("Colour scale: red = negative IR, yellow = 0.5, green = 1.0+.")
    pivot = disp_df.pivot_table(values="IR", index="Model",
                                columns="Regime", aggfunc="first")
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=[
            [0.0,  "#B71C1C"],
            [0.25, "#EF9A9A"],
            [0.45, "#FFFDE7"],
            [0.65, "#C8E6C9"],
            [1.0,  "#1B5E20"],
        ],
        color_continuous_midpoint=0.5,
        zmin=-0.5, zmax=2.0,
        aspect="auto",
        height=260,
        text_auto=".2f",
    )
    fig_hm.update_layout(
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font=dict(family="Arial, sans-serif", size=11),
        coloraxis_showscale=True,
        margin=dict(t=30, b=30, l=30, r=30),
        xaxis=dict(tickangle=-20, tickfont=dict(size=9)),
    )
    fig_hm.update_traces(textfont_size=10)
    st.plotly_chart(fig_hm, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Volatility Surface
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    # ── Section 1: Stochastic MC Portfolio Vol Surface ───────────────────────
    st.markdown("## Stochastic Portfolio Volatility Surface")
    st.caption(
        "Monte Carlo portfolio vol surface generated by 1,000 regime-switching paths per grid point — "
        "**not historical data**. "
        "X axis: forward horizon (months). Y axis: alpha tilt level (%). "
        "Z axis: expected annualised portfolio vol. "
        "Saddle points and ridges emerge from the interaction of alpha magnitude, "
        "horizon length, and regime transition probability: at medium horizons regime "
        "switches accumulate; at long horizons multiple regimes average out. "
        "Based on CS-Transformer results: IR 1.874, TE 2.22%, benchmark vol 16%."
    )

    with st.expander("MC parameters", expanded=False):
        col_mc1, col_mc2, col_mc3 = st.columns(3)
        with col_mc1:
            switch_prob = st.slider(
                "Regime switch prob (%/month)", 5, 25, 12, key="switch_prob"
            ) / 100.0
        with col_mc2:
            stress_mult = st.slider(
                "Stress vol multiplier", 1.2, 3.0, 2.0, step=0.1, key="stress_mult"
            )
        with col_mc3:
            n_mc_vol = st.select_slider(
                "MC paths", options=[500, 1000, 1500, 2000], value=1000, key="n_mc_vol"
            )

    # Fixed CS-T model parameters
    _BASE_IR    = 1.874
    _BASE_TE    = 0.0222   # annualised, at alpha=1%
    _BASE_ALPHA = 0.01
    _BENCH_VOL  = 0.16

    @st.cache_data(show_spinner=False)
    def compute_mc_vol_surface(base_ir, base_te, bench_vol, base_alpha,
                                n_alpha, max_horizon, n_sims,
                                sw_prob, st_mult, seed=42):
        """
        Build stochastic vol surface via Monte Carlo with Markov regime switching.

        For each (alpha, horizon) grid point, simulate n_sims paths of monthly
        portfolio returns. At each step the regime can switch (risk-on ↔ risk-off)
        with probability sw_prob. In risk-off regime, vol is multiplied by st_mult.
        Z[i, j] = std(cumulative return over j months) × sqrt(12 / j)  (annualised).
        """
        rng        = np.random.default_rng(seed)
        alpha_arr  = np.linspace(0.005, 0.05, n_alpha)
        h_arr      = np.arange(1, max_horizon + 1)
        bench_m    = bench_vol / np.sqrt(12)
        Z          = np.zeros((n_alpha, max_horizon))

        for i, alpha in enumerate(alpha_arr):
            te_ann = base_te * alpha / base_alpha
            te_m   = te_ann  / np.sqrt(12)
            mean_m = base_ir * te_ann / 12.0

            # Simulate Markov regime chain: shape (n_sims, max_horizon)
            r_state = np.zeros(n_sims)
            regimes = np.zeros((n_sims, max_horizon))
            for t in range(max_horizon):
                sw      = rng.random(n_sims) < sw_prob
                r_state = np.where(sw, 1.0 - r_state, r_state)
                regimes[:, t] = r_state

            vm      = np.where(regimes == 1, st_mult, 1.0)          # vol multiplier
            port_vm = np.sqrt((bench_m * vm) ** 2 + (te_m * vm) ** 2)  # (n, T)

            # Draw returns: normal with state-dependent vol
            rets = rng.normal(loc=mean_m, scale=port_vm)   # (n, T)

            for j, h in enumerate(h_arr):
                cum     = rets[:, :h].sum(axis=1)
                Z[i, j] = cum.std() * np.sqrt(12.0 / h)   # annualise

        return alpha_arr, h_arr, Z

    with st.spinner("Running Monte Carlo simulation ..."):
        alpha_arr, h_arr, Z_mc = compute_mc_vol_surface(
            _BASE_IR, _BASE_TE, _BENCH_VOL, _BASE_ALPHA,
            n_alpha=24, max_horizon=36,
            n_sims=n_mc_vol,
            sw_prob=switch_prob, st_mult=stress_mult,
        )

    alpha_pct = alpha_arr * 100   # convert to %

    fig_mc = go.Figure(data=[go.Surface(
        x=h_arr,
        y=alpha_pct,
        z=Z_mc * 100,   # convert to %
        colorscale=[
            [0.00, "#08306B"],
            [0.25, "#2171B5"],
            [0.50, "#6BAED6"],
            [0.70, "#FDAE6B"],
            [0.85, "#E6550D"],
            [1.00, "#7F2704"],
        ],
        opacity=0.95,
        showscale=True,
        colorbar=dict(
            title=dict(text="Ann. Vol (%)", side="right", font=dict(size=10)),
            thickness=14, len=0.65,
            tickfont=dict(size=9),
            tickformat=".1f",
        ),
        hovertemplate=(
            "Horizon: %{x} months<br>"
            "Alpha tilt: %{y:.1f}%<br>"
            "Portfolio vol: %{z:.1f}%<extra></extra>"
        ),
    )])

    fig_mc.update_layout(
        height=620,
        paper_bgcolor="#1A1D27",
        font=dict(family="Arial, sans-serif", size=10),
        scene=dict(
            xaxis=dict(
                title=dict(text="Forward Horizon (months)", font=dict(size=10)),
                tickfont=dict(size=8), gridcolor="#E0E0E0",
                backgroundcolor="#1A1D27",
            ),
            yaxis=dict(
                title=dict(text="Alpha Tilt (%)", font=dict(size=10)),
                tickfont=dict(size=8), gridcolor="#E0E0E0",
                backgroundcolor="#1A1D27",
            ),
            zaxis=dict(
                title=dict(text="Annualised Vol (%)", font=dict(size=10)),
                tickfont=dict(size=8), gridcolor="#E0E0E0",
                backgroundcolor="#1A1D27",
            ),
            camera=dict(eye=dict(x=1.6, y=-1.9, z=1.1)),
            aspectratio=dict(x=2.0, y=1.0, z=0.85),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        title=dict(
            text=(
                f"Stochastic Portfolio Vol Surface  —  "
                f"MC n={n_mc_vol:,} paths  |  "
                f"switch prob {switch_prob*100:.0f}%/mo  |  "
                f"stress mult ×{stress_mult:.1f}"
            ),
            font=dict(size=11), x=0.02,
        ),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── Backtest Replay Animation ─────────────────────────────────────────────
    st.divider()
    st.markdown("## Backtest Replay — Vol Surface Through Time")
    st.caption(
        "Precompute all 192 monthly frames once (~25s), then play them instantly. "
        "Watch the surface spike during COVID crash (2020) and rate-hike bear (2022), "
        "then flatten during QE bull (2013–2019). Fixed colour scale so the shape "
        "changes are visible — not the colours."
    )

    import time as _time

    REPLAY_MONTHS = pd.date_range("2010-01-01", "2025-12-01", freq="MS")

    def _regime_params(dt):
        yr, mo = dt.year, dt.month
        if yr == 2008 or yr == 2009:             return 0.22, 2.8   # GFC
        if yr == 2020 and mo <= 5:               return 0.25, 3.0   # COVID crash
        if yr == 2020 and mo >= 6:               return 0.10, 1.6   # COVID recovery
        if yr == 2022:                           return 0.20, 2.5   # rate-hike bear
        if 2013 <= yr <= 2019:                   return 0.05, 1.3   # QE bull
        return 0.12, 2.0                                            # default

    # ── Controls ─────────────────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
    with col_a:
        play_speed = st.select_slider(
            "Frame speed (sec)", options=[0.05, 0.1, 0.2, 0.4], value=0.1,
            key="play_speed"
        )
    with col_b:
        n_sims_replay = st.select_slider(
            "MC paths per frame", options=[200, 400, 800], value=400,
            key="n_sims_replay"
        )
    with col_c:
        precompute_btn = st.button("⚙  Precompute Frames", key="precompute_btn",
                                   use_container_width=True)
    with col_d:
        play_btn = st.button("▶  Play Animation", key="play_btn",
                             use_container_width=True,
                             disabled="replay_frames" not in st.session_state)

    # ── Precompute ────────────────────────────────────────────────────────────
    if precompute_btn:
        frames = []
        prog   = st.progress(0.0, text="Precomputing frames…")
        n      = len(REPLAY_MONTHS)
        for i, dt in enumerate(REPLAY_MONTHS):
            sw, sm = _regime_params(dt)
            a_arr, h_arr, Z = compute_mc_vol_surface(
                _BASE_IR, _BASE_TE, _BENCH_VOL, _BASE_ALPHA,
                n_alpha=16, max_horizon=24,
                n_sims=n_sims_replay,
                sw_prob=sw, st_mult=sm,
                seed=int(dt.timestamp()) % 10000,
            )
            frames.append({
                "dt": dt, "sw": sw, "sm": sm,
                "a":  a_arr, "h": h_arr, "Z": Z,
            })
            prog.progress((i + 1) / n,
                          text=f"Frame {i+1}/{n}  —  {dt.strftime('%b %Y')}")
        st.session_state["replay_frames"] = frames
        prog.empty()
        st.success(f"Done — {n} frames precomputed. Press ▶ Play Animation.")

    # ── Play ──────────────────────────────────────────────────────────────────
    if play_btn and "replay_frames" in st.session_state:
        chart_slot  = st.empty()
        status_slot = st.empty()

        for f in st.session_state["replay_frames"]:
            dt, sw, sm = f["dt"], f["sw"], f["sm"]
            fig_r = go.Figure(data=[go.Surface(
                x=f["h"], y=f["a"] * 100, z=f["Z"] * 100,
                colorscale=[
                    [0.00, "#08306B"], [0.30, "#2171B5"],
                    [0.55, "#6BAED6"], [0.75, "#FDAE6B"],
                    [0.90, "#E6550D"], [1.00, "#7F2704"],
                ],
                cmin=14, cmax=36,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Vol (%)", side="right",
                               font=dict(size=9, color="#E8EDF2")),
                    thickness=12, len=0.6, tickformat=".0f",
                    tickfont=dict(color="#E8EDF2"),
                ),
            )])
            fig_r.update_layout(
                height=540,
                paper_bgcolor="#1A1D27",
                font=dict(family="Arial, sans-serif", size=9, color="#E8EDF2"),
                scene=dict(
                    xaxis=dict(title="Horizon (months)",
                               backgroundcolor="#1A1D27", gridcolor="#2A2D3A",
                               tickfont=dict(size=8, color="#B0B8C4")),
                    yaxis=dict(title="Alpha (%)",
                               backgroundcolor="#1A1D27", gridcolor="#2A2D3A",
                               tickfont=dict(size=8, color="#B0B8C4")),
                    zaxis=dict(title="Vol (%)", range=[14, 36],
                               backgroundcolor="#1A1D27", gridcolor="#2A2D3A",
                               tickfont=dict(size=8, color="#B0B8C4")),
                    camera=dict(eye=dict(x=1.6, y=-1.9, z=1.1)),
                    aspectratio=dict(x=2.0, y=1.0, z=0.85),
                ),
                margin=dict(l=0, r=0, t=44, b=0),
                title=dict(
                    text=(f"{dt.strftime('%b %Y')}  ·  "
                          f"switch {sw*100:.0f}%/mo  ·  stress ×{sm:.1f}"),
                    font=dict(size=12, color="#E8EDF2"), x=0.02,
                ),
            )
            chart_slot.plotly_chart(fig_r, use_container_width=True)
            status_slot.caption(
                f"{dt.strftime('%Y-%m')}  |  switch prob {sw*100:.0f}%/mo  |  "
                f"stress ×{sm:.1f}  |  "
                + ("🔴 CRASH" if sm >= 2.8 else "🟠 BEAR" if sm >= 2.2
                   else "🟡 RECOVERY" if sm >= 1.5 else "🟢 BULL")
            )
            _time.sleep(play_speed)

        status_slot.caption("▶ Replay complete — press Play to watch again.")

    st.divider()

    # ── Section 2: Historical Cross-Sectional Return Distribution ────────────
    st.markdown("## Historical Cross-Sectional Return Distribution")
    st.caption(
        "Historical view: 3-D surface of the cross-sectional distribution of S&P 500 "
        "monthly stock returns over time. X axis: month. Y axis: cross-sectional "
        "percentile. Z axis: actual monthly return. COVID crash (Mar 2020) shows a "
        "sharp spike downward; AI bull (2023–2025) shows persistent upward skew in "
        "the top decile."
    )

    PANEL_PATH = DATA_DIR / "panel_monthly_enriched.parquet"

    if not PANEL_PATH.exists():
        st.warning("panel_monthly_enriched.parquet not found. "
                   "Run 1g_feature_engineering.py first.")
    else:
        year_range = st.slider(
            "Year range:", 2010, 2025, (2010, 2025), key="yr_range",
        )

        @st.cache_data(show_spinner=False)
        def load_hist_surface(yr_start: int, yr_end: int):
            cols  = ["date", "ticker", "fwd_ret_1m"]
            panel = pd.read_parquet(PANEL_PATH, columns=cols)
            panel = panel[
                (panel["date"].dt.year >= yr_start) &
                (panel["date"].dt.year <= yr_end)
            ].dropna(subset=["fwd_ret_1m"])
            months = sorted(panel["date"].unique())
            pcts   = np.arange(0, 101, 5)
            Z      = np.full((len(pcts), len(months)), np.nan)
            for j, dt in enumerate(months):
                vals = panel.loc[panel["date"] == dt, "fwd_ret_1m"].values
                if len(vals) >= 20:
                    lo, hi = np.percentile(vals, [1, 99])
                    Z[:, j] = np.percentile(np.clip(vals, lo, hi), pcts)
            m_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in months]
            return Z, pcts, m_labels, months

        with st.spinner("Loading historical return surface ..."):
            Z_hist, pcts_h, m_labels_h, months_h = load_hist_surface(
                year_range[0], year_range[1]
            )

        m_idx_h     = np.arange(len(months_h))
        tick_idx_h  = m_idx_h[::12]
        tick_lbl_h  = [m_labels_h[i] for i in tick_idx_h]

        fig_hist = go.Figure(data=[go.Surface(
            x=m_idx_h,
            y=pcts_h,
            z=Z_hist,
            colorscale=[
                [0.0, "#8B0000"], [0.35, "#E87070"],
                [0.5,  "#F5F5F5"],
                [0.65, "#6BAED6"], [1.0, "#08306B"],
            ],
            cmid=0.0,
            opacity=0.95,
            showscale=True,
            colorbar=dict(
                title=dict(text="fwd_ret_1m", side="right", font=dict(size=10)),
                thickness=14, len=0.65,
                tickfont=dict(size=9), tickformat=".1%",
            ),
            hovertemplate=(
                "Month: %{x}<br>Percentile: %{y}<br>"
                "Return: %{z:.3f}<extra></extra>"
            ),
        )])
        fig_hist.update_layout(
            height=560,
            paper_bgcolor="#1A1D27",
            font=dict(family="Arial, sans-serif", size=10),
            scene=dict(
                xaxis=dict(
                    title="Month", tickmode="array",
                    tickvals=tick_idx_h, ticktext=tick_lbl_h,
                    tickfont=dict(size=8), gridcolor="#E0E0E0",
                    backgroundcolor="#1A1D27",
                ),
                yaxis=dict(
                    title="Cross-Sectional Percentile",
                    tickfont=dict(size=8), gridcolor="#E0E0E0",
                    backgroundcolor="#1A1D27",
                ),
                zaxis=dict(
                    title="Monthly Return",
                    tickfont=dict(size=8), gridcolor="#E0E0E0",
                    backgroundcolor="#1A1D27",
                ),
                camera=dict(eye=dict(x=1.4, y=-1.8, z=1.0)),
                aspectratio=dict(x=2.2, y=1.0, z=0.9),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f"Historical fwd_ret_1m Surface  ({year_range[0]}–{year_range[1]})",
                font=dict(size=12), x=0.02,
            ),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Percentile time series
        st.markdown("### Percentile Time Series")
        pct_options  = [int(p) for p in pcts_h]
        pct_defaults = [p for p in [10, 25, 50, 75, 90] if p in pct_options]
        show_pct = st.multiselect(
            "Percentiles to display:",
            options=pct_options,
            default=pct_defaults,
            key="show_pcts",
        )
        fig_ts = go.Figure()
        pct_colors = px.colors.sample_colorscale("Viridis", max(len(show_pct), 1))
        for pct_val, col in zip(show_pct, pct_colors):
            idx = int(np.argmin(np.abs(pcts_h - pct_val)))
            fig_ts.add_trace(go.Scatter(
                x=[pd.Timestamp(m) for m in months_h],
                y=Z_hist[idx, :],
                mode="lines",
                name=f"P{pct_val}",
                line=dict(color=col, width=1.5),
            ))
        for reg in ["COVID Crash (Q1 2020)", "Rate Hike Bear (2022)"]:
            if reg in ALL_REGIMES:
                s, e, c = ALL_REGIMES[reg]
                fig_ts.add_vrect(
                    x0=s, x1=e, fillcolor=c, opacity=0.08,
                    annotation_text=reg.split(" (")[0][:14],
                    annotation_font_size=8, annotation_position="top left",
                )
        fig_ts.add_hline(y=0, line_color="#CCCCCC", line_width=0.8)
        apply_layout(fig_ts, height=320,
                     xaxis_title="Date", yaxis_title="fwd_ret_1m",
                     title=dict(text="fwd_ret_1m — Selected Percentile Bands",
                                font=dict(size=11), x=0))
        st.plotly_chart(fig_ts, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — Walk-Forward RL Backtest
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Walk-Forward RL Backtest — No Data Leakage")
    st.markdown(
        "The L2 SAC agent is re-trained from scratch on each fold using only "
        "past data. Factor IC weights, state normalisation, and the RL policy "
        "are all computed on the training window only — nothing from the test "
        "period leaks in. Results below are fully out-of-sample across 95 months."
    )

    WF_FILE   = DATA_DIR / "bt_wf_rl.csv"
    FOLD_FILE = DATA_DIR / "wf_fold_summary.csv"

    if not WF_FILE.exists() or not FOLD_FILE.exists():
        st.warning(
            "Walk-forward data not found. Run `python 5c_walk_forward.py` first."
        )
    else:
        wf   = pd.read_csv(WF_FILE,   parse_dates=["date"]).sort_values("date")
        fold = pd.read_csv(FOLD_FILE)

        # ── Headline metrics ─────────────────────────────────────────────────
        rl_alpha_ann = wf["active_ret"].mean() * 12
        rl_te        = wf["active_ret"].std() * (12 ** 0.5)
        rl_ir        = rl_alpha_ann / rl_te if rl_te > 0 else 0.0

        # Fixed alpha benchmark (alpha_used constant at 1%)
        fixed_alpha = 0.01
        # approximate fixed active return using same bench returns and fixed alpha
        fixed_active = wf["active_ret"] * (fixed_alpha / wf["alpha_used"].replace(0, np.nan))
        fixed_active = fixed_active.dropna()
        fixed_alpha_ann = fixed_active.mean() * 12
        fixed_te        = fixed_active.std() * (12 ** 0.5)
        fixed_ir        = fixed_alpha_ann / fixed_te if fixed_te > 0 else 0.0

        n_wins = int((fold["rl_ir"] > fold["fixed_ir"]).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RL Info Ratio",     f"{rl_ir:.3f}",
                  f"vs Fixed {fixed_ir:.3f}")
        c2.metric("RL Ann Alpha",      f"{rl_alpha_ann*100:.2f}%",
                  f"vs Fixed {fixed_alpha_ann*100:.2f}%")
        c3.metric("RL Tracking Error", f"{rl_te*100:.2f}%")
        c4.metric("Folds RL Wins",     f"{n_wins}/5",
                  "out-of-sample folds")

        st.markdown("---")

        # ── Cumulative active return chart ───────────────────────────────────
        st.markdown("### Cumulative Out-of-Sample Active Return")

        wf = wf.sort_values("date").copy()
        wf["cum_rl"]    = (1 + wf["active_ret"]).cumprod() - 1
        # rebuild fixed active return series aligned to wf index
        wf["fixed_ar"]  = wf["active_ret"] * (fixed_alpha / wf["alpha_used"].replace(0, np.nan))
        wf["cum_fixed"] = (1 + wf["fixed_ar"].fillna(0)).cumprod() - 1

        fig_cum = go.Figure()

        # Shade fold boundaries
        fold_starts = wf.groupby("fold")["date"].min().reset_index()
        fold_colors = ["#1A2A3A", "#1A3A2A", "#2A1A3A", "#3A2A1A", "#1A3A3A"]
        for i, row in fold_starts.iterrows():
            end_dt = wf[wf["fold"] == row["fold"]]["date"].max()
            fig_cum.add_vrect(
                x0=row["date"], x1=end_dt,
                fillcolor=fold_colors[i % len(fold_colors)],
                opacity=0.25, layer="below", line_width=0,
                annotation_text=row["fold"],
                annotation_position="top left",
                annotation_font=dict(size=8, color="#9AAABB"),
            )

        fig_cum.add_trace(go.Scatter(
            x=wf["date"], y=wf["cum_rl"] * 100,
            mode="lines", name="RL Agent",
            line=dict(color="#4A9EE0", width=2.2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=wf["date"], y=wf["cum_fixed"] * 100,
            mode="lines", name="Fixed α=1%",
            line=dict(color="#E07A4A", width=1.8, dash="dot"),
        ))
        fig_cum.add_hline(y=0, line_color="#555", line_width=0.8)

        _l = dict(**_LAYOUT)
        fig_cum.update_layout(
            **_l,
            height=380,
            yaxis_title="Cumulative Active Return (%)",
            xaxis_title="Date",
            title=dict(text="Walk-Forward Cumulative Alpha (95 out-of-sample months)",
                       font=dict(size=12), x=0),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Per-fold IR bar chart ─────────────────────────────────────────────
        st.markdown("### Per-Fold Information Ratio")

        bar_colors_rl    = ["#4A9EE0"] * len(fold)
        bar_colors_fixed = ["#E07A4A"] * len(fold)

        fig_fold = go.Figure()
        fig_fold.add_trace(go.Bar(
            name="RL Agent",
            x=fold["label"],
            y=fold["rl_ir"],
            marker_color=bar_colors_rl,
            text=[f"{v:.2f}" for v in fold["rl_ir"]],
            textposition="outside",
            textfont=dict(size=9, color="#E8EDF2"),
        ))
        fig_fold.add_trace(go.Bar(
            name="Fixed α=1%",
            x=fold["label"],
            y=fold["fixed_ir"],
            marker_color=bar_colors_fixed,
            text=[f"{v:.2f}" for v in fold["fixed_ir"]],
            textposition="outside",
            textfont=dict(size=9, color="#E8EDF2"),
        ))
        fig_fold.add_hline(y=0, line_color="#888", line_width=0.8)

        _l2 = dict(**_LAYOUT)
        fig_fold.update_layout(
            **_l2,
            height=360,
            barmode="group",
            bargap=0.25,
            yaxis_title="Information Ratio",
            title=dict(text="IR by Fold — RL Agent vs Fixed Alpha",
                       font=dict(size=12), x=0),
        )
        st.plotly_chart(fig_fold, use_container_width=True)

        # ── Fold summary table ────────────────────────────────────────────────
        st.markdown("### Fold Summary")
        disp = fold.copy()
        disp.columns = ["Fold", "Months", "RL IR", "RL Alpha", "RL TE", "Fixed IR"]
        disp["RL Alpha"] = (disp["RL Alpha"] * 100).map("{:.1f}%".format)
        disp["RL TE"]    = (disp["RL TE"]    * 100).map("{:.1f}%".format)
        disp["RL IR"]    = disp["RL IR"].map("{:.3f}".format)
        disp["Fixed IR"] = disp["Fixed IR"].map("{:.3f}".format)
        disp["Beats?"]   = (fold["rl_ir"] > fold["fixed_ir"]).map({True: "✓", False: "✗"})
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown(
            "_Factor IC weights, state normalisation, and RL policy are all "
            "computed on training data only. No test-period information is used "
            "in any component of the pipeline._"
        )


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ML-Driven S&P 500 Index Enhancement  ·  "
    "CS-Transformer: IR 1.874, Ann α 4.16%, TE 2.22%  ·  "
    "Walk-Forward RL: IR 0.879, Ann α 7.19%, 4/5 folds  ·  "
    "Run: streamlit run 6_regime_dashboard.py"
)
