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
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial, sans-serif", size=11, color="#2c2c2c"),
    xaxis=dict(showgrid=True, gridcolor="#F2F2F2", linecolor="#CCCCCC",
               linewidth=1, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#F2F2F2", linecolor="#CCCCCC",
               linewidth=1, zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin=dict(t=60, b=40, l=50, r=30),
)

# ── Custom CSS — academic, clean ────────────────────────────────────────────
_CSS = """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

/* Headings */
h1 { font-size: 1.35rem !important; font-weight: 700; color: #12243a;
     letter-spacing: -0.01em; border-bottom: 2px solid #1565C0;
     padding-bottom: 0.35rem; margin-bottom: 0.8rem; }
h2 { font-size: 1.0rem !important; font-weight: 600; color: #1a2a3a;
     border-bottom: 1px solid #e4e8ee; padding-bottom: 0.25rem; margin-top: 1rem; }
h3 { font-size: 0.9rem !important; font-weight: 600; color: #2c3e50; }
p  { font-size: 0.86rem; color: #555; line-height: 1.55; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f9fafb;
    border: 1px solid #dde3ea;
    border-radius: 5px;
    padding: 0.7rem 0.9rem;
}
[data-testid="metric-container"] label { font-size: 0.78rem !important;
    color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.2rem !important; font-weight: 600; color: #12243a; }

/* Tabs */
[data-baseweb="tab-list"] { border-bottom: 2px solid #dde3ea; gap: 0; }
[data-baseweb="tab"] { font-size: 0.83rem !important; font-weight: 500;
    color: #5a6a7a; padding: 0.5rem 1rem; }
[data-baseweb="tab"][aria-selected="true"] { color: #1565C0 !important;
    border-bottom: 2px solid #1565C0; font-weight: 600; }

/* Sidebar */
[data-testid="stSidebar"] { background: #f5f7fa; border-right: 1px solid #dde3ea; }
[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }
section[data-testid="stSidebar"] h2 { font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 0.06em; color: #8a96a3;
    border-bottom: 1px solid #dde3ea; padding-bottom: 0.2rem; }

/* Captions / subtext */
.stCaption, [data-testid="stCaptionContainer"] { color: #8a96a3 !important;
    font-size: 0.8rem; font-style: normal; }

/* Info / warning boxes */
.stAlert { border-radius: 4px; font-size: 0.84rem; }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: white; border: 1px solid #c5ced6; color: #2c3e50;
    font-size: 0.83rem; font-weight: 500; padding: 0.35rem 0.9rem;
    border-radius: 4px; }
[data-testid="stDownloadButton"] > button:hover { background: #f0f4f8; }

/* Dividers */
hr { border: none; border-top: 1px solid #e4e8ee; margin: 1rem 0; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #dde3ea; border-radius: 4px; }

/* Spinner */
.stSpinner { color: #1565C0; }
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Regime Breakdown",
    "Bootstrap Analysis",
    "Monte Carlo Projection",
    "Statistics",
    "Volatility Surface",
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
        plot_bgcolor="white", paper_bgcolor="white",
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
    st.markdown("## Cross-Sectional Return Distribution Surface")
    st.caption(
        "3-D surface of the cross-sectional distribution of S&P 500 stock returns over time. "
        "X axis: month. Y axis: cross-sectional percentile (0 = worst stock, 100 = best stock). "
        "Z axis: actual monthly return at that percentile. "
        "The surface reveals regime structure: COVID crash (Mar 2020) shows a sharp spike downward "
        "across all percentiles; AI bull (2023–2025) shows persistent upward skew in the top decile. "
        "Factor scores (vol_252d etc.) are cross-sectionally rank-normalised to [−0.5, +0.5] "
        "and produce a flat surface by construction — fwd_ret_1m is the meaningful default."
    )

    PANEL_PATH = DATA_DIR / "panel_monthly_enriched.parquet"

    if not PANEL_PATH.exists():
        st.warning("panel_monthly_enriched.parquet not found. "
                   "Run 1g_feature_engineering.py first.")
    else:
        # Controls
        # fwd_ret_1m is raw (not normalised) → meaningful 3-D surface.
        # Factor scores are rank-normalised to [−0.5, +0.5] → always a nearly flat
        # linear ramp. For those, the 2-D percentile time-series chart below is the
        # useful view — it shows HOW each percentile band evolved over time.
        RAW_FACTORS    = {"fwd_ret_1m"}   # not rank-normalised
        RANKED_FACTORS = {"vol_252d", "idio_vol_252d", "vol_126d", "ret_12m", "beta_252d"}

        col_feat, col_pct, col_yr = st.columns([2, 2, 2])
        with col_feat:
            vol_factor = st.selectbox(
                "Factor:",
                ["fwd_ret_1m", "vol_252d", "idio_vol_252d", "vol_126d", "ret_12m", "beta_252d"],
                index=0,
                key="vol_factor",
                help="fwd_ret_1m shows actual monthly stock returns — the 3-D surface is most "
                     "meaningful here (regime spikes visible). Factor scores (vol_252d etc.) are "
                     "rank-normalised to [−0.5, +0.5]; their surface is flat by construction — "
                     "use the Percentile Time Series chart below for those.",
            )
        is_ranked = vol_factor in RANKED_FACTORS
        if is_ranked:
            st.info(
                f"**{vol_factor}** is cross-sectionally rank-normalised to [−0.5, +0.5]. "
                "The 3-D surface will appear nearly flat — this is expected. "
                "The **Percentile Time Series** chart below is the useful view for factor scores: "
                "it shows how the top/bottom percentile bands evolved over time."
            )
        with col_pct:
            pct_step = st.select_slider(
                "Percentile resolution:",
                options=[5, 10, 20], value=5, key="pct_step",
            )
        with col_yr:
            year_range = st.slider(
                "Year range:", 2010, 2025, (2010, 2025), key="yr_range",
            )

        @st.cache_data(show_spinner=False)
        def load_vol_surface(factor: str, pct_step: int,
                              yr_start: int, yr_end: int):
            cols   = ["date", "ticker", factor]
            panel  = pd.read_parquet(PANEL_PATH, columns=cols)
            panel  = panel[
                (panel["date"].dt.year >= yr_start) &
                (panel["date"].dt.year <= yr_end)
            ].dropna(subset=[factor])
            months  = sorted(panel["date"].unique())
            pcts    = np.arange(0, 101, pct_step)
            Z = np.full((len(pcts), len(months)), np.nan)
            for j, dt in enumerate(months):
                vals = panel.loc[panel["date"] == dt, factor].values
                if len(vals) >= 20:
                    # Winsorise at 1st/99th percentile to suppress extreme outliers
                    lo, hi = np.percentile(vals, [1, 99])
                    vals_w  = np.clip(vals, lo, hi)
                    Z[:, j] = np.percentile(vals_w, pcts)
            m_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in months]
            return Z, pcts, m_labels, months

        with st.spinner("Building volatility surface ..."):
            Z, pcts, m_labels, months = load_vol_surface(
                vol_factor, pct_step, year_range[0], year_range[1]
            )

        m_idx = np.arange(len(months))

        # Tick marks every 12 months
        tick_idx    = m_idx[::12]
        tick_labels = [m_labels[i] for i in tick_idx]

        # Symmetric diverging colorscale for returns (red=negative, blue=positive)
        # Viridis for non-return factors
        is_return = vol_factor in ("fwd_ret_1m", "ret_1m", "ret_3m", "ret_6m", "ret_12m")
        colorscale = (
            [[0.0, "#8B0000"], [0.35, "#E87070"],
             [0.5,  "#F5F5F5"],
             [0.65, "#6BAED6"], [1.0, "#08306B"]]
            if is_return else "Viridis"
        )
        fig_surf = go.Figure(data=[go.Surface(
            x=m_idx,
            y=pcts,
            z=Z,
            colorscale=colorscale,
            cmid=0.0 if is_return else None,
            opacity=0.95,
            showscale=True,
            colorbar=dict(
                title=dict(text=vol_factor, side="right", font=dict(size=10)),
                thickness=14, len=0.65,
                tickfont=dict(size=9),
                tickformat=".1%" if is_return else ".3f",
            ),
            hovertemplate=(
                "Month: %{x}<br>"
                "Percentile: %{y}<br>"
                f"{vol_factor}: " + "%{z:.3f}<extra></extra>"
            ),
        )])

        fig_surf.update_layout(
            height=600,
            paper_bgcolor="white",
            font=dict(family="Arial, sans-serif", size=10),
            scene=dict(
                xaxis=dict(
                    title=dict(text="Month", font=dict(size=10)),
                    tickmode="array",
                    tickvals=tick_idx,
                    ticktext=tick_labels,
                    tickfont=dict(size=8),
                    gridcolor="#E0E0E0",
                    backgroundcolor="rgb(248,249,250)",
                ),
                yaxis=dict(
                    title=dict(text="Cross-Sectional Percentile", font=dict(size=10)),
                    tickfont=dict(size=8),
                    gridcolor="#E0E0E0",
                    backgroundcolor="rgb(248,249,250)",
                ),
                zaxis=dict(
                    title=dict(text=vol_factor, font=dict(size=10)),
                    tickfont=dict(size=8),
                    gridcolor="#E0E0E0",
                    backgroundcolor="rgb(248,249,250)",
                ),
                camera=dict(eye=dict(x=1.4, y=-1.8, z=1.0)),
                aspectratio=dict(x=2.2, y=1.0, z=0.9),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=(f"Cross-Sectional {vol_factor} Surface  "
                      f"({year_range[0]}–{year_range[1]})"),
                font=dict(size=12), x=0.02,
            ),
        )
        st.plotly_chart(fig_surf, use_container_width=True)

        st.divider()

        # Companion chart: time series of selected percentiles
        st.markdown("## Percentile Time Series")
        st.caption("Select which percentile bands to overlay on the 2-D chart below.")

        pct_options  = [int(p) for p in pcts]
        pct_defaults = [p for p in [10, 25, 50, 75, 90] if p in pct_options]
        if not pct_defaults:
            pct_defaults = [pct_options[len(pct_options) // 2]]   # fallback: midpoint
        show_pct = st.multiselect(
            "Percentiles to display:",
            options=pct_options,
            default=pct_defaults,
            key="show_pcts",
        )

        fig_ts = go.Figure()
        pct_colors = px.colors.sample_colorscale("Viridis", len(show_pct))
        for pct_val, col in zip(show_pct, pct_colors):
            idx = int(np.argmin(np.abs(pcts - pct_val)))
            z_row = Z[idx, :]
            fig_ts.add_trace(go.Scatter(
                x=[pd.Timestamp(m) for m in months],
                y=z_row,
                mode="lines",
                name=f"P{pct_val}",
                line=dict(color=col, width=1.5),
            ))

        # Regime shading on time-series chart
        for reg in ["COVID Crash (Q1 2020)", "Rate Hike Bear (2022)"]:
            if reg in ALL_REGIMES:
                s, e, c = ALL_REGIMES[reg]
                short = reg.split(" (")[0]
                fig_ts.add_vrect(
                    x0=s, x1=e, fillcolor=c, opacity=0.08,
                    annotation_text=short[:14], annotation_font_size=8,
                    annotation_position="top left",
                )

        fig_ts.add_hline(y=0, line_color="#CCCCCC", line_width=0.8)
        apply_layout(fig_ts, height=320,
                     xaxis_title="Date",
                     yaxis_title=vol_factor,
                     title=dict(text=f"{vol_factor} — Selected Percentile Bands",
                                font=dict(size=11), x=0))
        st.plotly_chart(fig_ts, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ML-Driven S&P 500 Index Enhancement  ·  "
    "CS-Transformer: IR 1.874, Ann α 4.16%, TE 2.22%  ·  "
    "Out-of-sample test period: Jan 2023 – Nov 2025  ·  "
    "Run: streamlit run 6_regime_dashboard.py"
)
