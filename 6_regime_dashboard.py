"""
6_regime_dashboard.py — Interactive Regime Backtesting Dashboard
=================================================================
Streamlit app with:
  - Regime selector (Post-GFC, QE Bull, COVID Crash, COVID Recovery,
    Rate Hike Bear, AI Bull 2023+)
  - Model selector (CS-Transformer, FT-Transformer, LGBM, Factor-Combo)
  - Stochastic bootstrap engine (block bootstrap, 1000 iterations)
  - Monte Carlo forward projection
  - Full stats table with 95% confidence intervals

HOW TO RUN:
  streamlit run 6_regime_dashboard.py
  Opens at: http://localhost:8501

HOW TO STOP:
  Press Ctrl+C in the terminal where it is running.

HOW STREAMLIT WORKS:
  Runs a local web server on port 8501. Every time you interact with a widget
  (slider, dropdown, checkbox) the entire script reruns from top to bottom
  with the updated values. No need to refresh the browser manually.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

# All available regimes (extendable to 2008 if price data available)
ALL_REGIMES = {
    "🔴 GFC Crash (2008–2009)":          ("2008-01-01", "2009-12-31", "#D32F2F"),
    "🟡 Post-GFC Recovery (2010–2012)":  ("2010-01-01", "2012-12-31", "#FFA726"),
    "🟢 QE Bull (2013–2019)":            ("2013-01-01", "2019-12-31", "#388E3C"),
    "⚫ COVID Crash (2020 Q1)":           ("2020-01-01", "2020-05-31", "#212121"),
    "🔵 COVID Recovery (2020–2021)":      ("2020-06-01", "2021-12-31", "#1565C0"),
    "🟠 Rate Hike Bear (2022)":           ("2022-01-01", "2022-12-31", "#E65100"),
    "🟣 AI Bull (2023–present)":          ("2023-01-01", "2099-12-31", "#6A1B9A"),
}

MODEL_FILES = {
    "CS-Transformer":  "bt_ie_cs_transformer.csv",
    "FT-Transformer":  "bt_ie_transformer.csv",
    "LGBM":            "bt_ie_lgbm.csv",
    "Factor-Combo":    "bt_ie_factor_combo.csv",
}

MODEL_COLORS = {
    "CS-Transformer": "#2196F3",
    "FT-Transformer": "#FF9800",
    "LGBM":           "#4CAF50",
    "Factor-Combo":   "#9E9E9E",
}

# ── Helper functions ────────────────────────────────────────────────────────────

def load_backtest(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / MODEL_FILES[name]
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def tag_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = "Unclassified"
    for name, (start, end, _) in ALL_REGIMES.items():
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        df.loc[mask, "regime"] = name
    return df


def regime_stats(returns: pd.Series) -> dict:
    """Compute IR, alpha, TE, hit rate, max active DD from a monthly active return series."""
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
    roll_max  = cum.cummax()
    max_dd    = ((cum - roll_max) / roll_max).min() * 100
    return dict(n=n, ann_alpha=round(ann_alpha * 100, 2),
                track_err=round(track_err * 100, 2),
                ir=round(ir, 3), hit_rate=round(hit_rate, 1),
                max_dd=round(max_dd, 2))


def block_bootstrap(returns: pd.Series, n_boot=1000, block_size=3) -> np.ndarray:
    """
    Block bootstrap to estimate IR distribution.
    Resamples consecutive blocks of months to preserve autocorrelation.
    Returns array of bootstrapped IR values.
    """
    r = returns.dropna().values
    n = len(r)
    if n < 6:
        return np.full(n_boot, np.nan)

    irs = []
    for _ in range(n_boot):
        # Draw random starting points for blocks
        n_blocks  = max(1, n // block_size)
        starts    = np.random.randint(0, max(1, n - block_size + 1), size=n_blocks)
        resampled = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        ann_a     = resampled.mean() * 12
        te        = resampled.std() * np.sqrt(12) + 1e-8
        irs.append(ann_a / te)

    return np.array(irs)


def monte_carlo_projection(ir: float, te: float, n_months: int = 36,
                           n_sims: int = 1000) -> pd.DataFrame:
    """
    Monte Carlo forward projection of cumulative active return.
    Assumes monthly active returns ~ N(ir*te/12, te/sqrt(12)).
    """
    monthly_alpha = ir * te / 12
    monthly_vol   = te / np.sqrt(12)
    paths = np.random.normal(monthly_alpha, monthly_vol, (n_sims, n_months))
    cum_paths = np.cumprod(1 + paths, axis=1) - 1   # cumulative active return
    months = np.arange(1, n_months + 1)
    return pd.DataFrame({
        "month":   months,
        "p5":      np.percentile(cum_paths, 5,  axis=0) * 100,
        "p25":     np.percentile(cum_paths, 25, axis=0) * 100,
        "p50":     np.percentile(cum_paths, 50, axis=0) * 100,
        "p75":     np.percentile(cum_paths, 75, axis=0) * 100,
        "p95":     np.percentile(cum_paths, 95, axis=0) * 100,
    })


# ── Streamlit App ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Regime Backtesting Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ML Index Enhancement — Regime Backtesting Dashboard")
st.caption("Interactive analysis of model performance across market regimes with stochastic confidence intervals.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Models")
    selected_models = st.multiselect(
        "Select models to compare:",
        options=list(MODEL_FILES.keys()),
        default=["CS-Transformer", "FT-Transformer", "LGBM"],
    )

    st.subheader("Regimes")
    selected_regimes = st.multiselect(
        "Select regimes:",
        options=list(ALL_REGIMES.keys()),
        default=[k for k in ALL_REGIMES.keys() if "GFC Crash" not in k],
    )

    st.subheader("Stochastic Engine")
    n_bootstrap = st.slider("Bootstrap iterations", 100, 2000, 1000, step=100)
    block_size  = st.slider("Block size (months)", 1, 6, 3)
    n_mc_months = st.slider("Monte Carlo horizon (months)", 12, 60, 36)
    n_mc_sims   = st.slider("Monte Carlo simulations", 100, 2000, 1000, step=100)

    st.subheader("Display")
    show_ci     = st.checkbox("Show 95% CI on IR chart", value=True)
    show_gfc_note = "🔴 GFC Crash (2008–2009)" in selected_regimes

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    data = {}
    for name in MODEL_FILES:
        df = load_backtest(name)
        if df is not None:
            data[name] = tag_regime(df)
    return data

all_data = load_all_data()

if not all_data:
    st.error("No backtest CSV files found in data/. Run 4b_index_enhancement.py first.")
    st.stop()

available_models = [m for m in selected_models if m in all_data]
if not available_models:
    st.warning("No selected models have backtest data. Run 4b_index_enhancement.py first.")
    st.stop()

# GFC note
if show_gfc_note:
    st.info("⚠️ GFC (2008–2009): ML model scores not available for this period (models trained on 2010+). "
            "Factor-Combo baseline can be computed if prices data extends to 2007.")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Regime Breakdown",
    "🎲 Bootstrap Analysis",
    "🔮 Monte Carlo Projection",
    "📋 Full Stats Table",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Regime Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Information Ratio by Regime")
    st.caption("Higher IR = more consistent alpha per unit of active risk taken.")

    # Build stats table
    rows = []
    for model in available_models:
        df = all_data[model]
        # Full period
        s = regime_stats(df["active_ret"])
        rows.append({"Model": model, "Regime": "FULL PERIOD", **s})
        # Per regime
        for reg in selected_regimes:
            sub = df[df["regime"] == reg]["active_ret"]
            s   = regime_stats(sub)
            rows.append({"Model": model, "Regime": reg, **s})

    stats_df = pd.DataFrame(rows)

    # IR bar chart
    fig_ir = go.Figure()
    for model in available_models:
        sub = stats_df[stats_df["Model"] == model]
        sub = sub[sub["Regime"].isin(["FULL PERIOD"] + selected_regimes)]
        fig_ir.add_trace(go.Bar(
            name=model,
            x=sub["Regime"].str[:30],
            y=sub["ir"],
            marker_color=MODEL_COLORS.get(model, "#607D8B"),
            opacity=0.85,
        ))

    fig_ir.add_hline(y=0.5, line_dash="dash", line_color="green",
                     annotation_text="IR=0.5 (good)", annotation_position="right")
    fig_ir.add_hline(y=1.0, line_dash="dash", line_color="gold",
                     annotation_text="IR=1.0 (excellent)", annotation_position="right")
    fig_ir.add_hline(y=0,   line_dash="solid", line_color="gray", line_width=1)

    fig_ir.update_layout(
        barmode="group", height=420,
        xaxis_title="Regime", yaxis_title="Information Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_ir, width="stretch")

    # Cumulative active return chart
    st.subheader("Cumulative Active Return")
    fig_cum = go.Figure()
    for model in available_models:
        df = all_data[model]
        df_sorted = df.sort_values("date")
        cum = (1 + df_sorted["active_ret"].fillna(0)).cumprod() - 1
        fig_cum.add_trace(go.Scatter(
            x=df_sorted["date"], y=cum * 100,
            name=model, mode="lines",
            line=dict(color=MODEL_COLORS.get(model, "#607D8B"), width=2),
        ))

    # Add regime shading
    for reg in selected_regimes:
        start_str, end_str, color = ALL_REGIMES[reg]
        short_name = reg[2:].split(" (")[0]
        fig_cum.add_vrect(
            x0=start_str, x1=end_str,
            fillcolor=color, opacity=0.07,
            annotation_text=short_name[:12],
            annotation_position="top left",
            annotation_font_size=9,
        )

    fig_cum.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    fig_cum.update_layout(
        height=400,
        xaxis_title="Date", yaxis_title="Cumulative Active Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_cum, width="stretch")

    # Hit rate & Ann Alpha side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hit Rate by Regime (%)")
        fig_hr = go.Figure()
        for model in available_models:
            sub = stats_df[(stats_df["Model"] == model) &
                           (stats_df["Regime"].isin(selected_regimes))]
            fig_hr.add_trace(go.Bar(
                name=model, x=sub["Regime"].str[:25], y=sub["hit_rate"],
                marker_color=MODEL_COLORS.get(model, "#607D8B"), opacity=0.85,
            ))
        fig_hr.add_hline(y=50, line_dash="dash", line_color="gray",
                         annotation_text="50% (random)")
        fig_hr.update_layout(barmode="group", height=320, showlegend=False,
                              plot_bgcolor="white", xaxis=dict(tickangle=-30))
        st.plotly_chart(fig_hr, width="stretch")

    with col2:
        st.subheader("Annualised Alpha by Regime (%)")
        fig_alpha = go.Figure()
        for model in available_models:
            sub = stats_df[(stats_df["Model"] == model) &
                           (stats_df["Regime"].isin(selected_regimes))]
            fig_alpha.add_trace(go.Bar(
                name=model, x=sub["Regime"].str[:25], y=sub["ann_alpha"],
                marker_color=MODEL_COLORS.get(model, "#607D8B"), opacity=0.85,
            ))
        fig_alpha.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        fig_alpha.update_layout(barmode="group", height=320, showlegend=False,
                                plot_bgcolor="white", xaxis=dict(tickangle=-30))
        st.plotly_chart(fig_alpha, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bootstrap Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎲 Stochastic Bootstrap — IR Confidence Intervals")
    st.caption(f"Block bootstrap ({n_bootstrap} iterations, block size={block_size} months). "
               f"Bars show 5th–95th percentile range.")

    with st.spinner(f"Running {n_bootstrap} bootstrap iterations ..."):
        boot_rows = []
        for model in available_models:
            df = all_data[model]
            for reg in ["FULL PERIOD"] + selected_regimes:
                if reg == "FULL PERIOD":
                    sub = df["active_ret"]
                else:
                    sub = df[df["regime"] == reg]["active_ret"]

                if sub.dropna().shape[0] < 6:
                    continue

                boot_irs = block_bootstrap(sub, n_boot=n_bootstrap, block_size=block_size)
                point_ir = regime_stats(sub)["ir"]
                boot_rows.append({
                    "Model":   model,
                    "Regime":  reg,
                    "point":   point_ir,
                    "p5":      np.nanpercentile(boot_irs, 5),
                    "p50":     np.nanpercentile(boot_irs, 50),
                    "p95":     np.nanpercentile(boot_irs, 95),
                    "n":       sub.dropna().shape[0],
                })

    boot_df = pd.DataFrame(boot_rows)

    # IR with CI chart
    fig_boot = go.Figure()
    regime_labels = ["FULL PERIOD"] + selected_regimes
    x_positions   = list(range(len(regime_labels)))

    offsets = np.linspace(-0.3, 0.3, len(available_models))
    for i, model in enumerate(available_models):
        sub = boot_df[boot_df["Model"] == model]
        sub = sub[sub["Regime"].isin(regime_labels)].set_index("Regime").reindex(regime_labels).reset_index()

        x_jit = [x + offsets[i] for x in x_positions]
        color = MODEL_COLORS.get(model, "#607D8B")

        # CI range
        if show_ci:
            for j, row in sub.iterrows():
                if np.isnan(row["p5"]):
                    continue
                fig_boot.add_shape(
                    type="line",
                    x0=x_jit[j], x1=x_jit[j],
                    y0=row["p5"], y1=row["p95"],
                    line=dict(color=color, width=2),
                )

        # Point estimate
        fig_boot.add_trace(go.Scatter(
            x=x_jit, y=sub["point"],
            mode="markers",
            name=model,
            marker=dict(color=color, size=10, symbol="circle"),
        ))

    fig_boot.add_hline(y=0,   line_dash="solid", line_color="gray", line_width=1)
    fig_boot.add_hline(y=0.5, line_dash="dash",  line_color="green",
                       annotation_text="IR=0.5", annotation_position="right")
    fig_boot.add_hline(y=1.0, line_dash="dash",  line_color="gold",
                       annotation_text="IR=1.0", annotation_position="right")

    fig_boot.update_layout(
        height=450,
        xaxis=dict(tickmode="array", tickvals=x_positions,
                   ticktext=[r[:25] for r in regime_labels], tickangle=-30),
        yaxis_title="Information Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        title="IR Point Estimates with 90% Bootstrap CI",
    )
    st.plotly_chart(fig_boot, width="stretch")

    # Bootstrap distribution for selected regime
    st.subheader("Bootstrap IR Distribution")
    col_model, col_regime = st.columns(2)
    with col_model:
        boot_model = st.selectbox("Model:", available_models, key="boot_model")
    with col_regime:
        boot_regime = st.selectbox("Regime:", ["FULL PERIOD"] + selected_regimes, key="boot_regime")

    df_sel = all_data[boot_model]
    if boot_regime == "FULL PERIOD":
        sub_sel = df_sel["active_ret"]
    else:
        sub_sel = df_sel[df_sel["regime"] == boot_regime]["active_ret"]

    if sub_sel.dropna().shape[0] >= 6:
        boot_irs_sel = block_bootstrap(sub_sel, n_boot=n_bootstrap, block_size=block_size)
        point_ir_sel = regime_stats(sub_sel)["ir"]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=boot_irs_sel, nbinsx=50,
            marker_color=MODEL_COLORS.get(boot_model, "#607D8B"),
            opacity=0.75, name="Bootstrap IR",
        ))
        fig_dist.add_vline(x=point_ir_sel, line_color="red", line_width=2,
                           annotation_text=f"Point IR={point_ir_sel:.3f}",
                           annotation_position="top right")
        fig_dist.add_vline(x=np.nanpercentile(boot_irs_sel, 5), line_dash="dash",
                           line_color="orange", annotation_text="5th pct")
        fig_dist.add_vline(x=np.nanpercentile(boot_irs_sel, 95), line_dash="dash",
                           line_color="orange", annotation_text="95th pct")
        fig_dist.add_vline(x=0, line_color="gray", line_width=1)
        fig_dist.update_layout(
            height=320, showlegend=False,
            xaxis_title="IR", yaxis_title="Count",
            plot_bgcolor="white", paper_bgcolor="white",
            title=f"{boot_model} — {boot_regime} (n={sub_sel.dropna().shape[0]} months)",
        )
        st.plotly_chart(fig_dist, width="stretch")

        p5, p50, p95 = (np.nanpercentile(boot_irs_sel, q) for q in [5, 50, 95])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Point IR",  f"{point_ir_sel:.3f}")
        c2.metric("Median IR", f"{p50:.3f}")
        c3.metric("5th pct",   f"{p5:.3f}")
        c4.metric("95th pct",  f"{p95:.3f}")
    else:
        st.warning(f"Not enough data for {boot_model} / {boot_regime} ({sub_sel.dropna().shape[0]} months).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Monte Carlo Projection
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔮 Monte Carlo Forward Projection")
    st.caption("Simulates future cumulative active return paths based on estimated IR and tracking error.")

    col_m, col_r = st.columns(2)
    with col_m:
        mc_model  = st.selectbox("Model:", available_models, key="mc_model")
    with col_r:
        mc_regime = st.selectbox("Base regime:", ["FULL PERIOD"] + selected_regimes, key="mc_regime")

    df_mc = all_data[mc_model]
    if mc_regime == "FULL PERIOD":
        sub_mc = df_mc["active_ret"]
    else:
        sub_mc = df_mc[df_mc["regime"] == mc_regime]["active_ret"]

    s_mc = regime_stats(sub_mc)
    ir_mc = s_mc["ir"] if not np.isnan(s_mc["ir"]) else 0.0
    te_mc = s_mc["track_err"] / 100 if not np.isnan(s_mc["track_err"]) else 0.02

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

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p95"], mode="lines",
        line=dict(width=0), showlegend=False, name="95th pct",
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(33,150,243,0.15)", name="5th–95th pct", showlegend=True,
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p25"], mode="lines",
        line=dict(width=0), showlegend=False, name="25th pct",
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p75"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(33,150,243,0.25)", name="25th–75th pct", showlegend=True,
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_df["month"], y=mc_df["p50"], mode="lines",
        line=dict(color="#2196F3", width=2.5), name="Median", showlegend=True,
    ))
    fig_mc.add_hline(y=0, line_color="gray", line_width=1)

    fig_mc.update_layout(
        height=420,
        xaxis_title="Months Forward", yaxis_title="Cumulative Active Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        title=f"{mc_model} | IR={ir_use:.3f} | TE={te_use*100:.1f}% | {n_mc_sims} simulations",
    )
    st.plotly_chart(fig_mc, width="stretch")

    c1, c2, c3, c4, c5 = st.columns(5)
    last = mc_df.iloc[-1]
    c1.metric(f"Median ({n_mc_months}m)", f"{last['p50']:.1f}%")
    c2.metric("5th pct",  f"{last['p5']:.1f}%")
    c3.metric("25th pct", f"{last['p25']:.1f}%")
    c4.metric("75th pct", f"{last['p75']:.1f}%")
    c5.metric("95th pct", f"{last['p95']:.1f}%")

    prob_positive = (mc_df["p50"] > 0).mean() * 100
    st.info(f"📊 Probability of positive cumulative alpha at {n_mc_months} months: "
            f"**{mc_df['p50'].iloc[-1] > 0 and 'likely' or 'uncertain'}** "
            f"(median = {last['p50']:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Full Stats Table
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Full Statistics Table")
    st.caption("All metrics per model per regime. IR > 0.5 = good, > 1.0 = excellent.")

    display_rows = []
    for model in available_models:
        df = all_data[model]
        for reg in ["FULL PERIOD"] + selected_regimes:
            if reg == "FULL PERIOD":
                sub = df["active_ret"]
            else:
                sub = df[df["regime"] == reg]["active_ret"]
            s = regime_stats(sub)
            display_rows.append({
                "Model":      model,
                "Regime":     reg[:35],
                "N months":   s["n"],
                "Ann α (%)":  s["ann_alpha"],
                "TE (%)":     s["track_err"],
                "IR":         s["ir"],
                "Hit Rate %": s["hit_rate"],
                "Max DD %":   s["max_dd"],
            })

    disp_df = pd.DataFrame(display_rows)

    def colour_ir(val):
        if pd.isna(val): return ""
        if val >= 1.0:   return "background-color: #C8E6C9"
        if val >= 0.5:   return "background-color: #FFF9C4"
        if val < 0:      return "background-color: #FFCDD2"
        return ""

    styled = (disp_df.style
              .applymap(colour_ir, subset=["IR"])
              .format({"Ann α (%)": "{:.2f}", "TE (%)": "{:.2f}",
                       "IR": "{:.3f}", "Hit Rate %": "{:.1f}", "Max DD %": "{:.2f}"})
              .hide(axis="index"))
    st.dataframe(styled, width="stretch", height=500)

    st.download_button(
        "⬇️ Download as CSV",
        data=disp_df.to_csv(index=False),
        file_name="regime_stats.csv",
        mime="text/csv",
    )

    # Heatmap: IR across models × regimes
    st.subheader("IR Heatmap — Models × Regimes")
    pivot = disp_df.pivot_table(values="IR", index="Model", columns="Regime", aggfunc="first")
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.5,
        aspect="auto", height=300,
        text_auto=".2f",
        title="Information Ratio by Model and Regime",
        zmin=-0.5, zmax=2.0,
    )
    fig_hm.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_hm, width="stretch")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ML-Driven S&P 500 Index Enhancement | "
    "CS-Transformer IR 1.874 (test period 2023–2025) | "
    "Run `streamlit run 6_regime_dashboard.py` to launch"
)
