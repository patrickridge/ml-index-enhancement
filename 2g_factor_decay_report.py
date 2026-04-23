"""
2g_factor_decay_report.py
==========================
IS vs OOS factor-decay diagnostics.

Reads factor_ic_summary.csv, factor_oos_ic.csv, and factor_ic_decay.csv
(outputs from 2a_factor_analysis.py) and produces:

  1. IS vs OOS IC comparison table — sorted by OOS decay severity
  2. IC decay curves plot — shows how fast IC drops at lags 1–12m
  3. Regime-breakdown analysis — which regimes each factor works in OOS
  4. Factor health scorecard — green/amber/red per factor
  5. Written narrative report → reports/factor_decay_report.md

Run AFTER 2a_factor_analysis.py.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr
import time as _time
_t0 = _time.time()

try:
    from config import MACRO_COLS, DATA_DIR as _cfg_dir, TRAIN_END, START_DATE
except ImportError:
    MACRO_COLS  = []
    _cfg_dir    = Path("data")
    TRAIN_END   = "2022-12-31"
    START_DATE  = "2010-01-01"

DATA_DIR    = Path("data")
FIG_DIR     = Path("figures")
REPORT_DIR  = Path("reports")
PANEL_PATH  = DATA_DIR / "panel_monthly_enriched.parquet"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_STOCKS = 20  # minimum stocks per month for IC computation

# Regime date ranges (used for OOS regime breakdown)
REGIMES = {
    "QE Bull (2010–2019)":    ("2010-01-01", "2019-12-31"),
    "COVID Crash (2020)":     ("2020-01-01", "2020-12-31"),
    "Rate Hike Bear (2022)":  ("2022-01-01", "2022-12-31"),
    "AI Bull (2023–2024)":    ("2023-01-01", "2024-06-30"),
    "OOS Test (2024+)":       ("2024-07-01", "2099-12-31"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def monthly_ic(panel: pd.DataFrame, factor: str,
               target: str = "fwd_ret_1m") -> pd.Series:
    results = {}
    for dt, grp in panel.groupby("date"):
        sub = grp[[factor, target]].dropna()
        if len(sub) < MIN_STOCKS:
            continue
        corr, _ = spearmanr(sub[factor], sub[target])
        if not np.isnan(corr):
            results[dt] = corr
    return pd.Series(results, name=factor)


def ic_decay_at_lag(panel: pd.DataFrame, factor: str, lag: int,
                    target: str = "fwd_ret_1m") -> float:
    """IC of factor at month t vs return at month t+lag."""
    panel_s = panel.sort_values(["ticker", "date"])
    panel_s = panel_s.copy()
    panel_s["fwd_shifted"] = (panel_s
                              .groupby("ticker")[target]
                              .shift(-lag))
    sub = panel_s[[factor, "fwd_shifted"]].dropna()
    if len(sub) < MIN_STOCKS * 5:
        return np.nan
    corr, _ = spearmanr(sub[factor], sub["fwd_shifted"])
    return float(corr)


def health_label(row: pd.Series) -> str:
    """Green/Amber/Red based on OOS IC vs IS IC."""
    if pd.isna(row["ic_oos"]) or pd.isna(row["ic_is"]):
        return "GREY"
    if row["sign_flip"]:
        return "RED"
    dr = row.get("decay_ratio", np.nan)
    if pd.isna(dr):
        return "GREY"
    if abs(dr) >= 0.7:
        return "GREEN"
    if abs(dr) >= 0.3:
        return "AMBER"
    return "RED"


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

    # Panel
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    always_exclude  = {"date", "ticker", "fwd_ret_1m"}
    macro_in_panel  = [c for c in MACRO_COLS if c in panel.columns]
    feat_cols = [c for c in panel.columns
                 if c not in always_exclude and c not in macro_in_panel]

    # IS / OOS IC summary (from 2a)
    oos_path = DATA_DIR / "factor_oos_ic.csv"
    ic_path  = DATA_DIR / "factor_ic_summary.csv"

    if not oos_path.exists():
        raise FileNotFoundError(
            f"{oos_path} not found — run 2a_factor_analysis.py first.")
    if not ic_path.exists():
        raise FileNotFoundError(
            f"{ic_path} not found — run 2a_factor_analysis.py first.")

    oos_df  = pd.read_csv(oos_path)
    ic_df   = pd.read_csv(ic_path)

    # IC decay CSV (optional — recompute if missing)
    decay_path = DATA_DIR / "factor_ic_decay.csv"
    decay_df   = pd.read_csv(decay_path, index_col=0) if decay_path.exists() else None

    return panel, feat_cols, oos_df, ic_df, decay_df


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — IS vs OOS comparison table
# ─────────────────────────────────────────────────────────────────────────────

def section_is_vs_oos(oos_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*70)
    print("SECTION 1 — IS vs OOS IC Comparison (sorted by decay severity)")
    print("="*70)

    df = oos_df.copy()
    df["health"] = df.apply(health_label, axis=1)

    # Sort: REDs first, then AMBERs, then GREENs; within each group by |OOS IC|
    order_map = {"RED": 0, "AMBER": 1, "GREEN": 2, "GREY": 3}
    df["_sort"] = df["health"].map(order_map)
    df = df.sort_values(["_sort", "ic_oos"], key=lambda s: s.abs() if s.name == "ic_oos" else s,
                        ascending=[True, False])

    print(f"\n{'Factor':<26} {'IC_IS':>8} {'IC_OOS':>8} {'ICIR_OOS':>9} "
          f"{'Decay':>7}  Health")
    print("─"*72)
    for _, row in df.iterrows():
        decay_s  = f"{row['decay_ratio']:+.2f}" if not pd.isna(row.get("decay_ratio")) else "   n/a"
        ic_is_s  = f"{row['ic_is']:+.4f}"       if not pd.isna(row.get("ic_is"))       else "    n/a"
        ic_oos_s = f"{row['ic_oos']:+.4f}"      if not pd.isna(row.get("ic_oos"))      else "    n/a"
        icir_s   = f"{row['icir_oos']:+.3f}"    if not pd.isna(row.get("icir_oos"))    else "   n/a"
        sym = {"GREEN": "✓", "AMBER": "~", "RED": "✗", "GREY": "?"}[row["health"]]
        print(f"  {row['factor']:<24} {ic_is_s:>8} {ic_oos_s:>8} "
              f"{icir_s:>9} {decay_s:>7}  {sym} {row['health']}")

    counts = df["health"].value_counts()
    print(f"\nSummary:")
    for label in ["GREEN", "AMBER", "RED", "GREY"]:
        n = counts.get(label, 0)
        print(f"  {label:5s}: {n}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — IC decay curves
# ─────────────────────────────────────────────────────────────────────────────

def section_ic_decay_curves(panel: pd.DataFrame, oos_df: pd.DataFrame,
                             decay_df_precomp: pd.DataFrame = None):
    print("\n" + "="*70)
    print("SECTION 2 — IC Decay Curves (lags 0–12 months, IS vs OOS)")
    print("="*70)

    # Pick top 10 factors by |OOS IC| (any direction)
    top10 = (oos_df.dropna(subset=["ic_oos"])
             .reindex(oos_df["ic_oos"].abs().sort_values(ascending=False).index)
             .head(10)["factor"].tolist())

    train_panel = panel[panel["date"] <= TRAIN_END].reset_index(drop=True)
    oos_start   = pd.Timestamp(TRAIN_END) + pd.DateOffset(days=1)
    oos_panel   = panel[panel["date"] >= oos_start].reset_index(drop=True)

    LAGS = list(range(0, 13))  # 0 to 12 months

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
    fig.suptitle("IC Decay Curves — IS (blue) vs OOS (red) | top 10 factors by |OOS IC|",
                 fontsize=13, fontweight="bold")

    for idx, fac in enumerate(top10):
        ax = axes[idx // 5][idx % 5]

        # IS decay
        is_vals  = []
        oos_vals = []
        for lag in LAGS:
            if decay_df_precomp is not None and str(lag) in decay_df_precomp.columns and fac in decay_df_precomp.index:
                is_ic = float(decay_df_precomp.loc[fac, str(lag)])
            else:
                is_ic = ic_decay_at_lag(train_panel, fac, lag)
            oos_ic = ic_decay_at_lag(oos_panel, fac, lag)
            is_vals.append(is_ic)
            oos_vals.append(oos_ic)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.plot(LAGS, is_vals,  "b-o", markersize=4, label="IS",  linewidth=1.5)
        ax.plot(LAGS, oos_vals, "r-s", markersize=4, label="OOS", linewidth=1.5)
        ax.set_title(fac, fontsize=9, fontweight="bold")
        ax.set_xlabel("Lag (months)", fontsize=8)
        ax.set_ylabel("IC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "factor_decay_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Regime breakdown (OOS regimes)
# ─────────────────────────────────────────────────────────────────────────────

def section_regime_breakdown(panel: pd.DataFrame, oos_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*70)
    print("SECTION 3 — Regime-by-Regime IC Breakdown (all periods)")
    print("="*70)

    # Only analyse factors with non-trivial OOS IC
    top_factors = (oos_df.dropna(subset=["ic_oos"])
                   .reindex(oos_df["ic_oos"].abs().sort_values(ascending=False).index)
                   .head(20)["factor"].tolist())

    regime_records = []
    for regime_name, (r_start, r_end) in REGIMES.items():
        sub = panel[(panel["date"] >= r_start) & (panel["date"] <= r_end)]
        n_months = sub["date"].nunique()
        for fac in top_factors:
            ic_s = monthly_ic(sub, fac)
            regime_records.append({
                "regime":   regime_name,
                "factor":   fac,
                "ic_mean":  round(ic_s.mean(), 4)   if len(ic_s) > 0 else np.nan,
                "ic_std":   round(ic_s.std(), 4)    if len(ic_s) > 1 else np.nan,
                "n_months": n_months,
            })

    rdf = pd.DataFrame(regime_records)

    # Pivot: rows = factor, columns = regime
    pivot = rdf.pivot_table(index="factor", columns="regime",
                            values="ic_mean", aggfunc="first")
    # Re-order columns by time
    col_order = [c for c in REGIMES.keys() if c in pivot.columns]
    pivot = pivot[col_order]

    print(f"\n{'Factor':<26} " + "  ".join(f"{c[:18]:>18}" for c in col_order))
    print("─" * (26 + 20 * len(col_order)))
    for fac in top_factors:
        if fac not in pivot.index:
            continue
        row_vals = "  ".join(
            f"{pivot.loc[fac, c]:>+18.4f}" if not pd.isna(pivot.loc[fac, c]) else f"{'n/a':>18}"
            for c in col_order
        )
        print(f"  {fac:<24} {row_vals}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, len(top_factors) * 0.45)))
    data  = pivot.values.astype(float)
    vmax  = np.nanpercentile(np.abs(data), 95)
    im    = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                      vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([c.replace(" (", "\n(") for c in col_order], fontsize=9)
    ax.set_yticks(range(len(top_factors)))
    ax.set_yticklabels(top_factors, fontsize=9)
    for i in range(len(top_factors)):
        for j in range(len(col_order)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7.5, color="black" if abs(v) < vmax * 0.6 else "white")
    plt.colorbar(im, ax=ax, label="IC mean")
    ax.set_title("Factor IC by Market Regime (top 20 by |OOS IC|)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "factor_regime_heatmap.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    return rdf


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Health scorecard bar chart
# ─────────────────────────────────────────────────────────────────────────────

def section_health_scorecard(health_df: pd.DataFrame):
    print("\n" + "="*70)
    print("SECTION 4 — Factor Health Scorecard")
    print("="*70)

    colour_map = {"GREEN": "#2ecc71", "AMBER": "#f39c12",
                  "RED": "#e74c3c",   "GREY": "#95a5a6"}
    df = health_df.dropna(subset=["ic_oos"]).copy()
    df = df.sort_values("ic_oos", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.28)))
    colours = [colour_map[h] for h in df["health"]]
    bars = ax.barh(df["factor"], df["ic_oos"], color=colours, edgecolor="white",
                   linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    # Add IS IC as markers
    for i, (_, row) in enumerate(df.iterrows()):
        if not pd.isna(row["ic_is"]):
            ax.plot(row["ic_is"], i, "k|", markersize=8, markeredgewidth=2,
                    label="IS IC" if i == 0 else "")
    ax.set_xlabel("IC (OOS test period)")
    ax.set_title("Factor Health Scorecard\n"
                 "Bar = OOS IC  |  Black tick = IS IC\n"
                 "Green = stable  Amber = decayed  Red = flipped/dead",
                 fontsize=11)
    ax.legend(["IS IC (black tick)"], loc="lower right", fontsize=9)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=colour_map[k], label=k)
                    for k in ["GREEN", "AMBER", "RED", "GREY"]]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "factor_health_scorecard.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Written Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown_report(health_df: pd.DataFrame, regime_df: pd.DataFrame,
                          n_oos_months: int):
    print("\n" + "="*70)
    print("SECTION 5 — Writing Markdown Diagnostic Report")
    print("="*70)

    green  = health_df[health_df["health"] == "GREEN"]["factor"].tolist()
    amber  = health_df[health_df["health"] == "AMBER"]["factor"].tolist()
    red    = health_df[health_df["health"] == "RED"]["factor"].tolist()
    flipped = health_df[health_df["sign_flip"] == True]["factor"].tolist()

    # Best OOS factors
    best_oos = (health_df.dropna(subset=["ic_oos"])
                .sort_values("ic_oos", key=abs, ascending=False)
                .head(5)[["factor", "ic_is", "ic_oos", "icir_oos", "decay_ratio",
                           "health"]]
                .to_markdown(index=False, floatfmt="+.4f"))

    # Worst (most decayed / flipped)
    worst_oos = (health_df[health_df["health"].isin(["RED", "AMBER"])]
                 .sort_values("decay_ratio", key=abs, ascending=True)
                 .head(10)[["factor", "ic_is", "ic_oos", "decay_ratio", "health"]]
                 .to_markdown(index=False, floatfmt="+.4f"))

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    lines = [
        f"# Factor Decay Diagnostic Report",
        f"**Generated:** {today}  ",
        f"**IS training period:** {START_DATE} → {TRAIN_END}  ",
        f"**OOS test period:** {TRAIN_END[:7]} → present ({n_oos_months} months)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **{len(green)} factors are GREEN** (OOS IC stable, decay ratio ≥ 0.7) — safe to use",
        f"- **{len(amber)} factors are AMBER** (some decay, ratio 0.3–0.7) — use with caution",
        f"- **{len(red)} factors are RED** (severe decay or sign flip OOS) — review or drop",
        f"- **{len(flipped)} factors flipped sign OOS**: `{', '.join(flipped) if flipped else 'none'}`",
        "",
        "### Top 5 Factors by |OOS IC|",
        "",
        best_oos,
        "",
        "### Most Decayed / Flipped Factors",
        "",
        worst_oos if worst_oos else "_None_",
        "",
        "---",
        "",
        "## 2. Key Findings on Factor Decay",
        "",
        "### Why factors decay OOS",
        "1. **Regime shift**: A factor trained on the QE bull market (2010–2019) may not work",
        "   in the rate-hike bear (2022) or AI mega-cap bull (2023–2024). Factors with",
        "   IC sign flips across regimes are the most dangerous.",
        "2. **Overfitting**: Factors computed from the same price data can have spurious",
        "   IS IC that doesn't survive. IC < 0.02 in training is typically noise.",
        "3. **Crowding**: Widely-used factors (momentum, RSI) get arbitraged away.",
        "   OOS ICIR < 0.1 suggests the alpha is crowded out.",
        "4. **Market structure change**: The 2023–2024 'Magnificent 7' concentration means",
        "   cross-sectional models face structural headwinds — 7 stocks drive 60% of SPX",
        "   returns, and no within-S&P cross-section factor captures this.",
        "",
        "### Green factors — what's working OOS",
        "",
        f"```\n{chr(10).join(green)}\n```",
        "",
        "These have consistent IC across both IS and OOS periods. Prioritise these in",
        "the CS-Transformer feature set.",
        "",
        "### Red/Amber factors — what needs investigation",
        "",
        f"```\n{chr(10).join(red + amber)}\n```",
        "",
        "Consider: (a) dropping from the model, (b) applying regime-conditional weights,",
        "or (c) orthogonalising against known regime exposures before use.",
        "",
        "---",
        "",
        "## 3. Recommended Next Steps",
        "",
        "1. **Retrain CS-Transformer** with GREEN factors prioritised; use extended",
        "   training window (2010–2022) and 18-month validation (Jan 2023 – Jun 2024).",
        "2. **Add log_mktcap** (size factor via yfinance shares × price) — directly",
        "   addresses the 2024 mega-cap underperformance.",
        "3. **Investigate AMBER factors by regime**: run `2f_factor_diagnostics.py`",
        "   and check if the decay is concentrated in one specific regime.",
        "4. **Apply IC-weighted combination**: in `2d_factor_weights.py`, weight factors",
        "   by OOS ICIR rather than IS ICIR to avoid overweighting decayed signals.",
        "5. **Add residual momentum**: `ret_6m - beta × spx_ret_6m` (distinct from",
        "   `residual_ret_12m` already in the panel — add a 6m version).",
        "",
        "---",
        "",
        "## 4. Figures Generated",
        "",
        "| Figure | Description |",
        "|--------|-------------|",
        "| `figures/factor_decay_curves.png` | IS vs OOS IC decay curves, lags 0–12m |",
        "| `figures/factor_regime_heatmap.png` | IC heatmap across all 5 regimes |",
        "| `figures/factor_health_scorecard.png` | Green/Amber/Red health bar chart |",
        "",
        "---",
        "",
        "*Report auto-generated by `2g_factor_decay_report.py`*",
    ]

    out = REPORT_DIR / "factor_decay_report.md"
    out.write_text("\n".join(lines))
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FACTOR DECAY DIAGNOSTIC REPORT")
    print("=" * 70)

    panel, feat_cols, oos_df, ic_df, decay_df = load_data()

    oos_start   = pd.Timestamp(TRAIN_END) + pd.DateOffset(days=1)
    n_oos_months = panel[panel["date"] >= oos_start]["date"].nunique()
    print(f"  Panel: {panel.shape}  |  Features: {len(feat_cols)}")
    print(f"  IS: {START_DATE} → {TRAIN_END}  |  OOS: {n_oos_months} months")

    # Section 1: IS vs OOS table + health labels
    health_df = section_is_vs_oos(oos_df)

    # Section 2: IC decay curves (IS vs OOS, lags 0–12)
    section_ic_decay_curves(panel, oos_df, decay_df_precomp=decay_df)

    # Section 3: regime breakdown heatmap
    regime_df = section_regime_breakdown(panel, oos_df)

    # Section 4: health scorecard bar chart
    section_health_scorecard(health_df)

    # Section 5: markdown narrative report
    write_markdown_report(health_df, regime_df, n_oos_months)

    print(f"\n{'='*70}")
    print("FACTOR DECAY REPORT COMPLETE")
    print(f"{'='*70}")
    print(f"  figures/factor_decay_curves.png")
    print(f"  figures/factor_regime_heatmap.png")
    print(f"  figures/factor_health_scorecard.png")
    print(f"  reports/factor_decay_report.md")
    print(f"\nDone in {(_time.time() - _t0)/60:.1f} min")


if __name__ == "__main__":
    main()
