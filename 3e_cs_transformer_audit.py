"""
3e_cs_transformer_audit.py
Audit what the CS-Transformer signal is actually ranking.

Requires (run first):
  1h_feature_engineering.py → data/panel_monthly_enriched.parquet
  3c_cs_transformer.py      → data/scores_cs_transformer.parquet

Outputs:
  figures/cs_transformer_loadings.png   - factor loading bar chart
  data/cs_transformer_loadings.csv      - loadings table
  figures/cs_transformer_ic_stability.png - rolling IC + regime breakdown
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# Paths
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
FIG_DIR    = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

PANEL_FILE       = DATA_DIR / "panel_monthly_enriched.parquet"
FACTOR_FILE      = DATA_DIR / "factor_selected.csv"

# Candidate score files in priority order
SCORE_CANDIDATES = [
    DATA_DIR / "scores_cs_transformer.parquet",
    DATA_DIR / "bt_ie_cs_transformer.csv",
    DATA_DIR / "bt_cs_transformer.csv",
]

OUT_LOADINGS_PNG = FIG_DIR / "cs_transformer_loadings.png"
OUT_LOADINGS_CSV = DATA_DIR / "cs_transformer_loadings.csv"
OUT_IC_PNG       = FIG_DIR / "cs_transformer_ic_stability.png"


# Helpers

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Cross-sectional Spearman rank correlation; returns NaN on degenerate input."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan
    return stats.spearmanr(a[mask], b[mask]).statistic


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns NaN on degenerate input."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan
    return stats.pearsonr(a[mask], b[mask])[0]


def zscore_cs(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score (within a month group)."""
    mu, sd = s.mean(), s.std(ddof=1)
    if sd < 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


# 1. Load score file

scores_path = None
for candidate in SCORE_CANDIDATES:
    if candidate.exists():
        scores_path = candidate
        break

if scores_path is None:
    print("=" * 70)
    print("ERROR: No CS-Transformer score file found.")
    print()
    print("Expected one of:")
    for c in SCORE_CANDIDATES:
        print(f"  {c}")
    print()
    print("Generate scores first by running:")
    print("  python 3c_cs_transformer.py")
    print("=" * 70)
    sys.exit(1)

print(f"Loading CS-T scores from: {scores_path.name}")
if scores_path.suffix == ".parquet":
    scores_df = pd.read_parquet(scores_path)
else:
    scores_df = pd.read_csv(scores_path, parse_dates=["date"])

# Normalise column names - we need at minimum: date, ticker, score
scores_df.columns = [c.strip().lower() for c in scores_df.columns]
if "score" not in scores_df.columns:
    # Fallback: first numeric column that is not a known return column
    skip = {"fwd_ret_1m", "ret_1m", "date", "ticker"}
    numeric_cols = [c for c in scores_df.select_dtypes("number").columns if c not in skip]
    if not numeric_cols:
        print("ERROR: Cannot identify a score column in", scores_path.name)
        sys.exit(1)
    scores_df = scores_df.rename(columns={numeric_cols[0]: "score"})
    print(f"  (using column '{numeric_cols[0]}' as score)")

scores_df["date"] = pd.to_datetime(scores_df["date"])
print(f"  Rows: {len(scores_df):,} | "
      f"Months: {scores_df['date'].nunique()} | "
      f"Tickers: {scores_df['ticker'].nunique()}")


# 2. Load panel and factor list

print(f"\nLoading panel: {PANEL_FILE.name}")
panel = pd.read_parquet(PANEL_FILE)
panel["date"] = pd.to_datetime(panel["date"])

factor_meta = pd.read_csv(FACTOR_FILE)
factors = factor_meta["factor"].tolist()

# Keep only factors actually present in panel
factors = [f for f in factors if f in panel.columns]
print(f"  Factors available: {len(factors)} / {len(factor_meta)}")

# Identify market-vol proxy for regime (use spx_vol_63d if present, else vol_252d median)
if "spx_vol_63d" in panel.columns:
    regime_col = "spx_vol_63d"
elif "market_vol_regime" in panel.columns:
    regime_col = "market_vol_regime"
else:
    regime_col = None
print(f"  Regime proxy column: {regime_col or 'None (regime split skipped)'}")


# 3. Merge scores with panel

print("\nMerging scores with panel…")
panel_sub = panel[["date", "ticker"] + factors + (["fwd_ret_1m"] if "fwd_ret_1m" in panel.columns else [])
                  + ([regime_col] if regime_col and regime_col in panel.columns else [])].copy()

merged = pd.merge(scores_df[["date", "ticker", "score"]
                             + (["fwd_ret_1m"] if "fwd_ret_1m" in scores_df.columns else [])],
                  panel_sub,
                  on=["date", "ticker"],
                  how="inner",
                  suffixes=("_cst", "_panel"))

# Reconcile fwd_ret_1m: prefer the one from scores (already aligned)
if "fwd_ret_1m_cst" in merged.columns:
    merged["fwd_ret_1m"] = merged["fwd_ret_1m_cst"].fillna(merged.get("fwd_ret_1m_panel", np.nan))
elif "fwd_ret_1m_panel" in merged.columns:
    merged["fwd_ret_1m"] = merged["fwd_ret_1m_panel"]

# Clean up suffix columns
merged = merged[[c for c in merged.columns if not c.endswith(("_cst", "_panel"))
                 or c in ("fwd_ret_1m",)]]
if "fwd_ret_1m_cst" in merged.columns:
    merged = merged.drop(columns=[c for c in merged.columns
                                   if c.endswith(("_cst", "_panel")) and c != "fwd_ret_1m"])

print(f"  Merged rows: {len(merged):,} | Months: {merged['date'].nunique()}")

months = sorted(merged["date"].unique())


# SECTION A - Factor Loadings (cross-sectional Spearman)

print("\n--- Section A: Factor Loadings ---")

loading_records = []  # (factor, month, corr)

for month in months:
    sub = merged[merged["date"] == month].copy()
    if len(sub) < 20:
        continue

    # Cross-sectional z-score of the CS-T score itself
    sub["score_z"] = zscore_cs(sub["score"])

    for fac in factors:
        if fac not in sub.columns:
            continue
        fac_z = zscore_cs(sub[fac])
        rho = spearman_corr(sub["score_z"].values, fac_z.values)
        loading_records.append({"factor": fac, "date": month, "corr": rho})

loadings_df = pd.DataFrame(loading_records)

# Aggregate per factor
loading_agg = (
    loadings_df.groupby("factor")["corr"]
    .agg(mean_corr="mean", std_corr="std", n_months="count")
    .reset_index()
)
loading_agg["abs_mean"] = loading_agg["mean_corr"].abs()
loading_agg = loading_agg.sort_values("abs_mean", ascending=False).reset_index(drop=True)

print("\nTop 15 factor loadings (|mean Spearman ρ|):")
print(loading_agg.head(15).to_string(index=False))

# Save CSV
loading_agg.to_csv(OUT_LOADINGS_CSV, index=False)
print(f"\nSaved: {OUT_LOADINGS_CSV}")


# Plot A: Horizontal bar chart

fig_a, ax_a = plt.subplots(figsize=(10, max(6, len(factors) * 0.38)))

# Sort ascending for bottom→top display
plot_df = loading_agg.sort_values("mean_corr", ascending=True)
colors = ["#d62728" if v < 0 else "#1f77b4" for v in plot_df["mean_corr"]]
y_pos  = np.arange(len(plot_df))

ax_a.barh(y_pos, plot_df["mean_corr"], xerr=plot_df["std_corr"],
          color=colors, alpha=0.80, ecolor="#555555",
          error_kw=dict(linewidth=0.8, capsize=3))

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(plot_df["factor"], fontsize=8)
ax_a.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax_a.set_xlabel("Mean Cross-Sectional Spearman ρ  (±1 std)", fontsize=10)
ax_a.set_title(
    "CS-Transformer Signal - Factor Loadings\n(Cross-Sectional Spearman Correlation)",
    fontsize=12, fontweight="bold"
)
pos_patch = mpatches.Patch(color="#1f77b4", alpha=0.8, label="Positive loading")
neg_patch = mpatches.Patch(color="#d62728", alpha=0.8, label="Negative loading")
ax_a.legend(handles=[pos_patch, neg_patch], fontsize=9)
ax_a.grid(axis="x", linestyle=":", alpha=0.5)

fig_a.tight_layout()
fig_a.savefig(OUT_LOADINGS_PNG, dpi=150, bbox_inches="tight")
plt.close(fig_a)
print(f"Saved: {OUT_LOADINGS_PNG}")


# SECTION B - IC Stability Analysis

print("\n--- Section B: IC Stability ---")

fwd_col = "fwd_ret_1m"
if fwd_col not in merged.columns:
    print("  WARNING: fwd_ret_1m not available - IC analysis skipped.")
else:
    # Monthly IC series
    ic_records = []
    for month in months:
        sub = merged[merged["date"] == month]
        ic = pearson_corr(sub["score"].values, sub[fwd_col].values)
        ic_records.append({"date": month, "ic": ic})

    ic_series = pd.DataFrame(ic_records).set_index("date")["ic"].dropna().sort_index()

    overall_ic   = ic_series.mean()
    overall_std  = ic_series.std(ddof=1)
    overall_icir = overall_ic / overall_std if overall_std > 0 else np.nan
    t_stat       = overall_ic / (overall_std / np.sqrt(len(ic_series)))

    print(f"\n  Overall IC  : {overall_ic:.4f}")
    print(f"  IC Std Dev  : {overall_std:.4f}")
    print(f"  ICIR        : {overall_icir:.3f}")
    print(f"  t-statistic : {t_stat:.3f}  (n={len(ic_series)} months)")

    # Rolling 6-month IC
    roll_mean = ic_series.rolling(6, min_periods=3).mean()
    roll_std  = ic_series.rolling(6, min_periods=3).std(ddof=1)

    # Regime split
    regime_results = {}
    if regime_col and regime_col in merged.columns:
        # Monthly median of regime proxy (one value per month)
        regime_monthly = (
            merged.groupby("date")[regime_col].median()
            .reindex(ic_series.index)
        )
        global_median = regime_monthly.median()
        risk_on_mask  = regime_monthly <= global_median
        risk_off_mask = regime_monthly >  global_median

        ic_risk_on  = ic_series[risk_on_mask].dropna()
        ic_risk_off = ic_series[risk_off_mask].dropna()

        def regime_stats(s: pd.Series, label: str) -> dict:
            if len(s) == 0:
                return {"regime": label, "n": 0, "ic": np.nan, "icir": np.nan}
            mu = s.mean(); sd = s.std(ddof=1)
            return {"regime": label, "n": len(s),
                    "ic": mu, "icir": mu / sd if sd > 0 else np.nan}

        regime_results["Risk-On"]  = regime_stats(ic_risk_on,  "Risk-On  (low vol)")
        regime_results["Risk-Off"] = regime_stats(ic_risk_off, "Risk-Off (high vol)")

        print(f"\n  Regime proxy: {regime_col} (median split = {global_median:.4f})")
        print(f"  {'Regime':<25} {'n':>5} {'IC':>8} {'ICIR':>8}")
        print(f"  {'-'*50}")
        for v in regime_results.values():
            print(f"  {v['regime']:<25} {v['n']:>5} {v['ic']:>8.4f} {v['icir']:>8.3f}")

    # Plot B

    n_rows = 2 if regime_results else 1
    fig_b, axes = plt.subplots(n_rows, 1,
                                figsize=(12, 4.5 * n_rows),
                                gridspec_kw={"hspace": 0.45})
    if n_rows == 1:
        axes = [axes]

    ax0 = axes[0]
    # Bar chart of monthly IC
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ic_series.values]
    ax0.bar(ic_series.index, ic_series.values, color=bar_colors,
            alpha=0.35, width=20, label="Monthly IC")
    # Rolling mean ± std band
    ax0.plot(roll_mean.index, roll_mean.values,
             color="#1f77b4", linewidth=1.8, label="6m rolling mean IC")
    ax0.fill_between(roll_mean.index,
                     (roll_mean - roll_std).values,
                     (roll_mean + roll_std).values,
                     color="#1f77b4", alpha=0.15, label="±1 std")
    ax0.axhline(0,           color="black",   linewidth=0.8, linestyle="--")
    ax0.axhline(overall_ic,  color="#ff7f0e", linewidth=1.2, linestyle=":",
                label=f"Mean IC = {overall_ic:.4f}")
    ax0.set_title(
        f"CS-Transformer - IC Stability (ICIR = {overall_icir:.3f}, "
        f"t = {t_stat:.2f}, n = {len(ic_series)})",
        fontsize=11, fontweight="bold"
    )
    ax0.set_ylabel("IC (Pearson)", fontsize=10)
    ax0.legend(fontsize=9, loc="upper left")
    ax0.grid(axis="y", linestyle=":", alpha=0.4)
    ax0.tick_params(axis="x", rotation=30)

    if regime_results and len(axes) > 1:
        ax1 = axes[1]
        reg_labels = list(regime_results.keys())
        reg_ic     = [regime_results[k]["ic"]   for k in reg_labels]
        reg_icir   = [regime_results[k]["icir"] for k in reg_labels]
        reg_n      = [regime_results[k]["n"]    for k in reg_labels]

        x = np.arange(len(reg_labels))
        w = 0.35
        bars1 = ax1.bar(x - w/2, reg_ic,   width=w, color="#1f77b4", alpha=0.8, label="Mean IC")
        bars2 = ax1.bar(x + w/2, reg_icir, width=w, color="#ff7f0e", alpha=0.8, label="ICIR")

        # Annotate with n months
        for i, (bar, n) in enumerate(zip(bars1, reg_n)):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"n={n}", ha="center", va="bottom", fontsize=8)

        ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax1.set_xticks(x)
        ax1.set_xticklabels(reg_labels, fontsize=10)
        ax1.set_title("IC by Market Regime  (vol proxy: " + regime_col + ")",
                      fontsize=11, fontweight="bold")
        ax1.set_ylabel("IC / ICIR", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", linestyle=":", alpha=0.4)

    fig_b.savefig(OUT_IC_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print(f"\nSaved: {OUT_IC_PNG}")


# Summary

print("\n" + "=" * 65)
print("CS-TRANSFORMER AUDIT COMPLETE")
print("=" * 65)
print(f"  Loadings chart  : {OUT_LOADINGS_PNG}")
print(f"  Loadings CSV    : {OUT_LOADINGS_CSV}")
print(f"  IC stability    : {OUT_IC_PNG}")
print()

top3 = loading_agg.head(3)
print("Top 3 factors the CS-T signal most resembles:")
for _, row in top3.iterrows():
    direction = "positively" if row["mean_corr"] > 0 else "negatively"
    print(f"  {row['factor']:<25}  ρ = {row['mean_corr']:+.4f} ± {row['std_corr']:.4f}  ({direction} correlated)")
print()
print("Interpretation guide:")
print("  High |ρ| → CS-T score strongly resembles that factor's cross-sectional ranking.")
print("  Near-zero ρ for all factors → CS-T is learning a genuinely novel combination.")
print("  Dominant single factor → CS-T may be largely a complex re-implementation of that factor.")
