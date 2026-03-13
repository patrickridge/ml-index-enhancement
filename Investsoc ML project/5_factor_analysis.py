"""
5_factor_analysis.py — Per-Factor IC Analysis and Quantile Backtests
=====================================================================
Evaluates how predictive each of the 80 features is vs next-month returns.
This is about PREDICTIVE POWER, not feature redundancy (that's 7_factor_diagnostics.py).

Metrics computed per factor:
  IC       — Spearman rank correlation of factor vs fwd_ret_1m, averaged across months
  ICIR     — IC Information Ratio: mean(IC) / std(IC)  (consistency of signal)
  t-stat   — statistical significance: mean(IC) / (std(IC) / sqrt(N_months))
  % pos    — fraction of months where IC > 0  (directional consistency)
  IC decay — IC at lags 1–6 months ahead (signal persistence)
  Quintile spread — mean return spread between top and bottom quintile stocks

Interpretation:
  IC   > 0.02  : weak but meaningful signal
  IC   > 0.05  : good signal
  ICIR > 0.5   : consistent signal (analogous to IR > 0.5 for portfolios)
  t-stat > 2.0 : statistically significant at ~5% level
  IC decay     : a factor with IC that drops quickly has a short signal half-life

Outputs (all to data/):
  factor_ic_summary.csv        — ranked table of all factors
  factor_ic_decay.csv          — IC at lags 0–6 for each factor
  factor_quintile_returns.csv  — quintile spread per factor
  factor_ic_summary.png        — bar chart of top/bottom 20 by ICIR
  factor_ic_decay_top10.png    — IC decay curves for top 10 factors

Run AFTER 1_feature_engineering.py.
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

try:
    from config import MACRO_COLS, DATA_DIR as _cfg_dir
    _DATA_DIR = _cfg_dir
except ImportError:
    MACRO_COLS = []
    _DATA_DIR  = Path("data")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
PANEL_IN = DATA_DIR / "panel_monthly_enriched.parquet"

MAX_IC_DECAY_LAGS = 6     # compute IC at lags 0–6 months ahead
N_QUINTILES       = 5     # quintile backtest buckets
MIN_STOCKS        = 20    # skip month if fewer stocks


# ═══════════════════════════════════════════════════════════════════════════════
# IC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def monthly_ic(panel: pd.DataFrame, factor: str, target: str = "fwd_ret_1m") -> pd.Series:
    """
    Compute Spearman IC for one factor, one IC per calendar month.
    Returns a Series indexed by date.
    """
    results = {}
    for dt, grp in panel.groupby("date"):
        sub = grp[[factor, target]].dropna()
        if len(sub) < MIN_STOCKS:
            continue
        corr, _ = spearmanr(sub[factor], sub[target])
        if not np.isnan(corr):
            results[dt] = corr
    return pd.Series(results, name=factor)


def ic_summary(ic_series: pd.Series) -> dict:
    """Compute IC statistics from a monthly IC series."""
    ic = ic_series.dropna()
    n  = len(ic)
    if n < 3:
        return dict(ic_mean=np.nan, ic_std=np.nan, icir=np.nan,
                    t_stat=np.nan, pct_positive=np.nan, n_months=n)
    mu   = ic.mean()
    sig  = ic.std(ddof=1)
    icir = mu / sig if sig > 1e-8 else np.nan
    tstat = mu / (sig / np.sqrt(n)) if sig > 1e-8 else np.nan
    return dict(
        ic_mean     = round(float(mu),   4),
        ic_std      = round(float(sig),  4),
        icir        = round(float(icir), 4) if not np.isnan(icir) else np.nan,
        t_stat      = round(float(tstat),3) if not np.isnan(tstat) else np.nan,
        pct_positive= round((ic > 0).mean(), 4),
        n_months    = n,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUINTILE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def quintile_backtest(panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    Each month, sort stocks by factor into N_QUINTILES buckets.
    Compute equal-weight return per bucket per month.
    Returns DataFrame: index=date, columns=Q1..Q5 and 'spread' (Q5 - Q1).
    Assumes higher factor score = better (long Q5, short Q1).
    """
    records = []
    for dt, grp in panel.groupby("date"):
        sub = grp[[factor, "fwd_ret_1m"]].dropna()
        if len(sub) < N_QUINTILES * 4:
            continue
        sub = sub.copy()
        sub["quintile"] = pd.qcut(sub[factor], N_QUINTILES,
                                   labels=list(range(1, N_QUINTILES + 1)),
                                   duplicates="drop")
        q_rets = sub.groupby("quintile")["fwd_ret_1m"].mean()
        row = {"date": dt}
        for q in range(1, N_QUINTILES + 1):
            row[f"Q{q}"] = q_rets.get(q, np.nan)
        # Spread: Q5 (top) - Q1 (bottom)
        if f"Q{N_QUINTILES}" in row and "Q1" in row:
            row["spread"] = row[f"Q{N_QUINTILES}"] - row["Q1"]
        records.append(row)
    return pd.DataFrame(records).set_index("date")


# ═══════════════════════════════════════════════════════════════════════════════
# IC DECAY
# ═══════════════════════════════════════════════════════════════════════════════

def ic_decay_series(panel: pd.DataFrame, factor: str) -> dict:
    """
    IC at lag 0 (standard), 1, 2, ..., MAX_IC_DECAY_LAGS months ahead.
    Lag k: Spearman(factor at t, return at t+k).
    Uses the forward return at t+k by shifting the target column.
    Returns dict {lag: mean_ic}.
    """
    dates = sorted(panel["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}

    decay = {}
    for lag in range(MAX_IC_DECAY_LAGS + 1):
        ics = []
        for dt, grp in panel.groupby("date"):
            didx = date_to_idx[dt]
            future_idx = didx + lag
            if future_idx >= len(dates):
                continue
            future_dt = dates[future_idx]
            # Factor at t, return at t+lag
            factor_vals = grp[["ticker", factor]].dropna().set_index("ticker")[factor]
            ret_vals    = (panel[panel["date"] == future_dt]
                           [["ticker", "fwd_ret_1m"]].dropna()
                           .set_index("ticker")["fwd_ret_1m"])
            common = factor_vals.index.intersection(ret_vals.index)
            if len(common) < MIN_STOCKS:
                continue
            corr, _ = spearmanr(factor_vals[common], ret_vals[common])
            if not np.isnan(corr):
                ics.append(corr)
        decay[lag] = round(float(np.mean(ics)), 4) if ics else np.nan
    return decay


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ic_summary(summary_df: pd.DataFrame, out_path: Path, top_n: int = 20):
    """Horizontal bar chart of ICIR for top/bottom N factors."""
    df = summary_df.dropna(subset=["icir"]).sort_values("icir")
    n  = len(df)

    # Take top N and bottom N
    show = pd.concat([df.head(top_n), df.tail(top_n)]).drop_duplicates()
    show = show.sort_values("icir")

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in show["icir"]]
    fig, ax = plt.subplots(figsize=(10, max(8, len(show) * 0.28)))
    bars = ax.barh(range(len(show)), show["icir"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["factor"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 0.5, color="green", linewidth=0.8, linestyle="--", label="ICIR=0.5 (good)")
    ax.axvline(-0.5, color="red",   linewidth=0.8, linestyle="--")
    ax.set_xlabel("IC Information Ratio (ICIR = mean IC / std IC)")
    ax.set_title(f"Factor ICIR — Top/Bottom {top_n} Factors\n"
                 f"(ICIR > 0.5 = consistent positive signal, < -0.5 = consistent negative)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_ic_decay(decay_df: pd.DataFrame, factors: list, out_path: Path):
    """IC decay curves for a list of factors."""
    lags = [c for c in decay_df.columns if isinstance(c, int)]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, fac in enumerate(factors[:10]):
        if fac not in decay_df.index:
            continue
        vals = decay_df.loc[fac, lags].values
        ax.plot(lags, vals, marker="o", label=fac, color=cmap(i), linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (months ahead)")
    ax.set_ylabel("Mean IC (Spearman)")
    ax.set_title("IC Decay — Top 10 Factors by |ICIR|")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("FACTOR ANALYSIS — Predictive IC and Quantile Backtests")
    print("=" * 65)

    # ── Load panel ───────────────────────────────────────────────────────────────
    print(f"\nLoading {PANEL_IN} …")
    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= "2010-01-01"].reset_index(drop=True)

    always_exclude = {"date", "ticker", "fwd_ret_1m"}
    macro_in_panel = [c for c in MACRO_COLS if c in panel.columns]
    feat_cols = [c for c in panel.columns
                 if c not in always_exclude and c not in macro_in_panel]

    n_months = panel["date"].nunique()
    n_tickers = panel["ticker"].nunique()
    print(f"  {len(feat_cols)} features | {n_months} months | {n_tickers} tickers")
    if macro_in_panel:
        print(f"  Macro features skipped (not cross-sectional): {macro_in_panel}")

    # ── Per-factor IC ────────────────────────────────────────────────────────────
    print(f"\nComputing monthly IC for {len(feat_cols)} features …")
    ic_records = []
    ic_series_all = {}

    for i, fac in enumerate(feat_cols, 1):
        ic_s = monthly_ic(panel, fac)
        ic_series_all[fac] = ic_s
        stats = ic_summary(ic_s)
        stats["factor"] = fac
        ic_records.append(stats)
        if i % 20 == 0 or i == len(feat_cols):
            print(f"  {i}/{len(feat_cols)} done")

    summary_df = (pd.DataFrame(ic_records)
                  [["factor", "ic_mean", "ic_std", "icir", "t_stat",
                    "pct_positive", "n_months"]]
                  .sort_values("icir", ascending=False, key=abs)
                  .reset_index(drop=True))

    summary_df.to_csv(DATA_DIR / "factor_ic_summary.csv", index=False)
    print(f"\nSaved → factor_ic_summary.csv")

    # ── Print top/bottom factors ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("TOP 15 FACTORS by |ICIR|")
    print(f"{'─'*65}")
    print(f"{'Factor':<28} {'IC':>7} {'ICIR':>7} {'t-stat':>7} {'%pos':>6}")
    print(f"{'─'*65}")
    for _, row in summary_df.head(15).iterrows():
        sign = "+" if row["ic_mean"] > 0 else "-"
        print(f"  {row['factor']:<26} {row['ic_mean']:>+7.4f}  "
              f"{row['icir']:>6.3f}  {row['t_stat']:>6.2f}  {row['pct_positive']*100:>5.1f}%")

    print(f"\n{'─'*65}")
    print("BOTTOM 10 FACTORS (weakest / reversed signal)")
    print(f"{'─'*65}")
    for _, row in summary_df.tail(10).sort_values("icir").iterrows():
        print(f"  {row['factor']:<26} {row['ic_mean']:>+7.4f}  "
              f"{row['icir']:>6.3f}  {row['t_stat']:>6.2f}  {row['pct_positive']*100:>5.1f}%")

    sig_positive = summary_df[(summary_df["icir"] > 0.3) & (summary_df["t_stat"] > 2)]["factor"].tolist()
    sig_negative = summary_df[(summary_df["icir"] < -0.3) & (summary_df["t_stat"].abs() > 2)]["factor"].tolist()
    print(f"\nSignificant predictors (|ICIR|>0.3, |t|>2): {len(sig_positive)+len(sig_negative)}")
    print(f"  Positive IC: {sig_positive}")
    print(f"  Negative IC (contrarian): {sig_negative}")

    # ── IC Decay ────────────────────────────────────────────────────────────────
    print(f"\nComputing IC decay (lags 0–{MAX_IC_DECAY_LAGS}) for top 20 factors …")
    top20 = summary_df.head(20)["factor"].tolist()
    decay_records = {}
    for i, fac in enumerate(top20, 1):
        decay_records[fac] = ic_decay_series(panel, fac)
        print(f"  {i}/{len(top20)}: {fac}")

    decay_df = pd.DataFrame(decay_records).T
    decay_df.index.name = "factor"
    decay_df.columns = list(range(MAX_IC_DECAY_LAGS + 1))
    decay_df.to_csv(DATA_DIR / "factor_ic_decay.csv")
    print(f"Saved → factor_ic_decay.csv")

    # ── Quintile Backtest ────────────────────────────────────────────────────────
    print(f"\nRunning quintile backtests for top 20 factors …")
    quintile_records = []
    for i, fac in enumerate(top20, 1):
        qt = quintile_backtest(panel, fac)
        if qt.empty:
            continue
        spread = qt["spread"].dropna()
        ann_spread = spread.mean() * 12 * 100   # annualised in %
        sharpe_spread = (spread.mean() / spread.std(ddof=1) * np.sqrt(12)
                         if spread.std(ddof=1) > 0 else np.nan)
        quintile_records.append({
            "factor":         fac,
            "ann_q5_q1_spread_pct": round(ann_spread, 2),
            "spread_sharpe":  round(sharpe_spread, 3) if not np.isnan(sharpe_spread) else np.nan,
            "n_months":       len(spread),
        })
        print(f"  {i}/{len(top20)}: {fac:30s}  "
              f"Ann spread={ann_spread:+.1f}%  Sharpe={sharpe_spread:.2f}")

    quint_df = pd.DataFrame(quintile_records).sort_values("ann_q5_q1_spread_pct",
                                                           ascending=False)
    quint_df.to_csv(DATA_DIR / "factor_quintile_returns.csv", index=False)
    print(f"Saved → factor_quintile_returns.csv")

    # ── Plots ────────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_ic_summary(summary_df, DATA_DIR / "factor_ic_summary.png", top_n=20)
    plot_ic_decay(decay_df, top20[:10], DATA_DIR / "factor_ic_decay_top10.png")

    # ── Final summary table ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FACTOR ANALYSIS COMPLETE")
    print(f"{'='*65}")
    print(f"  Features analysed:  {len(feat_cols)}")
    print(f"  Months:             {n_months}  ({panel['date'].min().date()} → {panel['date'].max().date()})")
    print(f"  Significant factors (|ICIR|>0.3, |t|>2): "
          f"{len(sig_positive)+len(sig_negative)} / {len(feat_cols)}")
    print(f"\n  Files saved to data/:")
    print(f"    factor_ic_summary.csv        ← ranked table of all factors")
    print(f"    factor_ic_decay.csv          ← IC persistence at lags 0–{MAX_IC_DECAY_LAGS}")
    print(f"    factor_quintile_returns.csv  ← Q5–Q1 spread per factor")
    print(f"    factor_ic_summary.png        ← ICIR bar chart")
    print(f"    factor_ic_decay_top10.png    ← decay curves for top 10")
    print(f"{'='*65}")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
