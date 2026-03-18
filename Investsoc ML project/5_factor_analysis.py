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
import pyarrow.parquet as pq
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
FIG_DIR  = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PANEL_IN = DATA_DIR / "panel_monthly_enriched.parquet"

MAX_IC_DECAY_LAGS = 60    # compute IC at lags 0–60 months ahead (~5 years)
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

def plot_ic_summary(summary_df: pd.DataFrame, out_dir: Path, top_n: int = 10):
    """
    Two separate horizontal bar charts:
      - Top N by ICIR  (positive signal — buy these)
      - Bottom N by ICIR (negative signal — short these / use inverted)
    """
    df = summary_df.dropna(subset=["icir"]).sort_values("icir", ascending=False)

    top_pos = df.head(top_n).sort_values("icir")          # highest positive ICIR
    top_neg = df.tail(top_n).sort_values("icir")          # most negative ICIR

    for subset, label, color, fname in [
        (top_pos, f"Top {top_n} — Positive Signal (buy high)",  "#2ecc71", "factor_ic_summary_positive.png"),
        (top_neg, f"Bottom {top_n} — Negative Signal (buy low)", "#e74c3c", "factor_ic_summary_negative.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, max(5, len(subset) * 0.45)))
        ax.barh(range(len(subset)), subset["icir"], color=color, alpha=0.85)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(subset["factor"], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("ICIR (mean IC / std IC)")
        ax.set_title(label)
        plt.tight_layout()
        out_path = out_dir / fname
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


# Market regimes (COVID crash Mar-May 2020 excluded — black swan, stress test handles it)
REGIMES = {
    "qe_bull":        ("2010-01-01", "2019-12-31"),   # QE-era bull market
    "covid_recovery": ("2020-06-01", "2021-12-31"),   # Stimulus-driven rally
    "rate_hike_bear": ("2022-01-01", "2022-12-31"),   # Fed hiking cycle bear
    "ai_bull":        ("2023-01-01", "2099-12-31"),   # AI-driven bull (current)
}
COVID_EXCLUDE = ("2020-03-01", "2020-05-31")          # excluded from all regimes
REGIME_PASS_THRESHOLD = 0.75   # must have positive IC in ≥ 75% of regimes to pass


def regime_stability(panel: pd.DataFrame, factor: str) -> dict:
    """
    Compute mean IC for each named market regime (COVID period excluded).
    A factor is 'regime_stable' if IC > 0 in >= REGIME_PASS_THRESHOLD of regimes.
    Returns dict with per-regime IC + pass/fail flag.
    """
    covid_start = pd.Timestamp(COVID_EXCLUDE[0])
    covid_end   = pd.Timestamp(COVID_EXCLUDE[1])
    panel_clean = panel[~((panel["date"] >= covid_start) &
                          (panel["date"] <= covid_end))].copy()

    results = {}
    regime_ics = {}
    for name, (start, end) in REGIMES.items():
        mask = (panel_clean["date"] >= pd.Timestamp(start)) & \
               (panel_clean["date"] <= pd.Timestamp(end))
        sub = panel_clean[mask]
        if sub.empty or sub["date"].nunique() < 6:
            results[f"ic_{name}"] = np.nan
            continue
        ic_vals = []
        for _, grp in sub.groupby("date"):
            vals = grp[[factor, "fwd_ret_1m"]].dropna()
            if len(vals) < 50:
                continue
            corr, _ = spearmanr(vals[factor], vals["fwd_ret_1m"])
            if not np.isnan(corr):
                ic_vals.append(corr)
        mean_ic = float(np.mean(ic_vals)) if ic_vals else np.nan
        results[f"ic_{name}"] = round(mean_ic, 4) if not np.isnan(mean_ic) else np.nan
        regime_ics[name] = mean_ic

    # Sign consistency: stable if IC sign is same direction in ≥ 75% of valid regimes
    # (consistently positive OR consistently negative — both are usable signals)
    valid_ics = [v for v in regime_ics.values() if not np.isnan(v)]
    n_valid = len(valid_ics)
    if n_valid > 0:
        n_positive = sum(1 for v in valid_ics if v > 0)
        n_negative = n_valid - n_positive
        n_consistent = max(n_positive, n_negative)   # how many agree on direction
        majority_sign = "+" if n_positive >= n_negative else "-"
        results["regime_stable"] = (n_consistent / n_valid >= REGIME_PASS_THRESHOLD)
        results["regimes_positive"] = f"{n_positive}/{n_valid}"
        results["majority_sign"]    = majority_sign
    else:
        results["regime_stable"]    = False
        results["regimes_positive"] = "0/0"
        results["majority_sign"]    = "?"
    return results


def factor_turnover(panel: pd.DataFrame, factor: str) -> float:
    """
    Turnover = average % of stocks that change quintile month-to-month.
    Low turnover (< 20%) is ideal — means the signal is stable and cheap to trade.
    High turnover (> 50%) means excessive rebalancing costs would kill real-world returns.
    """
    sub = panel[["date", "ticker", factor]].dropna().copy()
    sub["quintile"] = sub.groupby("date")[factor].transform(
        lambda x: pd.qcut(x, N_QUINTILES, labels=False, duplicates="drop")
    )
    sub = sub.sort_values(["ticker", "date"])
    sub["prev_q"] = sub.groupby("ticker")["quintile"].shift(1)
    sub = sub.dropna(subset=["prev_q"])
    sub["changed"] = sub["quintile"] != sub["prev_q"]
    return float(sub.groupby("date")["changed"].mean().mean())


def ras_permutation_test(panel: pd.DataFrame, factor: str,
                         n_permutations: int = 100) -> tuple[float, float]:
    """
    RAS (Random Asset Shuffling) permutation test.
    Randomly shuffles factor values cross-sectionally N times.
    p-value = fraction of random runs that beat the real spread.
    Low p-value (< 0.05) = factor is genuinely predictive, not noise.
    """
    qt_real = quintile_backtest(panel, factor)
    if qt_real.empty:
        return np.nan, np.nan
    real_spread = qt_real["spread"].mean() * 12 * 100   # annualised %

    rng = np.random.default_rng(seed=42)
    perm_spreads = []
    perm_panel = panel[["date", "ticker", factor, "fwd_ret_1m"]].copy()
    for _ in range(n_permutations):
        # Shuffle factor within each month (preserves cross-date structure)
        perm_panel[factor] = (perm_panel.groupby("date")[factor]
                              .transform(lambda x: rng.permutation(x.values)))
        qt_p = quintile_backtest(perm_panel, factor)
        perm_spreads.append(qt_p["spread"].mean() * 12 * 100 if not qt_p.empty else 0.0)

    perm_arr = np.array(perm_spreads)
    # p-value: fraction of permutations as extreme as real (two-sided by magnitude)
    p_val = float((np.abs(perm_arr) >= abs(real_spread)).mean())
    return real_spread, p_val


def plot_quintile_bars(quint_df: pd.DataFrame, out_path: Path):
    """Bar chart showing Q1–Q5 mean annualised returns for top factors."""
    q_cols = [f"Q{q}_ann_pct" for q in range(1, N_QUINTILES + 1)]
    factors = quint_df["factor"].tolist()
    n = len(factors)
    if n == 0:
        return

    x = np.arange(N_QUINTILES)
    width = 0.7 / n
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("tab10")

    for i, row in quint_df.iterrows():
        fac = row["factor"]
        vals = [row.get(c, np.nan) for c in q_cols]
        offset = (list(quint_df["factor"]).index(fac) - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=fac, color=cmap(list(quint_df["factor"]).index(fac) % 10),
                      alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q}\n(low→high)" if q == 1 else f"Q{q}" for q in range(1, N_QUINTILES + 1)])
    ax.set_ylabel("Ann. Return (%)")
    ax.set_title("Quintile Backtest — Mean Annualised Return per Bucket\n(Q5=top factor score, Q1=bottom)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
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
    panel = pq.read_table(PANEL_IN).to_pandas()
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

    # ── Filter 1: |IC| > 0 — keep any directional signal (positive or negative) ──
    nonzero_ic = summary_df[summary_df["ic_mean"].abs() > 0].copy()
    print(f"\nFilter 1 — |IC| > 0 (both directions kept): {len(nonzero_ic)} / {len(summary_df)} kept")
    print(f"  Positive IC: {nonzero_ic[nonzero_ic['ic_mean'] > 0]['factor'].tolist()}")
    print(f"  Negative IC (contrarian): {nonzero_ic[nonzero_ic['ic_mean'] < 0]['factor'].tolist()}")

    # ── Filter 2: Regime stability ────────────────────────────────────────────────
    # COVID crash (Mar–May 2020) excluded as black swan; stress test handles tail events.
    # Regimes: QE bull (2010–2019), COVID recovery (Jun 2020–2021),
    #          rate hike bear (2022), AI bull (2023–present)
    # Pass: IC > 0 in ≥ 75% of regimes.
    print(f"\nFilter 2 — Regime stability (COVID Mar–May 2020 excluded) …")
    regime_records = []
    for i, fac in enumerate(summary_df["factor"].tolist(), 1):
        res = regime_stability(panel, fac)
        res["factor"] = fac
        regime_records.append(res)
        if i % 10 == 0 or i == len(summary_df):
            print(f"  {i}/{len(summary_df)} done")

    regime_df = pd.DataFrame(regime_records)
    regime_df = regime_df[["factor", "ic_qe_bull", "ic_covid_recovery",
                            "ic_rate_hike_bear", "ic_ai_bull",
                            "regimes_positive", "majority_sign", "regime_stable"]]
    regime_df.to_csv(DATA_DIR / "factor_regime_stability.csv", index=False)
    print(f"Saved → factor_regime_stability.csv")

    # Print regime stability table for all factors
    print(f"\n{'─'*85}")
    print(f"REGIME STABILITY  (COVID Mar–May 2020 excluded)")
    print(f"{'─'*85}")
    print(f"{'Factor':<26} {'QE bull':>9} {'COVID rec':>10} {'Rate bear':>10} "
          f"{'AI bull':>9} {'Pass':>6} {'Stable':>7}")
    print(f"{'─'*85}")
    for _, row in regime_df.sort_values("ic_qe_bull", ascending=False).iterrows():
        stable_str = "✓" if row["regime_stable"] else "✗"
        print(f"  {row['factor']:<24} "
              f"{row.get('ic_qe_bull', np.nan):>+9.4f} "
              f"{row.get('ic_covid_recovery', np.nan):>+10.4f} "
              f"{row.get('ic_rate_hike_bear', np.nan):>+10.4f} "
              f"{row.get('ic_ai_bull', np.nan):>+9.4f} "
              f"{str(row['regimes_positive']):>6}  {stable_str}")

    # ── Combined filter: |IC| > 0 AND sign consistent across regimes ────────────
    stable_factors = set(regime_df[regime_df["regime_stable"]]["factor"].tolist())
    both_filters = nonzero_ic[nonzero_ic["factor"].isin(stable_factors)].copy()
    # Sort by |ICIR| descending
    both_filters = both_filters.reindex(
        both_filters["icir"].abs().sort_values(ascending=False).index)
    print(f"\nFinal filter (|IC|>0 + sign-consistent across regimes): "
          f"{len(both_filters)} / {len(summary_df)} factors kept")
    kept_pos = both_filters[both_filters["ic_mean"] > 0]["factor"].tolist()
    kept_neg = both_filters[both_filters["ic_mean"] < 0]["factor"].tolist()
    print(f"  Kept positive IC: {kept_pos}")
    print(f"  Kept negative IC (contrarian): {kept_neg}")
    dropped_regime = nonzero_ic[~nonzero_ic["factor"].isin(stable_factors)]["factor"].tolist()
    print(f"  Dropped (sign unstable across regimes): {dropped_regime}")

    # ── IC Decay ────────────────────────────────────────────────────────────────
    # Run decay on top 10 positive IC + top 10 negative IC (by |ICIR|)
    top10_pos = both_filters[both_filters["ic_mean"] > 0].head(10)["factor"].tolist()
    top10_neg = both_filters[both_filters["ic_mean"] < 0].head(10)["factor"].tolist()
    top20 = top10_pos + top10_neg
    print(f"\nComputing IC decay (lags 0–{MAX_IC_DECAY_LAGS}) for top 10 positive + top 10 negative factors …")
    decay_records = {}
    for i, fac in enumerate(top20, 1):
        decay_records[fac] = ic_decay_series(panel, fac)
        print(f"  {i}/{len(top20)}: {fac}")

    decay_df = pd.DataFrame(decay_records).T
    decay_df.index.name = "factor"
    decay_df.columns = list(range(MAX_IC_DECAY_LAGS + 1))
    decay_df.to_csv(DATA_DIR / "factor_ic_decay.csv")
    print(f"Saved → factor_ic_decay.csv")

    # ── Quintile Backtest + Turnover + RAS ───────────────────────────────────────
    print(f"\nRunning quintile backtests, turnover & RAS tests for top 20 positive-IC factors …")
    print(f"  (RAS uses 100 random permutations per factor — takes ~2 min)")
    quintile_records = []
    quintile_frames  = {}
    for i, fac in enumerate(top20, 1):
        qt = quintile_backtest(panel, fac)
        if qt.empty:
            continue
        spread = qt["spread"].dropna()
        ann_spread = spread.mean() * 12 * 100
        sharpe_spread = (spread.mean() / spread.std(ddof=1) * np.sqrt(12)
                         if spread.std(ddof=1) > 0 else np.nan)
        # Mean annualised return per quintile
        q_means = {f"Q{q}_ann_pct": round(qt[f"Q{q}"].mean() * 12 * 100, 2)
                   for q in range(1, N_QUINTILES + 1) if f"Q{q}" in qt.columns}
        # Monotonicity: Q5 > Q4 > Q3 > Q2 > Q1
        q_vals = [q_means.get(f"Q{q}_ann_pct", np.nan) for q in range(1, N_QUINTILES + 1)]
        monotonic = all(q_vals[q] > q_vals[q-1] for q in range(1, N_QUINTILES)
                        if not np.isnan(q_vals[q]) and not np.isnan(q_vals[q-1]))
        # Turnover: % of stocks changing quintile each month (lower = cheaper to trade)
        turnover = factor_turnover(panel, fac)
        # RAS permutation test: is the spread real or just noise?
        _, ras_pval = ras_permutation_test(panel, fac, n_permutations=100)
        quintile_records.append({
            "factor":               fac,
            **q_means,
            "ann_q5_q1_spread_pct": round(ann_spread, 2),
            "spread_sharpe":        round(sharpe_spread, 3) if not np.isnan(sharpe_spread) else np.nan,
            "monotonic":            monotonic,
            "turnover_pct":         round(turnover * 100, 1),
            "ras_p_value":          round(ras_pval, 3) if not np.isnan(ras_pval) else np.nan,
            "ras_significant":      ras_pval < 0.05 if not np.isnan(ras_pval) else False,
            "n_months":             len(spread),
        })
        quintile_frames[fac] = qt
        sig_str = "✓ real" if ras_pval < 0.05 else "✗ noise"
        print(f"  {i}/{len(top20)}: {fac:30s}  "
              f"Spread={ann_spread:+.1f}%  Mono={'✓' if monotonic else '✗'}  "
              f"Turnover={turnover*100:.0f}%  RAS p={ras_pval:.2f} [{sig_str}]")

    quint_df = pd.DataFrame(quintile_records).sort_values("ann_q5_q1_spread_pct",
                                                           ascending=False)
    quint_df.to_csv(DATA_DIR / "factor_quintile_returns.csv", index=False)
    print(f"Saved → factor_quintile_returns.csv")

    # ── Quintile bar chart (top 10 by |spread|) ──────────────────────────────────
    plot_quintile_bars(quint_df.head(10), FIG_DIR / "factor_quintile_plot.png")

    # ── Plots ────────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_ic_summary(summary_df, FIG_DIR, top_n=10)   # saves _positive.png + _negative.png
    plot_ic_decay(decay_df, top10_pos, FIG_DIR / "factor_ic_decay_positive.png")
    plot_ic_decay(decay_df, top10_neg, FIG_DIR / "factor_ic_decay_negative.png")

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
    print(f"    factor_quintile_returns.csv  ← Q1–Q5 returns, monotonicity, turnover, RAS p-value")
    print(f"    factor_quintile_plot.png          ← bar chart Q1–Q5 for top 10 factors")
    print(f"    factor_ic_summary_positive.png   ← top 10 positive IC factors")
    print(f"    factor_ic_summary_negative.png   ← top 10 negative IC (contrarian) factors")
    print(f"    factor_ic_decay_positive.png     ← decay curves for top 10 positive IC")
    print(f"    factor_ic_decay_negative.png     ← decay curves for top 10 negative IC")
    print(f"{'='*65}")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
