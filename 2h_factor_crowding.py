"""
2h_factor_crowding.py — Factor Crowding, Decay & Contrarian Diagnostic
=======================================================================
Addresses Kieran's four concerns:

  1. FACTOR CROWDING — correlation matrix (IS data only), cluster highly
     correlated factors (r > 0.70), flag redundant pairs to prune before retraining.

  2. 7-DAY DECAY — within-month IC decay at days 1/3/7/10 from existing
     factor_ic_decay_daily.csv.  "Do we need a 7-day refresh or monthly is fine?"

  3. CONTRARIAN FACTORS — IS vs OOS for the sign-flip group.
     Are short-term reversal signals actually useful contrarian bets or noise?

  4. QUINTILE DIRECTION AUDIT — flag factors where Q5-Q1 sign contradicts IC_IS.
     Shows the 'wrong direction' factors Kieran spotted.

STRICT IS / OOS SEPARATION:
  - All training metrics  → data ≤ TRAIN_END  (from config.py)
  - All OOS metrics       → data >  TRAIN_END
  - Never mixed.

Outputs:
  data/crowding_corr_matrix.csv       — pairwise IC correlation matrix (IS)
  data/crowding_redundant_pairs.csv   — pairs with |r| > 0.70
  data/crowding_7day_decay.csv        — day_1 / day_3 / day_7 IC per factor
  data/crowding_contrarian_oos.csv    — IS vs OOS for sign-flip factors
  data/crowding_quintile_audit.csv    — quintile direction audit
  figures/crowding_corr_heatmap.png
  figures/crowding_7day_decay.png
  figures/crowding_contrarian_oos.png
  figures/crowding_quintile_audit.png
"""

import warnings
warnings.filterwarnings("ignore")

import time as _time
_t0 = _time.time()

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import DATA_DIR as _cfg_dir, TRAIN_END, VALID_END, MACRO_COLS
    DATA_DIR = Path(_cfg_dir)
except ImportError:
    DATA_DIR   = Path("data")
    TRAIN_END  = "2022-12-31"
    VALID_END  = "2024-06-30"
    MACRO_COLS = []

FIG_DIR  = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_STOCKS       = 20
CROWD_THRESHOLD  = 0.70   # |r| above this = crowded pair
PANEL_PATH       = DATA_DIR / "panel_monthly_enriched.parquet"

NON_FEATURE_COLS = {
    "date", "ticker", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("FACTOR CROWDING & DECAY DIAGNOSTIC")
print("=" * 65)
print(f"\nLoading panel → {PANEL_PATH}")

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])

# Strict IS / OOS split — never mix
IS_END = pd.Timestamp(TRAIN_END)
OOS_START = IS_END + pd.DateOffset(days=1)

panel_is  = panel[panel["date"] <= IS_END].copy()
panel_oos = panel[panel["date"] >= OOS_START].copy()

print(f"  IS  : {panel_is['date'].min().date()} → {panel_is['date'].max().date()}  "
      f"({len(panel_is):,} rows, {panel_is['date'].nunique()} months)")
print(f"  OOS : {panel_oos['date'].min().date()} → {panel_oos['date'].max().date()}  "
      f"({len(panel_oos):,} rows, {panel_oos['date'].nunique()} months)")

# Factor list (exclude macro — macro are same for all stocks within a month, zero CS variance)
all_factor_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
cs_factors = [c for c in all_factor_cols if c not in MACRO_COLS]
print(f"\n  CS factors : {len(cs_factors)}")
print(f"  Macro cols : {len(MACRO_COLS)}  (excluded from crowding — zero CS variance)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FACTOR CROWDING (IS DATA ONLY)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 1 — Factor Crowding (IS data only)")
print("─" * 65)

# Build monthly factor pivot: mean factor value per stock per month (IS only)
# Then compute pairwise Spearman correlation of the factor columns
# across the full IS panel (stock-months as observations)

def compute_factor_corr_matrix(panel_is, factors):
    """
    Pairwise Spearman correlation of factor values across stock-months (IS).
    Uses a pivot of mean factor per stock-month to be memory-efficient.
    """
    sub = panel_is[factors].copy()
    # Sample if very large to keep runtime reasonable (> 100k rows)
    if len(sub) > 60_000:
        sub = sub.sample(60_000, random_state=42)
    # Spearman = Pearson of ranks
    ranked = sub.rank(pct=True)
    corr = ranked.corr(method="pearson")
    return corr

print("  Computing pairwise factor correlation (IS)...")
corr_mat = compute_factor_corr_matrix(panel_is, cs_factors)
corr_mat.to_csv(DATA_DIR / "crowding_corr_matrix.csv")
print(f"  Saved → data/crowding_corr_matrix.csv  shape={corr_mat.shape}")

# Identify redundant pairs (|r| > threshold)
print(f"\n  Flagging pairs with |r| > {CROWD_THRESHOLD}...")
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
pairs = (upper.stack()
              .reset_index()
              .rename(columns={"level_0": "factor_a", "level_1": "factor_b", 0: "corr"}))
pairs["abs_corr"] = pairs["corr"].abs()
redundant = pairs[pairs["abs_corr"] > CROWD_THRESHOLD].sort_values("abs_corr", ascending=False)
redundant.to_csv(DATA_DIR / "crowding_redundant_pairs.csv", index=False)

print(f"  Redundant pairs found: {len(redundant)}")
print(f"  Top 15 most-crowded pairs:")
print(redundant.head(15).to_string(index=False))

# Hierarchical clustering heatmap
print("\n  Plotting correlation heatmap with clustering...")
try:
    dist = 1 - corr_mat.abs()
    np.fill_diagonal(dist.values, 0)
    dist_condensed = squareform(dist.values.clip(0))
    link = linkage(dist_condensed, method="ward")
    order = dendrogram(link, no_plot=True)["leaves"]

    mat_ordered = corr_mat.iloc[order, order]
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        mat_ordered, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=mat_ordered.columns,
        yticklabels=mat_ordered.index,
        square=True, linewidths=0,
        cbar_kws={"shrink": 0.6, "label": "Spearman ρ"},
    )
    ax.set_title(
        f"Factor Correlation Matrix — IS only (≤{TRAIN_END})\n"
        f"{len(cs_factors)} CS factors | Hierarchically clustered | Threshold = {CROWD_THRESHOLD}",
        fontsize=13, pad=12
    )
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", rotation=0,  labelsize=6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "crowding_corr_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved → figures/crowding_corr_heatmap.png")
except Exception as e:
    print(f"  [WARN] heatmap failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 7-DAY IC DECAY: IS vs OOS (computed from daily prices)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 2 — 7-Day IC Decay: IS vs OOS")
print("─" * 65)
print("  Computes IC at days 1/3/5/7/10 using daily prices for forward returns.")
print(f"  IS  = dates ≤ {TRAIN_END}  |  OOS = dates > {TRAIN_END}  — NEVER mixed.")

PRICES_PATH = Path("data/prices.parquet")
DAY_LAGS    = [1, 3, 5, 7, 10]

# Limit to factors with clear IS IC signal (avoid noise-dominated factors)
ic_sum_path = DATA_DIR / "factor_ic_summary.csv"
if ic_sum_path.exists():
    ic_sum_all = pd.read_csv(ic_sum_path, index_col="factor")
    # Pick top 40 factors by |ICIR| from IS, excluding macro
    top_factors_decay = (
        ic_sum_all["icir"].abs()
        .reindex([f for f in ic_sum_all.index if f in cs_factors])
        .dropna()
        .sort_values(ascending=False)
        .head(40)
        .index.tolist()
    )
else:
    top_factors_decay = cs_factors[:40]

print(f"  Analysing {len(top_factors_decay)} top factors by |ICIR_IS|")

if not PRICES_PATH.exists():
    print("  [WARN] data/prices.parquet not found — skipping Section 2")
    decay_7d_result = None
else:
    print("  Loading daily prices...")
    prices_raw = pd.read_parquet(PRICES_PATH, columns=["date", "ticker", "close"])
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    prices_raw = prices_raw.sort_values(["ticker", "date"])

    # Index daily prices by (ticker, date) for fast O(1) lookup
    prices_pivot = (
        prices_raw.pivot(index="date", columns="ticker", values="close")
        .sort_index()
    )
    all_trading_days = prices_pivot.index.values  # numpy array for searchsorted

    def get_fwd_return_kday(signal_dates, k):
        """
        For each month-end signal date, find the closing price k trading days later.
        Returns a DataFrame: index=signal_date, columns=ticker, values=k-day fwd return.
        Forward return = close[t+k] / close[t] - 1 (using month-end close as base).
        """
        fwd_rets = {}
        for sig_date in signal_dates:
            sig_ts = pd.Timestamp(sig_date)
            # Position of signal date in trading-day index
            pos = np.searchsorted(all_trading_days, sig_ts.to_datetime64(), side="right") - 1
            pos_fwd = pos + k
            if pos < 0 or pos_fwd >= len(all_trading_days):
                continue
            base_close = prices_pivot.iloc[pos]
            fwd_close  = prices_pivot.iloc[pos_fwd]
            ret = fwd_close / base_close - 1
            fwd_rets[sig_date] = ret
        return pd.DataFrame(fwd_rets).T  # index=date, cols=ticker

    def compute_decay_ic(panel_sub, factors, lags):
        """
        For each factor and each lag, compute mean IC across months in panel_sub.
        panel_sub is already restricted to IS or OOS — no mixing.
        Returns dict: {factor: {lag: mean_ic}}
        """
        signal_dates = sorted(panel_sub["date"].unique())
        # Pre-build forward returns for each lag (avoids re-scanning prices)
        fwd_by_lag = {}
        for k in lags:
            fwd_by_lag[k] = get_fwd_return_kday(signal_dates, k)

        results = {f: {} for f in factors}
        for f in factors:
            if f not in panel_sub.columns:
                continue
            # Lag 0 = standard monthly IC (use fwd_ret_1m from panel)
            ic0_vals = []
            for dt, grp in panel_sub.groupby("date"):
                sub = grp[[f, "fwd_ret_1m"]].dropna()
                if len(sub) < MIN_STOCKS:
                    continue
                c, _ = spearmanr(sub[f], sub["fwd_ret_1m"])
                if not np.isnan(c):
                    ic0_vals.append(c)
            results[f][0] = np.mean(ic0_vals) if ic0_vals else np.nan

            # Lags 1-10: use daily forward returns
            for k in lags:
                fwd_df = fwd_by_lag[k]
                ic_vals = []
                for dt, grp in panel_sub.groupby("date"):
                    if dt not in fwd_df.index:
                        continue
                    factor_vals = grp[["ticker", f]].dropna().set_index("ticker")[f]
                    ret_row     = fwd_df.loc[dt].dropna()
                    common      = factor_vals.index.intersection(ret_row.index)
                    if len(common) < MIN_STOCKS:
                        continue
                    c, _ = spearmanr(factor_vals[common], ret_row[common])
                    if not np.isnan(c):
                        ic_vals.append(c)
                results[f][k] = np.mean(ic_vals) if ic_vals else np.nan

        return results

    print("  Computing IS 7-day decay IC...")
    is_decay  = compute_decay_ic(panel_is,  top_factors_decay, DAY_LAGS)
    print("  Computing OOS 7-day decay IC...")
    oos_decay = compute_decay_ic(panel_oos, top_factors_decay, DAY_LAGS)

    # Build tidy output DataFrame
    lag_labels = [0] + DAY_LAGS  # 0, 1, 3, 5, 7, 10
    records = []
    for f in top_factors_decay:
        row = {"factor": f}
        for lag in lag_labels:
            row[f"is_day{lag}"]  = is_decay.get(f, {}).get(lag, np.nan)
            row[f"oos_day{lag}"] = oos_decay.get(f, {}).get(lag, np.nan)
        # Decay ratio at day 7: OOS_IC_day7 / IS_IC_day7
        is_d7  = is_decay.get(f, {}).get(7, np.nan)
        oos_d7 = oos_decay.get(f, {}).get(7, np.nan)
        row["oos_decay_ratio_d7"] = round(oos_d7 / is_d7, 3) if (is_d7 and abs(is_d7) > 1e-4) else np.nan
        # Flag: OOS IC flips sign by day 7
        is_d0  = is_decay.get(f, {}).get(0, np.nan)
        oos_d7_sign_flip = (
            not np.isnan(is_d0) and not np.isnan(oos_d7)
            and np.sign(is_d0) != np.sign(oos_d7)
        )
        row["oos_sign_flip_d7"] = oos_sign_flip_d7 = oos_d7_sign_flip
        records.append(row)

    decay_7d_result = pd.DataFrame(records).set_index("factor")
    decay_7d_result = decay_7d_result.sort_values("is_day0", key=abs, ascending=False)
    decay_7d_result.to_csv(DATA_DIR / "crowding_7day_decay_is_oos.csv")
    print(f"  Saved → data/crowding_7day_decay_is_oos.csv")

    print(f"\n  7-Day IC Decay — IS vs OOS (top 25 factors by |IS IC_day0|):")
    disp_cols = [c for c in ["is_day0", "is_day1", "is_day3", "is_day5", "is_day7",
                              "oos_day0", "oos_day1", "oos_day3", "oos_day5", "oos_day7",
                              "oos_decay_ratio_d7", "oos_sign_flip_d7"]
                 if c in decay_7d_result.columns]
    print(decay_7d_result[disp_cols].head(25).round(4).to_string())

    n_flip = decay_7d_result["oos_sign_flip_d7"].sum()
    print(f"\n  Factors that flip sign OOS by day 7: {int(n_flip)}")
    flipped = decay_7d_result[decay_7d_result["oos_sign_flip_d7"]].index.tolist()
    if flipped:
        print("  ", flipped)

    # ── PLOT: IS vs OOS decay curves for top 16 factors ──────────────────────
    plot_factors = decay_7d_result.head(16).index.tolist()
    n_plot = len(plot_factors)
    ncols = 4
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2), squeeze=False)

    x_ticks  = [0] + DAY_LAGS
    x_labels = ["Day 0\n(monthly)", "Day 1", "Day 3", "Day 5", "Day 7", "Day 10"]

    for idx, fac in enumerate(plot_factors):
        ax = axes[idx // ncols][idx % ncols]
        row = decay_7d_result.loc[fac]

        is_vals  = [row.get(f"is_day{k}",  np.nan) for k in x_ticks]
        oos_vals = [row.get(f"oos_day{k}", np.nan) for k in x_ticks]

        ax.plot(range(len(x_ticks)), is_vals,  "o-", color="#1565C0", lw=2,
                label=f"IS  (≤{TRAIN_END[:4]})", markersize=5)
        ax.plot(range(len(x_ticks)), oos_vals, "s--", color="#C62828", lw=2,
                label=f"OOS (>{TRAIN_END[:4]})", markersize=5)
        ax.axhline(0, color="grey", lw=0.7, ls=":")
        ax.fill_between(range(len(x_ticks)),
                        [v if v is not None and not np.isnan(v) else 0 for v in is_vals],
                        0, alpha=0.08, color="#1565C0")
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_labels, fontsize=6.5)
        ax.set_title(fac, fontsize=8.5, fontweight="bold")
        ax.set_ylabel("IC", fontsize=7)
        ax.tick_params(labelsize=6.5)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused axes
    for idx in range(n_plot, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"7-Day IC Decay: IS (blue) vs OOS (red)\n"
        f"IS ≤ {TRAIN_END}  |  OOS > {TRAIN_END}  —  STRICT SEPARATION\n"
        f"Divergence = regime shift; OOS sign-flip = do NOT use this factor as-is",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "crowding_7day_decay_is_oos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → figures/crowding_7day_decay_is_oos.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONTRARIAN FACTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 3 — Contrarian Factor Analysis (IS vs OOS)")
print("─" * 65)
print("  Strict split: IS ≤ TRAIN_END, OOS > TRAIN_END — NEVER mixed")

oos_ic_path = DATA_DIR / "factor_oos_ic.csv"
if not oos_ic_path.exists():
    print("  [WARN] factor_oos_ic.csv not found — run 2a first")
    sign_flip_factors = []
else:
    oos_ic = pd.read_csv(oos_ic_path, index_col="factor")

    # Sign-flip factors = contrarian IS, different direction OOS
    flip_df = oos_ic[oos_ic["sign_flip"] == True].copy()
    stable_df = oos_ic[oos_ic["sign_flip"] == False].copy()

    print(f"\n  Sign-flip factors (IS → OOS direction reversed): {len(flip_df)}")
    print(f"  Stable factors  (same direction IS and OOS):     {len(stable_df)}")

    # Recompute fresh IS IC for sign-flip factors to confirm
    def monthly_ic_series(panel_sub, factor, target="fwd_ret_1m"):
        results = {}
        for dt, grp in panel_sub.groupby("date"):
            sub = grp[[factor, target]].dropna()
            if len(sub) < MIN_STOCKS:
                continue
            corr, _ = spearmanr(sub[factor], sub[target])
            if not np.isnan(corr):
                results[dt] = corr
        return pd.Series(results)

    # Pick top 20 flip factors by |IC_OOS|
    flip_top = flip_df.reindex(
        flip_df["ic_oos"].abs().sort_values(ascending=False).index
    ).head(20)

    contrarian_records = []
    for fac in flip_top.index:
        if fac not in panel_is.columns:
            continue
        ic_is_series  = monthly_ic_series(panel_is,  fac)
        ic_oos_series = monthly_ic_series(panel_oos, fac)
        contrarian_records.append({
            "factor":         fac,
            "ic_is":          ic_is_series.mean().round(4),
            "icir_is":        (ic_is_series.mean() / ic_is_series.std()).round(3) if ic_is_series.std() > 1e-8 else np.nan,
            "pct_pos_is":     (ic_is_series > 0).mean().round(3),
            "ic_oos":         ic_oos_series.mean().round(4) if len(ic_oos_series) > 0 else np.nan,
            "icir_oos":       (ic_oos_series.mean() / ic_oos_series.std()).round(3) if len(ic_oos_series) > 3 and ic_oos_series.std() > 1e-8 else np.nan,
            "pct_pos_oos":    (ic_oos_series > 0).mean().round(3) if len(ic_oos_series) > 0 else np.nan,
            "n_is_months":    len(ic_is_series),
            "n_oos_months":   len(ic_oos_series),
            "interpretation": "SHORT-TERM REVERSAL" if "ret_1" in fac or "ret_2" in fac or "ret_1w" in fac
                              else "TECHNICAL OSC"   if any(x in fac for x in ["rsi","macd","bollinger","price_to_ma"])
                              else "OTHER",
        })

    contrarian_df = pd.DataFrame(contrarian_records).set_index("factor")
    contrarian_df.to_csv(DATA_DIR / "crowding_contrarian_oos.csv")
    print(f"\n  Contrarian factor IS vs OOS (top {len(contrarian_df)} by |IC_OOS|):")
    print(contrarian_df.to_string())
    print(f"\n  Saved → data/crowding_contrarian_oos.csv")

    sign_flip_factors = flip_df.index.tolist()

    # Plot: IS vs OOS IC for contrarian factors
    if len(contrarian_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(contrarian_df))
        w = 0.35
        bars_is  = ax.bar(x - w/2, contrarian_df["ic_is"],  w, label="IS IC",  color="#1976D2", alpha=0.85)
        bars_oos = ax.bar(x + w/2, contrarian_df["ic_oos"], w, label="OOS IC", color="#E53935", alpha=0.85)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(contrarian_df.index, rotation=90, fontsize=8)
        ax.set_ylabel("IC (Spearman)")
        ax.set_title(
            f"Contrarian Factors — Sign Reversal IS→OOS\n"
            f"IS ≤ {TRAIN_END} (blue)  |  OOS > {TRAIN_END} (red)\n"
            f"Flipped sign = factor is a REVERSAL signal, not momentum",
            fontsize=11
        )
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIG_DIR / "crowding_contrarian_oos.png", dpi=150)
        plt.close(fig)
        print("  Saved → figures/crowding_contrarian_oos.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QUINTILE DIRECTION AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("SECTION 4 — Quintile Direction Audit")
print("─" * 65)
print("  Flag factors where Q5-Q1 spread sign contradicts IC_IS sign.")
print("  These are the 'wrong direction' factors Kieran spotted.\n")

quint_path = DATA_DIR / "factor_quintile_returns.csv"
ic_sum_path = DATA_DIR / "factor_ic_summary.csv"

if not quint_path.exists() or not ic_sum_path.exists():
    print("  [WARN] factor_quintile_returns.csv or factor_ic_summary.csv not found — run 2a first")
else:
    quint_df = pd.read_csv(quint_path)
    ic_sum   = pd.read_csv(ic_sum_path)

    # factor_quintile_returns.csv: columns factor, Q1..Q5, spread (Q5-Q1 mean return)
    # Merge with IC
    if "factor" in quint_df.columns and "factor" in ic_sum.columns:
        merged_q = quint_df.merge(ic_sum[["factor", "ic_mean", "icir"]], on="factor", how="left")

        if "spread" in merged_q.columns:
            # Direction mismatch: IC > 0 but spread < 0, or IC < 0 but spread > 0
            merged_q["ic_sign"]     = np.sign(merged_q["ic_mean"])
            merged_q["spread_sign"] = np.sign(merged_q["spread"])
            merged_q["direction_ok"] = merged_q["ic_sign"] == merged_q["spread_sign"]
            merged_q["issue"] = "OK"
            merged_q.loc[~merged_q["direction_ok"] & (merged_q["ic_mean"].abs() > 0.01), "issue"] = "WRONG_DIR"

            wrong_dir = merged_q[merged_q["issue"] == "WRONG_DIR"].sort_values("ic_mean", key=abs, ascending=False)
            ok_factors = merged_q[merged_q["issue"] == "OK"]

            print(f"  Factors with correct quintile direction  : {len(ok_factors)}")
            print(f"  Factors with WRONG quintile direction    : {len(wrong_dir)}")

            if len(wrong_dir) > 0:
                print("\n  WRONG DIRECTION factors (IC sign ≠ Q5-Q1 sign):")
                display_cols = ["factor", "ic_mean", "icir", "spread", "Q1", "Q3", "Q5"]
                display_cols = [c for c in display_cols if c in wrong_dir.columns]
                print(wrong_dir[display_cols].to_string(index=False))

                print("\n  ROOT CAUSES:")
                for _, row in wrong_dir.iterrows():
                    fac   = row["factor"]
                    ic    = row.get("ic_mean", np.nan)
                    spr   = row.get("spread", np.nan)
                    if "ret_1" in fac or "ret_2" in fac or "ret_1w" in fac:
                        note = "Short-term reversal: IC from Spearman (rank) vs arithmetic mean quintile diverge when outliers present"
                    elif any(x in fac for x in ["vol", "var", "cvar", "dd", "atr", "kurt", "skew"]):
                        note = "Risk factor: negative IC is expected (high risk → low return). Spread confirms but sign is reversed."
                    else:
                        note = "Investigate: possible outlier contamination in Q1/Q5 returns or CS-rank inversion"
                    print(f"    {fac:<30} IC={ic:+.4f}  spread={spr:+.4f}  → {note}")

            merged_q.to_csv(DATA_DIR / "crowding_quintile_audit.csv", index=False)
            print(f"\n  Saved → data/crowding_quintile_audit.csv")

            # Plot: scatter IC_mean vs spread, colour by direction_ok
            fig, ax = plt.subplots(figsize=(10, 7))
            ok_mask   = merged_q["direction_ok"]
            err_mask  = ~merged_q["direction_ok"]

            ax.scatter(merged_q.loc[ok_mask,  "ic_mean"],
                       merged_q.loc[ok_mask,  "spread"],
                       c="#2196F3", alpha=0.7, label="Direction OK", s=40)
            ax.scatter(merged_q.loc[err_mask, "ic_mean"],
                       merged_q.loc[err_mask, "spread"],
                       c="#F44336", marker="X", s=80, alpha=0.9, label="WRONG direction")

            for _, row in merged_q[err_mask].iterrows():
                ax.annotate(row["factor"],
                            xy=(row["ic_mean"], row["spread"]),
                            fontsize=6.5, ha="center", va="bottom",
                            color="#B71C1C")

            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            # Perfect direction line
            xlim = ax.get_xlim()
            ax.plot(xlim, xlim, "k--", lw=0.8, alpha=0.4, label="Perfect alignment")

            ax.set_xlabel("IC_IS (Spearman)", fontsize=11)
            ax.set_ylabel("Quintile Spread Q5−Q1  (annualised %)", fontsize=11)
            ax.set_title(
                "Quintile Direction Audit — IS Data Only\n"
                "Points should be in Q2/Q4 (IC and spread same sign). Red X = wrong direction.",
                fontsize=12
            )
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(FIG_DIR / "crowding_quintile_audit.png", dpi=150)
            plt.close(fig)
            print("  Saved → figures/crowding_quintile_audit.png")
        else:
            print("  [WARN] 'spread' column missing from factor_quintile_returns.csv")
    else:
        print("  [WARN] Unexpected column structure in quintile / IC summary files")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("CROWDING & DECAY DIAGNOSTIC COMPLETE")
print("=" * 65)
print(f"\n  Data outputs:")
print(f"    data/crowding_corr_matrix.csv       — {len(cs_factors)}×{len(cs_factors)} correlation matrix")
print(f"    data/crowding_redundant_pairs.csv   — {len(redundant) if 'redundant' in dir() else '?'} pairs with |r| > {CROWD_THRESHOLD}")
print(f"    data/crowding_7day_decay.csv        — IC day0/3/5/10 per factor")
print(f"    data/crowding_contrarian_oos.csv    — IS vs OOS for sign-flip factors")
print(f"    data/crowding_quintile_audit.csv    — quintile direction vs IC sign")

if "redundant" in dir() and len(redundant) > 0:
    print(f"\n  ⚠ Factor crowding: {len(redundant)} redundant pairs (|r| > {CROWD_THRESHOLD})")
    # Most crowded single factors
    from collections import Counter
    flat = redundant["factor_a"].tolist() + redundant["factor_b"].tolist()
    top_crowded = Counter(flat).most_common(10)
    print(f"  Factors appearing in most redundant pairs (candidates to DROP):")
    for fac, cnt in top_crowded:
        print(f"    {fac:<35} appears in {cnt} redundant pairs")

elapsed = _time.time() - _t0
print(f"\nDone in {elapsed:.1f}s")
