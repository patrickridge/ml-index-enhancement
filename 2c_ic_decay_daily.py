"""
2c_ic_decay_daily.py — Daily IC Decay (1–90 trading days)
==========================================================
Tests how long each factor's predictive signal lasts at daily resolution.
Complements 5b_ic_decay_all.py (which uses monthly lags).

Method:
  - Factor values taken from panel_monthly_enriched.parquet at each month-end
  - Forward returns computed from prices.parquet at 1,2,3,5,10,15,20,30,45,60,90 days ahead
  - Spearman IC computed between factor and fwd return at each horizon
  - Averaged across all month-end dates in training period (2010–2020)

Why this matters:
  - If IC peaks at day 5–10 → signal is short-lived, weekly rebalancing would help
  - If IC is flat/rising to day 20–30 → monthly rebalancing is optimal
  - If IC is still strong at day 60–90 → factor is a slow structural tilt

Outputs:
  data/factor_ic_decay_daily.csv         — IC at each horizon per factor
  figures/factor_ic_decay_daily_grid.png — grid of all factors (like 5b)
  figures/factor_ic_decay_daily_top20.png — top 20 factors by |ICIR| zoomed in

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
import math

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-12-31"

# Daily horizons to test (trading days)
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90]
MIN_STOCKS = 30   # skip month-end if fewer stocks have valid data

MACRO_COLS = [
    "vix_level", "vix_change_21d", "yield_10y", "yield_spread_10y2y",
    "yield_change_21d", "dollar_index", "credit_proxy_change",
    "market_trend_spx", "market_vol_regime",
    "spx_ret_1m", "spx_ret_3m", "spx_ret_6m", "spx_ret_12m", "spx_vol_63d",
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading monthly panel …")
panel = pq.read_table(DATA_DIR / "panel_monthly_enriched.parquet").to_pandas()
panel["date"] = pd.to_datetime(panel["date"])
panel = panel[(panel["date"] >= TRAIN_START) & (panel["date"] <= TRAIN_END)].reset_index(drop=True)

exclude   = {"date", "ticker", "fwd_ret_1m"} | set(MACRO_COLS)
feat_cols = [c for c in panel.columns if c not in exclude and panel[c].nunique() > 1]
print(f"  {len(feat_cols)} factors | {panel['date'].nunique()} month-ends | {panel['ticker'].nunique()} tickers")

print("Loading daily prices …")
prices = pq.read_table(DATA_DIR / "prices.parquet").to_pandas()
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

# Build a pivot table: date × ticker → close price (for fast lookups)
close_pivot = prices.pivot(index="date", columns="ticker", values="close")
all_trading_days = close_pivot.index.sort_values()

# ── Forward return at N trading days ──────────────────────────────────────────
def fwd_return_n_days(ref_date: pd.Timestamp, n: int) -> pd.Series:
    """
    Return (close_{ref_date + n trading days} / close_{ref_date}) - 1
    for all tickers with valid prices on both days.
    """
    # Find ref_date in trading calendar (or nearest prior day)
    valid_days = all_trading_days[all_trading_days >= ref_date]
    if len(valid_days) == 0:
        return pd.Series(dtype=float)
    t0_day = valid_days[0]

    idx = all_trading_days.get_loc(t0_day)
    if idx + n >= len(all_trading_days):
        return pd.Series(dtype=float)
    t1_day = all_trading_days[idx + n]

    c0 = close_pivot.loc[t0_day].dropna()
    c1 = close_pivot.loc[t1_day].dropna()
    common = c0.index.intersection(c1.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    return (c1[common] / c0[common] - 1).rename(f"fwd_{n}d")


# ── IC at each horizon for one factor ─────────────────────────────────────────
def daily_ic_decay(factor: str) -> list:
    """
    Returns list of mean IC values, one per horizon in HORIZONS.
    """
    month_ends = sorted(panel["date"].unique())
    horizon_ics = {h: [] for h in HORIZONS}

    for dt in month_ends:
        grp = panel[panel["date"] == dt][["ticker", factor]].dropna()
        if len(grp) < MIN_STOCKS:
            continue
        factor_vals = grp.set_index("ticker")[factor]

        for h in HORIZONS:
            ret_vals = fwd_return_n_days(dt, h)
            common = factor_vals.index.intersection(ret_vals.index)
            if len(common) < MIN_STOCKS:
                continue
            corr, _ = spearmanr(factor_vals[common], ret_vals[common])
            if not np.isnan(corr):
                horizon_ics[h].append(corr)

    return [float(np.mean(horizon_ics[h])) if horizon_ics[h] else np.nan for h in HORIZONS]


# ── Compute for all factors ────────────────────────────────────────────────────
print(f"\nComputing daily IC decay ({HORIZONS}) for {len(feat_cols)} factors …")
all_decay = {}
for i, fac in enumerate(feat_cols, 1):
    all_decay[fac] = daily_ic_decay(fac)
    print(f"  {i:2d}/{len(feat_cols)}: {fac}")

decay_df = pd.DataFrame(all_decay, index=HORIZONS).T
decay_df.index.name = "factor"
decay_df.columns = [f"day_{h}" for h in HORIZONS]
decay_df.to_csv(DATA_DIR / "factor_ic_decay_daily.csv")
print("Saved → factor_ic_decay_daily.csv")


# ── Grid plot (all factors) ────────────────────────────────────────────────────
n_factors = len(feat_cols)
n_cols    = 8
n_rows    = math.ceil(n_factors / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.2))
axes = axes.flatten()

for idx, fac in enumerate(feat_cols):
    ax     = axes[idx]
    values = decay_df.loc[fac].values
    ic0    = values[0] if not np.isnan(values[0]) else 0
    color  = "#2196F3" if ic0 >= 0 else "#F44336"

    ax.plot(HORIZONS, values, color=color, linewidth=1.2, marker="o", markersize=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(20, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)  # ~1 month mark
    ax.set_title(fac, fontsize=7, pad=2)
    ax.set_xlim(0, 95)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("Days ahead", fontsize=5)
    ax.set_ylabel("IC", fontsize=5)

for idx in range(n_factors, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    f"Daily IC Decay (1–90 trading days) — All {n_factors} Factors  |  Train: {TRAIN_START[:4]}–{TRAIN_END[:4]}\n"
    f"Blue = positive IC, Red = contrarian  |  Grey dotted = ~1 month (day 20)",
    fontsize=10, y=1.01,
)
plt.tight_layout()
out = FIG_DIR / "factor_ic_decay_daily_grid.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")


# ── Top 20 factors zoomed plot ─────────────────────────────────────────────────
# Rank by |IC| at day 20 (closest to our monthly rebalance horizon)
ic_at_20 = decay_df["day_20"].abs().sort_values(ascending=False)
top20     = ic_at_20.head(20).index.tolist()

n_cols2 = 5
n_rows2 = math.ceil(len(top20) / n_cols2)
fig2, axes2 = plt.subplots(n_rows2, n_cols2, figsize=(n_cols2 * 3.5, n_rows2 * 2.5))
axes2 = axes2.flatten()

for idx, fac in enumerate(top20):
    ax     = axes2[idx]
    values = decay_df.loc[fac].values
    ic0    = decay_df.loc[fac, "day_20"]
    color  = "#2196F3" if ic0 >= 0 else "#F44336"

    ax.plot(HORIZONS, values, color=color, linewidth=1.5, marker="o", markersize=4)
    ax.fill_between(HORIZONS, 0, values, alpha=0.15, color=color)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(20, color="grey", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_title(f"{fac}\n(IC@20d={ic0:.3f})", fontsize=8, pad=2)
    ax.set_xlim(0, 95)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Days ahead", fontsize=6)
    ax.set_ylabel("IC", fontsize=6)

for idx in range(len(top20), len(axes2)):
    axes2[idx].set_visible(False)

fig2.suptitle(
    f"Top 20 Factors by |IC| at Day 20  |  Train: {TRAIN_START[:4]}–{TRAIN_END[:4]}",
    fontsize=11,
)
plt.tight_layout()
out2 = FIG_DIR / "factor_ic_decay_daily_top20.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {out2}")

print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")
