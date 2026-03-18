"""
5b_ic_decay_all.py — IC Decay Grid for ALL Factors
===================================================
Generates a grid plot showing IC decay curves (0-60 months) for every factor.
Used for meeting review to visually identify short-term vs long-term factors.
"""

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
import math, time as _time

_t0 = _time.time()

DATA_DIR = Path("data")
FIG_DIR  = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_IC_DECAY_LAGS = 60
MIN_STOCKS        = 20

MACRO_COLS = [
    "vix_level","vix_change_21d","yield_10y","yield_spread_10y2y",
    "yield_change_21d","dollar_index","credit_proxy_change",
    "market_trend_spx","market_vol_regime",
    "spx_ret_1m","spx_ret_3m","spx_ret_6m","spx_ret_12m","spx_vol_63d"
]

# ── Load data ──────────────────────────────────────────────────────────────────
TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-12-31"

print("Loading panel …")
panel = pq.read_table(DATA_DIR / "panel_monthly_enriched.parquet").to_pandas()
panel["date"] = pd.to_datetime(panel["date"])
panel = panel[(panel["date"] >= TRAIN_START) & (panel["date"] <= TRAIN_END)].reset_index(drop=True)
all_cols   = [c for c in panel.columns if c not in ["date","ticker","fwd_ret_1m"] + MACRO_COLS]
feat_cols  = [c for c in all_cols if panel[c].nunique() > 1]
n_months   = panel["date"].nunique()
print(f"  {len(feat_cols)} features | {n_months} train months ({TRAIN_START[:4]}–{TRAIN_END[:4]}) | {panel['ticker'].nunique()} tickers")


# ── IC decay helper ────────────────────────────────────────────────────────────
def ic_decay_series(factor: str) -> list:
    dates        = sorted(panel["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(dates)}
    decay        = []
    for lag in range(MAX_IC_DECAY_LAGS + 1):
        ics = []
        for dt, grp in panel.groupby("date"):
            future_idx = date_to_idx[dt] + lag
            if future_idx >= len(dates):
                continue
            future_dt   = dates[future_idx]
            factor_vals = grp[["ticker", factor]].dropna().set_index("ticker")[factor]
            ret_vals    = (panel[panel["date"] == future_dt]
                           [["ticker","fwd_ret_1m"]].dropna()
                           .set_index("ticker")["fwd_ret_1m"])
            common = factor_vals.index.intersection(ret_vals.index)
            if len(common) < MIN_STOCKS:
                continue
            corr, _ = spearmanr(factor_vals[common], ret_vals[common])
            if not np.isnan(corr):
                ics.append(corr)
        decay.append(float(np.mean(ics)) if ics else np.nan)
    return decay


# ── Compute IC decay for all factors ──────────────────────────────────────────
print(f"\nComputing IC decay (lags 0–{MAX_IC_DECAY_LAGS}) for all {len(feat_cols)} factors …")
all_decay = {}
for i, fac in enumerate(feat_cols, 1):
    all_decay[fac] = ic_decay_series(fac)
    print(f"  {i}/{len(feat_cols)}: {fac}")

decay_df = pd.DataFrame(all_decay, index=range(MAX_IC_DECAY_LAGS + 1)).T
decay_df.index.name = "factor"
decay_df.to_csv(DATA_DIR / "factor_ic_decay_all.csv")
print("Saved → factor_ic_decay_all.csv")


# ── Grid plot ─────────────────────────────────────────────────────────────────
n_factors = len(feat_cols)
n_cols    = 8
n_rows    = math.ceil(n_factors / n_cols)
lags      = list(range(MAX_IC_DECAY_LAGS + 1))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.2))
axes      = axes.flatten()

for idx, fac in enumerate(feat_cols):
    ax     = axes[idx]
    values = decay_df.loc[fac].values
    ic0    = values[0] if not np.isnan(values[0]) else 0
    color  = "#2196F3" if ic0 >= 0 else "#F44336"

    ax.plot(lags, values, color=color, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(fac, fontsize=7, pad=2)
    ax.set_xlim(0, MAX_IC_DECAY_LAGS)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("Lag (months)", fontsize=5)
    ax.set_ylabel("IC", fontsize=5)

# Hide unused subplots
for idx in range(n_factors, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f"IC Decay (0–{MAX_IC_DECAY_LAGS} months) — All {n_factors} Factors\n"
             f"Blue = positive IC at lag 0, Red = contrarian signal",
             fontsize=11, y=1.01)
plt.tight_layout()
out = FIG_DIR / "factor_ic_decay_all_factors.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")
