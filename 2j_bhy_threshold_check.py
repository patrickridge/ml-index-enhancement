"""
Are the BHY survivors an artifact of the universe filter?

The filter cuts at spx_weight >= 5e-6, which is itself a size and liquidity
threshold. Every factor that survived BHY is a size, liquidity or volatility
factor. Filtering on a size boundary and then discovering size effects is a
mechanism that could manufacture exactly that result.

Recomputes IC t-statistics for the survivors plus a control group of factors
that failed, across a range of weight floors. If the survivors are real their
t-stats should be broadly stable; if the filter is generating them, they should
move sharply with the threshold.

Focused on ~25 factors rather than re-running all of 2a three times.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

D = Path("/Users/patrick/Desktop/neural-network/data")
TRAIN_END = pd.Timestamp("2022-12-31")

SURVIVORS = [
    "size_proxy", "log_mktcap", "amihud_illiq_21d", "dollar_vol_21d",
    "dollar_vol_63d", "cand_kyle_lambda_21d", "vol_above_avg",
    "high_vol_week", "volume",
]
CONTROLS = [
    "ret_12m", "ret_6m", "ret_1m", "rsi_14", "macd_hist",
    "bollinger_pct", "vol_252d", "idio_vol_252d", "ir_12m",
]
THRESHOLDS = [0.0, 1e-6, 5e-6, 1e-5, 5e-5]


def ic_tstat(panel, factor):
    ics = []
    for _, g in panel.groupby("date"):
        sub = g[[factor, "fwd_ret_1m"]].dropna()
        if len(sub) < 20:
            continue
        c = spearmanr(sub[factor], sub["fwd_ret_1m"]).correlation
        if np.isfinite(c):
            ics.append(c)
    if len(ics) < 12:
        return np.nan, np.nan
    s = pd.Series(ics)
    return s.mean(), s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


panel = pd.read_parquet(D / "panel_monthly_enriched.parquet")
w     = pd.read_parquet(D / "spx_weights.parquet",
                        columns=["date", "ticker", "spx_weight"])
for df in (panel, w):
    df["date"] = pd.to_datetime(df["date"])

m = panel.merge(w, on=["date", "ticker"], how="inner")
m = m[m["date"] <= TRAIN_END]
m = m[m["fwd_ret_1m"].abs() <= 1.0]

avail_s = [f for f in SURVIVORS if f in m.columns]
avail_c = [f for f in CONTROLS  if f in m.columns]
print(f"Testing {len(avail_s)} survivors + {len(avail_c)} controls "
      f"across {len(THRESHOLDS)} weight floors\n")

rows = []
for thr in THRESHOLDS:
    sub = m[m["spx_weight"] >= thr] if thr > 0 else m
    n_obs = len(sub)
    for grp, feats in (("survivor", avail_s), ("control", avail_c)):
        for f in feats:
            ic, t = ic_tstat(sub, f)
            rows.append({"threshold": thr, "group": grp, "factor": f,
                         "ic": ic, "t": t, "n_obs": n_obs})
    print(f"  floor {thr:.0e}: {n_obs:,} rows done")

r = pd.DataFrame(rows)
piv = r.pivot_table(index=["group", "factor"], columns="threshold", values="t")

print("\n" + "=" * 78)
print("IC t-STATISTIC BY WEIGHT FLOOR")
print("=" * 78)
print(piv.to_string(float_format=lambda x: f"{x:+.2f}"))

print("\n" + "=" * 78)
print("STABILITY")
print("=" * 78)
for grp in ("survivor", "control"):
    g = piv.loc[grp]
    rng = (g.max(axis=1) - g.min(axis=1)).abs()
    print(f"  {grp:<9} mean |t| range across floors: {rng.mean():.2f}  "
          f"(max {rng.max():.2f})")
    flips = ((g.min(axis=1) < 0) & (g.max(axis=1) > 0)).sum()
    print(f"  {grp:<9} factors changing sign across floors: {flips}/{len(g)}")

print("\nIf survivors are genuine their t-stats hold up across floors.")
print("If the filter is generating them, they collapse as the floor drops to 0.")
