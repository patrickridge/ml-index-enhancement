"""
4i_capacity.py - At what AUM does market impact eat the alpha?

The backtests charge a flat 10 bps per side, which is a reasonable proxy for
spread and commission but says nothing about size. A strategy turning over ~64%
of the book each month is fine at 100m and may be impossible at 10bn, and the
flat-cost model cannot tell the two apart because it does not know how large the
trades are relative to the stocks being traded.

Applies a square-root impact model. Cost in return terms for trading Q dollars
of a stock whose average daily volume is ADV:

    impact = C * sigma_daily * sqrt(Q / ADV)

This is the standard Almgren-style form: impact grows with the square root of
participation, not linearly, and scales with the stock's own volatility. C is
the one free parameter; 0.5 to 1.0 is the usual empirical range and 1.0 is used
here as the conservative choice.

Reconstructs portfolio weights rather than reading them, because 4b saves
returns only. The construction mirrors 4b exactly: z-score tilt, long-only clip,
renormalise.

ADV is approximated as a fixed fraction of market cap. The enriched panel's
volume columns are cross-sectionally ranked into [-0.5, 0.5] and so cannot be
read as dollars. S&P 500 names typically turn over 0.5-1.0% of cap per day; 0.7%
is used, and the sensitivity to that choice is reported.

Reads:  data/scores_cs_transformer.parquet, data/spx_weights.parquet
Writes: data/capacity_analysis.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
OUT_PATH = DATA_DIR / "capacity_analysis.csv"

ALPHA          = 0.0005      # tilt strength; matches the 4b sweep selection
MIN_WEIGHT     = 5e-6
C_IMPACT       = 1.0         # square-root impact coefficient (conservative)
SIGMA_DAILY    = 0.020       # ~2%/day, typical S&P 500 single-name vol
ADV_FRAC_OF_CAP = 0.007      # daily volume as a share of market cap
EXEC_DAYS      = 3           # spread each month's trade over this many days

AUM_LEVELS = [1e8, 2.5e8, 5e8, 1e9, 2.5e9, 5e9, 1e10, 2.5e10, 5e10]

SCORE_FILES = {
    "CS-Transformer": "scores_cs_transformer.parquet",
    "LightGBM":       "scores_lgbm.parquet",
}


def build_weights(scores, weights, alpha=ALPHA):
    """Reproduce 4b's z-score tilt and return per-month weights plus mktcap."""
    df = scores.merge(weights[["date", "ticker", "spx_weight", "mktcap"]],
                      on=["date", "ticker"], how="inner")
    df = df.dropna(subset=["score", "spx_weight", "fwd_ret_1m", "mktcap"])
    df = df[df["spx_weight"] >= MIN_WEIGHT]

    out = []
    for dt, g in df.groupby("date"):
        g = g.copy()
        sd = g["score"].std(ddof=0)
        if sd < 1e-12:
            continue
        cov = g["spx_weight"].sum()
        if cov < 1e-9:
            continue
        bench_w = g["spx_weight"] / cov
        tilt    = alpha * (g["score"] - g["score"].mean()) / sd
        raw     = (bench_w + tilt).clip(lower=0.0)
        tot     = raw.sum()
        if tot < 1e-9:
            continue
        g["w"] = raw / tot
        out.append(g[["date", "ticker", "w", "mktcap"]])
    return pd.concat(out, ignore_index=True)


def main():
    wpath = DATA_DIR / "spx_weights.parquet"
    if not wpath.exists():
        print(f"ERROR: {wpath} not found.")
        return
    weights = pd.read_parquet(wpath)
    weights["date"] = pd.to_datetime(weights["date"])

    rows = []
    for model, fname in SCORE_FILES.items():
        spath = DATA_DIR / fname
        if not spath.exists():
            print(f"[skip] {model}: {fname} not found")
            continue

        scores = pd.read_parquet(spath)
        scores["date"] = pd.to_datetime(scores["date"])
        w = build_weights(scores, weights)
        if w.empty:
            print(f"[skip] {model}: no usable months")
            continue

        # Month-on-month weight change per ticker = what has to be traded
        piv   = w.pivot_table(index="date", columns="ticker", values="w").fillna(0.0)
        cap   = w.pivot_table(index="date", columns="ticker", values="mktcap")
        dw    = piv.diff().abs().iloc[1:]
        cap   = cap.reindex(dw.index)
        n_mo  = len(dw)
        turn  = dw.sum(axis=1).mean()

        print(f"\n{model}: {n_mo} months, mean turnover {turn*100:.1f}% of book/month")
        print(f"  {'AUM':>10}  {'impact/yr':>10}  {'flat 10bps':>11}  {'total':>8}")
        print("  " + "-" * 46)

        for aum in AUM_LEVELS:
            traded = dw * aum                       # dollars traded per name
            adv    = cap * ADV_FRAC_OF_CAP          # dollars of daily volume
            part   = (traded / (adv * EXEC_DAYS)).replace([np.inf, -np.inf], np.nan)
            imp    = C_IMPACT * SIGMA_DAILY * np.sqrt(part.clip(lower=0))
            # weight each name's impact by its share of that month's trading
            share  = dw.div(dw.sum(axis=1), axis=0)
            mo_cost = (imp * share).sum(axis=1)

            impact_yr = mo_cost.mean() * 12
            flat_yr   = turn * 0.0010 * 12
            total_yr  = impact_yr + flat_yr

            print(f"  {aum/1e9:>8.2f}bn  {impact_yr*100:>9.2f}%  "
                  f"{flat_yr*100:>10.2f}%  {total_yr*100:>7.2f}%")
            rows.append({"model": model, "aum_usd": aum,
                         "turnover_pm": round(turn, 4),
                         "impact_annual": round(impact_yr, 5),
                         "flat_annual": round(flat_yr, 5),
                         "total_annual": round(total_yr, 5)})

    if not rows:
        print("\nNo score files found.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print("\nCompare total cost against the strategy's gross alpha. Where the")
    print("two cross is the capacity limit. Assumptions: C=1.0, 2%/day vol,")
    print(f"ADV={ADV_FRAC_OF_CAP:.1%} of cap, execution over {EXEC_DAYS} days.")
    print("Halving C or doubling EXEC_DAYS roughly halves the impact estimate,")
    print("so treat the level as indicative and the ranking across AUM as robust.")


if __name__ == "__main__":
    main()
