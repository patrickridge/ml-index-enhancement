"""
1q_survivorship_bias.py - Estimate how much survivorship bias inflates the backtests.

The panel holds 697 tickers against roughly 1,200 that passed through the S&P 500
between 2010 and 2025. The 247 that could not be fetched are overwhelmingly
companies that were acquired, went bankrupt, or were dropped from the index -
exactly the population whose returns were worst before they disappeared.
Excluding them biases every backtest upward.

Fixing this needs CRSP or Compustat. Quantifying it does not, and an estimate of
the bias is more useful than an unqualified caveat.

Method: find tickers that do appear in the panel but stop partway through, which
means they left the index during the sample. Measure their active returns in the
months before they vanish. If departing names underperform, the same effect
applies to the 247 that were never fetched at all, and the magnitude here is a
lower bound on what is missing.

Reads:  data/panel_monthly_enriched.parquet, data/spx_weights.parquet
Writes: data/survivorship_estimate.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
OUT_PATH = DATA_DIR / "survivorship_estimate.csv"

MIN_WEIGHT   = 5e-6    # matches config.clean_universe
EXIT_WINDOW  = 6       # months before disappearance to measure
TAIL_BUFFER  = 6       # ignore exits this close to the panel end (not real exits)


def main():
    panel_path = DATA_DIR / "panel_monthly_enriched.parquet"
    w_path     = DATA_DIR / "spx_weights.parquet"
    if not panel_path.exists() or not w_path.exists():
        print("ERROR: need panel_monthly_enriched.parquet and spx_weights.parquet")
        return

    panel = pd.read_parquet(panel_path, columns=["date", "ticker", "fwd_ret_1m"])
    w     = pd.read_parquet(w_path, columns=["date", "ticker", "spx_weight"])
    for d in (panel, w):
        d["date"] = pd.to_datetime(d["date"])

    df = panel.merge(w, on=["date", "ticker"], how="inner")
    df = df[df["spx_weight"] >= MIN_WEIGHT].dropna(subset=["fwd_ret_1m"])

    months    = np.sort(df["date"].unique())
    panel_end = months[-1]
    cutoff    = months[max(0, len(months) - 1 - TAIL_BUFFER)]

    last_seen = df.groupby("ticker")["date"].max()
    leavers   = last_seen[last_seen < cutoff].index
    stayers   = last_seen[last_seen >= cutoff].index

    print("=" * 66)
    print("SURVIVORSHIP BIAS ESTIMATE")
    print("=" * 66)
    print(f"  Panel window           : {months[0].date()} -> {panel_end.date()}")
    print(f"  Tickers ever present   : {df['ticker'].nunique()}")
    print(f"  Left during sample     : {len(leavers)}")
    print(f"  Present at the end     : {len(stayers)}")

    if len(leavers) == 0:
        print("\n  No mid-sample exits found; cannot estimate from this panel.")
        return

    # Market return each month, as the benchmark to measure against
    mkt = df.groupby("date").apply(
        lambda g: np.average(g["fwd_ret_1m"], weights=g["spx_weight"]),
        include_groups=False).rename("mkt_ret")

    d = df.merge(mkt, on="date")
    d["active"] = d["fwd_ret_1m"] - d["mkt_ret"]

    # Active return in the final EXIT_WINDOW months of a leaver's life
    exit_rows = []
    for tkr in leavers:
        sub = d[d["ticker"] == tkr].sort_values("date")
        tail = sub.tail(EXIT_WINDOW)
        if len(tail) >= 2:
            exit_rows.append(tail["active"].mean())

    exit_active  = np.array(exit_rows)
    stay_active  = d[d["ticker"].isin(stayers)]["active"].mean()

    print(f"\n  Mean monthly active return, last {EXIT_WINDOW} months before exit:")
    print(f"    leavers  : {exit_active.mean()*100:+.3f}%  (n={len(exit_active)})")
    print(f"    stayers  : {stay_active*100:+.3f}%")
    drag = exit_active.mean() - stay_active
    print(f"    gap      : {drag*100:+.3f}% / month  ({drag*12*100:+.2f}% / yr)")

    # Scale to the 247 names that were never fetched.
    n_missing = 247
    n_present = df["ticker"].nunique()
    share     = n_missing / (n_present + n_missing)
    est_bias  = -drag * share * 12

    print(f"\n  Missing tickers        : {n_missing}")
    print(f"  Share of true universe : {share*100:.1f}%")
    print(f"  Implied upward bias    : ~{est_bias*100:.2f}% / yr of alpha")
    print("\n  Lower bound. Names that could not be fetched at all are likely to")
    print("  have failed harder than those that merely left the index, and the")
    print("  estimate assumes the same drag applies to both groups.")

    pd.DataFrame([{
        "leaver_active_pm":  exit_active.mean(),
        "stayer_active_pm":  stay_active,
        "gap_pm":            drag,
        "n_leavers":         len(exit_active),
        "n_missing_assumed": n_missing,
        "missing_share":     share,
        "est_annual_bias":   est_bias,
    }]).to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
