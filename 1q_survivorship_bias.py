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

    # .unique() on a datetime column returns numpy datetime64, which has no
    # .date(); wrap so the formatting below works either way
    months    = pd.to_datetime(pd.Series(df["date"].unique())).sort_values()
    months    = list(months)
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

    # Active return over the months leading up to an exit.
    #
    # The final observation is dropped. fwd_ret_1m on a ticker's last month is
    # the return into a month where the row no longer exists, so it is either
    # fabricated or a full write-off - the same dead-ticker data problem this
    # whole exercise is about. Including it produced a -6.15%/month estimate,
    # which is an artifact rather than a measurement.
    exit_rows = []
    for tkr in leavers:
        sub = d[d["ticker"] == tkr].sort_values("date")
        if len(sub) < 3:
            continue
        tail = sub.iloc[:-1].tail(EXIT_WINDOW)   # drop the terminal month
        if len(tail) >= 2:
            m = tail["active"].mean()
            if np.isfinite(m) and abs(m) < 0.5:  # guard against residual junk
                exit_rows.append(m)

    exit_active  = np.array(exit_rows)
    stay_active  = d[d["ticker"].isin(stayers)]["active"].mean()

    print(f"\n  Mean monthly active return, last {EXIT_WINDOW} months before exit:")
    print(f"    leavers  : {exit_active.mean()*100:+.3f}%  (n={len(exit_active)})")
    print(f"    stayers  : {stay_active*100:+.3f}%")
    drag = exit_active.mean() - stay_active
    print(f"    gap      : {drag*100:+.3f}% / month  ({drag*12*100:+.2f}% / yr)")

    # Scale to the names that were never fetched.
    #
    # A departing stock only drags while it is declining, then it leaves the
    # universe. So the right denominator is total universe-months, not the
    # number of tickers: each leaver contributes EXIT_WINDOW months of drag
    # spread across every stock-month in the panel.
    #
    # Multiplying the monthly gap by 12 and by the missing share instead treats
    # a transient event as permanent, which is how an earlier version of this
    # script produced an implausible 14.7%/yr.
    n_missing   = 247
    n_present   = df["ticker"].nunique()
    n_months    = len(months)
    univ_months = n_present * n_months

    observed_drag_pm = (len(exit_active) * EXIT_WINDOW * drag) / univ_months
    scale            = n_missing / max(len(exit_active), 1)
    est_bias_pm      = -observed_drag_pm * scale
    est_bias_yr      = est_bias_pm * 12

    print(f"\n  Observed leavers       : {len(exit_active)}")
    print(f"  Universe stock-months  : {univ_months:,}")
    print(f"  Drag already in panel  : {observed_drag_pm*12*100:+.3f}% / yr")
    print(f"  Missing tickers        : {n_missing}  ({scale:.1f}x the observed leavers)")
    print(f"  Implied additional bias: ~{est_bias_yr*100:.2f}% / yr of alpha")
    print("\n  Assumes missing names decline like observed leavers and sit in the")
    print("  index for a comparable spell. Names that could not be fetched at all")
    print("  probably failed harder, so treat this as a lower bound.")

    pd.DataFrame([{
        "leaver_active_pm":     exit_active.mean(),
        "stayer_active_pm":     stay_active,
        "gap_pm":               drag,
        "n_leavers":            len(exit_active),
        "universe_stock_months": univ_months,
        "observed_drag_annual": observed_drag_pm * 12,
        "n_missing_assumed":    n_missing,
        "est_annual_bias":      est_bias_yr,
    }]).to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
