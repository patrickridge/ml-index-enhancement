"""
1t_validate_refetch.py - Is the refetched price file actually better?

Gate between 1s and using the new data. The refetch is only worth taking if it
fixes the price scale and the volume gap without breaking what already worked.

The decisive check is return agreement. 0_data_audit.py verified the existing
file's returns against known annual figures and they were right to within
dividends, so the old returns are the reference. If the new file disagrees with
them, the new file is wrong, not the old one - a scale factor cannot change a
return, so a return that moved means something else did.

Also reports which tickers did not come back. That list should be delisted
names, which is the correct outcome rather than a loss: they are the zombies
from Notes item 22 and they should never have been scored.

Usage:
    python 1t_validate_refetch.py

Reads:  data/prices.parquet, data/prices_refetched.parquet,
        data/refetch_coverage.csv, data/spx_weights.parquet
Writes: nothing. Prints a verdict.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))


def monthly(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    m = (df.set_index("date").groupby("ticker")["close"]
           .resample("ME").last().reset_index())
    m["ret"] = m.groupby("ticker")["close"].pct_change()
    return m


def main():
    old = pd.read_parquet(DATA_DIR / "prices.parquet")
    new = pd.read_parquet(DATA_DIR / "prices_refetched.parquet")
    for d in (old, new):
        d["date"] = pd.to_datetime(d["date"])

    print("REFETCH VALIDATION")
    print("=" * 62)
    print(f"old {len(old):>10,} rows  {old['ticker'].nunique():>4} tickers")
    print(f"new {len(new):>10,} rows  {new['ticker'].nunique():>4} tickers")

    print("\nReturn agreement (the old returns are the reference)")
    print("-" * 62)
    mo, mn = monthly(old), monthly(new)
    j = mo.merge(mn, on=["date", "ticker"], suffixes=("_old", "_new"))
    j = j.dropna(subset=["ret_old", "ret_new"])
    j = j[(j["ret_old"].abs() < 1.0) & (j["ret_new"].abs() < 1.0)]

    corr = float(j["ret_old"].corr(j["ret_new"]))
    mad = float((j["ret_old"] - j["ret_new"]).abs().median())
    print(f"  overlapping monthly returns : {len(j):,}")
    print(f"  correlation                 : {corr:+.4f}")
    print(f"  median absolute difference  : {mad:.4f}")

    per = (j.groupby("ticker")
             .apply(lambda g: g["ret_old"].corr(g["ret_new"])
                    if len(g) > 12 else np.nan)
             .dropna())
    bad = per[per < 0.95]
    print(f"  tickers with corr < 0.95    : {len(bad)} of {len(per)}")
    if len(bad):
        print("    " + ", ".join(bad.nsmallest(8).index.tolist()))

    ok_returns = corr > 0.98 and len(bad) < 0.05 * len(per)
    print(f"  -> {'PASS' if ok_returns else 'FAIL'}: "
          f"{'returns preserved' if ok_returns else 'the refetch changed returns'}")

    print("\nPrice scale")
    print("-" * 62)

    # Measured on current index members, not on every ticker that has ever
    # appeared. Dead names carry stale quotes at both extremes - MHS last
    # printed at $15,869 and CPWR at $0.01 - which widens the spread without
    # saying anything about the live universe. The weight floor removes them
    # downstream, so they should not be in the test either.
    members = set()
    wpath = DATA_DIR / "spx_weights.parquet"
    if wpath.exists():
        w = pd.read_parquet(wpath)
        w["date"] = pd.to_datetime(w["date"])
        members = set(w[(w["spx_weight"] >= 5e-6) &
                        (w["date"] >= w["date"].max() - pd.DateOffset(months=6))
                        ]["ticker"])

    for lbl, d in (("old", old), ("new", new)):
        yr = d["date"].dt.year.max()
        px = (d[d["date"].dt.year == yr].sort_values("date")
                .groupby("ticker")["close"].last())
        px = px[px > 0]
        for scope, sel in (("all   ", px),
                           ("members", px[px.index.isin(members)] if members
                            else px)):
            if not len(sel):
                continue
            print(f"  {lbl} {scope}: log-sd {np.log(sel).std():.2f} "
                  f"| p5 ${sel.quantile(.05):>9,.0f} "
                  f"| median ${sel.median():>8,.0f} "
                  f"| p95 ${sel.quantile(.95):>9,.0f}")
    print("  real S&P 500 : log-sd ~0.85 | p5 ~$25 | median ~$100 | p95 ~$500")

    print("\nVolume")
    print("-" * 62)
    for lbl, d in (("old", old), ("new", new)):
        per_t = d.groupby("ticker")["volume"].apply(
            lambda s: s.notna().mean() if len(s) else 0.0)
        print(f"  {lbl}: {float(per_t.gt(0.9).mean()):.0%} of tickers have volume, "
              f"{int(per_t.lt(0.1).sum())} have none")

    print("\nTickers that did not come back")
    print("-" * 62)
    missing = sorted(set(old["ticker"]) - set(new["ticker"]))
    print(f"  {len(missing)} missing")
    if missing:
        print("  " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))

    wpath = DATA_DIR / "spx_weights.parquet"
    if wpath.exists() and missing:
        w = pd.read_parquet(wpath)
        w["date"] = pd.to_datetime(w["date"])
        recent = w[(w["date"] >= w["date"].max() - pd.DateOffset(years=2))
                   & (w["spx_weight"] >= 5e-6)]
        still_in = sorted(set(missing) & set(recent["ticker"]))
        print(f"\n  of those, still index members in the last 2 years: "
              f"{len(still_in)}")
        if still_in:
            print("  " + ", ".join(still_in))
            print("  These are real losses and need another source.")
        else:
            print("  None. Every missing name had already left the index,")
            print("  so losing them removes zombies rather than data.")

    print("\n" + "=" * 62)
    if ok_returns:
        print("VERDICT: safe to adopt.")
        print("  mv data/prices_refetched.parquet data/prices.parquet")
        print("  then rerun 1c, 1h, and 0_data_audit.py")
        print("\nDo NOT do this while a GPU run is scoring the old panel.")
    else:
        print("VERDICT: do not adopt. Returns moved, so investigate first.")


if __name__ == "__main__":
    main()
