"""
0_data_audit.py - Invariant checks on the data every other script trusts.

Written after a question about index weights turned up market caps in the
quadrillions. Three separate problems had been sitting in the pipeline without
anything failing, because nothing checked. The pattern each time is the same:
the bug is silent, the output looks plausible, and it only surfaces when
somebody compares a number to the outside world.

So this compares numbers to the outside world. It asserts what the data must
look like if the pipeline is behaving, and it fails loudly when it is not.

Numbered 0 because it is a precondition, not a stage. Run it before trusting
any result, and after any change to the 1* fetch scripts.

Usage:
    python 0_data_audit.py

Reads:  data/prices.parquet, data/spx_weights.parquet,
        data/panel_monthly_enriched.parquet, data/factor_ic_summary.csv
Writes: nothing. Prints a report and exits non-zero if any check fails.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))

# Known values from outside the pipeline. The whole point is that these are
# not derived from the data being checked, so they cannot agree by accident.
KNOWN_RETURNS = {
    "AAPL.O": {2019: 0.890, 2020: 0.810, 2021: 0.340,
               2022: -0.264, 2023: 0.481, 2024: 0.303},
    "MSFT.O": {2019: 0.578, 2020: 0.427, 2021: 0.525,
               2022: -0.280, 2023: 0.583, 2024: 0.132},
}
# Month-end closes, split and dividend adjusted, from the market rather than
# from anything in this repo. An earlier version of this file carried values
# typed from memory and they were 15% out, which made the per-ticker scale
# factors it reported wrong even though the conclusion held.
KNOWN_PRICE_DEC2025 = {"AAPL.O": 271, "MSFT.O": 481, "WMT.O": 111,
                       "NVDA.O": 186, "XOM.N": 118}

# Real S&P 500 concentration, for comparison. Top-10 share of index weight.
PLAUSIBLE_TOP10 = {2005: 0.20, 2015: 0.18, 2025: 0.40}

results = []


def check(name, passed, detail="", warn=False):
    tag = "WARN" if warn and not passed else ("PASS" if passed else "FAIL")
    results.append((tag, name))
    print(f"  [{tag}] {name}")
    if detail:
        for line in detail.rstrip().split("\n"):
            print(f"         {line}")


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def main():
    print("DATA AUDIT")
    print("=" * 62)

    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    section("Prices")

    # Returns are what the models predict and the backtests compound, so they
    # matter more than levels. Checked against outside knowledge.
    worst = 0.0
    detail = []
    for tkr, years in KNOWN_RETURNS.items():
        s = prices[prices["ticker"] == tkr].set_index("date")["close"]
        if s.empty:
            continue
        annual = s.resample("YE").last().pct_change()
        for yr, actual in years.items():
            got = annual[annual.index.year == yr]
            if not len(got):
                continue
            err = abs(float(got.iloc[0]) - actual)
            worst = max(worst, err)
            if err > 0.02:
                detail.append(f"{tkr} {yr}: {got.iloc[0]:+.1%} vs {actual:+.1%}")
    check("Annual returns match known actuals", worst < 0.02,
          "\n".join(detail) or f"worst deviation {worst:.1%} (dividends explain ~1%)")

    # Levels are a different question. A per-ticker scale factor leaves returns
    # correct and silently corrupts everything computed from price levels.
    dec = prices[(prices["date"].dt.year == 2025) & (prices["date"].dt.month == 12)]
    dec = dec.sort_values("date").groupby("ticker")["close"].last()
    scales = {t: float(dec[t]) / p for t, p in KNOWN_PRICE_DEC2025.items()
              if t in dec.index}
    if scales:
        logs = np.log(list(scales.values()))
        spread = float(logs.max() - logs.min())
        detail = [f"{t}: {v:,.0f}x" for t, v in scales.items()]
        detail.append(f"log spread {spread:.2f} against a real log-cap "
                      f"range of about 8.2")
        check("Price levels are on a real scale",
              all(0.5 < v < 2 for v in scales.values()), "\n".join(detail))
        check("Scale factor is uniform across tickers", spread < 0.1,
              "A per-ticker factor corrupts every price-level feature:\n"
              "log_mktcap, size_proxy, dollar_vol_*, amihud_illiq_*.\n"
              "Constant over time, so the same stocks are misranked every month.")

    # The stronger version of the same question, because it needs no reference
    # price at all. Multiplying every stock by one constant shifts log-prices
    # without spreading them, so dispersion is invariant to a uniform scale and
    # only moves if the factor varies per ticker. Measured on index members:
    # dead names carry stale quotes at both extremes and the weight floor drops
    # them downstream anyway.
    wpath = DATA_DIR / "spx_weights.parquet"
    if wpath.exists():
        wt = pd.read_parquet(wpath)
        wt["date"] = pd.to_datetime(wt["date"])
        members = set(wt[(wt["spx_weight"] >= 5e-6) &
                         (wt["date"] >= wt["date"].max() - pd.DateOffset(months=6))
                         ]["ticker"])
        yr = prices["date"].dt.year.max()
        px = (prices[prices["date"].dt.year == yr].sort_values("date")
                     .groupby("ticker")["close"].last())
        px = px[(px > 0) & (px.index.isin(members))]
        if len(px) > 50:
            sd = float(np.log(px).std())
            excess = float(np.sqrt(max(sd ** 2 - 0.85 ** 2, 0.0)))
            check("Cross-sectional price dispersion is realistic", sd < 1.15,
                  f"log-sd {sd:.2f} on {len(px)} members against a real S&P 500 "
                  f"at about 0.85\n"
                  f"median ${px.median():,.0f}, p5 ${px.quantile(.05):,.0f}, "
                  f"p95 ${px.quantile(.95):,.0f}\n"
                  f"implied sd of the per-ticker factor: {excess:.2f}")

    check("No non-positive closes", bool((prices["close"] > 0).all()))

    # Volume feeds every liquidity factor, and those are the factors that
    # survive multiple-testing correction. Whether it is missing at random or
    # by ticker decides whether those factors mean anything.
    per_tkr = prices.groupby("ticker")["volume"].apply(lambda s: s.notna().mean())
    always = int((per_tkr > 0.9).sum())
    never = int((per_tkr < 0.1).sum())
    partial = len(per_tkr) - always - never
    check("Volume covers the universe", never == 0,
          f"{always} tickers always have it, {never} never do, {partial} partly\n"
          f"all-or-nothing by ticker, so dollar_vol, amihud, size_proxy and\n"
          f"kyle_lambda only vary across {always}/{len(per_tkr)} names and are\n"
          f"constant for the rest, which makes them partly an indicator of\n"
          f"whether the fetch succeeded rather than of liquidity")

    section("Index weights")

    w = pd.read_parquet(DATA_DIR / "spx_weights.parquet")
    w["date"] = pd.to_datetime(w["date"])
    live = w[w["spx_weight"] >= 5e-6]

    # Tolerance is loose on purpose. The sub-floor names dropped above account
    # for a few parts in 10,000, which is expected. A real failure here looks
    # like the 0.8138 that produced 3.84%/yr of phantom alpha, not 0.9997.
    sums = live.groupby("date")["spx_weight"].sum()
    check("Weights sum to 1 each month",
          bool(((sums - 1).abs() < 1e-3).all()),
          f"range {sums.min():.4f} to {sums.max():.4f}")

    # Winsorisation must survive the renormalisation that follows it, or the
    # cap is not a cap.
    last = live[live["date"] == live["date"].max()]["spx_weight"]
    top = float(last.max())
    n_tied = int(np.isclose(last, top).sum())
    check("No stock exceeds the 8% winsorisation cap", top <= 0.0801,
          f"largest is {top:.2%}, and {n_tied} stocks sit at exactly that value\n"
          f"clip-then-renormalise pushes capped names back above the ceiling")

    detail = []
    ok = True
    for yr, expected in PLAUSIBLE_TOP10.items():
        g = live[live["date"].dt.year == yr]
        if g.empty:
            continue
        d = g[g["date"] == g["date"].max()]["spx_weight"]
        got = float(d.nlargest(10).sum())
        detail.append(f"{yr}: top-10 {got:.0%} against a real {expected:.0%}")
        if got > expected * 1.4:
            ok = False
    check("Concentration resembles the real index", ok, "\n".join(detail))

    counts = live.groupby("date")["ticker"].nunique()
    check("Roughly 500 constituents", bool(counts.min() >= 450),
          f"{counts.min()} to {counts.max()} names; the index has 500, so "
          f"coverage is incomplete", warn=True)

    section("Panel")

    panel = pd.read_parquet(DATA_DIR / "panel_monthly_enriched.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    dup = int(panel.duplicated(subset=["date", "ticker"]).sum())
    check("No duplicate (date, ticker) rows", dup == 0, f"{dup} duplicates")

    # The target must be the NEXT month's return. If it is the current month's,
    # every model in the repo is reading the answer off the page.
    px = prices.set_index("date").groupby("ticker")["close"].resample("ME").last()
    px = px.reset_index().rename(columns={"close": "px"})
    px["ret_this"] = px.groupby("ticker")["px"].pct_change()
    px["ret_next"] = px.groupby("ticker")["ret_this"].shift(-1)
    chk = panel[["date", "ticker", "fwd_ret_1m"]].copy()
    chk["date"] = chk["date"] + pd.offsets.MonthEnd(0)
    px["date"] = px["date"] + pd.offsets.MonthEnd(0)
    m = chk.merge(px, on=["date", "ticker"], how="inner").dropna(
        subset=["fwd_ret_1m", "ret_this", "ret_next"])
    m = m[m["fwd_ret_1m"].abs() < 1.0]
    if len(m) > 1000:
        c_next = float(m["fwd_ret_1m"].corr(m["ret_next"]))
        c_this = float(m["fwd_ret_1m"].corr(m["ret_this"]))
        check("Target is next month, not this month",
              c_next > 0.95 and c_next > c_this,
              f"corr with next month {c_next:+.3f}, with current month {c_this:+.3f}\n"
              f"on {len(m):,} matched rows")

    exclude = {"date", "ticker", "fwd_ret_1m"}
    feats = [c for c in panel.columns if c not in exclude]
    flat = [c for c in feats if panel[c].nunique(dropna=True) <= 1]
    check("No constant feature columns", not flat,
          f"{len(flat)} columns hold one value across the whole panel:\n"
          + ", ".join(flat[:8]) + (" ..." if len(flat) > 8 else ""))

    monthly = [c for c in feats
               if c not in flat and panel.groupby("date")[c].nunique().max() <= 1]
    non_macro = [c for c in monthly if not any(
        k in c for k in ("vix", "yield", "spx_", "fed_", "recession",
                         "policy", "dollar_index", "credit", "market_"))]
    check("No stock features constant within a month", not non_macro,
          f"{len(non_macro)} cannot rank anything: " + ", ".join(non_macro),
          warn=True)

    section("Factor diagnostics")

    ic_path = DATA_DIR / "factor_ic_summary.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path)
        n_nan = int(ic["t_stat"].isna().sum())
        check("Every factor has a computable t-statistic", n_nan == 0,
              f"{n_nan} of {len(ic)} lack one, so they are silently dropped "
              f"before multiple-testing correction")

        # Any survivor computed from price levels inherits the scale problem.
        price_level = ["log_mktcap", "size_proxy", "dollar_vol", "amihud",
                       "kyle_lambda"]
        top = ic.dropna(subset=["t_stat"]).nlargest(20, "t_stat", keep="all")
        hits = [f for f in top["factor"]
                if any(k in str(f) for k in price_level)]
        check("Strongest factors do not depend on price levels", not hits,
              f"{len(hits)} of the top 20 are price-level derived: "
              + ", ".join(hits[:6]) + "\n"
              "These inherit the per-ticker scale factor above and need "
              "rechecking once prices are on a real scale.")

    print("\n" + "=" * 62)
    n_fail = sum(1 for t, _ in results if t == "FAIL")
    n_warn = sum(1 for t, _ in results if t == "WARN")
    print(f"{len(results)} checks | {n_fail} failed | {n_warn} warnings")
    if n_fail:
        print("\nFailed:")
        for t, name in results:
            if t == "FAIL":
                print(f"  - {name}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
