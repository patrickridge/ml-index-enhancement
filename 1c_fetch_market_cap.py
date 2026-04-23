"""
1c_fetch_market_cap.py — Fetch market-cap weights for index enhancement
========================================================================
Gets current shares outstanding for every ticker via yfinance fast_info,
then multiplies by monthly close prices to build a time-varying market-cap
matrix. Saves monthly SPX-constituent weights to data/spx_weights.parquet.

Approach: use current shares outstanding as a constant proxy across all months.
For S&P 500 stocks, shares change slowly (buybacks/issuances are gradual), so
current shares × historical price is a good approximation of relative cap weights.
Weights are what matter for portfolio construction, not absolute market cap values.

Output
------
data/spx_weights.parquet
    date        : month-end timestamp
    ticker      : str  (original format, e.g. AAPL.O — matches panel/scores files)
    mktcap      : float  — proxy market cap (current shares × close price)
    spx_weight  : float  — fraction of total market cap that month (sums to ≈ 1)

Runtime: ~2–4 min  (one fast_info call per ticker)
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
PRICES_IN   = DATA_DIR / "prices.parquet"
OUT_WEIGHTS = DATA_DIR / "spx_weights.parquet"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_ticker(tk: str) -> str:
    """
    Convert exchange-suffixed ticker to yfinance format.
    Examples:  AAPL.O → AAPL,  A.N → A,  BRK.B.N → BRK-B
    """
    # Strip trailing single-char exchange suffix (.O NASDAQ, .N NYSE, .A AMEX)
    if len(tk) >= 2 and tk[-2] == '.' and tk[-1].upper() in ('O', 'N', 'A'):
        tk = tk[:-2]
    # Replace remaining dots with hyphens (BRK.B → BRK-B for Yahoo Finance)
    return tk.replace('.', '-')


def fetch_current_shares(ticker: str) -> float | None:
    """
    Fetch implied shares outstanding for one ticker via yfinance.

    Strategy: use fast_info.market_cap / fast_info.last_price to derive
    implied float shares. This is more reliable than fast_info.shares, which
    can return stale or non-float-adjusted values (causing WMT / NVDA distortion).

    Falls back to fast_info.shares, then info dict if market_cap unavailable.
    Returns float implied_shares or None if unavailable.
    """
    yf_tk = clean_ticker(ticker)
    try:
        t = yf.Ticker(yf_tk)
        fi = t.fast_info

        # Primary: implied shares = market_cap / last_price
        try:
            mktcap = fi.market_cap
            price  = fi.last_price
            if mktcap and price and mktcap > 0 and price > 0:
                return float(mktcap / price)
        except Exception:
            pass

        # Fallback 1: fast_info.shares
        try:
            shares = fi.shares
            if shares and shares > 0:
                return float(shares)
        except Exception:
            pass

        # Fallback 2: info dict
        info = t.info
        for key in ('sharesOutstanding', 'impliedSharesOutstanding', 'floatShares'):
            val = info.get(key)
            if val and val > 0:
                return float(val)
        return None
    except Exception:
        return None


def build_monthly_close(prices: pd.DataFrame) -> pd.DataFrame:
    """
    From daily OHLC, take the last close of each calendar month per ticker.
    Returns wide DataFrame: index=month-end dates, columns=original tickers.
    """
    prices["date"] = pd.to_datetime(prices["date"])
    monthly = (
        prices
        .set_index("date")
        .groupby("ticker")["close"]
        .resample("ME")
        .last()
        .unstack("ticker")
    )
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load prices ─────────────────────────────────────────────────────────────
    print("Loading prices.parquet …")
    prices  = pd.read_parquet(PRICES_IN)
    tickers = sorted(prices["ticker"].unique())
    print(f"  {len(tickers)} tickers | {prices['date'].min()} → {prices['date'].max()}")

    monthly_close = build_monthly_close(prices)
    print(f"  Monthly close: {monthly_close.shape[0]} months × {monthly_close.shape[1]} tickers")

    # ── Fetch current shares outstanding ─────────────────────────────────────────
    print(f"\nFetching current shares outstanding for {len(tickers)} tickers …")
    print("(~2–4 min)\n")

    shares_map = {}   # original_ticker → float shares
    failed     = []

    for i, tk in enumerate(tickers, 1):
        s = fetch_current_shares(tk)
        if s is not None:
            shares_map[tk] = s
        else:
            failed.append(tk)
        if i % 50 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)} done  |  ok={len(shares_map)}  failed={len(failed)}")

    print(f"\nShares fetched: {len(shares_map)} / {len(tickers)}")
    if failed:
        print(f"Failed tickers ({len(failed)}): {failed[:20]}{'…' if len(failed)>20 else ''}")

    if len(shares_map) == 0:
        print("\nERROR: No shares data fetched.")
        print(f"  Sample yfinance ticker: {clean_ticker(tickers[0])} (from {tickers[0]})")
        print("  Check internet connection and yfinance version.")
        return

    # ── Build market cap matrix ──────────────────────────────────────────────────
    # Market cap = current_shares × historical monthly price
    # Weights are relative so using current (constant) shares is a good approximation.
    print("\nBuilding market cap matrix …")

    # Align shares with monthly_close columns
    shares_series = pd.Series(shares_map)                    # index = original tickers
    common_tickers = monthly_close.columns.intersection(shares_series.index)

    close_m  = monthly_close[common_tickers]
    shares_s = shares_series[common_tickers]

    # Broadcast: each column × its constant shares count
    mktcap_wide = close_m.multiply(shares_s, axis="columns")

    # ── Convert to long format ───────────────────────────────────────────────────
    mktcap_long = (
        mktcap_wide
        .stack()
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker", 0: "mktcap"})
    )
    mktcap_long = mktcap_long.dropna(subset=["mktcap"])
    mktcap_long = mktcap_long[mktcap_long["mktcap"] > 0]

    # ── Compute SPX weights (cap-weighted fraction per month) ────────────────────
    total_cap = mktcap_long.groupby("date")["mktcap"].transform("sum")
    mktcap_long["spx_weight"] = mktcap_long["mktcap"] / total_cap

    # ── Winsorise: cap any single stock at 8% and renormalise ────────────────────
    # Prevents any outlier (wrong shares data) from dominating the benchmark.
    # Real S&P 500 largest holdings are ~6–7% (AAPL, NVDA, MSFT).
    CAP = 0.08
    mktcap_long["spx_weight"] = mktcap_long["spx_weight"].clip(upper=CAP)
    total_w = mktcap_long.groupby("date")["spx_weight"].transform("sum")
    mktcap_long["spx_weight"] = mktcap_long["spx_weight"] / total_w

    # ── Coverage diagnostics ─────────────────────────────────────────────────────
    print("\nCoverage check:")
    n_per_month = mktcap_long.groupby("date")["ticker"].count()
    print(f"  Tickers covered (shares fetched) : {len(shares_map)}")
    print(f"  Avg tickers per month            : {n_per_month.mean():.0f}")
    wsum = mktcap_long[mktcap_long["date"] == mktcap_long["date"].max()]["spx_weight"].sum()
    print(f"  Weight sum (most recent month)   : {wsum:.4f}")

    # Top-10 constituents in most recent month
    latest = mktcap_long[mktcap_long["date"] == mktcap_long["date"].max()]
    top10  = latest.nlargest(10, "spx_weight")[["ticker", "spx_weight"]]
    print(f"\nTop 10 by market cap ({mktcap_long['date'].max().date()}):")
    for _, row in top10.iterrows():
        print(f"  {row['ticker']:<10}  {row['spx_weight']*100:.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────────────
    mktcap_long[["date", "ticker", "mktcap", "spx_weight"]].to_parquet(
        OUT_WEIGHTS, index=False
    )
    print(f"\nSaved → {OUT_WEIGHTS}  ({len(mktcap_long):,} rows)")
    print("Run 4b_index_enhancement.py next.")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
