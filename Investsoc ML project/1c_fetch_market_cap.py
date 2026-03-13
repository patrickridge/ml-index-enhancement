"""
1c_fetch_market_cap.py — Fetch historical market-cap weights for index enhancement
====================================================================================
Downloads quarterly shares-outstanding history for every ticker in prices.parquet
via yfinance, multiplies by monthly close price, and saves monthly SPX-constituent
weights to data/spx_weights.parquet.

These weights are consumed by 6_index_enhancement.py to build the index-enhancement
portfolio (overweight high-scoring stocks, underweight low-scoring ones vs the index).

Output
------
data/spx_weights.parquet
    date        : month-end timestamp  (e.g. 2023-01-31)
    ticker      : str
    mktcap      : float  — market cap in USD (shares × close)
    spx_weight  : float  — fraction of total market cap that month  (sums to ≈ 1)

Runtime: ~10 min on a standard broadband connection (504 tickers × 1 API call each)
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
PRICES_IN   = DATA_DIR / "prices.parquet"
OUT_WEIGHTS = DATA_DIR / "spx_weights.parquet"

# ── Date range ─────────────────────────────────────────────────────────────────
START_DATE = "2009-01-01"   # one year before pipeline start to ensure coverage
END_DATE   = "2026-01-01"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_shares(ticker: str, start: str, end: str) -> pd.Series | None:
    """
    Return a monthly (ME) series of shares outstanding for one ticker.
    yfinance get_shares_full() returns sporadic dates (quarterly filings).
    We resample to month-end and forward-fill.
    Returns None if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        shares = t.get_shares_full(start=start, end=end)
        if shares is None or len(shares) == 0:
            return None
        shares = shares.sort_index()
        # Resample to month-end, forward-fill within the date range
        idx = pd.date_range(start=start, end=end, freq="ME")
        shares = shares.reindex(shares.index.union(idx)).ffill().reindex(idx)
        shares.name = ticker
        return shares
    except Exception:
        return None


def build_monthly_close(prices: pd.DataFrame) -> pd.DataFrame:
    """
    From daily OHLC, take the last close of each calendar month per ticker.
    Returns wide DataFrame: index=month-end dates, columns=tickers.
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

    # ── Fetch shares outstanding ─────────────────────────────────────────────────
    print(f"\nFetching shares outstanding for {len(tickers)} tickers via yfinance …")
    print("(this takes ~10 min — one API call per ticker)\n")

    shares_dict = {}
    failed      = []
    for i, tk in enumerate(tickers, 1):
        s = fetch_shares(tk, START_DATE, END_DATE)
        if s is not None:
            shares_dict[tk] = s
        else:
            failed.append(tk)
        if i % 50 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)} done  |  ok={len(shares_dict)}  failed={len(failed)}")

    print(f"\nShares fetched: {len(shares_dict)} / {len(tickers)}")
    if failed:
        print(f"Failed tickers ({len(failed)}): {failed[:20]}{'…' if len(failed)>20 else ''}")

    # ── Build market cap matrix ──────────────────────────────────────────────────
    print("\nBuilding market cap matrix …")
    shares_wide = pd.DataFrame(shares_dict)                # rows=months, cols=tickers

    # Align index
    common_dates   = monthly_close.index.intersection(shares_wide.index)
    common_tickers = monthly_close.columns.intersection(shares_wide.columns)

    close_m  = monthly_close.loc[common_dates, common_tickers]
    shares_m = shares_wide.loc[common_dates, common_tickers]

    mktcap_wide = close_m * shares_m      # elementwise: price × shares = market cap

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

    # ── Coverage diagnostics ─────────────────────────────────────────────────────
    print("\nCoverage check:")
    n_tickers_per_month = mktcap_long.groupby("date")["ticker"].count()
    print(f"  Avg tickers per month  : {n_tickers_per_month.mean():.0f}")
    print(f"  Min tickers per month  : {n_tickers_per_month.min()} ({n_tickers_per_month.idxmin().date()})")
    print(f"  Max tickers per month  : {n_tickers_per_month.max()} ({n_tickers_per_month.idxmax().date()})")
    print(f"  Weight sum check (sample month): "
          f"{mktcap_long[mktcap_long['date'] == mktcap_long['date'].max()]['spx_weight'].sum():.4f}")

    # Show top-10 constituents in most recent month
    latest = mktcap_long[mktcap_long["date"] == mktcap_long["date"].max()]
    top10  = latest.nlargest(10, "spx_weight")[["ticker", "spx_weight"]]
    print(f"\nTop 10 by market cap ({mktcap_long['date'].max().date()}):")
    for _, row in top10.iterrows():
        print(f"  {row['ticker']:<8}  {row['spx_weight']*100:.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────────────
    mktcap_long[["date", "ticker", "mktcap", "spx_weight"]].to_parquet(
        OUT_WEIGHTS, index=False
    )
    print(f"\nSaved → {OUT_WEIGHTS}  ({len(mktcap_long):,} rows)")
    print("Run 6_index_enhancement.py next.")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
