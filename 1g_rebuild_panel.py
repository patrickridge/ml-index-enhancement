"""
1g_rebuild_panel.py - Rebuild panel_monthly.parquet from prices.parquet
The original panel_monthly.parquet was built from data.xlsx (Wind platform
export), which has been deleted. This script recreates it directly from
prices.parquet, now containing 692 tickers including historical S&P 500
members added by 1d_fetch_missing_tickers.py.

Columns produced (matching original panel_monthly.parquet schema):
  date        - month-end date
  ticker      - ticker symbol
  ret_1m      - 1-month return
  ret_3m      - 3-month return
  ret_6m      - 6-month return
  ret_12m     - 12-month return
  rev_1m      - reversal signal (-ret_1m, kept for backward compatibility)
  vol_20d     - 20-day realised volatility (annualised), sampled at month-end
  vol_60d     - 60-day realised volatility (annualised)
  vol_252d    - 252-day realised volatility (annualised)
  hl_range    - normalised high-low range (monthly avg daily (H-L)/close)
  fwd_ret_1m  - forward 1-month return (target variable)

Run AFTER 1d_fetch_missing_tickers.py.
Run BEFORE 1h_feature_engineering.py.
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Paths
DATA_DIR  = Path("data")
PRICES_IN = DATA_DIR / "prices.parquet"
PANEL_OUT = DATA_DIR / "panel_monthly.parquet"

START_DATE = "2005-01-01"


def main():
    print("=" * 65)
    print("REBUILD BASE PANEL FROM PRICES")
    print("=" * 65)

    # Load prices
    print(f"\nLoading {PRICES_IN} …")
    prices = pq.read_table(PRICES_IN).to_pandas()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["date"] >= START_DATE].copy()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = sorted(prices["ticker"].unique())
    print(f"  {len(tickers)} tickers | "
          f"{prices['date'].min().date()} → {prices['date'].max().date()}")

    # Daily returns
    print("\nComputing daily returns …")
    prices["ret_d"] = prices.groupby("ticker")["close"].pct_change()

    # Rolling volatility
    print("Computing rolling volatility (20d / 60d / 252d) …")
    for w, col in [(20, "vol_20d"), (60, "vol_60d"), (252, "vol_252d")]:
        prices[col] = (
            prices.groupby("ticker")["ret_d"]
            .transform(lambda x, w=w: x.rolling(w, min_periods=max(5, w // 4)).std() * np.sqrt(252))
        )

    # HL range
    if "high" in prices.columns and "low" in prices.columns:
        prices["hl_range_d"] = (prices["high"] - prices["low"]) / prices["close"].replace(0, np.nan)
    else:
        prices["hl_range_d"] = np.nan

    # Sample at month-end
    print("Sampling at month-end dates …")
    prices["ym"] = prices["date"].dt.to_period("M")
    month_ends = (
        prices.groupby(["ticker", "ym"])["date"].max()
        .reset_index().rename(columns={"date": "month_end"})
    )
    snap = prices.merge(
        month_ends,
        left_on=["ticker", "date", "ym"],
        right_on=["ticker", "month_end", "ym"],
        how="inner"
    ).copy()

    # Monthly average HL range
    hl_monthly = (
        prices.groupby(["ticker", "ym"])["hl_range_d"]
        .mean().reset_index()
        .rename(columns={"hl_range_d": "hl_range"})
    )
    snap = snap.merge(hl_monthly, on=["ticker", "ym"], how="left")
    snap = snap[["date", "ticker", "close",
                 "vol_20d", "vol_60d", "vol_252d", "hl_range"]].copy()
    snap = snap.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Monthly returns
    print("Computing monthly returns …")
    snap["ret_1m"]  = snap.groupby("ticker")["close"].pct_change(1)
    snap["ret_3m"]  = snap.groupby("ticker")["close"].pct_change(3)
    snap["ret_6m"]  = snap.groupby("ticker")["close"].pct_change(6)
    snap["ret_12m"] = snap.groupby("ticker")["close"].pct_change(12)
    snap["rev_1m"]  = -snap["ret_1m"]

    # Forward return (target)
    print("Computing forward 1-month return …")
    snap["fwd_ret_1m"] = snap.groupby("ticker")["ret_1m"].shift(-1)
    snap = snap.dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)

    # Output
    out_cols = ["date", "ticker", "ret_1m", "ret_3m", "ret_6m", "ret_12m",
                "rev_1m", "vol_20d", "vol_60d", "vol_252d", "hl_range", "fwd_ret_1m"]
    panel = snap[out_cols].copy()

    print(f"\nPanel built:")
    print(f"  Shape:   {panel.shape}")
    print(f"  Tickers: {panel['ticker'].nunique()}")
    print(f"  Months:  {panel['date'].nunique()}  "
          f"({panel['date'].min().date()} → {panel['date'].max().date()})")

    pq.write_table(pa.Table.from_pandas(panel, preserve_index=False),
                   PANEL_OUT, compression="snappy")
    print(f"\nSaved → {PANEL_OUT}  ({len(panel):,} rows)")
    print("Next: run 1h_feature_engineering.py")
    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
