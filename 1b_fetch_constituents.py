"""
1b_fetch_constituents.py — Fix Survivorship Bias
============================================================
Downloads the full historical S&P 500 constituent list (1996–present) from
a free GitHub dataset, finds which tickers are missing from our prices.parquet,
and tries to fetch their price history via yfinance.

This fixes survivorship bias: our current prices.parquet only contains ~504
stocks that are CURRENTLY in the S&P 500. Companies that were removed between
2010–2025 (acquired, bankrupt, delisted) are missing — their absence inflates
backtest returns by ~1–2% per year.

Data source:
  https://github.com/fja05680/sp500
  CSV has one row per date, listing all S&P 500 members that day.
  Tickers with -YYYYMM suffix = removed from index in that month.

Steps:
  1. Download constituent CSV from GitHub
  2. Extract all unique historical tickers (2010–2025 scope)
  3. Find which are missing from current prices.parquet
  4. Fetch OHLCV via yfinance for each missing ticker
  5. Merge with existing prices.parquet → save updated file

Outputs:
  data/prices.parquet          — updated with historical members
  data/historical_tickers.csv  — full list of historical tickers + status

Runtime: ~20–40 min depending on number of missing tickers
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from io import StringIO

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
PRICES_IN  = DATA_DIR / "prices.parquet"
PRICES_OUT = DATA_DIR / "prices.parquet"   # overwrite with expanded version
TICKER_LOG = DATA_DIR / "historical_tickers.csv"

# ── Scope ──────────────────────────────────────────────────────────────────────
# Only fetch data for tickers that were S&P 500 members during our backtest window
BACKTEST_START = "2005-01-01"   # a few years before 2010 for warmup features
BACKTEST_END   = "2025-12-31"

# GitHub CSV URL
GITHUB_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)

MIN_MONTHS = 6   # skip tickers with fewer than this many months of data


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def strip_date_suffix(tk: str) -> str:
    """
    Historical constituent CSV uses TICKER-YYYYMM to indicate removal date.
    E.g. 'AAMRQ-201312' → 'AAMRQ',  'BF.B' → 'BF.B'
    """
    return re.sub(r"-\d{6}$", "", tk.strip())


def base_ticker(tk: str) -> str:
    """
    Strip exchange suffix from prices.parquet tickers.
    E.g. 'AAPL.O' → 'AAPL',  'BRK_B.N' → 'BRK_B'
    """
    if len(tk) >= 2 and tk[-2] == '.' and tk[-1].upper() in ('O', 'N', 'A'):
        return tk[:-2]
    return tk


def to_yf_ticker(tk: str) -> str:
    """
    Convert plain ticker to yfinance format.
    BRK.B → BRK-B,  BF.B → BF-B
    """
    return tk.replace('.', '-').replace('_', '-')


def fetch_prices_yf(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download OHLCV from yfinance. Returns long-format DataFrame matching
    prices.parquet schema, or None if unavailable / insufficient data.
    """
    yf_tk = to_yf_ticker(ticker)
    try:
        df = yf.download(
            yf_tk,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty or len(df) < MIN_MONTHS * 20:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.reset_index()

        # Keep only required columns
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"]
                if c in df.columns]
        df = df[keep].copy()
        df["ticker"] = ticker   # use plain ticker (no exchange suffix)
        df = df.dropna(subset=["close"])
        return df if len(df) > 0 else None

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("HISTORICAL S&P 500 CONSTITUENT FETCH")
    print("(Survivorship Bias Fix)")
    print("=" * 65)

    # ── Load existing prices ─────────────────────────────────────────────────
    print(f"\nLoading existing prices.parquet …")
    prices_existing = pq.read_table(PRICES_IN).to_pandas()
    prices_existing["date"] = pd.to_datetime(prices_existing["date"])
    existing_raw = sorted(prices_existing["ticker"].unique())

    # Base tickers (without exchange suffix) for comparison
    existing_base = {base_ticker(tk): tk for tk in existing_raw}
    print(f"  Existing: {len(existing_raw)} tickers | "
          f"{prices_existing['date'].min().date()} → {prices_existing['date'].max().date()}")

    # ── Download historical constituent list ─────────────────────────────────
    print(f"\nDownloading historical S&P 500 constituent list from GitHub …")
    try:
        if HAS_REQUESTS:
            resp = requests.get(GITHUB_URL, timeout=30)
            resp.raise_for_status()
            csv_text = resp.text
        else:
            import urllib.request
            with urllib.request.urlopen(GITHUB_URL, timeout=30) as r:
                csv_text = r.read().decode("utf-8")
        print("  Download successful.")
    except Exception as e:
        print(f"  ERROR downloading constituent list: {e}")
        print("  Cannot continue — check internet connection.")
        return

    # Parse CSV: date | tickers (comma-separated in quotes)
    const_df = pd.read_csv(StringIO(csv_text))
    const_df["date"] = pd.to_datetime(const_df["date"])

    # Filter to backtest scope: only rows within our date range
    scope_mask = (const_df["date"] >= BACKTEST_START) & \
                 (const_df["date"] <= BACKTEST_END)
    const_df = const_df[scope_mask].copy()
    print(f"  Constituent rows in scope ({BACKTEST_START[:4]}–{BACKTEST_END[:4]}): {len(const_df)}")

    # Extract all unique tickers that ever appeared in scope
    all_hist_tickers = set()
    for _, row in const_df.iterrows():
        tks = str(row["tickers"]).split(",")
        for tk in tks:
            clean = strip_date_suffix(tk)
            if clean and clean != "nan":
                all_hist_tickers.add(clean)

    print(f"  Unique historical tickers in scope: {len(all_hist_tickers)}")

    # ── Find missing tickers ─────────────────────────────────────────────────
    missing = []
    already_have = []
    for tk in sorted(all_hist_tickers):
        if tk in existing_base:
            already_have.append(tk)
        else:
            missing.append(tk)

    print(f"\n  Already in prices.parquet: {len(already_have)}")
    print(f"  Missing (need to fetch):   {len(missing)}")

    if not missing:
        print("\nNo missing tickers — prices.parquet already has full history.")
        return

    # ── Fetch missing tickers via yfinance ───────────────────────────────────
    print(f"\nFetching {len(missing)} missing tickers via yfinance …")
    print(f"(This may take 20–40 min)\n")

    new_frames = []
    results = []   # for logging

    for i, tk in enumerate(missing, 1):
        df = fetch_prices_yf(tk, BACKTEST_START, BACKTEST_END)
        status = "ok" if df is not None else "failed"
        n_rows = len(df) if df is not None else 0
        results.append({"ticker": tk, "status": status, "rows": n_rows})

        if df is not None:
            new_frames.append(df)

        if i % 25 == 0 or i == len(missing):
            n_ok   = sum(1 for r in results if r["status"] == "ok")
            n_fail = sum(1 for r in results if r["status"] == "failed")
            elapsed = (_time.time() - _t0) / 60
            print(f"  {i}/{len(missing)}  |  ok={n_ok}  failed={n_fail}  "
                  f"| {elapsed:.1f} min elapsed")

    # ── Summary of fetch results ─────────────────────────────────────────────
    ok_list   = [r["ticker"] for r in results if r["status"] == "ok"]
    fail_list = [r["ticker"] for r in results if r["status"] == "failed"]

    print(f"\nFetch complete:")
    print(f"  Succeeded: {len(ok_list)} / {len(missing)}")
    print(f"  Failed:    {len(fail_list)} / {len(missing)}")
    if fail_list:
        print(f"  Failed tickers (likely bankrupt/fully delisted):")
        for tk in fail_list[:30]:
            print(f"    {tk}")
        if len(fail_list) > 30:
            print(f"    ... and {len(fail_list) - 30} more")

    # ── Save ticker log ──────────────────────────────────────────────────────
    log_df = pd.DataFrame(results)
    log_df["in_existing"] = False
    existing_log = pd.DataFrame(
        [{"ticker": tk, "status": "existing", "rows": 0, "in_existing": True}
         for tk in already_have]
    )
    full_log = pd.concat([log_df, existing_log], ignore_index=True)
    full_log.to_csv(TICKER_LOG, index=False)
    print(f"\nSaved ticker log → {TICKER_LOG}")

    if not new_frames:
        print("\nNo new price data fetched — prices.parquet unchanged.")
        return

    # ── Merge with existing prices ────────────────────────────────────────────
    print(f"\nMerging {len(new_frames)} new ticker DataFrames with existing prices …")
    new_combined = pd.concat(new_frames, ignore_index=True)

    # Standardise columns to match existing prices.parquet
    # Existing prices.parquet may have exchange suffixes — new ones won't
    # We keep new tickers as plain (no suffix) since exchange is unknown
    col_order = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"]
                 if c in new_combined.columns or c in prices_existing.columns]

    # Align dtypes
    for col in ["open", "high", "low", "close", "volume"]:
        if col in new_combined.columns:
            new_combined[col] = pd.to_numeric(new_combined[col], errors="coerce")
        if col in prices_existing.columns:
            prices_existing[col] = pd.to_numeric(prices_existing[col], errors="coerce")

    merged = pd.concat([prices_existing, new_combined], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"  Old rows: {len(prices_existing):,}")
    print(f"  New rows: {len(new_combined):,}")
    print(f"  Total:    {len(merged):,}")
    print(f"  Tickers:  {merged['ticker'].nunique()}")

    # ── Save updated prices.parquet ──────────────────────────────────────────
    table = pa.Table.from_pandas(merged)
    pq.write_table(table, PRICES_OUT, compression="snappy")
    print(f"\nSaved → {PRICES_OUT}  ({len(merged):,} rows | "
          f"{merged['ticker'].nunique()} tickers)")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SURVIVORSHIP BIAS FIX COMPLETE")
    print(f"{'='*65}")
    print(f"  Original universe:  {len(existing_raw)} tickers (current S&P 500 only)")
    print(f"  Historical members: {len(all_hist_tickers)} unique tickers in scope")
    print(f"  Successfully added: {len(ok_list)} tickers")
    print(f"  Unavailable:        {len(fail_list)} tickers (bankrupt/fully delisted)")
    print(f"  Final universe:     {merged['ticker'].nunique()} tickers")
    print(f"\nNext step: re-run 1_feature_engineering.py to rebuild the panel.")
    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
