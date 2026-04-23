"""
1d_fetch_missing_tickers.py — Recover Price Data for Failed Historical Tickers
===============================================================================
Script 1b_fetch_constituents.py fetched ~440 historical S&P 500
members but ~252 tickers failed, typically due to a yfinance timezone bug
(YFTzMissingError) or because the ticker was fully delisted.

This script attempts two recovery methods for each failed ticker:

  Method 1 — yfinance Ticker.history()
    Uses yf.Ticker(ticker).history() instead of yf.download(). This avoids
    the YFTzMissingError that affects yf.download() for some older tickers.

  Method 2 — Stooq via pandas_datareader
    Stooq is a free data source with good historical coverage of delisted US
    equities. No API key required. US tickers use format "{ticker}.us".

Inputs:
  data/historical_tickers.csv  — ticker log from 1d; rows with status='failed'
  data/prices.parquet          — existing price database

Outputs:
  data/prices.parquet          — updated with any newly recovered tickers
  data/historical_tickers.csv  — statuses updated ('failed' → 'ok' or 'missing')

Runtime: ~15–30 min depending on number of tickers recovered.
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

# Silence yfinance's own logging output
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

import re
import numpy as np
import pandas as pd
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

try:
    import pandas_datareader.data as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False
    print("WARNING: pandas_datareader not installed — Method 2 (Stooq) unavailable.")
    print("         Install with: pip install pandas-datareader")

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
PRICES_PATH    = DATA_DIR / "prices.parquet"
TICKER_LOG = DATA_DIR / "historical_tickers.csv"

# ── Fetch settings ──────────────────────────────────────────────────────────────
FETCH_START = "2005-01-01"
FETCH_END   = "2026-01-01"
MIN_ROWS    = 20   # minimum trading days required to accept a result

# Progress print every N tickers
PROGRESS_EVERY = 25


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def to_yf_ticker(tk: str) -> str:
    """
    Convert plain ticker to yfinance format.
    BRK.B → BRK-B,  BF.B → BF-B,  BRK_B → BRK-B
    """
    return tk.replace('.', '-').replace('_', '-')


def normalise_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Normalise an OHLCV DataFrame (from either source) to prices.parquet schema:
      columns: date, ticker, open, high, low, close, volume
    Returns None if the result is empty or lacks a 'close' column.
    """
    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns if yf.download() produced them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure date is a column (not the index)
    if df.index.name and "date" in df.index.name.lower():
        df = df.reset_index().rename(columns={df.index.name: "date"})
    elif "date" not in df.columns:
        df = df.reset_index()
        # rename first column to 'date' if needed
        first_col = df.columns[0]
        if first_col != "date":
            df = df.rename(columns={first_col: "date"})

    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    # Strip timezone info so concat with existing prices (tz-naive) works cleanly
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df = df.dropna(subset=["date"])

    if "close" not in df.columns:
        return None

    # Coerce numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["ticker"] = ticker   # plain ticker, no exchange suffix

    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["close"])

    return df if len(df) >= MIN_ROWS else None


# ── Method 1: yf.Ticker.history() ───────────────────────────────────────────────

def fetch_yf_history(ticker: str) -> pd.DataFrame | None:
    """
    Try yf.Ticker(ticker).history() — avoids the YFTzMissingError that
    affects yf.download() for some older / delisted tickers.
    """
    yf_tk = to_yf_ticker(ticker)
    try:
        tk_obj = yf.Ticker(yf_tk)
        # period="max" retrieves all available history; auto_adjust=True applies
        # corporate-action adjustments (splits, dividends) to OHLC columns.
        raw = tk_obj.history(period="max", auto_adjust=True, raise_errors=False)

        if raw is None or raw.empty:
            return None

        # Restrict to our date window
        raw.index = pd.to_datetime(raw.index, utc=False)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        raw = raw[(raw.index >= FETCH_START) & (raw.index < FETCH_END)]

        return normalise_ohlcv(raw, ticker)

    except Exception:
        return None


# ── Method 2: Stooq via pandas_datareader ───────────────────────────────────────

def fetch_stooq(ticker: str) -> pd.DataFrame | None:
    """
    Try pandas_datareader with the Stooq data source.
    US tickers use format "{ticker}.us" (lowercase).
    Stooq has good coverage of US-listed equities including many delisted names.
    """
    if not HAS_PDR:
        return None

    stooq_ticker = ticker.lower() + ".us"
    try:
        raw = pdr.DataReader(
            stooq_ticker,
            "stooq",
            start=FETCH_START,
            end=FETCH_END,
        )

        if raw is None or raw.empty:
            return None

        # Stooq returns data newest-first — sort ascending
        raw = raw.sort_index()

        return normalise_ohlcv(raw, ticker)

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("RECOVER MISSING TICKERS (YFTzMissingError Fix + Stooq)")
    print("=" * 65)

    # ── Load ticker log ──────────────────────────────────────────────────────
    if not TICKER_LOG.exists():
        print(f"\nERROR: {TICKER_LOG} not found.")
        print("Run 1b_fetch_constituents.py first.")
        return

    log_df = pd.read_csv(TICKER_LOG)
    failed_mask = log_df["status"] == "failed"
    failed_tickers = sorted(log_df.loc[failed_mask, "ticker"].tolist())
    n_total = len(failed_tickers)

    print(f"\nTicker log: {len(log_df)} entries")
    print(f"  Failed (to retry): {n_total}")
    print(f"  Stooq fallback:    {'enabled' if HAS_PDR else 'DISABLED (install pandas-datareader)'}")

    if n_total == 0:
        print("\nNo failed tickers to retry — nothing to do.")
        return

    # ── Load existing prices ─────────────────────────────────────────────────
    print(f"\nLoading {PRICES_PATH} …")
    prices_existing = pq.read_table(PRICES_PATH).to_pandas()
    prices_existing["date"] = pd.to_datetime(prices_existing["date"])
    print(f"  {prices_existing['ticker'].nunique()} tickers | "
          f"{len(prices_existing):,} rows")

    # ── Retry each failed ticker ─────────────────────────────────────────────
    print(f"\nRetrying {n_total} failed tickers …\n")

    new_frames = []
    retry_results = {}   # ticker → {'status': 'ok'/'missing', 'method': str, 'rows': int}

    for i, tk in enumerate(failed_tickers, 1):
        df = None
        method_used = None

        # Method 1: yfinance Ticker.history()
        try:
            df = fetch_yf_history(tk)
            if df is not None:
                method_used = "yf_history"
        except Exception:
            df = None

        # Method 2: Stooq (only if Method 1 failed)
        if df is None:
            try:
                df = fetch_stooq(tk)
                if df is not None:
                    method_used = "stooq"
            except Exception:
                df = None

        # Record result
        if df is not None:
            retry_results[tk] = {
                "status": "ok",
                "method": method_used,
                "rows": len(df),
            }
            new_frames.append(df)
        else:
            retry_results[tk] = {
                "status": "missing",   # truly unavailable after both methods
                "method": "none",
                "rows": 0,
            }

        # Progress counter
        if i % PROGRESS_EVERY == 0 or i == n_total:
            n_ok   = sum(1 for r in retry_results.values() if r["status"] == "ok")
            n_fail = i - n_ok
            elapsed = (_time.time() - _t0) / 60
            print(f"  {i}/{n_total}  |  ok={n_ok}  failed={n_fail}  "
                  f"| elapsed {elapsed:.1f}m")

    # ── Summary of retry results ─────────────────────────────────────────────
    ok_tickers      = [tk for tk, r in retry_results.items() if r["status"] == "ok"]
    still_failed    = [tk for tk, r in retry_results.items() if r["status"] == "missing"]
    via_yf_history  = [tk for tk, r in retry_results.items() if r["method"] == "yf_history"]
    via_stooq       = [tk for tk, r in retry_results.items() if r["method"] == "stooq"]

    print(f"\nRetry complete:")
    print(f"  Recovered:       {len(ok_tickers)} / {n_total}")
    print(f"    via yf.history:  {len(via_yf_history)}")
    print(f"    via Stooq:       {len(via_stooq)}")
    print(f"  Still missing:   {len(still_failed)} / {n_total}  "
          f"(bankrupt / fully delisted / not on Stooq)")

    if still_failed:
        # Show first 30 for diagnosis
        preview = still_failed[:30]
        print(f"\n  Still-missing tickers (first {len(preview)}):")
        for tk in preview:
            print(f"    {tk}")
        if len(still_failed) > 30:
            print(f"    … and {len(still_failed) - 30} more")

    # ── Update ticker log ────────────────────────────────────────────────────
    for idx, row in log_df.iterrows():
        tk = row["ticker"]
        if tk in retry_results:
            result = retry_results[tk]
            log_df.at[idx, "status"] = result["status"]
            if result["rows"] > 0:
                log_df.at[idx, "rows"] = result["rows"]

    log_df.to_csv(TICKER_LOG, index=False)
    print(f"\nUpdated ticker log → {TICKER_LOG}")

    # ── Merge recovered data with existing prices ────────────────────────────
    if not new_frames:
        print("\nNo new data recovered — prices.parquet unchanged.")
        print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")
        return

    print(f"\nMerging {len(new_frames)} recovered tickers into prices.parquet …")
    new_combined = pd.concat(new_frames, ignore_index=True)

    # Coerce numeric dtypes to be safe before concat
    for col in ["open", "high", "low", "close", "volume"]:
        if col in new_combined.columns:
            new_combined[col] = pd.to_numeric(new_combined[col], errors="coerce")
        if col in prices_existing.columns:
            prices_existing[col] = pd.to_numeric(prices_existing[col], errors="coerce")

    merged = pd.concat([prices_existing, new_combined], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])

    # Deduplicate: keep one row per (ticker, date)
    before_dedup = len(merged)
    merged = (
        merged
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )
    dropped = before_dedup - len(merged)
    if dropped:
        print(f"  Deduplicated: removed {dropped:,} duplicate (ticker, date) rows")

    print(f"  Old rows: {len(prices_existing):,}")
    print(f"  New rows: {len(new_combined):,}")
    print(f"  Total:    {len(merged):,}")
    print(f"  Tickers:  {merged['ticker'].nunique()}")

    # ── Save updated prices.parquet ──────────────────────────────────────────
    table = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(table, PRICES_PATH, compression="snappy")
    print(f"\nSaved → {PRICES_PATH}  ({len(merged):,} rows | "
          f"{merged['ticker'].nunique()} tickers)")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RECOVERY COMPLETE")
    print(f"{'='*65}")
    print(f"  Retried:    {n_total} tickers")
    print(f"  Recovered:  {len(ok_tickers)} tickers  "
          f"({len(via_yf_history)} via yf.history, {len(via_stooq)} via Stooq)")
    print(f"  Still lost: {len(still_failed)} tickers  (truly unavailable)")
    print(f"\nNext step: re-run 1g_rebuild_panel.py then 1h_feature_engineering.py")
    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
