"""
1e_parse_wind_prices.py - Parse Wind Missing Data & Merge into prices.parquet
==============================================================================
Parses the Missing data.xlsx file (historical S&P 500 constituents with
OHLCV data from Wind) and merges valid tickers into data/prices.parquet.

This fixes survivorship bias: prices.parquet currently only covers tickers
that survived to be in the S&P 500 in 2024. The xlsx adds ~179 tickers that
were constituents in earlier years but have since been removed.

Excel structure (per ticker sheet):
  Rows 0-5  : Chinese/English header rows - skip
  Row 6+    : Daily OHLCV data
  Column 0  : Date
  Column 1  : Open
  Column 2  : High
  Column 3  : Low
  Column 4  : Close
  Column 5  : Volume

Excluded tickers:
  AW  - CBOT Dow Jones-UBS Commodity Index Futures (wrong asset class)
  ABC  - Italian company "CIA DELLA RUOTA" (wrong geography)

Run:
  python 1e_parse_wind_prices.py

Input:
  $DATA_XLSX (env var) - defaults to ~/Downloads/Missing data.xlsx
  data/prices.parquet

Output:
  data/prices.parquet  (updated in-place with new tickers appended)
  data/wind_parse_log.csv  (summary of what was added / skipped)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
XLSX_PATH  = Path(os.environ.get(
    "DATA_XLSX",
    str(Path.home() / "Downloads" / "Missing data.xlsx"),
))
PRICES_OUT = DATA_DIR / "prices.parquet"

# ── Tickers to exclude regardless of data presence ────────────────────────────
EXCLUDE_TICKERS = {"AW", "ABC"}

# ── Non-ticker sheets to skip ─────────────────────────────────────────────────
META_SHEETS = {"工作表1", "tickers_missing", "ticker still missing"}

MIN_VALID_ROWS = 10   # require at least 10 non-NaN close prices to include


def parse_ticker_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame | None:
    """
    Parse one ticker sheet.  Returns a clean DataFrame with columns:
      ticker, date, open, high, low, close, volume
    or None if the sheet has no usable data.
    """
    try:
        # header=None so we get positional columns; skip first 6 rows (headers)
        raw = xl.parse(sheet, header=None, skiprows=6)
    except Exception as e:
        print(f"  [{sheet}] Read error: {e}")
        return None

    if raw.empty or raw.shape[1] < 5:
        return None

    # Rename positional columns
    col_map = {0: "date", 1: "open", 2: "high", 3: "low", 4: "close"}
    if raw.shape[1] >= 6:
        col_map[5] = "volume"

    raw = raw.rename(columns=col_map)

    # Parse dates
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])

    # Parse OHLCV as numeric
    for col in ["open", "high", "low", "close"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if "volume" in raw.columns:
        raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    else:
        raw["volume"] = np.nan

    # Filter: need at least MIN_VALID_ROWS non-NaN closes
    valid_close = raw["close"].notna().sum()
    if valid_close < MIN_VALID_ROWS:
        return None

    # Drop rows where close is NaN (weekends/holidays leaked in)
    raw = raw.dropna(subset=["close"]).copy()

    raw["ticker"] = sheet
    cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    return raw[cols].reset_index(drop=True)


def main():
    print("=" * 65)
    print("1e_parse_wind_prices.py - Parse Wind OHLCV & Merge")
    print("=" * 65)

    if not XLSX_PATH.exists():
        print(f"\nERROR: Excel file not found at {XLSX_PATH}")
        print("Set DATA_XLSX env var or place file at that path.")
        return

    # ── Load existing prices ──────────────────────────────────────────────────
    print(f"\nLoading existing prices.parquet ...")
    if PRICES_OUT.exists():
        prices_existing = pd.read_parquet(PRICES_OUT)
        prices_existing["date"] = pd.to_datetime(prices_existing["date"])
        existing_tickers = set(prices_existing["ticker"].unique())
        print(f"  Existing: {len(existing_tickers)} tickers, "
              f"{len(prices_existing):,} rows")
    else:
        prices_existing  = pd.DataFrame()
        existing_tickers = set()
        print("  prices.parquet not found - will create fresh.")

    # ── Open Excel ────────────────────────────────────────────────────────────
    print(f"\nOpening {XLSX_PATH.name} ...")
    xl = pd.ExcelFile(XLSX_PATH)
    all_sheets    = xl.sheet_names
    ticker_sheets = [s for s in all_sheets
                     if s not in META_SHEETS and s not in EXCLUDE_TICKERS]
    print(f"  Total sheets: {len(all_sheets)}  |  "
          f"Ticker sheets to process: {len(ticker_sheets)}")

    # ── Parse each ticker sheet ───────────────────────────────────────────────
    print("\nParsing ticker sheets ...")
    new_frames = []
    log_rows   = []

    for i, sheet in enumerate(ticker_sheets, 1):
        if i % 25 == 0 or i == 1:
            print(f"  Processing {i}/{len(ticker_sheets)} - {sheet} ...")

        if sheet in existing_tickers:
            log_rows.append({"ticker": sheet, "status": "already_in_prices",
                             "rows_added": 0})
            continue

        df = parse_ticker_sheet(xl, sheet)
        if df is None:
            log_rows.append({"ticker": sheet, "status": "blank_or_invalid",
                             "rows_added": 0})
        else:
            new_frames.append(df)
            log_rows.append({"ticker": sheet, "status": "added",
                             "rows_added": len(df),
                             "date_min": str(df["date"].min().date()),
                             "date_max": str(df["date"].max().date())})

    xl.close()

    # ── Merge and save ────────────────────────────────────────────────────────
    added_tickers   = [r["ticker"] for r in log_rows if r["status"] == "added"]
    skipped_blank   = [r["ticker"] for r in log_rows if r["status"] == "blank_or_invalid"]
    already_present = [r["ticker"] for r in log_rows if r["status"] == "already_in_prices"]

    print(f"\n  Added:           {len(added_tickers)} new tickers")
    print(f"  Already present: {len(already_present)} tickers (skipped)")
    print(f"  Blank/invalid:   {len(skipped_blank)} tickers (skipped)")
    print(f"  Excluded:        {sorted(EXCLUDE_TICKERS)} (hardcoded exclusions)")

    if not new_frames:
        print("\nNothing new to add. prices.parquet unchanged.")
        return

    new_df = pd.concat(new_frames, ignore_index=True)
    print(f"\n  New rows to add: {len(new_df):,}")

    # Combine with existing
    if not prices_existing.empty:
        combined = pd.concat([prices_existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate (ticker + date)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "date"]).sort_values(
        ["ticker", "date"]).reset_index(drop=True)
    after = len(combined)
    if before != after:
        print(f"  Deduplicated: removed {before - after} duplicate rows")

    combined.to_parquet(PRICES_OUT, index=False)
    print(f"\nSaved -> {PRICES_OUT}")
    print(f"  Total tickers: {combined['ticker'].nunique()}")
    print(f"  Total rows:    {len(combined):,}")
    print(f"  Date range:    {combined['date'].min().date()} -> "
          f"{combined['date'].max().date()}")

    # ── Save log ──────────────────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_path = DATA_DIR / "wind_parse_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Saved -> {log_path}")

    # ── Print sample of added tickers ─────────────────────────────────────────
    if added_tickers:
        sample = added_tickers[:20]
        print(f"\nSample of added tickers: {sample}"
              + (" ..." if len(added_tickers) > 20 else ""))

    print("\nNext steps:")
    print("  1. python 1f_build_panel.py         - rebuild panel with new tickers")
    print("  2. python 1g_feature_engineering.py  - recompute all factors")
    print("  3. python 2a_factor_analysis.py      - re-run IC selection")
    print("  4. [Kaggle] 3d_cs_transformer_kaggle.py - retrain model")
    print("  5. python 5c_walk_forward.py         - re-run walk-forward backtest")
    print("\nDone.")


if __name__ == "__main__":
    main()
