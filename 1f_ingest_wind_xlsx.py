"""
1f_ingest_wind_xlsx.py - Ingest Wind platform XLSX exports into prices.parquet
The Wind platform can export historical OHLCV data for any S&P 500
constituent (including delisted stocks). This script reads all XLSX files
from data/wind_exports/ and merges them into prices.parquet.

Wind export format (from 工作表1):
  Row 3: 开始日期  (start date)
  Row 4: 截止日期  (end date)
  Row 5: 证券代码  (security code, e.g. "AABA.O")
  Row 6: 证券简称  (security name)
  Row 7: column headers (Chinese)
  Row 8: column headers (English)
  Row 9+: data (Date, Open, High, Low, Close)

Usage:
  1. Export the missing tickers from Wind (one file per ticker, or multiple)
  2. Place all XLSX files in  data/wind_exports/
  3. Run:  python 1f_ingest_wind_xlsx.py
  4. Then: python 1g_rebuild_panel.py
           python 1h_feature_engineering.py

Also handles the single-ticker format (like missing data.xlsx placed anywhere
in the data/ folder if passed as --file argument).
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Paths
DATA_DIR    = Path("data")
WIND_DIR    = DATA_DIR / "wind_exports"       # drop xlsx files here
PRICES_IN   = DATA_DIR / "prices.parquet"
TICKER_LOG  = DATA_DIR / "historical_tickers.csv"

# Wind data starts from 2005 at the earliest; filter early junk
START_DATE  = "2005-01-01"


def parse_wind_xlsx(path: Path) -> tuple[str, pd.DataFrame] | None:
    """
    Parse a single Wind-exported XLSX file.

    Returns (ticker_str, df) where df has columns:
        date, open, high, low, close  (volume set to NaN - Wind doesn't export it)
    Returns None if parsing fails.
    """
    try:
        # Use openpyxl engine; data_only=True evaluates cached formula results
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheet = xl.sheet_names[0]

        # Read raw to extract metadata (ticker in row 5 col C, 0-indexed row 4)
        raw = pd.read_excel(
            path,
            sheet_name=sheet,
            header=None,
            engine="openpyxl"
        )

        # Extract ticker
        ticker = None
        for r in range(min(10, len(raw))):
            for c in range(min(5, raw.shape[1])):
                cell = str(raw.iloc[r, c]) if pd.notna(raw.iloc[r, c]) else ""
                # Wind ticker format: AAPL.O / MSFT.O / AABA.O etc.
                if "." in cell and any(
                    cell.upper().endswith(s) for s in (".O", ".N", ".A", ".OQ")
                ):
                    ticker = cell.strip()
                    break
            if ticker:
                break

        if ticker is None:
            # Try filename as fallback
            stem = path.stem.upper().replace(" ", "")
            if stem and stem != "MISSINGDATA":
                ticker = stem
                print(f"  Warning: ticker not found in {path.name}, using filename: {ticker}")
            else:
                print(f"  Error: cannot determine ticker from {path.name}")
                return None

        # Find data start row
        # Data starts after the bilingual header block.
        # Look for the first row where column B is a parseable date.
        data_start = None
        for r in range(len(raw)):
            val = raw.iloc[r, 1] if raw.shape[1] > 1 else None
            if val is None or pd.isna(val):
                continue
            try:
                ts = pd.to_datetime(val, errors="coerce")
                if pd.notna(ts) and ts.year >= 2000:
                    data_start = r
                    break
            except Exception:
                continue

        if data_start is None:
            print(f"  Error: no date data found in {path.name}")
            return None

        # Read data block
        data_raw = raw.iloc[data_start:].copy()
        data_raw = data_raw.iloc[:, 1:6]   # columns B–F: date, open, high, low, close

        data_raw.columns = ["date", "open", "high", "low", "close"]
        data_raw = data_raw.dropna(subset=["date"]).reset_index(drop=True)

        # Parse dates
        data_raw["date"] = pd.to_datetime(data_raw["date"], errors="coerce")
        data_raw = data_raw.dropna(subset=["date"])
        data_raw = data_raw[data_raw["date"] >= START_DATE]

        # Parse OHLC
        for col in ["open", "high", "low", "close"]:
            data_raw[col] = pd.to_numeric(data_raw[col], errors="coerce")

        # Drop rows with no close
        data_raw = data_raw.dropna(subset=["close"])
        # Drop rows where close = 0 (Wind sometimes puts 0 for unavailable dates)
        data_raw = data_raw[data_raw["close"] > 0]

        if len(data_raw) < 20:
            print(f"  Warning: only {len(data_raw)} valid rows in {path.name}")
            return None

        data_raw["volume"] = np.nan
        data_raw["ticker"] = ticker
        data_raw = data_raw.reset_index(drop=True)

        return ticker, data_raw[["date", "ticker", "open", "high", "low", "close", "volume"]]

    except Exception as e:
        print(f"  Error parsing {path.name}: {e}")
        return None


def main():
    print("=" * 65)
    print("INGEST WIND XLSX EXPORTS → prices.parquet")
    print("=" * 65)

    # Collect xlsx files
    xlsx_files = []

    # Check wind_exports/ folder
    if WIND_DIR.exists():
        xlsx_files.extend(sorted(WIND_DIR.glob("*.xlsx")))
        xlsx_files.extend(sorted(WIND_DIR.glob("*.xls")))

    # Also check for "missing data.xlsx" in data/ (ad-hoc single-file export)
    for candidate in sorted(DATA_DIR.glob("*.xlsx")):
        if candidate not in xlsx_files and candidate.name != "factor data.xlsx":
            xlsx_files.append(candidate)

    # Command-line override: python 1f_ingest_wind_xlsx.py path/to/file.xlsx
    if len(sys.argv) > 1:
        extra = Path(sys.argv[1])
        if extra.exists() and extra not in xlsx_files:
            xlsx_files = [extra] + xlsx_files

    if not xlsx_files:
        print(f"\nNo XLSX files found.")
        print(f"  Place Wind exports in:  {WIND_DIR}/")
        print(f"  Or run:  python 1f_ingest_wind_xlsx.py path/to/file.xlsx")
        return

    print(f"\nFound {len(xlsx_files)} XLSX file(s):")
    for f in xlsx_files:
        print(f"  {f}")

    # Load existing prices
    print(f"\nLoading {PRICES_IN} …")
    prices = pq.read_table(PRICES_IN).to_pandas()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    existing_tickers = set(prices["ticker"].unique())
    print(f"  {len(existing_tickers)} tickers | {len(prices):,} rows")

    # Parse all xlsx files
    new_dfs  = []
    ok_list  = []
    bad_list = []

    for fpath in xlsx_files:
        print(f"\nParsing: {fpath.name}")
        result = parse_wind_xlsx(fpath)
        if result is None:
            bad_list.append(fpath.name)
            continue
        ticker, df = result
        n_rows = len(df)
        date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
        print(f"  Ticker: {ticker}  |  {n_rows:,} rows  |  {date_range}")

        # If ticker already exists in prices, merge (fill gaps only)
        if ticker in existing_tickers:
            existing = prices[prices["ticker"] == ticker]["date"]
            new_dates = df[~df["date"].isin(existing)]
            if len(new_dates) == 0:
                print(f"  Already fully covered - skipping")
                continue
            else:
                print(f"  Adding {len(new_dates)} new rows (extending existing data)")
                df = new_dates
        else:
            print(f"  New ticker - adding all {n_rows} rows")

        new_dfs.append(df)
        ok_list.append(ticker)

    if not new_dfs:
        print("\nNo new data to add to prices.parquet.")
    else:
        # Merge
        print(f"\nMerging {len(new_dfs)} new dataset(s) …")
        new_data = pd.concat(new_dfs, ignore_index=True)

        # Ensure column schema matches existing prices
        existing_cols = list(prices.columns)
        for col in existing_cols:
            if col not in new_data.columns:
                new_data[col] = np.nan

        prices_updated = pd.concat(
            [prices[existing_cols], new_data[existing_cols]],
            ignore_index=True
        )
        prices_updated["date"] = pd.to_datetime(prices_updated["date"]).dt.tz_localize(None)
        prices_updated = prices_updated.sort_values(["ticker", "date"]).reset_index(drop=True)
        prices_updated = prices_updated.drop_duplicates(
            subset=["ticker", "date"], keep="last"
        ).reset_index(drop=True)

        n_new = prices_updated["ticker"].nunique()
        print(f"  Updated: {n_new} tickers | {len(prices_updated):,} rows")

        # Save
        date_col = pa.field("date", pa.timestamp("us"))
        table = pa.Table.from_pandas(prices_updated, preserve_index=False)
        # Ensure date has no timezone
        if pa.types.is_timestamp(table.schema.field("date").type):
            table = table.set_column(
                table.schema.get_field_index("date"),
                "date",
                table.column("date").cast(pa.timestamp("us"))
            )
        pq.write_table(table, PRICES_IN, compression="snappy")
        print(f"  Saved → {PRICES_IN}")

    # Update ticker log
    if ok_list and TICKER_LOG.exists():
        print("\nUpdating ticker log …")
        log_df = pd.read_csv(TICKER_LOG)
        log_df = log_df.set_index("ticker")
        for tk in ok_list:
            if tk in log_df.index:
                log_df.loc[tk, "status"] = "ok_wind"
            else:
                # New ticker not in log - add it
                log_df.loc[tk] = {"status": "ok_wind"}
        log_df = log_df.reset_index()
        log_df.to_csv(TICKER_LOG, index=False)
        print(f"  Updated {len(ok_list)} ticker(s) → status: ok_wind")

    # Summary
    print(f"\n{'='*65}")
    print("WIND INGEST COMPLETE")
    print(f"{'='*65}")
    print(f"  Parsed successfully : {len(ok_list)}  ({ok_list})")
    if bad_list:
        print(f"  Failed to parse    : {len(bad_list)}  ({bad_list})")

    if TICKER_LOG.exists():
        log_df = pd.read_csv(TICKER_LOG)
        print(f"\n  Ticker status summary:")
        print(log_df["status"].value_counts().to_string())
        n_still_missing = (log_df["status"] == "missing").sum()
        if n_still_missing > 0:
            print(f"\n  {n_still_missing} tickers still missing.")
            print("  Export these from Wind and place in data/wind_exports/")

    print(f"\nNext steps:")
    print("  python 1g_rebuild_panel.py")
    print("  python 1h_feature_engineering.py")
    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
