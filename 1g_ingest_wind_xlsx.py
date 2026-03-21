"""
1g_ingest_wind_xlsx.py — Ingest Wind Platform XLSX Exports
===========================================================
Parses OHLC price data exported from the Wind financial data platform
(Kieran's manual exports) and merges it into prices.parquet.

Wind xlsx file format
---------------------
Rows 0–7:   metadata / header block
Row 4:      col 2 = ticker  (e.g. "AABA.O", "ATVI.O", "PCLN.N")
Row 5:      col 2 = company name
Row 8+:     daily data rows
  col 0:    (empty / row label)
  col 1:    Date
  col 2:    Open
  col 3:    High
  col 4:    Low
  col 5:    Close
  (No volume column in Wind exports)

Ticker normalisation
--------------------
Wind uses Bloomberg-style exchange suffixes: .O (NASDAQ), .N (NYSE), .A (AMEX).
These are stripped to match the plain-ticker convention used in prices.parquet
for the 188 historical tickers added by 1d / 1f scripts.

Usage
-----
1. Drop Kieran's Wind xlsx files into  data/wind_exports/
2. Run:  python 1g_ingest_wind_xlsx.py
3. Script merges all newly parsed data into data/prices.parquet

Outputs:
  data/prices.parquet   — updated with Wind data
  (wind_exports/ folder is created automatically if missing)
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

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
WIND_DIR     = DATA_DIR / "wind_exports"
PRICES_PATH  = DATA_DIR / "prices.parquet"

# Wind metadata row indices (0-based)
ROW_TICKER   = 4   # ticker is in this row
ROW_NAME     = 5   # company name is in this row
COL_META     = 2   # column index for ticker / name values

# Data starts at this row
DATA_START_ROW = 8

# Wind data column positions within the data block (0-based within file)
COL_DATE  = 1
COL_OPEN  = 2
COL_HIGH  = 3
COL_LOW   = 4
COL_CLOSE = 5


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def strip_exchange_suffix(raw_ticker: str) -> str:
    """
    Remove Wind/Bloomberg exchange suffix from a ticker.
    Examples:
      'AABA.O'  → 'AABA'
      'PCLN.N'  → 'PCLN'
      'LUV.A'   → 'LUV'
      'BRK_B.N' → 'BRK_B'
      'AAPL'    → 'AAPL'   (no suffix, unchanged)
    Only strips single-letter suffixes .O / .N / .A (Wind exchange codes).
    """
    raw_ticker = str(raw_ticker).strip()
    if len(raw_ticker) >= 3 and raw_ticker[-2] == '.' and raw_ticker[-1].upper() in ('O', 'N', 'A'):
        return raw_ticker[:-2]
    return raw_ticker


def parse_wind_xlsx(path: Path) -> tuple[pd.DataFrame | None, str, str]:
    """
    Parse a single Wind xlsx export file.

    Returns (df, ticker, company_name) where df has columns:
      date, ticker, open, high, low, close, volume
    or (None, raw_ticker, company_name) on failure.
    """
    try:
        # Read the full sheet without any header processing so we control row access
        raw = pd.read_excel(path, header=None, dtype=str)
    except Exception as e:
        print(f"    ERROR reading {path.name}: {e}")
        return None, path.stem, ""

    # ── Extract ticker and company name from metadata rows ───────────────────
    try:
        raw_ticker   = str(raw.iloc[ROW_TICKER, COL_META]).strip()
        company_name = str(raw.iloc[ROW_NAME,   COL_META]).strip()
    except IndexError:
        print(f"    ERROR: {path.name} — metadata rows not at expected positions "
              f"(expected rows {ROW_TICKER}/{ROW_NAME}, col {COL_META})")
        return None, path.stem, ""

    # Guard against empty/nan metadata
    if raw_ticker in ("", "nan", "None"):
        print(f"    WARNING: {path.name} — ticker cell empty, skipping file")
        return None, path.stem, ""

    plain_ticker = strip_exchange_suffix(raw_ticker)

    # ── Extract data rows ────────────────────────────────────────────────────
    if len(raw) <= DATA_START_ROW:
        print(f"    WARNING: {path.name} ({plain_ticker}) — no data rows after header")
        return None, raw_ticker, company_name

    data_block = raw.iloc[DATA_START_ROW:].copy().reset_index(drop=True)

    # Select and rename the relevant columns
    try:
        df = pd.DataFrame({
            "date":  data_block.iloc[:, COL_DATE],
            "open":  data_block.iloc[:, COL_OPEN],
            "high":  data_block.iloc[:, COL_HIGH],
            "low":   data_block.iloc[:, COL_LOW],
            "close": data_block.iloc[:, COL_CLOSE],
        })
    except IndexError as e:
        print(f"    ERROR: {path.name} ({plain_ticker}) — unexpected column layout: {e}")
        return None, raw_ticker, company_name

    # ── Clean and coerce types ───────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where date or close is missing (Wind sometimes has trailing blank rows)
    df = df.dropna(subset=["date", "close"]).reset_index(drop=True)

    if df.empty:
        print(f"    WARNING: {path.name} ({plain_ticker}) — no valid rows after cleaning")
        return None, raw_ticker, company_name

    # ── Finalise schema ──────────────────────────────────────────────────────
    # Wind exports have no volume column; set to NaN to match prices.parquet schema
    df["volume"] = np.nan
    df["ticker"] = plain_ticker

    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df, raw_ticker, company_name


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("INGEST WIND PLATFORM XLSX EXPORTS")
    print("=" * 65)

    # ── Ensure wind_exports/ directory exists ────────────────────────────────
    WIND_DIR.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(WIND_DIR.glob("*.xlsx"))
    if not xlsx_files:
        print(f"\nNo xlsx files found in {WIND_DIR}/")
        print("Drop Kieran's Wind export files there and re-run.")
        print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")
        return

    print(f"\nFound {len(xlsx_files)} xlsx file(s) in {WIND_DIR}/")

    # ── Parse each xlsx file ─────────────────────────────────────────────────
    print("\nParsing files …")
    parsed_frames = []
    parse_log = []   # list of dicts for summary table

    for path in xlsx_files:
        df, raw_ticker, company_name = parse_wind_xlsx(path)

        if df is not None:
            parsed_frames.append(df)
            plain_ticker = df["ticker"].iloc[0]
            parse_log.append({
                "file":         path.name,
                "raw_ticker":   raw_ticker,
                "ticker":       plain_ticker,
                "company":      company_name,
                "rows":         len(df),
                "date_start":   df["date"].min().date(),
                "date_end":     df["date"].max().date(),
                "status":       "ok",
            })
            print(f"  {plain_ticker:12s}  {len(df):5,} rows  "
                  f"({df['date'].min().date()} → {df['date'].max().date()})  "
                  f"← {path.name}")
        else:
            parse_log.append({
                "file":       path.name,
                "raw_ticker": raw_ticker,
                "ticker":     strip_exchange_suffix(raw_ticker),
                "company":    company_name,
                "rows":       0,
                "status":     "failed",
            })

    n_ok     = sum(1 for r in parse_log if r["status"] == "ok")
    n_failed = sum(1 for r in parse_log if r["status"] == "failed")
    print(f"\nParsed: {n_ok} ok, {n_failed} failed")

    if not parsed_frames:
        print("No data extracted from any file — prices.parquet unchanged.")
        print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")
        return

    # ── Combine all parsed data ──────────────────────────────────────────────
    new_data = pd.concat(parsed_frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"])

    print(f"\nNew data combined:")
    print(f"  {new_data['ticker'].nunique()} ticker(s) | {len(new_data):,} rows")

    # ── Load existing prices.parquet ─────────────────────────────────────────
    if not PRICES_PATH.exists():
        print(f"\nWARNING: {PRICES_PATH} not found — creating new file from Wind data only.")
        prices_existing = pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )
    else:
        print(f"\nLoading {PRICES_PATH} …")
        prices_existing = pq.read_table(PRICES_PATH).to_pandas()
        prices_existing["date"] = pd.to_datetime(prices_existing["date"])
        print(f"  {prices_existing['ticker'].nunique()} tickers | "
              f"{len(prices_existing):,} rows | "
              f"{prices_existing['date'].min().date()} → {prices_existing['date'].max().date()}")

    # ── Merge ────────────────────────────────────────────────────────────────
    print("\nMerging …")

    # Coerce numeric columns in both frames before concat
    for col in ["open", "high", "low", "close", "volume"]:
        for frame in [new_data, prices_existing]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    merged = pd.concat([prices_existing, new_data], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])

    # Deduplicate: for any (ticker, date) overlap, keep the Wind row last
    # (new_data was appended after prices_existing, so keep="last" favours Wind)
    before_dedup = len(merged)
    merged = (
        merged
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )
    dropped = before_dedup - len(merged)
    if dropped:
        print(f"  Deduplicated: removed {dropped:,} overlapping (ticker, date) rows "
              f"(Wind data kept where both sources existed)")

    print(f"  Old rows: {len(prices_existing):,}")
    print(f"  New rows: {len(new_data):,}")
    print(f"  Total:    {len(merged):,}")
    print(f"  Tickers:  {merged['ticker'].nunique()}")

    # ── Save updated prices.parquet ──────────────────────────────────────────
    table = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(table, PRICES_PATH, compression="snappy")
    print(f"\nSaved → {PRICES_PATH}  ({len(merged):,} rows | "
          f"{merged['ticker'].nunique()} tickers)")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("WIND INGEST COMPLETE")
    print(f"{'='*65}")
    print(f"  xlsx files processed:  {len(xlsx_files)}")
    print(f"  Files parsed ok:       {n_ok}")
    print(f"  Files failed:          {n_failed}")

    if n_ok:
        new_tickers = [r["ticker"] for r in parse_log if r["status"] == "ok"]
        existing_tickers = set(prices_existing["ticker"].unique())
        truly_new = [t for t in new_tickers if t not in existing_tickers]
        updated   = [t for t in new_tickers if t in existing_tickers]
        print(f"\n  Tickers added (new):   {len(truly_new)}")
        for tk in truly_new:
            print(f"    {tk}")
        if updated:
            print(f"  Tickers updated (overlap merged): {len(updated)}")
            for tk in updated:
                print(f"    {tk}")

    print(f"\nNext step: re-run 1e_rebuild_base_panel.py then 1_feature_engineering.py")
    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
