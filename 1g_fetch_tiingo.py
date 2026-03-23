"""
1g_fetch_tiingo.py — Fetch missing historical tickers via Tiingo API
=====================================================================
Tiingo has delisted/acquired stock data going back to 2005, making it
ideal for filling survivorship-bias gaps in our prices.parquet.

Free tier: 500 requests/hour, 50 requests/minute.
This script fetches in batches with rate limiting built in.

Run AFTER 1f_fetch_missing_tickers.py (which already tried yfinance + Stooq).
Run BEFORE 1e_rebuild_base_panel.py.
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
PRICES_IN  = DATA_DIR / "prices.parquet"
TICKER_LOG = DATA_DIR / "historical_tickers.csv"

API_KEY    = "641922103acc798dda62e76b562133ad24df602d"
START_DATE = "2005-01-01"
END_DATE   = "2026-01-01"

# Rate limiting: free tier = 500/hour, 50/minute
# Use 40/minute to be safe (1.5s between requests)
SLEEP_BETWEEN = 1.5   # seconds between API calls


def clean_ticker_tiingo(tk: str) -> str:
    """
    Convert exchange-suffixed ticker to Tiingo format.
    AAPL.O → aapl,  BRK.B.N → brk-b,  A.N → a
    Tiingo uses lowercase with hyphens (not dots).
    """
    # Strip trailing single-char exchange suffix (.O .N .A)
    if len(tk) >= 2 and tk[-2] == '.' and tk[-1].upper() in ('O', 'N', 'A'):
        tk = tk[:-2]
    # Replace remaining dots with hyphens (BRK.B → BRK-B)
    tk = tk.replace('.', '-')
    return tk.lower()


def fetch_tiingo(tk_orig: str) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV from Tiingo for one ticker.
    Returns DataFrame with columns: date, open, high, low, close, volume
    or None if unavailable.
    """
    tk = clean_ticker_tiingo(tk_orig)
    url = f"https://api.tiingo.com/tiingo/daily/{tk}/prices"
    params = {
        "startDate": START_DATE,
        "endDate":   END_DATE,
        "token":     API_KEY,
        "resampleFreq": "daily",
        "format":    "json",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            return "rate_limited"
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data)
        if df.empty:
            return None

        # Use adjusted columns (split/dividend adjusted)
        col_map = {
            "adjOpen":   "open",
            "adjHigh":   "high",
            "adjLow":    "low",
            "adjClose":  "close",
            "adjVolume": "volume",
        }
        # Fall back to unadjusted if adjusted not present
        for adj, raw in [("adjOpen","open"),("adjHigh","high"),("adjLow","low"),
                          ("adjClose","close"),("adjVolume","volume")]:
            if adj not in df.columns and raw in df.columns:
                col_map[raw] = raw
            elif adj in df.columns:
                col_map[adj] = raw

        df = df.rename(columns=col_map)
        keep = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].copy()

        # Fix date — strip timezone info
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

        # Add ticker
        df["ticker"] = tk_orig

        # Ensure numeric
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float64")

        # Drop rows with no close price
        df = df.dropna(subset=["close"])
        if len(df) < 20:   # need at least a month of data
            return None

        return df

    except Exception as e:
        return None


def main():
    print("=" * 65)
    print("FETCH MISSING TICKERS VIA TIINGO")
    print("=" * 65)

    # ── Load ticker log ──────────────────────────────────────────────────────
    log_df = pd.read_csv(TICKER_LOG)
    missing = log_df[log_df["status"] == "missing"]["ticker"].tolist()
    print(f"\nMissing tickers to fetch: {len(missing)}")

    # ── Load existing prices ─────────────────────────────────────────────────
    print(f"Loading {PRICES_IN} …")
    prices = pq.read_table(PRICES_IN).to_pandas()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    existing_tickers = set(prices["ticker"].unique())
    print(f"  Existing: {len(existing_tickers)} tickers, {len(prices):,} rows")

    # ── Fetch loop ──────────────────────────────────────────────────────────
    fetched_dfs  = []
    ok_tickers   = []
    still_missing = []
    rate_limited  = False

    print(f"\nFetching from Tiingo (≤40 req/min, ~{len(missing)*1.5/60:.0f} min estimated) …\n")

    for i, tk in enumerate(missing, 1):
        result = fetch_tiingo(tk)

        if result == "rate_limited":
            print(f"\n⚠️  Rate limit hit at ticker {i}/{len(missing)} ({tk})")
            print("   Waiting 65 seconds then retrying …")
            _time.sleep(65)
            result = fetch_tiingo(tk)
            if result == "rate_limited":
                print("   Still rate limited — stopping. Re-run in 1 hour.")
                still_missing.extend(missing[i-1:])
                rate_limited = True
                break

        if result is not None and isinstance(result, pd.DataFrame):
            fetched_dfs.append(result)
            ok_tickers.append(tk)
            status = f"✓  {len(result):5d} rows"
        else:
            still_missing.append(tk)
            status = "✗  not available"

        print(f"  [{i:3d}/{len(missing)}] {tk:<14} {status}")
        _time.sleep(SLEEP_BETWEEN)

    print(f"\n{'─'*50}")
    print(f"Tiingo results:  {len(ok_tickers)} fetched,  {len(still_missing)} not available")

    if not fetched_dfs:
        print("Nothing new to save.")
    else:
        # ── Merge with existing prices ───────────────────────────────────────
        print(f"\nMerging {len(fetched_dfs)} new ticker DataFrames …")
        new_data = pd.concat(fetched_dfs, ignore_index=True)

        # Ensure column schema matches
        existing_cols = list(prices.columns)
        for col in existing_cols:
            if col not in new_data.columns:
                new_data[col] = np.nan

        # Combine
        prices_updated = pd.concat(
            [prices[existing_cols], new_data[existing_cols]],
            ignore_index=True
        )
        prices_updated["date"] = pd.to_datetime(prices_updated["date"]).dt.tz_localize(None)
        prices_updated = prices_updated.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Remove any duplicates (same ticker + date)
        prices_updated = prices_updated.drop_duplicates(
            subset=["ticker", "date"], keep="last"
        ).reset_index(drop=True)

        n_tickers_new = prices_updated["ticker"].nunique()
        print(f"  Updated prices: {n_tickers_new} tickers, {len(prices_updated):,} rows")

        # ── Save ─────────────────────────────────────────────────────────────
        schema = pa.Schema.from_pandas(prices_updated, preserve_index=False)
        # Force date to timestamp[us] (no timezone)
        date_idx = schema.get_field_index("date")
        schema = schema.set(date_idx, pa.field("date", pa.timestamp("us")))

        table = pa.Table.from_pandas(prices_updated, schema=schema, preserve_index=False)
        pq.write_table(table, PRICES_IN, compression="snappy")
        print(f"  Saved → {PRICES_IN}")

    # ── Update ticker log ────────────────────────────────────────────────────
    print("\nUpdating ticker log …")
    log_df = log_df.set_index("ticker")
    for tk in ok_tickers:
        log_df.loc[tk, "status"] = "ok_tiingo"
    log_df.to_csv(TICKER_LOG)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("TIINGO FETCH COMPLETE")
    print(f"{'='*65}")
    log_df_reset = log_df.reset_index()
    print(f"  Status breakdown:")
    print(log_df_reset["status"].value_counts().to_string())

    n_still_missing = len(still_missing)
    if n_still_missing > 0:
        print(f"\n  {n_still_missing} tickers still missing (need Wind/CRSP manual export):")
        for tk in still_missing[:20]:
            print(f"    {tk}")
        if n_still_missing > 20:
            print(f"    … and {n_still_missing - 20} more")

    if rate_limited:
        print("\n⚠️  Hit rate limit — re-run in 1 hour to continue.")
    else:
        print("\nNext steps:")
        print("  1. Run 1e_rebuild_base_panel.py")
        print("  2. Run 1_feature_engineering.py")
        print("  3. Upload to Kaggle, retrain models")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
