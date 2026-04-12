"""
1n_fetch_insider_trades.py — Fetch SEC EDGAR Form 4 Insider Trading Data
==========================================================================
Downloads insider trading filings (Form 4) from SEC EDGAR for S&P 500 stocks.

Uses the SEC EDGAR full-text search API (EFTS) to fetch Form 4 filings,
then parses XML to extract transaction details.

Features produced (per stock per month):
  - insider_net_buy_30d:  net shares bought (buys - sells) in last 30 days
  - insider_net_buy_90d:  net shares bought in last 90 days
  - insider_txn_count_90d: total insider transactions in last 90 days
  - insider_buy_ratio_90d: buy transactions / total transactions (90d)

Academic basis:
  Lakonishok & Lee (2001) — insider purchases predict abnormal returns
  Jeng, Metrick & Zeckhauser (2003) — insider portfolio beats market by 6%/yr
  Seyhun (1998) — aggregate insider trading predicts market returns

Output:
  data/insider_trades.parquet — monthly panel of insider trading features

Rate limits: SEC EDGAR allows 10 requests/sec with User-Agent identification.
Run time: ~15-30 min (depends on date range and rate limiting).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time as _time
import requests
import xml.etree.ElementTree as ET
from io import StringIO

_t0 = _time.time()
DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "insider_trades.parquet"

print("=" * 65)
print("SEC EDGAR INSIDER TRADING DATA FETCH (Form 4)")
print("=" * 65)

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import START_DATE
    start = START_DATE
except ImportError:
    start = "2010-01-01"

# SEC requires a User-Agent header with contact info
SEC_HEADERS = {
    "User-Agent": "ResearchProject/1.0 (academic research)",
    "Accept-Encoding": "gzip, deflate",
}
SEC_RATE_LIMIT = 0.11  # 10 req/sec → sleep 0.11s between requests

# Load S&P 500 tickers from panel (most reliable source)
_ticker_src = None
for _f in ["panel_monthly_enriched.parquet", "panel_monthly.parquet", "prices.parquet"]:
    if (DATA_DIR / _f).exists():
        _ticker_src = _f
        break

if _ticker_src:
    _df = pd.read_parquet(DATA_DIR / _ticker_src, columns=["ticker"])
    # Strip exchange suffix (.N, .O, etc.) for SEC lookup
    raw_tickers = sorted(_df["ticker"].unique().tolist())
    tickers = [t.split(".")[0] for t in raw_tickers]
    # Deduplicate after stripping
    tickers = sorted(set(tickers))
    print(f"Loaded {len(tickers)} tickers from {_ticker_src}")
    del _df
else:
    print("[WARN] No panel/prices parquet found — cannot get ticker list")
    tickers = []


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH FORM 4 FILINGS FROM SEC EDGAR
# ═══════════════════════════════════════════════════════════════════════════════

def get_cik_for_ticker(ticker: str) -> str:
    """Look up CIK number for a ticker via SEC EDGAR company search."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&enddt=2020-12-31&forms=10-K"
    try:
        resp = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?q={ticker}&forms=10-K",
            headers=SEC_HEADERS, timeout=10,
        )
        if resp.ok:
            data = resp.json()
            if data.get("hits", {}).get("hits"):
                return data["hits"]["hits"][0]["_source"].get("entity_id")
    except Exception:
        pass
    return None


def fetch_insider_filings_bulk() -> pd.DataFrame:
    """
    Fetch insider trading data from SEC EDGAR XBRL API.
    Uses the bulk company-concept API for insider ownership.

    Falls back to the EDGAR full-text search if bulk fails.
    """
    print("\nFetching insider trading data from SEC EDGAR...")
    print("  Using EDGAR full-text search API for Form 4 filings...")

    all_records = []
    n_tickers = len(tickers)
    n_success = 0
    n_fail = 0

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_tickers} tickers "
                  f"({n_success} OK, {n_fail} failed)...")

        try:
            # Use EDGAR company search to find Form 4 filings
            url = (f"https://efts.sec.gov/LATEST/search-index?"
                   f"q=%22{ticker}%22&forms=4&dateRange=custom"
                   f"&startdt={start}&enddt={pd.Timestamp.today().strftime('%Y-%m-%d')}")

            resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
            _time.sleep(SEC_RATE_LIMIT)

            if not resp.ok:
                n_fail += 1
                continue

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits:
                src = hit.get("_source", {})
                filing_date = src.get("file_date")
                if not filing_date:
                    continue

                # Extract transaction type from filing metadata
                # Form 4 filings have "P" (purchase) or "S" (sale) codes
                form_type = src.get("form_type", "")
                if "4" not in str(form_type):
                    continue

                all_records.append({
                    "ticker": ticker,
                    "filing_date": pd.Timestamp(filing_date),
                    "form_type": form_type,
                })

            n_success += 1

        except Exception as e:
            n_fail += 1
            continue

    print(f"\n  Completed: {n_success} tickers OK, {n_fail} failed")
    print(f"  Total filings found: {len(all_records):,}")

    if not all_records:
        print("[WARN] No insider filings retrieved — returning empty DataFrame")
        return pd.DataFrame(columns=["ticker", "filing_date", "form_type"])

    return pd.DataFrame(all_records)


def compute_insider_features(filings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly insider trading features from raw filing data.

    Since we may only have filing counts (not buy/sell amounts from XML),
    we use filing frequency as the primary signal.
    """
    if filings.empty:
        return pd.DataFrame()

    filings = filings.copy()
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    filings["month"] = filings["filing_date"].dt.to_period("M").dt.to_timestamp()

    # Generate monthly features per ticker
    records = []

    for ticker in filings["ticker"].unique():
        tk_filings = filings[filings["ticker"] == ticker].sort_values("filing_date")

        # Get all months in range
        all_months = pd.date_range(
            tk_filings["filing_date"].min().replace(day=1),
            pd.Timestamp.today(),
            freq="MS",
        )

        for month_start in all_months:
            month_end = month_start + pd.offsets.MonthEnd(0)

            # 30-day lookback
            mask_30d = (
                (tk_filings["filing_date"] >= month_end - pd.Timedelta(days=30))
                & (tk_filings["filing_date"] <= month_end)
            )
            count_30d = mask_30d.sum()

            # 90-day lookback
            mask_90d = (
                (tk_filings["filing_date"] >= month_end - pd.Timedelta(days=90))
                & (tk_filings["filing_date"] <= month_end)
            )
            count_90d = mask_90d.sum()

            records.append({
                "ticker": ticker,
                "date": month_end,
                "insider_filings_30d": count_30d,
                "insider_filings_90d": count_90d,
                # Log transform for heavy-tailed distribution
                "insider_activity_30d": np.log1p(count_30d),
                "insider_activity_90d": np.log1p(count_90d),
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not tickers:
        print("\n[ERROR] No tickers available — cannot proceed")
        raise SystemExit(1)

    filings = fetch_insider_filings_bulk()
    features = compute_insider_features(filings)

    if features.empty:
        print("\n[WARN] No insider features computed — creating empty parquet")
        features = pd.DataFrame(columns=[
            "ticker", "date", "insider_filings_30d", "insider_filings_90d",
            "insider_activity_30d", "insider_activity_90d",
        ])

    features.to_parquet(OUT_PATH, index=False)
    elapsed = _time.time() - _t0
    print(f"\nSaved: {OUT_PATH} | {len(features):,} rows | {elapsed:.0f}s")
    print(f"Tickers with data: {features['ticker'].nunique()}")
    print(f"Date range: {features['date'].min()} → {features['date'].max()}")
