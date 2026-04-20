"""
1n_fetch_insider_trades.py — SEC EDGAR Form 4 insider filings
===============================================================
For each S&P 500 ticker, looks up the SEC CIK from the official
ticker-to-CIK mapping, then pulls the filing history from the EDGAR
submissions API and counts Form 4 (insider transaction) filings per
month.

The output is filing-frequency only, not buy/sell dollars — the EDGAR
submissions endpoint doesn't return transaction amounts, just form
metadata. Frequency has been used as a stand-in in the literature
(more filings roughly = more insider activity around a name).

Features produced (per stock per month)
---------------------------------------
  insider_filings_30d   raw filing count in last 30 days
  insider_filings_90d   raw filing count in last 90 days
  insider_activity_30d  log1p of the 30-day count
  insider_activity_90d  log1p of the 90-day count

References
----------
  Lakonishok & Lee (2001) — insider purchases predict abnormal returns
  Jeng, Metrick & Zeckhauser (2003) — insider portfolio beats market
  Seyhun (1998) — aggregate insider trading predicts market returns

Output: data/insider_trades.parquet.
Rate limits: SEC allows 10 req/sec with a proper User-Agent.
Run time: ~15-30 min for ~500 tickers.

Environment
-----------
  SEC_USER_AGENT="Your Name your_email@example.com"   (required)
"""

import os
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

# SEC EDGAR requires a User-Agent header formatted as "Name Email".
# Set SEC_USER_AGENT in your environment, e.g.
#   export SEC_USER_AGENT="Your Name your_email@example.com"
_sec_ua = os.environ.get("SEC_USER_AGENT")
if not _sec_ua:
    print("[WARN] SEC_USER_AGENT not set — SEC EDGAR may return 403.")
    print("       Set it with: export SEC_USER_AGENT='Your Name your_email@example.com'")
    _sec_ua = "Anonymous User anonymous@example.com"

SEC_HEADERS = {
    "User-Agent": _sec_ua,
    "Accept-Encoding": "gzip, deflate",
}
SEC_HEADERS_DATA = dict(SEC_HEADERS)
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

def load_ticker_cik_map() -> dict:
    """
    Download SEC's official ticker→CIK mapping.
    Returns dict: { "AAPL": "0000320193", ... }
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # Format: { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"}, ... }
    mapping = {}
    for entry in data.values():
        tk = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)  # zero-pad to 10 digits
        mapping[tk] = cik
    return mapping


def fetch_insider_filings_bulk() -> pd.DataFrame:
    """
    Fetch insider trading data via SEC EDGAR submissions API.

    For each ticker:
    1. Look up CIK from SEC's official ticker map
    2. Fetch filing history from https://data.sec.gov/submissions/CIK{cik}.json
    3. Extract Form 4 filing dates
    """
    print("\nStep 1: Loading SEC ticker→CIK mapping...")
    try:
        cik_map = load_ticker_cik_map()
        print(f"  Loaded {len(cik_map):,} ticker→CIK mappings from SEC")
    except Exception as e:
        print(f"  [ERROR] Failed to load CIK mapping: {e}")
        return pd.DataFrame(columns=["ticker", "filing_date", "form_type"])

    print("\nStep 2: Fetching Form 4 filings per ticker...")
    all_records = []
    n_tickers = len(tickers)
    n_matched = 0
    n_success = 0
    n_fail = 0
    start_ts = pd.Timestamp(start)

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_tickers} tickers "
                  f"({n_success} with filings, {n_matched} CIK matched, "
                  f"{n_fail} no CIK)...")

        # Look up CIK
        cik = cik_map.get(ticker.upper())
        if cik is None:
            n_fail += 1
            continue
        n_matched += 1

        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=SEC_HEADERS_DATA, timeout=15)
            _time.sleep(SEC_RATE_LIMIT)

            if not resp.ok:
                continue

            data = resp.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])

            ticker_count = 0
            for form, date_str in zip(forms, dates):
                if form not in ("4", "4/A"):
                    continue
                fdate = pd.Timestamp(date_str)
                if fdate < start_ts:
                    continue
                all_records.append({
                    "ticker": ticker,
                    "filing_date": fdate,
                    "form_type": form,
                })
                ticker_count += 1

            if ticker_count > 0:
                n_success += 1

        except Exception:
            continue

    print(f"\n  CIK matched: {n_matched}/{n_tickers} | "
          f"Tickers with Form 4s: {n_success} | No CIK: {n_fail}")
    print(f"  Total Form 4 filings found: {len(all_records):,}")

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
