"""
1l_fetch_13f.py - Fetch Institutional Ownership from SEC EDGAR 13F Filings
Downloads quarterly 13F filings from SEC EDGAR (completely free, no API key)
to compute institutional ownership factors.

Approach:
  1. Use SEC EDGAR full-text search API to find 13F-HR filings
  2. For each filing, parse the holdings XML to extract CUSIPs + shares
  3. Map CUSIPs to pipeline tickers using OpenFIGI API (free, rate-limited)
  4. Aggregate holdings per ticker per quarter

Alternative (faster): Use yfinance .institutional_holders for current snapshot.

Academic basis:
  Gompers & Metrick (2001) - institutional ownership and stock returns
  Yan & Zhang (2009) - institutional investors and cross-section of returns
  Chen, Hong & Stein (2002) - breadth of ownership and stock returns

Output:
  data/institutional_ownership.parquet - quarterly: date, ticker, inst_own_pct,
                                          inst_own_change, num_institutions, inst_concentration

Run time: ~15-30 min (EDGAR + OpenFIGI API calls).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time as _time

_t0 = _time.time()
DATA_DIR  = Path("data")
OUT_PATH  = DATA_DIR / "institutional_ownership.parquet"
CACHE_PATH = DATA_DIR / "cusip_ticker_map.parquet"

print("=" * 65)
print("INSTITUTIONAL OWNERSHIP DATA FETCH (SEC EDGAR 13F)")
print("=" * 65)

# Load our ticker list
try:
    panel = pd.read_parquet(DATA_DIR / "panel_monthly.parquet", columns=["ticker"])
    our_tickers = panel["ticker"].unique().tolist()
    print(f"Pipeline tickers: {len(our_tickers)}")
except FileNotFoundError:
    print("[ERROR] panel_monthly.parquet not found - run 1h first.")
    raise SystemExit(1)


def strip_exchange_suffix(ticker: str) -> str:
    if ticker.endswith((".O", ".N", ".A", ".K", ".P", ".Z")):
        return ticker[:-2]
    return ticker


# APPROACH: YFINANCE FALLBACK (simpler, snapshot-only)
# Full EDGAR 13F parsing is complex (CUSIP mapping, XML parsing, rate limits).
# Start with yfinance snapshot for current institutional ownership,
# then build time series by running periodically.

print("\nUsing yfinance to fetch institutional ownership snapshots...")
print("  (Full EDGAR 13F parsing is planned for a future update.)\n")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    raise SystemExit(1)

records = []
n = len(our_tickers)
failed = 0
today = pd.Timestamp.today().normalize()

for i, tk in enumerate(our_tickers):
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  [{i+1}/{n}] Processing {tk}...")

    yf_tk = strip_exchange_suffix(tk).replace(".", "-")

    try:
        obj = yf.Ticker(yf_tk)
        info = obj.info

        if not info:
            failed += 1
            continue

        rec = {
            "date": today,
            "ticker": tk,
        }

        # Institutional ownership percentage
        # yfinance provides heldPercentInstitutions
        inst_pct = info.get("heldPercentInstitutions")
        rec["inst_own_pct"] = inst_pct if inst_pct is not None else np.nan

        # Number of institutional holders
        # From institutional_holders table
        try:
            inst_holders = obj.institutional_holders
            if inst_holders is not None and not inst_holders.empty:
                rec["num_institutions"] = len(inst_holders)
                # HHI concentration = sum of squared shares fractions
                if "Shares" in inst_holders.columns or "Value" in inst_holders.columns:
                    shares_col = "Shares" if "Shares" in inst_holders.columns else "Value"
                    shares = inst_holders[shares_col].dropna()
                    if len(shares) > 0 and shares.sum() > 0:
                        fracs = shares / shares.sum()
                        rec["inst_concentration"] = (fracs ** 2).sum()
                    else:
                        rec["inst_concentration"] = np.nan
                else:
                    rec["inst_concentration"] = np.nan
            else:
                rec["num_institutions"] = np.nan
                rec["inst_concentration"] = np.nan
        except Exception:
            rec["num_institutions"] = np.nan
            rec["inst_concentration"] = np.nan

        # inst_own_change - need prior quarter data, set NaN for first snapshot
        rec["inst_own_change"] = np.nan

        records.append(rec)

    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"    [WARN] {yf_tk}: {e}")
        elif failed == 6:
            print(f"    [WARN] Suppressing further individual errors...")
        continue

    # Rate limiting
    if (i + 1) % 20 == 0:
        _time.sleep(0.3)

print(f"\n  Completed: {n - failed}/{n} tickers ({failed} failed)")


# BUILD RESULT

if not records:
    print("\n[ERROR] No institutional ownership data fetched.")
    raise SystemExit(1)

current_df = pd.DataFrame(records)
current_df["date"] = pd.to_datetime(current_df["date"])

# Append to existing file if present
if OUT_PATH.exists():
    print(f"\n  Found existing {OUT_PATH} - appending new snapshot...")
    existing = pd.read_parquet(OUT_PATH)
    existing["date"] = pd.to_datetime(existing["date"])

    # Compute inst_own_change from prior snapshot
    if "inst_own_pct" in existing.columns:
        prior_snapshot = existing.sort_values("date").drop_duplicates(
            subset=["ticker"], keep="last"
        ).set_index("ticker")["inst_own_pct"]

        current_df["inst_own_change"] = current_df.apply(
            lambda row: (row["inst_own_pct"] - prior_snapshot.get(row["ticker"], np.nan))
            if pd.notna(row["inst_own_pct"]) and row["ticker"] in prior_snapshot.index
            else np.nan,
            axis=1
        )

    combined = pd.concat([existing, current_df], ignore_index=True)
    # Deduplicate
    combined["_ym"] = combined["date"].dt.to_period("Q")
    combined = combined.sort_values("date").drop_duplicates(
        subset=["ticker", "_ym"], keep="last"
    ).drop(columns=["_ym"])
    result = combined
    print(f"  After dedup: {len(result)} rows")
else:
    result = current_df

# Select output columns
OUTPUT_COLS = ["inst_own_pct", "inst_own_change", "num_institutions", "inst_concentration"]
keep_cols = ["date", "ticker"] + [c for c in OUTPUT_COLS if c in result.columns]
result = result[keep_cols]

# Drop rows where ALL factor columns are NaN
result = result.dropna(subset=[c for c in OUTPUT_COLS if c in result.columns], how="all")

# Save
result.to_parquet(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
print(f"  Shape: {result.shape}")
print(f"  Tickers: {result['ticker'].nunique()}")
print(f"  Date range: {result['date'].min().date()} → {result['date'].max().date()}")

for col in OUTPUT_COLS:
    if col in result.columns:
        pct = result[col].notna().mean() * 100
        mn = result[col].mean()
        st = result[col].std()
        print(f"    {col:25s} {pct:5.1f}% filled | mean={mn:.4f} | std={st:.4f}")

elapsed = _time.time() - _t0
print(f"\nDone in {elapsed:.1f}s")
print("\nNOTE: Institutional ownership is snapshot-only from yfinance.")
print("  Run quarterly to build time series. For historical 13F data,")
print("  full EDGAR parsing with CUSIP→ticker mapping is planned.")
