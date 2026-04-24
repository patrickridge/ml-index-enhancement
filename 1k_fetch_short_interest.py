"""
1k_fetch_short_interest.py - Fetch Short Interest Data
========================================================
Downloads short interest data from free sources:

  1. yfinance key_stats - provides current short interest, short ratio,
     and short % of float for individual tickers. Only snapshot data (no history),
     so we collect current values and append over time.

  2. FINRA bi-monthly short interest reports (if available via API).

These are cross-sectional factors: different for each stock on each date.

Academic basis:
  Drechsler & Drechsler (2016) - short interest cost and equity returns
  Asquith, Pathak & Ritter (2005) - short interest and stock returns
  Diether, Lee & Werner (2009) - short-selling and daily returns

Output:
  data/short_interest.parquet - columns: date, ticker, short_pct_float,
                                 short_interest_ratio, short_change_2w

Run time: ~10-20 min (yfinance ticker-by-ticker).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time as _time

_t0 = _time.time()
DATA_DIR  = Path("data")
OUT_PATH  = DATA_DIR / "short_interest.parquet"

print("=" * 65)
print("SHORT INTEREST DATA FETCH")
print("=" * 65)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    raise SystemExit(1)

# Load our ticker list
try:
    panel = pd.read_parquet(DATA_DIR / "panel_monthly.parquet", columns=["ticker"])
    our_tickers = panel["ticker"].unique().tolist()
    print(f"Pipeline tickers: {len(our_tickers)}")
except FileNotFoundError:
    print("[ERROR] panel_monthly.parquet not found - run 1h first.")
    raise SystemExit(1)


def strip_exchange_suffix(ticker: str) -> str:
    """AAPL.O → AAPL, MSFT.O → MSFT, BRK.B → BRK.B (keep share class)"""
    if ticker.endswith((".O", ".N", ".A", ".K", ".P", ".Z")):
        return ticker[:-2]
    return ticker


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH SHORT INTEREST FROM YFINANCE
# ═══════════════════════════════════════════════════════════════════════════════
# yfinance provides short interest data through Ticker.info dict:
#   - shortPercentOfFloat:  short interest as % of float
#   - sharesShortPriorMonth: short interest from prior month
#   - shortRatio:           days to cover (short interest / avg daily volume)
#   - sharesShort:          current short interest (shares)
#   - sharesShortPreviousMonthDate / sharesShortPriorMonth
#
# This gives us CURRENT snapshot + 1 month prior. Not ideal for a long backtest,
# but the factor can still be cross-sectionally ranked at any point in time.
#
# Strategy: fetch current snapshot for all tickers. If we already have a prior
# snapshot file, append to it for building time series.

print("\nFetching short interest from yfinance (ticker-by-ticker)...")
print("  This may take 10-20 min for 500+ tickers.\n")

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

        if not info or "shortPercentOfFloat" not in info:
            failed += 1
            continue

        rec = {
            "date": today,
            "ticker": tk,
            "short_pct_float": info.get("shortPercentOfFloat"),
            "short_ratio": info.get("shortRatio"),  # days to cover
            "shares_short": info.get("sharesShort"),
            "shares_short_prior": info.get("sharesShortPriorMonth"),
            "avg_volume": info.get("averageDailyVolume10Day") or info.get("averageVolume"),
            "float_shares": info.get("floatShares"),
        }

        # Compute short_change_2w proxy: current vs prior month
        if rec["shares_short"] and rec["shares_short_prior"] and rec["shares_short_prior"] > 0:
            rec["short_change_2w"] = (rec["shares_short"] / rec["shares_short_prior"]) - 1.0
        else:
            rec["short_change_2w"] = np.nan

        # short_squeeze_risk = short_pct_float / (avg_daily_volume_pct + eps)
        if rec["short_pct_float"] and rec["avg_volume"] and rec["float_shares"]:
            vol_pct = rec["avg_volume"] / rec["float_shares"] if rec["float_shares"] > 0 else np.nan
            rec["short_squeeze_risk"] = rec["short_pct_float"] / (vol_pct + 1e-8) if vol_pct else np.nan
        else:
            rec["short_squeeze_risk"] = np.nan

        # Also rename short_ratio → short_interest_ratio for pipeline
        rec["short_interest_ratio"] = rec.pop("short_ratio")

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


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

if not records:
    print("\n[ERROR] No short interest data fetched.")
    raise SystemExit(1)

current_df = pd.DataFrame(records)
current_df["date"] = pd.to_datetime(current_df["date"])

# ── Append to existing file if present (build time series over time) ─────────
if OUT_PATH.exists():
    print(f"\n  Found existing {OUT_PATH} - appending new snapshot...")
    existing = pd.read_parquet(OUT_PATH)
    existing["date"] = pd.to_datetime(existing["date"])
    combined = pd.concat([existing, current_df], ignore_index=True)
    # Deduplicate: keep latest per (ticker, date-month)
    combined["_ym"] = combined["date"].dt.to_period("M")
    combined = combined.sort_values("date").drop_duplicates(
        subset=["ticker", "_ym"], keep="last"
    ).drop(columns=["_ym"])
    result = combined
    print(f"  After dedup: {len(result)} rows")
else:
    result = current_df

# Select output columns
OUTPUT_COLS = [
    "short_pct_float", "short_interest_ratio", "short_change_2w", "short_squeeze_risk"
]
keep_cols = ["date", "ticker"] + [c for c in OUTPUT_COLS if c in result.columns]
# Also keep raw shares for future rebuilds
extra_cols = [c for c in ["shares_short", "shares_short_prior", "avg_volume", "float_shares"]
              if c in result.columns]
result = result[keep_cols + extra_cols]

# Drop rows where ALL factor columns are NaN
result = result.dropna(subset=[c for c in OUTPUT_COLS if c in result.columns], how="all")

# ── Save ──────────────────────────────────────────────────────────────────────
result.to_parquet(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
print(f"  Shape: {result.shape}")
print(f"  Tickers: {result['ticker'].nunique()}")
print(f"  Date range: {result['date'].min().date()} → {result['date'].max().date()}")

for col in OUTPUT_COLS:
    if col in result.columns:
        pct = result[col].notna().mean() * 100
        print(f"    {col:25s} {pct:5.1f}% filled | "
              f"mean={result[col].mean():.4f} | std={result[col].std():.4f}")

elapsed = _time.time() - _t0
print(f"\nDone in {elapsed:.1f}s")
print("\nNOTE: Short interest data is snapshot-only from yfinance.")
print("  Run this script periodically (e.g., bi-weekly) to build time series.")
print("  For historical data, consider Nasdaq Data Link (quandl) or FINRA API.")
