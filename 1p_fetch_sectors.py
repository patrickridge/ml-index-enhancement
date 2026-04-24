"""
1p_fetch_sectors.py - Fetch GICS sector mapping for all tickers
==================================================================
One-off script to build a ticker → GICS sector mapping by calling
yfinance `.info` for each ticker. Caches to data/sectors.parquet.

Needed by the macro-sector fundamental agent (4f_agent_ensemble_backtest.py).

Run time: ~5-10 min for ~500 tickers.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
import time as _time

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "sectors.parquet"

print("=" * 65)
print("FETCH SECTOR MAPPING (via yfinance)")
print("=" * 65)

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] pip install yfinance")
    raise SystemExit(1)

# Load tickers from the panel
panel = pd.read_parquet(DATA_DIR / "panel_monthly_enriched.parquet", columns=["ticker"])
raw_tickers = sorted(panel["ticker"].unique().tolist())
# Strip exchange suffix for yfinance lookups
ticker_map = {t: t.split(".")[0] for t in raw_tickers}
unique_bases = sorted(set(ticker_map.values()))
print(f"Loaded {len(raw_tickers)} panel tickers ({len(unique_bases)} unique bases)")

# ── Resume from existing cache if present ────────────────────────────────────
if OUT_PATH.exists():
    existing = pd.read_parquet(OUT_PATH)
    done_bases = set(existing["base_ticker"].unique())
    print(f"Cache found: {len(existing)} tickers already fetched")
else:
    existing = pd.DataFrame(columns=["base_ticker", "sector", "industry"])
    done_bases = set()

to_fetch = [b for b in unique_bases if b not in done_bases]
print(f"Remaining to fetch: {len(to_fetch)}")

records = existing.to_dict("records")
for i, base in enumerate(to_fetch):
    if (i + 1) % 25 == 0:
        print(f"  Progress: {i+1}/{len(to_fetch)} - {base}")
    try:
        info = yf.Ticker(base).info
        sector   = info.get("sector")
        industry = info.get("industry")
    except Exception:
        sector, industry = None, None
    records.append({"base_ticker": base, "sector": sector, "industry": industry})

    # Save every 50 to be safe
    if (i + 1) % 50 == 0:
        pd.DataFrame(records).to_parquet(OUT_PATH, index=False)

# Final save - also write a ticker-level table (with exchange suffix) for merge ease
sectors = pd.DataFrame(records).drop_duplicates("base_ticker")

expanded = []
for t in raw_tickers:
    base = ticker_map[t]
    row = sectors[sectors["base_ticker"] == base]
    if len(row) > 0:
        expanded.append({
            "ticker":   t,
            "base":     base,
            "sector":   row.iloc[0]["sector"],
            "industry": row.iloc[0]["industry"],
        })
    else:
        expanded.append({"ticker": t, "base": base, "sector": None, "industry": None})

out = pd.DataFrame(expanded)
out.to_parquet(OUT_PATH, index=False)

n_cov = out["sector"].notna().sum()
print(f"\nSaved: {OUT_PATH}")
print(f"Coverage: {n_cov}/{len(out)} tickers with sector ({n_cov/len(out)*100:.0f}%)")
print(f"\nSector distribution:")
print(out["sector"].value_counts().to_string())
