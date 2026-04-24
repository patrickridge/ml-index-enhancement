"""
1m_fetch_prediction_markets.py - Fetch Prediction Market / Fed Futures Data
Downloads forward-looking market expectations from free sources:

  1. CME Fed Funds Futures (via yfinance) - implied Fed rate expectations
     going back to 2000s. The spread between contract months implies
     market-expected rate changes.

  2. Treasury yield curve - recession probability proxy from 10Y-2Y spread
     (already partially in Cat 10, but here we compute a calibrated probability).

  3. VIX term structure - VIX futures spread (contango vs backwardation)
     as a forward-looking risk sentiment indicator.

These are macro-level time-signal factors: same for all stocks in a given month.
They get TS z-scored (not CS-ranked) and added to MACRO_COLS.

Academic basis:
  Cieslak & Povala (2016) - Fed rate expectations predict equity returns
  Wolfers & Zitzewitz (2004) - prediction market price = probability
  Baker, Bloom & Davis (2016) - policy uncertainty and equity returns
  Adrian & Estrella (2010) - yield curve predicts recessions

Output:
  data/prediction_markets.parquet - daily time series of probability/expectation factors

Run time: ~1-2 min (yfinance downloads).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time as _time

_t0 = _time.time()
DATA_DIR  = Path("data")
OUT_PATH  = DATA_DIR / "prediction_markets.parquet"

print("=" * 65)
print("PREDICTION MARKET / FED EXPECTATIONS DATA FETCH")
print("=" * 65)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    raise SystemExit(1)

# Date range (match pipeline)
try:
    from config import START_DATE
    start = START_DATE
except ImportError:
    start = "2005-01-01"
end = pd.Timestamp.today().strftime("%Y-%m-%d")


# 1. FED FUNDS RATE EXPECTATIONS
print("\n1. Fetching Fed Funds Effective Rate (DFF from FRED proxy)...")
# Use the 1-month and 3-month T-bill rates as Fed expectation proxies
# Fed Funds Rate itself from yfinance: ^IRX (13-week T-bill) as close proxy
# The spread between 3m and current effective rate implies expected change

tickers_fed = {
    "^IRX": "tbill_3m",     # 13-week T-bill yield
    "^FVX": "yield_5y",     # 5-year treasury yield
}

fed_frames = []
for tk, name in tickers_fed.items():
    try:
        data = yf.download(tk, start=start, end=end, progress=False)
        if data.empty:
            print(f"  [WARN] {tk} ({name}) - no data")
            continue
        s = data["Close"]
        # yfinance may return MultiIndex columns - flatten
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.dropna()
        s.name = name
        fed_frames.append(s)
        print(f"  {tk} ({name}): {len(s)} rows")
    except Exception as e:
        print(f"  [WARN] {tk} failed: {e}")

# Also fetch Fed Funds Futures (ZQ=F) if available
print("\n  Fetching Fed Funds Futures (ZQ=F)...")
try:
    zq = yf.download("ZQ=F", start=start, end=end, progress=False)
    if not zq.empty:
        zq_close = zq["Close"]
        if isinstance(zq_close, pd.DataFrame):
            zq_close = zq_close.iloc[:, 0]
        # Fed Funds Futures price: 100 - implied rate
        # So implied rate = 100 - price
        zq_close = zq_close.dropna()
        implied_rate = 100 - zq_close
        implied_rate.name = "ff_implied_rate"
        fed_frames.append(implied_rate)
        print(f"  ZQ=F (Fed Funds Futures): {len(implied_rate)} rows")
    else:
        print("  [WARN] ZQ=F - no data (may need futures data subscription)")
except Exception as e:
    print(f"  [WARN] ZQ=F failed: {e}")


# 2. YIELD CURVE RECESSION PROBABILITY
print("\n2. Fetching yield curve data for recession probability...")
# 10Y-2Y spread → calibrated recession probability using probit model
# Adrian & Estrella (2010): P(recession in 12m) = Φ(-0.57 + (-0.83) × spread)
# where spread = 10Y - 2Y yield

tickers_yc = {
    "^TNX": "yield_10y_raw",  # 10-year treasury
    "2YY=F": "yield_2y_raw",  # 2-year treasury (futures proxy)
}

yc_frames = []
for tk, name in tickers_yc.items():
    try:
        data = yf.download(tk, start=start, end=end, progress=False)
        if data.empty:
            continue
        s = data["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = name
        yc_frames.append(s)
        print(f"  {tk} ({name}): {len(s)} rows")
    except Exception as e:
        print(f"  [WARN] {tk} failed: {e}")


# 3. VIX TERM STRUCTURE (contango/backwardation)
print("\n3. Fetching VIX term structure...")
# VIX spot vs VIX 3-month futures - contango = complacency, backwardation = fear

tickers_vix = {
    "^VIX":  "vix_spot",
    "^VIX3M": "vix_3m",   # CBOE 3-month VIX
}

vix_frames = []
for tk, name in tickers_vix.items():
    try:
        data = yf.download(tk, start=start, end=end, progress=False)
        if data.empty:
            continue
        s = data["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = name
        vix_frames.append(s)
        print(f"  {tk} ({name}): {len(s)} rows")
    except Exception as e:
        print(f"  [WARN] {tk} failed: {e}")


# COMPUTE DERIVED FACTORS
print("\n4. Computing derived prediction market factors...")

# Combine all series into a single DataFrame
all_series = fed_frames + yc_frames + vix_frames
if not all_series:
    print("[ERROR] No data downloaded. Check internet connection.")
    raise SystemExit(1)

combined = pd.concat(all_series, axis=1).sort_index()
combined.index.name = "date"
print(f"  Combined raw data: {len(combined)} rows, columns: {list(combined.columns)}")

# Fed rate change expectation
# If we have Fed Funds Futures, use implied rate vs current 3m T-bill
if "ff_implied_rate" in combined.columns and "tbill_3m" in combined.columns:
    # Positive = market expects rate HIKE, negative = expects CUT
    combined["fed_rate_exp_change"] = combined["ff_implied_rate"] - combined["tbill_3m"]
    # Probability proxies: clip to [-2, 2] range for stability
    change = combined["fed_rate_exp_change"].clip(-2, 2)
    combined["fed_hike_prob"] = change.clip(lower=0) / 2.0  # 0 to 1
    combined["fed_cut_prob"]  = (-change).clip(lower=0) / 2.0  # 0 to 1
    print("  ✓ fed_rate_exp_change, fed_hike_prob, fed_cut_prob")
elif "tbill_3m" in combined.columns:
    # Fallback: use 3m T-bill 1-month change as rate change expectation proxy
    combined["fed_rate_exp_change"] = combined["tbill_3m"].diff(21)
    change = combined["fed_rate_exp_change"].clip(-2, 2)
    combined["fed_hike_prob"] = change.clip(lower=0) / 2.0
    combined["fed_cut_prob"]  = (-change).clip(lower=0) / 2.0
    print("  ✓ fed_rate_exp_change (T-bill proxy), fed_hike_prob, fed_cut_prob")

# Recession probability (yield curve probit)
if "yield_10y_raw" in combined.columns and "yield_2y_raw" in combined.columns:
    spread = combined["yield_10y_raw"] - combined["yield_2y_raw"]
    # Estrella-Mishkin probit: P(recession) = Φ(-0.57 - 0.83 × spread)
    from scipy.stats import norm
    combined["recession_prob"] = norm.cdf(-0.57 - 0.83 * spread)
    print("  ✓ recession_prob (yield curve probit)")
elif "yield_10y_raw" in combined.columns:
    # Fallback: just use 10Y level change as policy tightness
    combined["recession_prob"] = np.nan
    print("  ✗ recession_prob (need 2Y yield data)")
else:
    combined["recession_prob"] = np.nan

# Policy uncertainty = max(hike_prob, cut_prob)
if "fed_hike_prob" in combined.columns:
    combined["policy_uncertainty"] = combined[["fed_hike_prob", "fed_cut_prob"]].max(axis=1)
    print("  ✓ policy_uncertainty")

# VIX term structure
if "vix_spot" in combined.columns and "vix_3m" in combined.columns:
    # VIX contango ratio: VIX_3M / VIX_spot
    # > 1 = contango (complacency), < 1 = backwardation (fear/stress)
    combined["vix_term_structure"] = combined["vix_3m"] / combined["vix_spot"].replace(0, np.nan)
    print("  ✓ vix_term_structure")
elif "vix_spot" in combined.columns:
    # Proxy: VIX 21-day change direction
    combined["vix_term_structure"] = np.nan
    print("  ✗ vix_term_structure (need VIX3M data)")

# Composite prediction market sentiment
# Simple equal-weight of normalised components
sentiment_cols = []
for c in ["fed_rate_exp_change", "recession_prob", "vix_term_structure"]:
    if c in combined.columns and combined[c].notna().sum() > 50:
        sentiment_cols.append(c)

if sentiment_cols:
    # Z-score each, then average (negative = risk-off sentiment)
    z_frames = []
    for c in sentiment_cols:
        s = combined[c]
        z = (s - s.expanding(min_periods=21).mean()) / s.expanding(min_periods=21).std()
        z_frames.append(z)
    combined["pred_market_sentiment"] = pd.concat(z_frames, axis=1).mean(axis=1)
    print(f"  ✓ pred_market_sentiment (from {sentiment_cols})")

# Select final output columns
OUTPUT_COLS = [
    "fed_hike_prob", "fed_cut_prob", "recession_prob",
    "policy_uncertainty", "vix_term_structure", "pred_market_sentiment",
]

final_cols = [c for c in OUTPUT_COLS if c in combined.columns]
result = combined[final_cols].copy()
result.index.name = "date"
result = result.reset_index()
result["date"] = pd.to_datetime(result["date"])

# Drop rows where everything is NaN
result = result.dropna(subset=final_cols, how="all")

# Save
result.to_parquet(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
print(f"  Shape: {result.shape}")
print(f"  Date range: {result['date'].min().date()} → {result['date'].max().date()}")
print(f"  Columns: {final_cols}")

for col in final_cols:
    pct = result[col].notna().mean() * 100
    print(f"    {col:30s} {pct:5.1f}% filled | "
          f"mean={result[col].mean():.4f} | std={result[col].std():.4f}")

elapsed = _time.time() - _t0
print(f"\nDone in {elapsed:.1f}s")
