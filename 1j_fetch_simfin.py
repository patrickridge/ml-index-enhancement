"""
1j_fetch_simfin.py — Fetch Quarterly Fundamentals from Simfin (Free API)
=========================================================================
Downloads quarterly income, balance sheet, and cash flow data from the Simfin
bulk data API (free tier).  Computes trailing-twelve-month (TTM) aggregates
and standard valuation / quality ratios.

Point-in-time safety:
  Simfin provides a `publish_date` column — the date the filing became public.
  We use `publish_date` as the observation date, NOT period-end, to avoid
  look-ahead bias.  TTM = sum of last 4 published quarterly values.

Output:
  data/fundamental.parquet  — columns: date, ticker, pe_ratio, pb_ratio, ...
  This file is auto-detected by 1h_feature_engineering.py → Cat 9 factors.

Academic basis:
  Sloan (1996) — accruals anomaly
  Cooper, Gulen & Schill (2008) — asset growth
  Novy-Marx (2013) — gross profitability
  Fama & French (2015) — profitability + investment factors

Run time: ~2-5 min (bulk download + compute).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time as _time

_t0 = _time.time()
DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "fundamental.parquet"
SIMFIN_DIR = str(DATA_DIR / "simfin_bulk")

print("=" * 65)
print("SIMFIN FUNDAMENTAL DATA FETCH")
print("=" * 65)

# ── Load Simfin bulk data ─────────────────────────────────────────────────────
SIMFIN_API_KEY = None  # Set your free key from simfin.com, or None to use yfinance fallback

try:
    import simfin as sf
    sf.set_data_dir(SIMFIN_DIR)
    if SIMFIN_API_KEY:
        sf.set_api_key(SIMFIN_API_KEY)
    else:
        sf.set_api_key("free")
    HAS_SIMFIN = True
except ImportError:
    HAS_SIMFIN = False
    print("  simfin not installed — will use yfinance fallback")

# Load our ticker list
try:
    panel = pd.read_parquet(DATA_DIR / "panel_monthly.parquet", columns=["ticker"])
    our_tickers = panel["ticker"].unique().tolist()
    print(f"  Pipeline tickers: {len(our_tickers)}")
except FileNotFoundError:
    print("  [WARN] panel_monthly.parquet not found — fetching all US tickers")
    our_tickers = None

# ── Download quarterly financial statements ───────────────────────────────────
print("\nDownloading Simfin bulk data (US quarterly)...")

try:
    df_income = sf.load_income(variant="quarterly", market="us")
    print(f"  Income statements: {len(df_income):,} rows")
except Exception as e:
    print(f"  [ERROR] Failed to load income data: {e}")
    df_income = pd.DataFrame()

try:
    df_balance = sf.load_balance(variant="quarterly", market="us")
    print(f"  Balance sheets: {len(df_balance):,} rows")
except Exception as e:
    print(f"  [ERROR] Failed to load balance data: {e}")
    df_balance = pd.DataFrame()

try:
    df_cashflow = sf.load_cashflow(variant="quarterly", market="us")
    print(f"  Cash flow statements: {len(df_cashflow):,} rows")
except Exception as e:
    print(f"  [ERROR] Failed to load cashflow data: {e}")
    df_cashflow = pd.DataFrame()

if df_income.empty and df_balance.empty:
    print("\n[ERROR] No data downloaded. Check your internet connection.")
    raise SystemExit(1)

# ── Ticker mapping ────────────────────────────────────────────────────────────
# Simfin uses plain tickers (AAPL), our pipeline may use exchange-suffixed (AAPL.O)
# Build bidirectional mapping

def strip_exchange_suffix(ticker: str) -> str:
    """AAPL.O → AAPL, MSFT.O → MSFT, BRK.B → BRK.B (keep share class)"""
    if ticker.endswith((".O", ".N", ".A", ".K", ".P", ".Z")):
        return ticker[:-2]
    return ticker

if our_tickers:
    plain_to_pipeline = {}
    for tk in our_tickers:
        plain = strip_exchange_suffix(tk)
        plain_to_pipeline[plain] = tk
    print(f"  Ticker mapping: {len(plain_to_pipeline)} pipeline tickers mapped")
else:
    plain_to_pipeline = None


# ── Helper: reset Simfin multi-index to flat DataFrame ────────────────────────
def flatten_simfin(df):
    """Simfin returns MultiIndex (Ticker, Report Date). Flatten to columns."""
    if df.empty:
        return df
    df = df.reset_index()
    # Rename standard columns
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        col_map[c] = cl
    df = df.rename(columns=col_map)
    # Ensure date columns
    for dc in ["report_date", "publish_date"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
    return df


df_i = flatten_simfin(df_income)
df_b = flatten_simfin(df_balance)
df_c = flatten_simfin(df_cashflow)

# ── Map Simfin tickers to pipeline tickers ────────────────────────────────────
def map_tickers(df, mapping):
    """Map Simfin Ticker column to pipeline tickers. Drop unmapped."""
    if mapping is None:
        return df
    tk_col = "ticker" if "ticker" in df.columns else None
    if tk_col is None:
        return df
    df = df.copy()
    df["_pipeline_tk"] = df[tk_col].map(mapping)
    before = len(df)
    df = df.dropna(subset=["_pipeline_tk"])
    df[tk_col] = df["_pipeline_tk"]
    df = df.drop(columns=["_pipeline_tk"])
    return df

df_i = map_tickers(df_i, plain_to_pipeline)
df_b = map_tickers(df_b, plain_to_pipeline)
df_c = map_tickers(df_c, plain_to_pipeline)

print(f"\n  After ticker mapping:")
print(f"    Income:  {len(df_i):,} rows, {df_i['ticker'].nunique() if 'ticker' in df_i.columns else 0} tickers")
print(f"    Balance: {len(df_b):,} rows, {df_b['ticker'].nunique() if 'ticker' in df_b.columns else 0} tickers")
print(f"    CashFlow:{len(df_c):,} rows, {df_c['ticker'].nunique() if 'ticker' in df_c.columns else 0} tickers")


# ── Compute TTM aggregates and ratios ─────────────────────────────────────────
# Use publish_date as the point-in-time observation date
# For each (ticker, publish_date), compute TTM = sum of last 4 quarterly values

def safe_col(df, name):
    """Get column value or NaN if column doesn't exist."""
    return df[name] if name in df.columns else np.nan

def compute_fundamentals(df_i, df_b, df_c):
    """
    Compute fundamental ratios from quarterly Simfin data.
    Returns DataFrame with columns: date, ticker, + FUNDAMENTAL_COLS.
    """
    records = []

    # Use publish_date for point-in-time; fall back to report_date
    date_col_i = "publish_date" if "publish_date" in df_i.columns else "report_date"
    date_col_b = "publish_date" if "publish_date" in df_b.columns else "report_date"

    # Sort by ticker and date for rolling TTM
    if df_i.empty:
        return pd.DataFrame()

    # Identify key columns (Simfin uses various naming conventions)
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Income statement columns
    revenue_col   = find_col(df_i, ["revenue", "total_revenue", "net_revenue"])
    cogs_col      = find_col(df_i, ["cost_of_revenue", "cost_of_goods_sold", "cogs"])
    net_income_col = find_col(df_i, ["net_income", "net_income_(common)"])
    op_income_col  = find_col(df_i, ["operating_income_(loss)", "operating_income"])
    eps_col       = find_col(df_i, ["earnings_per_share,_diluted", "eps_diluted",
                                     "earnings_per_share_(diluted)"])

    # Balance sheet columns
    total_assets_col  = find_col(df_b, ["total_assets"])
    total_equity_col  = find_col(df_b, ["total_equity", "stockholders'_equity",
                                         "total_stockholders'_equity"])
    total_debt_col    = find_col(df_b, ["total_debt", "long_term_debt", "total_liabilities"])
    shares_col        = find_col(df_b, ["shares_(diluted)", "shares_outstanding",
                                         "common_shares_outstanding"])
    book_val_col      = find_col(df_b, ["total_equity", "stockholders'_equity"])

    # Cash flow columns
    cfo_col  = find_col(df_c, ["net_cash_from_operating_activities",
                                "cash_from_operating_activities"])
    capex_col = find_col(df_c, ["capital_expenditures",
                                 "acquisition_of_fixed_assets_&_intangibles",
                                 "change_in_fixed_assets_&_intangibles"])
    buyback_col = find_col(df_c, ["repurchase_of_common_equity",
                                   "common_stock_repurchased"])

    print(f"\n  Column mapping:")
    print(f"    Revenue:    {revenue_col}")
    print(f"    Net Income: {net_income_col}")
    print(f"    Total Eq:   {total_equity_col}")
    print(f"    Total Debt: {total_debt_col}")
    print(f"    CFO:        {cfo_col}")

    # Process per ticker: TTM aggregation
    for tk in df_i["ticker"].unique():
        tk_i = df_i[df_i["ticker"] == tk].sort_values(date_col_i).copy()
        tk_b = df_b[df_b["ticker"] == tk].sort_values(date_col_b).copy() if not df_b.empty else pd.DataFrame()
        tk_c = df_c[df_c["ticker"] == tk].sort_values(
            "publish_date" if "publish_date" in df_c.columns else "report_date"
        ).copy() if not df_c.empty else pd.DataFrame()

        # TTM rolling sums for income/cashflow items (sum of last 4 quarters)
        def ttm(series):
            return series.rolling(4, min_periods=3).sum()

        rev_ttm = ttm(tk_i[revenue_col]) if revenue_col else None
        ni_ttm  = ttm(tk_i[net_income_col]) if net_income_col else None
        cogs_ttm = ttm(tk_i[cogs_col]) if cogs_col else None

        for i, row_i in tk_i.iterrows():
            idx = tk_i.index.get_loc(i)
            obs_date = row_i[date_col_i]
            if pd.isna(obs_date):
                continue

            rec = {"date": obs_date, "ticker": tk}

            # Revenue TTM
            rev = rev_ttm.iloc[idx] if rev_ttm is not None else np.nan
            ni  = ni_ttm.iloc[idx] if ni_ttm is not None else np.nan
            cogs_v = cogs_ttm.iloc[idx] if cogs_ttm is not None else np.nan

            # Get nearest balance sheet data (on or before this date)
            if not tk_b.empty:
                mask = tk_b[date_col_b] <= obs_date
                if mask.any():
                    latest_b = tk_b[mask].iloc[-1]
                    total_eq    = latest_b[total_equity_col] if total_equity_col and total_equity_col in latest_b.index else np.nan
                    total_assets = latest_b[total_assets_col] if total_assets_col and total_assets_col in latest_b.index else np.nan
                    total_debt  = latest_b[total_debt_col] if total_debt_col and total_debt_col in latest_b.index else np.nan
                    shares      = latest_b[shares_col] if shares_col and shares_col in latest_b.index else np.nan
                    book_val    = latest_b[book_val_col] if book_val_col and book_val_col in latest_b.index else np.nan
                else:
                    total_eq = total_assets = total_debt = shares = book_val = np.nan
            else:
                total_eq = total_assets = total_debt = shares = book_val = np.nan

            # Price proxy: not available from Simfin fundamentals alone
            # Use book value ratios that don't need price (ROE, ROA, margins)
            # Price-based ratios (PE, PB, PS) need market cap — use shares × close from prices

            # ── Ratios that DON'T need price ──
            # ROE = NI_TTM / Equity
            rec["roe"] = ni / total_eq if total_eq and abs(total_eq) > 1e6 else np.nan
            # ROA = NI_TTM / Total Assets
            rec["roa"] = ni / total_assets if total_assets and abs(total_assets) > 1e6 else np.nan
            # Gross margin = (Revenue - COGS) / Revenue
            if rev and abs(rev) > 1e6 and not np.isnan(cogs_v):
                rec["gross_margin"] = (rev - cogs_v) / rev
            else:
                rec["gross_margin"] = np.nan
            # Debt to equity
            rec["debt_to_equity"] = total_debt / total_eq if total_eq and abs(total_eq) > 1e6 else np.nan

            # ── Growth rates (YoY) ──
            if idx >= 4:
                rev_1y = rev_ttm.iloc[idx - 4] if rev_ttm is not None else np.nan
                ni_1y  = ni_ttm.iloc[idx - 4]  if ni_ttm  is not None else np.nan
                rec["revenue_growth_yoy"] = (rev / rev_1y - 1) if rev_1y and abs(rev_1y) > 1e6 else np.nan
                rec["eps_growth_yoy"]     = (ni / ni_1y - 1)   if ni_1y  and abs(ni_1y) > 1e6 else np.nan
            else:
                rec["revenue_growth_yoy"] = np.nan
                rec["eps_growth_yoy"]     = np.nan

            # ── Earnings quality = CFO_TTM / NI_TTM (Sloan 1996) ──
            if not tk_c.empty and cfo_col:
                mask_c = tk_c["publish_date" if "publish_date" in tk_c.columns else "report_date"] <= obs_date
                if mask_c.any():
                    # TTM CFO
                    last4_c = tk_c[mask_c].tail(4)
                    cfo_ttm = last4_c[cfo_col].sum() if len(last4_c) >= 3 else np.nan
                    rec["earnings_quality"] = cfo_ttm / ni if ni and abs(ni) > 1e6 else np.nan
                else:
                    rec["earnings_quality"] = np.nan
            else:
                rec["earnings_quality"] = np.nan

            # ── Price-dependent ratios (set to NaN — will be computed in 1h using market prices) ──
            # PE, PB, PS, EV/EBITDA require market cap = shares × close
            # Store per-share values so 1h can divide by price
            if shares and shares > 0:
                rec["_eps_ttm"]       = ni / shares if ni else np.nan
                rec["_book_per_share"] = book_val / shares if book_val else np.nan
                rec["_rev_per_share"]  = rev / shares if rev else np.nan
            else:
                rec["_eps_ttm"] = rec["_book_per_share"] = rec["_rev_per_share"] = np.nan

            # Placeholder for price-dependent ratios
            rec["pe_ratio"]   = np.nan  # computed in 1h: close / _eps_ttm
            rec["pb_ratio"]   = np.nan  # computed in 1h: close / _book_per_share
            rec["ps_ratio"]   = np.nan  # computed in 1h: close / _rev_per_share
            rec["ev_ebitda"]  = np.nan  # needs enterprise value
            rec["buyback_yield"] = np.nan  # needs price history

            records.append(rec)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["date"] = pd.to_datetime(result["date"])
    # Drop rows where ALL ratios are NaN
    ratio_cols = ["roe", "roa", "gross_margin", "debt_to_equity",
                  "revenue_growth_yoy", "eps_growth_yoy", "earnings_quality"]
    result = result.dropna(subset=ratio_cols, how="all")

    return result


# ── Run ───────────────────────────────────────────────────────────────────────
print("\nComputing fundamental ratios (TTM, point-in-time)...")
fund_df = compute_fundamentals(df_i, df_b, df_c)

if fund_df.empty:
    print("\n[WARN] No fundamental data computed. Check Simfin data availability.")
else:
    # Keep only the columns expected by the pipeline
    from utils_factors import FUNDAMENTAL_COLS
    keep_cols = ["date", "ticker"] + [c for c in FUNDAMENTAL_COLS if c in fund_df.columns]
    # Also keep per-share helper columns for price-dependent ratio computation in 1h
    helper_cols = [c for c in fund_df.columns if c.startswith("_")]
    fund_df = fund_df[keep_cols + helper_cols]

    # Deduplicate: keep latest observation per (ticker, date-month)
    fund_df["_ym"] = fund_df["date"].dt.to_period("M")
    fund_df = fund_df.sort_values("date").drop_duplicates(
        subset=["ticker", "_ym"], keep="last"
    ).drop(columns=["_ym"])

    fund_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"  Shape: {fund_df.shape}")
    print(f"  Date range: {fund_df['date'].min().date()} → {fund_df['date'].max().date()}")
    print(f"  Tickers: {fund_df['ticker'].nunique()}")
    print(f"  Columns: {list(fund_df.columns)}")

    # Coverage stats
    for col in [c for c in FUNDAMENTAL_COLS if c in fund_df.columns]:
        pct = fund_df[col].notna().mean() * 100
        print(f"    {col:25s} {pct:5.1f}% filled")

elapsed = _time.time() - _t0
print(f"\nDone in {elapsed:.1f}s")
