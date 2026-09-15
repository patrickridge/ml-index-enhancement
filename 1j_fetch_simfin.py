"""
1j_fetch_simfin.py - Fetch Quarterly Fundamentals from Simfin (Free API)
Downloads quarterly income, balance sheet, and cash flow data from the Simfin
bulk data API (free tier).  Computes trailing-twelve-month (TTM) aggregates
and standard valuation / quality ratios.

Point-in-time safety:
  Simfin provides a `publish_date` column - the date the filing became public.
  We use `publish_date` as the observation date, NOT period-end, to avoid
  look-ahead bias.  TTM = sum of last 4 published quarterly values.

Output:
  data/fundamental.parquet  - columns: date, ticker, pe_ratio, pb_ratio, ...
  This file is auto-detected by 1h_feature_engineering.py → Cat 9 factors.

Academic basis:
  Sloan (1996) - accruals anomaly
  Cooper, Gulen & Schill (2008) - asset growth
  Novy-Marx (2013) - gross profitability
  Fama & French (2015) - profitability + investment factors

Run time: ~2-5 min (bulk download + compute).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path
import time as _time

_t0 = _time.time()
DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
OUT_PATH = DATA_DIR / "fundamental.parquet"
SIMFIN_DIR = str(DATA_DIR / "simfin_bulk")

print("=" * 65)
print("SIMFIN FUNDAMENTAL DATA FETCH")
print("=" * 65)

# Load Simfin bulk data
# Read from the environment, never from the file. A key committed to a public
# repo is leaked the moment it is pushed, and rotating it afterwards does not
# un-leak it. Set it in your shell instead:
#     export SIMFIN_API_KEY=your_key_here
SIMFIN_API_KEY = os.environ.get("SIMFIN_API_KEY")

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
    print("  simfin not installed - will use yfinance fallback")

# Load our ticker list
try:
    panel = pd.read_parquet(DATA_DIR / "panel_monthly.parquet", columns=["ticker"])
    our_tickers = panel["ticker"].unique().tolist()
    print(f"  Pipeline tickers: {len(our_tickers)}")
except FileNotFoundError:
    print("  [WARN] panel_monthly.parquet not found - fetching all US tickers")
    our_tickers = None

# Download quarterly financial statements
df_income = pd.DataFrame()
df_balance = pd.DataFrame()
df_cashflow = pd.DataFrame()
USE_YFINANCE_FALLBACK = False

if HAS_SIMFIN and SIMFIN_API_KEY:
    print("\nDownloading Simfin bulk data (US quarterly)...")
    try:
        df_income = sf.load_income(variant="quarterly", market="us")
        print(f"  Income statements: {len(df_income):,} rows")
    except Exception as e:
        print(f"  [WARN] Simfin income failed: {e}")

    try:
        df_balance = sf.load_balance(variant="quarterly", market="us")
        print(f"  Balance sheets: {len(df_balance):,} rows")
    except Exception as e:
        print(f"  [WARN] Simfin balance failed: {e}")

    try:
        df_cashflow = sf.load_cashflow(variant="quarterly", market="us")
        print(f"  Cash flow statements: {len(df_cashflow):,} rows")
    except Exception as e:
        print(f"  [WARN] Simfin cashflow failed: {e}")

if df_income.empty and df_balance.empty:
    print("\n  Simfin unavailable - falling back to yfinance fundamentals.")
    print("  NOTE: To use Simfin, register free at simfin.com and set SIMFIN_API_KEY.")
    USE_YFINANCE_FALLBACK = True

# Ticker mapping
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


# Helper: reset Simfin multi-index to flat DataFrame
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

# Map Simfin tickers to pipeline tickers
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


# Compute TTM aggregates and ratios
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
            # Price-based ratios (PE, PB, PS) need market cap - use shares × close from prices

            # Ratios that DON'T need price
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

            # Growth rates (YoY)
            if idx >= 4:
                rev_1y = rev_ttm.iloc[idx - 4] if rev_ttm is not None else np.nan
                ni_1y  = ni_ttm.iloc[idx - 4]  if ni_ttm  is not None else np.nan
                rec["revenue_growth_yoy"] = (rev / rev_1y - 1) if rev_1y and abs(rev_1y) > 1e6 else np.nan
                rec["eps_growth_yoy"]     = (ni / ni_1y - 1)   if ni_1y  and abs(ni_1y) > 1e6 else np.nan
            else:
                rec["revenue_growth_yoy"] = np.nan
                rec["eps_growth_yoy"]     = np.nan

            # Earnings quality = CFO_TTM / NI_TTM (Sloan 1996)
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

            # Price-dependent ratios (set to NaN - will be computed in 1h using market prices)
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


# YFINANCE FALLBACK - fetch fundamentals ticker-by-ticker

def fetch_fundamentals_yfinance(tickers, batch_size=20):
    """
    Fetch quarterly fundamentals from yfinance for a list of tickers.
    Returns DataFrame with columns: date, ticker, + fundamental ratios.

    yfinance provides ~4-5 years of quarterly data per ticker.
    We use the filing date (column date) as point-in-time observation.

    NOTE: This is slower than Simfin bulk download (~1-2 sec per ticker).
    For 500 tickers, expect ~10-20 min.
    """
    import yfinance as yf

    records = []
    n = len(tickers)
    failed = 0

    for i, tk in enumerate(tickers):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n}] Processing {tk}...")

        # yfinance wants plain ticker (no .O/.N suffix)
        yf_tk = strip_exchange_suffix(tk)
        # Special cases: BRK.B → BRK-B for yfinance
        yf_tk = yf_tk.replace(".", "-")

        try:
            obj = yf.Ticker(yf_tk)

            # Quarterly financials (income statement)
            inc = obj.quarterly_financials  # line items × dates
            if inc is None or inc.empty:
                failed += 1
                continue

            # Quarterly balance sheet
            bal = obj.quarterly_balance_sheet
            if bal is None:
                bal = pd.DataFrame()

            # Quarterly cash flow
            cf = obj.quarterly_cashflow
            if cf is None:
                cf = pd.DataFrame()

            # Transpose: dates become rows, line items become columns
            inc_t = inc.T.sort_index()
            bal_t = bal.T.sort_index() if not bal.empty else pd.DataFrame()
            cf_t  = cf.T.sort_index()  if not cf.empty  else pd.DataFrame()

            # Helper to safely get a column
            def get(df, names, default=np.nan):
                if df.empty:
                    return pd.Series(default, index=inc_t.index)
                for name in names:
                    if name in df.columns:
                        return df[name].reindex(inc_t.index)
                return pd.Series(default, index=inc_t.index)

            # Extract key line items
            revenue    = get(inc_t, ["Total Revenue", "Revenue"])
            net_income = get(inc_t, ["Net Income", "Net Income Common Stockholders"])
            cogs       = get(inc_t, ["Cost Of Revenue", "Cost of Revenue"])
            op_income  = get(inc_t, ["Operating Income", "EBIT"])

            total_eq     = get(bal_t, ["Total Stockholders Equity", "Stockholders Equity",
                                        "Total Equity Gross Minority Interest"])
            total_assets = get(bal_t, ["Total Assets"])
            total_debt   = get(bal_t, ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"])
            shares_out   = get(bal_t, ["Share Issued", "Ordinary Shares Number"])

            cfo   = get(cf_t, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
            capex = get(cf_t, ["Capital Expenditure"])

            # TTM (trailing 4 quarters)
            def ttm(s):
                return s.rolling(4, min_periods=3).sum()

            rev_ttm = ttm(revenue)
            ni_ttm  = ttm(net_income)
            cogs_ttm = ttm(cogs)
            cfo_ttm = ttm(cfo)

            # For each quarter, compute ratios
            for j, dt in enumerate(inc_t.index):
                rec = {"date": pd.Timestamp(dt), "ticker": tk}

                rev_v = rev_ttm.iloc[j] if not pd.isna(rev_ttm.iloc[j]) else np.nan
                ni_v  = ni_ttm.iloc[j]  if not pd.isna(ni_ttm.iloc[j])  else np.nan
                cogs_v = cogs_ttm.iloc[j] if not pd.isna(cogs_ttm.iloc[j]) else np.nan
                cfo_v  = cfo_ttm.iloc[j] if not pd.isna(cfo_ttm.iloc[j]) else np.nan

                eq_v     = total_eq.iloc[j]     if j < len(total_eq) and not pd.isna(total_eq.iloc[j]) else np.nan
                assets_v = total_assets.iloc[j]  if j < len(total_assets) and not pd.isna(total_assets.iloc[j]) else np.nan
                debt_v   = total_debt.iloc[j]    if j < len(total_debt) and not pd.isna(total_debt.iloc[j]) else np.nan
                shares_v = shares_out.iloc[j]    if j < len(shares_out) and not pd.isna(shares_out.iloc[j]) else np.nan

                # ROE = NI_TTM / Equity
                rec["roe"] = ni_v / eq_v if eq_v and abs(eq_v) > 1e6 else np.nan
                # ROA = NI_TTM / Assets
                rec["roa"] = ni_v / assets_v if assets_v and abs(assets_v) > 1e6 else np.nan
                # Gross margin
                if rev_v and abs(rev_v) > 1e6 and not np.isnan(cogs_v):
                    rec["gross_margin"] = (rev_v - cogs_v) / rev_v
                else:
                    rec["gross_margin"] = np.nan
                # Debt to equity
                rec["debt_to_equity"] = debt_v / eq_v if eq_v and abs(eq_v) > 1e6 else np.nan

                # YoY growth (need 4 prior quarters)
                if j >= 4:
                    rev_1y = rev_ttm.iloc[j - 4]
                    ni_1y  = ni_ttm.iloc[j - 4]
                    rec["revenue_growth_yoy"] = (rev_v / rev_1y - 1) if rev_1y and abs(rev_1y) > 1e6 else np.nan
                    rec["eps_growth_yoy"]     = (ni_v / ni_1y - 1)   if ni_1y and abs(ni_1y) > 1e6 else np.nan
                else:
                    rec["revenue_growth_yoy"] = np.nan
                    rec["eps_growth_yoy"] = np.nan

                # Earnings quality = CFO_TTM / NI_TTM (Sloan 1996)
                rec["earnings_quality"] = cfo_v / ni_v if ni_v and abs(ni_v) > 1e6 else np.nan

                # Per-share helpers (for price-based ratios in 1h)
                if shares_v and shares_v > 0:
                    rec["_eps_ttm"]        = ni_v / shares_v if ni_v else np.nan
                    rec["_book_per_share"] = eq_v / shares_v if eq_v else np.nan
                    rec["_rev_per_share"]  = rev_v / shares_v if rev_v else np.nan
                else:
                    rec["_eps_ttm"] = rec["_book_per_share"] = rec["_rev_per_share"] = np.nan

                # Placeholder for price-dependent ratios
                rec["pe_ratio"]  = np.nan
                rec["pb_ratio"]  = np.nan
                rec["ps_ratio"]  = np.nan
                rec["ev_ebitda"] = np.nan
                rec["buyback_yield"] = np.nan

                records.append(rec)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"    [WARN] {yf_tk}: {e}")
            elif failed == 6:
                print(f"    [WARN] Suppressing further individual errors...")
            continue

        # Rate limiting: small sleep every batch_size tickers
        if (i + 1) % batch_size == 0:
            _time.sleep(0.5)

    print(f"  Completed: {n - failed}/{n} tickers ({failed} failed)")

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"])

    # yfinance dates are period-end, not publish dates
    # Add ~45 day lag as conservative point-in-time estimate
    # (10-Q filing deadline = 40 days for large accelerated filers)
    result["date"] = result["date"] + pd.Timedelta(days=45)

    # Drop all-NaN ratio rows
    ratio_cols = ["roe", "roa", "gross_margin", "debt_to_equity",
                  "revenue_growth_yoy", "eps_growth_yoy", "earnings_quality"]
    result = result.dropna(subset=ratio_cols, how="all")

    return result


# Run
if USE_YFINANCE_FALLBACK:
    print("\nFetching fundamentals via yfinance (ticker-by-ticker)...")
    print("  This may take 10-20 min for 500+ tickers.")
    if our_tickers:
        fund_df = fetch_fundamentals_yfinance(our_tickers)
    else:
        print("  [ERROR] No ticker list available. Run 1h first to generate panel_monthly.parquet.")
        fund_df = pd.DataFrame()
else:
    print("\nComputing fundamental ratios (TTM, point-in-time) from Simfin...")
    fund_df = compute_fundamentals(df_i, df_b, df_c)

if fund_df.empty:
    print("\n[WARN] No fundamental data computed.")
    if USE_YFINANCE_FALLBACK:
        print("  yfinance fallback returned no data. Check internet connection.")
    else:
        print("  Simfin returned no data. Set SIMFIN_API_KEY or use yfinance fallback.")
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
