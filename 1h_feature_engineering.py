"""
1h_feature_engineering.py  (v2 — 100+ factors)
===============================================
Reads daily OHLCV prices and the existing monthly panel, engineers 100+ factors
across 10 categories, and saves the enriched panel.

Data sources:
  REQUIRED   data/prices.parquet          — daily OHLC (+ optional volume column)
  REQUIRED   data/panel_monthly.parquet   — monthly panel with existing 9 momentum factors
  OPTIONAL   data/fundamental.parquet     — fundamental data (Cat 9)
  FETCHED    yfinance: ^GSPC, ^VIX, ^TNX, ^IRX, DX-Y.NYB, HYG (Cats 6, 8, 10)

Output:
  data/panel_monthly_enriched.parquet   — all factors, cross-sectionally ranked

Factor categories built here:
  Cat 1  — Multi-horizon momentum (6 new: ret_1w/2w/9m/18m/24m/36m)
  Cat 2  — Volatility regimes (8 new)
  Cat 3  — Tail risk (6 new)
  Cat 4  — Price level / trend (11 new)
  Cat 5  — Volume & liquidity (6 new, optional)
  Cat 6  — Market beta / correlation (8 new)
  Cat 7  — Intraday / microstructure (5 new)
  Cat 8  — Cross-sectional relative (6 new)
  Cat 9  — Fundamental / quality (12 optional)
  Cat 10 — Macro / regime (9, time-series z-scored NOT cross-sectionally ranked)
  Cat 11 — Time signal factors (5 new: ir_6m, trend_r2_126d, ret_consistency_12m, skew_60d, drawdown_pct_252d)
  Cat 12 — Barra-style factors (4 new: amihud_illiq_21d, size_proxy, vol_of_vol_63d, beta_stability_63d)
  Cat 13 — Cross-sectional interaction factors (5 new, monthly: residual_ret_1m, beta_x_idiovol, up_down_beta_spread, vol_excess, mom_decel)
  Cat 14 — Size / market cap (1 new: log_mktcap via yfinance)
  Cat 15 — Tail ranking features (51 new: _top/_bot/_tail dummies for 17 base factors)
  Cat 16 — Mined alpha factors (12 new: nearness_52w_high, max_ret_21d, risk_adj_mom_6m/12m,
            residual_mom_6m/12m, up_down_vol_ratio, co_skewness_63d, vol_contraction_signal,
            mom_quality, price_range_ratio, reversal_size)

Run time: ~5-10 min on a laptop.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from utils_factors import (
    add_momentum_daily,
    add_volatility_features,
    add_tail_risk,
    add_price_trend,
    add_volume_liquidity,
    add_beta_correlation,
    add_microstructure,
    add_cross_sectional_relative,
    add_fundamental_factors,
    fetch_macro_data,
    add_macro_factors,
    add_time_signal_factors,
    add_barra_style_factors,
    add_macro_interaction_factors,
    fetch_market_cap_data,
    add_size_factor,
    add_tail_ranking_features,
    add_mined_factors,
    add_time_signal_v2,
    add_seasonality_factors,
    add_short_interest_factors,
    add_institutional_factors,
)
from config import MACRO_COLS
import time as _time; _t0 = _time.time()

DATA_DIR  = Path("data")
PANEL_IN  = DATA_DIR / "panel_monthly.parquet"
PRICES_IN = DATA_DIR / "prices.parquet"
FUND_IN   = DATA_DIR / "fundamental.parquet"
OUT_PATH  = DATA_DIR / "panel_monthly_enriched.parquet"

# Intermediate column prefixes to exclude when sampling daily features at month-end
_SKIP_PREFIXES = ("_", "ret_d", "spx_ret", "ret_neg", "ret_pos")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def rank_norm(s: pd.Series) -> pd.Series:
    """Cross-sectional rank normalised to [-0.5, 0.5]. NaN stays NaN."""
    r = s.rank(pct=True, na_option="keep")
    return r - 0.5


def cs_rank_all(df: pd.DataFrame, feat_cols: list, date_col: str = "date") -> pd.DataFrame:
    """Apply rank_norm cross-sectionally within each date for feat_cols."""
    df = df.copy()
    for c in feat_cols:
        df[c] = df.groupby(date_col)[c].transform(rank_norm)
    return df


def ts_zscore(series: pd.Series, window: int = 36) -> pd.Series:
    """
    Time-series rolling z-score (window periods lookback).
    Used for macro columns which have zero cross-sectional variance.
    """
    mu  = series.rolling(window, min_periods=max(6, window // 4)).mean()
    sig = series.rolling(window, min_periods=max(6, window // 4)).std()
    return (series - mu) / (sig + 1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# SPX DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_spx_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPX daily close and returns via yfinance."""
    try:
        import yfinance as yf
        spx = yf.download("^GSPC", start=start_date, end=end_date,
                           auto_adjust=True, progress=False)
        spx_df = spx["Close"].reset_index()
        spx_df.columns = ["date", "spx_close"]
        spx_df["date"]    = pd.to_datetime(spx_df["date"])
        spx_df["spx_ret"] = spx_df["spx_close"].pct_change()
        return spx_df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] Could not fetch SPX data: {e}")
        return pd.DataFrame(columns=["date", "spx_close", "spx_ret"])


def compute_spx_monthly_stats(spx_daily: pd.DataFrame,
                               panel_dates) -> pd.DataFrame:
    """
    Compute SPX monthly returns and vol sampled at each month-end date.
    Returns a date-indexed DataFrame used for Cat 8 cross-sectional factors.
    """
    if spx_daily.empty:
        return pd.DataFrame()

    spx  = spx_daily.set_index("date")["spx_close"].sort_index()
    spx_r = spx_daily.set_index("date")["spx_ret"].sort_index()

    rows = []
    for dt in sorted(panel_dates):
        slice_spx = spx.loc[:dt]
        slice_ret = spx_r.loc[:dt]
        if len(slice_spx) < 22:
            continue

        row = {"date": dt}
        for w, label in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]:
            if len(slice_spx) > w:
                row[f"spx_ret_{label}"] = (
                    slice_spx.iloc[-1] / slice_spx.iloc[-w - 1] - 1
                )
            else:
                row[f"spx_ret_{label}"] = np.nan
        row["spx_vol_63d"] = slice_ret.iloc[-63:].std() if len(slice_ret) >= 63 else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY FEATURE PIPELINE  (Cats 1-7)
# ═══════════════════════════════════════════════════════════════════════════════

def build_daily_features(prices: pd.DataFrame,
                          spx_daily: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all per-ticker daily feature computation via utils_factors."""
    prices = prices.sort_values(["ticker", "date"]).copy()

    # Base daily return
    prices["ret_d"] = prices.groupby("ticker", group_keys=False)["close"].transform(
        lambda x: x.pct_change()
    )

    # Merge SPX daily returns for beta/correlation (Cats 2 & 6)
    if not spx_daily.empty:
        prices = prices.merge(spx_daily[["date", "spx_ret"]], on="date", how="left")

    print("  Cat 1: multi-horizon momentum...")
    prices = add_momentum_daily(prices)

    print("  Cat 2: volatility regimes...")
    prices = add_volatility_features(prices)

    print("  Cat 3: tail risk...")
    prices = add_tail_risk(prices)

    print("  Cat 4: price level / trend...")
    prices = add_price_trend(prices)

    print("  Cat 5: volume & liquidity...")
    prices = add_volume_liquidity(prices)

    print("  Cat 6: market beta / correlation...")
    prices = add_beta_correlation(prices)

    print("  Cat 7: intraday / microstructure...")
    prices = add_microstructure(prices)

    print("  Cat 11: time signal factors...")
    prices = add_time_signal_factors(prices)

    print("  Cat 12: Barra-style factors...")
    prices = add_barra_style_factors(prices)

    print("  Cat 16: mined factors...")
    prices = add_mined_factors(prices)

    print("  Cat 17: time-signal v2 factors...")
    prices = add_time_signal_v2(prices)

    print("  Cat 18: seasonality factors...")
    prices = add_seasonality_factors(prices)

    # Carry month-end close for market cap computation (Cat 14).
    # Named close_me — not in exclude_cols so it survives sample_at_month_end.
    prices["close_me"] = prices["close"]

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DAILY FEATURES AT MONTH-END
# ═══════════════════════════════════════════════════════════════════════════════

def sample_at_month_end(daily: pd.DataFrame,
                         panel_dates: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, ticker) in panel_dates, extract daily feature values
    at or just before that date via merge_asof.

    Auto-detects feature columns: all columns except date, ticker, and
    those starting with underscore or known intermediate prefixes.
    """
    # Exclude raw OHLC price levels — not cross-sectional signals, only spurious size bias
    exclude_cols = {"date", "ticker", "open", "high", "low", "close"}
    new_feat_cols = [
        c for c in daily.columns
        if c not in exclude_cols
        and not any(c.startswith(p) for p in _SKIP_PREFIXES)
    ]

    keep_cols = ["date", "ticker"] + new_feat_cols
    daily_sub = daily[keep_cols].dropna(subset=["date", "ticker"])
    daily_sub = daily_sub.sort_values(["ticker", "date"])

    left  = panel_dates[["date", "ticker"]].sort_values(["date", "ticker"]).reset_index(drop=True)
    right = daily_sub.sort_values(["date", "ticker"]).reset_index(drop=True)

    result = pd.merge_asof(
        left, right,
        on="date", by="ticker",
        direction="backward",
        tolerance=pd.Timedelta("35 days"),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM EXTENSIONS FROM MONTHLY PANEL (carried over from v1)
# ═══════════════════════════════════════════════════════════════════════════════

def add_momentum_extensions(panel: pd.DataFrame) -> pd.DataFrame:
    """Build mom_2_12, mom_accel, ir_12m, ir_3m, rev_signal from monthly ret_Xm."""
    panel = panel.copy()
    if "ret_12m" in panel.columns and "ret_1m" in panel.columns:
        panel["mom_2_12"] = panel["ret_12m"] - panel["ret_1m"]
    if "ret_3m" in panel.columns and "ret_12m" in panel.columns:
        panel["mom_accel"] = panel["ret_3m"] - (panel["ret_12m"] - panel["ret_3m"])
    if "ret_12m" in panel.columns and "vol_252d" in panel.columns:
        panel["ir_12m"] = panel["ret_12m"] / (panel["vol_252d"].abs() + 1e-6)
    if "ret_3m" in panel.columns and "vol_60d" in panel.columns:
        panel["ir_3m"] = panel["ret_3m"] / (panel["vol_60d"].abs() + 1e-6)
    # rev_signal = -ret_1m after ranking → exact duplicate of ret_1m, removed
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("FEATURE ENGINEERING v3 — 100+ factors (Cats 1-13)")
    print("=" * 65)

    # Load base data
    print("\nLoading data...")
    panel  = pd.read_parquet(PANEL_IN)
    # Drop rev_1m from v1 base panel — it is -ret_1m after ranking (exact duplicate)
    panel  = panel.drop(columns=[c for c in ["rev_1m"] if c in panel.columns])
    prices = pd.read_parquet(PRICES_IN)
    panel["date"]  = pd.to_datetime(panel["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    panel  = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  Panel:  {panel.shape} | {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"  Prices: {prices.shape} | tickers={prices['ticker'].nunique()}")
    vol_avail = "volume" in prices.columns
    print(f"  Volume column: {'PRESENT (Cat 5 enabled)' if vol_avail else 'absent (Cat 5 skipped)'}")

    # Fetch SPX daily
    start_str = str(prices["date"].min().date())
    end_str   = str((prices["date"].max() + pd.Timedelta("5 days")).date())
    print(f"\nFetching SPX daily data ({start_str} → {end_str})...")
    spx_daily = fetch_spx_daily(start_str, end_str)
    print(f"  SPX rows: {len(spx_daily):,}")

    # Build daily features (Cats 1-7)
    print("\nBuilding daily features (Cats 1-7)...")
    daily = build_daily_features(prices, spx_daily)
    print(f"  Daily columns after feature engineering: {daily.shape[1]}")

    # Sample at month-end
    print("Sampling daily features at month-end dates...")
    daily_at_month = sample_at_month_end(daily, panel[["date", "ticker"]])

    # Merge into panel
    panel = panel.merge(daily_at_month, on=["date", "ticker"], how="left")

    # Momentum extensions (from existing monthly returns)
    print("Adding momentum extensions...")
    panel = add_momentum_extensions(panel)

    # Cat 8: cross-sectional relative
    spx_monthly = compute_spx_monthly_stats(spx_daily, panel["date"].unique())
    if not spx_monthly.empty:
        print("Adding Cat 8: cross-sectional relative factors...")
        panel = add_cross_sectional_relative(panel, spx_monthly)

    # Cat 9: fundamental (optional)
    if FUND_IN.exists():
        print(f"\nFound {FUND_IN} — adding Cat 9: fundamental factors...")
        fundamental = pd.read_parquet(FUND_IN)
        fundamental["date"] = pd.to_datetime(fundamental["date"])
        panel = add_fundamental_factors(panel, fundamental)
    else:
        print(f"\n{FUND_IN} not found — skipping Cat 9 fundamental factors.")

    # Cat 19: short interest (optional)
    SI_IN = DATA_DIR / "short_interest.parquet"
    if SI_IN.exists():
        print(f"\nFound {SI_IN} — adding Cat 19: short interest factors...")
        short_df = pd.read_parquet(SI_IN)
        short_df["date"] = pd.to_datetime(short_df["date"])
        panel = add_short_interest_factors(panel, short_df)
    else:
        print(f"\n{SI_IN} not found — skipping Cat 19 short interest factors.")

    # Cat 20: institutional ownership (optional)
    IO_IN = DATA_DIR / "institutional_ownership.parquet"
    if IO_IN.exists():
        print(f"\nFound {IO_IN} — adding Cat 20: institutional ownership factors...")
        inst_df = pd.read_parquet(IO_IN)
        inst_df["date"] = pd.to_datetime(inst_df["date"])
        panel = add_institutional_factors(panel, inst_df)
    else:
        print(f"\n{IO_IN} not found — skipping Cat 20 institutional ownership factors.")

    # Cat 10: macro / regime
    print("\nFetching macro data (Cat 10)...")
    macro_daily = fetch_macro_data(start_str, end_str, spx_daily=spx_daily)
    if not macro_daily.empty:
        print("Adding macro factors to panel...")
        panel = add_macro_factors(panel, macro_daily)
        found_macro = [c for c in MACRO_COLS if c in panel.columns]
        print(f"  Macro columns added: {found_macro}")
        print("\nAdding Cat 13: macro interaction factors...")
        panel = add_macro_interaction_factors(panel)
    else:
        print("  Macro data unavailable — skipping Cat 10 and Cat 13.")

    # Cat 21: prediction market / Fed expectations (macro-level, TS z-scored)
    PRED_IN = DATA_DIR / "prediction_markets.parquet"
    if PRED_IN.exists():
        print("\nAdding Cat 21: prediction market factors...")
        pred_df = pd.read_parquet(PRED_IN)
        pred_df["date"] = pd.to_datetime(pred_df["date"]).astype("datetime64[us]")
        pred_sorted = pred_df.sort_values("date")
        panel_sorted = panel.sort_values("date")
        panel_sorted["date"] = panel_sorted["date"].astype("datetime64[us]")
        # merge_asof: each month-end gets the latest prediction market data
        pred_cols = [c for c in pred_df.columns if c != "date"]
        panel = pd.merge_asof(
            panel_sorted,
            pred_sorted,
            on="date",
            direction="backward",
            tolerance=pd.Timedelta("35 days"),
        )
        found_pred = [c for c in pred_cols if c in panel.columns]
        print(f"  Prediction market columns added: {found_pred}")
    else:
        print(f"\n{PRED_IN} not found — skipping Cat 21 prediction market factors.")

    # Cat 14: size factor — log(market cap) = log(shares × close_me)
    mktcap_cache = str(DATA_DIR / "mktcap_shares.parquet")
    all_tickers  = panel["ticker"].unique().tolist()
    print("\nFetching market cap data (Cat 14: log_mktcap)...")
    shares_df = fetch_market_cap_data(all_tickers, start_str, end_str,
                                      cache_path=mktcap_cache)
    panel = add_size_factor(panel, shares_df)
    # Drop the helper column — it is not a model feature
    panel = panel.drop(columns=["close_me"], errors="ignore")

    # Cat 15: Tail-ranking features — explicit top/bottom decile dummies
    # Capture non-linear tail effects on contrarian + momentum + vol factors.
    # These are already in {0,1}/{-1,0,1} space; do NOT CS-rank them again.
    print("\nAdding Cat 15: tail-ranking features...")
    TAIL_BASE_FACTORS = [
        # Short-term reversal (contrarian candidates)
        "ret_1m", "ret_2w", "ret_1w",
        # Multi-horizon momentum
        "ret_6m", "ret_9m", "ret_12m", "ret_18m", "ret_24m",
        # Volatility
        "vol_momentum_21d", "vol_21d", "vol_252d",
        # Quality/risk
        "idio_vol_252d", "beta_252d", "maxdd_126d",
        # Barra-style size/liquidity
        "size_proxy", "amihud_illiq_21d", "log_mktcap",
    ]
    tail_base_available = [f for f in TAIL_BASE_FACTORS if f in panel.columns]
    panel = add_tail_ranking_features(panel, tail_base_available, top_pct=0.10, bot_pct=0.10)

    # Track tail columns so they skip CS-ranking (already {0,1} / {-1,0,1})
    TAIL_COLS = set()
    for f in tail_base_available:
        for suffix in ("_top", "_bot", "_tail"):
            col = f"{f}{suffix}"
            if col in panel.columns:
                TAIL_COLS.add(col)
    # Calendar dummies have zero CS variance (same for all stocks on a date)
    # Keep as binary {0,1} — no CS-ranking or TS z-scoring
    for cal_col in ["turn_of_month", "january_dummy"]:
        if cal_col in panel.columns:
            TAIL_COLS.add(cal_col)

    # Identify all feature columns
    always_exclude = {"date", "ticker", "fwd_ret_1m"}
    macro_in_panel = [c for c in MACRO_COLS if c in panel.columns]
    # Tail cols already normalised ({0,1}/{-1,0,1}) — skip CS-ranking
    feat_cols_cs   = [c for c in panel.columns
                      if c not in always_exclude
                      and c not in macro_in_panel
                      and c not in TAIL_COLS]
    all_feat_cols  = feat_cols_cs + list(TAIL_COLS) + macro_in_panel

    print(f"\nTotal features: {len(all_feat_cols)}")
    print(f"  Cross-sectionally ranked: {len(feat_cols_cs)}")
    print(f"  Time-series z-scored (macro): {len(macro_in_panel)}")

    # Drop rows with no target
    before = len(panel)
    panel  = panel.dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)
    print(f"Dropped {before - len(panel):,} rows with NaN target.")

    # Cross-sectional rank all non-macro features
    print("Cross-sectional rank normalising features...")
    panel = cs_rank_all(panel, feat_cols_cs, date_col="date")

    # Time-series z-score macro features
    if macro_in_panel:
        print("Time-series z-scoring macro features...")
        macro_ts = (panel.drop_duplicates("date")
                        .set_index("date")[macro_in_panel]
                        .sort_index())
        macro_z  = macro_ts.apply(lambda s: ts_zscore(s, window=36)).reset_index()
        panel    = panel.drop(columns=macro_in_panel).merge(macro_z, on="date", how="left")

    # Fill remaining NaNs with 0 (neutral rank)
    panel[all_feat_cols] = panel[all_feat_cols].fillna(0.0)

    # Save
    panel.to_parquet(OUT_PATH, index=False)

    print(f"\n{'=' * 65}")
    print(f"Done! Saved → {OUT_PATH}")
    print(f"Shape: {panel.shape}")
    print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Features ({len(all_feat_cols)}): {sorted(all_feat_cols)}")
    print("=" * 65)
    print(f"Done in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
