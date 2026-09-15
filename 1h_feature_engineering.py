"""
1h_feature_engineering.py - build the enriched monthly factor panel.

Reads daily OHLCV prices and the base monthly panel, computes the factor
library from utils_factors.py plus the optional-source categories that need
this pipeline's context (macro/regime, fundamentals, mined alpha, etc.), and
writes the cross-sectionally ranked panel consumed by every downstream model.

Downstream Cats 19-24 (short interest, 13F, prediction markets, mined
candidates, insider, sentiment) are added in-place by later fetchers; this
file covers Cats 1-16.

Inputs:
  data/prices.parquet            required - daily OHLC (+ optional volume)
  data/panel_monthly.parquet     required - base monthly panel (9 momentum factors)
  data/fundamental.parquet       optional - fundamentals (enables Cat 9)
  yfinance                       fetched  - ^GSPC, ^VIX, ^TNX, ^IRX, DX-Y.NYB, HYG
                                            (Cats 6, 8, 10)

Output:
  data/panel_monthly_enriched.parquet

Categories added here (Cats 17-18 for seasonality / time-signal v2 are appended
by the same run; README has the full 24-category list):
  Cat 1   multi-horizon momentum               (6 factors: 1w/2w/9m/18m/24m/36m)
  Cat 2   volatility regimes                    (8)
  Cat 3   tail risk                             (6)
  Cat 4   price level / trend                   (11)
  Cat 5   volume & liquidity                    (6, needs volume column)
  Cat 6   market beta / correlation             (8)
  Cat 7   intraday / microstructure             (5)
  Cat 8   cross-sectional relative              (6)
  Cat 9   fundamental / quality                 (12, optional)
  Cat 10  macro / regime                        (9, TS-z-scored, not CS-ranked)
  Cat 11  time-signal v1                        (5)
  Cat 12  Barra-style                           (4)
  Cat 13  cross-sectional interactions          (5)
  Cat 14  size / market cap                     (1)
  Cat 15  tail ranking (top/bot/tail dummies)   (51)
  Cat 16  mined alpha                           (12)

Run time: ~5-10 min on a laptop.
"""

import numpy as np
import pandas as pd
import os
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
    add_proper_time_signals,
    add_seasonality_factors,
    add_short_interest_factors,
    add_institutional_factors,
)
from config import MACRO_COLS, USE_CANDIDATE_FEATURES
import time as _time; _t0 = _time.time()

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
PANEL_IN  = DATA_DIR / "panel_monthly.parquet"
PRICES_IN = DATA_DIR / "prices.parquet"
FUND_IN   = DATA_DIR / "fundamental.parquet"
OUT_PATH  = DATA_DIR / "panel_monthly_enriched.parquet"

# Intermediate column prefixes to exclude when sampling daily features at month-end
_SKIP_PREFIXES = ("_", "ret_d", "spx_ret", "ret_neg", "ret_pos")


# HELPERS

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


# SPX DATA FETCH

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


# DAILY FEATURE PIPELINE  (Cats 1-7)

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

    print("  Cat 17: proper per-stock time-signal factors (TSMOM, MA regime, 52w, volume, earnings, serial corr)...")
    prices = add_proper_time_signals(prices)

    print("  Cat 18: seasonality factors...")
    prices = add_seasonality_factors(prices)

    # Cat 22: candidate factors from factor mining (69 OHLCV-derived signals)
    if USE_CANDIDATE_FEATURES:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "research", "factor_mining"))
        from candidate_factory import build_all_candidates
        from regime_entropy import add_entropy_regime_signals
        print("  Cat 22: factor mining candidates (9 families, ~69 features)...")
        prices = build_all_candidates(prices)
        prices = add_entropy_regime_signals(prices)

    # Carry month-end close for market cap computation (Cat 14).
    # Named close_me - not in exclude_cols so it survives sample_at_month_end.
    prices["close_me"] = prices["close"]

    return prices


# SAMPLE DAILY FEATURES AT MONTH-END

def sample_at_month_end(daily: pd.DataFrame,
                         panel_dates: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, ticker) in panel_dates, extract daily feature values
    at or just before that date via merge_asof.

    Auto-detects feature columns: all columns except date, ticker, and
    those starting with underscore or known intermediate prefixes.
    """
    # Exclude raw OHLC price levels - not cross-sectional signals, only spurious size bias
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


# MOMENTUM EXTENSIONS FROM MONTHLY PANEL (carried over from v1)

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


# MAIN

def main():
    print("=" * 65)
    print("FEATURE ENGINEERING v3 - 100+ factors (Cats 1-13)")
    print("=" * 65)

    # Load base data
    print("\nLoading data...")
    panel  = pd.read_parquet(PANEL_IN)
    # Drop rev_1m from v1 base panel - it is -ret_1m after ranking (exact duplicate)
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
        print(f"\nFound {FUND_IN} - adding Cat 9: fundamental factors...")
        fundamental = pd.read_parquet(FUND_IN)
        fundamental["date"] = pd.to_datetime(fundamental["date"])
        panel = add_fundamental_factors(panel, fundamental)
    else:
        print(f"\n{FUND_IN} not found - skipping Cat 9 fundamental factors.")

    # Cat 19: short interest (optional)
    SI_IN = DATA_DIR / "short_interest.parquet"
    if SI_IN.exists():
        print(f"\nFound {SI_IN} - adding Cat 19: short interest factors...")
        short_df = pd.read_parquet(SI_IN)
        short_df["date"] = pd.to_datetime(short_df["date"])
        panel = add_short_interest_factors(panel, short_df)
    else:
        print(f"\n{SI_IN} not found - skipping Cat 19 short interest factors.")

    # Cat 20: institutional ownership (optional)
    IO_IN = DATA_DIR / "institutional_ownership.parquet"
    if IO_IN.exists():
        print(f"\nFound {IO_IN} - adding Cat 20: institutional ownership factors...")
        inst_df = pd.read_parquet(IO_IN)
        inst_df["date"] = pd.to_datetime(inst_df["date"])
        panel = add_institutional_factors(panel, inst_df)
    else:
        print(f"\n{IO_IN} not found - skipping Cat 20 institutional ownership factors.")

    # Helper: merge a per-ticker external dataset that uses BASE ticker format
    # (e.g. "AAPL") onto our panel which uses exchange-suffixed format (e.g. "AAPL.O").
    def _merge_base_ticker_data(panel, ext_df, ext_cols, label):
        ext_df = ext_df.copy()
        ext_df["date"] = pd.to_datetime(ext_df["date"]) + pd.offsets.MonthEnd(0)
        # Build base→panel ticker map (one base maps to one panel ticker in our universe)
        panel_tickers = panel["ticker"].unique()
        base_to_panel = {}
        for t in panel_tickers:
            base = str(t).split(".")[0]
            if base not in base_to_panel:
                base_to_panel[base] = t
        ext_df["ticker"] = ext_df["ticker"].map(base_to_panel)
        ext_df = ext_df.dropna(subset=["ticker"])

        panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
        panel = panel.merge(
            ext_df[["date", "ticker"] + ext_cols],
            on=["date", "ticker"], how="left",
        )
        # Forward-fill per ticker; 0-fill initial NaNs
        panel = panel.sort_values(["ticker", "date"])
        panel[ext_cols] = panel.groupby("ticker")[ext_cols].ffill().fillna(0.0)
        n_merged = ext_df["ticker"].nunique()
        print(f"  {label} columns added: {ext_cols}  "
              f"(matched {n_merged}/{ext_df['ticker'].nunique()} tickers to panel)")
        return panel

    # Cat 23: news sentiment (optional - from 1o_fetch_sentiment.py)
    SENT_IN = DATA_DIR / "sentiment.parquet"
    if SENT_IN.exists():
        sent_df = pd.read_parquet(SENT_IN)
        if len(sent_df) > 0:
            print(f"\nFound {SENT_IN} - adding Cat 23: news sentiment factors...")
            sent_cols = [c for c in sent_df.columns if c not in ("date", "ticker")]
            panel = _merge_base_ticker_data(panel, sent_df, sent_cols, "Sentiment")
        else:
            print(f"\n{SENT_IN} exists but is empty - skipping Cat 23.")
    else:
        print(f"\n{SENT_IN} not found - skipping Cat 23 sentiment factors.")

    # Cat 24: insider trading (optional - from 1n_fetch_insider_trades.py)
    INS_IN = DATA_DIR / "insider_trades.parquet"
    if INS_IN.exists():
        ins_df = pd.read_parquet(INS_IN)
        if len(ins_df) > 0:
            print(f"\nFound {INS_IN} - adding Cat 24: insider trading factors...")
            ins_cols = [c for c in ins_df.columns if c not in ("date", "ticker")]
            panel = _merge_base_ticker_data(panel, ins_df, ins_cols, "Insider")
        else:
            print(f"\n{INS_IN} exists but is empty - skipping Cat 24.")
    else:
        print(f"\n{INS_IN} not found - skipping Cat 24 insider trading factors.")

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
        print("  Macro data unavailable - skipping Cat 10 and Cat 13.")

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
        print(f"\n{PRED_IN} not found - skipping Cat 21 prediction market factors.")

    # Cat 14: size factor - log(market cap).
    #
    # Prefer the caps 1c already computed. The shares cache this used to rely
    # on holds 152 tickers and writes them without the exchange suffix, so
    # merging it against a panel keyed on AAPL.O matched 142 of 677 names and
    # log_mktcap came out valid for 9% of rows. The other 91% were filled, which
    # left four fifths of every cross-section sharing one value: a factor that
    # cannot rank, yet one that had been surviving multiple-testing correction.
    #
    # spx_weights.parquet carries mktcap for 625 tickers in the panel's own
    # ticker format, so it is both wider and already aligned.
    weights_path = DATA_DIR / "spx_weights.parquet"
    panel["log_mktcap"] = np.nan

    if weights_path.exists():
        print("\nCat 14: log_mktcap from spx_weights.parquet...")
        caps = pd.read_parquet(weights_path)[["date", "ticker", "mktcap"]]
        caps["date"] = pd.to_datetime(caps["date"]) + pd.offsets.MonthEnd(0)
        panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
        panel = panel.merge(caps, on=["date", "ticker"], how="left")
        good = panel["mktcap"] > 0
        panel.loc[good, "log_mktcap"] = np.log(panel.loc[good, "mktcap"])
        panel = panel.drop(columns=["mktcap"])
        print(f"  log_mktcap: {int(good.sum()):,}/{len(panel):,} valid "
              f"({good.mean() * 100:.1f}%)")

    # Fall back to the shares cache only where 1c had nothing to say.
    if panel["log_mktcap"].isna().any():
        mktcap_cache = str(DATA_DIR / "mktcap_shares.parquet")
        all_tickers  = panel["ticker"].unique().tolist()
        print("Cat 14: filling gaps from the shares cache...")
        shares_df = fetch_market_cap_data(all_tickers, start_str, end_str,
                                          cache_path=mktcap_cache)
        filled = add_size_factor(panel.drop(columns=["log_mktcap"]), shares_df)
        panel["log_mktcap"] = panel["log_mktcap"].fillna(filled["log_mktcap"])
        print(f"  after fallback: {int(panel['log_mktcap'].notna().sum()):,}/"
              f"{len(panel):,} valid "
              f"({panel['log_mktcap'].notna().mean() * 100:.1f}%)")

    # Drop the helper column - it is not a model feature
    panel = panel.drop(columns=["close_me"], errors="ignore")

    # Cat 15: Tail-ranking features - explicit top/bottom decile dummies
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
    # Keep as binary {0,1} - no CS-ranking or TS z-scoring
    for cal_col in ["turn_of_month", "january_dummy"]:
        if cal_col in panel.columns:
            TAIL_COLS.add(cal_col)
    # Cat 17 proper time signals: binary {0,1} or signed {+1,-1} per stock
    # These are already absolute signals - CS-ranking would destroy their meaning
    for ts_col in [
        "above_ma_200", "above_ma_50", "new_52w_high", "new_52w_low",
        "tsmom_sign_12m", "tsmom_sign_6m", "vol_above_avg", "high_vol_week",
        "serial_corr_sign",
    ]:
        if ts_col in panel.columns:
            TAIL_COLS.add(ts_col)

    # Identify all feature columns
    always_exclude = {"date", "ticker", "fwd_ret_1m"}
    macro_in_panel = [c for c in MACRO_COLS if c in panel.columns]
    # Tail cols already normalised ({0,1}/{-1,0,1}) - skip CS-ranking
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
