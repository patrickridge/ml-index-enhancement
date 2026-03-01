"""
1_feature_engineering.py
=========================
Reads data/panel_monthly.parquet (your existing 10 features) and
data/prices.parquet (daily OHLCV), then engineers ~25 additional
features grounded in factor literature. Saves enriched panel to
data/panel_monthly_enriched.parquet.

New feature groups:
  A) Momentum extensions     (Jegadeesh & Titman, 1993)
  B) Short-term reversal     (Jegadeesh, 1990)
  C) Volatility ratios       (low-vol anomaly, Ang et al. 2006)
  D) Volume / liquidity      (Datar et al. 1998; Amihud 2002)
  E) Price level / 52w high  (George & Hwang 2004)
  F) Trend / MA signals      (Han et al. 2016)
  G) Cross-sectional rank    (rank-normalise key signals -- tree models love this)

All features are cross-sectionally rank-normalised within each month
to [-0.5, 0.5] to make them comparable and reduce outlier impact.

Run time: ~2-5 minutes on a laptop.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
PANEL_IN  = DATA_DIR / "panel_monthly.parquet"
PRICES_IN = DATA_DIR / "prices.parquet"
OUT_PATH  = DATA_DIR / "panel_monthly_enriched.parquet"

# ── rolling window lengths (trading days) ─────────────────────────────────────
VOL_SHORT  = 20
VOL_MED    = 60
VOL_LONG   = 252
MA_SHORT   = 20
MA_MED     = 60
MA_LONG    = 200
AMIHUD_WIN = 21   # ~1 month of daily observations


def rank_norm(s: pd.Series) -> pd.Series:
    """Cross-sectional rank normalised to [-0.5, 0.5]. NaN stays NaN."""
    r = s.rank(pct=True, na_option="keep")
    return r - 0.5


def cs_rank_all(df: pd.DataFrame, feat_cols: list, date_col: str = "date") -> pd.DataFrame:
    """Apply rank_norm to feat_cols cross-sectionally within each date."""
    df = df.copy()
    for c in feat_cols:
        df[c] = df.groupby(date_col)[c].transform(rank_norm)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: build daily feature table from prices
# ─────────────────────────────────────────────────────────────────────────────
def build_daily_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker daily features that we will then sample at month-end.
    Input: prices with columns [date, ticker, open, high, low, close]
    Output: same index, with added feature columns
    """
    prices = prices.sort_values(["ticker", "date"]).copy()

    grp = prices.groupby("ticker", group_keys=False)

    # --- returns ---
    prices["ret_d"] = grp["close"].pct_change()

    # dollar volume proxy (close * volume not available; use close as proxy for price level)
    # We DO have open/high/low so we can build range-based liquidity
    prices["hl_ratio_d"] = (prices["high"] - prices["low"]) / prices["close"].clip(lower=1e-6)

    # --- volatility features ---
    for w in [VOL_SHORT, VOL_MED, VOL_LONG]:
        prices[f"vol_{w}d_raw"] = grp["ret_d"].transform(
            lambda x: x.rolling(w, min_periods=max(5, w // 2)).std()
        )

    # volatility ratios (short/long) -- captures vol regime change
    prices["vol_ratio_sm"] = prices[f"vol_{VOL_SHORT}d_raw"] / (prices[f"vol_{VOL_MED}d_raw"] + 1e-9)
    prices["vol_ratio_ml"] = prices[f"vol_{VOL_MED}d_raw"] / (prices[f"vol_{VOL_LONG}d_raw"] + 1e-9)

    # --- moving average features ---
    for w in [MA_SHORT, MA_MED, MA_LONG]:
        prices[f"ma_{w}d"] = grp["close"].transform(
            lambda x: x.rolling(w, min_periods=max(5, w // 2)).mean()
        )

    # price relative to MA (trend signal)
    prices["price_to_ma20"]  = prices["close"] / (prices["ma_20d"]  + 1e-9) - 1
    prices["price_to_ma60"]  = prices["close"] / (prices["ma_60d"]  + 1e-9) - 1
    prices["price_to_ma200"] = prices["close"] / (prices["ma_200d"] + 1e-9) - 1

    # MA cross-over ratio
    prices["ma_20_60_cross"]  = prices["ma_20d"]  / (prices["ma_60d"]  + 1e-9) - 1
    prices["ma_60_200_cross"] = prices["ma_60d"]  / (prices["ma_200d"] + 1e-9) - 1

    # --- 52-week high (George & Hwang 2004) ---
    prices["high_52w"] = grp["close"].transform(
        lambda x: x.rolling(252, min_periods=120).max()
    )
    prices["nearness_52w"] = prices["close"] / (prices["high_52w"] + 1e-9) - 1

    # --- Amihud illiquidity (abs_ret / range as proxy; no volume col) ---
    # Use |ret| / hl_ratio as a rough liquidity measure
    prices["amihud_proxy"] = prices["ret_d"].abs() / (prices["hl_ratio_d"] + 1e-9)
    prices["amihud_21d"] = grp["amihud_proxy"].transform(
        lambda x: x.rolling(AMIHUD_WIN, min_periods=10).mean()
    )

    # --- downside deviation (Sortino proxy) ---
    prices["ret_neg"] = prices["ret_d"].clip(upper=0)
    prices["downvol_60d"] = grp["ret_neg"].transform(
        lambda x: x.rolling(VOL_MED, min_periods=20).std()
    )
    # upside / downside vol ratio
    prices["ret_pos"] = prices["ret_d"].clip(lower=0)
    prices["upvol_60d"] = grp["ret_pos"].transform(
        lambda x: x.rolling(VOL_MED, min_periods=20).std()
    )
    prices["up_down_vol"] = prices["upvol_60d"] / (prices["downvol_60d"] + 1e-9)

    # --- realised skewness (60d) ---
    prices["skew_60d"] = grp["ret_d"].transform(
        lambda x: x.rolling(VOL_MED, min_periods=20).skew()
    )

    # --- realised max drawdown (60d) ---
    def rolling_maxdd(x: pd.Series, w: int = 60) -> pd.Series:
        nav = (1 + x.fillna(0)).cumprod()
        roll_max = nav.rolling(w, min_periods=10).max()
        dd = nav / roll_max - 1
        return dd.rolling(w, min_periods=10).min()

    prices["maxdd_60d"] = grp["ret_d"].transform(rolling_maxdd)

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: sample daily features at each month-end date in panel
# ─────────────────────────────────────────────────────────────────────────────
def sample_at_month_end(daily: pd.DataFrame, panel_dates: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, ticker) in panel_dates, find the last available daily
    row on or before that date and extract the feature columns.
    """
    # columns we want to bring in
    new_feat_cols = [
        "vol_ratio_sm", "vol_ratio_ml",
        "price_to_ma20", "price_to_ma60", "price_to_ma200",
        "ma_20_60_cross", "ma_60_200_cross",
        "nearness_52w",
        "amihud_21d",
        "up_down_vol",
        "skew_60d",
        "maxdd_60d",
    ]
    keep_cols = ["date", "ticker"] + new_feat_cols

    daily_sub = daily[keep_cols].dropna(subset=["date", "ticker"])
    daily_sub = daily_sub.sort_values(["ticker", "date"])

    # merge_asof requires both sides sorted by the merge key (date)
    left = panel_dates[["date", "ticker"]].copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    right = daily_sub.sort_values(["date", "ticker"]).reset_index(drop=True)

    result = pd.merge_asof(
        left,
        right,
        on="date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta("35 days"),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: momentum extensions from panel (already monthly)
# ─────────────────────────────────────────────────────────────────────────────
def add_momentum_extensions(panel: pd.DataFrame) -> pd.DataFrame:
    """
    From existing ret_1m .. ret_12m, build additional momentum signals.
    """
    panel = panel.copy()

    # skip-1-month momentum (2-12, classic Jegadeesh & Titman)
    # ret_12m includes last 12 months, ret_1m is most recent; difference ~ 2-12m mom
    if "ret_12m" in panel.columns and "ret_1m" in panel.columns:
        panel["mom_2_12"] = panel["ret_12m"] - panel["ret_1m"]

    # momentum acceleration: recent vs older
    if "ret_3m" in panel.columns and "ret_12m" in panel.columns:
        panel["mom_accel"] = panel["ret_3m"] - (panel["ret_12m"] - panel["ret_3m"])

    # vol-adjusted momentum (return / vol -- information ratio proxy)
    if "ret_12m" in panel.columns and "vol_252d" in panel.columns:
        panel["ir_12m"] = panel["ret_12m"] / (panel["vol_252d"].abs() + 1e-6)

    if "ret_3m" in panel.columns and "vol_60d" in panel.columns:
        panel["ir_3m"] = panel["ret_3m"] / (panel["vol_60d"].abs() + 1e-6)

    # reversal signal: if large prior move, expect mean reversion
    if "ret_1m" in panel.columns:
        panel["rev_signal"] = -panel["ret_1m"]   # short-term reversal

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    prices = pd.read_parquet(PRICES_IN)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"Panel: {panel.shape} | Prices: {prices.shape}")

    # --- Step A: daily features ---
    print("Building daily features (this takes ~2-3 min)...")
    daily = build_daily_features(prices)

    # --- Step B: sample at month ends ---
    print("Sampling daily features at month-end dates...")
    daily_at_month = sample_at_month_end(daily, panel[["date", "ticker"]])

    # --- Step C: merge into panel ---
    panel = panel.merge(daily_at_month, on=["date", "ticker"], how="left")

    # --- Step D: add momentum extensions ---
    print("Adding momentum extensions...")
    panel = add_momentum_extensions(panel)

    # --- Step E: cross-sectional rank normalise ALL features ---
    # (keep fwd_ret_1m raw as target, rank everything else)
    print("Cross-sectional rank normalising features...")
    exclude = {"date", "ticker", "fwd_ret_1m"}
    feat_cols = [c for c in panel.columns if c not in exclude]
    panel = cs_rank_all(panel, feat_cols, date_col="date")

    # --- Step F: drop rows where target is NaN ---
    before = len(panel)
    panel = panel.dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)
    print(f"Dropped {before - len(panel)} rows with NaN target.")

    # --- Step G: fill remaining NaNs with 0 (neutral rank = no signal) ---
    panel[feat_cols] = panel[feat_cols].fillna(0.0)

    # --- Save ---
    panel.to_parquet(OUT_PATH, index=False)

    print(f"\nDone! Saved to {OUT_PATH}")
    print(f"Shape: {panel.shape}")
    print(f"Features ({len(feat_cols)}): {feat_cols}")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Tickers: {panel['ticker'].nunique()}")


if __name__ == "__main__":
    main()
