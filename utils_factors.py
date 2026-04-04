"""
utils_factors.py
================
Centralised factor library for the ML trading pipeline.

All per-ticker functions accept a prices DataFrame sorted by [ticker, date] with columns:
  date, ticker, open, high, low, close   ← always required
  volume                                 ← optional (Cat 5 volume factors)
  spx_ret                                ← optional (Cat 2 beta, Cat 6 beta/corr)
  ret_d                                  ← pre-computed daily return (close.pct_change)

Convention:
  - Each add_*() function takes prices and returns prices with NEW columns appended in-place.
  - Intermediate columns used only as building blocks are prefixed with an underscore or
    kept only if explicitly listed in NEW COLUMNS comments.
  - Cross-sectional ranking happens LATER in 1_feature_engineering.py — these functions
    produce raw (unranked) values.
  - Macro functions return a date-indexed DataFrame (one row per date, merged on date).

Factor categories:
  Cat 1  — Multi-Horizon Momentum         (6 new daily-computed factors)
  Cat 2  — Volatility Regimes             (8 new factors)
  Cat 3  — Tail Risk                      (6 new factors)
  Cat 4  — Price Level / Trend            (11 new factors)
  Cat 5  — Volume & Liquidity             (6 new factors, requires volume)
  Cat 6  — Market Beta / Correlation      (8 new factors, requires spx_ret)
  Cat 7  — Intraday / Microstructure      (5 new factors)
  Cat 8  — Cross-Sectional Relative       (6 new monthly factors)
  Cat 9  — Fundamental / Quality          (12 optional factors)
  Cat 10 — Macro / Regime                 (9 factors, fetched from yfinance)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _rolling_maxdd(x: pd.Series, w: int) -> pd.Series:
    """
    Rolling maximum drawdown computed within each window of length w.
    Moved here from 1_feature_engineering.py and parameterised by window.
    """
    nav = (1 + x.fillna(0)).cumprod()
    roll_max = nav.rolling(w, min_periods=max(5, w // 3)).max()
    dd = nav / roll_max - 1
    return dd.rolling(w, min_periods=max(5, w // 3)).min()


def _wilder_rsi(close: pd.Series, w: int) -> pd.Series:
    """
    Wilder's RSI approximated via EWM (alpha = 1/w).
    Standard pandas-compatible implementation.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / w, min_periods=w, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / w, min_periods=w, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _trend_slope_norm(x: pd.Series, w: int) -> pd.Series:
    """
    Rolling OLS slope of x over w periods, normalised by the mean of x.
    Returns a dimensionless trend strength signal.
    """
    t = np.arange(w, dtype=float)
    t_mean = t.mean()
    t_dev  = t - t_mean
    ss_t   = float((t_dev ** 2).sum())

    def _slope(arr):
        if np.any(np.isnan(arr)) or ss_t < 1e-12:
            return np.nan
        y_mean = arr.mean()
        if abs(y_mean) < 1e-9:
            return 0.0
        return float((t_dev * (arr - y_mean)).sum() / ss_t) / (y_mean + 1e-9)

    return x.rolling(w, min_periods=w).apply(_slope, raw=True)


def _trend_r2(x: pd.Series, w: int) -> pd.Series:
    """
    R² of a rolling OLS linear fit of x over w periods.
    Measures how "trendy" the price movement is (1 = perfect trend).
    """
    t = np.arange(w, dtype=float)
    t_mean = t.mean()
    t_dev  = t - t_mean
    ss_t   = float((t_dev ** 2).sum())

    def _r2(arr):
        if np.any(np.isnan(arr)):
            return np.nan
        y_mean = arr.mean()
        ss_tot = float(((arr - y_mean) ** 2).sum())
        if ss_tot < 1e-14:
            return 1.0
        b = float((t_dev * (arr - y_mean)).sum() / ss_t)
        y_hat = y_mean + b * t_dev
        ss_res = float(((arr - y_hat) ** 2).sum())
        return max(0.0, 1.0 - ss_res / ss_tot)

    return x.rolling(w, min_periods=w).apply(_r2, raw=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 1 — MULTI-HORIZON MOMENTUM  (6 new daily-computed factors)
# ═══════════════════════════════════════════════════════════════════════════════

def add_momentum_daily(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add short and long-horizon momentum to daily prices.

    NEW COLUMNS: ret_1w, ret_2w, ret_9m, ret_18m, ret_24m, ret_36m
    Requires: close

    Note: ret_36m requires 756 trading days of history (≈3 years).
    For stocks with shorter histories the value is NaN → filled with 0 (neutral rank)
    in 1_feature_engineering.py.
    """
    grp = prices.groupby("ticker", group_keys=False)

    windows = {
        "ret_1w":  5,
        "ret_2w":  10,
        "ret_9m":  189,
        "ret_18m": 378,
        "ret_24m": 504,
        "ret_36m": 756,
    }
    for col, w in windows.items():
        min_p = max(3, w // 4)
        prices[col] = grp["close"].transform(
            lambda x, w=w, mp=min_p: x.pct_change(periods=w)
        )
    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 2 — VOLATILITY REGIMES  (8 new factors)
# ═══════════════════════════════════════════════════════════════════════════════

def add_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility regime features to daily prices.

    NEW COLUMNS:
      vol_5d, vol_21d, vol_126d      — rolling realised vol (std of ret_d)
      vol_ratio_st                   — vol_5d / vol_21d  (short-term regime)
      vol_trend                      — vol_21d / vol_63d (trend of vol)
      downvol_21d                    — downside deviation over 21d
      beta_21d, beta_63d             — CAPM beta to market (if spx_ret present)

    Requires: ret_d
    Optional: spx_ret (needed for beta_21d, beta_63d)
    """
    grp = prices.groupby("ticker", group_keys=False)

    for col, w in [("vol_5d", 5), ("vol_21d", 21), ("_vol_63d", 63), ("vol_126d", 126)]:
        prices[col] = grp["ret_d"].transform(
            lambda x, w=w: x.rolling(w, min_periods=max(3, w // 3)).std()
        )

    prices["vol_ratio_st"] = prices["vol_5d"]   / (prices["vol_21d"] + 1e-9)
    prices["vol_trend"]    = prices["vol_21d"]   / (prices["_vol_63d"] + 1e-9)

    # Downside deviation
    prices["_ret_neg"] = prices["ret_d"].clip(upper=0)
    prices["downvol_21d"] = grp["_ret_neg"].transform(
        lambda x: x.rolling(21, min_periods=10).std()
    )

    # CAPM betas — only when market returns are available
    if "spx_ret" in prices.columns:
        for w, col in [(21, "beta_21d"), (63, "beta_63d")]:
            def _beta(g, w=w):
                spx = prices.loc[g.index, "spx_ret"]
                cov_ = g.rolling(w, min_periods=max(10, w // 2)).cov(spx)
                var_ = spx.rolling(w, min_periods=max(10, w // 2)).var()
                return cov_ / (var_ + 1e-12)
            prices[col] = grp["ret_d"].transform(_beta)

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 3 — TAIL RISK  (6 new factors)
# ═══════════════════════════════════════════════════════════════════════════════

def add_tail_risk(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add tail-risk features to daily prices.

    NEW COLUMNS:
      kurt_60d        — rolling kurtosis (fat-tail signal)
      maxdd_21d       — 21-day rolling max drawdown
      maxdd_126d      — 126-day rolling max drawdown
      var_95_21d      — 5th percentile of daily returns over 21d (Value-at-Risk proxy)
      cvar_95_21d     — Conditional VaR: mean return below var_95_21d
      omega_ratio_21d — sum of positive returns / |sum of negative returns| over 21d

    Requires: ret_d
    """
    grp = prices.groupby("ticker", group_keys=False)

    # Rolling kurtosis
    prices["kurt_60d"] = grp["ret_d"].transform(
        lambda x: x.rolling(60, min_periods=20).kurt()
    )

    # Rolling max drawdowns
    prices["maxdd_21d"]  = grp["ret_d"].transform(lambda x: _rolling_maxdd(x, w=21))
    prices["maxdd_126d"] = grp["ret_d"].transform(lambda x: _rolling_maxdd(x, w=126))

    # VaR and CVaR (5th percentile = 95% VaR)
    prices["var_95_21d"] = grp["ret_d"].transform(
        lambda x: x.rolling(21, min_periods=10).quantile(0.05)
    )

    def _cvar(x: pd.Series, w: int = 21) -> pd.Series:
        """Mean of returns below the 5th percentile — rolling CVaR."""
        results = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            window = x.iloc[max(0, i - w + 1): i + 1].dropna().values
            if len(window) < 5:
                continue
            var_5 = np.percentile(window, 5)
            tail  = window[window <= var_5]
            if len(tail) > 0:
                results[i] = tail.mean()
        return pd.Series(results, index=x.index)

    prices["cvar_95_21d"] = grp["ret_d"].transform(_cvar)

    # Omega ratio — upside / downside mass
    def _omega(x: pd.Series, w: int = 21) -> pd.Series:
        pos = x.clip(lower=0).rolling(w, min_periods=10).sum()
        neg = (-x.clip(upper=0)).rolling(w, min_periods=10).sum()
        return pos / (neg + 1e-9)

    prices["omega_ratio_21d"] = grp["ret_d"].transform(_omega)

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 4 — PRICE LEVEL / TREND  (11 new factors)
# ═══════════════════════════════════════════════════════════════════════════════

def add_price_trend(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add price level and trend features to daily prices.

    NEW COLUMNS:
      price_to_ma10, price_to_ma50, price_to_ma100  — close relative to MAs
      nearness_52w_low    — close / 52-week low (distance from bottom)
      rsi_14, rsi_21      — Wilder RSI
      macd_signal         — (EMA12 - EMA26) / close  (normalized MACD)
      bollinger_pct       — Bollinger Band %B: (close - lower) / (upper - lower)
      trend_slope_21d     — OLS slope / price mean over 21d
      trend_slope_63d     — OLS slope / price mean over 63d
      trend_r2_21d        — R² of 21d OLS trend

    Requires: close
    """
    grp = prices.groupby("ticker", group_keys=False)

    # Moving averages (10, 20, 50, 100, 200 day)
    for w in [10, 20, 50, 100, 200]:
        ma_col = f"_ma_{w}d"
        prices[ma_col] = grp["close"].transform(
            lambda x, w=w: x.rolling(w, min_periods=max(5, w // 2)).mean()
        )
        prices[f"price_to_ma{w}"] = prices["close"] / (prices[ma_col] + 1e-9) - 1

    # MA crossover signals (ratio of short MA to long MA, normalised)
    # >0 means short MA above long MA (bullish trend)
    prices["ma_cross_10_50"]  = prices["_ma_10d"]  / (prices["_ma_50d"]  + 1e-9) - 1
    prices["ma_cross_50_200"] = prices["_ma_50d"]  / (prices["_ma_200d"] + 1e-9) - 1

    # 52-week low nearness (distance from 52-week low — high = stock broke out of bottom)
    prices["_low_52w"] = grp["close"].transform(
        lambda x: x.rolling(252, min_periods=120).min()
    )
    prices["nearness_52w_low"] = prices["close"] / (prices["_low_52w"] + 1e-9) - 1

    # Wilder RSI
    prices["rsi_14"] = grp["close"].transform(lambda x: _wilder_rsi(x, w=14))
    # rsi_21 removed — R²=0.99 vs rsi_14, near-exact duplicate (VIF > 100)

    # MACD: line = (EMA12 - EMA26) / close, histogram = line - 9-day EMA of line
    ema12 = grp["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = grp["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd_line = ema12 - ema26
    prices["macd_signal"] = macd_line / (prices["close"] + 1e-9)
    # MACD histogram: MACD line minus its 9-day signal line (momentum of momentum)
    macd_sig_line = (prices.groupby("ticker", group_keys=False)["macd_signal"]
                     .transform(lambda x: x.ewm(span=9, adjust=False).mean()))
    prices["macd_hist"] = prices["macd_signal"] - macd_sig_line

    # Bollinger Band %B  (window=20, 2σ)
    bb_mid   = grp["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    bb_std   = grp["close"].transform(lambda x: x.rolling(20, min_periods=10).std())
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    band_width = bb_upper - bb_lower
    prices["bollinger_pct"] = (prices["close"] - bb_lower) / (band_width + 1e-9)

    # OLS trend slope (normalised) and R²
    prices["trend_slope_21d"] = grp["close"].transform(
        lambda x: _trend_slope_norm(x, w=21)
    )
    prices["trend_slope_63d"] = grp["close"].transform(
        lambda x: _trend_slope_norm(x, w=63)
    )
    prices["trend_r2_21d"] = grp["close"].transform(
        lambda x: _trend_r2(x, w=21)
    )

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 5 — VOLUME & LIQUIDITY  (6 new factors; degrades gracefully without volume)
# ═══════════════════════════════════════════════════════════════════════════════

def add_volume_liquidity(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume and liquidity features to daily prices.

    NEW COLUMNS (requires 'volume' column):
      dollar_vol_21d    — rolling 21d mean of close * volume
      dollar_vol_63d    — rolling 63d mean of close * volume
      vol_momentum_21d  — volume / 21d_avg_volume - 1  (volume surge signal)
      vol_trend_ratio   — 21d_avg_volume / 63d_avg_volume
      obv_signal        — OBV momentum: 21d pct change in cumulative OBV

    NEW COLUMNS (always, without volume):
      [hl_ratio_d and amihud_21d are computed in 1_feature_engineering.py existing code;
       this function adds the volume-based ones only]

    Degrades gracefully: if 'volume' not in prices.columns, skips all volume factors.
    """
    if "volume" not in prices.columns:
        return prices   # no volume data — skip silently

    grp = prices.groupby("ticker", group_keys=False)

    dv = prices["close"] * prices["volume"]
    prices["dollar_vol_21d"] = grp.apply(
        lambda g: (g["close"] * g["volume"]).rolling(21, min_periods=10).mean()
    ).reset_index(level=0, drop=True)
    prices["dollar_vol_63d"] = grp.apply(
        lambda g: (g["close"] * g["volume"]).rolling(63, min_periods=30).mean()
    ).reset_index(level=0, drop=True)

    avg_vol_21 = grp["volume"].transform(
        lambda x: x.rolling(21, min_periods=10).mean()
    )
    avg_vol_63 = grp["volume"].transform(
        lambda x: x.rolling(63, min_periods=30).mean()
    )
    prices["vol_momentum_21d"] = prices["volume"] / (avg_vol_21 + 1e-9) - 1
    prices["vol_trend_ratio"]  = avg_vol_21 / (avg_vol_63 + 1e-9)

    # OBV momentum: cumulative OBV, then 21-day pct change
    def _obv_signal(g: pd.DataFrame) -> pd.Series:
        direction = np.sign(g["ret_d"].fillna(0))
        obv = (direction * g["volume"]).cumsum()
        obv_lag = obv.shift(21)
        return (obv - obv_lag) / (obv_lag.abs() + 1e-9)

    prices["obv_signal"] = (
        prices.groupby("ticker", group_keys=False)
        .apply(_obv_signal)
        .reset_index(level=0, drop=True)
    )

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 6 — MARKET BETA / CORRELATION  (8 new factors)
# ═══════════════════════════════════════════════════════════════════════════════

def add_beta_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add market beta and correlation features.

    NEW COLUMNS (requires spx_ret):
      beta_252d       — rolling 252d CAPM beta to SPX
      corr_spx_21d    — rolling 21d Pearson correlation with SPX
      corr_spx_63d    — rolling 63d Pearson correlation with SPX
      idio_vol_252d   — std of (ret_d - beta_252d * spx_ret) over 252d
      up_beta_63d     — beta on up-market days (spx_ret > 0)
      down_beta_63d   — beta on down-market days (spx_ret < 0)

    Note: beta_21d and beta_63d may already be present from add_volatility_features;
    this function skips them if already computed.
    Requires: ret_d, spx_ret
    """
    if "spx_ret" not in prices.columns:
        return prices

    grp = prices.groupby("ticker", group_keys=False)

    # beta_252d
    def _beta_w(g, w):
        spx  = prices.loc[g.index, "spx_ret"]
        cov_ = g.rolling(w, min_periods=w // 2).cov(spx)
        var_ = spx.rolling(w, min_periods=w // 2).var()
        return cov_ / (var_ + 1e-12)

    prices["beta_252d"] = grp["ret_d"].transform(lambda g: _beta_w(g, 252))

    # Pearson correlations
    def _corr_w(g, w):
        spx = prices.loc[g.index, "spx_ret"]
        return g.rolling(w, min_periods=max(10, w // 2)).corr(spx)

    prices["corr_spx_21d"] = grp["ret_d"].transform(lambda g: _corr_w(g, 21))
    prices["corr_spx_63d"] = grp["ret_d"].transform(lambda g: _corr_w(g, 63))

    # Idiosyncratic volatility: std of residuals from CAPM
    if "beta_252d" in prices.columns:
        spx_ret = prices["spx_ret"]
        prices["_idio_ret"] = prices["ret_d"] - prices["beta_252d"] * spx_ret
        prices["idio_vol_252d"] = grp["_idio_ret"].transform(
            lambda x: x.rolling(252, min_periods=126).std()
        )

    # Conditional (asymmetric) beta: zeroing approximation
    # up_beta: use return on SPX-up days, zero on SPX-down days
    spx = prices["spx_ret"]
    prices["_up_ret"]   = prices["ret_d"].where(spx > 0, 0.0)
    prices["_up_spx"]   = spx.where(spx > 0, 0.0)
    prices["_down_ret"] = prices["ret_d"].where(spx < 0, 0.0)
    prices["_down_spx"] = spx.where(spx < 0, 0.0)

    def _cond_beta(g_ret, g_spx, w):
        cov_ = g_ret.rolling(w, min_periods=w // 3).cov(g_spx)
        var_ = g_spx.rolling(w, min_periods=w // 3).var()
        return cov_ / (var_ + 1e-12)

    up_ret_grp   = prices.groupby("ticker")["_up_ret"]
    down_ret_grp = prices.groupby("ticker")["_down_ret"]

    prices["up_beta_63d"] = (
        up_ret_grp.transform(
            lambda g: _cond_beta(g, prices.loc[g.index, "_up_spx"], 63)
        )
    )
    prices["down_beta_63d"] = (
        down_ret_grp.transform(
            lambda g: _cond_beta(g, prices.loc[g.index, "_down_spx"], 63)
        )
    )

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 7 — INTRADAY / MICROSTRUCTURE  (5 new factors)
# ═══════════════════════════════════════════════════════════════════════════════

def add_microstructure(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add intraday and microstructure features.

    NEW COLUMNS:
      avg_hl_range_21d   — average (high-low)/close over 21d (range proxy)
      avg_gap_21d        — average |open/prev_close - 1| over 21d (overnight gap)
      open_to_close_21d  — average (close-open)/open over 21d (intraday direction)
      atr_21d_norm       — 14-period EWM ATR normalised by close
      intraday_vol_21d   — rolling std of (close-open)/open over 21d

    Requires: open, high, low, close
    """
    grp = prices.groupby("ticker", group_keys=False)

    # High-low range
    hl = (prices["high"] - prices["low"]) / (prices["close"] + 1e-9)
    prices["avg_hl_range_21d"] = (
        prices.groupby("ticker")["close"].transform(
            lambda x: x  # placeholder
        )
    )
    # Computed per-ticker for alignment
    prices["_hl"] = hl
    prices["avg_hl_range_21d"] = grp["_hl"].transform(
        lambda x: x.rolling(21, min_periods=10).mean()
    )

    # Overnight gap: |open / previous close - 1|
    prev_close = grp["close"].transform(lambda x: x.shift(1))
    prices["_gap"] = (prices["open"] / (prev_close + 1e-9) - 1).abs()
    prices["avg_gap_21d"] = grp["_gap"].transform(
        lambda x: x.rolling(21, min_periods=10).mean()
    )

    # Intraday direction: (close - open) / open
    prices["_oc"] = (prices["close"] - prices["open"]) / (prices["open"] + 1e-9)
    prices["open_to_close_21d"] = grp["_oc"].transform(
        lambda x: x.rolling(21, min_periods=10).mean()
    )
    prices["intraday_vol_21d"] = grp["_oc"].transform(
        lambda x: x.rolling(21, min_periods=10).std()
    )

    # Average True Range (EWM Wilder smoothing, span=14), normalised by close
    high, low, close = prices["high"], prices["low"], prices["close"]
    prev_c = grp["close"].transform(lambda x: x.shift(1))
    tr = pd.concat([
        (high - low).rename("hl"),
        (high - prev_c).abs().rename("hpc"),
        (low  - prev_c).abs().rename("lpc"),
    ], axis=1).max(axis=1)
    prices["_tr"] = tr
    prices["atr_21d_norm"] = (
        grp["_tr"].transform(
            lambda x: x.ewm(span=14, min_periods=7, adjust=False).mean()
        ) / (close + 1e-9)
    )

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 8 — CROSS-SECTIONAL RELATIVE  (6 new monthly factors)
# ═══════════════════════════════════════════════════════════════════════════════

def add_cross_sectional_relative(
    panel: pd.DataFrame,
    spx_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add market-relative factors to the MONTHLY panel (called after sample_at_month_end).

    Parameters
    ----------
    panel : pd.DataFrame
        Monthly panel with columns including ret_1m, ret_3m, ret_6m, ret_12m,
        beta_252d, vol_21d (already sampled at month-end).
    spx_monthly : pd.DataFrame
        Date-indexed DataFrame with columns: spx_ret_1m, spx_ret_3m, spx_ret_6m,
        spx_ret_12m, spx_vol_63d  (market monthly returns and vol).

    NEW COLUMNS:
      residual_ret_12m  — ret_12m - beta_252d * spx_ret_12m  (CAPM alpha proxy)

    REMOVED (were exact duplicates after cross-sectional rank normalisation):
      ret_rel_spx_1m/3m/6m — identical to ret_1m/3m/6m (SPX return is constant cross-sectionally)
      vol_rel_spx_63d       — identical to vol_21d (denominator spx_vol_63d is constant)
      beta_adj_ret_12m      — alias of residual_ret_12m

    spx_ret_* and spx_vol_63d are kept in panel for MACRO_COLS time-series z-scoring.
    """
    panel = panel.copy()
    spx_monthly = spx_monthly.copy()
    spx_monthly.index = pd.to_datetime(spx_monthly.index)

    # Merge SPX monthly stats into panel on date (used as macro regime cols)
    panel = panel.merge(spx_monthly, on="date", how="left")

    # CAPM residual return: ret_12m - beta * spx_ret_12m
    if "ret_12m" in panel.columns and "beta_252d" in panel.columns and "spx_ret_12m" in panel.columns:
        panel["residual_ret_12m"] = (
            panel["ret_12m"] - panel["beta_252d"] * panel["spx_ret_12m"]
        )

    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 9 — FUNDAMENTAL / QUALITY  (12 factors, optional)
# ═══════════════════════════════════════════════════════════════════════════════

FUNDAMENTAL_COLS = [
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe", "roa", "gross_margin",
    "revenue_growth_yoy", "eps_growth_yoy",
    "debt_to_equity", "earnings_quality", "buyback_yield",
]


def add_fundamental_factors(
    panel: pd.DataFrame,
    fundamental: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge optional fundamental/quality factors into the monthly panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Monthly panel (date, ticker, features…).
    fundamental : pd.DataFrame
        Must have columns: date, ticker, plus any subset of FUNDAMENTAL_COLS.
        Expected to be point-in-time (no look-ahead): each row is the most
        recently reported value as of that month-end date.

    Returns panel with available fundamental columns left-joined.
    Missing values → NaN → filled with 0 (neutral rank) in 1_feature_engineering.py.
    """
    fund_cols_available = [c for c in FUNDAMENTAL_COLS if c in fundamental.columns]
    if not fund_cols_available:
        return panel

    fund_sub = fundamental[["date", "ticker"] + fund_cols_available].copy()
    fund_sub["date"] = pd.to_datetime(fund_sub["date"])
    panel = panel.merge(fund_sub, on=["date", "ticker"], how="left")
    print(f"  Added {len(fund_cols_available)} fundamental factors: {fund_cols_available}")
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 10 — MACRO / REGIME  (9 factors, fetched from yfinance)
# ═══════════════════════════════════════════════════════════════════════════════

MACRO_TICKERS = {
    "^VIX":      "vix",
    "^TNX":      "yield_10y_raw",   # 10Y Treasury yield (×0.01 to get decimal)
    "^IRX":      "yield_3m_raw",    # 3-month T-bill yield
    "DX-Y.NYB":  "dollar_index",    # US Dollar Index
    "HYG":       "hyg",             # High-yield bond ETF (credit proxy)
}


def fetch_macro_data(
    start_date: str,
    end_date: str,
    spx_daily: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fetch macro / regime data from yfinance.
    Returns a date-indexed DataFrame with daily macro observations.

    Outputs:
      vix_level           — VIX closing level
      vix_change_21d      — 21-day pct change in VIX
      yield_10y           — 10Y Treasury yield (decimal)
      yield_spread_10y2y  — proxy: yield_10y - yield_3m  (2Y not available free)
      yield_change_21d    — 21-day change in yield_10y
      dollar_index        — DXY level
      credit_proxy_change — 21-day pct change in HYG (high-yield ETF)
      market_trend_spx    — SPX close / 200d MA - 1  (from spx_daily if provided)
      market_vol_regime   — VIX / 20.0  (continuous normalised; >1 = high-vol regime)
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [WARN] yfinance not installed — skipping macro factors")
        return pd.DataFrame()

    tickers = list(MACRO_TICKERS.keys())
    raw = yf.download(tickers, start=start_date, end=end_date,
                      auto_adjust=True, progress=False)["Close"]
    raw.columns = [MACRO_TICKERS[t] for t in raw.columns if t in MACRO_TICKERS]
    raw = raw.reset_index().rename(columns={"Date": "date"})
    raw["date"] = pd.to_datetime(raw["date"])

    macro = pd.DataFrame({"date": raw["date"]})

    # VIX
    if "vix" in raw.columns:
        macro["vix_level"]      = raw["vix"].values
        macro["vix_change_21d"] = raw["vix"].pct_change(21).values
        macro["market_vol_regime"] = raw["vix"].values / 20.0   # continuous normalisation

    # Yields
    if "yield_10y_raw" in raw.columns:
        macro["yield_10y"]       = raw["yield_10y_raw"].values / 100.0
        macro["yield_change_21d"] = (raw["yield_10y_raw"] / 100.0).diff(21).values
    if "yield_3m_raw" in raw.columns and "yield_10y_raw" in raw.columns:
        macro["yield_spread_10y2y"] = (
            (raw["yield_10y_raw"] - raw["yield_3m_raw"]) / 100.0
        ).values

    # Dollar index
    if "dollar_index" in raw.columns:
        macro["dollar_index"] = raw["dollar_index"].values

    # Credit proxy (HYG 21d pct change)
    if "hyg" in raw.columns:
        macro["credit_proxy_change"] = raw["hyg"].pct_change(21).values

    # Market trend: SPX vs 200d MA (from pre-fetched spx_daily if provided)
    if spx_daily is not None and "spx_close" in spx_daily.columns:
        spx = spx_daily.set_index("date")["spx_close"]
        ma200 = spx.rolling(200, min_periods=100).mean()
        market_trend = (spx / (ma200 + 1e-9) - 1).reset_index()
        market_trend.columns = ["date", "market_trend_spx"]
        macro = macro.merge(market_trend, on="date", how="left")

    macro = macro.set_index("date").sort_index()
    return macro


def add_macro_factors(
    panel: pd.DataFrame,
    macro_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge macro / regime factors into the monthly panel.
    Macro factors are sampled at each month-end date (merge_asof).

    NOTE: Macro columns are NOT cross-sectionally ranked (same for all stocks
    in a month → zero cross-sectional variance). They are time-series z-scored
    in 1_feature_engineering.py after this merge.
    """
    if macro_daily.empty:
        return panel

    macro_reset = macro_daily.reset_index()
    macro_reset = macro_reset.sort_values("date")
    macro_reset["date"] = pd.to_datetime(macro_reset["date"]).astype("datetime64[us]")

    panel_sorted = panel.sort_values("date")
    panel_sorted["date"] = panel_sorted["date"].astype("datetime64[us]")
    result = pd.merge_asof(
        panel_sorted,
        macro_reset,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("35 days"),
    )
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 11 — TIME SIGNAL FACTORS  (5 new factors)
# Trend quality, momentum consistency, skewness — regime-sensitive signals
# ═══════════════════════════════════════════════════════════════════════════════

def add_time_signal_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-series signal quality factors.

    NEW COLUMNS:
      ir_6m               — 6m info ratio: ret_6m / vol_126d
      trend_r2_126d       — R² of 126d OLS linear trend (trend persistence)
      ret_consistency_12m — fraction of last 252 trading days with positive return
      skew_60d            — 60d rolling return skewness (negative = crash risk)
      drawdown_pct_252d   — current price / 252d rolling max - 1 (distance from ATH)

    Requires: close, ret_d
    """
    grp = prices.groupby("ticker", group_keys=False)

    # 6m information ratio: annualised return / vol
    if "ret_6m" not in prices.columns:
        prices["_ret_6m_ts"] = grp["close"].transform(
            lambda x: x.pct_change(126)
        )
        ret_6m_col = "_ret_6m_ts"
    else:
        ret_6m_col = "ret_6m"

    vol_126 = grp["ret_d"].transform(
        lambda x: x.rolling(126, min_periods=60).std()
    )
    prices["ir_6m"] = prices[ret_6m_col] / (vol_126 + 1e-9)

    # 126d trend R² (persistence of trend — higher = more trending)
    prices["trend_r2_126d"] = grp["close"].transform(
        lambda x: _trend_r2(x, w=126)
    )

    # Return consistency: % of last 252 daily returns that were positive
    prices["ret_consistency_12m"] = grp["ret_d"].transform(
        lambda x: x.rolling(252, min_periods=120).apply(
            lambda w: (w > 0).mean(), raw=True
        )
    )

    # 60d rolling skewness
    prices["skew_60d"] = grp["ret_d"].transform(
        lambda x: x.rolling(60, min_periods=20).skew()
    )

    # Distance from 252d rolling high (current drawdown from ATH)
    rolling_max_252 = grp["close"].transform(
        lambda x: x.rolling(252, min_periods=120).max()
    )
    prices["drawdown_pct_252d"] = prices["close"] / (rolling_max_252 + 1e-9) - 1

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 12 — BARRA-STYLE FACTORS  (4 new factors)
# Size proxy, vol-of-vol, beta stability — cross-sectional risk exposures
# ═══════════════════════════════════════════════════════════════════════════════

def add_barra_style_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add Barra-inspired cross-sectional risk exposure factors.

    NEW COLUMNS:
      amihud_illiq_21d  — Amihud illiquidity: mean(|ret_d| / dollar_vol) over 21d
                          High = illiquid = small-cap proxy (Barra "SIZE" factor)
      vol_of_vol_63d    — rolling std of vol_21d over 63d (Barra "DASTD" factor)
      beta_stability_63d — rolling std of beta_21d over 63d (unstable beta → risk)
      size_proxy        — log(dollar_vol_63d + 1) as continuous size proxy

    amihud_illiq_21d and size_proxy require 'volume' column.
    vol_of_vol_63d and beta_stability_63d require vol_21d and beta_21d columns.
    All degrade gracefully if inputs are absent.
    """
    grp = prices.groupby("ticker", group_keys=False)

    # Amihud illiquidity and size proxy (need volume)
    if "volume" in prices.columns:
        dollar_vol = prices["close"] * prices["volume"]
        prices["_dollar_vol_d"] = dollar_vol

        # Amihud: mean(|ret_d| / dollar_vol) — high = illiquid = small
        prices["_amihud_d"] = prices["ret_d"].abs() / (dollar_vol + 1e-9)
        prices["amihud_illiq_21d"] = grp["_amihud_d"].transform(
            lambda x: x.rolling(21, min_periods=10).mean() * 1e6  # scale for readability
        )

        # Size proxy: log of 63d mean dollar volume
        dv63 = grp["_dollar_vol_d"].transform(
            lambda x: x.rolling(63, min_periods=30).mean()
        )
        prices["size_proxy"] = np.log(dv63.clip(lower=1e-9) + 1)
    else:
        prices["amihud_illiq_21d"] = np.nan
        prices["size_proxy"] = np.nan

    # Vol-of-vol: rolling std of vol_21d (requires vol_21d from Cat 2)
    if "vol_21d" in prices.columns:
        prices["vol_of_vol_63d"] = grp["vol_21d"].transform(
            lambda x: x.rolling(63, min_periods=30).std()
        )
    else:
        prices["vol_of_vol_63d"] = np.nan

    # Beta stability: rolling std of beta_21d over 63d (requires beta_21d from Cat 2)
    if "beta_21d" in prices.columns:
        prices["beta_stability_63d"] = grp["beta_21d"].transform(
            lambda x: x.rolling(63, min_periods=30).std()
        )
    else:
        prices["beta_stability_63d"] = np.nan

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 13 — CROSS-SECTIONAL INTERACTION FACTORS  (5 new factors, monthly level)
# Multi-dimensional cross-sectional signals and CAPM residuals
# Called AFTER add_macro_factors() so spx_ret_* cols are present in panel
# ═══════════════════════════════════════════════════════════════════════════════

def add_macro_interaction_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional interaction factors to the monthly panel.

    Note on design: products of (macro_scalar × CS_var) collapse to rank(CS_var)
    after cross-sectional ranking (macro scalar is constant within each month).
    Instead, these factors combine TWO cross-sectional variables or a CS variable
    with a market-relative benchmark, producing genuinely 2D cross-sectional signals.

    NEW COLUMNS:
      residual_ret_1m     — ret_1m - beta_252d × spx_ret_1m   (1m CAPM alpha)
      beta_x_idiovol      — beta_252d × idio_vol_252d          (systematic × idiosyncratic risk)
      up_down_beta_spread — up_beta_63d - down_beta_63d         (directional beta asymmetry)
      vol_excess          — vol_21d / (|beta_252d| × spx_vol_63d + ε) - 1
                            (stock vol in excess of market-implied vol)
      mom_decel           — ret_1m / (|ret_6m| / 6 + ε) - 1   (short-term vs avg 6m momentum)

    All degrade gracefully if underlying columns are missing.
    """
    panel = panel.copy()

    # 1-month CAPM alpha: idiosyncratic return stripped of market move
    if all(c in panel.columns for c in ["ret_1m", "beta_252d", "spx_ret_1m"]):
        panel["residual_ret_1m"] = (
            panel["ret_1m"] - panel["beta_252d"] * panel["spx_ret_1m"]
        )
    else:
        panel["residual_ret_1m"] = np.nan

    # Systematic × idiosyncratic risk: highest for stocks with both high beta and high IVOL
    if "beta_252d" in panel.columns and "idio_vol_252d" in panel.columns:
        panel["beta_x_idiovol"] = panel["beta_252d"].abs() * panel["idio_vol_252d"]
    else:
        panel["beta_x_idiovol"] = np.nan

    # Directional beta asymmetry: stocks with high up-beta and low down-beta are desirable
    if "up_beta_63d" in panel.columns and "down_beta_63d" in panel.columns:
        panel["up_down_beta_spread"] = panel["up_beta_63d"] - panel["down_beta_63d"]
    else:
        panel["up_down_beta_spread"] = np.nan

    # Excess volatility over market-implied (vol not explained by market beta)
    if all(c in panel.columns for c in ["vol_21d", "beta_252d", "spx_vol_63d"]):
        market_implied_vol = panel["beta_252d"].abs() * panel["spx_vol_63d"]
        panel["vol_excess"] = panel["vol_21d"] / (market_implied_vol + 1e-9) - 1
    else:
        panel["vol_excess"] = np.nan

    # Momentum deceleration: recent 1m return vs average monthly run rate over 6m
    if "ret_1m" in panel.columns and "ret_6m" in panel.columns:
        avg_monthly_6m = panel["ret_6m"].abs() / 6.0
        panel["mom_decel"] = panel["ret_1m"] / (avg_monthly_6m + 1e-9) - 1
    else:
        panel["mom_decel"] = np.nan

    cat13_cols = [
        "residual_ret_1m", "beta_x_idiovol", "up_down_beta_spread",
        "vol_excess", "mom_decel",
    ]
    added = [c for c in cat13_cols if c in panel.columns]
    print(f"  Cat 13 interaction factors added: {added}")
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# CAT 14 — SIZE FACTOR  (log market cap, fetched from yfinance with caching)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_market_cap_data(
    tickers: list,
    start_date: str,
    end_date: str,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch current shares outstanding from yfinance for each ticker.
    Returns long-form DataFrame: [date, ticker, shares].
    Uses fast_info.shares (lightweight endpoint, ~0.3s/ticker, no 404 spam).
    Caches to cache_path after first run — subsequent runs load instantly.
    """
    import warnings as _warnings
    try:
        import yfinance as yf
    except ImportError:
        print("  [WARN] yfinance not installed — skipping market cap (Cat 14)")
        return pd.DataFrame()

    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        print(f"  Loading market cap cache: {cache}")
        return pd.read_parquet(cache)

    print(f"  Fetching shares outstanding for {len(tickers)} tickers "          f"(~2-3 min first time, then cached)...")
    rows     = []
    n_ok     = 0
    n_failed = 0
    ref_date = pd.Timestamp(end_date)

    for i, tk in enumerate(tickers):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(tickers)}  ok={n_ok}  failed={n_failed}")
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                t  = yf.Ticker(tk)
                so = None
                # fast_info.shares — quickest, no full info page needed
                try:
                    fi = t.fast_info
                    v  = getattr(fi, "shares", None)
                    if v and float(v) > 0:
                        so = float(v)
                except Exception:
                    pass
                # Fallback: info dict
                if not so:
                    try:
                        info = t.info
                        for key in ("sharesOutstanding",
                                    "impliedSharesOutstanding"):
                            v = info.get(key)
                            if v and float(v) > 0:
                                so = float(v)
                                break
                    except Exception:
                        pass
            if so:
                rows.append({"date": ref_date, "ticker": tk, "shares": so})
                n_ok += 1
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    print(f"  Done: {n_ok} fetched, {n_failed} failed "          f"({n_ok}/{len(tickers)} tickers covered)")

    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
          .assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
          .dropna(subset=["shares"])
          .sort_values(["ticker", "date"])
          .reset_index(drop=True))

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        print(f"  Saved cache → {cache}")

    return df


def add_size_factor(
    panel: pd.DataFrame,
    shares_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add log_mktcap to the monthly panel.
    Requires 'close_me' in panel (month-end close price, added in build_daily_features).
    Market cap = shares_outstanding × close_me.

    NEW COLUMN: log_mktcap — log(market cap in $).
    Cross-sectionally ranked in 1g (higher rank = larger cap).
    Falls back to NaN (→ filled with 0 = neutral rank) if data unavailable.
    """
    panel = panel.copy()

    if shares_df.empty or "close_me" not in panel.columns:
        print("  [WARN] add_size_factor: missing shares_df or close_me → log_mktcap = NaN")
        panel["log_mktcap"] = np.nan
        return panel

    shares_sorted = shares_df.sort_values("date")   # merge_asof requires sort by `on` key only
    shares_sorted["date"] = shares_sorted["date"].astype("datetime64[us]")
    panel_sorted  = panel.sort_values(["date", "ticker"])
    panel_sorted["date"] = panel_sorted["date"].astype("datetime64[us]")

    merged = pd.merge_asof(
        panel_sorted,
        shares_sorted,
        on="date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta("185 days"),   # up to 6-month-old quarterly filing
    )

    valid = (merged["shares"].notna()
             & merged["close_me"].notna()
             & (merged["close_me"] > 0)
             & (merged["shares"] > 0))
    mktcap = np.where(valid, merged["shares"] * merged["close_me"], np.nan)
    merged["log_mktcap"] = np.where(
        np.isfinite(mktcap) & (mktcap > 0),
        np.log(mktcap),
        np.nan,
    )
    merged = merged.drop(columns=["shares"], errors="ignore")

    n_valid = merged["log_mktcap"].notna().sum()
    print(f"  log_mktcap: {n_valid:,}/{len(merged):,} valid "
          f"({100*n_valid/max(len(merged),1):.1f}%)")

    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# CAT 15 — TAIL RANKING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_tail_ranking_features(
    panel: pd.DataFrame,
    base_factors: list,
    top_pct: float = 0.10,
    bot_pct: float = 0.10,
) -> pd.DataFrame:
    """
    Cat 15: Non-linear tail membership features.

    Rationale (per Kieran):
      "Use contrarian factors, use quantile test to make those tail ranking factors."
      The CS-Transformer sees continuous ranks [0,1].  Explicit tail dummies give
      it a hard signal: this stock is in the EXTREME TOP/BOTTOM decile — a regime
      the model might under-weight with smooth inputs alone.

    For each base factor we add:
      {f}_top  : 1 if CS-rank >= (1 - top_pct), else 0  (top decile long candidate)
      {f}_bot  : 1 if CS-rank <= bot_pct,        else 0  (bottom decile short candidate)
      {f}_tail : +1 top, -1 bottom, 0 middle              (signed tail indicator)

    These are computed CROSS-SECTIONALLY within each month so there is no
    lookahead.  The resulting columns are already in {0,1} / {-1,0,+1} space —
    no additional CS ranking is applied (marked via TAIL_COLS list in 1g).

    Parameters
    ----------
    panel        : monthly enriched panel (must already have base_factors columns)
    base_factors : list of factor column names to build tail features from
    top_pct      : fraction defining "top" tail (default 10%)
    bot_pct      : fraction defining "bottom" tail (default 10%)

    Returns
    -------
    panel with new {f}_top / {f}_bot / {f}_tail columns appended.
    """
    panel = panel.copy()
    new_cols_added = []

    valid_factors = [f for f in base_factors if f in panel.columns]

    for dt, grp_idx in panel.groupby("date").groups.items():
        grp = panel.loc[grp_idx, valid_factors]
        # Cross-sectional percentile rank within this month
        pct_rank = grp.rank(pct=True, method="average")

        for f in valid_factors:
            top_col  = f"{f}_top"
            bot_col  = f"{f}_bot"
            tail_col = f"{f}_tail"
            col = pct_rank[f]
            panel.loc[grp_idx, top_col]  = (col >= 1.0 - top_pct).astype(np.float32)
            panel.loc[grp_idx, bot_col]  = (col <= bot_pct).astype(np.float32)
            panel.loc[grp_idx, tail_col] = (
                (col >= 1.0 - top_pct).astype(float)
                - (col <= bot_pct).astype(float)
            ).astype(np.float32)

    for f in valid_factors:
        new_cols_added += [f"{f}_top", f"{f}_bot", f"{f}_tail"]

    n_new = sum(1 for c in new_cols_added if c in panel.columns)
    print(f"  Cat 15: {n_new} tail-ranking features added "
          f"(top {int(top_pct*100)}% / bot {int(bot_pct*100)}% per {len(valid_factors)} base factors)")
    return panel
