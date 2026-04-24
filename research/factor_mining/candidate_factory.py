"""
candidate_factory.py
====================
Generate ~80-100 candidate features from raw OHLCV data for Track 1 factor mining.
Total time: 7950s

Each candidate is designed to capture information NOT already in the existing 205 features.
All functions operate per-ticker on daily data and return values to be sampled at month-end.

Usage:
    from research.factor_mining.candidate_factory import build_all_candidates
    prices_with_candidates = build_all_candidates(prices)
"""

import numpy as np
import pandas as pd
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 1: PATH-DEPENDENT RETURN FEATURES
# These capture the *shape* of the price path, not just start-to-end returns.
# ══════���════════════���═══════════════════════════════════════════════════════════

def add_path_dependent(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Path-dependent return features - capture how prices got from A to B.

    NEW COLUMNS:
      cand_path_efficiency_21d   - |net return| / sum(|daily returns|) over 21d
                                   (1 = straight line, 0 = choppy path)
      cand_path_efficiency_63d   - same over 63d
      cand_up_path_ratio_21d     - sum(positive rets) / sum(|all rets|) over 21d
      cand_max_consec_up_63d     - max consecutive up-days in past 63d
      cand_max_consec_down_63d   - max consecutive down-days in past 63d
      cand_signed_path_21d       - sum(|daily ret|) × sign(net return)
                                   (measures path length with direction)
      cand_return_dispersion_21d - std of daily returns / |mean daily return| over 21d
                                   (coefficient of variation of daily returns)
      cand_path_skew_21d         - asymmetry of daily return distribution within window
    """
    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        ret_s = pd.Series(ret_d, index=idx)

        # Path efficiency: |net move| / total absolute path
        for w, suffix in [(21, "21d"), (63, "63d")]:
            net_ret = ret_s.rolling(w, min_periods=max(10, w // 3)).sum()
            total_path = ret_s.abs().rolling(w, min_periods=max(10, w // 3)).sum()
            eff = net_ret.abs() / (total_path + 1e-9)
            prices.loc[idx, f"cand_path_efficiency_{suffix}"] = eff.values

        # Up-path ratio: what fraction of total absolute movement was upward?
        pos_path = ret_s.clip(lower=0).rolling(21, min_periods=10).sum()
        total_abs = ret_s.abs().rolling(21, min_periods=10).sum()
        prices.loc[idx, "cand_up_path_ratio_21d"] = (pos_path / (total_abs + 1e-9)).values

        # Max consecutive up/down days
        def _max_consec(x, direction="up"):
            sign = (x > 0) if direction == "up" else (x < 0)
            max_run = 0
            current_run = 0
            for v in sign:
                if v:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            return max_run

        prices.loc[idx, "cand_max_consec_up_63d"] = (
            ret_s.rolling(63, min_periods=21).apply(
                lambda x: _max_consec(x, "up"), raw=True
            ).values
        )
        prices.loc[idx, "cand_max_consec_down_63d"] = (
            ret_s.rolling(63, min_periods=21).apply(
                lambda x: _max_consec(x, "down"), raw=True
            ).values
        )

        # Signed path length
        total_path_21 = ret_s.abs().rolling(21, min_periods=10).sum()
        net_ret_21 = ret_s.rolling(21, min_periods=10).sum()
        prices.loc[idx, "cand_signed_path_21d"] = (
            total_path_21 * np.sign(net_ret_21)
        ).values

        # Return dispersion (CV of daily returns)
        mean_abs = ret_s.abs().rolling(21, min_periods=10).mean()
        std_ret = ret_s.rolling(21, min_periods=10).std()
        prices.loc[idx, "cand_return_dispersion_21d"] = (std_ret / (mean_abs + 1e-9)).values

        # Path skewness
        prices.loc[idx, "cand_path_skew_21d"] = (
            ret_s.rolling(21, min_periods=10).skew().values
        )

    return prices


# ═══════════���════════════════════��═════════════════════════════════���════════════
# FAMILY 2: TREND EFFICIENCY / CHOPPINESS
# Measures how "trendy" vs "choppy" a price series is.
# Different from trend_r2 which measures R² of linear fit.
# ════���═════════════════���════════════════════════════════════════════════════════

def add_trend_efficiency(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Trend efficiency and choppiness features.

    NEW COLUMNS:
      cand_kaufman_er_21d      ��� Kaufman Efficiency Ratio (|net|/sum(|daily|))
                                  DIFFERENT from path_efficiency: uses close-to-close
                                  vs day-by-day returns. ER is close[t]-close[t-n] /
                                  sum(|close[i]-close[i-1]|). Standard TA indicator.
      cand_kaufman_er_63d      - same over 63d
      cand_choppiness_14d      - Choppiness Index: 100 × log(sum(ATR)/range) / log(n)
                                  High = choppy/sideways, Low = trending
      cand_fractal_dim_63d     - Fractal dimension proxy from Higuchi method
                                  D ≈ 1.0 = trending, D ≈ 1.5 = random, D ≈ 2.0 = mean-reverting
      cand_directional_persist - Fraction of days where sign(ret[t]) == sign(ret[t-1])
                                  High = persistent trends, Low = frequent reversals
      cand_adx_proxy_14d       - ADX proxy: ratio of directional movement to true range
      cand_trend_break_63d     - Number of MA crossovers in 63d (low = persistent trend)
      cand_trend_acceleration   - d(trend_slope_21d)/dt - is the trend accelerating?
    """
    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        close = grp["close"].values
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        ret_s = pd.Series(ret_d, index=idx)
        cls_s = pd.Series(close, index=idx)

        # Kaufman ER: |close[t]-close[t-n]| / sum(|close[i]-close[i-1]|)
        for w, suffix in [(21, "21d"), (63, "63d")]:
            net_move = cls_s.diff(w).abs()
            daily_moves = cls_s.diff().abs().rolling(w, min_periods=max(10, w // 3)).sum()
            er = net_move / (daily_moves + 1e-9)
            prices.loc[idx, f"cand_kaufman_er_{suffix}"] = er.values

        # Choppiness Index (14-day)
        if "high" in grp.columns and "low" in grp.columns:
            high = grp["high"].values
            low = grp["low"].values
            prev_c = np.concatenate([[np.nan], close[:-1]])
            tr = np.maximum(high - low,
                   np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
            tr_s = pd.Series(tr, index=idx)
            atr_sum_14 = tr_s.rolling(14, min_periods=7).sum()
            range_14_high = pd.Series(high, index=idx).rolling(14, min_periods=7).max()
            range_14_low = pd.Series(low, index=idx).rolling(14, min_periods=7).min()
            range_14 = range_14_high - range_14_low
            chop = 100 * np.log10(atr_sum_14 / (range_14 + 1e-9)) / np.log10(14)
            prices.loc[idx, "cand_choppiness_14d"] = chop.values

        # Fractal dimension proxy (simplified Higuchi-like)
        def _fractal_dim(x):
            x = x[~np.isnan(x)]
            n = len(x)
            if n < 20:
                return np.nan
            # Simple estimate: D ≈ 2 - H (where H is Hurst)
            # Use path length method
            L = np.sum(np.abs(np.diff(x)))
            net = abs(x[-1] - x[0])
            if L < 1e-10:
                return np.nan
            return 1 + np.log(L / (net + 1e-10)) / np.log(n)

        prices.loc[idx, "cand_fractal_dim_63d"] = (
            ret_s.rolling(63, min_periods=30)
                 .apply(_fractal_dim, raw=True).values
        )

        # Directional persistence: fraction of same-sign consecutive days
        sign_match = (np.sign(ret_d[1:]) == np.sign(ret_d[:-1])).astype(float)
        sign_match = np.concatenate([[np.nan], sign_match])
        sign_s = pd.Series(sign_match, index=idx)
        prices.loc[idx, "cand_directional_persist"] = (
            sign_s.rolling(63, min_periods=21).mean().values
        )

        # ADX proxy: directional movement / true range
        if "high" in grp.columns and "low" in grp.columns:
            high_s = pd.Series(high, index=idx)
            low_s = pd.Series(low, index=idx)
            plus_dm = (high_s.diff()).clip(lower=0)
            minus_dm = (-low_s.diff()).clip(lower=0)
            # Zero out when opposite DM is larger
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
            tr_s = pd.Series(tr, index=idx)
            plus_di = plus_dm.rolling(14, min_periods=7).mean() / (tr_s.rolling(14, min_periods=7).mean() + 1e-9)
            minus_di = minus_dm.rolling(14, min_periods=7).mean() / (tr_s.rolling(14, min_periods=7).mean() + 1e-9)
            dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
            adx = dx.rolling(14, min_periods=7).mean()
            prices.loc[idx, "cand_adx_proxy_14d"] = adx.values

        # Trend break count: number of MA10/MA50 crossovers in 63d
        ma10 = cls_s.rolling(10, min_periods=5).mean()
        ma50 = cls_s.rolling(50, min_periods=25).mean()
        cross_signal = np.sign(ma10 - ma50)
        cross_changes = (cross_signal.diff().abs() > 0).astype(float)
        prices.loc[idx, "cand_trend_break_63d"] = (
            cross_changes.rolling(63, min_periods=21).sum().values
        )

        # Trend acceleration: change in 21d trend slope over 21d
        # (requires trend_slope_21d already computed)
        if "trend_slope_21d" in grp.columns:
            ts = grp["trend_slope_21d"].values
            prices.loc[idx, "cand_trend_acceleration"] = (
                pd.Series(ts, index=idx).diff(21).values
            )

    return prices


# ══════════════���════════════════════════════════════��═════════════════���═════════
# FAMILY 3: DRAWDOWN / RECOVERY DYNAMICS
# Goes beyond maxdd to capture temporal structure of drawdowns.
# ═══════════════��═══════════════════════════════════════════════════════════════

def add_drawdown_dynamics(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Drawdown and recovery dynamics features.

    NEW COLUMNS:
      cand_time_since_high_252d  - Trading days since 252d rolling high (0 = at high)
      cand_drawdown_duration     - Consecutive days in current drawdown
      cand_recovery_speed_63d    - Average daily return during recovery from drawdown
      cand_time_underwater_126d  - Fraction of last 126d spent below prior peak
      cand_drawdown_depth_speed  - maxdd_21d / max(1, days_in_drawdown) - how fast did it drop?
      cand_bounce_from_low_21d   - Return from 21d low to current price
      cand_recovery_ratio_63d    - (current - 63d_low) / (63d_high - 63d_low) - where in range?
      cand_pain_index_63d        - Mean of drawdowns over 63d (not just max)
    """
    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        close = grp["close"].values
        n = len(close)
        cls_s = pd.Series(close, index=idx)

        # Time since 252d high
        rolling_max = cls_s.rolling(252, min_periods=126).max()
        at_high = (cls_s >= rolling_max * 0.995)  # within 0.5% of high
        days_since = np.zeros(n)
        count = 0
        for i in range(n):
            if at_high.iloc[i]:
                count = 0
            else:
                count += 1
            days_since[i] = count
        prices.loc[idx, "cand_time_since_high_252d"] = days_since

        # Drawdown duration: consecutive days below running max
        running_max = np.maximum.accumulate(close)
        in_dd = close < running_max * 0.995
        dd_duration = np.zeros(n)
        count = 0
        for i in range(n):
            if in_dd[i]:
                count += 1
            else:
                count = 0
            dd_duration[i] = count
        prices.loc[idx, "cand_drawdown_duration"] = dd_duration

        # Recovery speed: mean daily return on days where we're recovering from DD
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(n, np.nan)
        ret_s = pd.Series(ret_d, index=idx)
        # Recovery = in drawdown but positive return
        recovery_ret = ret_s.where((dd_duration > 0) & (ret_s > 0), np.nan)
        prices.loc[idx, "cand_recovery_speed_63d"] = (
            recovery_ret.rolling(63, min_periods=10).mean().values
        )

        # Time underwater: fraction of last 126d in drawdown
        in_dd_s = pd.Series(in_dd.astype(float), index=idx)
        prices.loc[idx, "cand_time_underwater_126d"] = (
            in_dd_s.rolling(126, min_periods=63).mean().values
        )

        # Drawdown depth/speed ratio
        dd_pct = close / running_max - 1
        dd_pct_s = pd.Series(dd_pct, index=idx)
        min_dd = dd_pct_s.rolling(21, min_periods=10).min()
        dd_dur_s = pd.Series(dd_duration, index=idx)
        max_dur = dd_dur_s.rolling(21, min_periods=10).max().clip(lower=1)
        prices.loc[idx, "cand_drawdown_depth_speed"] = (min_dd / max_dur).values

        # Bounce from 21d low
        low_21 = cls_s.rolling(21, min_periods=10).min()
        prices.loc[idx, "cand_bounce_from_low_21d"] = ((cls_s - low_21) / (low_21 + 1e-9)).values

        # Recovery ratio: position within 63d range
        high_63 = cls_s.rolling(63, min_periods=21).max()
        low_63 = cls_s.rolling(63, min_periods=21).min()
        range_63 = high_63 - low_63
        prices.loc[idx, "cand_recovery_ratio_63d"] = (
            (cls_s - low_63) / (range_63 + 1e-9)
        ).values

        # Pain index: mean drawdown over 63d
        prices.loc[idx, "cand_pain_index_63d"] = (
            dd_pct_s.rolling(63, min_periods=21).mean().values
        )

    return prices


# ═════��═════════════════════════════════════════════��═══════════════════════════
# FAMILY 4: VOLATILITY SHAPE & HIGHER MOMENTS
# Beyond simple vol levels - captures vol dynamics and distribution shape.
# ═════════════════════════════════════════════════════════���═════════════════════

def add_vol_shape(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility shape and higher moment features.

    NEW COLUMNS:
      cand_realized_skew_21d     - Rolling skewness of returns, 21d
      cand_realized_skew_63d     - Rolling skewness, 63d (partially overlaps skew_60d)
      cand_kurt_change_63d       - Change in kurtosis over 63d (fat-tail momentum)
      cand_vol_term_slope        - vol_5d / vol_126d (wider term structure than existing)
      cand_jump_intensity_63d    - Count of |ret| > 3σ days in past 63d
      cand_vol_asymmetry_63d     - (upvol - downvol) / (upvol + downvol) - vol skew
      cand_realized_var_ratio    - realized variance / (realized vol)² - should be 1 for normal
                                   dist, deviates for fat tails / jumps
    """
    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        ret_s = pd.Series(ret_d, index=idx)

        # Realized skewness at multiple horizons
        prices.loc[idx, "cand_realized_skew_21d"] = (
            ret_s.rolling(21, min_periods=10).skew().values
        )
        prices.loc[idx, "cand_realized_skew_63d"] = (
            ret_s.rolling(63, min_periods=21).skew().values
        )

        # Kurtosis change (momentum of tail risk)
        kurt = ret_s.rolling(63, min_periods=21).kurt()
        prices.loc[idx, "cand_kurt_change_63d"] = kurt.diff(21).values

        # Vol term structure slope: short vol / long vol
        vol_5 = ret_s.rolling(5, min_periods=3).std()
        vol_126 = ret_s.rolling(126, min_periods=63).std()
        prices.loc[idx, "cand_vol_term_slope"] = (vol_5 / (vol_126 + 1e-9)).values

        # Jump intensity: count of extreme moves
        rolling_std = ret_s.rolling(63, min_periods=21).std()
        is_jump = (ret_s.abs() > 3 * rolling_std).astype(float)
        prices.loc[idx, "cand_jump_intensity_63d"] = (
            is_jump.rolling(63, min_periods=21).sum().values
        )

        # Vol asymmetry: upside vol vs downside vol
        up_vol = ret_s.clip(lower=0).rolling(63, min_periods=21).std()
        dn_vol = ret_s.clip(upper=0).rolling(63, min_periods=21).std()
        prices.loc[idx, "cand_vol_asymmetry_63d"] = (
            (up_vol - dn_vol) / (up_vol + dn_vol + 1e-9)
        ).values

        # Realized variance ratio
        realized_var = (ret_s ** 2).rolling(63, min_periods=21).mean()
        realized_vol_sq = ret_s.rolling(63, min_periods=21).std() ** 2
        prices.loc[idx, "cand_realized_var_ratio"] = (
            realized_var / (realized_vol_sq + 1e-9)
        ).values

    return prices


# ═══════════════════════════════════════════════���═══════════════════════════════
# FAMILY 5: VOLUME-PRICE INTERACTIONS
# How volume relates to price movement - beyond simple volume metrics.
# ══════════��═══════════��════════════════════════════════════════════════════════

def add_volume_price(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-price interaction features.

    NEW COLUMNS:
      cand_price_vol_corr_21d    - Pearson corr(return, volume) over 21d
      cand_price_vol_corr_63d    - same over 63d
      cand_vwap_deviation_21d    - close / 21d VWAP - 1 (above/below volume-weighted avg)
      cand_volume_at_highs       - avg volume on up days / avg volume on down days (63d)
      cand_volume_return_mom_21d - 21d return weighted by volume rank
      cand_accumulation_21d      - (close - low) / (high - low) × volume, cumulated 21d
      cand_smart_money_flow_21d  - volume × sign(close - (high+low)/2), cumulated 21d
      cand_vol_price_divergence  - sign(21d ret) != sign(21d vol change) → divergence
    """
    if "volume" not in prices.columns:
        return prices

    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        vol = grp["volume"].values
        close = grp["close"].values
        ret_s = pd.Series(ret_d, index=idx)
        vol_s = pd.Series(vol, index=idx)
        cls_s = pd.Series(close, index=idx)

        # Price-volume correlation
        for w, suffix in [(21, "21d"), (63, "63d")]:
            prices.loc[idx, f"cand_price_vol_corr_{suffix}"] = (
                ret_s.rolling(w, min_periods=max(10, w // 3)).corr(vol_s).values
            )

        # VWAP deviation
        dv = cls_s * vol_s
        vwap_21 = dv.rolling(21, min_periods=10).sum() / (vol_s.rolling(21, min_periods=10).sum() + 1e-9)
        prices.loc[idx, "cand_vwap_deviation_21d"] = ((cls_s - vwap_21) / (vwap_21 + 1e-9)).values

        # Volume on up days vs down days
        up_mask = ret_s > 0
        vol_up = vol_s.where(up_mask, np.nan).rolling(63, min_periods=15).mean()
        vol_dn = vol_s.where(~up_mask, np.nan).rolling(63, min_periods=15).mean()
        prices.loc[idx, "cand_volume_at_highs"] = (vol_up / (vol_dn + 1e-9)).values

        # Volume-weighted return momentum
        vol_rank = vol_s.rolling(21, min_periods=10).rank(pct=True)
        prices.loc[idx, "cand_volume_return_mom_21d"] = (
            (ret_s * vol_rank).rolling(21, min_periods=10).sum().values
        )

        # Accumulation/distribution
        if "high" in grp.columns and "low" in grp.columns:
            high = grp["high"].values
            low = grp["low"].values
            hl_range = high - low
            clv = np.where(hl_range > 1e-9,
                          (close - low) / hl_range * 2 - 1,  # Chaikin CLV
                          0.0)
            ad_flow = pd.Series(clv * vol, index=idx)
            prices.loc[idx, "cand_accumulation_21d"] = (
                ad_flow.rolling(21, min_periods=10).sum().values
            )

            # Smart money flow: Chaikin Money Flow variant
            mid = (high + low) / 2
            smf = pd.Series(np.sign(close - mid) * vol, index=idx)
            prices.loc[idx, "cand_smart_money_flow_21d"] = (
                smf.rolling(21, min_periods=10).sum().values
            )

        # Volume-price divergence
        ret_21 = cls_s.pct_change(21)
        vol_chg_21 = vol_s.rolling(21, min_periods=10).mean().pct_change(21)
        divergence = (np.sign(ret_21) != np.sign(vol_chg_21)).astype(float)
        prices.loc[idx, "cand_vol_price_divergence"] = divergence.values

    return prices


# ══════���════════════════════════════════════════════════════════════��═══════════
# FAMILY 6: GAP / OVERNIGHT FEATURES
# Decompose returns into overnight (gap) and intraday components.
# ════════════════════════════════════════════════════════���══════════════════════

def add_gap_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Gap and overnight return features.

    NEW COLUMNS:
      cand_cum_gap_ret_21d       - Cumulative overnight gap return over 21d
      cand_cum_intraday_ret_21d  - Cumulative intraday return over 21d
      cand_gap_reversal_21d      - Correlation between gap and subsequent intraday move
                                   (negative = gaps tend to fill)
      cand_gap_persistence_21d   - Autocorrelation of gap direction over 21d
      cand_overnight_vs_intra    - overnight_ret_21d / (intraday_ret_21d + overnight_ret_21d)
                                   - which component drives total return?
      cand_large_gap_count_63d   - Count of gaps > 2% in past 63d
    """
    if "open" not in prices.columns:
        return prices

    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        close = grp["close"].values
        open_p = grp["open"].values
        n = len(close)

        # Overnight return: open[t] / close[t-1] - 1
        prev_close = np.concatenate([[np.nan], close[:-1]])
        overnight_ret = np.where(prev_close > 0, open_p / prev_close - 1, np.nan)
        on_s = pd.Series(overnight_ret, index=idx)

        # Intraday return: close[t] / open[t] - 1
        intraday_ret = np.where(open_p > 0, close / open_p - 1, np.nan)
        id_s = pd.Series(intraday_ret, index=idx)

        # Cumulative components
        prices.loc[idx, "cand_cum_gap_ret_21d"] = on_s.rolling(21, min_periods=10).sum().values
        prices.loc[idx, "cand_cum_intraday_ret_21d"] = id_s.rolling(21, min_periods=10).sum().values

        # Gap reversal tendency
        prices.loc[idx, "cand_gap_reversal_21d"] = (
            on_s.rolling(21, min_periods=10).corr(id_s).values
        )

        # Gap persistence
        gap_sign = np.sign(overnight_ret)
        gap_sign_s = pd.Series(gap_sign, index=idx)
        def _autocorr(x):
            x = x[~np.isnan(x)]
            if len(x) < 5:
                return np.nan
            return pd.Series(x).autocorr(1)
        prices.loc[idx, "cand_gap_persistence_21d"] = (
            gap_sign_s.rolling(21, min_periods=10).apply(_autocorr, raw=True).values
        )

        # Overnight vs intraday ratio
        on_sum = on_s.rolling(21, min_periods=10).sum()
        total_sum = on_sum + id_s.rolling(21, min_periods=10).sum()
        prices.loc[idx, "cand_overnight_vs_intra"] = (
            on_sum / (total_sum.abs() + 1e-9)
        ).values

        # Large gap count
        large_gap = (on_s.abs() > 0.02).astype(float)
        prices.loc[idx, "cand_large_gap_count_63d"] = (
            large_gap.rolling(63, min_periods=21).sum().values
        )

    return prices


# ═════════════════════════════════���════════════════════════════���════════════════
# FAMILY 7: ALTERNATIVE MOMENTUM CONSTRUCTIONS
# Momentum variants not already in the existing library.
# ═════════════════════════════════════════════════���═════════════════════════════

def add_alt_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative momentum constructions.

    NEW COLUMNS:
      cand_frog_in_pan_6m        - Da, Gurun, Warachka (2014): continuous small moves
                                   (many same-sign days) vs discrete jumps. FIP = sign(ret_6m) ×
                                   fraction of same-sign days. High FIP = underreaction.
      cand_mom_breadth_63d       - Fraction of positive daily returns in past 63d
      cand_mom_acceleration_63d  - ret_21d - ret_42d (short-term momentum picking up?)
      cand_expected_shortfall_mom - Mean of worst 10% daily returns over 126d as signal
      cand_idio_mom_6m           - 6m return after removing top-5-PC exposure
                                   (SIMPLIFIED: residual_mom already captures single-factor;
                                    this uses vol-adjusted version)
      cand_neg_mom_6m            - -ret_6m (intermediate reversal signal, 1-6 month)
      cand_mom_vol_interaction   - ret_12m × vol_contraction_signal (momentum + low-vol timing)
      cand_momentum_gap          - ret_12m - ret_1m (Jegadeesh 1990 skip-month purity)
    """
    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        close = grp["close"].values
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        ret_s = pd.Series(ret_d, index=idx)
        cls_s = pd.Series(close, index=idx)

        # Frog-in-the-pan
        ret_126 = cls_s.pct_change(126)
        positive_frac = ret_s.rolling(126, min_periods=63).apply(
            lambda x: (x > 0).mean(), raw=True
        )
        prices.loc[idx, "cand_frog_in_pan_6m"] = (
            np.sign(ret_126) * positive_frac
        ).values

        # Momentum breadth
        prices.loc[idx, "cand_mom_breadth_63d"] = (
            (ret_s > 0).astype(float).rolling(63, min_periods=21).mean().values
        )

        # Momentum acceleration
        ret_21 = cls_s.pct_change(21)
        ret_42 = cls_s.pct_change(42)
        prices.loc[idx, "cand_mom_acceleration_63d"] = (ret_21 - ret_42 / 2).values

        # Expected shortfall momentum (mean of worst 10% daily returns)
        def _es_10(x):
            x = x[~np.isnan(x)]
            if len(x) < 20:
                return np.nan
            cutoff = np.percentile(x, 10)
            tail = x[x <= cutoff]
            return tail.mean() if len(tail) > 0 else np.nan

        prices.loc[idx, "cand_expected_shortfall_mom"] = (
            ret_s.rolling(126, min_periods=63).apply(_es_10, raw=True).values
        )

        # Negative 6-month momentum (intermediate reversal)
        prices.loc[idx, "cand_neg_mom_6m"] = (-cls_s.pct_change(126)).values

        # Momentum × vol contraction interaction
        ret_252 = cls_s.pct_change(252)
        vol_21 = ret_s.rolling(21, min_periods=10).std()
        vol_63 = ret_s.rolling(63, min_periods=21).std()
        vol_contract = (vol_21 < vol_63 * 0.85).astype(float)
        prices.loc[idx, "cand_mom_vol_interaction"] = (ret_252 * vol_contract).values

        # Momentum gap (skip-month: 12m - 1m)
        ret_21_val = cls_s.pct_change(21)
        prices.loc[idx, "cand_momentum_gap"] = (ret_252 - ret_21_val).values

    return prices


# ═════════════��════════════════════════���═════════════════════════��══════════════
# FAMILY 8: NONLINEAR INTERACTIONS
# Combinations of existing primitives that create nonlinear signals.
# ═══��════════════��══════════════════════════════════════��═══════════════════════

def add_nonlinear_interactions(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Nonlinear interaction features combining existing primitives.

    NEW COLUMNS:
      cand_mom_x_vol_regime      - ret_12m × (1 if vol_ratio_st > 1.5 else 0)
                                   (momentum only when vol is spiking)
      cand_reversal_x_vol_spike  - ret_1m × (1 if vol_momentum_21d > 1 else 0)
                                   (reversal stronger with volume spike)
      cand_trend_eff_x_mom_sign  - path_efficiency × sign(ret_6m)
                                   (efficient trends with positive direction)
      cand_beta_change_x_vol     - diff(beta_63d, 21) × vol_21d
                                   (beta increasing while vol rising = risk)
      cand_hurst_x_mom           - hurst_63d × ret_12m
                                   (trending stock with positive momentum)
      cand_gap_x_volume          - avg_gap_21d × vol_momentum_21d
                                   (big gaps with volume confirmation)
      cand_skew_x_mom            - skew_60d × ret_6m
                                   (negatively skewed + positive momentum = risk premium)
      cand_drawdown_x_recovery   - maxdd_21d × recovery_speed_63d
                                   (deep drawdown + fast recovery)
      cand_vol_of_vol_x_mom      - vol_of_vol_63d × abs(ret_12m)
                                   (unstable vol + strong momentum = lottery)
      cand_corr_change_x_ret     - diff(corr_spx_63d, 21) × ret_1m
                                   (correlation regime change with recent return)
    """
    prices = prices.copy()

    # These interact with features that may already exist or were just computed
    # Safe: check column existence and fill NaN if missing

    def _safe_col(df, col):
        return df[col].values if col in df.columns else np.zeros(len(df))

    ret_d = prices["ret_d"] if "ret_d" in prices.columns else pd.Series(0, index=prices.index)
    ret_s = ret_d

    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        close = grp["close"].values
        cls_s = pd.Series(close, index=idx)
        ret_d_tk = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        ret_s_tk = pd.Series(ret_d_tk, index=idx)

        ret_252 = cls_s.pct_change(252).values
        ret_126 = cls_s.pct_change(126).values
        ret_21 = cls_s.pct_change(21).values

        # mom × vol_regime
        vol_ratio = _safe_col(grp, "vol_ratio_st")
        prices.loc[idx, "cand_mom_x_vol_regime"] = ret_252 * (vol_ratio > 1.5).astype(float)

        # reversal × vol_spike
        vol_mom = _safe_col(grp, "vol_momentum_21d")
        prices.loc[idx, "cand_reversal_x_vol_spike"] = -ret_21 * (vol_mom > 1.0).astype(float)

        # trend_eff × mom_sign
        path_eff = _safe_col(grp, "cand_path_efficiency_63d")
        prices.loc[idx, "cand_trend_eff_x_mom_sign"] = path_eff * np.sign(ret_126)

        # beta_change × vol
        beta_63 = _safe_col(grp, "beta_63d")
        vol_21 = _safe_col(grp, "vol_21d")
        beta_chg = pd.Series(beta_63, index=idx).diff(21).values
        prices.loc[idx, "cand_beta_change_x_vol"] = beta_chg * vol_21

        # hurst × momentum
        hurst = _safe_col(grp, "hurst_63d")
        prices.loc[idx, "cand_hurst_x_mom"] = hurst * ret_252

        # gap × volume
        avg_gap = _safe_col(grp, "avg_gap_21d")
        vol_mom2 = _safe_col(grp, "vol_momentum_21d")
        prices.loc[idx, "cand_gap_x_volume"] = avg_gap * vol_mom2

        # skew × momentum
        skew = _safe_col(grp, "skew_60d")
        prices.loc[idx, "cand_skew_x_mom"] = skew * ret_126

        # drawdown × recovery
        maxdd = _safe_col(grp, "maxdd_21d")
        rec_speed = _safe_col(grp, "cand_recovery_speed_63d")
        prices.loc[idx, "cand_drawdown_x_recovery"] = maxdd * rec_speed

        # vol_of_vol × |momentum|
        vov = _safe_col(grp, "vol_of_vol_63d")
        prices.loc[idx, "cand_vol_of_vol_x_mom"] = vov * np.abs(ret_252)

        # corr_change × return
        corr_spx = _safe_col(grp, "corr_spx_63d")
        corr_chg = pd.Series(corr_spx, index=idx).diff(21).values
        prices.loc[idx, "cand_corr_change_x_ret"] = corr_chg * ret_21

    return prices


# ═══════════════════════════════════════════════════════════════════════���═══════
# FAMILY 9: MICROSTRUCTURE PROXIES
# Bid-ask spread and price impact approximations from OHLCV.
# ══════════════════════��════════════════════════════════��═══════════════════════

def add_microstructure_proxies(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Microstructure proxies estimated from OHLCV data.

    NEW COLUMNS:
      cand_corwin_schultz_21d    - Corwin-Schultz (2012) bid-ask spread estimator
                                   from daily high-low prices. Higher = less liquid.
      cand_roll_spread_21d       - Roll (1984) implied spread from autocov of returns.
                                   sqrt(-2 × cov(ret[t], ret[t-1])).
      cand_kyle_lambda_21d       - Kyle's lambda proxy: abs(ret) / sqrt(volume)
                                   averaged over 21d. Price impact per unit of volume.
      cand_price_impact_asym_21d - Asymmetric price impact: lambda on down days /
                                   lambda on up days. > 1 = more impact selling.
    """
    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        ret_s = pd.Series(ret_d, index=idx)

        if "high" in grp.columns and "low" in grp.columns:
            high = grp["high"].values
            low = grp["low"].values
            close = grp["close"].values

            # Corwin-Schultz spread estimator
            # β = E[ln(H/L)²], γ = [ln(H2/L2)]² where H2,L2 are 2-day high,low
            ln_hl = np.log(high / (low + 1e-9))
            ln_hl_sq = ln_hl ** 2

            # 2-day high and low
            high_2d = np.maximum(high[1:], high[:-1])
            low_2d = np.minimum(low[1:], low[:-1])
            ln_hl2 = np.log(high_2d / (low_2d + 1e-9))
            ln_hl2_sq = np.concatenate([[np.nan], ln_hl2 ** 2])

            beta = pd.Series(ln_hl_sq, index=idx).rolling(21, min_periods=10).mean()
            gamma = pd.Series(ln_hl2_sq, index=idx).rolling(21, min_periods=10).mean()

            # CS formula: S = 2(e^α - 1)/(1 + e^α), where α = f(β, γ)
            k = np.sqrt(2) - 1
            alpha_sq = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
            alpha_sq = alpha_sq.clip(lower=0)
            alpha = np.sqrt(alpha_sq)
            cs_spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            prices.loc[idx, "cand_corwin_schultz_21d"] = cs_spread.values

        # Roll spread: sqrt(-2 × autocovariance)
        autocov = ret_s.rolling(21, min_periods=10).apply(
            lambda x: pd.Series(x).autocorr(1) * np.var(x) if len(x) > 5 else np.nan,
            raw=True
        )
        # Only defined when autocov is negative (as per Roll model)
        roll_sq = -2 * autocov
        roll_spread = np.where(roll_sq > 0, np.sqrt(roll_sq), 0)
        prices.loc[idx, "cand_roll_spread_21d"] = roll_spread

        # Kyle's lambda: |return| / sqrt(volume)
        if "volume" in grp.columns:
            vol = grp["volume"].values
            sqrt_vol = np.sqrt(np.maximum(vol, 1))
            lambda_d = np.abs(ret_d) / sqrt_vol
            lambda_s = pd.Series(lambda_d, index=idx)
            prices.loc[idx, "cand_kyle_lambda_21d"] = (
                lambda_s.rolling(21, min_periods=10).mean().values
            )

            # Asymmetric price impact
            lambda_down = lambda_s.where(ret_s < 0, np.nan).rolling(21, min_periods=5).mean()
            lambda_up = lambda_s.where(ret_s > 0, np.nan).rolling(21, min_periods=5).mean()
            prices.loc[idx, "cand_price_impact_asym_21d"] = (
                lambda_down / (lambda_up + 1e-9)
            ).values

    return prices


# ═════════════���═════════════════════════════════��═══════════════════════════════
# MASTER FUNCTION: BUILD ALL CANDIDATES
# ════════════════════════════��═══════════════════════════════���══════════════════

ALL_CANDIDATE_FAMILIES = [
    ("path_dependent", add_path_dependent),
    ("trend_efficiency", add_trend_efficiency),
    ("drawdown_dynamics", add_drawdown_dynamics),
    ("vol_shape", add_vol_shape),
    ("volume_price", add_volume_price),
    ("gap_features", add_gap_features),
    ("alt_momentum", add_alt_momentum),
    ("nonlinear_interactions", add_nonlinear_interactions),
    ("microstructure_proxies", add_microstructure_proxies),
]


def build_all_candidates(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all candidate factor families to the prices DataFrame.

    Returns prices with all cand_* columns appended.
    """
    for name, func in ALL_CANDIDATE_FAMILIES:
        print(f"  Computing {name}...")
        prices = func(prices)

    # Report
    cand_cols = [c for c in prices.columns if c.startswith("cand_")]
    print(f"\n  Total candidate features generated: {len(cand_cols)}")
    return prices


def get_candidate_columns(prices: pd.DataFrame) -> List[str]:
    """Return list of all candidate column names."""
    return sorted([c for c in prices.columns if c.startswith("cand_")])


def get_candidate_metadata() -> pd.DataFrame:
    """Return metadata about all candidates for the catalog."""
    candidates = [
        # Family 1: Path-Dependent
        ("cand_path_efficiency_21d", "path_dependent", "|net ret| / sum(|daily ret|) 21d - path straightness"),
        ("cand_path_efficiency_63d", "path_dependent", "|net ret| / sum(|daily ret|) 63d"),
        ("cand_up_path_ratio_21d", "path_dependent", "Fraction of total path that is upward 21d"),
        ("cand_max_consec_up_63d", "path_dependent", "Max consecutive up-days in 63d"),
        ("cand_max_consec_down_63d", "path_dependent", "Max consecutive down-days in 63d"),
        ("cand_signed_path_21d", "path_dependent", "Path length × sign(net return) 21d"),
        ("cand_return_dispersion_21d", "path_dependent", "CV of daily returns 21d"),
        ("cand_path_skew_21d", "path_dependent", "Skewness of daily returns 21d"),
        # Family 2: Trend Efficiency
        ("cand_kaufman_er_21d", "trend_efficiency", "Kaufman Efficiency Ratio 21d"),
        ("cand_kaufman_er_63d", "trend_efficiency", "Kaufman Efficiency Ratio 63d"),
        ("cand_choppiness_14d", "trend_efficiency", "Choppiness Index 14d (high=choppy)"),
        ("cand_fractal_dim_63d", "trend_efficiency", "Fractal dimension proxy 63d"),
        ("cand_directional_persist", "trend_efficiency", "Fraction of same-sign consecutive days 63d"),
        ("cand_adx_proxy_14d", "trend_efficiency", "ADX proxy from directional movement 14d"),
        ("cand_trend_break_63d", "trend_efficiency", "MA crossover count in 63d (low=persistent)"),
        ("cand_trend_acceleration", "trend_efficiency", "Change in trend_slope_21d over 21d"),
        # Family 3: Drawdown Dynamics
        ("cand_time_since_high_252d", "drawdown_dynamics", "Days since 252d rolling high"),
        ("cand_drawdown_duration", "drawdown_dynamics", "Consecutive days in current drawdown"),
        ("cand_recovery_speed_63d", "drawdown_dynamics", "Mean daily ret during recovery 63d"),
        ("cand_time_underwater_126d", "drawdown_dynamics", "Fraction of 126d in drawdown"),
        ("cand_drawdown_depth_speed", "drawdown_dynamics", "Drawdown depth / duration ratio"),
        ("cand_bounce_from_low_21d", "drawdown_dynamics", "Return from 21d low to current"),
        ("cand_recovery_ratio_63d", "drawdown_dynamics", "Position within 63d range [0,1]"),
        ("cand_pain_index_63d", "drawdown_dynamics", "Mean drawdown over 63d"),
        # Family 4: Vol Shape
        ("cand_realized_skew_21d", "vol_shape", "Rolling skewness 21d"),
        ("cand_realized_skew_63d", "vol_shape", "Rolling skewness 63d (near skew_60d - check dedup)"),
        ("cand_kurt_change_63d", "vol_shape", "Change in kurtosis over 63d"),
        ("cand_vol_term_slope", "vol_shape", "vol_5d / vol_126d (wider term structure)"),
        ("cand_jump_intensity_63d", "vol_shape", "Count of |ret| > 3σ days in 63d"),
        ("cand_vol_asymmetry_63d", "vol_shape", "(upvol - downvol) / (upvol + downvol) 63d"),
        ("cand_realized_var_ratio", "vol_shape", "Realized variance / vol² (jump indicator)"),
        # Family 5: Volume-Price
        ("cand_price_vol_corr_21d", "volume_price", "Corr(return, volume) 21d"),
        ("cand_price_vol_corr_63d", "volume_price", "Corr(return, volume) 63d"),
        ("cand_vwap_deviation_21d", "volume_price", "Close / 21d VWAP - 1"),
        ("cand_volume_at_highs", "volume_price", "Avg vol on up days / down days 63d"),
        ("cand_volume_return_mom_21d", "volume_price", "Volume-weighted return momentum 21d"),
        ("cand_accumulation_21d", "volume_price", "Chaikin accumulation 21d"),
        ("cand_smart_money_flow_21d", "volume_price", "Chaikin money flow variant 21d"),
        ("cand_vol_price_divergence", "volume_price", "Price-volume direction disagreement"),
        # Family 6: Gap Features
        ("cand_cum_gap_ret_21d", "gap_features", "Cumulative overnight gap return 21d"),
        ("cand_cum_intraday_ret_21d", "gap_features", "Cumulative intraday return 21d"),
        ("cand_gap_reversal_21d", "gap_features", "Corr(gap, intraday) 21d - negative=fills"),
        ("cand_gap_persistence_21d", "gap_features", "Autocorr of gap direction 21d"),
        ("cand_overnight_vs_intra", "gap_features", "Overnight share of total return 21d"),
        ("cand_large_gap_count_63d", "gap_features", "Count of >2% gaps in 63d"),
        # Family 7: Alt Momentum
        ("cand_frog_in_pan_6m", "alt_momentum", "Da et al 2014: sign(ret) × frac positive days"),
        ("cand_mom_breadth_63d", "alt_momentum", "Fraction of positive daily returns 63d"),
        ("cand_mom_acceleration_63d", "alt_momentum", "ret_21d - ret_42d/2 (acceleration)"),
        ("cand_expected_shortfall_mom", "alt_momentum", "Mean of worst 10% daily rets 126d"),
        ("cand_neg_mom_6m", "alt_momentum", "-ret_6m (intermediate reversal)"),
        ("cand_mom_vol_interaction", "alt_momentum", "ret_12m × vol_contraction_signal"),
        ("cand_momentum_gap", "alt_momentum", "ret_12m - ret_1m (skip-month)"),
        # Family 8: Nonlinear Interactions
        ("cand_mom_x_vol_regime", "nonlinear", "ret_12m × (vol_ratio > 1.5)"),
        ("cand_reversal_x_vol_spike", "nonlinear", "-ret_1m × (vol_mom > 1)"),
        ("cand_trend_eff_x_mom_sign", "nonlinear", "path_eff × sign(ret_6m)"),
        ("cand_beta_change_x_vol", "nonlinear", "Δbeta_63d × vol_21d"),
        ("cand_hurst_x_mom", "nonlinear", "hurst_63d × ret_12m"),
        ("cand_gap_x_volume", "nonlinear", "avg_gap × vol_momentum"),
        ("cand_skew_x_mom", "nonlinear", "skew_60d × ret_6m"),
        ("cand_drawdown_x_recovery", "nonlinear", "maxdd_21d × recovery_speed"),
        ("cand_vol_of_vol_x_mom", "nonlinear", "vol_of_vol × |ret_12m|"),
        ("cand_corr_change_x_ret", "nonlinear", "Δcorr_spx × ret_1m"),
        # Family 9: Microstructure
        ("cand_corwin_schultz_21d", "microstructure", "Corwin-Schultz bid-ask spread 21d"),
        ("cand_roll_spread_21d", "microstructure", "Roll implied spread 21d"),
        ("cand_kyle_lambda_21d", "microstructure", "Kyle lambda (price impact) 21d"),
        ("cand_price_impact_asym_21d", "microstructure", "Asymmetric price impact (down/up)"),
    ]

    return pd.DataFrame(candidates, columns=["factor_name", "family", "description"])
