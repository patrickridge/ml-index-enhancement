# Feature Audit — Existing 205-Feature Factor Library

**Date:** 2026-04-06
**Panel:** `data/panel_monthly_enriched.parquet` — 147,733 rows × 208 columns
**Date range:** 2005-01 to 2025-11 | **Tickers:** 697 historical S&P 500 constituents

---

## Summary

208 total columns in the panel:
- 2 identifiers: `date`, `ticker`
- 1 target: `fwd_ret_1m`
- ~205 features across 21 categories (Cats 1–21)

All factor functions are in `utils_factors.py` (1,600+ lines). Cross-sectional ranking to [-0.5, 0.5] happens in `1h_feature_engineering.py`. Macro columns (Cat 10, 21) are time-series z-scored instead.

---

## Feature Inventory by Category

### Cat 1 — Multi-Horizon Momentum (6 factors)
| Factor | Description | Window |
|--------|-------------|--------|
| `ret_1w` | 5-day return | 5d |
| `ret_2w` | 10-day return | 10d |
| `ret_9m` | 189-day return | 189d |
| `ret_18m` | 378-day return | 378d |
| `ret_24m` | 504-day return | 504d |
| `ret_36m` | 756-day return | 756d |

**Also in panel (from 1h):** `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`, `mom_2_12` (skip-month), `mom_accel`

**Gap:** No 1-3 month intermediate reversal factor explicitly separated from momentum.

### Cat 2 — Volatility Regimes (8 factors)
| Factor | Description |
|--------|-------------|
| `vol_5d`, `vol_21d`, `vol_126d` | Rolling realized vol |
| `vol_ratio_st` | vol_5d / vol_21d |
| `vol_trend` | vol_21d / vol_63d |
| `downvol_21d` | Downside deviation 21d |
| `beta_21d`, `beta_63d` | CAPM beta |

**Also in panel:** `vol_20d`, `vol_60d`, `vol_252d`

**Gap:** No vol term structure across 3+ horizons (e.g., vol_5d/vol_126d). No realized variance vs realized vol comparison.

### Cat 3 — Tail Risk (6 factors)
| Factor | Description |
|--------|-------------|
| `kurt_60d` | Rolling kurtosis |
| `maxdd_21d`, `maxdd_126d` | Rolling max drawdown |
| `var_95_21d` | 5th percentile VaR |
| `cvar_95_21d` | Conditional VaR |
| `omega_ratio_21d` | Upside/downside mass |

**Gap:** No drawdown duration, no time-since-max-drawdown, no recovery speed metrics.

### Cat 4 — Price Level / Trend (15 factors)
| Factor | Description |
|--------|-------------|
| `price_to_ma10/20/50/100/200` | Close relative to MAs |
| `ma_cross_10_50`, `ma_cross_50_200` | MA crossover ratios |
| `nearness_52w_low` | Close / 52w low |
| `rsi_14` | Wilder RSI |
| `macd_signal`, `macd_hist` | Normalized MACD + histogram |
| `bollinger_pct` | Bollinger Band %B |
| `trend_slope_21d/63d` | OLS slope normalized |
| `trend_r2_21d` | R² of 21d linear trend |

**Gap:** No ADX/DMI directional movement. No Choppiness Index. No trend efficiency ratio (net move / total path).

### Cat 5 — Volume & Liquidity (5 factors)
| Factor | Description |
|--------|-------------|
| `dollar_vol_21d/63d` | Rolling mean dollar volume |
| `vol_momentum_21d` | Volume surge (vol/avg - 1) |
| `vol_trend_ratio` | 21d avg vol / 63d avg vol |
| `obv_signal` | OBV 21d pct change |

**Gap:** No price-volume correlation. No volume-weighted return features. No Klinger oscillator.

### Cat 6 — Market Beta / Correlation (8 factors)
| Factor | Description |
|--------|-------------|
| `beta_252d` | 252d CAPM beta |
| `corr_spx_21d/63d` | Pearson correlation with SPX |
| `idio_vol_252d` | Idiosyncratic vol |
| `up_beta_63d`, `down_beta_63d` | Conditional (asymmetric) beta |

### Cat 7 — Microstructure (5 factors)
| Factor | Description |
|--------|-------------|
| `avg_hl_range_21d` | Average (H-L)/C |
| `avg_gap_21d` | Average |overnight gap| |
| `open_to_close_21d` | Average (C-O)/O |
| `atr_21d_norm` | ATR normalized by close |
| `intraday_vol_21d` | Std of (C-O)/O |

**Gap:** No cumulative gap returns. No gap reversal tendency. No overnight vs intraday return decomposition.

### Cat 8 — Cross-Sectional Relative (1 factor)
| Factor | Description |
|--------|-------------|
| `residual_ret_12m` | ret_12m - beta × spx_ret_12m |

### Cat 9 — Fundamental / Quality (12 factors, optional)
`pe_ratio`, `pb_ratio`, `ps_ratio`, `ev_ebitda`, `roe`, `roa`, `gross_margin`, `revenue_growth_yoy`, `eps_growth_yoy`, `debt_to_equity`, `earnings_quality`, `buyback_yield`

### Cat 10 — Macro / Regime (9 factors, time-series z-scored)
`vix_level`, `vix_change_21d`, `yield_10y`, `yield_spread_10y2y`, `yield_change_21d`, `dollar_index`, `credit_proxy_change`, `market_trend_spx`, `market_vol_regime`

### Cat 11 — Time Signal Factors (5 factors)
`ir_6m`, `ir_3m` (if computed), `trend_r2_126d`, `ret_consistency_12m`, `skew_60d`, `drawdown_pct_252d`

### Cat 12 — Barra-Style (4 factors)
`amihud_illiq_21d`, `vol_of_vol_63d`, `beta_stability_63d`, `size_proxy`

### Cat 13 — Interaction Factors (5 factors, monthly)
`residual_ret_1m`, `beta_x_idiovol`, `up_down_beta_spread`, `vol_excess`, `mom_decel`

### Cat 14 — Size (1 factor)
`log_mktcap`

### Cat 15 — Tail Rankings (~60 factors)
Binary top/bot/tail flags for ~20 base factors (3 cols each). Already in {0, 1, -1} space.

**Base factors with tails:** `ret_1w`, `ret_2w`, `ret_6m`, `ret_9m`, `ret_12m`, `ret_18m`, `ret_24m`, `ret_1m`, `vol_21d`, `vol_252d`, `maxdd_126d`, `beta_252d`, `idio_vol_252d`, `log_mktcap`, `size_proxy`, `vol_momentum_21d`, `amihud_illiq_21d`

### Cat 16 — Mined Alpha Signals (12 factors)
`nearness_52w_high`, `max_ret_21d`, `risk_adj_mom_6m/12m`, `residual_mom_6m/12m`, `up_down_vol_ratio`, `co_skewness_63d`, `vol_contraction_signal`, `mom_quality`, `price_range_ratio`, `reversal_size`

### Cat 17 — Time Signals v2 (17 factors, per-stock binary/directional)
- TSMOM: `tsmom_sign_12m/6m`, `tsmom_magnitude`
- MA regime: `above_ma_200/50`, `ma_200_slope`, `price_pct_above_ma200`
- 52w high: `new_52w_high/low`, `ret_since_52w_low`
- Volume: `vol_above_avg`, `volume_trend`, `high_vol_week`
- Earnings proxy: `earnings_gap`, `post_earnings_drift`
- Serial corr: `serial_corr_sign`, `hurst_63d`

### Cat 18 — Seasonality (5 factors)
`ret_same_month_1y/2y`, `ret_same_month_avg`, `turn_of_month`, `january_dummy`

### Cat 19 — Short Interest (4 factors, optional)
`short_pct_float`, `short_interest_ratio`, `short_change_2w`, `short_squeeze_risk`

### Cat 20 — Institutional Ownership (4 factors, optional)
`inst_own_pct`, `inst_own_change`, `num_institutions`, `inst_concentration`

### Cat 21 — Prediction Markets (6 factors, macro/time-series)
`fed_hike_prob`, `fed_cut_prob`, `recession_prob`, `policy_uncertainty`, `vix_term_structure`, `pred_market_sentiment`

---

## Known Redundancies

308 factor pairs with |corr| > 0.7 identified in prior diagnostics. Key clusters:

| Cluster | Highly Correlated Pairs |
|---------|------------------------|
| **Price-to-MA family** | price_to_ma10/20/50/100/200 — monotonic overlap at longer horizons |
| **Momentum family** | ret_6m ↔ ret_9m, ret_12m ↔ ret_18m, risk_adj_mom ↔ ret variants |
| **Volatility family** | vol_5d/21d/60d/126d/252d form a correlated chain |
| **Beta variants** | beta_21d ↔ beta_63d ↔ beta_252d |
| **Range/ATR** | avg_hl_range_21d ↔ atr_21d_norm (near-duplicate, corr ≈ 1.0) |
| **52w nearness** | nearness_52w_high ↔ drawdown_pct_252d (near-inverse) |

---

## Identified Gaps for Factor Mining

### 1. Path-Dependent Return Features (NOT COVERED)
- Signed path length: sum(|daily ret|) / net return — measures path efficiency
- Up-path vs down-path ratio
- Max consecutive up/down days
- Return dispersion within windows

### 2. Trend Efficiency / Choppiness (NOT COVERED)
- Efficiency ratio: |net move| / sum(|daily moves|) — Kaufman (1995)
- Choppiness Index: log(sum(ATR_n)) / log(High_n - Low_n)
- ADX / directional movement index
- Fractal dimension proxy

### 3. Drawdown Dynamics (PARTIALLY COVERED — only maxdd)
- Time since 52w high (NOT just nearness to it)
- Drawdown duration (how many days in current drawdown)
- Recovery speed from max drawdown
- Time underwater as fraction of lookback

### 4. Volatility Shape & Higher Moments (PARTIALLY COVERED)
- Realized skewness at multiple horizons (only skew_60d exists)
- Kurtosis change / momentum of kurtosis
- Vol term structure slope across 3+ horizons
- Jump intensity proxy (count of |ret| > 3σ days)

### 5. Volume-Price Interactions (NOT COVERED)
- Price-volume correlation (21d/63d)
- Volume-weighted return momentum
- Volume at new highs vs volume at new lows
- Accumulation/distribution line variants

### 6. Gap / Overnight Decomposition (PARTIALLY COVERED — only avg_gap_21d)
- Cumulative gap return over window
- Gap reversal tendency (do gaps fill?)
- Overnight return vs intraday return split
- Gap direction persistence

### 7. Nonlinear Interactions (NOT SYSTEMATICALLY COVERED)
- momentum × volatility_regime
- reversal × volume_spike
- trend_efficiency × momentum_sign
- beta_change × vol_change

### 8. Regime-Conditional Transforms (NOT COVERED)
- Entropy-based volatility regime detector (Singha paper concept)
- Vol transition detector (regime change signal)
- Beta regime shift detector
- Conditional factor performance by regime

### 9. Alternative Momentum Constructions (PARTIALLY COVERED)
- Frog-in-the-pan: continuous small moves vs discrete jumps (Da et al. 2014)
- Momentum breadth: % positive days in window (partially via ret_consistency_12m)
- Acceleration: change in momentum slope
- Expected shortfall momentum

### 10. Microstructure from OHLCV (PARTIALLY COVERED)
- Bid-ask spread proxies from OHLCV (Corwin-Schultz, Roll estimator)
- Kyle's lambda approximation
- Price impact asymmetry

---

## Duplicate-Risk Areas for New Candidates

Any new candidate in these families MUST be checked for correlation > 0.85 against:

| Family | Existing features to check against |
|--------|-----------------------------------|
| Momentum variants | ret_1w/2w/1m/3m/6m/9m/12m/18m/24m/36m, mom_2_12, risk_adj_mom, residual_mom |
| Volatility variants | vol_5d/20d/21d/60d/126d/252d, vol_ratio_st, vol_trend, downvol_21d |
| Trend signals | price_to_ma*, ma_cross_*, trend_slope_*, trend_r2_*, rsi_14, bollinger_pct |
| Beta/correlation | beta_21d/63d/252d, corr_spx_21d/63d, up/down_beta |
| Drawdown/52w | maxdd_21d/126d, nearness_52w_low/high, drawdown_pct_252d |
| Volume | dollar_vol_*, vol_momentum_21d, vol_trend_ratio, obv_signal |
