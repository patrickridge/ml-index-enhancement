# Factor Decay Diagnostic Report
**Generated:** 2026-04-04  
**IS training period:** 2010-01-01 → 2022-12-31  
**OOS test period:** 2022-12 → present (39 months)  

---

## 1. Executive Summary

- **70 factors are GREEN** (OOS IC stable, decay ratio ≥ 0.7) — safe to use
- **7 factors are AMBER** (some decay, ratio 0.3–0.7) — use with caution
- **74 factors are RED** (severe decay or sign flip OOS) — review or drop
- **69 factors flipped sign OOS**: `macd_signal, trend_slope_21d, ma_cross_10_50, price_to_ma50, ret_1m, omega_ratio_21d, open_to_close_21d, ret_1m_bot, ret_1m_tail, mom_decel, rsi_14, residual_ret_1m, price_to_ma20, maxdd_126d_top, up_down_vol_ratio, cum_vol_ratio, co_skewness_63d, price_to_ma100, ir_3m, skew_60d, cvar_95_21d, ret_2w, ret_2w_bot, ret_1m_top, bollinger_pct, avg_hl_range_21d, high_low_range_21d, beta_252d_top, hl_range, idio_vol_252d, log_mktcap_tail, ret_3m, atr_21d_norm, reversal_size, ret_2w_tail, obv_signal, trend_slope_63d, price_to_ma10, idio_vol_252d_bot, macd_hist, max_ret_21d, log_mktcap, avg_gap_21d, vol_252d_bot, kurt_60d, adx_regime, amihud_illiq_21d, vol_5d, adx_14, vol_21d_bot, vol_252d, vol_of_vol_63d, dollar_vol_63d, size_proxy, beta_252d_bot, size_proxy_tail, ret_1w_top, vol_expansion, ret_1w, vol_126d, ret_2w_top, ret_1w_tail, vol_momentum_21d_top, dollar_vol_21d, vol_trend_ratio, vol_contraction_signal, beta_x_idiovol, ret_1w_bot, price_range_ratio`

### Top 5 Factors by |OOS IC|

| factor           |   ic_is |   ic_oos |   icir_oos |   decay_ratio | health   |
|:-----------------|--------:|---------:|-----------:|--------------:|:---------|
| ret_36m          | +0.0096 |  +0.0609 |    +0.4680 |       +6.3200 | GREEN    |
| ret_18m          | +0.0013 |  +0.0606 |    +0.3760 |      +46.4300 | GREEN    |
| ir_12m           | +0.0113 |  +0.0552 |    +0.3920 |       +4.8800 | GREEN    |
| risk_adj_mom_12m | +0.0096 |  +0.0550 |    +0.3780 |       +5.7100 | GREEN    |
| residual_mom_12m | +0.0123 |  +0.0523 |    +0.3180 |       +4.2600 | GREEN    |

### Most Decayed / Flipped Factors

| factor               |   ic_is |   ic_oos |   decay_ratio | health   |
|:---------------------|--------:|---------:|--------------:|:---------|
| vol_trend            | -0.0019 |  -0.0000 |       +0.0100 | RED      |
| log_mktcap_top       | -0.0177 |  -0.0002 |       +0.0100 | RED      |
| beta_x_idiovol       | +0.0097 |  -0.0008 |       -0.0900 | RED      |
| amihud_illiq_21d_bot | -0.0301 |  -0.0031 |       +0.1000 | RED      |
| price_range_ratio    | +0.0051 |  -0.0006 |       -0.1100 | RED      |
| beta_252d_bot        | -0.0281 |  +0.0041 |       -0.1500 | RED      |
| ret_1w               | -0.0173 |  +0.0036 |       -0.2100 | RED      |
| ret_1w_bot           | +0.0038 |  -0.0008 |       -0.2200 | RED      |
| volume               | +0.0130 |  +0.0028 |       +0.2200 | RED      |
| vol_expansion        | -0.0142 |  +0.0037 |       -0.2600 | RED      |

---

## 2. Key Findings on Factor Decay

### Why factors decay OOS
1. **Regime shift**: A factor trained on the QE bull market (2010–2019) may not work
   in the rate-hike bear (2022) or AI mega-cap bull (2023–2024). Factors with
   IC sign flips across regimes are the most dangerous.
2. **Overfitting**: Factors computed from the same price data can have spurious
   IS IC that doesn't survive. IC < 0.02 in training is typically noise.
3. **Crowding**: Widely-used factors (momentum, RSI) get arbitraged away.
   OOS ICIR < 0.1 suggests the alpha is crowded out.
4. **Market structure change**: The 2023–2024 'Magnificent 7' concentration means
   cross-sectional models face structural headwinds — 7 stocks drive 60% of SPX
   returns, and no within-S&P cross-section factor captures this.

### Green factors — what's working OOS

```
ret_36m
ret_18m
ir_12m
risk_adj_mom_12m
residual_mom_12m
mom_quality
residual_ret_12m
ret_24m
ret_12m
ret_consistency_12m
ret_18m_tail
corr_spx_63d
ret_18m_top
mom_2_12
vol_excess
maxdd_21d
nearness_52w_low
nearness_52w_high
ret_12m_tail
drawdown_pct_252d
ret_24m_tail
mom_accel
ret_24m_top
price_to_ma200
ret_9m
up_beta_63d
ret_12m_top
maxdd_126d_tail
ir_6m
risk_adj_mom_6m
corr_spx_21d
residual_mom_6m
amihud_illiq_21d_top
down_beta_63d
beta_63d
maxdd_126d
ret_6m
maxdd_126d_bot
ret_18m_bot
trend_r2_21d
ret_9m_tail
ret_12m_bot
log_mktcap_bot
beta_21d
downvol_21d
beta_252d
ret_9m_top
ret_6m_tail
vol_momentum_21d_tail
var_95_21d
ret_24m_bot
ret_6m_bot
amihud_illiq_21d_tail
up_down_beta_spread
idio_vol_252d_tail
ma_cross_50_200
trend_strength
ret_9m_bot
vol_252d_tail
intraday_vol_21d
idio_vol_252d_top
vol_252d_top
beta_stability_63d
ret_6m_top
vol_21d_tail
vol_20d
vol_21d
vol_60d
autocorr_ret_21d
vol_ratio_st
```

These have consistent IC across both IS and OOS periods. Prioritise these in
the CS-Transformer feature set.

### Red/Amber factors — what needs investigation

```
macd_signal
trend_slope_21d
ma_cross_10_50
price_to_ma50
ret_1m
omega_ratio_21d
open_to_close_21d
ret_1m_bot
ret_1m_tail
mom_decel
rsi_14
residual_ret_1m
price_to_ma20
maxdd_126d_top
up_down_vol_ratio
cum_vol_ratio
co_skewness_63d
price_to_ma100
ir_3m
skew_60d
cvar_95_21d
ret_2w
ret_2w_bot
ret_1m_top
bollinger_pct
avg_hl_range_21d
high_low_range_21d
beta_252d_top
hl_range
idio_vol_252d
log_mktcap_tail
ret_3m
atr_21d_norm
reversal_size
ret_2w_tail
obv_signal
trend_slope_63d
price_to_ma10
idio_vol_252d_bot
macd_hist
size_proxy_bot
max_ret_21d
log_mktcap
avg_gap_21d
vol_252d_bot
kurt_60d
adx_regime
amihud_illiq_21d
vol_5d
adx_14
vol_21d_bot
vol_252d
vol_of_vol_63d
dollar_vol_63d
size_proxy
beta_252d_bot
size_proxy_tail
ret_1w_top
vol_expansion
ret_1w
vol_126d
ret_2w_top
ret_1w_tail
amihud_illiq_21d_bot
vol_momentum_21d_top
volume
dollar_vol_21d
vol_trend_ratio
vol_contraction_signal
beta_x_idiovol
ret_1w_bot
price_range_ratio
log_mktcap_top
vol_trend
vol_momentum_21d_bot
size_proxy_top
vol_21d_top
beta_252d_tail
vol_momentum_21d
roll_spread
trend_r2_126d
```

Consider: (a) dropping from the model, (b) applying regime-conditional weights,
or (c) orthogonalising against known regime exposures before use.

---

## 3. Recommended Next Steps

1. **Retrain CS-Transformer** with GREEN factors prioritised; use extended
   training window (2010–2022) and 18-month validation (Jan 2023 – Jun 2024).
2. **Add log_mktcap** (size factor via yfinance shares × price) — directly
   addresses the 2024 mega-cap underperformance.
3. **Investigate AMBER factors by regime**: run `2f_factor_diagnostics.py`
   and check if the decay is concentrated in one specific regime.
4. **Apply IC-weighted combination**: in `2d_factor_weights.py`, weight factors
   by OOS ICIR rather than IS ICIR to avoid overweighting decayed signals.
5. **Add residual momentum**: `ret_6m - beta × spx_ret_6m` (distinct from
   `residual_ret_12m` already in the panel — add a 6m version).

---

## 4. Figures Generated

| Figure | Description |
|--------|-------------|
| `figures/factor_decay_curves.png` | IS vs OOS IC decay curves, lags 0–12m |
| `figures/factor_regime_heatmap.png` | IC heatmap across all 5 regimes |
| `figures/factor_health_scorecard.png` | Green/Amber/Red health bar chart |

---

*Report auto-generated by `2g_factor_decay_report.py`*