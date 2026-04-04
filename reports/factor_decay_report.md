# Factor Decay Diagnostic Report
**Generated:** 2026-04-03  
**IS training period:** 2010-01-01 → 2022-12-31  
**OOS test period:** 2022-12 → present (39 months)  

---

## 1. Executive Summary

- **36 factors are GREEN** (OOS IC stable, decay ratio ≥ 0.7) — safe to use
- **2 factors are AMBER** (some decay, ratio 0.3–0.7) — use with caution
- **41 factors are RED** (severe decay or sign flip OOS) — review or drop
- **39 factors flipped sign OOS**: `macd_signal, trend_slope_21d, ma_cross_10_50, price_to_ma50, ret_1m, omega_ratio_21d, open_to_close_21d, mom_decel, rsi_14, residual_ret_1m, price_to_ma20, price_to_ma100, ir_3m, skew_60d, cvar_95_21d, ret_2w, bollinger_pct, avg_hl_range_21d, hl_range, idio_vol_252d, ret_3m, atr_21d_norm, obv_signal, trend_slope_63d, price_to_ma10, macd_hist, avg_gap_21d, kurt_60d, amihud_illiq_21d, vol_5d, vol_252d, vol_of_vol_63d, dollar_vol_63d, size_proxy, ret_1w, vol_126d, dollar_vol_21d, vol_trend_ratio, beta_x_idiovol`

### Top 5 Factors by |OOS IC|

| factor           |   ic_is |   ic_oos |   icir_oos |   decay_ratio | health   |
|:-----------------|--------:|---------:|-----------:|--------------:|:---------|
| ret_36m          | +0.0096 |  +0.0609 |    +0.4680 |       +6.3200 | GREEN    |
| ret_18m          | +0.0013 |  +0.0606 |    +0.3760 |      +46.4300 | GREEN    |
| ir_12m           | +0.0113 |  +0.0552 |    +0.3920 |       +4.8800 | GREEN    |
| residual_ret_12m | +0.0109 |  +0.0510 |    +0.3280 |       +4.6900 | GREEN    |
| ret_24m          | +0.0048 |  +0.0507 |    +0.2810 |      +10.6500 | GREEN    |

### Most Decayed / Flipped Factors

| factor           |   ic_is |   ic_oos |   decay_ratio | health   |
|:-----------------|--------:|---------:|--------------:|:---------|
| vol_trend        | -0.0019 |  -0.0000 |       +0.0100 | RED      |
| beta_x_idiovol   | +0.0097 |  -0.0008 |       -0.0900 | RED      |
| ret_1w           | -0.0173 |  +0.0036 |       -0.2100 | RED      |
| volume           | +0.0130 |  +0.0028 |       +0.2200 | RED      |
| dollar_vol_21d   | -0.0072 |  +0.0026 |       -0.3600 | RED      |
| price_to_ma10    | -0.0303 |  +0.0111 |       -0.3700 | RED      |
| vol_momentum_21d | +0.0182 |  +0.0068 |       +0.3800 | AMBER    |
| trend_r2_126d    | +0.0034 |  +0.0016 |       +0.4600 | AMBER    |
| macd_hist        | -0.0186 |  +0.0101 |       -0.5400 | RED      |
| dollar_vol_63d   | -0.0071 |  +0.0042 |       -0.5900 | RED      |

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
residual_ret_12m
ret_24m
ret_12m
ret_consistency_12m
corr_spx_63d
mom_2_12
vol_excess
maxdd_21d
nearness_52w_low
drawdown_pct_252d
mom_accel
price_to_ma200
ret_9m
up_beta_63d
ir_6m
corr_spx_21d
down_beta_63d
beta_63d
maxdd_126d
ret_6m
trend_r2_21d
beta_21d
downvol_21d
beta_252d
var_95_21d
up_down_beta_spread
ma_cross_50_200
intraday_vol_21d
beta_stability_63d
vol_20d
vol_21d
vol_60d
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
mom_decel
rsi_14
residual_ret_1m
price_to_ma20
price_to_ma100
ir_3m
skew_60d
cvar_95_21d
ret_2w
bollinger_pct
avg_hl_range_21d
hl_range
idio_vol_252d
ret_3m
atr_21d_norm
obv_signal
trend_slope_63d
price_to_ma10
macd_hist
avg_gap_21d
kurt_60d
amihud_illiq_21d
vol_5d
vol_252d
vol_of_vol_63d
dollar_vol_63d
size_proxy
ret_1w
vol_126d
volume
dollar_vol_21d
vol_trend_ratio
beta_x_idiovol
vol_trend
vol_momentum_21d
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