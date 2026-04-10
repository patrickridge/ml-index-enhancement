# Factor Mining Results Summary

**Date:** 2026-04-10 12:47
**IS Period:** 2010-01 to 2022-12
**Candidates Generated:** 177
**Survivors (pass all screens):** 8

## Screening Funnel

| Stage | Count |
|-------|-------|
| Total candidates | 177 |
| Pass IC > 0.015 | 20 |
| Pass ICIR > 0.4 | 3 |
| Pass persistence | 128 |
| Pass dedup (<0.85 corr) | 105 |
| Pass RAS (p<0.05) | 14 |
| Pass BHY (FDR 5%) | 8 |
| **Pass ALL** | **8** |

## Surviving Factors

### vol_momentum_21d
- IC: 0.0160, ICIR: 0.267
- IC lag1: 0.0044
- Closest existing: vol_above_avg (corr=0.405)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

### cand_vol_price_divergence
- IC: 0.0203, ICIR: 0.259
- IC lag1: 0.0201
- Closest existing: high_vol_week (corr=0.345)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

### amihud_illiq_21d_top
- IC: -0.0285, ICIR: -0.353
- IC lag1: -0.0330
- Closest existing: size_proxy_bot (corr=0.814)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

### log_mktcap_top
- IC: -0.0177, ICIR: -0.363
- IC lag1: -0.0127
- Closest existing: amihud_illiq_21d_bot (corr=0.338)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

### size_proxy_bot
- IC: -0.0302, ICIR: -0.371
- IC lag1: -0.0355
- Closest existing: amihud_illiq_21d_top (corr=0.814)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

### amihud_illiq_21d_bot
- IC: -0.0256, ICIR: -0.400
- IC lag1: -0.0245
- Closest existing: size_proxy_top (corr=0.679)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

### vol_momentum_21d_bot
- IC: -0.0317, ICIR: -0.422
- IC lag1: -0.0243
- Closest existing: size_proxy_bot (corr=0.551)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

### size_proxy_top
- IC: -0.0296, ICIR: -0.432
- IC lag1: -0.0249
- Closest existing: amihud_illiq_21d_bot (corr=0.679)
- RAS p-value: 0.0000, BHY adjusted: 0.0000

