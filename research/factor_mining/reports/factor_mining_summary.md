# Factor Mining Results Summary

**Date:** 2026-04-09 17:57
**IS Period:** 2010-01 to 2022-12
**Candidates Generated:** 177
**Survivors (pass all screens):** 3

## Screening Funnel

| Stage | Count |
|-------|-------|
| Total candidates | 177 |
| Pass IC > 0.015 | 20 |
| Pass ICIR > 0.4 | 3 |
| Pass persistence | 128 |
| Pass dedup (<0.85 corr) | 105 |
| Pass RAS (p<0.05) | 3 |
| Pass BHY (FDR 5%) | 3 |
| **Pass ALL** | **3** |

## Surviving Factors

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

