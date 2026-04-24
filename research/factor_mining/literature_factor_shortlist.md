# Literature Factor Shortlist - 2020–2025

**Date:** 2026-04-06
**Scope:** JF, JFE, RFS + top empirical finance, 2020–2025
**Filter:** Implementable signals not already in the 205-feature set

---

## Tier 1 - Implementable with Current Data (OHLCV + Fundamentals)

### 1. Frog-in-the-Pan Momentum (Da, Gurun & Warachka, JF 2014 / extended 2021)
- **Signal:** Continuous information = sign(ret_6m) × fraction of same-sign daily returns. High FIP → underreaction → momentum continuation.
- **Data:** OHLCV only
- **Feasibility:** ✅ Already implemented as `cand_frog_in_pan_6m` in candidate_factory.py
- **Covered by existing?** No - `ret_consistency_12m` counts positive days but doesn't weight by direction
- **Novelty:** Medium (well-known but not in current library)

### 2. Short-Term Reversal Decomposition (Avramov, Chordia & Goyal, RFS 2006 / Nagel 2012 update)
- **Signal:** Decompose 1-month reversal into overnight return vs intraday return. Overnight component carries most of the reversal signal.
- **Data:** OHLCV only
- **Feasibility:** ✅ Implemented as `cand_cum_gap_ret_21d` and `cand_overnight_vs_intra`
- **Covered by existing?** `avg_gap_21d` captures gap magnitude but not cumulative direction
- **Novelty:** Medium-High

### 3. Maximum Daily Return (Bali, Cakici & Whitelaw, JFE 2011 / extended Kumar 2021)
- **Signal:** MAX = highest daily return in past month. High-MAX stocks underperform (lottery demand).
- **Data:** OHLCV only
- **Feasibility:** ✅ Already in library as `max_ret_21d` (Cat 16)
- **Covered by existing?** Yes - skip
- **Novelty:** N/A (already implemented)

### 4. Corwin-Schultz Spread (Corwin & Schultz, JF 2012)
- **Signal:** Bid-ask spread estimated from daily high-low prices. Higher spread = less liquid = higher expected return.
- **Data:** OHLCV only
- **Feasibility:** ✅ Implemented as `cand_corwin_schultz_21d`
- **Covered by existing?** `amihud_illiq_21d` is a different liquidity proxy - complementary, not duplicate
- **Novelty:** High (not widely used in ML pipelines)

### 5. Realized Skewness (Amaya, Christoffersen, Jacobs & Vasquez, JFE 2015 / Bali et al. 2020)
- **Signal:** Stocks with positive realized skewness earn lower returns (lottery preference). Monthly cross-sectional.
- **Data:** OHLCV only
- **Feasibility:** ✅ Implemented as `cand_realized_skew_21d/63d`
- **Covered by existing?** `skew_60d` exists - `cand_realized_skew_63d` likely duplicate, but 21d version is new
- **Novelty:** Medium (21d horizon is new information)

### 6. Trend Factor (Han, Zhou & Zhu, JFE 2016 / updated 2022)
- **Signal:** Weighted combination of MA signals across horizons. "Which MA crossovers matter most?" Weights estimated IS.
- **Data:** OHLCV only
- **Feasibility:** ✅ Could construct from existing price_to_ma* columns
- **Covered by existing?** Partially - individual MAs exist but not the composite
- **Novelty:** Medium (composite adds value over individual MAs)
- **Implementation:** `cand_trend_composite = Σ w_k × price_to_ma_k` where weights from IS IC

### 7. Chaikin Money Flow (Granville 1963, formalized in quant literature ~2020)
- **Signal:** Volume-weighted accumulation indicator. Persistent positive CMF = institutional buying.
- **Data:** OHLCV
- **Feasibility:** ✅ Implemented as `cand_smart_money_flow_21d` and `cand_accumulation_21d`
- **Covered by existing?** `obv_signal` is similar in spirit but different construction
- **Novelty:** Medium

### 8. Expected Shortfall Factor (Luo, JFE 2022)
- **Signal:** Expected shortfall (mean of worst-decile returns) predicts cross-section. More informative than VaR alone.
- **Data:** OHLCV only
- **Feasibility:** ✅ Implemented as `cand_expected_shortfall_mom`
- **Covered by existing?** `cvar_95_21d` is over 21d - 126d horizon is different
- **Novelty:** Medium-High

### 9. Kyle's Lambda from OHLCV (Hasbrouck, JF 2009 / Abdi & Ranaldo, JFE 2017)
- **Signal:** Price impact coefficient. Approximated as |ret|/sqrt(volume). Higher = more price impact = less liquid.
- **Data:** OHLCV
- **Feasibility:** ✅ Implemented as `cand_kyle_lambda_21d`
- **Covered by existing?** Different from Amihud (which uses |ret|/dollar_vol, not sqrt(vol))
- **Novelty:** High

---

## Tier 2 - Partially Implementable (Requires Proxy or Approximation)

### 10. Earnings Announcement Premium (Barber, De George, Lehavy & Trueman, JFE 2013 / Savor & Wilson 2016)
- **Signal:** Stocks earn higher returns around scheduled earnings announcements. Tradeable if you know the calendar.
- **Data:** Earnings calendar (not in repo) - proxy via largest gap days
- **Feasibility:** ⚠️ Proxied by `earnings_gap` and `post_earnings_drift` (Cat 17)
- **Novelty:** Medium (proxy is rough)

### 11. Anomaly Momentum (Huang, Li, Wang & Zhou, JFE 2023)
- **Signal:** Returns to anomalies are persistent - use past 12m anomaly return as predictor.
- **Data:** Requires multiple anomaly returns (could use our factor IC history)
- **Feasibility:** ⚠️ Could proxy via rolling IC of individual factors
- **Implementation:** Track which factors had high IC last 12 months, overweight them
- **Novelty:** High (meta-factor, captures factor momentum)
- **Note:** More relevant for Layer 1 RL (adaptive factor weighting) than as standalone

### 12. Downside Beta (Ang, Chen & Xing, RFS 2006 / Lettau, Maggiori & Weber, RFS 2014)
- **Signal:** Stocks with high downside beta earn higher returns. Up-down beta asymmetry.
- **Data:** OHLCV + SPX
- **Feasibility:** ✅ Already implemented as `up_beta_63d`, `down_beta_63d`, `up_down_beta_spread`
- **Covered by existing?** Yes - skip
- **Novelty:** N/A

---

## Tier 3 - Requires External Data (Not Currently Feasible)

### 13. News Sentiment (Tetlock, JF 2007 / Ke, Kelly & Xiu, RFS 2024)
- **Signal:** NLP-derived sentiment from news articles. Negative sentiment predicts negative returns.
- **Data:** News text corpus (not available)
- **Feasibility:** ❌ Requires NLP pipeline + news data subscription
- **Novelty:** Very High
- **Recommendation:** Future data sourcing priority

### 14. Options-Implied Volatility Skew (Xing, Zhang & Zhao, JFQA 2010 / An et al., JFE 2023)
- **Signal:** Put-call implied vol spread predicts returns. Steep skew = negative outlook.
- **Data:** Options data (not available)
- **Feasibility:** ❌
- **Novelty:** Very High

### 15. Earnings Call Tone (Loughran & McDonald, JF 2011 / updated 2020)
- **Signal:** Sentiment analysis of earnings call transcripts. Negative tone predicts drift.
- **Data:** Earnings transcripts (not available)
- **Feasibility:** ❌

### 16. ESG Scores (Pedersen, Fitzgibbons & Pomorski, JFE 2021)
- **Signal:** ESG-sin stock return spread. High-ESG stocks may carry a "green premium."
- **Data:** ESG ratings (not available)
- **Feasibility:** ❌

### 17. Retail Order Flow (Boehmer, Jones, Zhang & Zhang, RFS 2021)
- **Signal:** Aggregate retail buying pressure predicts short-term returns.
- **Data:** Trade-level data (not available)
- **Feasibility:** ❌

---

## Summary Ranking

| Rank | Factor | Novelty | Feasibility | Priority |
|------|--------|---------|-------------|----------|
| 1 | Corwin-Schultz spread | High | ✅ Done | Screen via validation |
| 2 | Kyle's lambda | High | ✅ Done | Screen via validation |
| 3 | Frog-in-the-pan | Med-High | ✅ Done | Screen via validation |
| 4 | Expected shortfall factor | Med-High | ✅ Done | Screen via validation |
| 5 | Overnight return decomposition | Med-High | ✅ Done | Screen via validation |
| 6 | Realized skewness 21d | Medium | ✅ Done | Screen via validation |
| 7 | Trend composite | Medium | To build | Construct from existing MAs |
| 8 | Anomaly momentum (meta-factor) | High | ⚠️ | Better for RL Layer 1 |
| 9 | News sentiment | Very High | ❌ | Future data priority |
| 10 | Options-implied skew | Very High | ❌ | Future data priority |

**Key finding:** Most Tier 1 literature factors are already implemented in `candidate_factory.py`. The biggest untapped opportunities require external data (news, options, earnings transcripts).
