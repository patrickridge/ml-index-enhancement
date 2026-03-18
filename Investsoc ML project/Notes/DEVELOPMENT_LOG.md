# Development Log — Investsoc ML Project

Chronological record of all significant changes, decisions, and findings.

---

## Phase 1 — Initial Pipeline (pre-15 Mar 2026)

**Goal:** Build ML models to rank S&P 500 stocks monthly and construct a portfolio that beats the index.

### What was built
- `1_feature_engineering.py` — 80-feature monthly panel from raw OHLCV price data (504 tickers, 2006–2025)
- `1b_orthogonalize.py` — PCA orthogonalization (reduces mean off-diagonal feature correlation 0.22 → 0.10)
- `2_lgbm_backtest.py` — LightGBM gradient boosted trees
- `2b_nn_backtest.py` — FT-Transformer (attention-based neural net, GPU required)
- `2c_cs_transformer.py` — Cross-Sectional Transformer (all 500 stocks seen simultaneously)
- `3_pca_rp_backtest.py` — PCA Risk Parity (experimental, abandoned — high beta)
- `4_benchmark_spx.py` — comparison table vs S&P 500

### Phase 1 Results (test period Jan 2023 – Nov 2025, 35 months)

| Strategy | Sharpe | Ann Ret | Beta | Alpha |
|----------|--------|---------|------|-------|
| LGBM Long-Only | **1.82** | 32.7% | 0.07 | +32.9% |
| LGBM Long-Short | 1.66 | 21.9% | -0.06 | +24.2% |
| FT-Transformer Long-Only | 1.60 | 32.7% | 0.27 | +28.9% |
| **S&P 500 benchmark** | 1.40 | 16.2% | 1.00 | — |
| CS Transformer Long-Only | 1.25 | 29.1% | 0.24 | +26.6% |

All 3 main models beat the S&P 500 on Sharpe ratio.

---

## Phase 2 — Index Enhancement (13–14 Mar 2026)

**Decision:** Goal is index enhancement (not unconstrained stock picking). Universe is naturally constrained to S&P 500 — this makes it structurally an IE problem.

**Target metrics:** Tracking error 2–4%, Information Ratio > 0.5.

**Portfolio construction change:**
- Old: equal-weight top-50 stocks
- New: `w_i = w_SPX_i + α × z-score(ML_score_i)` — tilt all 500 stocks around cap weights

### Scripts added
- `1c_fetch_market_cap.py` — fetches monthly SPX constituent weights via yfinance (shares × close)
- `6_index_enhancement.py` — runs IE portfolio construction, sweeps alpha values (0.005–0.05)

### IE Results

| Model | Alpha (α=0.02) | IR | Tracking Error |
|-------|---------------|-----|----------------|
| LGBM | best | 0.55–0.64 | 3.4–4.2% |
| FT-Transformer | — | — | — |
| CS Transformer | — | — | — |

IR > 0.5 = good (top-quartile institutional fund managers hit ~0.5).

---

## Phase 3 — Factor Analysis (15 Mar 2026)

**Decision:** Kieran's methodology — do factor analysis before ML training.

### Scripts added
- `5_factor_analysis.py` — IC analysis, IC decay (lags 0–6 months), quintile backtests

### Key findings from factor analysis (72 features at the time)

| Category | Best factor | ICIR | Ann Q5–Q1 spread | IC decay |
|----------|------------|------|-----------------|---------|
| Volatility | vol_252d | +0.155 | +14.6% | Flat — persistent signal |
| Idiosyncratic vol | idio_vol_252d | +0.172 | +13.4% | Flat — persistent |
| Mean reversion | bollinger_pct | -0.157 | -4.3% | Decays immediately at lag 1 |
| Price/trend | price_to_ma10 | -0.138 | -3.2% | Reverses at lag 1 (mean reversion) |
| Junk (removed) | open/high/low/close | ±0.26–0.27 | -10% | Spurious size bias |

**Key insight:** Dominant signal is volatility risk premium — high-vol stocks outperform in the 2010–2025 bull market. IC is flat over 6+ months → structural tilt, not timing signal. Fundamental factors (PE, ROE) absent — Kieran's new data will fill this gap.

### Kieran's feedback (15 Mar)
1. Remove `open`, `high`, `low`, `close` as features (raw dollar prices, not alpha)
2. Add more technical indicators: MACD variants, more MA windows/crossovers

### Changes made (15 Mar)
**`1_feature_engineering.py`:**
- Added `"open", "high", "low", "close"` to `exclude_cols` in `sample_at_month_end()`

**`utils_factors.py` Cat 4:**
- Expanded MA windows: `[10, 50, 100]` → `[10, 20, 50, 100, 200]` (adds `price_to_ma20`, `price_to_ma200`)
- Added `ma_cross_10_50`, `ma_cross_50_200` (golden/death cross signals)
- Added `macd_hist` = MACD line - 9-day EMA of MACD line (momentum of momentum)

**Net feature count:** 80 → 81 (−4 OHLC + 5 new technical)

---

## Phase 4 — Feature Deduplication (15 Mar 2026)

**Trigger:** Ran `7_factor_diagnostics.py`. Fixed Part D bug (`N` vs `N_orth` shape mismatch).

### Diagnostics findings (72 features)
- Mean off-diagonal correlation: **0.235** (high redundancy)
- RMT: only **10 signal eigenvalues** out of 72 (85.1% variance in 10 factors)
- **513 redundant pairs** with |corr| > 0.5
- **44 features with VIF > 10**, of which 17 with VIF = ∞ (perfect collinearity)

### Root causes of perfect collinearity
| Removed | Kept | Reason |
|---------|------|--------|
| `ret_rel_spx_1m` | `ret_1m` | SPX return constant cross-sectionally → rank(ret_rel) = rank(ret) |
| `ret_rel_spx_3m` | `ret_3m` | Same |
| `ret_rel_spx_6m` | `ret_6m` | Same |
| `vol_rel_spx_63d` | `vol_21d` | Denominator `spx_vol_63d` constant cross-sectionally → same rank |
| `beta_adj_ret_12m` | `residual_ret_12m` | Exact alias (same computation) |
| `rev_signal` | `ret_1m` | `rev_signal = -ret_1m` → perfect negative duplicate |
| `rev_1m` | `ret_1m` | Same — came from v1 `panel_monthly.parquet` |
| `rsi_21` | `rsi_14` | R² = 0.99, ICIR near-identical, VIF > 100 |

### SPX-level columns moved to MACRO_COLS (`config.py`)
`spx_ret_1m`, `spx_ret_3m`, `spx_ret_6m`, `spx_ret_12m`, `spx_vol_63d`

**Reason:** Same value for all stocks per month → after cross-sectional ranking they become constants. Moving to MACRO_COLS means they get time-series z-scored instead — preserving their regime information (market momentum/volatility state) for the ML models.

### Files changed
- `utils_factors.py` — Cat 8: removed redundant cross-sectional relatives, kept `residual_ret_12m`
- `utils_factors.py` — Cat 4: removed `rsi_21`
- `1_feature_engineering.py` — drop `rev_1m` on load; removed `rev_signal` computation
- `config.py` — expanded `MACRO_COLS` to include 5 SPX-level columns

### Result after cleanup
- **Cross-sectional features:** 72 → 59 (−13 duplicates/junk)
- **Macro (time-series z-scored):** 9 → 14 (+5 SPX regime cols)
- **Total:** 73 features (cleaner, no VIF=∞ in cross-sectional set)
- **RMT:** expect ~8–10 genuine signals after re-run (unchanged — noise was redundant, not signal)

---

## Priority 3 — RMT Denoising (PENDING)

**Finding:** Only 10 eigenvalues above the Marchenko-Pastur noise floor in 72 features. 62 eigenvalues are pure noise.

**Options:**
1. **Run `1b_orthogonalize.py`** — PCA residualization already built, reduces mean off-diag corr 0.235 → 0.100 (57.5% reduction). ML models then see decorrelated inputs.
2. **RMT covariance denoising** — already in `utils_rmt.py`, used by CS Transformer. Could apply to LGBM feature set too.
3. **Hard feature selection** — keep only factors with |ICIR| above threshold (currently best is 0.172, none above 0.3). Wait for Kieran's fundamental data — PE/ROE factors likely to score much higher.

**Recommended:** Run `1b_orthogonalize.py` after Kieran's data arrives and feature set is finalised. Orthogonalizing now before new data = wasted computation.

---

## Waiting On (as of 15 Mar 2026)

- **fundamental data** — PE ratio confirmed, others TBD
- Once data arrives: rebuild panel → re-run factor analysis → re-train all 3 models → re-run index enhancement
- Then: run `1b_orthogonalize.py` → retrain with orthogonalized features → compare IR

---

## Phase 5 — Factor Analysis Overhaul (17–18 Mar 2026)

### 5.1 — Extended IC Decay + Regime Stability

**Changes to `5_factor_analysis.py`:**
- Extended `MAX_IC_DECAY_LAGS` from 6 → 24 → 60 months (Kieran requested 2–3 year window to see longer trends)
- Added `factor_turnover()` — % stocks changing quintile month-to-month
- Added `ras_permutation_test()` — 100 permutations, p < 0.05 = real signal
- Added `regime_stability()` — 4 market regimes, COVID crash Mar–May 2020 excluded as black swan:
  - QE bull: 2010–2019
  - COVID recovery: Jun 2020–2021
  - Rate hike bear: 2022
  - AI bull: 2023–present
- Added `plot_quintile_bars()` — Q1–Q5 bar chart per factor
- All figures moved to `data/figures/`

**Regime stability results (59 factors):**
- 43/59 factors pass regime stability (sign consistent across ≥3/4 regimes)
- COVID crash excluded — "no one can predict COVID" (Kieran)
- Filters applied: |IC| > 0 AND sign consistent → 43 factors kept

### 5.2 — Abs IC Fix (17 Mar 2026)

**Bug found:** Original filter used `ic_mean > 0`, dropping all contrarian factors.

**Fix:** Changed to `|ic_mean| > 0` — keeps both positive and negative IC factors.

**Rationale:** Negative IC is not useless — it means the factor works in reverse (contrarian signal). `bollinger_pct` (IC = -0.037, ICIR = -0.157) is one of the strongest factors but was being dropped. The sign just tells you the direction to trade.

**New filter logic:**
1. Filter 1: `|ic_mean| > 0` — any directional signal kept
2. Filter 2: sign consistent across regimes — not flipping + to − between regimes

**Result:** 43/59 factors survive (26 positive IC + 17 contrarian)

**Separate plots for meeting:**
- `factor_ic_summary_positive.png` — top 10 positive IC
- `factor_ic_summary_negative.png` — top 10 contrarian
- `factor_ic_decay_positive.png` — decay curves positive
- `factor_ic_decay_negative.png` — decay curves contrarian

### 5.3 — All-factor IC Decay Grid (18 Mar 2026)

Added `5b_ic_decay_all.py` — computes and plots IC decay for all 59 factors in a grid layout (blue = positive IC, red = contrarian). 60-month window.

**Key visual findings:**
- `vol_252d`, `idio_vol_252d`, `vol_126d`, `beta_252d` — smooth persistent signal for 60 months (long-term factors)
- `nearness_52w_low` — distinctive hump, peaks ~lag 12 (medium-term)
- `ret_12m`, `ir_12m` — positive but dips at lag 12–24 (COVID recovery effect)
- Short-term momentum (`ret_1m`, `ret_1w`, `ret_2w`) — chaotic, bouncing around zero at all lags
- `maxdd_126d`, `var_95_21d`, `cvar_95_21d` — smooth persistent contrarian signal

**Kieran's observation:** IC half-life is not a good metric — IC doesn't decay monotonically, it oscillates and bounces back. Agreed — dropped IC half-life idea.

### 5.4 — Conceptual Clarifications (18 Mar 2026)

**IC decay ≠ factor filtering (Kieran's correction):**
- **Filtering** = is this factor worth using? (IC, ICIR, regime stability, RAS)
- **IC decay** = how long does the signal last? Informs rebalancing frequency and factor weighting, not inclusion/exclusion
- IC decay runs only on filtered survivors, not all factors

**Negative IC is not useless (confirmed):**
- IC = correlation between factor ranking and returns
- Negative IC = factor works in reverse direction
- |ICIR| is the quality metric — sign just tells you direction
- ICIR = IC_mean / IC_std measures consistency over time

**Factor longevity methods beyond IC decay:**
- Factor value autocorrelation (ACF) — how sticky is the factor itself month-to-month
- Quintile survival analysis — how long do top-ranked stocks stay in the top quintile
- IC decay is the industry standard; half-life approach rejected as IC oscillates rather than monotonically decaying

**Non-monotonic factors (Kieran's idea):**
- Factors without clean Q5>Q4>Q3>Q2>Q1 ordering → only use top/bottom 100 stocks
- Signal only works at extremes for these factors, not the middle quintiles

**Factor weighting by decay profile (Kieran's idea):**
- Short-term factors (fast IC decay) → higher weight for monthly rebalancing
- Long-term factors (persistent IC) → lower rebalancing frequency / lower weight
- To be implemented after model training

### 5.5 — Train/Test Split Fix (18 Mar 2026)

**Bug found:** Factor analysis (IC, ICIR, quintile, RAS) was using full 2010–2025 data. Since model test period is 2021–2025, this was lookahead bias — using test period data to select which factors to use.

**Fix:** Added `TRAIN_END = "2020-12-31"` cutoff:
- IC, ICIR, quintile backtests, RAS, turnover → **2010–2020 only** (training data)
- Regime stability → **full panel** (needs 2022 rate hike bear + 2023 AI bull regimes)
- IC decay → **training data only, survivors only**

**Code change in `5_factor_analysis.py`:**
```python
TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-12-31"   # hold out 2021–2025 as test set
panel_train = panel[panel["date"] <= TRAIN_END]
```

### 5.6 — Model Decision (18 Mar 2026)

**LightGBM dropped** — project moving forward with:
- **FT-Transformer** (Sharpe 1.60)
- **Cross-Sectional Transformer** (Sharpe 1.25)
- Plan: ensemble both models (blend stock scores) for final portfolio

**Rationale:** Transformer models better suited for the cross-sectional structure of the data and Kieran's firm is building toward live deployment.

### 5.7 — Kieran's Factor Data (18 Mar 2026)

Kieran shared `factor data.xlsx` from his quant platform (Wind/similar):
- **33 unique factors**, 503 S&P 500 tickers
- Factor categories: money flow (buy/sell amt, order count), valuation (PE TTM, PB, PS, PCF, EV/EBITDA, dividend yield, PE relative to history), risk (beta 20/60/120d, vol ratio, Treynor ratio, residual vol), technical (turnover 5/10/20/60d)
- All marked **PIT (Point-In-Time)** — no lookahead bias ✓
- **Problem:** Only December 2025 data. Need full 2010–2025 history for IC tests and model training

**Priority factors to pull (2010–2025 monthly):**
```
pe_ttm, val_petohist20/60/120/250, val_peindu_sw, val_pbindu_sw
ps_ttm, val_psindu_sw, pcf_ocf_ttm, pcf_ncf_ttm
dividendyield2, val_mvtoebitda_ttm
risk_beta20/60/120, risk_residvol252
mfd_buyamt_d, mfd_sellamt_d, tech_turnoverrate20/60
```

### 5.8 — Live Deployment Plan (18 Mar 2026)

Kieran's Shenzhen firm gave him access to a quant platform for live strategy deployment. End-to-end pipeline to automate:
1. Pull new factor data monthly from platform
2. Run FT-Transformer + CS-Transformer → generate stock scores
3. Compute portfolio weights: `w_i = w_SPX_i + α × z-score(ML_score_i)`
4. Execute rebalancing trades automatically

**Future integration:** Investsoc sector analyst picks could be incorporated as an overlay (quant-fundamental hybrid), boosting ML scores for stocks analysts also like.

---

## Waiting On (as of 18 Mar 2026)

1. **Kieran's full historical factor data** (2010–2025) — main blocker for model training
2. **Tonight's meeting** — align on: final filter criteria, factor weighting by decay, non-monotonic factor treatment, train/test setup
3. **After data arrives:** merge panel → re-run factor analysis (train period only) → train FT-Transformer + CS-Transformer → ensemble → backtest → automate
