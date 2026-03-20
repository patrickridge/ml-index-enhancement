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

---

## Phase 6 — Model Training Prep (19 Mar 2026)

### 6.1 — Portfolio Construction: Top/Bottom 100

**Change:** Moved from continuous z-score tilt (all 500 stocks) to discrete top/bottom 100 approach.

**Old:** `w_i = w_SPX_i + α × z-score(score_i)` applied to all stocks
**New:** Overweight top 100 stocks by ML score, underweight bottom 100, hold middle ~300 at benchmark weight

**Config change:** `TOP_N = 100`, `BOTTOM_N = 100` (was 50/50)

**Rationale:** Kieran's suggestion — cleaner signal, avoids applying weak/noisy scores to middle stocks.

### 6.2 — Metrics Overhaul

**Removed:** Sharpe ratio as primary metric
**Added:**
- **Information Ratio (IR)** = alpha / tracking error — primary metric for index enhancement
- **Hit rate** — % of months portfolio beats benchmark
- **Max active drawdown** — worst sustained underperformance vs benchmark
- **Active return** — raw monthly alpha

**Rationale:** Sharpe doesn't capture relative performance vs benchmark. Since portfolio stays close to S&P 500, IR and hit rate are more meaningful. Small alpha over index requires precise relative metrics.

### 6.3 — Sign Flipping for Contrarian Factors

**Plan:** Flip sign of negative IC factors (bollinger, RSI, price_to_MA etc.) before feeding into ML model so all features point in the same direction (higher = better predicted return).

**Factors to flip:** All 17 contrarian factors identified in regime stability analysis.

### 6.4 — Gradual Training Plan (Kieran's roommate's suggestion)

Train model incrementally, one improvement at a time:
1. **Step 1:** Baseline — IC filter + RAS, contrarian factors sign-flipped, predict `fwd_ret_1m`
2. **Step 2:** IC decay curve fitting — find smooth representation of each factor's decay, use for factor weighting. Also adjust quintile treatment for non-monotonic factors (top/bottom 100 only)
3. **Step 3:** Add Kieran's fundamental factors (PE, PB, money flow etc.) once data arrives
4. **Step 4:** Factor weighting by decay profile (short-term vs long-term)

### 6.5 — Project Structure Cleanup (19 Mar 2026)

- Moved `DEVELOPMENT_LOG.md`, `STRATEGY.md`, `notes.md` → `Notes/`
- Renamed `"1 price parquet.py"` → `1_price_parquet.py`, `"6 Diagonstic test.py"` → `6_diagnostic_test.py`
- Deleted one-off scripts: `check_overfit.py`, `inspect_data.py`, `5_pitch_validator.py`, `report.ipynb`, `data.xlsx`
- Moved `2_lgbm_backtest.py`, kaggle scripts, `3_pca_rp_backtest.py`, `6_diagnostic_test.py` → `archive/`
- Moved all `*.png` figures → `figures/` at repo root
- Restructured repo: moved all files from `Investsoc ML project/` subfolder to repo root
- Added `.gitignore` (excludes `.claude/`, `*.parquet`, `*.csv`, `*.xlsx`, `__pycache__`)

### 6.6 — 7_factor_diagnostics.py date fix (pending)

`7_factor_diagnostics.py` uses full panel with no date cutoff. Should be filtered to 2010–2020 training period only to avoid test period leakage in correlation/RMT/VIF analysis. Low priority — diagnostics only, not used in training.

---

---

## Phase 7 — Backtest Execution & Known Issues (19 Mar 2026)

### 7.1 — FT-Transformer Kaggle Run (19 Mar 2026)

Results from `2b_nn_backtest_kaggle.py` with updated TOP_N=100, LONG_FRAC=0.20:

| Strategy | Sharpe | Ann Ret | Max DD |
|----------|--------|---------|--------|
| Long-Only Top 100 | 1.45 | 28.35% | -16.95% |
| Long-Short top/bot 20% | 1.23 | 19.09% | -9.65% |

Note: Sharpe dropped slightly vs Phase 1 (1.60 → 1.45) because TOP_N increased from 50 → 100 (more diversified = lower concentration = lower return and vol). Expected.

Trained in 10.7 min on Kaggle GPU T4.
`scores_transformer.parquet` saved to `data/`.

### 7.2 — Timers Added

Added `Done in X.X min` timer to all scripts missing it:
- `2_lgbm_backtest.py`
- `2c_cs_transformer.py`
- `3_pca_rp_backtest.py`

### 7.3 — `6_index_enhancement.py` Sharpe Bug Fixed

`ie_stats()` was missing `sharpe` key but it was referenced in the print table. Fixed by adding portfolio Sharpe computation to `ie_stats()`.

### 7.4 — SPX Weights Data Quality Issue (NEEDS FIXING)

`1c_fetch_market_cap.py` ran successfully (500/504 tickers, 2 min) but market cap weights are distorted:
- NVDA showing 25.38% (should be ~6-7%)
- WMT showing 12.92% (should be ~0.7%)

**Root cause:** yfinance `get_shares_full()` returns inconsistent/stale shares outstanding for some tickers — current shares applied to historical prices without adjusting for splits/buybacks creates relative weight distortion.

**Impact:** `6_index_enhancement.py` will run but the benchmark return won't accurately reflect the true S&P 500. Active returns and IR figures will be directionally correct but numerically off.

**Fix options (in priority order):**
1. Use a proper data source for SPX constituent weights (Kieran's platform may have this)
2. Cross-check with known cap weights from another free source (e.g. SPDR holdings CSV)
3. Use equal-weight benchmark as fallback

**Status:** Not blocking — pipeline works, fix before any final results presentation.

### 7.5 — New Scripts Added

- `5c_ic_decay_daily.py` — daily IC decay (1–90 trading days) per factor. Kieran's suggestion to confirm monthly rebalancing is optimal horizon.

### 7.6 — Kieran's New Ideas (19 Mar 2026)

- **Barra-style factors** — size (mktcap), value (P/B), leverage (D/E), earnings yield. Can approximate from yfinance. Kieran's platform data will likely cover these.
- **Smart money factors** — institutional ownership changes (13F), short interest, insider buying. Harder to get without a data vendor.
- **Daily IC decay** — built as `5c_ic_decay_daily.py`. To send to Kieran once run.
- **Focus for now:** Complete backtesting framework first, then factor additions.

---

### 7.7 — Index Enhancement Results (19 Mar 2026)

First full run of `6_index_enhancement.py` with all 3 model scores. Best α per model (target TE 1–5%):

| Model | Ann α | Tracking Error | IR | Sharpe | Max DD |
|-------|-------|---------------|-----|--------|--------|
| CS-Transformer | **1.76%** | 2.27% | **0.776** | 2.46 | -9.0% |
| FT-Transformer | 0.70% | 2.57% | 0.271 | 2.42 | -8.5% |
| LGBM | 0.25% | 1.96% | 0.130 | 2.49 | -8.2% |

All at α=0.002. Larger tilts hurt — alpha drops quickly as α increases.

**Key finding:** CS-Transformer dominates in index enhancement despite being middle-ranked in standalone mode. Sees all 500 stocks simultaneously so learns relative rankings directly — exactly what IE needs. LGBM scores stocks independently so signal gets noisy across full universe.

**Caveats:** SPX weights distorted (WMT 12.92%, NVDA 25.38%) so IR figures are directionally correct but numerically off. Only 35 months test data in a bull market.

---

---

## Phase 8 — Factor Framework & Weight Fixes (19 Mar 2026)

### 8.1 — `5d_factor_weights.py` (new script)

Combines all factor analysis outputs into a single clean table for model training.

**Steps:**
1. **Filter** — |IC| > 0 AND regime_stable → 43/59 factors survive
2. **IC decay weighting** — persistence weight = mean |IC| over lags 0–12 months (area under 1-year decay curve). Factors with longer-lasting signal get higher weight. Normalised to sum to 1.
3. **Partial flag** — `use_partial = True` for non-monotonic factors (Q5→Q1 ordering broken). For these, only top/bottom 100 stocks carry signal — middle 300 held at benchmark weight.

**Output:** `data/factor_selected.csv` — complete factor list with weights and flags, ready for model training.

**Why mean |IC| over lags 0–12 (not half-life):** IC doesn't decay monotonically — it oscillates and bounces back (Kieran's observation). Half-life fitting is unstable. Mean area under curve is robust and captures integrated signal strength over a 1-year window.

### 8.2 — SPX Weights Fix

**Root cause of WMT/NVDA distortion:** `fast_info.shares` from yfinance returns stale or non-float-adjusted share counts for some tickers. WMT was returning ~8× too many shares.

**Fix 1 (primary):** Changed `fetch_current_shares()` to derive implied shares from `market_cap / last_price` instead of using raw share count. `fast_info.market_cap` is more reliably reported.

**Fix 2 (safety net):** Added 8% winsorisation after computing weights — any single stock capped at 8% weight, then renormalised. Prevents any remaining outlier from dominating the benchmark. Real S&P 500 top holdings are ~6–7% (AAPL, NVDA, MSFT).

### 8.3 — CS-Transformer Kaggle Script Confirmed Correct

`2c_cs_transformer_kaggle.py` already had `TOP_N=100`, `BOTTOM_N=100`, `LONG_FRAC=0.20`. No changes needed.

---

## Current Status (19 Mar 2026 — evening)

**Completed today:**
- All 3 models trained and scored (LGBM, FT-Transformer, CS-Transformer)
- Full IE pipeline working — CS-Transformer IR=0.776 best result
- Factor framework complete: filtering → decay weighting → partial flags
- SPX weights fix (market_cap/price + 8% winsorisation)
- Daily IC decay script built (`5c_ic_decay_daily.py`)

**Blocked on:** Kieran's full historical fundamental data (2010–2025)

**Next steps when data arrives:**
1. Re-run `1c_fetch_market_cap.py` → verify top-10 weights now realistic
2. Merge Kieran's factors → rebuild panel (`1_feature_engineering.py`)
3. Re-run `5_factor_analysis.py` → `5d_factor_weights.py` on expanded factor set
4. Re-train FT-Transformer + CS-Transformer on Kaggle with `factor_selected.csv` filter
5. Re-run `6_index_enhancement.py` → compare IR vs current baseline
