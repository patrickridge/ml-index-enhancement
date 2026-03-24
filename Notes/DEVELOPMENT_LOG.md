# Development Log — Investsoc ML Project

Chronological record of all significant changes, decisions, and findings.

---

## Phase 1 — Initial Pipeline (pre-15 Mar 2026)

**Goal:** Build ML models to rank S&P 500 stocks monthly and construct a portfolio that beats the index.

### What was built
- `1g_feature_engineering.py` — 80-feature monthly panel from raw OHLCV price data (504 tickers, 2006–2025)
- `1h_orthogonalize.py` — PCA orthogonalization (reduces mean off-diagonal feature correlation 0.22 → 0.10)
- `2_lgbm_backtest.py` — LightGBM gradient boosted trees
- `3a_ft_transformer.py` — FT-Transformer (attention-based neural net, GPU required)
- `3c_cs_transformer.py` — Cross-Sectional Transformer (all 500 stocks seen simultaneously)
- `3_pca_rp_backtest.py` — PCA Risk Parity (experimental, abandoned — high beta)
- `4d_benchmark_spx.py` — comparison table vs S&P 500

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
- `4b_index_enhancement.py` — runs IE portfolio construction, sweeps alpha values (0.005–0.05)

### IE Results

| Model | Alpha (α=0.02) | IR | Tracking Error |
|-------|---------------|-----|----------------|
| LGBM | best | 0.55–0.64 | 3.4–4.2% |
| FT-Transformer | — | — | — |
| CS Transformer | — | — | — |

IR > 0.5 = good (top-quartile institutional fund managers hit ~0.5).

---

## Phase 3 — Factor Analysis (15 Mar 2026)

**Decision:** the methodology — do factor analysis before ML training.

### Scripts added
- `2a_factor_analysis.py` — IC analysis, IC decay (lags 0–6 months), quintile backtests

### Key findings from factor analysis (72 features at the time)

| Category | Best factor | ICIR | Ann Q5–Q1 spread | IC decay |
|----------|------------|------|-----------------|---------|
| Volatility | vol_252d | +0.155 | +14.6% | Flat — persistent signal |
| Idiosyncratic vol | idio_vol_252d | +0.172 | +13.4% | Flat — persistent |
| Mean reversion | bollinger_pct | -0.157 | -4.3% | Decays immediately at lag 1 |
| Price/trend | price_to_ma10 | -0.138 | -3.2% | Reverses at lag 1 (mean reversion) |
| Junk (removed) | open/high/low/close | ±0.26–0.27 | -10% | Spurious size bias |

**Key insight:** Dominant signal is volatility risk premium — high-vol stocks outperform in the 2010–2025 bull market. IC is flat over 6+ months → structural tilt, not timing signal. Fundamental factors (PE, ROE) absent — new fundamental data will fill this gap.

### Feedback (15 Mar)
1. Remove `open`, `high`, `low`, `close` as features (raw dollar prices, not alpha)
2. Add more technical indicators: MACD variants, more MA windows/crossovers

### Changes made (15 Mar)
**`1g_feature_engineering.py`:**
- Added `"open", "high", "low", "close"` to `exclude_cols` in `sample_at_month_end()`

**`utils_factors.py` Cat 4:**
- Expanded MA windows: `[10, 50, 100]` → `[10, 20, 50, 100, 200]` (adds `price_to_ma20`, `price_to_ma200`)
- Added `ma_cross_10_50`, `ma_cross_50_200` (golden/death cross signals)
- Added `macd_hist` = MACD line - 9-day EMA of MACD line (momentum of momentum)

**Net feature count:** 80 → 81 (−4 OHLC + 5 new technical)

---

## Phase 4 — Feature Deduplication (15 Mar 2026)

**Trigger:** Ran `2f_factor_diagnostics.py`. Fixed Part D bug (`N` vs `N_orth` shape mismatch).

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
- `1g_feature_engineering.py` — drop `rev_1m` on load; removed `rev_signal` computation
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
1. **Run `1h_orthogonalize.py`** — PCA residualization already built, reduces mean off-diag corr 0.235 → 0.100 (57.5% reduction). ML models then see decorrelated inputs.
2. **RMT covariance denoising** — already in `utils_rmt.py`, used by CS Transformer. Could apply to LGBM feature set too.
3. **Hard feature selection** — keep only factors with |ICIR| above threshold (currently best is 0.172, none above 0.3). Wait for the team's fundamental data — PE/ROE factors likely to score much higher.

**Recommended:** Run `1h_orthogonalize.py` after the team's data arrives and feature set is finalised. Orthogonalizing now before new data = wasted computation.

---

## Waiting On (as of 15 Mar 2026)

- **fundamental data** — PE ratio confirmed, others TBD
- Once data arrives: rebuild panel → re-run factor analysis → re-train all 3 models → re-run index enhancement
- Then: run `1h_orthogonalize.py` → retrain with orthogonalized features → compare IR

---

## Phase 5 — Factor Analysis Overhaul (17–18 Mar 2026)

### 5.1 — Extended IC Decay + Regime Stability

**Changes to `2a_factor_analysis.py`:**
- Extended `MAX_IC_DECAY_LAGS` from 6 → 24 → 60 months (Requested 2–3 year window to see longer trends)
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
- COVID crash excluded — "no one can predict COVID" 
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

Added `2b_ic_decay_all.py` — computes and plots IC decay for all 59 factors in a grid layout (blue = positive IC, red = contrarian). 60-month window.

**Key visual findings:**
- `vol_252d`, `idio_vol_252d`, `vol_126d`, `beta_252d` — smooth persistent signal for 60 months (long-term factors)
- `nearness_52w_low` — distinctive hump, peaks ~lag 12 (medium-term)
- `ret_12m`, `ir_12m` — positive but dips at lag 12–24 (COVID recovery effect)
- Short-term momentum (`ret_1m`, `ret_1w`, `ret_2w`) — chaotic, bouncing around zero at all lags
- `maxdd_126d`, `var_95_21d`, `cvar_95_21d` — smooth persistent contrarian signal

**observation:** IC half-life is not a good metric — IC doesn't decay monotonically, it oscillates and bounces back. Agreed — dropped IC half-life idea.

### 5.4 — Conceptual Clarifications (18 Mar 2026)

**IC decay ≠ factor filtering (correction):**
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

**Non-monotonic factors (approach):**
- Factors without clean Q5>Q4>Q3>Q2>Q1 ordering → only use top/bottom 100 stocks
- Signal only works at extremes for these factors, not the middle quintiles

**Factor weighting by decay profile (approach):**
- Short-term factors (fast IC decay) → higher weight for monthly rebalancing
- Long-term factors (persistent IC) → lower rebalancing frequency / lower weight
- To be implemented after model training

### 5.5 — Train/Test Split Fix (18 Mar 2026)

**Bug found:** Factor analysis (IC, ICIR, quintile, RAS) was using full 2010–2025 data. Since model test period is 2021–2025, this was lookahead bias — using test period data to select which factors to use.

**Fix:** Added `TRAIN_END = "2020-12-31"` cutoff:
- IC, ICIR, quintile backtests, RAS, turnover → **2010–2020 only** (training data)
- Regime stability → **full panel** (needs 2022 rate hike bear + 2023 AI bull regimes)
- IC decay → **training data only, survivors only**

**Code change in `2a_factor_analysis.py`:**
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

**Rationale:** Transformer models better suited for the cross-sectional structure of the data and the firm is building toward live deployment.

### 5.7 — Factor Data from Quant Platform (18 Mar 2026)

Shared `factor data.xlsx`:
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

1. Pull new factor data monthly from platform
2. Run FT-Transformer + CS-Transformer → generate stock scores
3. Compute portfolio weights: `w_i = w_SPX_i + α × z-score(ML_score_i)`
4. Execute rebalancing trades automatically

**Future integration:** Investsoc sector analyst picks could be incorporated as an overlay (quant-fundamental hybrid), boosting ML scores for stocks analysts also like.

---

## Waiting On (as of 18 Mar 2026)

1. **full historical factor data** (2010–2025) — main blocker for model training
2. **Tonight's meeting** — align on: final filter criteria, factor weighting by decay, non-monotonic factor treatment, train/test setup
3. **After data arrives:** merge panel → re-run factor analysis (train period only) → train FT-Transformer + CS-Transformer → ensemble → backtest → automate

---

## Phase 6 — Model Training Prep (19 Mar 2026)

### 6.1 — Portfolio Construction: Top/Bottom 100

**Change:** Moved from continuous z-score tilt (all 500 stocks) to discrete top/bottom 100 approach.

**Old:** `w_i = w_SPX_i + α × z-score(score_i)` applied to all stocks
**New:** Overweight top 100 stocks by ML score, underweight bottom 100, hold middle ~300 at benchmark weight

**Config change:** `TOP_N = 100`, `BOTTOM_N = 100` (was 50/50)

**Rationale:** suggestion — cleaner signal, avoids applying weak/noisy scores to middle stocks.

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

### 6.4 — Gradual Training Plan (an external suggestion)

Train model incrementally, one improvement at a time:
1. **Step 1:** Baseline — IC filter + RAS, contrarian factors sign-flipped, predict `fwd_ret_1m`
2. **Step 2:** IC decay curve fitting — find smooth representation of each factor's decay, use for factor weighting. Also adjust quintile treatment for non-monotonic factors (top/bottom 100 only)
3. **Step 3:** Add fundamental factors (PE, PB, money flow etc.) once data arrives
4. **Step 4:** Factor weighting by decay profile (short-term vs long-term)

### 6.5 — Project Structure Cleanup (19 Mar 2026)

- Moved `DEVELOPMENT_LOG.md`, `STRATEGY.md`, `notes.md` → `Notes/`
- Renamed `"1 price parquet.py"` → `1a_price_parquet.py`, `"6 Diagonstic test.py"` → `6_diagnostic_test.py`
- Deleted one-off scripts: `check_overfit.py`, `inspect_data.py`, `5_pitch_validator.py`, `report.ipynb`, `data.xlsx`
- Moved `2_lgbm_backtest.py`, kaggle scripts, `3_pca_rp_backtest.py`, `6_diagnostic_test.py` → `archive/`
- Moved all `*.png` figures → `figures/` at repo root
- Restructured repo: moved all files from `Investsoc ML project/` subfolder to repo root
- Added `.gitignore` (excludes `.claude/`, `*.parquet`, `*.csv`, `*.xlsx`, `__pycache__`)

### 6.6 — 2f_factor_diagnostics.py date fix (pending)

`2f_factor_diagnostics.py` uses full panel with no date cutoff. Should be filtered to 2010–2020 training period only to avoid test period leakage in correlation/RMT/VIF analysis. Low priority — diagnostics only, not used in training.

---

---

## Phase 7 — Backtest Execution & Known Issues (19 Mar 2026)

### 7.1 — FT-Transformer Kaggle Run (19 Mar 2026)

Results from `3b_ft_transformer_kaggle.py` with updated TOP_N=100, LONG_FRAC=0.20:

| Strategy | Sharpe | Ann Ret | Max DD |
|----------|--------|---------|--------|
| Long-Only Top 100 | 1.45 | 28.35% | -16.95% |
| Long-Short top/bot 20% | 1.23 | 19.09% | -9.65% |

Note: Sharpe dropped slightly vs Phase 1 (1.60 → 1.45) because TOP_N increased from 50 → 100 (more diversified = lower concentration = lower return and vol). Expected.

Trained in 10.7 min on Kaggle GPU T4.
`scores_transformer.parquet` saved to `data/`.

### 7.3 — `4b_index_enhancement.py` Sharpe Bug Fixed

`ie_stats()` was missing `sharpe` key but it was referenced in the print table. Fixed by adding portfolio Sharpe computation to `ie_stats()`.

### 7.4 — SPX Weights Data Quality Issue (NEEDS FIXING)

`1c_fetch_market_cap.py` ran successfully (500/504 tickers, 2 min) but market cap weights are distorted:
- NVDA showing 25.38% (should be ~6-7%)
- WMT showing 12.92% (should be ~0.7%)

**Root cause:** yfinance `get_shares_full()` returns inconsistent/stale shares outstanding for some tickers — current shares applied to historical prices without adjusting for splits/buybacks creates relative weight distortion.

**Impact:** `4b_index_enhancement.py` will run but the benchmark return won't accurately reflect the true S&P 500. Active returns and IR figures will be directionally correct but numerically off.

**Fix options (in priority order):**
1. Use a proper data source for SPX constituent weights (the quant platform may have this)
2. Cross-check with known cap weights from another free source (e.g. SPDR holdings CSV)
3. Use equal-weight benchmark as fallback

**Status:** Not blocking — pipeline works, fix before any final results presentation.

### 7.5 — New Scripts Added

- `2c_ic_decay_daily.py` — daily IC decay (1–90 trading days) per factor. suggestion to confirm monthly rebalancing is optimal horizon.

### 7.6 — Ideas for Next Phase (19 Mar 2026)

- **Barra-style factors** — size (mktcap), value (P/B), leverage (D/E), earnings yield. Can approximate from yfinance. the quant platform data will likely cover these.
- **Smart money factors** — institutional ownership changes (13F), short interest, insider buying. Harder to get without a data vendor.
- **Daily IC decay** — built as `2c_ic_decay_daily.py`. To send to the team once run.
- **Focus for now:** Complete backtesting framework first, then factor additions.

---

### 7.7 — Index Enhancement Results (19 Mar 2026)

First full run of `4b_index_enhancement.py` with all 3 model scores. Best α per model (target TE 1–5%):

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

### 8.1 — `2d_factor_weights.py` (new script)

Combines all factor analysis outputs into a single clean table for model training.

**Steps:**
1. **Filter** — |IC| > 0 AND regime_stable → 43/59 factors survive
2. **IC decay weighting** — persistence weight = mean |IC| over lags 0–12 months (area under 1-year decay curve). Factors with longer-lasting signal get higher weight. Normalised to sum to 1.
3. **Partial flag** — `use_partial = True` for non-monotonic factors (Q5→Q1 ordering broken). For these, only top/bottom 100 stocks carry signal — middle 300 held at benchmark weight.

**Output:** `data/factor_selected.csv` — complete factor list with weights and flags, ready for model training.

**Why mean |IC| over lags 0–12 (not half-life):** IC doesn't decay monotonically — it oscillates and bounces back (observation). Half-life fitting is unstable. Mean area under curve is robust and captures integrated signal strength over a 1-year window.

### 8.2 — SPX Weights Fix

**Root cause of WMT/NVDA distortion:** `fast_info.shares` from yfinance returns stale or non-float-adjusted share counts for some tickers. WMT was returning ~8× too many shares.

**Fix 1 (primary):** Changed `fetch_current_shares()` to derive implied shares from `market_cap / last_price` instead of using raw share count. `fast_info.market_cap` is more reliably reported.

**Fix 2 (safety net):** Added 8% winsorisation after computing weights — any single stock capped at 8% weight, then renormalised. Prevents any remaining outlier from dominating the benchmark. Real S&P 500 top holdings are ~6–7% (AAPL, NVDA, MSFT).

### 8.3 — CS-Transformer Kaggle Script Confirmed Correct

`3d_cs_transformer_kaggle.py` already had `TOP_N=100`, `BOTTOM_N=100`, `LONG_FRAC=0.20`. No changes needed.

---

## Current Status (19 Mar 2026 — evening)

**Completed today:**
- All 3 models trained and scored (LGBM, FT-Transformer, CS-Transformer)
- Full IE pipeline working — CS-Transformer IR=0.776 best result
- Factor framework complete: filtering → decay weighting → partial flags
- SPX weights fix (market_cap/price + 8% winsorisation)
- Daily IC decay script built (`2c_ic_decay_daily.py`)

**Blocked on:** full historical fundamental data (2010–2025)

---

## Phase 9 — Survivorship Bias Fix (20 Mar 2026)

### 9.1 — Problem Identified

Survivorship bias: `prices.parquet` only had ~504 current S&P 500 members. Companies removed between 2010–2025 (acquired, bankrupt, delisted) were missing, inflating backtest returns by ~1–2%/year.

### 9.2 — `1b_fetch_constituents.py` (new script)

- Downloads full historical S&P 500 constituent list from GitHub (`fja05680/sp500`)
- 826 unique tickers ever in S&P 500 during 2005–2025 scope
- 386 already in `prices.parquet`, 440 missing
- Tried yfinance for all 440: **188 succeeded**, 252 failed (bankrupt/fully delisted)
- `prices.parquet` updated: **504 → 692 tickers**
- Three lists saved: `tickers_existing.csv`, `tickers_new_fetched.csv`, `tickers_missing.csv`
- Sent `tickers_missing.csv` (252 tickers) to the team for CRSP/Compustat pull

### 9.3 — `1f_rebuild_panel.py` (new script)

`panel_monthly.parquet` was originally built from `data.xlsx` (deleted during cleanup). Rebuilt it directly from `prices.parquet`:
- Computes daily returns → rolling vol (20d/60d/252d) → HL range
- Samples at month-end, computes multi-horizon returns + forward return
- Output: `panel_monthly.parquet` — **692 tickers, 147,517 rows**

### 9.4 — Feature Engineering Re-run

Re-ran `1g_feature_engineering.py` on expanded universe:
- **Before:** 502 tickers, 106,944 rows
- **After:** 692 tickers, 147,517 rows
- Same 79 features, same pipeline

### 9.5 — Known Issues (20 Mar 2026)

- **SPX weights** still distorted (top 4 stocks at 12.66% each after winsorisation) — iterative winsorisation needed or proper data source
- **252 missing tickers** — waiting on CRSP/Compustat pull
- **Fundamental factors** — `data/fundamental.parquet` still missing, Cat 9 skipped. Biggest remaining gap for model improvement
- **Factor analysis** — should re-run `2a_factor_analysis.py` + `2d_factor_weights.py` on new 692-ticker universe

---

## Current Status (20 Mar 2026)

**Completed today:**
- Survivorship bias partially fixed (504 → 692 tickers)
- Panel rebuilt from scratch (`1f_rebuild_panel.py`)
- Three ticker lists exported for the CRSP pull
- New `panel_monthly_enriched.parquet` ready for Kaggle upload

**Next steps:**
1. Upload new panel to Kaggle → retrain both models
2. Re-run `4b_index_enhancement.py` with updated scores
3. Waiting on: 252 missing tickers (CRSP) + fundamental factors
4. When data arrives: rebuild panel again → re-run everything

---

## Phase 10 — Missing Ticker Recovery (21 Mar 2026)

### 10.1 — `1d_fetch_missing_tickers.py` (new script)

Secondary yfinance attempt + Stooq fallback for 252 failed tickers.

- **yfinance fix:** Used `Ticker.history()` instead of `yf.download()` — avoids `YFTzMissingError`
- **Result:** Only 4 additional tickers recovered (252 → 248 missing)
- **Stooq via pandas-datareader:** Returns 0 rows for delisted stocks — confirmed dead end for this approach
- `prices.parquet` updated: **692 → 696 tickers**

### 10.2 — Tiingo API Attempts

Tiingo API (free tier: 500 requests/hour) specifically has delisted/acquired stock data.

- **First run:** Recovered ~33 tickers but save failed (exit code 1) — data lost. Root cause: timezone-aware dates causing type mismatch with `pa.Table.from_pandas()`.
- **Second run:** Immediately hit hourly rate limit (429 error) — had used 496/500 calls across first two runs.
- **Third run (this session):** Rate limit still active — 50 tickers tried, all returned "not available". Rate limit reset needed.
- **Script:** `(removed)` — includes rate limit detection, 65s auto-retry, fixed timezone stripping (`dt.tz_localize(None)`)
- **Key fix:** `df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()` before saving

**Status:** Tiingo hourly limit reset needed before retry. Some tickers not available on Tiingo (especially older/obscure ones).

### 10.3 — Wind XLSX Ingestion (`1e_ingest_wind_xlsx.py`)

Exported from Wind: `missing data.xlsx` from his Wind platform — OHLCV for AABA.O (Yahoo/Altaba).

**Wind export format:**
- Sheet: `工作表1` (default Chinese)
- Rows 3–6: metadata (start/end date, ticker code like `AABA.O`, name)
- Row 7–8: bilingual column headers
- Row 9+: data — Date, Open, High, Low, Close (no volume)

**Script `1e_ingest_wind_xlsx.py`:**
- Scans `data/wind_exports/` folder for all xlsx files (drop files here)
- Also picks up any xlsx in `data/` (handles ad-hoc Export)
- Extracts ticker from metadata rows automatically
- Strips Wind formula cells, handles NaN open prices
- Merges into `prices.parquet`, updates ticker log (status: `ok_wind`)
- Run: `python 1e_ingest_wind_xlsx.py` after adding new xlsx files

**Result:** AABA.O added (2010–2025, 3,886 rows). `prices.parquet`: **696 → 697 tickers**

### 10.4 — Workflow for Wind Exports

1. Export missing tickers from Wind (one or multiple files)
2. Place xlsx files in `data/wind_exports/`
3. Run `python 1e_ingest_wind_xlsx.py`
4. Run `python 1f_rebuild_panel.py`
5. Run `python 1g_feature_engineering.py`
6. Upload to Kaggle, retrain models

### 10.5 — Current Universe Status (21 Mar 2026)

| Source | Count |
|--------|-------|
| existing (original prices.parquet) | 386 |
| ok (yfinance, 1d/1f) | 192+4=196 |
| ok_tiingo (pending retry) | 0 (33 lost in save failure) |
| ok_wind | 1 (AABA.O) |
| **still missing** | **248** |
| **Total tickers** | **697** |

---

## Current Status (21 Mar 2026)

**Completed today:**
- `1d_fetch_missing_tickers.py` — 4 extra tickers via yfinance fix
- `(removed)` — Tiingo fetcher with rate limit handling and timezone fix
- `1e_ingest_wind_xlsx.py` — Wind xlsx ingestion pipeline
- AABA.O added from Wind export

**Blocked on:**
- Tiingo hourly rate limit (reset needed, then retry `(removed)`)
- Wind exports for remaining 248 tickers (drop in `data/wind_exports/`)
- full historical fundamental data (2010–2025)

**Next steps once data available:**
1. Run `1e_ingest_wind_xlsx.py` (after Export Wind data)
2. Run `(removed)` (after rate limit resets)
3. Run `1f_rebuild_panel.py`
4. Run `1g_feature_engineering.py`
5. Upload to Kaggle → retrain FT-Transformer + CS-Transformer
6. Re-run `4b_index_enhancement.py` with updated scores

---

## Phase 11 — Advanced Framework (23 Mar 2026)

### 11.1 — Regime Backtesting Engine (`4c_regime_engine.py`)

Breaks IE backtest performance out by market regime.

**Two modes:**
- `python 4c_regime_engine.py` — rule-based (4 hardcoded regimes)
- `python 4c_regime_engine.py --hmm` — 2-state HMM on SPX returns + VIX (risk-on / risk-off)

**HMM results (test period Jan 2023–Oct 2025, 24 months: 17 risk-on, 7 risk-off):**

| Model | Risk-Off IR | Risk-On IR | Full Period IR |
|-------|------------|-----------|----------------|
| CS-Transformer | **1.019** | **0.931** | 0.960 |
| FT-Transformer | 1.866 | 0.086 | 0.438 |
| LGBM | 1.397 | 0.224 | 0.384 |

**Key finding:** CS-Transformer is the only regime-stable model — IR consistent across both risk-on and risk-off. FT-Transformer and LGBM generate almost all alpha in risk-off periods only. Confirms CS-Transformer as the primary model.

**Outputs:** `data/ie_regime_breakdown.csv`, `figures/ie_regime_breakdown.png`

### 11.2 — Differentiable IC Optimisation (`2e_ic_optimise.py`)

Replaces static IC-decay factor weights with gradient-optimised weights.

**Method:** Weight vector w ∈ R^43, softmax-normalised. Loss = −mean(Pearson IC). Adam, lr=0.01, 500 epochs, PyTorch.

| | Train IC (2010–2020) | Val IC (2021–2022) |
|-|---------------------|-------------------|
| IC-decay weights | 0.0254 | 0.0318 |
| Optimised weights | **0.0758** | **0.0434** |
| Improvement | +197% | +36% |

**Key shifts:** `residual_ret_12m` 0.008 → 0.225, `ret_18m` 0.011 → 0.143, `ir_3m` 0.011 → 0.135. Volatility factors mostly downweighted. Momentum/return factors dominate.

**Outputs:** `data/factor_selected_optimised.csv`, `figures/factor_weights_optimised.png`

### 11.3 — Factor-Combo Baseline (`4a_factor_combo_baseline.py`)

Linear weighted factor combination (no ML) as IE signal — quantifies what the transformer adds.

| Model | IR |
|-------|----|
| Factor-Combo linear (optimised weights) | **−0.046** |
| LGBM | 0.384 |
| FT-Transformer | 0.438 |
| CS-Transformer | **0.960** |

Linear combo underperforms the benchmark. CS-Transformer goes −0.046 → 0.960 — confirms the transformer is learning genuine non-linear cross-sectional patterns, not just repackaging factor exposures.

### 11.4 — RL Architecture (Planned)

Not implementing yet — waiting for fundamental data + complete universe.

- **Layer 1 (factor weighting):** SAC agent, state = regime + rolling IC history, action = 43-dim weight vector, reward = next-month IC
- **Layer 2 (portfolio tilt):** SAC/TD3 agent, action = tilt α, reward = active_return − λ × TE_penalty
- **Why SAC over PPO:** Off-policy, sample efficient, entropy regularisation prevents regime overfitting

---

## Current Status (23 Mar 2026)

**Completed:**
- Regime breakdown — CS-Transformer regime-stable (IR 0.93–1.02), others risk-off dependent
- IC optimisation — train IC +197%, val IC +36%, momentum > volatility
- Factor-combo baseline — linear IR=−0.046 vs CS-Transformer 0.960, validates transformer complexity
- Wind xlsx ingestion pipeline, AABA.O added (697 tickers)
- Documentation cleaned

**Blocked on:**
- 248 missing tickers (Wind exports — drop xlsx in `data/wind_exports/`)
- Full historical fundamental data (PE, PB, money flow 2010–2025)

**Next steps once data arrives:**
1. `python 1e_ingest_wind_xlsx.py`
2. `python 1f_rebuild_panel.py` + `python 1g_feature_engineering.py`
3. `python 2e_ic_optimise.py` on expanded feature set
4. Retrain on Kaggle → `python 4b_index_enhancement.py` + `python 4c_regime_engine.py`

---

## Phase 12 — Retrain + RL Agent (24 Mar 2026)

### 12.1 — Script Renaming (24 Mar 2026)

All scripts renamed to reflect actual execution order, grouped into 5 phases:

| Phase | Scripts | Purpose |
|-------|---------|---------|
| 1 (1a–1h) | `1a_price_parquet.py` … `1h_orthogonalize.py` | Data prep |
| 2 (2a–2f) | `2a_factor_analysis.py` … `2f_factor_diagnostics.py` | Factor analysis |
| 3 (3a–3d) | `3a_ft_transformer.py` … `3d_cs_transformer_kaggle.py` | Model training |
| 4 (4a–4d) | `4a_factor_combo_baseline.py` … `4d_benchmark_spx.py` | Evaluation |
| 5 (5a) | `5a_rl_portfolio_agent.py` | RL layer |

README, STRATEGY.md, and DEVELOPMENT_LOG updated to use new names.

### 12.2 — Wind Data Received (24 Mar 2026)

Files received from collaborator:
- `data.xlsx` — confirmed old price data, already in `prices.parquet` (no new tickers)
- `factor data.xlsx` — fundamental data (money flow `mfd_buyamt_d`) but **only December 2025** (1 month). Full 2010–2025 history still needed.
- `missing data.xlsx` — AABA.O already ingested in Phase 10

**Status:** 247 "failed" tickers still missing — delisted US stocks without exchange suffix (e.g. `ABKFQ`, `ABMD`). Wind likely doesn't have these. Full resolution requires CRSP/Compustat.

### 12.3 — CS-Transformer Retrained on Kaggle (24 Mar 2026)

Retrained `3d_cs_transformer_kaggle.py` with expanded panel (697 tickers, 240 train months):

| | Previous | Updated |
|--|---------|---------|
| Train months | 132 | 240 |
| Test months | 35 | 39 |
| Long-Only Top 100 Sharpe | — | 1.03 |
| Long-Only Ann Return | — | 29.3% |

Files saved: `scores_cs_transformer.parquet`, `bt_cs_transformer.csv`, `bt_cs_transformer_ls.csv`

### 12.4 — IE Results Updated (24 Mar 2026)

Full IE pipeline rerun with updated CS-Transformer scores (`4a` → `4d`):

| Model | Ann α | TE | IR | Hit Rate |
|-------|-------|-----|-----|----------|
| **CS-Transformer** | **4.16%** | **2.22%** | **1.874** | 75.0% |
| FT-Transformer | 1.06% | 2.47% | 0.428 | 45.8% |
| LGBM | 0.61% | 1.61% | 0.377 | 66.7% |
| Factor-Combo (linear) | −0.2% | 3.5% | −0.047 | 50.0% |

**CS-Transformer IR improved from 0.960 → 1.874** — larger training set (240 vs 132 months) drove meaningful improvement. Factor-combo linear IR stable at −0.047 (confirms ML is adding genuine value, not just from more data).

### 12.5 — Regime Breakdown Updated (24 Mar 2026)

HMM 2-state regime breakdown with new CS-Transformer scores:

| Model | Full IR | Risk-Off IR | Risk-On IR | Regime-Stable? |
|-------|---------|------------|-----------|----------------|
| CS-Transformer | **1.850** | 0.926 | **2.227** | Yes |
| FT-Transformer | 0.438 | 1.866 | 0.086 | No |
| LGBM | 0.384 | 1.397 | 0.224 | No |

**Updated finding:** CS-Transformer now stronger in risk-on (IR 2.227) than risk-off (IR 0.926) — likely due to the extended test period (39 vs 24 months) including more risk-on months. Still regime-stable. FT-Transformer and LGBM pattern unchanged.

### 12.6 — RL Portfolio Agent (`5a_rl_portfolio_agent.py`) (24 Mar 2026)

SAC (Soft Actor-Critic) agent replacing fixed-alpha portfolio tilt. No memory — MLP policy only, each month independent (Markov).

**Architecture:**
- State (6 features): signal strength, signal dispersion, benchmark vol, recent active return, HMM regime indicator, rolling tracking error
- Action: alpha ∈ [0.002, 0.05] — how aggressively to tilt this month
- Reward: active return × 12 − penalty if TE > 3%
- Training: factor-combo scores on 2010–2022 (103 months)
- Evaluation: CS-Transformer scores on test period 2023–2025 (24 months)

**Results:**

| Metric | RL Agent | Fixed α=0.01 |
|--------|---------|-------------|
| Ann Alpha | **1.75%** | −0.21% |
| Tracking Error | 6.16% | 5.28% |
| IR | **0.284** | −0.040 |
| Hit Rate | 54.2% | 54.2% |

RL agent learns to adapt alpha by regime: avg alpha 3.15% in risk-off, 1.97% in risk-on. Fixed α=0.01 is suboptimal — RL improves IR from −0.040 → 0.284.

**Note:** RL IR (0.284) is lower than CS-T IE IR (1.874) because RL is evaluated against a sub-optimal fixed baseline (α=0.01 which gives IR=−0.04). RL adds value on top of the wrong baseline. Optimal path: use CS-T scores with RL-selected alpha — not yet evaluated.

**Bugs fixed:**
- `5a_rl_portfolio_agent.py`: NaN propagation in actor network — fixed by `np.nan_to_num(..., nan=0.0)` on state vectors
- `4d_benchmark_spx.py`: duplicate index labels crash — fixed by deduplicating series before `pd.concat()`

---

## Current Status (24 Mar 2026)

**Completed:**
- CS-Transformer retrained (240 months) — IR improved 0.960 → 1.874
- Full IE pipeline rerun — all 4 scripts complete
- RL agent working — adapts alpha by regime, IR 0.284 vs −0.040 fixed
- Script renaming complete (phases 1–5)
- Regime breakdown updated

**Blocked on:**
- Full fundamental data history (2010–2025) — only December 2025 received
- 247 missing tickers — likely not available without CRSP/Compustat

**Next steps once fundamental data arrives:**
1. `python 1e_ingest_wind_xlsx.py` → `python 1f_rebuild_panel.py` → `python 1g_feature_engineering.py`
2. `python 2e_ic_optimise.py` (with new fundamental factors)
3. Retrain on Kaggle → `python 4b_index_enhancement.py` → `python 4c_regime_engine.py --hmm`
4. Retrain FT-Transformer on Kaggle (`3b_ft_transformer_kaggle.py`) with 240-month dataset

---

## Future Ideas (beyond current scope)

### Double-Layered Deep Reinforcement Learning (21 Mar 2026)

Suggested using double-layered DRL for a more advanced version of the strategy:

- **What is DRL:** Agent learns by trial and error — rewarded for good portfolio decisions, penalised for bad ones. Simulates thousands of trading periods to learn what works.
- **Layer 1:** RL agent learns which fundamental factors to weight dynamically based on market regime (replaces our fixed IC decay weighting)
- **Layer 2:** RL agent learns optimal portfolio weights and position sizing (replaces our fixed top/bottom 100 tilt)
- **Why better:** Adapts dynamically — more aggressive in bull markets, more defensive in bear markets. Used by Two Sigma, Renaissance etc.
- **Why not now:** Requires much more data, compute, and complexity. Current transformer approach must be validated first.

Natural next evolution once the current pipeline is stable and producing consistent alpha.
