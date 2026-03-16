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
