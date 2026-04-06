# Validation Protocol — Factor Mining Anti-Leakage Framework

**Date:** 2026-04-06
**Purpose:** Prevent lookahead bias, data snooping, and multiple-testing inflation in the factor mining exercise.

---

## 1. Temporal Split (Strict Chronology)

| Period | Date Range | Use |
|--------|-----------|-----|
| **IS (Discovery)** | 2010-01-01 to 2022-12-31 | ALL factor generation, screening, selection, threshold tuning |
| **OOS (Frozen Final)** | 2023-01-01 onward | ONE-TIME evaluation after pipeline is completely frozen |

**Rules:**
- No OOS IC, OOS ICIR, or any OOS metric may influence which factors are kept or dropped.
- OOS evaluation happens ONCE. If results are poor, we document that — we do NOT go back and re-select.
- Any candidate that uses future data in its construction (e.g., forward-looking normalization windows) is rejected.

---

## 2. IS Screening Pipeline

Each candidate factor passes through this sequential funnel:

### Stage 1: Basic IS IC
- Compute monthly Spearman rank-IC: `corr(rank(factor), rank(fwd_ret_1m))` per month
- Restrict to IS months (≤ 2022-12-31)
- Require: `|mean(IC)| > 0.015` (weak but nonzero signal)

### Stage 2: IS ICIR
- ICIR = mean(IC) / std(IC)
- Require: `|ICIR| > 0.40` (consistent signal, not just one lucky month)

### Stage 3: IC Persistence
- Compute IC at lags 1, 2, 3, 6 months
- Require: `|IC_lag1| > 0.5 × |IC_lag0|` (signal doesn't instantly decay)

### Stage 4: Deduplication
- Compute cross-sectional correlation of candidate vs ALL existing 205 features + all other candidates, within each month
- Average across IS months
- Reject if `|mean_corr| > 0.85` with any existing feature (keep higher ICIR)

### Stage 5: RAS Test (Random Alpha Simulation)
- Generate 1,000 random "factors" by shuffling the actual factor values within each cross-section (preserving the cross-sectional structure but destroying any real signal)
- Compute IS ICIR for each random factor
- Candidate passes if its IS ICIR exceeds the **95th percentile** of the random distribution
- This tests: "is this factor's ICIR distinguishable from what you'd get by chance?"

### Stage 6: BHY Multiple Testing Correction
- After running all candidates through Stages 1-5, collect their RAS p-values
- Apply Benjamini-Hochberg-Yekutieli (BHY) correction at FDR = 5%
- BHY is preferred over Bonferroni because:
  - Candidate factors are correlated (Bonferroni is too conservative)
  - BHY controls FDR under arbitrary dependence
  - With ~100 candidates, Bonferroni would reject almost everything

### Stage 7: Final IS Validation
- Survivors must pass ALL of: IC threshold, ICIR threshold, persistence, dedup, RAS, BHY
- Compute additional IS diagnostics for survivors: turnover, sector neutrality, regime stability

---

## 3. Leakage Risk Checklist

| Risk | Mitigation |
|------|-----------|
| **Rolling window lookahead** | All rolling computations use `.shift(0)` or equivalent — window includes current day but never future days. Verified: pandas `.rolling()` is backward-looking by default. |
| **Feature normalization lookahead** | Cross-sectional ranking uses only data available at each month-end. Time-series z-scoring uses expanding or rolling windows fit on past data only. |
| **Survivorship bias** | Panel includes delisted tickers via `1d_fetch_missing_tickers.py`. 247 still missing — documented as limitation. |
| **Look-ahead from target** | `fwd_ret_1m` is the forward return — never used in feature computation. Only used as the target variable for IC computation. |
| **IS/OOS contamination** | `TRAIN_END = "2022-12-31"` hard-coded. All screening functions accept `end_date` parameter and filter strictly. |
| **Multiple testing** | BHY correction applied after all candidates evaluated. RAS test provides per-candidate calibration. |

---

## 4. Robustness Checks (IS-only)

For each surviving factor, compute:

1. **Sub-period stability:** Split IS into 2010-2016 and 2017-2022. Require consistent IC sign in both halves.
2. **Sector neutrality:** Compute IC within GICS sectors (if available) or within size quintiles. Check signal isn't driven by one sector.
3. **Turnover:** Monthly turnover = fraction of stocks that change quintile. High turnover (>60%) = signal is noisy/unstable.
4. **Regime conditioning:** Split IS by VIX regime (above/below median). Check IC sign stability.

---

## 5. OOS Evaluation (ONE-TIME, After Freezing)

After the entire discovery pipeline is frozen (no more additions or removals):

1. Compute OOS IC, ICIR for all survivors
2. Compare IS vs OOS metrics — document degradation
3. Expected: some IC decay is normal; reversed sign is a red flag
4. Do NOT re-select based on OOS results
5. Document everything in `research/factor_mining/reports/final_evaluation.md`

---

## 6. Converting Factors to Time Signals

Per Kieran's constraint: time signal factors must be per-stock binary or directional.

**Conversion rules:**
- **Binary (0/1):** Factor crosses a threshold calibrated on IS data only. Example: `vol_regime_high = 1 if entropy < IS_75th_percentile`
- **Directional (-1, 0, +1):** Sign of the factor value, or sign of factor change. Example: `trend_eff_signal = sign(efficiency_ratio - 0.5)`
- **Thresholds:** All thresholds computed from IS distribution only. Applied to OOS without refitting.
- **No cross-sectional ranking at this stage:** The signal says "is THIS stock in state X?" not "is this stock ranked higher than peers?"

---

## 7. Implementation Notes

- All validation functions live in `research/factor_mining/validation_engine.py`
- Screening results saved to `research/factor_mining/results/`
- Candidate catalog: `research/factor_mining/candidate_factor_catalog.csv`
- Each candidate gets a row with: factor_name, track, family, IC_IS, ICIR_IS, IC_persistence, dedup_pass, RAS_pvalue, BHY_adjusted_pvalue, selected
