# Validation Protocol - Factor Mining Anti-Leakage Framework

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
- OOS evaluation happens ONCE. If results are poor, we document that - we do NOT go back and re-select.
- Any candidate that uses future data in its construction (e.g., forward-looking normalization windows) is rejected.

---

## 2. IS Screening Pipeline

Each candidate factor passes through this sequential funnel:

### Stage 1: Basic IS IC
- Compute monthly Spearman rank-IC: `corr(rank(factor), rank(fwd_ret_1m))` per month
- Restrict to IS months (≤ 2022-12-31)
- Threshold: `|mean(IC)| > 0.015` - **diagnostic only, not a gate**. Rationale: weekly rebalancing makes noisy factors acceptable

### Stage 2: IS ICIR
- ICIR = mean(IC) / std(IC)
- Threshold: `|ICIR| > 0.40` - **diagnostic only, not a gate** (same reasoning)

### Stage 3: IC Persistence
- Compute IC at lags 1, 2, 3, 6 months
- Threshold: `|IC_lag1| > 0.5 × |IC_lag0|` - **diagnostic only, not a gate**

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
- **Pass/fail gates (binding):** BHY correction + dedup only
- **Diagnostic metrics (logged, not gates):** IC, ICIR, persistence, RAS p-value
- Compute additional IS diagnostics for survivors: turnover, sector neutrality, regime stability

> **Design decision (7 Apr 2026):** Weekly rebalancing tolerates noisier factors, so IC/ICIR thresholds are dropped as hard gates. BHY controls false discovery across the whole candidate set; dedup prevents redundancy. All features are fed to the ML screens with L1/L2 penalties - the model handles selection, not a manual pre-filter.

---

## 3. Leakage Risk Checklist

| Risk | Mitigation |
|------|-----------|
| **Rolling window lookahead** | All rolling computations use `.shift(0)` or equivalent - window includes current day but never future days. Verified: pandas `.rolling()` is backward-looking by default. |
| **Feature normalization lookahead** | Cross-sectional ranking uses only data available at each month-end. Time-series z-scoring uses expanding or rolling windows fit on past data only. |
| **Survivorship bias** | Panel includes delisted tickers via `1d_fetch_missing_tickers.py`. 247 still missing - documented as limitation. |
| **Look-ahead from target** | `fwd_ret_1m` is the forward return - never used in feature computation. Only used as the target variable for IC computation. |
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
2. Compare IS vs OOS metrics - document degradation
3. Expected: some IC decay is normal; reversed sign is a red flag
4. Do NOT re-select based on OOS results
5. Document everything in `research/factor_mining/reports/final_evaluation.md`

---

## 6. Converting Factors to Time Signals

Design constraint: time-signal factors must be per-stock binary or directional.

**Conversion rules:**
- **Binary (0/1):** Factor crosses a threshold calibrated on IS data only. Example: `vol_regime_high = 1 if entropy < IS_75th_percentile`
- **Directional (-1, 0, +1):** Sign of the factor value, or sign of factor change. Example: `trend_eff_signal = sign(efficiency_ratio - 0.5)`
- **Thresholds:** All thresholds computed from IS distribution only. Applied to OOS without refitting.
- **No cross-sectional ranking at this stage:** The signal says "is THIS stock in state X?" not "is this stock ranked higher than peers?"

---

## 7. ML Screening (No Pre-Filter)

No manual filtering before ML. All features (existing 205 + new candidates) are fed to penalised models:

1. **Lasso (L1)** - drives irrelevant feature coefficients to exactly zero. Purged 5-fold time-series CV.
2. **Random Forest** - permutation importance (not impurity-based, to avoid bias).
3. **LightGBM** - built-in L1+L2 regularisation. Gain-based importance.

Union of survivors from all three screens → BHY + dedup validation.

The CS-Transformer also has built-in feature selection via `FeatureTokenizer` L1/L2 penalties (`l1_lambda`, `l2_lambda` in `TRANSFORMER_CS_PARAMS`). This learns to zero out useless feature embeddings during training.

## 8. Two-Stage CS-Transformer Training

1. **MSE pre-train** - standard masked MSE + FeatureTokenizer L1/L2 penalty. Learns feature interactions and stock embeddings.
2. **RL fine-tune (GRPO/DAPO)** - portfolio-level reward. Differentiable sigmoid top-K allocation. Group-relative advantage with G=4-8 samples. Three methods:
   - GRPO: symmetric PPO clipping + KL penalty vs frozen reference (conservative)
   - DAPO: asymmetric clipping (ε_low=0.20, ε_high=0.28), dynamic G, no KL (aggressive)
   - Hybrid: regime-aware switching

Validation during RL uses Rank IC (Spearman correlation of predicted vs actual ranks) - no return leakage.

## 9. Implementation Notes

- All validation functions live in `research/factor_mining/validation_engine.py`
- Screening results saved to `research/factor_mining/results/`
- Candidate catalog: `research/factor_mining/candidate_factor_catalog.csv`
- Each candidate gets a row with: factor_name, track, family, IC_IS, ICIR_IS, IC_persistence, dedup_pass, RAS_pvalue, BHY_adjusted_pvalue, selected

## 10. Macro FiLM Conditioning

The CS-Transformer now separates macro features (16 `MACRO_COLS`: VIX, yields, spreads, SPX stats, prediction market signals) from stock features:

- **Stock features** → FeatureTokenizer → per-feature d_model embeddings
- **Macro features** → MacroEncoder MLP → 64-dim macro embedding → MacroFiLMLayer
- FiLM modulation: `tokens_out = gamma * tokens + beta` (per-feature gamma/beta generated from macro state)
- Identity-initialized (gamma=1, beta=0) - backward-compatible with existing pre-trained weights
- Creates implicit ~247 stock × 16 macro = ~3,950 feature-macro interactions

The design goal is to treat macro indicators as a separate conditioning layer rather than packing them into the cross-sectional feature vector: the model learns which stock factors to trust or distrust under each macro regime.

## 11. Factor Correlation Awareness

Optional `CorrelationAttentionBias` injects the pre-computed F×F Spearman correlation matrix as additive attention bias in Stage 1 feature attention. Each attention head learns a scalar weight on the correlation matrix.

- Correlation matrix computed from IS training data only (no future leakage)
- Recomputed at each walk-forward retrain point
- Disabled by default (`use_corr_bias=False`); enable after FiLM is validated
