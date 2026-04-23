## Notes from original model
---

## Assumptions & Limitations

| Assumption | Detail |
|---|---|
| **No transaction costs** | All results assume month-end close prices with zero slippage or commissions. A rough sensitivity check on the LGBM strategy (~30–40% monthly turnover) suggests real-world impact of ~0.05–0.15 Sharpe reduction at 5–10 bps per trade. |
| **Survivorship bias** | Universe is a fixed historical S&P 500 list. Stocks that were removed from the index before our data starts (e.g. companies that went bankrupt pre-2010) are not included, introducing mild upward bias. Stocks removed during the sample period are included up to their removal date. |
| **No look-ahead bias** | Enforced by walk-forward design: test months (2023–2025) are never seen during training or validation. Each feature uses only data available at the time of prediction. The forward return (`fwd_ret_1m`) is the label — it is never a feature. |
| **Equal weighting** | Long-only portfolio uses 2% per stock (50 stocks). Equal-weighting avoids portfolio optimisation over-fitting; any optimised weights would require a further layer of cross-validation to avoid data snooping. |
| **Monthly rebalancing** | Assumes full portfolio rebalance at month-end close. Intra-month price moves are ignored. |
| **2010 start date** | Excludes 2008–2009 GFC regime. The GFC had structurally different correlations and liquidity dynamics compared to post-crisis markets. Starting in 2010 gives a cleaner, more stationary training distribution. |
| **Cross-sectional rank normalisation** | All 26 features are ranked within each month and scaled to [−0.5, +0.5]. This removes the absolute level of each feature (e.g. whether 6-month momentum is 5% or 50% in absolute terms) and makes signals stable across different market conditions. A stock ranked in the top decile on momentum always maps to the same input value regardless of what year it is. |
| **No short-selling constraints** | Long-Short results assume you can freely short any stock in the bottom decile. In practice, borrow costs and availability vary. |

---

## Known Issues / What's Next

### 1. PCA-RP Beta Problem
The PCA Risk Parity weighting produces beta = 2.07 (twice the market exposure). This is a bug in the weight construction — the first PCA component always captures the market factor, and back-projecting to asset space amplifies it. **Fix:** replace with iterative equal-risk-contribution (ERC) weighting.

### 2. Validation Period (2022 Bear Market)
Early stopping on the old fixed 2022 validation split hit at just 4 trees because 2022 was a pure bear market — nothing looked good on the validation set. The current pipeline uses a full walk-forward retrain which avoids this. Not a critical issue, but the validation split could still be improved (e.g. a rolling validation window rather than a fixed one).

### 3. No Transaction Costs
All results assume zero slippage and zero commissions. Monthly turnover for the LGBM Long-Only strategy is approximately 30–40% per month (stocks entering and leaving the top 50). At 10 bps per trade, this reduces the Sharpe by approximately 0.1–0.15.

### 4. No Fundamental Data
All 26 features are price-derived. Adding fundamental signals (P/E ratio, earnings growth, quality metrics) could improve predictive power, particularly for the long-short strategy.

### 5. Regime Stability Filter Uses Full Panel (Minor Look-Ahead)
In `2a_factor_analysis.py`, the regime stability check (which decides whether a factor survives into the final 43) is run on the full panel including the test period (2023–2025 AI bull regime). The IC calculation, quintile tests, and all model training are strictly train-only (≤ 2020), but the regime filter uses data it should not technically see.

**Impact:** Factors that happen to have consistent IC in the 2023–2025 test period are slightly more likely to pass the stability filter. However, a factor must pass in ≥ 3 of 4 regimes — it cannot pass on AI bull data alone. The practical effect on results is small.

**Why it exists:** The AI bull regime (2023–present) is one of four named regimes. Without using 2023+ data there would only be 3 regimes to test against, making the filter less meaningful.

**Fix:** Use only regimes that fall entirely within the training period (pre-2021) for factor selection. Add a separate post-hoc check on test-period regime stability as a diagnostic only.

### 6. 247 Missing Historical Tickers
247 S&P 500 historical constituents could not be fetched from yfinance or Tiingo (mostly delisted/acquired companies without a clean exchange suffix). These are excluded from the universe, introducing mild survivorship bias. Wind platform does not appear to have these tickers either. Full resolution would require CRSP or Compustat access.

### 7. Fundamental Data Limited to One Month
The `factor data.xlsx` export from Wind only covered December 2025 (1 month). All 43 current factors are price-derived. Adding fundamental signals (PE, PB, money flow, turnover rate) covering the full 2010–2025 history would likely improve IC meaningfully — fundamental factors tend to score higher ICIR than pure price signals. Blocked on collaborator providing full historical export.

### 8. RL Agent Trained on Factor-Combo Scores (Not CS-Transformer)
The SAC agent (`5b_rl_portfolio_agent.py`) is trained on factor-combo scores (2010–2022) but evaluated using CS-Transformer scores (2023–2025). This creates a distribution mismatch — the agent learned alpha dynamics from a weaker signal source. Once more out-of-sample CS-Transformer test months accumulate, the agent should be retrained directly on CS-Transformer scores for a cleaner evaluation.

### 9. RL Test Period Too Short for Reliable IR
The RL agent is evaluated on 24 test months (2023–2025). IR of 0.284 on 24 months has a wide confidence interval (~±0.4). Cannot reliably distinguish skill from luck at this sample size. As more test months accumulate, the estimate will stabilise.

### 10. Layer 1 and Layer 2 RL Not Yet Connected End-to-End
Layer 1 (5b — factor weighting, objective: maximise IC/ICIR) and Layer 2 (5a — portfolio tilt, objective: maximise portfolio Sharpe) are currently trained and evaluated independently. The objectives are deliberately non-conflicting: L1 optimises signal quality, L2 optimises how aggressively to act on that signal. The full pipeline should feed L1's recent IC directly into L2's state vector so L2 can trust the signal more when L1 has high recent IC. Requires joint or sequential training once more CS-T test months accumulate.

### 11. FT-Transformer Dropped from IE
After retraining on 240 months, FT-Transformer IE IR collapsed to 0.015. It is no longer useful for index enhancement. Only CS-Transformer and LGBM are viable IE models. An ensemble of CS-T + LGBM has not yet been evaluated and could improve IR further.

### 12. DAPO vs GRPO — Unfair Candidate Count (G) Comparison ✓ Fixed
~~In `5e_dapo_agent.py`, GRPO always samples `G=4` candidates per state, but DAPO samples `G_MIN=2` in easy states.~~

**Fixed (29 Mar 2026):** Set `DAPO_G_MIN = DAPO_G_INIT = 4` to match GRPO. DAPO now uses 4 candidates in easy states and up to 8 in hard states. Rerun `5e_dapo_agent.py` to get updated numbers.

### 13. DAPO Dynamic Sampling Variance Trigger is Statistically Unreliable ✓ Fixed
~~The dynamic G decision was based on the variance of only `G_INIT=2` reward samples.~~

**Fixed (29 Mar 2026):** `G_INIT` raised to 4 so the variance estimate uses ≥4 samples before deciding to extend to G_MAX=8. The trigger is now statistically meaningful.

### 14. 5e Rebuilt from Scratch — Hybrid GRPO/DAPO + 3-State Regime Detection ✓ Done

**Rebuilt (3 Apr 2026):** `5e_dapo_agent.py` completely rewritten with:
- `HybridDAPOAgent`: single actor, switches update rule per timestep by detected regime (DAPO on Risk-On, GRPO+KL β=0.01 on Risk-Off, GRPO+KL β=0.02 on Transition)
- `PureGRPOAgent` and `PureDAPOAgent` as comparison baselines
- `MarketRegimeDetector`: 3-state (Risk-On/Transition/Risk-Off) using composite stress score across 5 features: bench_vol, bench_ret, bench_momentum, vol_trend, cs_dispersion
- State space expanded to 8 features (added bench_momentum, regime_id_norm, regime_conf)

**Walk-forward results (3 Apr 2026):**
| Agent | Avg IR | Folds | Beats Fixed |
|---|---|---|---|
| PureGRPO | 1.003 | 0.81 / 1.71 / 1.07 / 0.85 / 0.58 | 5/5 |
| PureDAPO | 0.830 | 0.19 / 1.64 / 0.71 / 0.84 / 0.77 | 5/5 |
| HybridDAPO | 0.721 | 0.35 / 1.59 / 0.71 / 0.34 / 0.63 | 4/5 |
| Fixed α=1% | 0.337 | −0.74 / 1.36 / 0.34 / 0.34 / 0.38 | — |

### 15. HybridDAPO Regime Detector Not Firing OOS — Fixed ✓

**Problem:** Regime detector classified 90%+ of test months as Transition across all folds. Root cause: `predict()` used sigmoid(z-score) to approximate percentile ranks, which clusters near 0.5 for average months — always falling in the Transition band between p33 and p67 thresholds.

**Fixed (3 Apr 2026):** `predict()` now uses `np.searchsorted` against sorted training feature columns to compute true percentile ranks of test months against the training distribution. Verified: COVID-crash months correctly classified as Risk-Off, calm bull months as Risk-On.

### 16. CS-Transformer Signal Has Near-Zero OOS Predictive Power

**Finding (3 Apr 2026):** `3e_cs_transformer_audit.py` shows IC = −0.002, ICIR = −0.013 over 38 out-of-sample months. The model has a mild contrarian/mean-reversion tilt (loads negatively on price_to_ma50, rsi_14, ir_3m) but no statistically significant predictive power. Marginal positive IC in Risk-Off regimes (IC = +0.006) vs negative in Risk-On (IC = −0.010), neither significant.

**Action needed:** CS-T needs retraining, architectural review, or replacement with LGBM ensemble before being used in the RL pipeline.

### 17. RL Agent Underperforms in 2024

**Finding (3 Apr 2026):** Fold 5 (2022–2025) shows HybridDAPO IR = 0.626, PureGRPO IR = 0.582 — both weaker than earlier folds, with degradation concentrated in 2024. Likely cause: AI bull market (2023–2025) driven by a narrow set of mega-cap tech stocks, which a cross-sectional 500-stock factor signal struggles to exploit. The regime detector classifies 2024 mostly as Transition (low vol + modest momentum), giving no clear signal to the agent.

**Action needed:** Investigate per-year active returns in fold 5. May need sector concentration features or a signal that captures mega-cap vs rest-of-market divergence.

### 18. Factor Correlation Analysis Added ✓

**Added (3 Apr 2026):** New section in `2f_factor_diagnostics.py` — `factor_correlation_analysis()`. Computes Spearman correlation between monthly IC time-series of all 43 factors, clusters via Ward linkage on (1 − |corr|), outputs `figures/factor_ic_correlation.png` and `data/factor_clusters.csv`. Identifies groups of redundant factors to inform future factor pruning.
