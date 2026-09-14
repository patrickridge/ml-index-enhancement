## Notes from original model
---

## Assumptions & Limitations

| Assumption | Detail |
|---|---|
| **Transaction costs** | ~~Zero slippage and commissions assumed.~~ Superseded: `4b_index_enhancement.py` charges 10 bps per side on every dollar of notional traded. Realised drag is 0.3–0.9 %/yr depending on model. See item 3 below. |
| **Survivorship bias** | Universe is a fixed historical S&P 500 list. Stocks that were removed from the index before our data starts (e.g. companies that went bankrupt pre-2010) are not included, introducing mild upward bias. Stocks removed during the sample period are included up to their removal date. |
| **No look-ahead bias** | Enforced by walk-forward design: test months (2023–2025) are never seen during training or validation. Each feature uses only data available at the time of prediction. The forward return (`fwd_ret_1m`) is the label - it is never a feature. |
| **Weighting** | ~~Long-only equal weight, 2% per stock across 50 names.~~ Superseded by index enhancement: weights tilt around SPX cap weights via `w_i = w_SPX_i + α × z_i`, long-only, renormalised. The old equal-weight top-50 construction survives only in the archived `bt_lgbm.csv` runs. |
| **Monthly rebalancing** | Assumes full portfolio rebalance at month-end close. Intra-month price moves are ignored. |
| **2010 start date** | Excludes 2008–2009 GFC regime. The GFC had structurally different correlations and liquidity dynamics compared to post-crisis markets. Starting in 2010 gives a cleaner, more stationary training distribution. |
| **Cross-sectional rank normalisation** | Per-stock features (~270 as of the current panel, 26 when this note was written) are ranked within each month and scaled to [−0.5, +0.5]. Macro columns are time-series z-scored instead, since ranking a value identical across all stocks yields a constant. This removes the absolute level of each feature (e.g. whether 6-month momentum is 5% or 50% in absolute terms) and makes signals stable across different market conditions. A stock ranked in the top decile on momentum always maps to the same input value regardless of what year it is. |
| **No short-selling constraints** | Long-Short results assume you can freely short any stock in the bottom decile. In practice, borrow costs and availability vary. |

---

## Known Issues / What's Next

### 1. PCA-RP Beta Problem
The PCA Risk Parity weighting produces beta = 2.07 (twice the market exposure). This is a bug in the weight construction - the first PCA component always captures the market factor, and back-projecting to asset space amplifies it. **Fix:** replace with iterative equal-risk-contribution (ERC) weighting.

### 2. Validation Period (2022 Bear Market)
Early stopping on the old fixed 2022 validation split hit at just 4 trees because 2022 was a pure bear market - nothing looked good on the validation set. The current pipeline uses a full walk-forward retrain which avoids this. Not a critical issue, but the validation split could still be improved (e.g. a rolling validation window rather than a fixed one).

### 3. No Transaction Costs ✓ Fixed
~~All results assume zero slippage and zero commissions.~~

**Fixed (Phase 19-20):** `4b_index_enhancement.py` applies `ONE_WAY_COST = 0.0010`
(10 bps per side) against `sum|Δw|` each month, so reported `port_ret` is net.
Backtests carry `port_ret_gross`, `turnover` and `txn_cost` columns alongside it.

Measured drag on the current runs:

| Model | Turnover / month | Cost drag | IR gross | IR net |
|---|---|---|---|---|
| CS-Transformer | 64.5 % | 0.77 %/yr | 1.399 | 1.322 |
| FT-Transformer | 72.7 % | 0.87 %/yr | 0.590 | 0.538 |
| LGBM | 24.4 % | 0.29 %/yr | 0.385 | 0.178 |

(Those IRs are from the block-tilt run; see item 19. The ranking of cost impact
holds either way.) Note how much harder costs bite LGBM: its gross alpha is
barely above the drag, so the same absolute cost more than halves its IR.

Remaining imprecision: turnover is measured against last month's *target*
weights rather than their drifted end-of-month values, so passive drift is
counted as traded. That overstates turnover and therefore overstates cost.

### 4. No Fundamental Data - Partly Addressed
~~All 26 features are price-derived.~~ The panel now carries ~270 features across
24 categories, including fundamentals (Cat 9), short interest (19), institutional
ownership (20), insider filings (23) and news sentiment (24). What remains true is
that fundamental *history* is thin: only a handful of recent quarterly snapshots,
so Cat 9 contributes far less than its column count suggests. See item 7.

### 5. Regime Stability Filter Uses Full Panel (Minor Look-Ahead)
In `2a_factor_analysis.py`, the regime stability check (which decides whether a factor survives into the final 43) is run on the full panel including the test period (2023–2025 AI bull regime). The IC calculation, quintile tests, and all model training are strictly train-only (≤ 2020), but the regime filter uses data it should not technically see.

**Impact:** Factors that happen to have consistent IC in the 2023–2025 test period are slightly more likely to pass the stability filter. However, a factor must pass in ≥ 3 of 4 regimes - it cannot pass on AI bull data alone. The practical effect on results is small.

**Why it exists:** The AI bull regime (2023–present) is one of four named regimes. Without using 2023+ data there would only be 3 regimes to test against, making the filter less meaningful.

**Fix:** Use only regimes that fall entirely within the training period (pre-2021) for factor selection. Add a separate post-hoc check on test-period regime stability as a diagnostic only.

### 6. 247 Missing Historical Tickers
247 S&P 500 historical constituents could not be fetched from yfinance or Tiingo (mostly delisted/acquired companies without a clean exchange suffix). These are excluded from the universe, introducing mild survivorship bias. Wind platform does not appear to have these tickers either. Full resolution would require CRSP or Compustat access.

### 7. Fundamental Data Limited to One Month
The `factor data.xlsx` export from Wind only covered December 2025 (1 month). All 43 current factors are price-derived. Adding fundamental signals (PE, PB, money flow, turnover rate) covering the full 2010–2025 history would likely improve IC meaningfully - fundamental factors tend to score higher ICIR than pure price signals. Blocked on collaborator providing full historical export.

### 8. RL Agent Trained on Factor-Combo Scores (Not CS-Transformer)
The SAC agent (`5b_rl_portfolio_agent.py`) is trained on factor-combo scores (2010–2022) but evaluated using CS-Transformer scores (2023–2025). This creates a distribution mismatch - the agent learned alpha dynamics from a weaker signal source. Once more out-of-sample CS-Transformer test months accumulate, the agent should be retrained directly on CS-Transformer scores for a cleaner evaluation.

### 9. RL Test Period Too Short for Reliable IR
The RL agent is evaluated on 24 test months (2023–2025). IR of 0.284 on 24 months has a wide confidence interval (~±0.4). Cannot reliably distinguish skill from luck at this sample size. As more test months accumulate, the estimate will stabilise.

### 10. Layer 1 and Layer 2 RL Not Yet Connected End-to-End
Layer 1 (5b - factor weighting, objective: maximise IC/ICIR) and Layer 2 (5a - portfolio tilt, objective: maximise portfolio Sharpe) are currently trained and evaluated independently. The objectives are deliberately non-conflicting: L1 optimises signal quality, L2 optimises how aggressively to act on that signal. The full pipeline should feed L1's recent IC directly into L2's state vector so L2 can trust the signal more when L1 has high recent IC. Requires joint or sequential training once more CS-T test months accumulate.

### 11. FT-Transformer Dropped from IE
After retraining on 240 months, FT-Transformer IE IR collapsed to 0.015. It is no longer useful for index enhancement. Only CS-Transformer and LGBM are viable IE models. An ensemble of CS-T + LGBM has not yet been evaluated and could improve IR further.

### 12. DAPO vs GRPO - Unfair Candidate Count (G) Comparison ✓ Fixed
~~In `5e_dapo_agent.py`, GRPO always samples `G=4` candidates per state, but DAPO samples `G_MIN=2` in easy states.~~

**Fixed (29 Mar 2026):** Set `DAPO_G_MIN = DAPO_G_INIT = 4` to match GRPO. DAPO now uses 4 candidates in easy states and up to 8 in hard states. Rerun `5e_dapo_agent.py` to get updated numbers.

### 13. DAPO Dynamic Sampling Variance Trigger is Statistically Unreliable ✓ Fixed
~~The dynamic G decision was based on the variance of only `G_INIT=2` reward samples.~~

**Fixed (29 Mar 2026):** `G_INIT` raised to 4 so the variance estimate uses ≥4 samples before deciding to extend to G_MAX=8. The trigger is now statistically meaningful.

### 14. 5e Rebuilt from Scratch - Hybrid GRPO/DAPO + 3-State Regime Detection ✓ Done

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
| Fixed α=1% | 0.337 | −0.74 / 1.36 / 0.34 / 0.34 / 0.38 | - |

### 15. HybridDAPO Regime Detector Not Firing OOS - Fixed ✓

**Problem:** Regime detector classified 90%+ of test months as Transition across all folds. Root cause: `predict()` used sigmoid(z-score) to approximate percentile ranks, which clusters near 0.5 for average months - always falling in the Transition band between p33 and p67 thresholds.

**Fixed (3 Apr 2026):** `predict()` now uses `np.searchsorted` against sorted training feature columns to compute true percentile ranks of test months against the training distribution. Verified: COVID-crash months correctly classified as Risk-Off, calm bull months as Risk-On.

### 16. CS-Transformer Signal Has Near-Zero OOS Predictive Power - Resolved, see item 21

**Finding (3 Apr 2026):** `3e_cs_transformer_audit.py` shows IC = −0.002, ICIR = −0.013 over 38 out-of-sample months. The model has a mild contrarian/mean-reversion tilt (loads negatively on price_to_ma50, rsi_14, ir_3m) but no statistically significant predictive power. Marginal positive IC in Risk-Off regimes (IC = +0.006) vs negative in Risk-On (IC = −0.010), neither significant.

**Action needed:** CS-T needs retraining, architectural review, or replacement with LGBM ensemble before being used in the RL pipeline.

### 17. RL Agent Underperforms in 2024

**Finding (3 Apr 2026):** Fold 5 (2022–2025) shows HybridDAPO IR = 0.626, PureGRPO IR = 0.582 - both weaker than earlier folds, with degradation concentrated in 2024. Likely cause: AI bull market (2023–2025) driven by a narrow set of mega-cap tech stocks, which a cross-sectional 500-stock factor signal struggles to exploit. The regime detector classifies 2024 mostly as Transition (low vol + modest momentum), giving no clear signal to the agent.

**Action needed:** Investigate per-year active returns in fold 5. May need sector concentration features or a signal that captures mega-cap vs rest-of-market divergence.

### 18. Factor Correlation Analysis Added ✓

**Added (3 Apr 2026):** New section in `2f_factor_diagnostics.py` - `factor_correlation_analysis()`. Computes Spearman correlation between monthly IC time-series of all 43 factors, clusters via Ward linkage on (1 − |corr|), outputs `figures/factor_ic_correlation.png` and `data/factor_clusters.csv`. Identifies groups of redundant factors to inform future factor pruning.

### 19. Tilt Formula Had Drifted From the Documented One ✓ Fixed

**Problem:** README and the `4b_index_enhancement.py` docstring both described a
conviction-weighted tilt, `w_i = w_SPX_i + α × z_i`. The code underneath applied a
flat ±α to the top and bottom 100 names instead, with the middle 300 pinned at
benchmark weight. Two different constructions, and the docstring sat directly on
top of code that contradicted it.

The consequence showed up in tracking error. Under the block tilt no α in the old
grid (which started at 0.002) came near the stated 2-4 % budget: the tightest
setting gave 5.31 % and the saved run was at 11.94 %. The README meanwhile quoted
2.22 %. None of the published headline numbers reproduced from the repo.

**Fixed:** `TILT_MODE` now selects between `"zscore"` (default, the documented
construction) and `"block"` (the previous behaviour, kept so older runs can be
reproduced). The α grid was extended down to 0.0002, because the zscore tilt
reaches a given tracking error at a far smaller α than the block tilt does.

**Result:** restoring the documented formula both lands inside the TE budget and
performs better. Matched as closely as the grid allows on tracking error:

| Model | Block tilt | zscore tilt |
|---|---|---|
| CS-Transformer | IR 1.509 @ 3.88 % TE | **IR 1.736 @ 3.82 % TE** |
| LGBM | IR 0.178 @ 1.62 % TE | **IR 0.738 @ 1.82 % TE** |
| FT-Transformer | IR 0.640 @ 1.74 % TE | IR 0.635 @ 1.00 % TE |

Conviction weighting wins or ties everywhere, which is itself informative: if the
signal's edge really were confined to the tails, spreading weight proportionally
across the whole cross-section should have diluted it. It did not.

**Still unreconciled:** even after the fix, the CS-Transformer's 1.736 does not
reproduce the README's old 1.87 at 2.22 % TE, and the window is 17 months rather
than the 35 once claimed. The scores in `data/` predate the Macro FiLM upgrade
(see README status), so the old figures were likely produced against a different
model that was never synced back. README now quotes what actually reproduces.

(Superseded by item 20: the 1.736 itself was inflated by a benchmark bug.)

### 20. Benchmark Leg Was Under-Invested ✓ Fixed

**Problem:** `build_enhanced_portfolio` renormalised the portfolio weights to sum
to 1 but computed `bench_ret` from the raw `spx_weight` column. Over the scored
universe that column sums to **0.8138**, not 1 - roughly 19 % of index market cap
has no score or no forward return and silently drops out of the merge.

So a fully-invested portfolio was being measured against an 81 %-invested
benchmark. In a rising market the missing exposure books straight through as
alpha. Measured cost of the bug on the CS-T window: benchmark understated by
**3.84 %/yr** (16.17 % as coded vs 20.01 % properly normalised).

**Fixed:** benchmark is renormalised over the traded universe before both the
tilt and the return calculation, so neither leg is under-invested. Tilting off
the raw column was separately inflating effective α by 1/coverage ≈ 1.23×; that
is gone too. A `coverage` column is now written to every backtest CSV so the
problem cannot recur silently.

**Effect at matched tracking error:** CS-T drops from IR 1.736 @ 3.82 % TE to
**IR 1.438 @ 4.04 % TE**, alpha 6.62 % → 5.81 %.

### 21. The IC ≈ 0 Paradox, Resolved

Item 16 flagged that the CS-T has essentially zero whole-universe IC while the
portfolio posts a strong IR. Both are true, and they are consistent.

**Forward return by score decile, 17 OOS months:**

| Decile | Ann. return | t |
|---|---|---|
| 10 | +86.8 % | 1.94 |
| 9 | +21.2 % | 1.32 |
| 1-8 | +8.5 % to +15.8 %, unordered | < 1.7 |

The edge is entirely in decile 10. Everything below it is flat. Restricting IC to
the tails makes it *more* negative (−0.0387 on the top and bottom 10 %), so the
model cannot rank within the extremes either - it identifies a group of winners
without ordering them. Spearman IC is a rank correlation weighting all ~500 names
equally and is close to blind to a single-decile effect like this.

This also explains why the z-score tilt beats the block tilt (item 19): flat ±α
across 100 names dilutes a top-decile effect, conviction weighting concentrates
on it.

**Permutation test (200 shuffles of scores within month, universe and
construction held fixed):** null mean alpha −0.33 %, sd 0.51 %, p95 +0.66 %.
Actual +5.81 % sits outside the entire null distribution, ~12 sd above its mean.
The alpha is not a construction artifact.

**What this does not establish:** t ≈ 1.9 on decile 10 over 17 months is
suggestive, not significant. Hit rate is 47.1 %, below half - the strategy wins on
magnitude, not frequency, which is a fragile profile. And +86.8 % annualised for
one decile is extreme enough that it should be checked for domination by a few
names before anyone leans on it.

**That check was run. See item 22 - the conclusion above is wrong.**

### 22. The Alpha Was Delisted Tickers ✓ Fixed

The decile-10 concentration check that item 21 deferred turned out to overturn it.

**Finding.** The single largest contributor to decile-10 return is `CPWR`,
Compuware, **delisted in 2014**. It sits in the 2024-25 test window printing
+1500 %, +900 %, +552 %, +212 % and +200 % in separate months. Alongside it:
FNMA and FMCC (OTC since the 2008 conservatorship), NKTR and FOSL (removed from
the index in 2019), KG (acquired 2011), ADCT.

| | Share of decile-10 total return |
|---|---|
| Top 1 ticker | **63.6 %** |
| Top 3 | 76.8 % |
| Top 10 | 97.3 % |

Mean monthly decile-10 return +5.34 % against a **median of +0.53 %**, skew 16.8.
Drop the single best name each month and +86.8 % annualised becomes +4.4 %. Drop
two and it goes negative.

**Mechanism.** These names carry `spx_weight ≈ 1e-9`, so the benchmark leg
effectively ignores them, but a score-driven tilt sizes positions on *score*, not
weight. The portfolio therefore holds real positions in delisted stocks whose
price series are stale stubs, and is measured against a benchmark that holds none
of them. Pure fictional alpha.

Root cause is upstream in `1c_fetch_market_cap.py`: it assigns a weight to every
ticker in `prices.parquet` by multiplying whatever yfinance reports as *current*
shares outstanding by the historical close, with no check on index membership at
the date in question.

**On the permutation test in item 21.** It was measuring the wrong thing.
Shuffling scores within a month destroys the model's ability to *locate* the
zombies, so the null collapsed and the real run looked like twelve sigma. The
test established that the selection was non-random. It could not distinguish
genuine skill from systematically selecting corrupted observations.

**Fixed:** `clean_universe()` in `config.py`. Minimum index weight 5e-6, plus a
hard gate on any monthly return above 100 %. Calibrated so every known zombie
observation drops while the largest surviving return falls to +99 % (AppLovin,
Oct 2024, real). Logs what it removes rather than filtering silently. Wired into
`4a`, `4b`, `5b`, `5c`, `5d`; `4f` and `4g` inherit it through `4b`.

**Corrected out-of-sample results** (net of 10 bps/side, clean universe):

| Model | Months | Ann. alpha | TE | IR |
|---|---|---|---|---|
| LightGBM | 35 | +0.83 % | 1.47 % | **+0.56** |
| FT-Transformer | 39 | −0.74 % | 1.44 % | −0.52 |
| CS-Transformer | 17 | −2.52 % | 4.64 % | −0.54 |

Both transformers are negative at every α in the sweep, and their alpha degrades
monotonically as the tilt grows (CS-T: −0.19 % at the smallest α to −5.35 % at
the largest). That is a signal that is actively wrong, not merely absent.
LightGBM degrades in the opposite direction, which is what a weak but real signal
looks like.

**Caveat on interpretation.** The CS-T scores in `data/` predate the Macro FiLM
upgrade, so this tests a stale artifact rather than the model currently in the
code. The correct claim is "the scored artifact shows negative alpha", not "the
architecture does not work". A Kaggle retrain is needed to separate those.
