# Deep-Learning Index Enhancement on the S&P 500

*Patrick Ridge, Kieran Chung*

## Background - what index enhancement is

Most active equity managers don't beat the S&P 500 after fees. Index
enhancement takes a different route: hold all ~500 constituents and tilt the
weights by a small amount based on a predictive model. The portfolio still
behaves mostly like the index - tracking error typically 2–4% annualised -
but earns a small, steady layer of alpha on top.

It's a real product category at large quant shops. AQR, Robeco, BlackRock's
systematic-active desk, Acadian and the enhanced-index books inside most
multi-strat hedge funds all run some version. The headline metric is the
**information ratio** (IR = annualised alpha ÷ tracking error). IR > 0.5 is
considered institutional-grade; > 1.0 is top-quartile.

This repo implements that strategy on the S&P 500 and tests whether a
cross-sectional transformer can outperform simpler models at the ranking step.

**Stock selection does not work here; tilt sizing does.** On clean data no
ranking model beats the benchmark over 140-152 out-of-sample months, and no
factor in a 253-strong library survives multiple-testing correction. What does
survive is the reinforcement-learning overlay that decides *how large* the tilt
should be: it beats a fixed tilt in 5 of 5 walk-forward folds at every risk
budget tested, cuts maximum active drawdown from −15.6 % to −5.9 %, and returns
IR 0.579 with a 95 % interval of [+0.10, +1.06] at 5 % tracking error. Those are
separable problems and only one of them was solvable with this data.

Earlier versions of this README reported IR 1.87 for the CS-Transformer, then
IR 0.56 for a LightGBM baseline once that was retracted, then 17 surviving
factors. All three traced to silent data faults: delisted tickers with
fabricated returns, a benchmark holding 19 % cash, volume missing for 72 % of
the universe, and a per-ticker price scale factor. The audit that found them is
documented in full below, along with the one result the same faults were
*suppressing* rather than inflating.

Written as a working record of what was built and what the numbers actually
show, including where they turned out to be wrong.

## How the strategy works

1. **Score the universe every month.** A Cross-Sectional Transformer reads a
   panel of ~250 usable features per stock plus 20 macro series, and produces
   one score per stock. Momentum, volatility, liquidity and microstructure come
   from price and volume; fundamentals come from SEC EDGAR (2005-2026) and
   short interest from FINRA (2018-2026), each stamped with the date it became
   public rather than the period it describes. The 13F and news-sentiment
   columns are still empty, and sentiment varies by month rather than across
   stocks, so neither is in the tested library.
2. **Tilt around benchmark weights.** `w_i = w_SPX_i + α × score_z_i`,
   long-only, renormalised to sum to 1. α is a single number controlling how
   aggressive the tilt is this month.
3. **Let an RL agent pick α.** A SAC policy observes the current market
   state (signal dispersion, benchmark vol, rolling tracking error, regime
   flag, yield-curve features) and outputs α, trained directly on the
   information ratio.

## The hypothesis being tested

Tree ensembles like LightGBM score each stock in isolation. Ranking 500 stocks
against each other is inherently relational: a volatility reading of 30% means
one thing when most of the universe sits at 15% and something else when most
sit at 45%. The hypothesis was that an architecture which sees the whole
cross-section at once should rank better than one that does not.

The Cross-Sectional Transformer implements that directly. A first attention
stage operates within each stock's feature vector; a second attends across
every stock in the universe that month, so each score is aware of its peers.
Macro state (VIX, yield curve, Fed rate expectations) enters through FiLM
conditioning. Training is two-stage: masked-MSE pre-train, then a
portfolio-level RL fine-tune (GRPO or DAPO) optimising portfolio return rather
than per-stock label accuracy.

**The hypothesis is not supported, and it has now been tested properly.** On
140 out-of-sample months of rebuilt data, with Macro FiLM active and
fundamentals in the feature set, the CS-Transformer posts IR −0.221 with a 95 %
interval of [−0.74, +0.36]. The LightGBM baseline it was supposed to beat posts
+0.143 over 152 months with an interval of [−0.36, +0.75]. **Those intervals
overlap across almost their entire range**, so the data cannot separate the two
architectures, and neither is separable from zero.

The architecture was never the binding constraint. No factor in the library
survives multiple-testing correction, so a better ranker has very little to
rank on. Cross-sectional attention is a reasonable idea that this dataset
cannot evaluate, because there is not enough cross-sectional signal present for
any ranker to exploit.

## Why reinforcement learning on top

The tracking-error budget is a hard constraint, not a soft preference. A
fixed α over-reacts in some regimes (e.g. when the cross-section has
collapsed and every stock looks the same, aggressive tilting just adds
tracking error without alpha) and under-reacts in others (in a dispersed
month with a clear top/bottom split, a timid tilt wastes signal). A learned
policy adapts tilt size to what the market is actually doing.

Validation is a 5-fold expanding-window walk-forward from 2014 to 2025 - no
shared training data between folds, factor weights refitted from scratch each
fold, agent retrained fresh. That design is sound and is the most defensible
part of the evaluation setup here.

On 144 out-of-sample months the SAC overlay beats a fixed-α baseline in **5 of
5 folds** at every risk budget tested, and cuts maximum active drawdown from
**−15.6 % to −5.9 %**. The information ratio depends on where the tilt cap is
set: **0.579 at 5.0 % tracking error**, where the 95 % interval excludes zero,
falling to 0.386 at the 4 % mandate boundary, where it does not. Both rows are
in the table below, because reporting only the first would repeat the mistake
that produced IR 1.87.

## What the pipeline does

1. **Data prep** (`1*` scripts): pull OHLCV, constituent history, fundamentals, short interest,
   institutional ownership, prediction markets, insider trades, news sentiment, sector mappings.
   Fundamentals come from SEC EDGAR and short interest from FINRA, both free and keyless. Each is
   stamped with the date the figure became *public* rather than the period it describes - a 10-K
   lands about 40 days after quarter end and short interest about 8 days after settlement, so
   using the period date would hand the model weeks of hindsight.
2. **Feature engineering** (`1h_feature_engineering.py`): compute ~283 factor columns across 24 categories,
   of which 253 carry cross-sectional information and are testable; the rest are 13F and
   news-sentiment columns that are empty or vary only by month (momentum, volatility, liquidity,
   microstructure, fundamentals, short interest, mined-alpha candidates, macro).
   Per-stock features are cross-sectionally ranked each month; macro features are kept at raw scale.
3. **Factor diagnostics** (`2*` scripts): IC, IC decay, quintile returns, crowding/correlation,
   IS-only screening of new candidates via Lasso / RF / LightGBM and BHY.
4. **Models** (`3*` scripts):
   - `3a_ft_transformer.py` - FT-Transformer (independent per-stock ranker)
   - `3c_cs_transformer.py` / `3d_cs_transformer_kaggle.py` - Cross-Sectional Transformer with
     Macro FiLM conditioning, optional correlation attention bias and a GRPO/DAPO RL fine-tune
     on portfolio return.
   - `3f_lgbm_baseline.py` - LightGBM reference model on the same walk-forward protocol,
     so the tree and transformer numbers are comparable rather than merely adjacent.
5. **Portfolio construction** (`4*` scripts): tilt ±α around SPX cap weights based on scores,
   sweep α to maximise information ratio within a 2–4 % tracking-error budget.
6. **RL overlay** (`5*` scripts): a two-layer policy that adjusts the tilt size month-to-month
   based on macro state.

## Repository layout

```
.
├── config.py                       shared hyper-parameters + paths
├── utils_factors.py                ~190 factor functions used by 1h
├── utils_rmt.py                    Random Matrix Theory covariance denoising
├── run_pipeline.sh                 end-to-end: fetch → panel → diagnostics
│
├── 1a–1p_*.py                      data fetch + feature engineering
├── 1s/1t_*.py                      price refetch + validation gate
├── 1u_fetch_edgar_fundamentals.py  SEC EDGAR fundamentals, point-in-time
├── 1v_fetch_short_interest.py      FINRA consolidated short interest
├── 2a–2k_*.py                      factor diagnostics + multiple-testing checks
├── 3a–3d_*.py                      ranking models
├── 3e_cs_transformer_audit.py      post-hoc diagnostics on CS-T scores
├── 3e_hp_sweep{,_kaggle}.py        CS-Transformer hyperparameter sweeps
├── 3f_lgbm_baseline.py             LightGBM reference on the same protocol
├── 4a–4g_*.py                      backtests + portfolio construction
├── 5a–5f_*.py                      RL overlays
├── 6_regime_dashboard.py           Streamlit regime analysis dashboard
├── 7_synthetic_regimes.py          DDPM-based synthetic-regime stress test
│
├── research/
│   └── factor_mining/              IS-only candidate discovery pipeline
│       ├── run_factor_mining.py
│       ├── candidate_factory.py    ~70 OHLCV candidate families
│       ├── validation_engine.py    BHY, RAS, dedup
│       ├── screen_*.py             Lasso / RF / LightGBM / autoencoder screens
│       └── regime_entropy.py       vol-regime entropy detector
│
├── Notes/                          development log, strategy notes
└── data/                           (gitignored - regenerated from the scripts)
```

## How to reproduce the results

### 1. Install

```bash
pip install -r requirements.txt

# optional secrets
export FINNHUB_API_KEY="..."                  # only needed for 1o_fetch_sentiment
export SEC_USER_AGENT="Your Name you@mail"    # only needed for 1n_fetch_insider_trades
```

### 2. Build the data panel

```bash
# One-time fetches
python 1a_price_parquet.py              # OHLC → prices.parquet
python 1b_fetch_constituents.py         # historical S&P 500 membership
python 1c_fetch_market_cap.py           # monthly SPX cap weights
python 1d_fetch_missing_tickers.py      # yfinance fallback for delisted tickers
python 1e_parse_wind_prices.py          # merge Wind export if present
python 1f_ingest_wind_xlsx.py           # same for multi-file Wind exports
python 1g_rebuild_panel.py              # base monthly panel from prices

# Optional data sources (run once; the panel handles missing ones gracefully)
python 1j_fetch_simfin.py               # quarterly fundamentals
python 1k_fetch_short_interest.py       # short interest
python 1l_fetch_13f.py                  # institutional ownership
python 1m_fetch_prediction_markets.py   # Fed/VIX/yield-curve signals
python 1n_fetch_insider_trades.py       # SEC EDGAR Form 4
python 1o_fetch_sentiment.py            # Finnhub news + VADER
python 1p_fetch_sectors.py              # yfinance GICS sector map

# Build the ~270-factor enriched panel (~2 hours)
python 1h_feature_engineering.py
```

### 3. Train and backtest

```bash
# Diagnostics (optional - not required for the final model)
python 2a_factor_analysis.py
python 2f_factor_diagnostics.py

# Models
python 3a_ft_transformer.py             # local CPU, ~10-15 min
# CS-Transformer runs on Kaggle - see below

# Backtests
python 4b_index_enhancement.py          # IE tilt, α-sweep per model
python 4d_benchmark_spx.py              # summary vs S&P 500
```

### 4. CS-Transformer on Kaggle

The local `3c_cs_transformer.py` uses the same model code but is CPU-slow.
For GPU runs, use the self-contained `3d_cs_transformer_kaggle.py`:

1. Upload `data/panel_monthly_enriched.parquet` to a Kaggle Dataset called
   `investsoc-ml-data`.
2. Open a Kaggle notebook → Settings → Accelerator = GPU T4.
3. Attach the dataset, paste `3d_cs_transformer_kaggle.py` into a cell, run.
4. Download `scores_cs_transformer.parquet` back into `data/`.

Expected GPU time: ~60–90 min including the GRPO/DAPO RL fine-tune.

## The CS-Transformer

The main model. Two-stage attention:

- **Stage 1 - per-stock feature attention.** Each stock's feature vector is tokenised and passed
  through a transformer. A learned `[CLS]` token summarises the stock.
- **Stage 2 - cross-stock attention.** All stocks for a month are treated as one sequence so each
  stock's representation is informed by its peers.

Two additions on top of the base model:

- **Macro FiLM.** 16 macro features (VIX, yields, spreads, SPX stats, prediction-market
  probabilities) are encoded separately and produce per-feature `(gamma, beta)` that modulate the
  Stage 1 tokens. Identity-initialised so pre-trained weights are preserved at the start of
  training. This gives the model an explicit handle on "factor X matters more in regime Y".
- **Factor correlation bias (optional).** An F×F Spearman correlation of the features is computed
  on the training set and fed as additive attention bias in Stage 1, with a small per-head scalar
  weight learned. Disabled by default - enable after FiLM has been validated.

Training is two-stage: a standard masked-MSE pre-train with L1/L2 feature-tokenizer penalties,
followed by a portfolio-level RL fine-tune using GRPO (PPO-style clipping + KL penalty vs a frozen
reference), DAPO (asymmetric clipping, dynamic group size, no KL) or a regime-aware hybrid.
Method is selectable in `config.py: RL_FINETUNE_PARAMS["method"]`.

## Factor categories (24 total, ~270 features)

| Cat | Name | Source |
|-----|------|--------|
| 1 | Momentum | OHLC (1/3/6/12m + skip-month reversal) |
| 2 | Volatility | realised, idiosyncratic, downside |
| 3 | Tail risk | CVaR, max DD, skew, kurt |
| 4 | Price trend | RSI, MACD, Bollinger, trend strength |
| 5 | Volume / liquidity | Amihud, turnover, bid-ask proxy |
| 6 | Beta / correlation | market / downside / rolling beta |
| 7 | Microstructure | spread, price impact, autocorr |
| 8 | Cross-sectional | SPX- and sector-relative moves |
| 9 | Fundamentals | ROE, ROA, margins, growth, EQ (`1u`, SEC EDGAR) |
| 10 | Macro / regime | VIX, yield curve, credit, PMI |
| 11 | Time-signal v1 | ADX, trend consistency |
| 12 | Barra-style | value, growth, leverage, EY |
| 13 | Macro × factor interactions | e.g. momentum × VIX |
| 14 | Size | log market cap |
| 15 | Tail ranking | top/bottom-decile dummies |
| 16 | Mined alpha | 12 hand-mined price signals |
| 17 | Time-signal v2 | ADX regime, roll spread, vol expansion |
| 18 | Seasonality | same-month, turn-of-month, January |
| 19 | Short interest | % float, days-to-cover (`1v`, FINRA) |
| 20 | Institutional | ownership %, HHI (`1l`) |
| 21 | Prediction markets | Fed/VIX/policy uncertainty (`1m`) |
| 22 | Factor-mining candidates | 9 families, ~69 features + entropy regime |
| 23 | Insider trading | 30/90-day filing activity (`1n`) |
| 24 | News sentiment | VADER mean / std / momentum / volume (`1o`) |

**Selection rules.**
Correlation pruning happens *after* training, not before.
OOS IC is a diagnostic only - never a selection criterion (avoids lookahead).
Prediction-market factors are used as *timing* signals at the portfolio level, not as cross-sectional
stock signals.

## Headline results

### Index enhancement, out-of-sample, net of 10 bps per side

Each model is shown at the α whose tracking error sits closest to the 2-4 %
budget. Windows differ because the three models were last scored at different
times, so these rows are not a like-for-like horse race.

| Model | Months | Ann. alpha | TE | IR | 95 % CI | P(IR>0) |
|---|---|---|---|---|---|---|
| **RL tilt overlay** | 144 | +2.90 % | 5.00 % | **+0.579** | **[+0.10, +1.06]** | **99 %** |
| LightGBM | 152 | +0.29 % | 2.02 % | +0.143 | [−0.36, +0.75] | 70 % |
| CS-Transformer | 140 | −1.04 % | 4.70 % | −0.221 | [−0.74, +0.36] | 21 % |
| Factor-combo | 44 | — | — | −0.264 | [−1.21, +0.45] | 24 % |
| *LightGBM, weight floor off* | 18 | +0.83 % | 1.47 % | *+0.56* | [−0.69, +1.99] | 81 % |

The overlay is the only row whose interval excludes zero, and it sits at 5 %
tracking error rather than inside the 2-4 % mandate; the sweep across tilt caps
is below.

**The CS-Transformer row is now a real test rather than a 17-month fragment.**
140 out-of-sample months on the rebuilt panel, trained with Macro FiLM active,
on a feature set that includes fundamentals and short interest. Its monthly IC
is *positive* at +0.0042 (ICIR +0.294) while its portfolio IR is −0.221, which
is the same IC-to-IR divergence documented in item 21: the z-score tilt sizes
positions by score magnitude, so the bet concentrates in the tails of the score
distribution and a mildly positive ranking with badly positioned extremes loses
money. Its interval spans zero either way.

The italic row is kept as a control rather than a result. It is the same script
on the same months with the same hyperparameters as the clean LightGBM run, and
the only difference is whether delisted tickers are filtered out of training.

The CS-Transformer row is from a model retrained on the cleaned universe. An
earlier run, trained on the panel that still contained delisted tickers, gave
−2.52 % alpha at 4.64 % TE for an IR of −0.54. Removing the zombies from
*training* cut the alpha drag by 85 % and the tracking error to a quarter:
validation MSE fell from 0.234 to 0.0072 and validation IC went from −0.002 to
+0.046. The model was previously spending its capacity trying to fit returns
that were fabricated.

It improved toward zero without crossing it, and at 17 months the interval is
far too wide to call either way.

**Every interval contains zero.** No ranking model in this study, clean-trained
or otherwise, is statistically distinguishable from no skill at all.

That cuts both ways, and it is worth being blunt about. LightGBM's +0.56 is not
evidence that gradient boosting works here, and it is now known to be an
artifact. The clean +0.143 over 152 months is not evidence that it works
either; its interval runs from −0.37 to +0.74. Nor was the earlier −0.31 for
the CS-Transformer evidence that cross-sectional attention fails. The intervals overlap across nearly their whole range, so the
data cannot separate the models from each other, let alone from zero. The point
estimates differ; the evidence does not.

The cause is sample size, not modelling. These models have 17 to 35
out-of-sample months, where the standard error on an information ratio is
roughly ±0.7 - far larger than any effect this strategy could plausibly
produce. Intervals are 10,000-resample circular block bootstraps with six-month
blocks, from `4h_bootstrap_ci.py`.

### The RL overlay, and an awkward tension

The overlay that sizes the tilt is the only part of the project with enough
out-of-sample data to measure - 144 months rather than 17. The α cap bounds the
action space and therefore the tracking error, so the result has to be read
across the whole sweep rather than at one convenient point:

| α cap | Ann. alpha | TE | IR | 95 % CI | Folds won | In mandate? |
|---|---|---|---|---|---|---|
| 1.80 % | 6.75 % | 9.39 % | 0.719 | [+0.244, +1.181] | 5/5 | no |
| 0.40 % | 2.90 % | 5.00 % | 0.579 | [+0.095, +1.062] | 5/5 | no |
| **0.25 %** | 1.56 % | **4.05 %** | 0.386 | [−0.114, +0.899] | **5/5** | **yes** |

Fixed-α baseline, for reference: IR 0.289 at 9.63 % TE.

**The significance boundary sits at roughly 5 % tracking error, just outside a
2-4 % mandate.** Shrink the budget to fit the product and the interval reopens
across zero. Quoting 0.719 without that sentence would be the same selective
reading that produced the original IR 1.87.

#### The baseline has to be the same size, or the test measures nothing

The fixed-α baseline originally ran at 1.0 % against an agent averaging 0.32 %,
three times the tilt and therefore three times the tracking error. Since IR
falls as α grows, a smaller tilt wins that comparison on arithmetic alone, and
"beats fixed in 5 of 5 folds" would have been true and empty. A cumulative
active return plot makes the problem obvious: the fixed leg earns nearly twice
as much in absolute terms and only loses on the denominator.

Re-run with the baseline at the agent's own mean tilt, so both carry the same
risk:

| | RL overlay | Fixed, same size |
|---|---|---|
| Ann. alpha | **2.90 %** | 1.73 % |
| Tracking error | 5.00 % | 5.04 % |
| Information ratio | **0.579** | 0.344 |
| Max active drawdown | **−5.78 %** | −7.77 % |
| Folds won | **5/5** | — |

**At matched tracking error the agent produces 67 % more alpha.** That cannot be
a level effect, because the levels are equal. `5c` now defaults to the matched
baseline; `ML_FIXED_ALPHA=0.010` reproduces the old comparison.

#### The agent tilts harder in the months that pay more per unit

A second test of the same claim, from a different direction. If the policy is
genuinely adapting rather than drifting, the months it chooses to tilt hard
should be months where tilting *pays better*, not merely months where it tilted
more. Dividing active return by α removes the mechanical part and leaves the
payoff per unit of tilt:

| | Mean active return per unit of α |
|---|---|
| High-tilt months (above median α) | **+2.02** |
| Low-tilt months (below median α) | −1.35 |

Welch t = 4.64, p = 7.8e-6, 72 months each side. Across all 144 months α
correlates with per-unit payoff at **r = +0.425, p = 1.1e-7**.

The agent only ever moves α between 0.20 % and 0.40 %, so this is a narrow
range producing a wide separation in outcome. Two independent pieces of
evidence now point the same way: more alpha at matched risk, and tilt size
lining up with when tilting works.

Two things do not depend on where the cap is set:

- **5 of 5 folds at every cap tested**, against a same-size baseline.
- **Maximum active drawdown falls from −15.6 % to −5.9 %** against the 1 % leg,
  and −5.78 % against −7.77 % at matched size. For a product whose entire
  promise is behaving like the index, that is arguably more relevant than the
  IR.

Before the data repair in Notes item 27 this same overlay measured IR 0.299
with an interval spanning zero, and the crossover sat near 7 % TE. Nothing about
the agent changed. The benchmark it is measured against had a top-10 weight of
72 % against a real 40 %, and active return is defined relative to that
benchmark. **This is the one result in the project that the broken data was
suppressing rather than inflating.**

What rescues the result is that those intervals test the wrong question. They
ask whether absolute IR exceeds zero. The claim worth making is relative:

- **5 of 5 folds** beat the fixed-α baseline. Under a coin-flip null that is
  p = 0.5⁵ = **0.031**.
- Paired t-test across matched folds: **p = 0.024** for PPO, with all four
  algorithms pointing the same way (`5h_algo_significance.py`).
- Max active drawdown falls from **−15.6 % to −5.9 %**, at every cap tested.

Matched-sample tests difference out the period effect, so they carry far more
power than comparing two noisy information ratios.

**The defensible statement:** adaptive tilt sizing reliably outperforms a static
tilt at every risk budget tested, cutting maximum active drawdown by roughly two
thirds. Its absolute information ratio is significant at 5 % tracking error and
not at 4 %, so whether it clears zero *inside the mandate* is still unsettled
after 144 months.

### α is a risk parameter, not a performance one

| α cap | Ann. alpha | TE | IR | Folds |
|---|---|---|---|---|
| 0.0025 | 1.56 % | 4.05 % | 0.386 | 5/5 |
| 0.0040 | 2.90 % | 5.00 % | 0.579 | 5/5 |
| 0.0180 | 6.75 % | 9.39 % | 0.719 | 5/5 |
| 0.050 | 3.05 % | 9.71 % | 0.314 | 4/5 |

IR spans 0.297 to 0.494 across the sweep - a range of 0.197 against a bootstrap
CI half-width near ±0.44. **The whole sweep fits inside one confidence
interval**, so the apparent optimum at 0.018 is noise and cannot be tuned
toward. Tracking error, by contrast, moves monotonically over a 2.5× range.

So α controls risk, not return. Pick it to meet the mandate, not to maximise a
backtested IR.

### Which RL algorithm? It does not matter

| | SAC | PPO | GRPO | DAPO | Fixed |
|---|---|---|---|---|---|
| IR | 0.462 | 0.672 | 0.339 | 0.599 | 0.025 |
| Folds won | 5/5 | 5/5 | 3/5 | 4/5 | — |

Every pairwise comparison is insignificant - PPO vs DAPO differ by 0.001
(p = 0.997), the closest pair p = 0.189. DAPO's *mean* improvement over the
baseline (+0.687) marginally exceeds PPO's (+0.686) yet fails significance
purely because it is more variable across folds, which is exactly the trap a
point-estimate ranking falls into.

An earlier version of this README reported GRPO 0.88 / PPO 0.87 / SAC 0.79. On
clean data that ordering inverts. The value comes from sizing the tilt at all,
not from the choice of policy gradient.

### Factor library, after multiple-testing correction

Of 253 factors with computable t-statistics, 15 pass a raw p < 0.05 against
12.7 expected by chance. **None survives Benjamini-Hochberg-Yekutieli at
FDR 0.10** - BHY rather than BH because factor IC series are heavily
cross-correlated and BH assumes independence.

Fifteen against twelve-point-seven is what 253 tests produce on their own. At
p < 0.001, where 0.25 are expected, **zero** clear it. The strongest factor in
the library is `dollar_vol_21d` at t = −3.03; BHY at 253 tests needs roughly
t > 4.0 for the leading factor, and plain BH needs 3.55.

Median |t| across the library is **0.566**, against 0.674 for the median of a
standard normal. The distribution of t-statistics here is not a weakened
version of a real signal. It is what 253 correlated tests on noise look like.

An earlier version of this section reported **17 survivors, every one a size,
liquidity or volatility factor**, and built the project's central explanation on
it. That result did not survive the data repair in Notes item 27. Volume was
missing for 505 of 697 tickers and `log_mktcap` was valid for 9 % of rows, so
four of those five survivors were substantially measuring *whether the fetch had
worked* rather than liquidity or size. On the rebuilt panel their t-statistics
roughly halve:

| Factor | before | after |
|---|---|---|
| `size_proxy` | −5.59 | −2.94 |
| `dollar_vol_21d` | −5.31 | −2.98 |
| `log_mktcap` | −5.05 | −2.62 |
| `amihud_illiq_21d` | +4.50 | +2.42 |
| `cand_kyle_lambda_21d` | +4.07 | +0.84 |

The signs are unchanged - size negative, illiquidity positive, both the
textbook directions - so the effects are real in direction. Their apparent
*strength* was about half artifact. Median |t| across the whole library barely
moved (0.59 to 0.56), which is what rules out a global degradation from the
rebuild: the damage is confined to exactly the factors that depended on the
broken inputs.

**This is not a threshold artifact.** `2k_fdr_sensitivity.py` returns zero at
every FDR from 0.01 to 0.30, and zero under plain BH as well, so it is neither
the dependence correction being strict nor the tolerance being tight.

**Read it narrowly.** It does not mean the factors are noise: the strongest sits
at |t| 2.98, against the t > 3.0 hurdle Harvey, Liu and Zhu argue for once
multiple testing is taken seriously. It means no single factor is strong enough
to clear a bar set for 237 simultaneous tests. BHY answers *have I discovered
something*, which is the right question for a research claim and the wrong one
for whether a combination is worth trading.

The result is also coherent rather than surprising. This library is entirely
price and volume derived, over a universe that is entirely large-cap and liquid.
**A size effect needs small caps and an illiquidity premium needs illiquid
names**, and the S&P 500 supplies neither. The old data manufactured both,
because "no volume data" was silently standing in for "illiquid".

So the explanation for why no ranking model works is simpler than the one it
replaces: there is no detectable cross-sectional signal in this data to find,
and four independent models agreeing on that is the consistent answer rather
than four separate failures.

### The escape hatch is now closed

An earlier version of this section said the tested library was entirely price
and volume derived, so value and quality were absent from the results because
they had **never been tested** rather than because they failed. That was true
and it was the single most likely thing to change the conclusion. It is no
longer true. `1u_fetch_edgar_fundamentals.py` and `1v_fetch_short_interest.py`
landed real point-in-time data, and both are now in the tested library:

| Factor | IC | t | Months |
|---|---|---|---|
| `eps_growth_yoy` | +0.0097 | +1.49 | 99 |
| `roa` | +0.0115 | +1.18 | 109 |
| `debt_to_equity` | −0.0078 | −1.09 | 109 |
| `roe` | +0.0078 | +1.03 | 107 |
| `revenue_growth_yoy` | +0.0058 | +0.96 | 97 |
| `pb_ratio` | +0.0103 | +0.82 | 109 |
| `pe_ratio` | +0.0065 | +0.57 | 109 |
| `short_change_2w` | −0.0061 | −1.02 | 60 |
| `short_pct_float` | −0.0080 | −0.45 | 60 |

**Nothing reaches |t| = 1.5.** The bar for BHY at 253 tests is near 4.0.

One honest caveat: these run on shorter samples than the price factors, because
EDGAR coverage starts later and FINRA's API history begins in 2017-12. A t-stat
scales with √months, so on the full 156 months the short-interest figures would
be about 1.6× larger - which moves the best of them from 1.02 to 1.65 and
changes no conclusion.

Ten columns still cannot be tested: four 13F columns, which are genuinely empty
(`1l_fetch_13f.py` produced nothing), and six that vary by month but are
identical across stocks - `january_dummy`, `turn_of_month`, and four
sentiment/news aggregates. A Spearman IC against a column with no
cross-sectional variation is undefined, not weak, which is why they drop out
before testing rather than failing it. `config.py` detects them at load at
runtime rather than hardcoding a list, so each returns on its own the day its
data arrives.

The remaining untested input is institutional ownership. It is the only one
left that could carry a different *kind* of effect, and on the evidence above
it is not where the money is.

#### Threshold sensitivity, and a caveat on the above

The universe filter cuts at `spx_weight >= 5e-6`, itself a size and liquidity
threshold. Finding size factors significant after filtering on a size boundary
is a mechanism that could manufacture the result, so the IC t-statistics were
recomputed across five weight floors.

| Factor | floor 0 | 1e-6 | **5e-6** | 1e-5 | 5e-5 |
|---|---|---|---|---|---|
| `log_mktcap` | −1.95 | −2.22 | **−5.05** | −3.03 | −2.05 |
| `size_proxy` | −4.28 | −4.43 | **−5.83** | −4.38 | −2.64 |
| `high_vol_week` | −2.76 | −3.81 | **−5.69** | −4.35 | −2.65 |
| `amihud_illiq_21d` | +4.50 | +4.98 | **+4.95** | +4.35 | +3.09 |
| `volume` | −0.87 | −2.15 | **−3.38** | −2.33 | −2.06 |

**Every survivor peaks at 5e-6 - the threshold actually used.** Mean |t| range
across floors is 2.50 for survivors against 0.48 for a control group of factors
that failed BHY.

The effects are real: signs are stable across all five floors (0 of 9 flip) and
most hold |t| > 2 everywhere. But **the |t| ≈ 5.8 quoted above is not robust.**
The defensible range is 2.5 to 4.5, and `log_mktcap` at no filter at all is
−1.95, below conventional significance. The survivor count of 17 would shrink at
other thresholds.

Two readings, both partly right. Benign: 5e-6 sits near the genuine
member/non-member boundary, so below it stubs dilute the signal and above it the
size distribution gets truncated - a peak at the correct cut is expected.
Concerning: the coincidence is close enough to disclose rather than explain
away. The threshold was calibrated on zombie removal before any factor testing,
not tuned for significance, but that is an assurance rather than a proof.

The control group is the reassuring part: factors that failed BHY stay
insignificant at every floor, so the filter is not distorting the cross-section
broadly. Override with `ML_MIN_WEIGHT` to reproduce any row above.

#### The thresholds are not what produce the answer

Three conventions sit behind a survivor count: the raw p < 0.05 screen, the FDR
level of 0.10, and the decision to correct for arbitrary dependence. None is
derived from anything. The 5% is Fisher's 1925 convention, which stuck largely
because statistical tables were printed at 5% and 1%. `2k_fdr_sensitivity.py`
sweeps all three.

| FDR q | BHY | plain BH |
|---|---|---|
| 0.01 | 0 | 0 |
| 0.05 | 0 | 0 |
| **0.10** | **0** | 0 |
| 0.15 | 0 | 0 |
| 0.20 | 0 | 0 |
| 0.30 | 0 | 3 |

**Zero, flat across a thirtyfold range of tolerance**, and zero under plain BH
until q = 0.30, where a 30 % false-discovery rate admits three. Neither the
dependence correction nor the FDR level is what produces the answer, which is
the only thing this sweep is for.

The raw level tells the same story from the other side:

| Raw p | Pass | Expected by chance | Ratio |
|---|---|---|---|
| 0.10 | 22 | 25.3 | 0.87 |
| 0.05 | 15 | 12.7 | 1.19 |
| 0.01 | 4 | 2.5 | 1.58 |
| 0.005 | 3 | 1.3 | 2.37 |
| 0.001 | 0 | 0.25 | 0.00 |

**At every conventional level the library produces about as many hits as chance
would.** The ratio never reaches 2.4 and the tail is empty: not one factor in
253 clears p < 0.001. A library with a few genuinely strong factors buried in
noise looks nothing like this - it has a fat tail of small p-values and a ratio
that climbs as the screen tightens. This one flattens.

So the correction is not what kills the factors, and reporting it as though a
harsher test defeated a real signal would overstate what is here. The
uncorrected numbers already sit at chance. BHY is reported because it is the
right test when 308 factor pairs correlate above 0.70, not because it is doing
the work.

On the wider question of whether the profession's conventional hurdle is too
lenient once you account for how many factors have been tested, see Harvey, Liu
and Zhu (2016), which argues for t > 3.0 rather than t > 2.0. Exactly one
factor here reaches it, at 3.03.

### Capacity

The backtests charge a flat 10 bps per side, which captures spread and
commission but is blind to size. `4i_capacity.py` applies a square-root impact
model instead - cost scales with the square root of participation in each name's
daily volume:

| AUM | Impact/yr | Flat 10 bps | Total |
|---|---|---|---|
| $1bn | 0.16 % | 0.14 % | 0.30 % |
| $5bn | 0.36 % | 0.14 % | 0.50 % |
| $10bn | 0.51 % | 0.14 % | 0.65 % |
| $50bn | 1.14 % | 0.14 % | 1.28 % |

There is no surviving positive alpha to measure these against, so the reference
point is a hypothetical 0.83 %/yr, the level the retracted LightGBM run showed.
Read the table as the size at which costs would consume an edge of that size,
not as a claim that the edge exists. Total cost stays below it out to roughly
**$10bn**.
Two structural reasons: real turnover is 12-15 % of the book per month, and at
α = 0.0005 most positions sit within a few basis points of index weight, so
trades are small fractions of names already trading billions daily.

(The 64 % turnover figure quoted elsewhere in `Notes/Improvements.md` counts
benchmark drift as trading and overstates the tradeable quantity about 4×. Costs
in the backtests are conservative as a result, not optimistic.)

**The binding constraint here is alpha, not capacity.** A strategy whose
information ratio has a confidence interval spanning zero does not have a size
problem. Assumptions: C = 1.0, 2 %/day single-name vol, ADV at 0.7 % of market
cap, execution over 3 days - halving C or doubling the execution window roughly
halves the impact estimate, so read the levels as indicative and the scaling
across AUM as the robust part.

### Survivorship bias, quantified

The 247 unfetchable tickers cannot be recovered without CRSP or Compustat, so
the bias is estimated instead (`1q_survivorship_bias.py`). Index leavers
underperform by **−4.16 %/month** over their final six months. Spread across
149,094 stock-months and scaled to the missing names, that implies roughly
**+0.50 %/yr of spurious alpha** in every backtest here.

Material against alphas of 0.8-3.6 %: read any headline figure as about half a
point per year optimistic.

Both transformers are negative at *every* α in the sweep, and their alpha
degrades monotonically as the tilt grows — CS-Transformer runs from −0.19 % at
the smallest α to −5.35 % at the largest. That is the signature of a signal
that is actively wrong rather than merely absent: acting on it harder loses
more.

The floor-on LightGBM does the same thing, from −0.29 % to −12.52 % across the
sweep at a 37 % monthly hit rate. The floor-off run is the only one that
behaves like a weak but genuine signal, with IR falling from 0.62 at small α to
0.08 at large as costs catch up with a thin edge. That contrast is the clearest
single picture of what the delisted tickers were doing: they are the difference
between a signal that looks real and one that is not there.

Reproduce with `python 4b_index_enhancement.py`. Full α sweep in
`data/ie_summary.csv`.

### How an earlier version of this README reported IR 1.87

Worth documenting, because the failure mode is instructive and the diagnostic
path is most of the actual research content here.

Earlier runs reported an IR of 1.87 for the CS-Transformer, later 1.74, then
1.44 as two separate bugs were fixed. All of it was spurious. Three compounding
problems:

**1. The tilt formula had drifted from the documented one.** README and
docstring described `w_i = w_SPX_i + α × z_i`; the code applied a flat ±α to the
top and bottom 100 names. Under the block tilt no α reached the stated 2-4 %
tracking-error budget, so no published number reproduced.

**2. The benchmark leg was under-invested.** Portfolio weights were renormalised
to sum to 1, benchmark weights were not. Over the scored universe `spx_weight`
sums to 0.81, not 1, because ~19 % of index market cap fails the merge. A
fully-invested portfolio measured against an 81 %-invested benchmark books the
missing exposure as alpha — worth 3.84 %/yr in a rising market.

**3. Delisted tickers with corrupt prices.** This was the fatal one.
`prices.parquet` retains names that left the index years ago, and `1c` assigns
them a market-cap weight anyway. Their weights come out near zero, so the
benchmark ignores them, but a score-driven tilt still opens real positions in
them. Their price series are stale stubs producing impossible returns.

Compuware (`CPWR`), **delisted in 2014**, appears in the 2024-25 test window
printing +1500 %, +900 %, +552 %, +212 % and +200 % in separate months.
Alongside it: FNMA and FMCC (OTC since 2008), NKTR and FOSL (removed 2019), KG
(acquired 2011).

Concentration in the top score decile before the fix:

| | Share of decile-10 return |
|---|---|
| Top 1 ticker | **63.6 %** |
| Top 3 tickers | 76.8 % |
| Top 10 tickers | 97.3 % |

Mean monthly decile-10 return was +5.34 % against a **median of +0.53 %**, skew
16.8. Dropping the single best name per month took it from +86.8 % annualised to
+4.4 %. Dropping two took it negative.

A permutation test had earlier appeared to vindicate the strategy at roughly
twelve standard deviations above a shuffled-score null. It was measuring the
wrong thing: shuffling destroys the model's ability to *locate the zombies*, so
the null collapsed. The test proved the selection was non-random, which is not
the same as proving it was skilful.

**The fix** is `clean_universe()` in `config.py`: a minimum index weight of 5e-6
plus a hard gate on any monthly return above 100 %. The threshold is calibrated
so that every known zombie observation is removed while the largest surviving
return falls to +99 % (AppLovin, October 2024, which is real). It logs what it
drops rather than filtering silently.

Applied to the CS-Transformer test panel it removes 2,620 of 8,840 stock-months,
all on the weight criterion. Those rows were carrying the entire result.

### Fold-level detail, at the widest cap

The α = 0.050 row from the sweep above, broken out by fold. It is the least
flattering configuration and the one with the loosest risk budget, kept because
the fold breakdown is where the result actually lives. The baseline here is the
old fixed α = 1 % leg rather than the matched-size one, so read the *pattern*
across folds rather than the IR gap.

Clean universe, benchmark renormalised, all parameters refit per fold.

|  | RL (SAC) | Fixed α = 1 % |
|---|---|---|
| Ann. alpha | **3.05 %** | 0.19 % |
| Tracking error | 9.71 % | 7.76 % |
| IR | **0.314** | 0.025 |
| Hit rate | 46.2 % | 47.6 % |
| Max active DD | **−14.37 %** | −20.42 % |

| Fold | Months | RL IR | Fixed IR |
|---|---|---|---|
| 2014-2015 | 24 | −0.103 | −1.231 |
| 2016-2017 | 24 | 0.596 | 0.680 |
| 2018-2019 | 24 | 0.447 | −0.168 |
| 2020-2021 | 24 | 0.184 | 0.122 |
| 2022-2025 | 47 | 0.486 | 0.274 |

At this cap the overlay beats the fixed tilt in **4 of 5 folds**, across four
distinct regimes, with factor weights, state normalisation and the policy all
refit inside each fold's training slice. Tightening the cap to anything in the
tested range takes it to 5 of 5. The fold-level consistency is what makes the
result worth anything - a single concatenated IR of 0.31 on its own would not
be.

An earlier version reported IR 0.88 against 0.28. Those runs carried the same
benchmark-normalisation bug and contaminated universe as everything else here.

Two things to be clear about:

- **Tracking error is 9.71 %, not 2-4 %.** At this cap the agent picks a mean α
  of 2.61 % and tops out at the 5 % limit, so this configuration is not
  operating inside an index-enhancement mandate. The capped runs in the sweep
  above are the ones to quote.
- **It runs on factor-combo scores, not CS-Transformer scores.** So this is a
  different signal from the model comparison above, and the two results cannot
  be chained together.

### Data splits

The ranking models use a fixed three-way split; the RL overlay uses the 5-fold
expanding walk-forward described above instead, which is why its sample is 144
months rather than 12.

| Split | Period | Months |
|---|---|---|
| Train | 2010-2022 | ~156 |
| Validation | 2023-01 to 2024-06 | 18 |
| Test | 2024-07 onward | ~12+ |

## Key output files

The data files are regenerated by the pipeline (gitignored):

| File | Description |
|---|---|
| `data/prices.parquet` | daily OHLC, ~697 tickers |
| `data/panel_monthly_enriched.parquet` | monthly panel, ~270 ranked features |
| `data/fundamental.parquet` | quarterly fundamentals |
| `data/short_interest.parquet` | short interest snapshot |
| `data/institutional_ownership.parquet` | institutional holdings snapshot |
| `data/prediction_markets.parquet` | daily macro / policy signals |
| `data/insider_trades.parquet` | monthly insider filing counts |
| `data/sentiment.parquet` | monthly VADER sentiment aggregates |
| `data/sectors.parquet` | GICS sector mapping |
| `data/scores_cs_transformer.parquet` | CS-Transformer scores (from Kaggle) |
| `data/bt_ie_*.csv` | index-enhancement backtest monthly returns |
| `data/bt_wf_rl.csv` | walk-forward RL monthly returns |

## Config (`config.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `START_DATE` | 2010-01-01 | post-GFC |
| `TRAIN_END` | 2022-12-31 | model training cutoff |
| `VALID_END` | 2024-06-30 | validation window end |
| `TOP_N` | 100 | stocks overweighted each month |
| `BOTTOM_N` | 100 | stocks underweighted |
| `RETRAIN_EVERY` | 12 | months between retrains |
| `TRANSFORMER_CS_PARAMS["use_macro_film"]` | True | Macro FiLM on Stage 1 tokens |
| `TRANSFORMER_CS_PARAMS["use_corr_bias"]` | False | correlation attention bias |
| `RL_FINETUNE_PARAMS["method"]` | `"grpo"` | `grpo`, `dapo` or `hybrid` |

## Known limitations

Stated rather than discovered by a reader. Each is measured where it can be.

**The benchmark is approximate.** `1c` builds index weights as current shares
outstanding times historical price. Companies that are giants today were small
in 2005, so projecting today's share count backwards tilts the index toward the
names that went on to win. The rebuilt benchmark returns **15.7 %/yr over
2010-2024 against a real S&P 500 near 13.5 %**, so roughly 2.2pp hot, down from
5.7pp before the price repair. Top-10 concentration comes out at 31 % for 2005
against a real 20 %, 23 % for 2015 against 18 %, and 43 % for 2025 against
40 % - the error shrinks toward the present, which is the signature of this
cause. Weights also use total shares rather than free float, and the 8 % single
name cap is a guard against bad data rather than an index rule.

Active return differences out the benchmark level, so the information ratios
are largely unaffected. It does reach the absolute Sharpe figures, the claim
that tracking error is 2-4 % against the S&P 500 *specifically*, and capacity.

**The universe is incomplete.** 677 tickers against the roughly 1,200 that
passed through the index between 2010 and 2025, and 478-612 names per month
rather than 500. Survivorship is estimated at about 0.50 %/yr in
`1q_survivorship_bias.py`, but an estimate is not a correction.

**Two tickers are reused.** `SW` and `TMUS` refer to different companies
earlier in the history, so those series stitch two firms together. The weight
floor does not catch this.

**Institutional ownership is still empty.** Four 13F columns are declared and
never populated, so ownership and crowding are untested. Fundamentals and short
interest used to sit in this paragraph; they now carry real point-in-time data
from SEC EDGAR and FINRA, are tested above, and reach |t| 1.49 at best.

**Fundamental and short-interest history is shorter than the price history.**
EDGAR coverage gives 97-109 months against 156 for price factors, and FINRA's
API starts in 2017-12, so short interest has 60. Those factors are tested on
less data than everything else and their t-statistics are correspondingly
smaller.

**Six live index members are missing** (AVB, BK, CTRA, EA, EQR, HOLX) because
they will not refetch from the price provider.

Run `python 0_data_audit.py` before trusting any result. It checks these
invariants against values from outside the repo, which is the only way a silent
fault gets caught - every bug in the audit trail below was silent.

## Status / open work

- **Widen the universe past the S&P 500.** Size and illiquidity are the two
  strongest effects in the library and both are structurally underpowered in a
  universe selected for being large and liquid. A Russell 2000 or all-cap panel
  is the one change that could plausibly move the factor result, and the
  pipeline is universe-agnostic below `1b_fetch_constituents.py`.
- **13F institutional ownership**, the last declared-and-empty group. EDGAR
  serves it free on the same endpoint pattern `1u` already uses.
- **Quarterly historical shares outstanding** would close most of the remaining
  benchmark gap. `mktcap_shares.parquet` has them for 152 of 677 tickers.
- More OHLCV factor mining is the *weakest* remaining option: 70 mined
  candidates already underperform the hand-built factors, and every additional
  test raises the BHY bar for all of them.
