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

**The result is negative.** Once the universe is cleaned of delisted tickers
with corrupt price data, neither transformer beats the benchmark
out-of-sample. A LightGBM baseline is the only model with positive alpha, at
IR 0.56. An earlier version of this README reported IR 1.87 for the
CS-Transformer; that number was an artifact, and the audit that found it is
documented in full below.

Written as a working record of what was built and what the numbers actually
show, including where they turned out to be wrong.

## How the strategy works

1. **Score the universe every month.** A Cross-Sectional Transformer reads a
   panel of ~270 features per stock (momentum, volatility, fundamentals,
   short interest, institutional ownership, insider filings, news sentiment,
   macro regime signals, mined OHLCV candidates) and produces one score per
   stock.
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

**The corrected results do not support the hypothesis.** On clean data the
CS-Transformer posts IR −0.54 against LightGBM's +0.56, and it is negative at
every tilt size tested. Whether that reflects the architecture, the training
setup, the 17-month evaluation window, or the fact that its scores predate the
Macro FiLM upgrade is not resolved here. What can be said is that the
cross-sectional attention did not earn its complexity on this data.

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

> **Numbers under re-validation.** Earlier runs reported IR 0.88 for the SAC
> overlay against 0.28 for a fixed-α baseline. Those runs carried the same
> benchmark-normalisation bug and the same contaminated universe as the index
> enhancement results above. `5b`, `5c` and `5d` have been fixed and are being
> re-run; the previous figures should not be quoted until they are replaced.

## What the pipeline does

1. **Data prep** (`1*` scripts): pull OHLCV, constituent history, fundamentals, short interest,
   institutional ownership, prediction markets, insider trades, news sentiment, sector mappings.
2. **Feature engineering** (`1h_feature_engineering.py`): compute ~270 factors across 24 categories
   (momentum, volatility, fundamentals, macro, mined-alpha candidates, insider, sentiment, etc.).
   Per-stock features are cross-sectionally ranked each month; macro features are kept at raw scale.
3. **Factor diagnostics** (`2*` scripts): IC, IC decay, quintile returns, crowding/correlation,
   IS-only screening of new candidates via Lasso / RF / LightGBM and BHY.
4. **Models** (`3*` scripts):
   - `3a_ft_transformer.py` - FT-Transformer (independent per-stock ranker)
   - `3c_cs_transformer.py` / `3d_cs_transformer_kaggle.py` - Cross-Sectional Transformer with
     Macro FiLM conditioning, optional correlation attention bias and a GRPO/DAPO RL fine-tune
     on portfolio return.
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
├── 2a–2h_*.py                      factor diagnostics
├── 3a–3d_*.py                      ranking models
├── 3e_cs_transformer_audit.py      post-hoc diagnostics on CS-T scores
├── 3e_hp_sweep{,_kaggle}.py        CS-Transformer hyperparameter sweeps
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
| 9 | Fundamentals | ROE, ROA, margins, growth, EQ (`1j`) |
| 10 | Macro / regime | VIX, yield curve, credit, PMI |
| 11 | Time-signal v1 | ADX, trend consistency |
| 12 | Barra-style | value, growth, leverage, EY |
| 13 | Macro × factor interactions | e.g. momentum × VIX |
| 14 | Size | log market cap |
| 15 | Tail ranking | top/bottom-decile dummies |
| 16 | Mined alpha | 12 hand-mined price signals |
| 17 | Time-signal v2 | ADX regime, roll spread, vol expansion |
| 18 | Seasonality | same-month, turn-of-month, January |
| 19 | Short interest | % float, days-to-cover (`1k`) |
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

| Model | Window | Months | Ann. alpha | Tracking error | IR |
|---|---|---|---|---|---|
| LightGBM | 2023-01 to 2025-11 | 35 | +0.83 % | 1.47 % | **+0.56** |
| FT-Transformer | 2023-01 to 2025-11 | 39 | −0.74 % | 1.44 % | −0.52 |
| CS-Transformer | 2024-07 to 2025-11 | 17 | −2.52 % | 4.64 % | −0.54 |

The headline finding of this project is negative: **neither transformer beats
the benchmark once the universe is clean.** The gradient-boosting baseline they
were built to improve on is the only model with positive out-of-sample alpha,
and even that is modest.

Both transformers are negative at *every* α in the sweep, and their alpha
degrades monotonically as the tilt grows — CS-Transformer runs from −0.19 % at
the smallest α to −5.35 % at the largest. That is the signature of a signal
that is actively wrong rather than merely absent: acting on it harder loses
more. LightGBM degrades in the opposite direction (IR 0.66 at small α down to
0.08 at large), which is what a weak but genuine signal looks like.

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

### RL walk-forward (95 OOS months, 2014–2025)

|  | RL (SAC) | Fixed α = 1 % |
|---|---|---|
| Ann. alpha | **7.19 %** | 1.87 % |
| Tracking error | 8.18 % | 6.77 % |
| IR | **0.88** | 0.28 |
| Hit rate | 50.5 % | 48.4 % |
| Max active DD | **−6.04 %** | −10.50 % |

### RL algorithm comparison

| Algorithm | IR | Notes |
|---|---|---|
| GRPO | **0.88** | PPO clipping + KL penalty |
| PPO | 0.87 | critic baseline |
| SAC | 0.79 | twin Q-critics, entropy reg |
| Fixed α = 1 % | 0.28 | no RL |

### Data splits

| Split | Period | Months |
|---|---|---|
| Train | 2010–2022 | ~156 |
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

## Status / open work

- Survivorship bias: the panel currently has 697 tickers vs the full ~1 200 that passed through
  the S&P 500 between 2010 and 2025. OOS numbers are not fully trustworthy until the history is
  filled in.
- The CS-Transformer scores currently in `data/` predate the Macro FiLM + 69-candidate upgrade.
  A fresh Kaggle retrain is the next step.
- Fundamental data is only a handful of recent quarterly snapshots. A Simfin API key pulls the
  full history in a few minutes.
