# Deep-Learning Index Enhancement on the S&P 500

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

This repo is a working implementation of that strategy on the S&P 500. Test
window (2023–2025, 35 months): **IR 1.87** at 2.22% tracking error, ~67% hit
rate. Written as a working record of what was built and what the numbers
actually show - not a marketing page.

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

## Why deep learning

Tree ensembles like LightGBM look at each stock in isolation. Ranking 500
stocks against each other is inherently relational - a volatility reading of
30% means one thing when most of the universe is at 15% and something else
when most are at 45%, and the signal strength at the *tails* of the
cross-section matters more than absolute levels. A Cross-Sectional
Transformer handles this directly: a first attention stage operates within
each stock's feature vector; a second stage attends across every stock in
the universe that month, so each score is aware of its peers. Macro state
(VIX, yield curve, Fed rate expectations) is injected through FiLM
conditioning so the model knows what regime it's scoring in.

Training is two-stage. First a standard masked-MSE pre-train so the model
learns cross-sectional rank structure. Then a portfolio-level RL fine-tune
(GRPO or DAPO) that optimises portfolio return directly - bridging the gap
between "predicting labels" and "actually making money", since those two
objectives are not the same.

## Why reinforcement learning on top

The tracking-error budget is a hard constraint, not a soft preference. A
fixed α over-reacts in some regimes (e.g. when the cross-section has
collapsed and every stock looks the same, aggressive tilting just adds
tracking error without alpha) and under-reacts in others (in a dispersed
month with a clear top/bottom split, a timid tilt wastes signal). A learned
policy adapts tilt size to what the market is actually doing.

Validation is a 5-fold expanding-window walk-forward from 2014 to 2025 - no
shared training data between folds, factor weights refitted from scratch
each fold, agent retrained fresh. On 95 OOS months the SAC overlay delivers
**IR 0.88 vs 0.28** for the fixed-α baseline, with max active drawdown
almost halved (−6.04% vs −10.50%). GRPO and PPO reach similar IR; SAC wins
on sample efficiency given the ~100-month fold sizes.

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

### Index enhancement over the OOS test period (Jan 2023 – Nov 2025)

| Model | Ann. alpha | Tracking error | IR | Hit rate |
|---|---|---|---|---|
| CS-Transformer | 4.16 % | 2.22 % | **1.87** | ~67 % |
| FT-Transformer | ~0.9 % | ~2.6 % | 0.44 | - |
| LightGBM | ~0.5 % | ~2.0 % | 0.38 | - |
| Linear factor combo baseline | - | - | −0.05 | - |

For context: IR > 0.5 is institutional-grade, > 1.0 is top-quartile.

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

## Authors

Patrick Ridge · Kieran Chung
