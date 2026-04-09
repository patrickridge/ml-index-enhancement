# Investsoc ML Project

ML-driven S&P 500 index enhancement pipeline. Three models (LightGBM, FT-Transformer, CS-Transformer) rank ~500 stocks monthly and tilt portfolio weights to beat the index with low tracking error. A two-layer Deep Reinforcement Learning (DRL) system then adaptively controls portfolio tilt each month.

---

## Requirements

```
pip install -r requirements.txt
```

---

## How to Run — Full Pipeline (in order)

### Phase 1 — Data Prep

```bash
python 1a_price_parquet.py             # parse OHLC source → data/prices.parquet
python 1b_fetch_constituents.py        # historical S&P 500 constituent list
python 1c_fetch_market_cap.py          # monthly SPX weights via yfinance
python 1d_fetch_missing_tickers.py     # recover delisted/missing tickers via yfinance
python 1e_parse_wind_prices.py         # parse Kieran's Missing data.xlsx → merge ~179 historical tickers
python 1f_ingest_wind_xlsx.py          # ingest Wind fundamentals xlsx (PE, ROE etc → data/wind_exports/)
python 1g_rebuild_panel.py             # build monthly panel from prices

# Optional external data fetches (run once; rebuild periodically)
python 1j_fetch_simfin.py              # quarterly fundamentals via Simfin/yfinance → data/fundamental.parquet
python 1k_fetch_short_interest.py      # short interest snapshot via yfinance → data/short_interest.parquet
python 1l_fetch_13f.py                 # institutional ownership snapshot → data/institutional_ownership.parquet
python 1m_fetch_prediction_markets.py  # Fed/VIX/yield prediction market signals → data/prediction_markets.parquet

python 1h_feature_engineering.py       # compute 190+ factors across 21 categories → data/panel_monthly_enriched.parquet
python 1i_orthogonalize.py             # PCA residualization (run after finalising feature set)
```

Or run everything at once:
```bash
bash run_pipeline.sh             # fetch data + rebuild panel + run diagnostics
bash run_pipeline.sh --skip-fetch  # skip external data fetch (use existing files)
```

### Phase 2 — Factor Analysis & Diagnostics

```bash
python 2a_factor_analysis.py           # IC, quintile, regime stability → data/factor_selected.csv
python 2b_ic_decay_all.py              # IC decay grid for all factors → figures/
python 2c_ic_decay_daily.py            # daily IC decay (1–90 trading days)
python 2d_factor_weights.py            # IC-decay based weights → data/factor_selected.csv
python 2e_ic_optimise.py               # gradient-optimised weights → data/factor_selected_optimised.csv
python 2f_factor_diagnostics.py        # IC correlation, VIF, RMT eigenvalue analysis
python 2g_factor_decay_report.py       # 7-day IC decay report (IS vs OOS)
python 2h_factor_crowding.py           # correlation matrix, IC decay, contrarian & quintile audits
```

### Phase 3 — Models

```bash
python 3a_ft_transformer.py            # FT-Transformer (CPU, ~10–15 min) → data/scores_transformer.parquet
```

CS-Transformer requires Kaggle GPU:
```
1. Upload data/panel_monthly_enriched.parquet to Kaggle Dataset "investsoc-ml-data"
2. Create a Kaggle notebook, add that dataset, enable GPU T4
3. Run 3d_cs_transformer_kaggle.py
4. Download: scores_cs_transformer.parquet → copy to data/
```

### Phase 4 — Evaluation

```bash
python 4a_factor_combo_baseline.py     # linear factor combo baseline (no ML)
python 4b_index_enhancement.py         # IE portfolio construction → data/bt_ie_*.csv
python 4c_regime_engine.py             # regime breakdown (rule-based) → figures/ie_regime_breakdown.png
python 4c_regime_engine.py --hmm       # HMM 2-state regime breakdown
python 4d_benchmark_spx.py             # full comparison vs S&P 500
```

### Phase 5 — Deep Reinforcement Learning

```bash
python 5a_rl_factor_agent.py           # Layer 1 SAC — adaptive factor IC weighting
python 5b_rl_portfolio_agent.py        # Layer 2 SAC — adaptive alpha tilt, 9-feature macro state
python 5c_walk_forward.py              # Walk-forward backtest — 5 folds, no leakage, 95 OOS months
python 5d_algorithm_comparison.py      # SAC vs PPO vs GRPO walk-forward comparison
python 5e_dapo_agent.py                # DAPO vs GRPO — clip-higher + dynamic sampling + no KL
```

### Research — Factor Mining (IS-only discovery)

```bash
python research/factor_mining/run_factor_mining.py   # full pipeline: candidates → ML screen → BHY validation → catalog
```

Pipeline: ~70 new OHLCV candidates + entropy regime overlay → screen ALL features (existing 205 + new ~70) together through Lasso/RF/LightGBM with no pre-filter (L1/L2 penalties do the work) → BHY + dedup validation → `candidate_factor_catalog.csv`

### CS-Transformer Two-Stage Training (MSE + RL)

`3c_cs_transformer.py` now runs two-stage training:
1. **MSE pre-train** — standard masked MSE with L1/L2 feature tokenizer penalties
2. **RL fine-tune** — GRPO/DAPO portfolio-level reward (configurable via `config.py: RL_FINETUNE_PARAMS["method"]`)

Methods: `"grpo"` (KL-penalised, default), `"dapo"` (asymmetric clip, no KL), `"hybrid"` (regime-aware switching)

---

## Factor Categories (21 total, ~190 features)

| Cat | Name | Description |
|-----|------|-------------|
| 1 | Momentum | 1/3/6/12m returns, skip-month reversal |
| 2 | Volatility | Realised vol, idiosyncratic vol, downside vol |
| 3 | Tail Risk | CVaR, max drawdown, skewness, kurtosis |
| 4 | Price Trend | RSI, MACD, Bollinger, trend strength |
| 5 | Volume/Liquidity | Amihud illiquidity, turnover, Bid-Ask spread proxy |
| 6 | Beta/Correlation | Market beta, rolling beta, downside beta |
| 7 | Microstructure | Bid-ask spread, price impact, autocorrelation |
| 8 | Cross-sectional | SPX-relative return, sector-relative vol |
| 9 | Fundamentals | ROE, ROA, gross margin, debt/equity, growth, earnings quality (from `1j`) |
| 10 | Macro/Regime | VIX, yield curve, credit spread, PMI (cross-ticker) |
| 11 | Time-Signal v1 | Stock-level regime timing: ADX, trend consistency |
| 12 | Barra Style | Value, growth, leverage, earnings variability |
| 13 | Macro Interactions | Momentum × VIX, value × yield curve |
| 14 | Size | log(market cap) |
| 15 | Tail Ranking | Binary top/bottom decile dummies for base factors |
| 16 | Mined Alpha | 12 novel price signals: nearness to 52w high, residual momentum, up/down vol ratio, co-skewness, vol contraction, etc. |
| 17 | Time-Signal v2 | ADX regime, roll spread, autocorr, vol expansion, trend strength |
| 18 | Seasonality | Same-month return (Heston & Sadka 2008), turn-of-month, January dummy |
| 19 | Short Interest | short % of float, days-to-cover, short change, squeeze risk (from `1k`) |
| 20 | Institutional | inst. ownership %, change, # holders, concentration HHI (from `1l`) |
| 21 | Prediction Markets | Fed hike/cut probability, recession probability, VIX term structure, policy uncertainty (from `1m`) |

**Crowding/selection rules (Kieran):**
- Correlation filter applied **after** backtest, not before training
- OOS IC is diagnostic only — never used for factor selection (lookahead bias)
- Prediction market factors used as **when-to-trade** signals (time-signals), not cross-sectional stock signals

---

## Out-of-Sample Results

### Index Enhancement (Test Period Jan 2023 – Nov 2025)

| Model | Ann. Alpha | Tracking Error | IR | Hit Rate |
|-------|-----------|---------------|-----|----------|
| CS-Transformer | 4.16% | 2.22% | **1.874** | ~67% |
| FT-Transformer | ~0.9% | ~2.6% | 0.438 | — |
| LGBM | ~0.5% | ~2.0% | 0.384 | — |
| Linear Factor Combo (baseline) | — | — | −0.046 | — |

IR > 0.5 is institutional-grade. IR > 1.0 is top-quartile.

### RL Walk-Forward Backtest (95 out-of-sample months, 2014–2025)

| | RL Agent (SAC) | Fixed α=1% |
|---|---|---|
| **Ann. Alpha** | **7.19%** | 1.87% |
| **Tracking Error** | 8.18% | 6.77% |
| **Info Ratio** | **0.879** | 0.276 |
| **Hit Rate** | 50.5% | 48.4% |
| **Max Active DD** | **−6.04%** | −10.50% |

| Fold | RL IR | Fixed IR |
|------|-------|----------|
| 2014–2015 | −0.755 | −1.659 ✓ |
| 2016–2017 | 1.771 | 1.778 |
| 2018–2019 | 0.042 | −0.073 ✓ |
| 2020–2021 | 1.125 | 0.751 ✓ |
| 2022–2025 | 1.188 | 0.380 ✓ |

### Algorithm Comparison — SAC vs PPO vs GRPO

| Algorithm | IR | Notes |
|-----------|----|-------|
| GRPO | **0.875** | DeepSeek-R1 method + KL penalty |
| PPO | 0.870 | Critic baseline |
| SAC | 0.788 | Twin Q-critics, entropy reg |
| Fixed α=1% | 0.276 | No RL |

### Data Splits

| Split | Period | Months |
|-------|--------|--------|
| Train | 2010–2020 | ~132 |
| Valid | 2021–2022 | ~24 |
| Test  | 2023–2025 | ~35+ |

---

## File Guide

| File | Purpose |
|------|---------|
| `1a_price_parquet.py` | Parse OHLC data → prices.parquet |
| `1b_fetch_constituents.py` | Download full historical S&P 500 constituent list |
| `1c_fetch_market_cap.py` | Fetch monthly SPX constituent weights via yfinance |
| `1d_fetch_missing_tickers.py` | Secondary yfinance fetch for missing/delisted tickers |
| `1e_parse_wind_prices.py` | Parse Missing data.xlsx → merge ~179 historical tickers |
| `1f_ingest_wind_xlsx.py` | Ingest Wind platform XLSX exports |
| `1g_rebuild_panel.py` | Rebuild monthly panel from prices.parquet |
| `1h_feature_engineering.py` | Build 190+ factors (21 categories) → panel_monthly_enriched.parquet |
| `1i_orthogonalize.py` | PCA residualization for factor orthogonality |
| `1j_fetch_simfin.py` | Quarterly fundamentals via Simfin or yfinance fallback |
| `1k_fetch_short_interest.py` | Short interest snapshot (yfinance, run periodically) |
| `1l_fetch_13f.py` | Institutional ownership snapshot (yfinance, run quarterly) |
| `1m_fetch_prediction_markets.py` | Fed/VIX/yield prediction market signals (daily, 2010–present) |
| `2a_factor_analysis.py` | IC, quintile, regime stability — IS diagnostic only |
| `2b_ic_decay_all.py` | IC decay grid for all factors |
| `2c_ic_decay_daily.py` | Daily IC decay (1–90 trading days) |
| `2d_factor_weights.py` | IC-decay based factor weights |
| `2e_ic_optimise.py` | Gradient-optimised factor weights (PyTorch) |
| `2f_factor_diagnostics.py` | IC correlation matrix, VIF, RMT eigenvalue analysis |
| `2g_factor_decay_report.py` | 7-day IC decay IS vs OOS report |
| `2h_factor_crowding.py` | Correlation matrix, IC decay, contrarian factors, quintile direction audit |
| `3a_ft_transformer.py` | FT-Transformer walk-forward backtest (local CPU) |
| `3b_ft_transformer_kaggle.py` | FT-Transformer (Kaggle GPU version) |
| `3c_cs_transformer.py` | CS-Transformer (local, slow) |
| `3d_cs_transformer_kaggle.py` | CS-Transformer (Kaggle GPU — primary) |
| `4a_factor_combo_baseline.py` | Linear factor combo baseline (no ML) |
| `4b_index_enhancement.py` | IE portfolio construction, alpha sweep |
| `4c_regime_engine.py` | Per-regime IE performance breakdown (rule-based + HMM) |
| `4d_benchmark_spx.py` | Compare all strategies vs S&P 500 |
| `5a_rl_factor_agent.py` | Layer 1 SAC — adaptive factor IC weighting |
| `5b_rl_portfolio_agent.py` | Layer 2 SAC — adaptive alpha tilt, 9-feature macro state |
| `5c_walk_forward.py` | Walk-forward RL backtest — 5 folds, 95 OOS months |
| `5d_algorithm_comparison.py` | SAC vs PPO vs GRPO walk-forward comparison |
| `5e_dapo_agent.py` | DAPO vs GRPO — clip-higher + dynamic sampling + no KL |
| `5f_dynamic_portfolio_rl.py` | 2D asymmetric tilt (alpha_long, alpha_short) |
| `config.py` | All shared parameters and hyperparameters |
| `utils_factors.py` | 190+ factor functions across 21 categories |
| `utils_rmt.py` | Random Matrix Theory covariance denoising |
| `run_pipeline.sh` | Automated: fetch data → rebuild panel → analyse → diagnostics |
| `research/factor_mining/run_factor_mining.py` | Master factor mining orchestrator |
| `research/factor_mining/candidate_factory.py` | ~70 OHLCV candidate features (9 families) |
| `research/factor_mining/validation_engine.py` | IS-only screening: IC, ICIR, RAS, BHY, dedup |
| `research/factor_mining/screen_lasso.py` | Lasso/Elastic Net with purged time-series CV |
| `research/factor_mining/screen_trees.py` | RF + LightGBM feature importance screening |
| `research/factor_mining/screen_autoencoder.py` | Autoencoder latent features from OHLCV windows |
| `research/factor_mining/regime_entropy.py` | Singha-inspired entropy vol regime detector |

---

## Key Output Files

| File | Description |
|------|-------------|
| `data/prices.parquet` | Daily OHLC — 697 tickers, 2010–2025 |
| `data/panel_monthly_enriched.parquet` | Monthly panel — 190+ factors, cross-sectionally ranked |
| `data/fundamental.parquet` | Quarterly fundamentals — ROE, ROA, margins, growth (from 1j) |
| `data/short_interest.parquet` | Short interest snapshot — % float, days-to-cover, etc. (from 1k) |
| `data/institutional_ownership.parquet` | Institutional ownership — % owned, # holders, HHI (from 1l) |
| `data/prediction_markets.parquet` | Daily Fed/VIX/yield signals — 6 factors, 2010–2026 (from 1m) |
| `data/factor_selected_optimised.csv` | Selected factors with IC-optimised weights |
| `data/scores_transformer.parquet` | FT-Transformer scores (test period) |
| `data/scores_cs_transformer.parquet` | CS-Transformer scores (from Kaggle) |
| `data/bt_ie_cs_transformer.csv` | CS-Transformer IE backtest monthly returns |
| `data/bt_wf_rl.csv` | Walk-forward RL monthly returns (95 OOS months) |
| `data/algo_comparison.csv` | SAC vs PPO vs GRPO per-fold metrics |

---

## Shared Config (`config.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `START_DATE` | 2010-01-01 | Post-GFC baseline |
| `TRAIN_END` | 2020-12-31 | Factor analysis and model training cutoff |
| `VALID_END` | 2022-12-31 | Validation window end |
| `TOP_N` | 100 | Stocks overweighted each month |
| `BOTTOM_N` | 100 | Stocks underweighted each month |
| `RETRAIN_EVERY` | 12 | Model retrain frequency (months) |
