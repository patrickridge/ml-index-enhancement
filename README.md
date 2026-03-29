# Investsoc ML Project

ML-driven S&P 500 index enhancement pipeline. Three models (LightGBM, FT-Transformer, CS-Transformer) rank ~500 stocks monthly and tilt portfolio weights to beat the index with low tracking error. A two-layer Deep Reinforcement Learning (DRL) system then adaptively controls portfolio tilt each month.

See [Notes/STRATEGY.md](Notes/STRATEGY.md) for a full plain-English explanation of the methodology and results.
See [Notes/DEVELOPMENT_LOG.md](Notes/DEVELOPMENT_LOG.md) for a chronological record of all changes and findings.

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
python 1d2_parse_wind_prices.py        # parse Kieran's Missing data.xlsx → merge ~179 historical tickers
python 1e_ingest_wind_xlsx.py          # ingest Wind fundamentals xlsx (PE, ROE etc → data/wind_exports/)
python 1f_rebuild_panel.py             # build monthly panel from prices
python 1g_feature_engineering.py       # compute 43+ factors → data/panel_monthly_enriched.parquet
python 1h_orthogonalize.py             # PCA residualization (run after finalising feature set)
```

### Phase 2 — Factor Analysis
```bash
python 2a_factor_analysis.py           # IC, quintile, regime stability → data/factor_selected.csv
python 2b_ic_decay_all.py              # IC decay grid for all factors → figures/
python 2c_ic_decay_daily.py            # daily IC decay (1–90 trading days)
python 2d_factor_weights.py            # IC-decay based weights → data/factor_selected.csv
python 2e_ic_optimise.py               # gradient-optimised weights → data/factor_selected_optimised.csv
python 2f_factor_diagnostics.py        # IC correlation, VIF, RMT eigenvalue analysis (optional)
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
python 5a_rl_factor_agent.py           # Layer 1 SAC — adaptive factor IC weighting (requires torch)
python 5b_rl_portfolio_agent.py        # Layer 2 SAC — adaptive alpha tilt, 9-feature macro state (run after 5a)
python 5c_walk_forward.py              # Walk-forward backtest — 5 folds, no leakage, 95 OOS months
python 5d_algorithm_comparison.py      # SAC vs PPO vs GRPO walk-forward comparison (GRPO + KL penalty)
python 5e_dapo_agent.py                # DAPO vs GRPO — clip-higher + dynamic sampling (no KL)
```

---

## Out-of-Sample Results

### Index Enhancement (Test Period Jan 2023 – Nov 2025)

| Model | Ann. Alpha | Tracking Error | IR | Hit Rate |
|-------|-----------|---------------|-----|----------|
| CS-Transformer | 4.16% | 2.22% | **1.874** | ~67% |
| FT-Transformer | ~0.9% | ~2.6% | 0.438 | — |
| LGBM | ~0.5% | ~2.0% | 0.384 | — |
| Linear Factor Combo (baseline) | — | — | −0.046 | — |

IR > 0.5 is institutional-grade. IR > 1.0 is top-quartile. CS-Transformer is regime-stable across all market conditions.

### RL Walk-Forward Backtest (95 out-of-sample months, 2014–2025)

Fully expanding-window validation — factor weights, state normalisation, and RL policy all trained on past data only. No test-period information used anywhere in the pipeline.

| | RL Agent (SAC) | Fixed α=1% |
|---|---|---|
| **Ann. Alpha** | **7.19%** | 1.87% |
| **Tracking Error** | 8.18% | 6.77% |
| **Info Ratio** | **0.879** | 0.276 |
| **Hit Rate** | 50.5% | 48.4% |
| **Max Active DD** | **−6.04%** | −10.50% |

RL beats fixed alpha in 4/5 folds. Max drawdown cut nearly in half — the agent scales back tilt in high-volatility regimes and sizes up when factor IC is high.

| Fold | RL IR | Fixed IR |
|------|-------|----------|
| 2014–2015 | −0.755 | −1.659 ✓ |
| 2016–2017 | 1.771 | 1.778 |
| 2018–2019 | 0.042 | −0.073 ✓ |
| 2020–2021 | 1.125 | 0.751 ✓ |
| 2022–2025 | 1.188 | 0.380 ✓ |

### Algorithm Comparison — SAC vs PPO vs GRPO (5d)

Same 5-fold walk-forward structure. GRPO implements the DeepSeek-R1 (2025) algorithm with KL penalty.

| Algorithm | Type | IR | Notes |
|-----------|------|----|-------|
| GRPO | No critic, group ranking | **0.875** | DeepSeek-R1 method + KL penalty |
| PPO | On-policy, clipped | 0.870 | Critic baseline |
| SAC | Off-policy, replay buffer | 0.788 | Twin Q-critics, entropy reg |
| Fixed α=1% | Baseline | 0.276 | No RL |

GRPO and PPO outperform SAC overall — in a deterministic simulation environment, SAC's off-policy advantage is reduced. GRPO's critic-free group ranking is the best fit for the small-data (~100 training months) regime.

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
| `1e_ingest_wind_xlsx.py` | Ingest Wind platform XLSX exports |
| `1f_rebuild_panel.py` | Rebuild monthly panel from prices.parquet |
| `1g_feature_engineering.py` | Build 43+ factors → panel_monthly_enriched.parquet |
| `1h_orthogonalize.py` | PCA residualization for factor orthogonality |
| `2a_factor_analysis.py` | IC, quintile, regime stability, factor selection |
| `2b_ic_decay_all.py` | IC decay grid for all 43 factors |
| `2c_ic_decay_daily.py` | Daily IC decay (1–90 trading days) |
| `2d_factor_weights.py` | IC-decay based factor weights |
| `2e_ic_optimise.py` | Gradient-optimised factor weights (PyTorch) |
| `2f_factor_diagnostics.py` | IC correlation matrix, VIF, RMT eigenvalue analysis |
| `3a_ft_transformer.py` | FT-Transformer walk-forward backtest (local CPU) |
| `3b_ft_transformer_kaggle.py` | FT-Transformer (Kaggle GPU version) |
| `3c_cs_transformer.py` | CS-Transformer (local, slow) |
| `3d_cs_transformer_kaggle.py` | CS-Transformer (Kaggle GPU — primary) |
| `4a_factor_combo_baseline.py` | Linear factor combo baseline (no ML) |
| `4b_index_enhancement.py` | IE portfolio construction, alpha sweep |
| `4c_regime_engine.py` | Per-regime IE performance breakdown (rule-based + HMM) |
| `4d_benchmark_spx.py` | Compare all strategies vs S&P 500 |
| `5a_rl_factor_agent.py` | Layer 1 SAC — adaptive factor IC weighting across 43 factors |
| `5b_rl_portfolio_agent.py` | Layer 2 SAC — adaptive alpha tilt, 9-feature state (base + macro: VIX, yield, spread) |
| `5c_walk_forward.py` | Walk-forward RL backtest — 5 folds, no data leakage, 95 OOS months |
| `5d_algorithm_comparison.py` | SAC vs PPO vs GRPO walk-forward comparison — GRPO with KL penalty (DeepSeek-R1) |
| `5e_dapo_agent.py` | DAPO vs GRPO — clip-higher (ε↑=0.28) + dynamic sampling (G=2–8) + no KL |
| `1d2_parse_wind_prices.py` | Parse Missing data.xlsx → merge ~179 historical S&P 500 tickers into prices.parquet |
| `config.py` | All shared parameters and hyperparameters |
| `utils_factors.py` | 100+ factor computation functions (10 categories) |
| `utils_rmt.py` | Random Matrix Theory covariance denoising |

---

## Key Output Files

| File | Description |
|------|-------------|
| `data/prices.parquet` | Daily OHLC — 697 tickers, 2010–2025 |
| `data/panel_monthly_enriched.parquet` | Monthly panel — 43+ factors, cross-sectionally ranked |
| `data/factor_selected_optimised.csv` | 43 selected factors with IC-optimised weights |
| `data/scores_transformer.parquet` | FT-Transformer scores (test period) |
| `data/scores_cs_transformer.parquet` | CS-Transformer scores (from Kaggle) |
| `data/bt_ie_cs_transformer.csv` | CS-Transformer IE backtest monthly returns |
| `data/bt_ie_transformer.csv` | FT-Transformer IE backtest monthly returns |
| `data/bt_ie_lgbm.csv` | LGBM IE backtest monthly returns |
| `data/ie_regime_breakdown.csv` | Per-regime IR breakdown (all models) |
| `data/bt_wf_rl.csv` | Walk-forward RL monthly returns (95 OOS months) |
| `data/wf_fold_summary.csv` | Per-fold IR, alpha, TE summary |
| `data/algo_comparison.csv` | SAC vs PPO vs GRPO per-fold metrics |
| `data/dapo_comparison.csv` | DAPO vs GRPO per-fold metrics |
| `data/wind_parse_log.csv` | Log of tickers added/skipped from Missing data.xlsx |
| `data/l1_rl_ic_full.csv` | Layer 1 RL IC series (used as L2 state feature) |
| `figures/ie_regime_breakdown.png` | Bar chart: IR by regime per model |
| `figures/wf_rl_comparison.png` | Walk-forward cumulative alpha: RL vs fixed |
| `figures/algo_comparison.png` | SAC vs PPO vs GRPO cumulative alpha + per-fold IR |
| `figures/dapo_comparison.png` | DAPO vs GRPO cumulative alpha + per-fold IR |
| `figures/factor_ic_summary_positive.png` | Top 10 positive-IC factors |
| `figures/factor_weights_optimised.png` | Optimised factor weight distribution |

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
