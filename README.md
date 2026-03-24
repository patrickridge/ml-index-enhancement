# Investsoc ML Project

ML-driven S&P 500 index enhancement pipeline. Three models (LightGBM, FT-Transformer, CS-Transformer) rank ~500 stocks monthly and tilt portfolio weights to beat the index with low tracking error.

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
python 1e_ingest_wind_xlsx.py          # ingest Wind xlsx exports (drop files in data/wind_exports/)
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

### Phase 5 — Reinforcement Learning
```bash
python 5b_rl_portfolio_agent.py        # SAC agent — adaptive alpha tilt (requires torch)
```

---

## Out-of-Sample Results (Test Period Jan 2023 – Oct 2025)

### Index Enhancement

| Model | Ann. Alpha | Tracking Error | IR | Hit Rate |
|-------|-----------|---------------|-----|----------|
| CS-Transformer | ~2.2% | ~2.3% | **0.960** | ~67% |
| FT-Transformer | ~0.9% | ~2.6% | 0.438 | — |
| LGBM | ~0.5% | ~2.0% | 0.384 | — |
| Linear Factor Combo (baseline) | — | — | −0.046 | — |

IR > 0.5 is top-quartile. IR > 0.9 is strong. CS-Transformer is also regime-stable (IR 1.02 risk-off, 0.93 risk-on).

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
| `5b_rl_portfolio_agent.py` | SAC RL agent — learns adaptive alpha tilt (MLP, no memory) |
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
| `figures/ie_regime_breakdown.png` | Bar chart: IR by regime per model |
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
