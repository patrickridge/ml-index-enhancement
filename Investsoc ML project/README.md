# Investsoc ML Project

LightGBM + FT-Transformer + Cross-Sectional Transformer pipeline for S&P 500 stock selection.
100+ engineered factors, Random Matrix Theory denoising, and PCA factor orthogonalization.

See [STRATEGY.md](STRATEGY.md) for a plain-English explanation of the methodology and results.

## Requirements

```
pip install -r requirements.txt
```

Place `data.xlsx` (OHLC data, one block per price field) in the project root.

---

## How to Run — Full Pipeline (in order)

### Step 1 — Data Prep
```bash
python "1 price parquet.py"           # parse data.xlsx → data/prices.parquet
```

### Step 2 — Feature Engineering (100+ factors, ~5–10 min)
```bash
python 1_feature_engineering.py       # 100+ factors → data/panel_monthly_enriched.parquet
python 1b_orthogonalize.py            # PCA residualization → data/panel_monthly_orthogonalized.parquet
```

### Step 3 — Factor Diagnostics (optional, read-only, ~2 min)
```bash
python 7_factor_diagnostics.py        # IC corr matrix, VIF, RMT eigenvalue plots → data/diag_*
```

### Step 4 — Models (run locally on CPU)
```bash
python 2_lgbm_backtest.py             # ~3–5 min  → data/scores_lgbm.parquet
python 2b_nn_backtest.py              # ~10–15 min → data/scores_transformer.parquet
python 3_pca_rp_backtest.py           # ~1–2 min  → data/bt_monthly_pca_rp.csv
```

### Step 5 — Cross-Sectional Transformer (run on Kaggle GPU)
The cross-sectional transformer processes all ~500 stocks simultaneously each month.
Too slow for CPU (~60 min); run on Kaggle T4 GPU (~5–10 min).

```
1. Upload data/panel_monthly_enriched.parquet to a Kaggle Dataset called "investsoc-ml-data"
2. Create a Kaggle notebook, add that dataset, enable GPU T4
3. Upload 2c_cs_transformer_kaggle.py and run it (or paste into a code cell)
4. Download: scores_cs_transformer.parquet, bt_cs_transformer.csv, bt_cs_transformer_ls.csv
5. Copy downloaded files into your local data/ folder
```

### Step 6 — Benchmark Comparison
```bash
python 4_benchmark_spx.py             # compare all models vs S&P 500
```

### Step 7 — Stock Pitch Validation (interactive)
```bash
python 5_pitch_validator.py           # enter any ticker for score + signal breakdown
```

---

## File Guide

| File | Type | Purpose |
|------|------|---------|
| `1 price parquet.py` | Script | Parse data.xlsx → prices.parquet |
| `1_feature_engineering.py` | Script | Build 100+ factors from OHLCV + macro data |
| `1b_orthogonalize.py` | Script | PCA residualization for factor orthogonality |
| `2_lgbm_backtest.py` | Script | LightGBM walk-forward model |
| `2b_nn_backtest.py` | Script | FT-Transformer (per-stock) walk-forward model |
| `2c_cs_transformer.py` | Script | Cross-sectional transformer (local version) |
| `2c_cs_transformer_kaggle.py` | Script | Cross-sectional transformer (Kaggle GPU version) |
| `3_pca_rp_backtest.py` | Script | PCA Risk Parity portfolio construction |
| `4_benchmark_spx.py` | Script | Compare all strategies vs S&P 500 |
| `5_pitch_validator.py` | Script | Interactive stock signal lookup |
| `6 Diagonstic test.py` | Script | Turnover, rank decay, covariance diagnostics |
| `7_factor_diagnostics.py` | Script | IC correlation, VIF, RMT eigenvalue analysis |
| `config.py` | Config | All shared parameters and hyperparameters |
| `utils_factors.py` | Utility | 100+ factor computation functions (10 categories) |
| `utils_rmt.py` | Utility | Random Matrix Theory covariance denoising |
| `check_overfit.py` | Diagnostic | Train/valid/test Sharpe comparison |
| `inspect_data.py` | Diagnostic | Print data shapes and date ranges |

---

## Data Splits

| Split | Period | Months |
|-------|--------|--------|
| Train | 2010–2020 | ~132 |
| Valid | 2021–2022 | ~24  |
| Test  | 2023–2025 | ~35+ |

---

## Out-of-Sample Performance (test period, 2023–2025)

| Strategy | Sharpe | Ann. Return | Beta | Alpha | Max DD |
|----------|--------|-------------|------|-------|--------|
| LGBM Long-Only (Top 50) | **1.54** | 33.7% | 0.26 | +30.3% | -17.8% |
| LGBM Long-Short | 0.83 | 11.7% | 0.15 | +9.5% | -8.7% |
| Transformer Long-Only | 1.32 | 32.3% | 0.22 | +30.6% | -20.2% |
| Transformer Long-Short | 0.97 | 20.7% | 0.29 | +16.9% | -18.2% |
| CS Transformer Long-Only | TBD | TBD | — | — | — |
| S&P 500 (benchmark) | 1.58 | 18.0% | 1.00 | — | — |
| LGBM + PCA-RP (fixed) | TBD | TBD | ~1.1 | — | — |

CS Transformer and fixed PCA-RP results pending. PCA-RP beta bug fixed via RMT covariance denoising.

---

## Shared Config

`config.py` contains all key parameters: date splits, LGBM params, Transformer params, portfolio params,
and new flags: `USE_ORTHOGONALIZED_FEATURES`, `USE_RMT_COV_DENOISING`, `MACRO_COLS`, `TRANSFORMER_CS_PARAMS`.

---

## Output Files

| File | Description |
|------|-------------|
| `data/prices.parquet` | Daily OHLC in long format |
| `data/panel_monthly_enriched.parquet` | Monthly panel (100+ factors, cross-sectionally ranked) |
| `data/panel_monthly_orthogonalized.parquet` | PCA-residualized version of enriched panel |
| `data/scores_lgbm.parquet` | LightGBM scores (test period) |
| `data/scores_transformer.parquet` | FT-Transformer scores (test period) |
| `data/scores_cs_transformer.parquet` | Cross-Sectional Transformer scores (from Kaggle) |
| `data/bt_lgbm.csv` | LGBM long-only top-50 monthly returns |
| `data/bt_lgbm_ls.csv` | LGBM long-short monthly returns |
| `data/bt_transformer.csv` | FT-Transformer long-only monthly returns |
| `data/bt_transformer_ls.csv` | FT-Transformer long-short monthly returns |
| `data/bt_cs_transformer.csv` | CS-Transformer long-only monthly returns |
| `data/bt_cs_transformer_ls.csv` | CS-Transformer long-short monthly returns |
| `data/bt_monthly_pca_rp.csv` | PCA-RP monthly returns |
| `data/benchmark_comparison.csv` | All strategies vs SPX summary table |
| `data/diag_ic_corr_heatmap.png` | Factor IC correlation heatmap |
| `data/diag_vif.csv` | Variance Inflation Factor table |
| `data/diag_rmt_eigenvalues.csv` | RMT eigenvalue analysis |
