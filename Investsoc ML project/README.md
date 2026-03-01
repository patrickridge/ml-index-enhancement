# Investsoc ML Project

LightGBM + FT-Transformer quantitative trading pipeline for S&P 500 stock selection.

See [STRATEGY.md](STRATEGY.md) for a plain-English explanation of the methodology, assumptions, and results.

## Requirements

```
pip install -r requirements.txt
```

Place `data.xlsx` (OHLC data, one block per price field) in the project root.

## Pipeline — Run in Order

```bash
python "1 price parquet.py"        # parse data.xlsx → data/prices.parquet
python 1_feature_engineering.py    # build 26 features → data/panel_monthly_enriched.parquet
python 2_lgbm_backtest.py          # train LightGBM, walk-forward scores + backtest
python 2b_nn_backtest.py           # train FT-Transformer, walk-forward scores + backtest
python 3_pca_rp_backtest.py        # PCA-RP weights + backtest (experimental)
python 4_benchmark_spx.py          # compare all strategies vs S&P 500
```

## Optional / Diagnostic Scripts

```bash
python 5_pitch_validator.py AAPL         # score + signal breakdown for any ticker
python 5_pitch_validator.py AAPL MSFT    # side-by-side comparison
jupyter notebook report.ipynb            # full results report with charts
python "6 Diagonstic test.py"            # turnover, rank decay, covariance diagnostics
python check_overfit.py                  # quick train/valid/test Sharpe check
python inspect_data.py                   # print data shapes and date ranges
```

## Data Splits

| Split | Period | Months |
|-------|--------|--------|
| Train | 2010–2020 | ~132 |
| Valid | 2021–2022 | ~24  |
| Test  | 2023–2025 | ~24+ |

## Out-of-Sample Performance (test period, 2023–2025)

| Strategy | Sharpe | Ann. Return | Beta | Alpha | Max DD |
|----------|--------|-------------|------|-------|--------|
| LGBM Long-Only (Top 50) | **1.54** | 33.7% | 0.26 | +30.3% | -17.8% |
| LGBM Long-Short | 0.83 | 11.7% | 0.15 | +9.5% | -8.7% |
| Transformer Long-Only | 1.32 | 32.3% | 0.22 | +30.6% | -20.2% |
| Transformer Long-Short | 0.97 | 20.7% | 0.29 | +16.9% | -18.2% |
| S&P 500 (benchmark) | 1.58 | 18.0% | 1.00 | — | — |
| LGBM + PCA-RP ⚠ | — | — | 2.07 | -7.6% | — |

Alpha = annualised excess return above what market beta explains. PCA-RP has a known beta bug — treat as experimental.

## Shared Config

`config.py` contains all key parameters (date splits, LGBM hyperparameters, Transformer hyperparameters, portfolio params).

## Output Files

| File | Description |
|------|-------------|
| `data/prices.parquet` | Daily OHLC in long format |
| `data/panel_monthly_enriched.parquet` | Monthly feature panel (26 features, cross-sectionally ranked) |
| `data/scores_lgbm.parquet` | Per-stock LightGBM scores (test period) |
| `data/scores_transformer.parquet` | Per-stock FT-Transformer scores (test period) |
| `data/bt_lgbm.csv` | LGBM long-only top-50 monthly returns |
| `data/bt_lgbm_ls.csv` | LGBM long-short monthly returns |
| `data/bt_transformer.csv` | Transformer long-only monthly returns |
| `data/bt_transformer_ls.csv` | Transformer long-short monthly returns |
| `data/bt_monthly_pca_rp.csv` | PCA-RP monthly returns (experimental) |
| `data/benchmark_comparison.csv` | All strategies vs SPX summary table |
| `data/weights_pca_rp.parquet` | Monthly PCA-RP portfolio weights |
