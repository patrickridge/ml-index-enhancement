# Investsoc ML Project

LightGBM + PCA Risk Parity quantitative trading pipeline.

## Requirements

```
pip install -r requirements.txt
```

Place `data.xlsx` (OHLC data, one block per price field) in the project root.

## Main Pipeline

Run scripts in order:

```bash
python "1 price parquet.py"        # parse data.xlsx → data/prices.parquet
python 1_feature_engineering.py    # build 25 features → data/panel_monthly_enriched.parquet
python 2_lgbm_backtest.py          # train LGBM, walk-forward scores → data/scores_lgbm.parquet
python 3_pca_rp_backtest.py        # PCA-RP weights + backtest → data/bt_monthly_pca_rp.csv
python 4_benchmark_spx.py          # compare vs S&P 500
```

## Standalone Alternative Pipeline

```bash
python "5.1 lgbm pca rp backtest.py"   # self-contained LGBM + PCA-RP in one script
```

## Data Splits (main pipeline)

| Split | Period | Months |
|-------|--------|--------|
| Train | 2010–2017 | ~96 |
| Valid | 2018–2019 | ~24 |
| Test  | 2020+ | ~60 |

## Performance (test set, main pipeline)

| Metric | Value |
|--------|-------|
| Ann. Return | ~12% |
| Volatility | ~17% |
| Sharpe | ~0.69 |

## Output Files

| File | Description |
|------|-------------|
| `data/prices.parquet` | Daily OHLC in long format |
| `data/panel_monthly_enriched.parquet` | Monthly feature panel (25 features) |
| `data/scores_lgbm.parquet` | Per-stock LGBM scores (test period) |
| `data/bt_lgbm.csv` | Long-only top-50 backtest returns |
| `data/bt_lgbm_ls.csv` | Long-short backtest returns |
| `data/weights_pca_rp.parquet` | Monthly portfolio weights |
| `data/nav_daily_pca_rp.parquet` | Daily NAV |
| `data/bt_monthly_pca_rp.csv` | Monthly backtest returns |

## Shared Config

`config.py` contains shared constants (date splits, LGBM params, portfolio params) used by the main pipeline scripts.
