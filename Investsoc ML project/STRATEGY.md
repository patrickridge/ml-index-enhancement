# ML Stock Selection Strategy — Simple Explainer

## What Are We Trying to Do?

Every month, pick the stocks most likely to go up next month. We use machine learning to rank ~500 S&P 500 stocks and build a portfolio from the top-ranked ones.

---

## The Data

- **Universe:** ~500 S&P 500 stocks
- **Time range:** 2010 – 2025 (post-GFC, cleaner regime)
- **Frequency:** Monthly predictions, daily prices for risk weighting
- **Target:** Each stock's return over the next calendar month (`fwd_ret_1m`)

---

## The Features (26 total)

These are signals computed for every stock each month — things the model learns from:

| Category | Examples |
|---|---|
| **Momentum** | 1-month, 3-month, 6-month, 12-month past returns |
| **Reversal** | Very short-term (1-week) return — tends to mean-revert |
| **Volatility** | Realised vol, vol ratio (recent vs longer-term) |
| **Liquidity** | Amihud illiquidity ratio (how hard is it to trade?) |
| **Trend** | Price vs 50-day and 200-day moving average |
| **Tail risk** | Skewness, max drawdown, max single-day gain |
| **Quality** | Information ratio (consistency of recent returns) |

No accounting data, no fundamentals — purely price-based signals.

---

## The Model: LightGBM

LightGBM is a gradient-boosted decision tree model. Think of it as a very smart ranking engine:

1. It looks at all 26 features for every stock every month
2. It learns which combinations of features predicted high returns in the past
3. Each month it outputs a **score** for every stock — higher score = more likely to outperform

### How We Train It (Walk-Forward)

We never let the model see future data:

```
2010 ──────────────── 2021 ── 2023 ── 2025
│← Training (~132m) →│ Valid │← Test →│
                      │ (24m) │
```

- **Train (2010–2020):** Model learns from historical data
- **Valid (2021–2022):** Early stopping — we stop training when improvement stalls
- **Test (2023–2025):** Never seen during training — real out-of-sample results

Every 12 months in the test period, we retrain the model with more data (expanding window).

---

## Portfolio Construction

### Strategy 1: Long-Only (Top 50)
- Each month: take the 50 highest-scoring stocks
- Equal-weight them (2% each)
- Hold for one month, then rebalance

### Strategy 2: Long-Short
- Long the top 10% of stocks by score (~50 stocks)
- Short the bottom 10% of stocks by score (~50 stocks)
- Market-neutral: gains come from the spread, not market direction

### Strategy 3: PCA Risk Parity (experimental)
- Same top-50 selection, but weights from PCA-based risk decomposition
- Currently has a beta problem (see below) — treat as experimental

---

## Results (Out-of-Sample, 2023–2025)

| Strategy | Ann. Return | Volatility | Sharpe | Market Beta |
|---|---|---|---|---|
| Long-Only Top 50 | ~30% | ~19% | **1.54** | ~1.0 (market-like) |
| Long-Short | ~15% | ~18% | **0.83** | **0.15** (market-neutral) |
| S&P 500 (benchmark) | ~25% | ~17% | ~1.4 | 1.0 (by definition) |

**Sharpe ratio** = return per unit of risk. Anything above 1.0 is considered good. Above 1.5 is strong.

**The L/S strategy is the cleaner result** — it generates ~+9.5% alpha (return above what the market explains) with almost no market exposure (beta = 0.15). This means it's working because the model actually ranks stocks well, not just because the market went up.

---

## Known Issues / What's Next

### 1. PCA-RP Beta Problem
The PCA Risk Parity weighting produces beta = 2.07 (twice the market exposure). This is a bug in the weight construction — the first PCA component is always the market factor, and back-projecting to asset space amplifies it. **Fix:** replace with iterative equal-risk-contribution (ERC) weighting.

### 2. Validation Period (2022 Bear Market)
Early stopping in `check_overfit.py` stops at just 4 trees because 2022 was a pure bear market — nothing looked good on the validation set. The main pipeline uses a full walk-forward retrain which avoids this. Not a critical issue, but the validation split could be improved.

### 3. No Transaction Costs
All results assume you can trade at month-end close prices with no slippage or commissions. Real-world returns would be lower.

---

## Files & Execution Order

```
1 price parquet.py          → downloads prices, saves data/prices.parquet
1_feature_engineering.py    → builds 26 features, saves data/panel_monthly_enriched.parquet
2_lgbm_backtest.py          → trains model, scores stocks, runs backtest
3_pca_rp_backtest.py        → applies PCA-RP weighting to LGBM scores (experimental)
4_benchmark_spx.py          → compares strategies vs S&P 500
6 Diagonstic test.py        → detailed diagnostics (turnover, rank decay, cov stats)
check_overfit.py            → quick train/valid/test Sharpe check
inspect_data.py             → prints data shapes and date ranges
```

Run scripts 1→2→3→4 in order. Scripts 6, check_overfit, inspect_data can be run any time after step 2.

---

## Shared Config (`config.py`)

All key parameters live in one place:

| Parameter | Value | Meaning |
|---|---|---|
| `START_DATE` | 2010-01-01 | Ignore pre-GFC data |
| `TRAIN_END` | 2020-12-31 | End of training window |
| `VALID_END` | 2022-12-31 | End of validation window |
| `TOP_N` | 50 | Stocks in long-only portfolio |
| `LONG_FRAC` | 0.10 | Top/bottom 10% for L/S |
| `RETRAIN_EVERY` | 12 | Retrain model every 12 months |
