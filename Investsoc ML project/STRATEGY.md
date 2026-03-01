# ML Stock Selection Strategy — Methodology & Results

## What Are We Trying to Do?

Every month, pick the stocks most likely to go up next month. We use machine learning to rank S&P 500 stocks and build a portfolio from the top-ranked ones.

---

## The Data

- **Universe:** ~500 S&P 500 stocks
- **Time range:** 2010 – 2025 (post-GFC, cleaner regime)
- **Frequency:** Monthly predictions, daily prices for feature construction
- **Target:** Each stock's return over the next calendar month (`fwd_ret_1m`)
- **Source:** Yahoo Finance via `yfinance` (adjusted close prices)

---

## The Features (26 total)

These are signals computed for every stock each month. All features are **cross-sectionally ranked within each month** and scaled to [−0.5, +0.5] — this means a value of +0.5 = top of the universe, −0.5 = bottom. Ranking removes the absolute level of each feature, making signals comparable across different market regimes.

| Category | Feature | Description |
|---|---|---|
| **Momentum** | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` | Past returns over 1, 3, 6, 12 months |
| **Reversal** | `rev_1m`, `rev_signal` | 1-month return (tends to mean-revert short-term) |
| **Momentum extensions** | `mom_2_12`, `mom_accel` | Skip-1-month momentum; recent vs older momentum |
| **Volatility** | `vol_60d`, `vol_252d` | Realised volatility over 60 and 252 days |
| **Volatility ratios** | `vol_ratio_sm`, `vol_ratio_ml` | Short/medium and medium/long vol ratios |
| **Trend / MA** | `price_to_ma20`, `price_to_ma60`, `price_to_ma200` | Price relative to 20, 60, 200-day moving averages |
| **MA crossovers** | `ma_20_60_cross`, `ma_60_200_cross` | Golden/death cross signals |
| **52-week high** | `nearness_52w` | How close the stock is to its annual high (breakout signal) |
| **Liquidity** | `amihud_21d` | Amihud (2002) illiquidity ratio — higher = harder to trade |
| **Tail risk** | `skew_60d`, `maxdd_60d`, `up_down_vol` | Return skewness, max drawdown, upside/downside vol ratio |
| **Quality** | `ir_12m`, `ir_3m` | Information ratio — consistency of risk-adjusted returns |

No accounting data, no fundamentals — purely price-based signals.

---

## The Models

### Model 1: LightGBM (Gradient-Boosted Trees)

LightGBM is a gradient-boosted decision tree model. Think of it as a very smart ranking engine:

1. It looks at all 26 features for every stock every month
2. It learns which combinations of features predicted high returns in the past
3. Each month it outputs a **score** for every stock — higher score = more likely to outperform

Key properties: handles non-linearity, feature interactions, and is fast to train. Interpretable via feature importance (which features it uses most).

### Model 2: FT-Transformer (Feature Tokenizer + Transformer)

The same transformer architecture that powers GPT and BERT — applied to the 26 financial features instead of words in a sentence.

**How it works:**
1. Each of the 26 features is embedded into a 64-dimensional vector via its own learned linear projection (Feature Tokenizer)
2. A learnable [CLS] token is prepended — like BERT's classification token
3. Multi-head self-attention (4 heads, 3 layers) learns which **combinations** of features matter: e.g. "high momentum + rising information ratio together predicts outperformance more than either alone"
4. The CLS token output is passed through a linear head to produce a score

```
Input: [vol_60d=0.2, ret_6m=0.3, ir_12m=0.4, ...]   (26 scalar features)
  ↓  Feature Tokenizer: each scalar → 64-dim embedding
  ↓  Prepend [CLS] token
  ↓  Multi-head self-attention (learns feature × feature interactions)
  ↓  Linear head on CLS output
Output: score (rank ≈ next-month return)
```

Both models use the same training framework, features, and backtesting code — results are directly comparable.

---

## Walk-Forward Training (No Look-Ahead)

We never let the model see future data:

```
2010 ──────────────────────── 2021 ──── 2023 ──── 2025
│← Training (~132 months)  →│ Valid  │←  Test  →│
                              │ (24m) │
```

- **Train (2010–2020):** Model learns from historical data
- **Valid (2021–2022):** Early stopping — we stop training when improvement stalls on unseen data
- **Test (2023–2025):** Never seen during training — real out-of-sample results

Every 12 months in the test period, we retrain with more data (**expanding window**), simulating a live deployment where you periodically refresh the model.

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
- Beta ≈ 0.15–0.29 (almost no market exposure)

### Strategy 3: PCA Risk Parity (experimental ⚠)
- Same top-50 selection as Strategy 1, but weights from PCA-based risk decomposition
- Currently has a beta = 2.07 bug — treat as experimental, do not use for live trading
- See Known Issues below

---

## Results (Out-of-Sample, 2023–2025)

| Strategy | Sharpe | Ann. Return | Volatility | Beta | Alpha | Max DD |
|---|---|---|---|---|---|---|
| LGBM Long-Only (Top 50) | **1.54** | 33.7% | 18.9% | 0.26 | +30.3% | -17.8% |
| LGBM Long-Short | 0.83 | 11.7% | 11.3% | 0.15 | +9.5% | -8.7% |
| Transformer Long-Only | 1.32 | 32.3% | 20.4% | 0.22 | +30.6% | -20.2% |
| Transformer Long-Short | 0.97 | 20.7% | 17.6% | 0.29 | +16.9% | -18.2% |
| S&P 500 (benchmark) | 1.58 | 18.0% | 16.8% | 1.00 | — | — |

**Sharpe ratio** = return per unit of risk. Anything above 1.0 is considered good. Above 1.5 is strong.

**Alpha** = annualised excess return above what market exposure (beta) explains. The L/S strategies generate alpha of +9.5% and +16.9% with very low market exposure — meaning the model is genuinely ranking stocks well, not just riding the bull market.

**Both models independently find similar alpha** (LGBM +30.3% vs Transformer +30.6% for Long-Only). This is a strong signal: when two fundamentally different model families (tree-based vs attention-based) agree on which stocks to buy, it confirms the underlying features have real predictive power.

**The L/S strategy is the cleaner result** for demonstrating model skill — most of the return is market-independent alpha, not market beta.

---

## Assumptions & Limitations

| Assumption | Detail |
|---|---|
| **No transaction costs** | All results assume month-end close prices with zero slippage or commissions. The transaction cost sensitivity table in `report.ipynb` shows real-world impact (~0.05–0.15 Sharpe reduction at 5–10 bps per trade). |
| **Survivorship bias** | Universe is a fixed historical S&P 500 list. Stocks that were removed from the index before our data starts (e.g. companies that went bankrupt pre-2010) are not included, introducing mild upward bias. Stocks removed during the sample period are included up to their removal date. |
| **No look-ahead bias** | Enforced by walk-forward design: test months (2023–2025) are never seen during training or validation. Each feature uses only data available at the time of prediction. The forward return (`fwd_ret_1m`) is the label — it is never a feature. |
| **Equal weighting** | Long-only portfolio uses 2% per stock (50 stocks). Equal-weighting avoids portfolio optimisation over-fitting; any optimised weights would require a further layer of cross-validation to avoid data snooping. |
| **Monthly rebalancing** | Assumes full portfolio rebalance at month-end close. Intra-month price moves are ignored. |
| **2010 start date** | Excludes 2008–2009 GFC regime. The GFC had structurally different correlations and liquidity dynamics compared to post-crisis markets. Starting in 2010 gives a cleaner, more stationary training distribution. |
| **Cross-sectional rank normalisation** | All 26 features are ranked within each month and scaled to [−0.5, +0.5]. This removes the absolute level of each feature (e.g. whether 6-month momentum is 5% or 50% in absolute terms) and makes signals stable across different market conditions. A stock ranked in the top decile on momentum always maps to the same input value regardless of what year it is. |
| **No short-selling constraints** | Long-Short results assume you can freely short any stock in the bottom decile. In practice, borrow costs and availability vary. |

---

## Known Issues / What's Next

### 1. PCA-RP Beta Problem
The PCA Risk Parity weighting produces beta = 2.07 (twice the market exposure). This is a bug in the weight construction — the first PCA component always captures the market factor, and back-projecting to asset space amplifies it. **Fix:** replace with iterative equal-risk-contribution (ERC) weighting.

### 2. Validation Period (2022 Bear Market)
Early stopping in `check_overfit.py` stops at just 4 trees because 2022 was a pure bear market — nothing looked good on the validation set. The main pipeline uses a full walk-forward retrain which avoids this. Not a critical issue, but the validation split could be improved (e.g. using a rolling validation window rather than a fixed one).

### 3. No Transaction Costs
All results assume zero slippage and zero commissions. Monthly turnover for the LGBM Long-Only strategy is approximately 30–40% per month (stocks entering and leaving the top 50). At 10 bps per trade, this reduces the Sharpe by approximately 0.1–0.15.

### 4. No Fundamental Data
All 26 features are price-derived. Adding fundamental signals (P/E ratio, earnings growth, quality metrics) could improve predictive power, particularly for the long-short strategy.

---

## Files & Execution Order

```
"1 price parquet.py"          → downloads prices,  saves data/prices.parquet
1_feature_engineering.py      → builds 26 features, saves data/panel_monthly_enriched.parquet
2_lgbm_backtest.py            → trains LightGBM,   saves data/scores_lgbm.parquet + bt_lgbm*.csv
2b_nn_backtest.py             → trains FT-Transformer, saves data/scores_transformer.parquet + bt_transformer*.csv
3_pca_rp_backtest.py          → PCA-RP weights (experimental)
4_benchmark_spx.py            → compares all strategies vs S&P 500, saves data/benchmark_comparison.csv
5_pitch_validator.py TICKER   → score + signal breakdown for any stock ticker
"6 Diagonstic test.py"        → turnover, rank decay, covariance diagnostics
check_overfit.py              → quick train/valid/test Sharpe check
inspect_data.py               → prints data shapes and date ranges
report.ipynb                  → full results report with charts (run after step 4)
```

Run scripts 1 → 2 → 2b → 3 → 4 in order. The others can be run any time after step 2.

---

## Shared Config (`config.py`)

All key parameters live in one place:

| Parameter | Value | Meaning |
|---|---|---|
| `START_DATE` | 2010-01-01 | Ignore pre-GFC data |
| `TRAIN_END` | 2020-12-31 | End of training window |
| `VALID_END` | 2022-12-31 | End of validation window |
| `TOP_N` | 50 | Stocks in long-only portfolio |
| `LONG_FRAC` | 0.10 | Top/bottom 10% for Long-Short |
| `RETRAIN_EVERY` | 12 | Retrain model every 12 months in test period |
| `LGBM_PARAMS` | see config.py | LightGBM hyperparameters (n_estimators, depth, etc.) |
| `TRANSFORMER_PARAMS` | see config.py | FT-Transformer hyperparameters (d_model=64, n_heads=4, n_layers=3) |
