## Notes from original model
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

### 5. Regime Stability Filter Uses Full Panel (Minor Look-Ahead)
In `2a_factor_analysis.py`, the regime stability check (which decides whether a factor survives into the final 43) is run on the full panel including the test period (2023–2025 AI bull regime). The IC calculation, quintile tests, and all model training are strictly train-only (≤ 2020), but the regime filter uses data it should not technically see.

**Impact:** Factors that happen to have consistent IC in the 2023–2025 test period are slightly more likely to pass the stability filter. However, a factor must pass in ≥ 3 of 4 regimes — it cannot pass on AI bull data alone. The practical effect on results is small.

**Why it exists:** The AI bull regime (2023–present) is one of four named regimes. Without using 2023+ data there would only be 3 regimes to test against, making the filter less meaningful.

**Fix:** Use only regimes that fall entirely within the training period (pre-2021) for factor selection. Add a separate post-hoc check on test-period regime stability as a diagnostic only.

### 6. 247 Missing Historical Tickers
247 S&P 500 historical constituents could not be fetched from yfinance or Tiingo (mostly delisted/acquired companies without a clean exchange suffix). These are excluded from the universe, introducing mild survivorship bias. Wind platform does not appear to have these tickers either. Full resolution would require CRSP or Compustat access.
