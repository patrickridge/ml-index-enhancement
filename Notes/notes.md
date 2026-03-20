## Aim
Beat the S&P 500 by a small consistent margin (2-4% per year) without taking big risks.

## How 
Hold all 500 S&P 500 stocks but tilt the weights using an ML model — overweight stocks the model thinks will do well, underweight stocks it thinks will underperform. The model is trained on factors (volatility, momentum, PE ratio etc.) that historically predict returns.

This is a real strategy used by institutional quant funds. The key insight is you're not trying to pick stocks dramatically, just nudge weights slightly, which keeps risk low. The ML model (transformers) is better than traditional approaches because it can learn non-linear relationships between factors and returns.

## Where you are now 
Factor testing is done. Waiting for historical fundamental data (PE, PB etc.) to train the model. Once that arrives it's model training → backtest → automate.

"enchanced index" funds 
Basically every large quant asset manager offers some version of this.

Integrating investsoc sector picks:

Simple idea — your ML model generates a score for every stock. The sector analysts pick their favourite stocks. You could:

Boost the ML score of stocks the analysts like — if analyst likes Apple AND the model likes Apple, overweight it even more
Hard constraint — ensure the portfolio always holds at least X% of each sector team's top picks
Overlay approach — run the ML strategy normally, then separately add a small tilt toward analyst picks on top
Option 1 or 3 is cleanest. It means the strategy has two alpha sources — quantitative (ML factors) and fundamental (human analysts) — which is actually how many real hedge funds operate. Called quant-fundamental hybrid.


The honest difference from a real hedge fund product:

Real funds have more factors, more data (alternative data, earnings calls, satellite imagery etc.)
Real funds have stricter risk management (drawdown limits, factor exposure caps)
Real funds execute more efficiently (transaction cost minimisation, smart order routing)

## Rest of project — what's needed:

Get historical data (2010–2025) 
Merge into panel — add 33 fundamental factors to our existing 59
Re-run factor analysis — IC, regime stability, quintile tests on combined set
Train FT-Transformer + Cross-Sectional Transformer — walk-forward validation, IR as metric
Ensemble the two models — blend scores into one stock ranking
Portfolio construction — translate scores into weight tilts around S&P 500 weights
Backtest — measure alpha, tracking error, Information Ratio vs S&P 500
Automate the pipeline — monthly rebalancing loop using platform
Integrate analyst picks (optional) — overlay sector team's picks on top

# original model performance

python "5 light gbm backtest.py"
TRAIN: months=144 | ann= 77.52% | vol= 23.02% | sharpe= 2.55
VALID: months= 36 | ann= 20.53% | vol= 21.17% | sharpe= 0.89
TEST: months= 59 | ann= 12.07% | vol= 16.69% | sharpe= 0.69


Key TEST stats:
Top50 Long-only (LGBM + PCA-RP): ann= 22.52% | vol= 22.36% | sharpe= 1.01 | maxDD=-27.39%
Top10 Long-only (LGBM + PCA-RP): ann= 27.87% | vol= 38.39% | sharpe= 0.73 | maxDD=-46.48%
L/S Top10 - Bottom50 (EqualW): ann= 11.74% | vol= 34.00% | sharpe= 0.35 | maxDD=-50.02%
L/S Top10 - Bottom10 (EqualW): ann=  7.33% | vol= 28.87% | sharpe= 0.25 | maxDD=-47.84%

LGBM
TRAIN: months=180 | ann= 36.83% | vol= 28.17% | sharpe=1.31
VALID: months= 24 | ann= 16.44% | vol= 30.84% | sharpe=0.53
TEST: months= 35 | ann= 33.95% | vol= 21.41% | sharpe=1.59

Update Todos
 Analyse all results and explain clearly to user
 Fixing verbosity in config.py
 Fix diagnostic script: switch to enriched panel
 Create STRATEGY.md: simple explanation of everything

 # extra notes

- could use yfinance daily hourly if don't need much data
- vectorbt for backtest
- hypothesis for testing before
- read papers on similiar projects

 - make plan
 - pull more data
 - factors
 - LLM


 # original project idea
Machine Learning for Financial Forecasting

Build a Neural Network Model to predict asset returns using historical factors (e.g. momentum, value, volatility) and macroeconomic data. Train and test the model, calculate its cost, and compare it to other forecasting models (e.g. Random Forest, Ridge Regression).