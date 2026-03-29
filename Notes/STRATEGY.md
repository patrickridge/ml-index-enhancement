# ML-Driven S&P 500 Index Enhancement — Strategy Overview

## What Are We Trying to Do?

Build a portfolio that tracks the S&P 500 closely but consistently beats it. Every month, we use machine learning to rank all ~500 stocks and tilt portfolio weights slightly toward the stocks expected to outperform. The goal is a high Information Ratio (IR) with low tracking error — generating steady alpha relative to the benchmark rather than making big concentrated bets.

---

## The Data

- **Universe:** ~697 S&P 500 stocks (historical constituents, including delisted/acquired companies to reduce survivorship bias)
- **Time range:** 2010–2025 (post-GFC, avoids structurally different 2008–2009 crisis)
- **Frequency:** Monthly predictions, daily prices for feature construction
- **Target:** Each stock's return over the next calendar month (`fwd_ret_1m`)
- **Price source:** Yahoo Finance (`yfinance`), Tiingo (delisted), Wind platform (manual exports)

---

## The Features (43 factors)

All features start as raw price-derived signals and go through three processing steps:

1. **Regime stability filter** — factor must show consistent sign across ≥ 3 of 4 market regimes (QE bull 2010–2019, COVID recovery 2020–2021, rate hike bear 2022, AI bull 2023+). 59 → 43 factors survive.
2. **Sign normalisation** — contrarian factors (negative IC) are sign-flipped so higher always means better predicted return.
3. **IC optimisation** — a PyTorch gradient descent pass finds the optimal weight for each factor by maximising mean Pearson IC over the training period (2010–2020). Train IC improved +197% vs static IC-decay weights.

| Category | Example Factors | IC Direction |
|----------|----------------|--------------|
| Momentum | `ret_12m`, `ret_18m`, `ret_6m`, `mom_2_12` | Positive |
| Quality/consistency | `ir_3m`, `ir_12m`, `residual_ret_12m` | Positive |
| Volatility | `vol_252d`, `idio_vol_252d`, `vol_126d` | Positive (risk premium) |
| Technical | `nearness_52w_low`, `macd_hist` | Positive (after flip) |
| Mean reversion | `bollinger_pct`, `rsi_14`, `price_to_ma20` | Contrarian (sign-flipped) |

All features are cross-sectionally z-scored within each month — a stock's absolute level doesn't matter, only its rank relative to the full universe that month. This makes signals stable across different market regimes.

---

## Factor Quality Tests

Before any factor is used, it must pass:

**1. IC Analysis (Information Coefficient)**
- IC = Pearson correlation between factor ranking and next-month return, computed each month
- ICIR = mean(IC) / std(IC) — measures consistency. Best factors: `idio_vol_252d` ICIR +0.172, `vol_252d` ICIR +0.155
- Both positive and contrarian factors kept — sign just tells you which direction to trade

**2. Quintile Test**
- Sort all stocks by factor → split into 5 buckets (Q1 bottom 20%, Q5 top 20%)
- Spread = avg return(Q5) − avg return(Q1)
- A clean monotonic staircase (Q1 < Q2 < Q3 < Q4 < Q5) confirms the signal works across the full distribution

**3. Regime Stability**
- Factor sign must be consistent across ≥ 3/4 market regimes (COVID crash excluded as black swan)
- Filters out signals that only work in one market environment

**4. IC Decay**
- Measures how many months ahead each factor predicts
- Volatility factors persist 60+ months (structural tilt). Short-term momentum factors fade within 1–2 months.
- Used to weight factors by signal longevity

---

## Walk-Forward Training (No Look-Ahead Bias)

```
2010 ─────────────────────── 2021 ──── 2023 ─────── 2025
│← Train (~132 months)     →│ Valid  │←  Test      →│
                              │ (24m) │   (35+ months)
```

- **Factor analysis** runs only on training data (2010–2020) — factors are selected before seeing any test period data
- **Model training** uses expanding window — every 12 months in the test period, the model retrains on all available history
- **Test period (2023–2025)** is never seen during training, validation, or factor selection

---

## The Models

### Cross-Sectional Transformer (Primary Model)

Processes all ~500 stocks simultaneously every month. Each stock's 43 factors become a vector; the transformer applies self-attention across the entire cross-section, learning how stocks relate to each other.

```
Month t:  [Stock 1 factors]  →
          [Stock 2 factors]  →  Multi-head Self-Attention  →  Score per stock
          [Stock 3 factors]  →
               ...
```

**Why it outperforms:** It sees every stock relative to every other stock in the same month — directly learning cross-sectional ranking, which is exactly what index enhancement requires. LGBM and the FT-Transformer score stocks independently and miss this relative information.

**Key property:** Regime-stable — IR of 0.926 in risk-off periods and 2.227 in risk-on periods (consistent across both). Full period IR 1.850.

### FT-Transformer (Secondary Model)

Per-stock transformer that treats each of the 43 factors as a "token" and learns feature interactions using self-attention. Captures non-linear relationships between factors (e.g. high momentum + rising information ratio together). Less regime-stable than CS-Transformer.

### LightGBM (Baseline)

Gradient-boosted decision trees — fast, interpretable via feature importance. Used as a baseline. Good ICIR properties but weaker at cross-sectional ranking.

---

## Portfolio Construction

Every month the portfolio is fully rebalanced using the following process:

**Step 1 — Score every stock**
The ML model scores all ~490–500 active S&P 500 stocks. Higher score = predicted to outperform next month.

**Step 2 — Rank and split into three groups**
Stocks are sorted by score and split into:
- **Top 100** — the 100 highest-scoring stocks (predicted best performers)
- **Middle ~300** — held at their normal S&P 500 benchmark weight, no change
- **Bottom 100** — the 100 lowest-scoring stocks (predicted worst performers)

**Step 3 — Apply tilt**
Each stock's portfolio weight is nudged away from its S&P 500 benchmark weight by a small amount α:

```
Top 100    →  weight = S&P500 weight + α   (overweighted vs index)
Middle 300 →  weight = S&P500 weight       (unchanged)
Bottom 100 →  weight = S&P500 weight − α   (underweighted vs index)
```

Weights are clipped at 0 (no shorting) and renormalised to sum to 1.

**Step 4 — Measure performance**
Active return each month = our portfolio return − S&P 500 return.
If we overweighted stocks that went up and underweighted stocks that went down, active return is positive.

**Why top and bottom 100, not all 500?**
Middle-ranked stocks have weak, near-zero signal. Tilting all 500 adds noise. Only the extreme ends of the ranking carry reliable predictive information.

**Why long-only?**
Index enhancement stays close to the benchmark. Shorting increases tracking error and introduces borrow costs. Weights are clipped at 0 so no stock is ever held short.

---

## Factor-Combo Baseline

To confirm the ML models add genuine value, we built a simple linear factor combination using the same 43 optimised IC weights (no ML):

```
score_i = sum(optimised_weight_j × z-scored_factor_j)  for each stock i
```

| Signal | IR |
|--------|----|
| Linear factor combo (no ML) | −0.047 |
| LGBM | 0.377 |
| FT-Transformer | 0.428 |
| **CS-Transformer** | **1.874** |

The linear combination underperforms the index. The transformer goes from −0.047 → 1.874 — it is learning genuine non-linear cross-sectional patterns, not just repackaging the factor exposures.

---

## Results (Out-of-Sample, Jan 2023 – Nov 2025)

### Index Enhancement Performance

| Model | Ann. Alpha | Tracking Error | IR | Hit Rate |
|-------|-----------|---------------|-----|----------|
| **CS-Transformer** | **4.16%** | **2.22%** | **1.874** | **75%** |
| FT-Transformer | 0.03% | 1.87% | 0.015 | 54% |
| LGBM | 0.61% | 1.61% | 0.377 | 67% |
| Factor-Combo (linear baseline) | −0.2% | 3.5% | −0.047 | 50% |

IR > 0.5 = good (top-quartile institutional fund managers typically hit ~0.3–0.5). IR > 1.0 is excellent.

*Both CS-Transformer and FT-Transformer retrained on 240 months (2010–2020).*

### Regime Breakdown (HMM, 2-state: risk-on / risk-off)

| Model | Full IR | Risk-Off IR | Risk-On IR | Regime-Stable? |
|-------|---------|------------|-----------|----------------|
| **CS-Transformer** | **1.850** | 0.926 | **2.227** | **Yes** |
| FT-Transformer | 0.024 | 0.210 | −0.044 | No |
| LGBM | 0.384 | 1.397 | 0.224 | No |

CS-Transformer is the only model that generates consistent alpha regardless of market regime. FT-Transformer IR collapses to near-zero after retraining — confirms CS-T's cross-sectional architecture is the key differentiator.

### RL Agent Results

**Walk-Forward Backtest — 5 Folds, No Data Leakage (5c_walk_forward.py)**

95 out-of-sample months across 2014–2025. Factor weights, state normalisation, and RL policy all retrained from scratch per fold using past data only.

| Metric | RL Agent (SAC) | Fixed α=1% |
|--------|---------------|------------|
| Ann Alpha | **7.19%** | 1.87% |
| Tracking Error | 8.18% | 6.77% |
| **Info Ratio** | **0.879** | 0.276 |
| Hit Rate | 50.5% | 48.4% |
| Max Active DD | **−6.04%** | −10.50% |

RL beats fixed alpha in 4/5 folds. Max drawdown cut nearly in half.

**Algorithm Comparison — SAC vs PPO vs GRPO (5d_algorithm_comparison.py)**

Same 5-fold structure, three different DRL algorithms:

| Algorithm | What It Does | IR |
|-----------|-------------|-----|
| **GRPO** | No critic — samples 4 alphas per state, ranks by reward. DeepSeek-R1 algorithm + KL penalty. | **0.875** |
| PPO | On-policy clipped update with critic baseline | 0.870 |
| SAC | Off-policy replay buffer, twin Q-critics, entropy regularisation | 0.788 |
| Fixed α=1% | No RL | 0.276 |

GRPO's critic-free group ranking is the best fit for small-data regimes (~100 training months). No value function = no estimation error.

**DAPO (5e_dapo_agent.py) — ByteDance 2025**

DAPO is the next evolution of GRPO with three improvements:
1. **Clip-higher**: Asymmetric clipping — positive advantages allow larger policy updates (ε=0.28) vs negative (ε=0.20)
2. **Dynamic sampling**: Hard market states (high reward variance) get up to 8 candidate alphas; easy states get 2
3. **No KL penalty**: Clip-higher replaces the KL stability mechanism

**Layer 2 State Vector (9 features)**

The SAC/GRPO/DAPO agents observe:
- Signal strength & dispersion (quality of the CS-T ranking)
- Benchmark vol, rolling tracking error, recent active return
- Regime (risk-on / risk-off)
- **VIX level, 10Y yield, yield spread (10Y–2Y)** ← macro features added Mar 2026

**Layer 1 — Adaptive Factor Weighting (5a_rl_factor_agent.py)**

| Method | Mean IC | ICIR | Hit Rate |
|--------|---------|------|---------|
| **RL adaptive weights** | **0.0507** | **0.368** | **65.8%** |
| Equal weights | 0.0119 | 0.058 | 57.9% |
| IC-decay weights | 0.0110 | 0.051 | 57.9% |
| IC-optimised weights | 0.0040 | 0.033 | 47.4% |

RL IC is 4× higher than next best. Static optimised weights underperform — overfit to training period signal. The RL agent dynamically upweights factors with recent positive IC and downweights those that have stopped working.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Monthly rebalancing | Signal IC decays over ~6–12 months; daily rebalancing adds cost with no benefit |
| Top/bottom 100 tilt | Middle stocks carry weak signal; focusing on extremes improves IR |
| Cross-sectional z-scoring | Makes features regime-invariant; model sees relative rank not absolute value |
| Survivorship bias fix | 697 tickers (historical constituents) vs 504 (current only) — avoids inflating backtest returns by ~1–2%/year |
| No shorting | Long-only index enhancement; avoids borrow costs and concentration risk |
| 2010 start date | Post-GFC baseline; GFC had structurally different correlations and regime |
| Regime stability filter | Factors that flip sign in different market environments are unreliable signals |

---

## Limitations & Caveats

| Limitation | Detail |
|------------|--------|
| No transaction costs | Results assume zero slippage/commissions. At 10 bps per trade with ~30% monthly turnover, Sharpe reduces ~0.1–0.15 |
| Remaining survivorship bias | 248 historical constituents still missing — blocked on Wind/CRSP data pull |
| No fundamental data | All 43 factors are price-derived. PE, PB, money flow factors not yet incorporated — likely to improve IC significantly |
| Bull market test period | Test period (2023–2025) is almost entirely the AI bull run. Regime stability analysis (HMM) partially mitigates this by testing risk-off sub-periods separately |
| SPX weights approximation | Benchmark weights estimated from market cap / price via yfinance. Top 4 holdings capped at 8% (winsorised). Real SPDR constituent data would improve accuracy |
| 35 test months | IR estimated on ~35 months. Confidence interval is wide — IR could vary ±0.3 with more data |

---

## What's Next

**Immediate (blocked on data):**
1. Collaborator exports remaining 248 tickers + fundamental data from Wind
2. Run `1e_ingest_wind_xlsx.py` → `1f_rebuild_panel.py` → `1g_feature_engineering.py`
3. Re-run `2e_ic_optimise.py` with fundamental factors included
4. Retrain CS-Transformer + FT-Transformer on Kaggle with expanded universe
5. Re-run `4b_index_enhancement.py` + `4c_regime_engine.py`

**Two-layer RL design — objectives are complementary, not conflicting:**

| Layer | File | Objective | What it optimises |
|-------|------|-----------|-------------------|
| **Layer 1** | `5a_rl_factor_agent.py` | Maximise IC / ICIR | *What signal to generate* — which factors to trust each month |
| **Layer 2** | `5b_rl_portfolio_agent.py` | Maximise portfolio Sharpe | *How aggressively to act* on that signal each month |

No conflict: Layer 1 never sees portfolio vol. Layer 2 takes signal quality as given and adjusts tilt to maximise risk-adjusted total return. A stronger signal (high recent IC from L1) should cause L2 to tilt more aggressively, as more alpha per unit of tilt raises Sharpe.

**Layer 2 — Portfolio Tilt (`5b_rl_portfolio_agent.py`):**

- **State (6 features):** signal strength, signal dispersion, benchmark vol, recent active return, regime indicator, rolling tracking error
- **Action:** alpha in [0.002, 0.05] — how aggressively to tilt this month
- **Reward:** annualised active return (`active_ret × 12`). TE constraint is implicit — high alpha × weak signal → negative active return → agent learns to reduce alpha when signal is unreliable.
- **Policy:** MLP only — NO memory, NO recurrence. Each month is fully independent (Markov).
- **Training:** factor-combo scores on full panel 2010–2022 (12 years, ~103 months)
- **Evaluation:** CS-Transformer scores on test period 2023–2025

**Layer 1 — Adaptive Factor Weighting (`5a_rl_factor_agent.py`):**

- **State (133-dim):** rolling 6m IC per factor (43) + rolling 12m IC per factor (43) + IC momentum (43) + macro (4)
- **Action:** 43-dim weight vector — how much to trust each factor this month
- **Reward:** direct monthly IC of the combined weighted factor signal (IC/ICIR maximisation)
- **Policy:** MLP only, no memory, fully Markov
- In momentum regimes: upweights momentum/quality factors. In mean-reversion regimes: shifts to contrarian factors.

---

## Diffusion Model Stress Test (`7_synthetic_regimes.py`)

The test period (2023–2025) contains only ~7 genuine risk-off months — too few for reliable bear-regime IR estimation (confidence interval ≈ ±0.6). A regime-conditional DDPM generates 1,000 synthetic bear market months to stress-test performance.

**Pipeline:**
1. Train DDPM on 312 months of macro features (SPX return, SPX vol, market trend) conditioned on regime label
2. Generate 1,000 synthetic bear feature vectors via DDPM reverse process
3. Translate to active returns via OLS bridge + residual noise (preserves realistic return variance)
4. Compute IR distribution across all 1,000 synthetic scenarios

**Results:**

| Scenario | IR |
|---------|-----|
| CS-T full period (real) | 1.890 |
| CS-T risk-off (real, n=7) | 2.595 |
| Synthetic bear mean | 2.294 |
| Synthetic bear 5th pct | 2.107 |
| Synthetic bear 95th pct | 2.495 |

74.6% of synthetic bear months generate positive active alpha. The CS-T strategy appears robust to stress scenarios — cross-sectional ranking signal survives market crashes (defensive/quality stocks still separate from the rest).

---

## Interactive Dashboard

```bash
streamlit run 6_regime_dashboard.py
```

5-tab Streamlit app: Regime Breakdown, Bootstrap Analysis, Monte Carlo Projection, Statistics table with IR heatmap, 3-D Volatility Surface. Sidebar controls for model/regime selection and stochastic engine settings. Run: `streamlit run 6_regime_dashboard.py` (opens localhost:8501).

---

## Files & Execution Order

```
Step 1 — Data Prep
  python 1a_price_parquet.py          # parse OHLC data → data/prices.parquet
  python 1f_rebuild_panel.py    # build monthly panel from prices
  python 1e_ingest_wind_xlsx.py      # add Wind xlsx exports (run after dropping files)
  python 1g_feature_engineering.py    # compute 43+ factors → panel_monthly_enriched.parquet
  python 1h_orthogonalize.py         # PCA residualization (optional, run after finalising features)

Step 2 — Factor Analysis
  python 2a_factor_analysis.py        # IC, quintile, regime stability → factor_selected.csv
  python 2b_ic_decay_all.py          # IC decay grid (all 43 factors)
  python 2d_factor_weights.py        # IC-decay based weights → factor_selected.csv
  python 2e_ic_optimise.py           # gradient-optimised weights → factor_selected_optimised.csv
  python 4a_factor_combo_baseline.py # linear combo baseline (no ML)

Step 3 — Models (GPU recommended)
  python 3a_ft_transformer.py           # FT-Transformer → scores_transformer.parquet
  [Kaggle] 3d_cs_transformer_kaggle.py # CS-Transformer → scores_cs_transformer.parquet

Step 4 — Evaluation
  python 4b_index_enhancement.py      # IE portfolio → bt_ie_*.csv
  python 4c_regime_engine.py          # regime breakdown (rule-based)
  python 4c_regime_engine.py --hmm    # HMM 2-state regime breakdown
  python 4d_benchmark_spx.py          # full comparison table

Step 5 — Reinforcement Learning
  python 5b_rl_portfolio_agent.py     # SAC agent — adaptive alpha (requires torch)
```
