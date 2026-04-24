# RL Portfolio Agent - Full Technical Report

Complete explanation of every variable, design choice, and algorithm used in the
reinforcement learning layer (`5b`, `5c`, `5d`, `5e`).

---

## 1. What Problem Are We Solving?

Every month we have a ranked list of ~500 S&P 500 stocks from our ML model (CS-Transformer
or LightGBM). We want to build a portfolio that beats the S&P 500 index.

The ML model answers **"which stocks to favour"** - but not **"how aggressively to tilt"**.
Tilting too hard increases tracking error and drawdown. Tilting too softly wastes the signal.

The RL agent answers the second question: given the current market environment, what is the
optimal tilt size α this month?

---

## 2. Portfolio Construction

### The Tilt Mechanism

Each stock `i` has an S&P 500 benchmark weight `w_SPX_i` (proportional to market cap).
We adjust these weights by ±α based on the ML model's ranking:

```
Top 100 stocks:    portfolio_weight_i = w_SPX_i + α   (overweight)
Middle ~300:       portfolio_weight_i = w_SPX_i        (unchanged)
Bottom 100 stocks: portfolio_weight_i = w_SPX_i − α   (underweight)
```

Weights are clipped at 0 (no shorting) and renormalised to sum to 1.

**α (alpha)** is the single number the RL agent controls each month.
- Range: **[0.002, 0.05]** - 0.2% to 5% tilt per stock
- Low α → stay close to index, low tracking error, low upside
- High α → bigger active bets, higher upside but also higher risk

**Active return** = our portfolio return − S&P 500 return.
The agent's goal is to maximise risk-adjusted active return (Information Ratio).

---

## 3. The State Vector - What the Agent "Sees"

Each month the agent observes a 6-dimensional state vector before deciding α.
All features are z-scored using training data statistics (mean subtracted, divided by std),
clipped to [−5, +5] to remove outliers.

| # | Feature | Raw computation | What it tells the agent |
|---|---------|----------------|------------------------|
| 0 | `signal_strength` | Mean of `|z-scored ML scores|` across all stocks | How strong/clear the model's ranking is this month. High = confident ranking, tilting is worth it. Low = all stocks look similar, tilting is risky. |
| 1 | `signal_dispersion` | Std dev of z-scored ML scores | How spread out the scores are. High dispersion = big gap between top and bottom stocks, strong long-short opportunity. |
| 2 | `bench_vol` | Rolling 3-month std dev of S&P 500 monthly returns × √12 (annualised) | Market volatility regime. High = volatile market, tilting tends to generate unpredictable outcomes. |
| 3 | `recent_active_ret` | 3-month rolling mean of active returns using fixed α=1% | Has the factor signal been working recently? Positive = strategy has momentum, keep tilting. Negative = signal has been failing, reduce tilt. |
| 4 | `regime` | 1 if `bench_vol > rolling median bench_vol`, else 0 | Binary risk-on/risk-off flag. **Not z-scored** - stays as 0.0 or 1.0. Used by DAPOSwitch to detect regime transitions. |
| 5 | `rolling_te` | Rolling 3-month std dev of active returns × √12 (annualised) | Recent tracking error. High TE = strategy has been making large active bets (for better or worse). |

**Why z-score?** Raw values are not comparable across time. Benchmark vol of 15% in 2015 means
something different to 15% in 2020 (COVID). Z-scoring makes the same number mean the same
thing regardless of when you're looking at it. Exception: `regime` is a binary flag - z-scoring
would destroy its meaning.

---

## 4. The Action

**Output:** a single scalar α ∈ [0.002, 0.05]

The neural network outputs a value in (−∞, +∞). We squash it into [α_min, α_max] using tanh:

```
raw_output ~ Normal(μ(state), σ(state))   ← stochastic during training
y = tanh(raw_output)                        ← squash to (−1, +1)
α = α_min + (y + 1) / 2 × (α_max − α_min) ← scale to [0.002, 0.05]
```

During evaluation (live deployment), we use the deterministic mean: `y = tanh(μ(state))`.

---

## 5. The Reward

```python
reward = active_ret × 12
```

`active_ret` is the monthly active return (portfolio return minus benchmark return) from
simulating the chosen α for that month. Multiplying by 12 annualises it - the agent thinks
in terms of annual active return.

**Why not penalise tracking error directly?** The KL penalty (in GRPO) and the clip
mechanism (in DAPO) act as implicit regularisers on how much the policy can change per
update. Tracking error is captured indirectly - if the agent tilts aggressively and
the signal fails, it gets a negative reward, and it learns to reduce α.

---

## 6. The Neural Network (Policy)

All three algorithms (GRPO, DAPO, DAPOSwitch) use the same neural network architecture:

```
GaussianActor:
  Input:  state vector (6 numbers)
  → Linear(6, 64) → ReLU
  → Linear(64, 64) → ReLU
  → Linear(64, 1)  ← mean head (μ)
     and separately:
  → Linear(64, 1)  ← log-std head (log σ), clamped to [−10, +2]
  Output: Normal distribution parameterised by (μ, σ) over action space
```

64 hidden units per layer. Small by ML standards - appropriate because we only have
~50–100 training months per fold, so a larger network would overfit badly.

---

## 7. Walk-Forward Structure - No Data Leakage

5 folds, each using an expanding training window:

```
Fold 1: Train 2010–2013 → Test 2014–2015
Fold 2: Train 2010–2015 → Test 2016–2017
Fold 3: Train 2010–2017 → Test 2018–2019
Fold 4: Train 2010–2019 → Test 2020–2021
Fold 5: Train 2010–2021 → Test 2022–2025
```

For each fold:
1. Factor IC weights are computed from training data only
2. Episode state features (z-score normalisation params) are computed from training data only
3. The RL agent is trained from scratch on training episodes
4. The agent is evaluated on test episodes - never seen during training

All 95 test months concatenated give the out-of-sample IR.

---

## 8. Episode Building (`build_episodes`)

Converts the raw panel data into a sequence of `(state, simulation_function)` pairs.

**For each month `t` in the dataset:**
1. Z-score the ML model scores across all stocks in that month
2. Compute `signal_strength` = mean(|z-scores|), `signal_dispersion` = std(z-scores)
3. Run a reference simulation with α=0.01 to get `bench_ret` and `active_ret_ref`
4. Compute rolling features (`bench_vol`, `rolling_te`, `recent_active_ret`) using past months
5. Determine `regime` (1 if bench_vol > rolling median, 0 otherwise)
6. Z-score all features except `regime` using training statistics
7. Store a reference to the raw scores + benchmark weights for the simulation function

The "simulation function" is not called until the agent chooses its α - this is what makes
it reinforcement learning: the agent's action determines the outcome it observes.

---

## 9. GRPO - Group Relative Policy Optimisation

**Origin:** DeepSeek-R1 (2025). Designed for LLMs, adapted here for portfolio tilting.

**Key idea:** Instead of estimating a "value function" (how good is this state in general),
sample multiple actions from the same state, rank them by reward, and use the relative
ranking as the advantage signal.

### Training loop (per epoch, per month):

```
1. Sample G=4 candidate alphas: α₁, α₂, α₃, α₄  ← stochastic draws from policy
2. Simulate each: r₁, r₂, r₃, r₄  ← run portfolio with each α
3. Compute advantages (group-relative):
      Aᵢ = (rᵢ − mean(r)) / std(r)
   → positive = above-average alpha, push policy toward it
   → negative = below-average alpha, push policy away
4. Policy gradient loss:
      L_PG = −mean(log_prob(αᵢ) × Aᵢ)
   This is REINFORCE: increase probability of actions that beat the group mean.
5. KL penalty:
      KL = KL(current_policy || reference_policy)
   The reference policy is a frozen copy from the start of training.
   KL term prevents the policy from drifting too far too fast.
6. Total loss = L_PG + β_KL × KL   (β_KL = 0.01)
```

**Why no value function?** Estimating V(s) requires many samples per state. With only
~50–100 training months (states), a learned value function would overfit - the noise in
V̂(s) would be larger than the signal. Group relative advantage bypasses this entirely.

---

## 10. DAPO - Dynamic Sampling Policy Optimisation

**Origin:** ByteDance/Seed (2025). Three improvements over GRPO.

### Improvement 1: Clip-Higher (asymmetric clipping)

Standard PPO (and GRPO's implicit update) clips how much the policy can change per step.
DAPO uses different clip bounds depending on whether the advantage is positive or negative:

```
For positive advantage (good action):   clip ratio to [1 − ε_low, 1 + ε_high]  = [0.80, 1.28]
For negative advantage (bad action):    clip ratio to [1 − ε_low, 1 + ε_low]   = [0.80, 1.20]
```

Where `ratio = π_new(α) / π_old(α)` - how much the probability of taking action α
changed after the update.

**Effect:** The policy is allowed to move more aggressively toward rewarding actions
(ratio up to 1.28) but uses standard conservatism for avoiding bad actions (ratio capped
at 1.20). Asymmetric because upside exploration matters more than downside avoidance in
a world with clip protection.

### Improvement 2: Dynamic G (adaptive candidate count)

```
Start with G_INIT=4 candidates → compute variance of rewards
If var(rewards) > 0.05 (hard/uncertain state):
    sample 4 more candidates (total G_MAX=8)
Else:
    keep G=4 (easy/certain state)
```

Hard states = months where outcomes are volatile (e.g. rate hike month - some alphas
work well, others blow up). More candidates → better estimate of the advantage distribution.
Easy states = outcomes are predictable regardless of α choice, extra samples add noise.

### Improvement 3: No KL Penalty

DAPO removes the KL term entirely. The clip mechanism provides sufficient stability.
Removing KL allows the policy to drift further from its initialisation, which can be
helpful when the optimal policy is far from the initial random policy.

---

## 11. DAPOSwitch - Regime-Aware Algorithm Selection

**New (29 Mar 2026).** Combines DAPO and GRPO depending on market regime stability.

### Regime detection:

```python
regime_t = 1  if bench_vol_t > rolling_median(bench_vol)  else 0
is_transition = (regime_t != regime_{t-1})
```

A month is a "transition" if the binary regime flag changed from the previous month.
Roughly 10–15% of months are transitions in typical market data.

### Update routing:

```
Stable month (same regime as last month):
    → DAPO update: clip-higher, no KL, G=4–8
    Rationale: regime is established, signal is predictable, explore more freely.

Transition month (regime just changed):
    → GRPO update: REINFORCE + KL, G=4
    Rationale: market environment just shifted, old alpha intuitions may be wrong.
               KL anchors the policy to its last-known-good behaviour until
               the agent observes enough new data to update confidently.
```

---

## 12. Results Summary (Walk-Forward, 95 OOS months, 2014–2025)

| Algorithm | Avg IR | Fold IRs | Beats Fixed |
|-----------|--------|----------|-------------|
| **GRPO** | **0.629** | −1.168, 1.961, −0.173, 1.132, 1.394 | 4/5 |
| DAPOSwitch | 0.566 | −1.191, 1.954, −0.176, 1.114, 1.128 | 4/5 |
| DAPO | 0.525 | −1.222, 1.963, −0.238, 1.156, 0.965 | 4/5 |
| Fixed α=1% | 0.235 | −1.659, 1.778, −0.073, 0.751, 0.380 | - |

All three RL algorithms beat fixed alpha by 2× or more in average IR.

**Bad folds (2014–15 and 2018–19):** All algorithms produce negative IR. These periods
correspond to choppy, directionless markets where the factor signal had low IC - no RL
algorithm can generate alpha from a weak signal. The RL advantage is that it sometimes
reduces α during these periods (lower max drawdown), but the signal itself is the
binding constraint.

---

## 13. Key Design Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `ALPHA_MIN` | 0.002 | Minimum tilt per stock (0.2%) |
| `ALPHA_MAX` | 0.050 | Maximum tilt per stock (5.0%) |
| `FIXED_ALPHA` | 0.010 | Baseline fixed tilt for comparison |
| `HIDDEN_DIM` | 64 | MLP hidden layer width |
| `GRPO_G` | 4 | Candidate alphas per state (GRPO) |
| `GRPO_KL_BETA` | 0.01 | KL penalty weight |
| `GRPO_EPOCHS` | 400 | Training epochs per fold |
| `DAPO_EPS_LOW` | 0.20 | Clip lower bound (both directions) |
| `DAPO_EPS_HIGH` | 0.28 | Clip upper bound (positive advantage only) |
| `DAPO_G_INIT` | 4 | Initial candidates before variance check |
| `DAPO_G_MAX` | 8 | Max candidates for hard states |
| `DAPO_VAR_THRESH` | 0.05 | Reward variance threshold to trigger G extension |

---

## 14. What Each File Does

| File | Role |
|------|------|
| `5a_rl_factor_agent.py` | Layer 1: RL agent that adapts *which factors to trust* each month (maximises IC/ICIR) |
| `5b_rl_portfolio_agent.py` | Layer 2: SAC agent for adaptive alpha (original implementation) |
| `5c_walk_forward.py` | Walk-forward harness for SAC - establishes 5-fold evaluation structure |
| `5d_algorithm_comparison.py` | SAC vs PPO vs GRPO comparison |
| `5e_dapo_agent.py` | DAPO vs GRPO vs DAPOSwitch comparison (current main RL file) |

---

## 15. Open Questions / Research Directions

1. **Why does GRPO beat DAPO with fair G?**
   The KL penalty may actually be *helping* in our small-data regime (~50 training months)
   by preventing the policy from overfitting to any single fold's training distribution.
   DAPO without KL has more freedom to drift, which helps in LLMs (millions of training
   steps) but may hurt here.

2. **DAPOSwitch improvement direction:**
   Currently uses a simple vol-threshold regime. Replacing with the HMM regime labels
   from `4c_regime_engine.py` would give cleaner transition signals. Alternatively,
   a learned regime detector (separate small network trained on macro features) could
   identify transitions earlier.

3. **Reward function:**
   Current reward = `active_ret × 12`. A Sharpe-shaped reward
   (`active_ret / rolling_te`) would explicitly incentivise risk-adjusted alpha rather
   than raw alpha. Worth testing - may reduce tracking error in stress folds.

4. **Connecting Layer 1 and Layer 2:**
   The Layer 1 agent's recent IC (how well the factor signal is working) should feed
   directly into Layer 2's state vector. A high recent IC from Layer 1 should cause
   Layer 2 to tilt more aggressively (high-quality signal → exploit it). Currently
   they are trained and evaluated independently.
