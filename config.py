"""
config.py - shared constants for the pipeline.

Date splits, portfolio sizing, model hyperparameters. Imported by the 2*, 3*,
4*, and 5* scripts. Kaggle scripts (3b, 3d, 3e_hp_sweep_kaggle) inline a copy
of the subset they need so they run without this module.
"""
import os
from pathlib import Path

# Defaults to ./data. Override to run from a git worktree or a Kaggle input
# mount without copying the panel, which is 100MB and not worth duplicating.
DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))

# Date splits
# With 20 years of data (~2004-2024):
#   Option A - keep GFC cut (cleaner regime, ~168 months from 2010):
#     START_DATE = "2010-01-01", TRAIN_END = "2020-12-31", VALID_END = "2022-12-31"
#   Option B - use all 20 years (~240 months, includes pre-GFC):
#     START_DATE = "2004-01-01", TRAIN_END = "2018-12-31", VALID_END = "2021-12-31"
#
# Current: Option A (recommended - avoids GFC regime in training)
START_DATE    = "2010-01-01"   # cut pre-GFC data
TRAIN_END     = "2022-12-31"   # ~156 months training (includes 2021-2022 vol regime)
VALID_END     = "2024-06-30"   # ~18 months validation (Jan 2023–Jun 2024); test = Jul 2024+ (~12 months)

# Universe hygiene
#
# prices.parquet carries tickers that left the index years ago (CPWR delisted
# 2014, FNMA/FMCC in OTC conservatorship since 2008, and so on). 1c assigns
# them a market-cap weight anyway, because it multiplies whatever yfinance
# reports as current shares outstanding by the historical close. The weights
# come out at ~1e-9, effectively zero, so the benchmark ignores them - but a
# score-driven tilt still opens real positions in them, and their price series
# are stale stubs that produce absurd returns (CPWR printed +1500% in a single
# month of 2025). That combination books pure fictional alpha.
#
# The floor below is calibrated on the OOS panel: 5e-6 removes every known
# zombie while the largest surviving monthly return falls to +99% (AppLovin,
# Oct 2024), which is real. The return gate is a second line of defence
# against data errors in names that clear the weight floor.
MIN_SPX_WEIGHT       = float(os.environ.get("ML_MIN_WEIGHT", 5e-6))
MAX_ABS_MONTHLY_RET  = 1.0     # |ret| > 100% in one month = treat as bad data


def drop_dead_features(df, feat_cols, label="", verbose=True):
    """
    Drop feature columns that carry no cross-sectional information.

    Two kinds qualify, and both are undefined rather than weak: a column with
    one value across the whole panel, and a column whose value is identical for
    every stock within each month. Neither can rank anything, so a Spearman IC
    against them is undefined and 2i drops them before testing anyway. Feeding
    them to a model is not harmful but it is not free either, and quoting them
    in a feature count overstates what the model actually had.

    21 fundamental, short-interest and 13F columns are currently in the first
    category because those fetches never landed usable data. The five
    sentiment and calendar columns are in the second: real month-level series
    sitting in the stock feature set, where they cannot do anything.

    Detected at runtime rather than hardcoded, so a column comes back on its
    own the day its data arrives.

    Returns the surviving column list.
    """
    live, flat, monthly = [], [], []

    for c in feat_cols:
        if df[c].nunique(dropna=True) <= 1:
            flat.append(c)
        elif "date" in df.columns and df.groupby("date")[c].nunique().max() <= 1:
            monthly.append(c)
        else:
            live.append(c)

    if verbose and (flat or monthly):
        tag = f" ({label})" if label else ""
        print(f"  dead features{tag}: {len(feat_cols)} -> {len(live)} "
              f"[-{len(flat)} empty, -{len(monthly)} not cross-sectional]")
        if flat:
            print(f"    empty: {', '.join(flat[:6])}"
                  f"{' ...' if len(flat) > 6 else ''}")
        if monthly:
            print(f"    month-level: {', '.join(monthly)}")

    return live


def clean_universe(df, weight_col="spx_weight", ret_col="fwd_ret_1m",
                   label="", verbose=True):
    """
    Drop stub tickers and impossible returns before any portfolio maths.

    Returns the filtered frame. Prints what it removed when verbose, because a
    silent universe filter is its own kind of bug.
    """
    n0 = len(df)
    out = df[df[weight_col] >= MIN_SPX_WEIGHT]
    n_weight = n0 - len(out)

    if ret_col in out.columns:
        bad = out[ret_col].abs() > MAX_ABS_MONTHLY_RET
        n_ret = int(bad.sum())
        if verbose and n_ret:
            worst = out.loc[bad].nlargest(min(3, n_ret), ret_col)
            for _, r in worst.iterrows():
                print(f"    [data] dropped {r.get('ticker', '?')} "
                      f"{r.get('date', '')} ret={r[ret_col]:+.1%}")
        out = out[~bad]
    else:
        n_ret = 0

    if verbose:
        tag = f" ({label})" if label else ""
        print(f"  universe filter{tag}: {n0:,} -> {len(out):,} rows "
              f"[-{n_weight:,} sub-threshold weight, -{n_ret:,} bad return]")
    return out


# Portfolio construction
TOP_N         = 100            # overweight top N stocks vs benchmark
BOTTOM_N      = 100            # underweight bottom N stocks vs benchmark
LONG_FRAC     = 0.20           # top/bottom fraction for long-short

# Walk-forward retraining
RETRAIN_EVERY = 12             # retrain every N months (expanding window)

# PCA Risk Parity
VOL_LOOKBACK  = 252            # trading days for covariance estimation
MIN_OBS       = 150            # minimum daily observations per ticker
N_PCA         = 10             # PCA components to keep
RIDGE         = 1e-3           # ridge regularization on covariance

# FT-Transformer (3a_ft_transformer.py)
# Architecture: each of the N input features is embedded into d_model-dim vectors,
# then processed by multi-head self-attention to learn feature interactions.
# CPU training time: ~45-60 min (3 folds × ~15-20 min each). GPU: ~15-25 min total.
TRANSFORMER_PARAMS = dict(
    d_model=64,         # embedding dimension per feature
    n_heads=4,          # attention heads (must divide d_model)
    n_layers=3,         # transformer encoder layers
    dropout=0.1,        # regularisation
    lr=1e-3,            # Adam learning rate
    weight_decay=1e-4,  # L2 regularisation
    epochs=80,          # max training epochs (was 150 - converges well before 80)
    patience=10,        # early stopping patience (was 20 - halves worst-case time)
    batch_size=512,     # mini-batch size
)

# LightGBM
LGBM_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=5,
    min_child_samples=60,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,          # suppress -inf/-nan split-gain warnings
)

# 100+ Factor Pipeline additions
# Feature variants
USE_ORTHOGONALIZED_FEATURES = False   # True → load panel_monthly_orthogonalized.parquet
USE_CANDIDATE_FEATURES      = True    # True → include 69 factor mining candidates in panel
PCA_VARIANCE_THRESHOLD      = 0.80    # variance explained threshold for 1i_orthogonalize.py

# RMT covariance denoising (fixes beta=2.07 in archive/3_pca_rp_backtest.py)
USE_RMT_COV_DENOISING = True

# Macro factor columns - NOT cross-sectionally ranked (same value for all stocks per month)
MACRO_COLS = [
    "vix_level", "vix_change_21d", "yield_10y", "yield_spread_10y2y",
    "yield_change_21d", "dollar_index", "credit_proxy_change",
    "market_trend_spx", "market_vol_regime",
    # SPX-level stats - same value for all stocks per month, must be time-series z-scored
    # (cross-sectional ranking makes these constants → useless as CS factors)
    "spx_ret_1m", "spx_ret_3m", "spx_ret_6m", "spx_ret_12m", "spx_vol_63d",
    # Prediction market / forward-looking macro (Cat 21)
    "fed_hike_prob", "fed_cut_prob", "recession_prob",
    "policy_uncertainty", "vix_term_structure", "pred_market_sentiment",
]

# Cross-sectional transformer (3c_cs_transformer.py)
TRANSFORMER_CS_PARAMS = dict(
    d_model=128, n_heads_s1=4, n_layers_s1=2,
    n_heads_s2=4, n_layers_s2=2,
    dropout=0.1,
    lr=1e-4,               # lowered from 5e-4 - bigger model / feature set
    weight_decay=1e-4,
    epochs=150, patience=20, max_stocks=520,
    # Feature tokenizer penalties - scaled down 100× so they don't drown MSE.
    # Previous values (1e-4) were summing to ~48k and dominating the loss.
    l1_lambda=1e-6,
    l2_lambda=1e-6,
    # Macro FiLM conditioning - disabled for first-pass retrain to isolate
    # training stability. Re-enable once the base model converges cleanly.
    use_macro_film=False,
    d_macro=64,
    # Factor correlation attention bias - remains disabled until FiLM is back on.
    use_corr_bias=False,
)

# RL Fine-Tuning (Stage 2: GRPO/DAPO after MSE pre-train)
# Methods: "grpo" (KL-penalised), "dapo" (asymmetric clip, no KL), "hybrid"
# Patterns from 5e_dapo_agent.py adapted for CS-Transformer monthly cross-sections
RL_FINETUNE_PARAMS = dict(
    method="grpo",              # "grpo", "dapo", or "hybrid"
    epochs=30,                  # RL fine-tuning epochs
    lr=1e-5,                    # lower LR for fine-tuning
    top_k=100,                  # long portfolio: top K stocks
    bottom_k=100,               # short portfolio: bottom K stocks
    # GRPO params
    grpo_G=4,                   # group size (samples per month)
    grpo_clip_epsilon=0.2,      # PPO-style clipping
    grpo_kl_beta=0.01,          # KL penalty coefficient (GRPO only)
    # DAPO params
    dapo_clip_low=0.20,         # symmetric clip for negative advantages
    dapo_clip_high=0.28,        # higher clip for positive advantages
    dapo_G_min=4,               # dynamic group size min
    dapo_G_max=8,               # dynamic group size max
    # Shared
    patience=10,                # early stopping on validation Rank IC
    freeze_backbone=False,      # True = only train score_head + noise_head
    sigmoid_temperature=0.5,    # for differentiable top-K portfolio
)
