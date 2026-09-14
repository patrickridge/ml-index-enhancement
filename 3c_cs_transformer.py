"""
3c_cs_transformer.py
Cross-Sectional Transformer for S&P 500 stock ranking.

Architecture (two-stage):
  Stage 1 - Per-Stock Feature Attention:
    Each stock's 100+ features are tokenized (one d_model-dim embedding per feature),
    a learnable [CLS] token is prepended, and a TransformerEncoder attends over the
    feature dimension. The CLS output becomes the stock's summary embedding.
    Weights are SHARED across all stocks (like a shared encoder in set transformers).

  Stage 2 - Cross-Stock Attention:
    All stock embeddings for one month are stacked into a sequence of length N_stocks.
    A MARKET_CLS token is prepended. A second TransformerEncoder attends across the
    full cross-section so each stock "sees" its peers before generating its score.
    That's the main lift over the original FT-Transformer, which processed
    each stock independently and had no way to see the rest of the universe.

  Score Head:
    LayerNorm → Linear(d_model, 1) → scalar score per stock.

Training:
  Each month is one forward pass (batch=1 for Stage 2).
  Months are padded to MAX_N_STOCKS=520; padding_mask prevents attention to pad positions.
  Loss: masked MSE on non-padded stocks. Shuffle at the month level each epoch.
  Walk-forward expanding window, same as 3a_ft_transformer.py.

Outputs:
  data/scores_cs_transformer.parquet
  data/bt_cs_transformer.csv      (long-only top 50)
  data/bt_cs_transformer_ls.csv   (long-short)

Run AFTER 1h_feature_engineering.py (and optionally 1i_orthogonalize.py).
"""

import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats as sp_stats

from config import (
    DATA_DIR, START_DATE, TRAIN_END, VALID_END,
    MIN_SPX_WEIGHT, MAX_ABS_MONTHLY_RET,
    TOP_N, BOTTOM_N, LONG_FRAC, RETRAIN_EVERY,
    TRANSFORMER_CS_PARAMS, RL_FINETUNE_PARAMS,
    USE_ORTHOGONALIZED_FEATURES, MACRO_COLS,
)

# Select feature panel
_panel_file = ("panel_monthly_orthogonalized.parquet" if USE_ORTHOGONALIZED_FEATURES
               else "panel_monthly_enriched.parquet")
PANEL_IN    = DATA_DIR / _panel_file
OUT_SCORES  = DATA_DIR / "scores_cs_transformer.parquet"
OUT_BT_LO   = DATA_DIR / "bt_cs_transformer.csv"
OUT_BT_LS   = DATA_DIR / "bt_cs_transformer_ls.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PERFORMANCE HELPERS  (identical to 3a_ft_transformer.py)

def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return dict(months=len(r), ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    ann    = (1 + r).prod() ** (12 / len(r)) - 1
    vol    = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else np.nan
    nav    = (1 + r).cumprod()
    maxdd  = (nav / nav.cummax() - 1).min()
    return dict(months=len(r), ann=ann, vol=vol, sharpe=sharpe, maxdd=maxdd)


def print_stats(label: str, r: pd.Series):
    s = perf_stats(r)
    print(f"  {label}")
    print(f"    months={s['months']} | ann={s['ann']*100:.2f}% | "
          f"vol={s['vol']*100:.2f}% | sharpe={s['sharpe']:.2f} | maxdd={s['maxdd']*100:.2f}%")


def long_only_ret(df_month: pd.DataFrame, top_n: int) -> float:
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    if len(sub) < top_n:
        return np.nan
    return sub.nlargest(top_n, "score")["fwd_ret_1m"].mean()


def long_short_ret(df_month: pd.DataFrame, frac: float) -> float:
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    n = len(sub)
    if n < 20:
        return np.nan
    k = max(1, int(np.floor(n * frac)))
    sub_s = sub.sort_values("score", ascending=False)
    return sub_s.head(k)["fwd_ret_1m"].mean() - sub_s.tail(k)["fwd_ret_1m"].mean()


# MODEL

class FeatureTokenizer(nn.Module):
    """
    Per-feature linear projection: each scalar → d_model-dim embedding.
    Separate weights per feature (same as in FTTransformer).

    Built-in L1/L2 penalty on per-feature weight norms - lets the model
    learn to zero out useless features instead of manual pre-filtering.
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, n_features) → (N, n_features, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    def feature_penalty(self, l1_lambda: float = 1e-4, l2_lambda: float = 1e-4) -> torch.Tensor:
        """
        Per-feature L1 + L2 penalty on weight norms.

        L1 encourages sparsity: drives entire feature embeddings to zero.
        L2 shrinks all weights, prevents any single feature from dominating.

        Returns scalar penalty to add to the loss.
        """
        # Per-feature norm: (n_features,) - L2 norm of each feature's embedding
        feat_norms = self.weight.norm(dim=1)  # (n_features,)
        l1_penalty = l1_lambda * feat_norms.sum()
        l2_penalty = l2_lambda * (feat_norms ** 2).sum()
        return l1_penalty + l2_penalty

    def feature_importance(self) -> np.ndarray:
        """Return per-feature importance as L2 norm of each feature's weight vector."""
        with torch.no_grad():
            return self.weight.norm(dim=1).cpu().numpy()


class MacroEncoder(nn.Module):
    """
    Encode macro state (VIX, yields, SPX returns, etc.) into a dense embedding.
    Macro features are the same for all stocks in a given month.

    Input:  (n_macro,)
    Output: (d_macro,)
    """
    def __init__(self, n_macro: int, d_macro: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_macro, d_macro),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_macro, d_macro),
            nn.LayerNorm(d_macro),
        )

    def forward(self, macro: torch.Tensor) -> torch.Tensor:
        return self.net(macro)


class MacroFiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on macro state.

    Given macro embedding and per-stock feature tokens, produces per-feature
    (gamma, beta) and modulates: tokens_out = gamma * tokens + beta.

    This creates implicit n_features × n_macro interactions - the model learns
    which factors to trust/distrust under each macro regime.

    Identity-initialized (gamma=1, beta=0) so pre-trained weights still work.
    """
    def __init__(self, n_stock_features: int, d_model: int, d_macro: int):
        super().__init__()
        self.film_gen = nn.Linear(d_macro, n_stock_features * 2)
        self.n_f = n_stock_features
        # Identity init: gamma=1, beta=0 → no modulation at start
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)
        with torch.no_grad():
            self.film_gen.bias[:n_stock_features] = 1.0

    def forward(self, tokens: torch.Tensor, macro_emb: torch.Tensor) -> torch.Tensor:
        """tokens: (N, F, d), macro_emb: (d_macro,) → (N, F, d)"""
        params = self.film_gen(macro_emb)           # (2*F,)
        gamma = params[:self.n_f].unsqueeze(0).unsqueeze(-1)   # (1, F, 1)
        beta  = params[self.n_f:].unsqueeze(0).unsqueeze(-1)   # (1, F, 1)
        return gamma * tokens + beta


class CorrelationAttentionBias(nn.Module):
    """
    Inject pre-computed factor correlation matrix as attention bias in Stage 1.

    Learns per-head scalar weights on the F×F correlation matrix, added to
    attention logits so the model knows which features are correlated.
    """
    def __init__(self, n_features: int, n_heads: int):
        super().__init__()
        self.head_weights = nn.Parameter(torch.zeros(n_heads))
        self.n_features = n_features

    def forward(self, corr_matrix: torch.Tensor) -> torch.Tensor:
        """corr_matrix: (F, F) → bias: (n_heads, F+1, F+1) with CLS padding"""
        F = self.n_features
        padded = torch.zeros(F + 1, F + 1, device=corr_matrix.device)
        padded[1:, 1:] = corr_matrix
        # (n_heads,) → (n_heads, 1, 1) × (F+1, F+1) → (n_heads, F+1, F+1)
        return self.head_weights.unsqueeze(-1).unsqueeze(-1) * padded.unsqueeze(0)


class Stage1AttentionLayer(nn.Module):
    """
    Custom transformer encoder layer that accepts additive attention bias.
    Pre-norm architecture matching the existing norm_first=True setting.
    Used only when use_corr_bias=True (otherwise standard TransformerEncoder).
    """
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2, attn_mask=attn_bias)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class CrossSectionalTransformer(nn.Module):
    """
    Two-stage cross-sectional transformer for monthly stock ranking.

    Stage 1 (feature attention):
      - Input: (MAX_N, n_features) - all stocks in a month (padded)
      - FeatureTokenizer → (MAX_N, n_features, d_model)
      - Prepend CLS_S1 → (MAX_N, n_features+1, d_model)
      - TransformerEncoder over feature dim → (MAX_N, n_features+1, d_model)
      - Extract CLS output → stock_embeddings: (MAX_N, d_model)

    Stage 2 (cross-stock attention):
      - Input: stock_embeddings: (MAX_N, d_model) → treated as seq of length MAX_N
      - Prepend MARKET_CLS → (1, MAX_N+1, d_model)
      - TransformerEncoder with padding_mask → (1, MAX_N+1, d_model)
      - Drop MARKET_CLS → enriched_embeddings: (MAX_N, d_model)

    Score Head:
      - LayerNorm → Linear → scalar → (MAX_N,)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads_s1: int = 4,
        n_layers_s1: int = 2,
        n_heads_s2: int = 4,
        n_layers_s2: int = 2,
        dropout: float = 0.1,
        max_stocks: int = 520,
        n_macro: int = 0,
        d_macro: int = 64,
        use_macro_film: bool = True,
        use_corr_bias: bool = False,
    ):
        super().__init__()
        self.d_model    = d_model
        self.max_stocks = max_stocks
        self.use_macro_film = use_macro_film and n_macro > 0
        self.use_corr_bias  = use_corr_bias

        # Macro FiLM conditioning (modulates feature tokens by macro state)
        if self.use_macro_film:
            self.macro_encoder = MacroEncoder(n_macro, d_macro, dropout)
            self.macro_film    = MacroFiLMLayer(n_features, d_model, d_macro)

        # Stage 1: per-stock feature attention
        self.tokenizer    = FeatureTokenizer(n_features, d_model)
        self.cls_token_s1 = nn.Parameter(torch.zeros(1, 1, d_model))

        if use_corr_bias:
            # Custom layers that accept additive attention bias
            self.corr_bias = CorrelationAttentionBias(n_features, n_heads_s1)
            self.s1_layers = nn.ModuleList([
                Stage1AttentionLayer(d_model, n_heads_s1, d_model * 4, dropout)
                for _ in range(n_layers_s1)
            ])
        else:
            s1_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads_s1,
                dim_feedforward=d_model * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.transformer_s1 = nn.TransformerEncoder(
                s1_layer, num_layers=n_layers_s1, enable_nested_tensor=False
            )

        # Stage 2: cross-stock attention
        self.market_cls = nn.Parameter(torch.zeros(1, 1, d_model))

        s2_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads_s2,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer_s2 = nn.TransformerEncoder(
            s2_layer, num_layers=n_layers_s2, enable_nested_tensor=False
        )

        # Score head
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,            # (MAX_N, n_features) - stock features only
        padding_mask: torch.Tensor,  # (MAX_N,) bool - True = padded/invalid stock
        x_macro: torch.Tensor = None,  # (n_macro,) - macro features (one per month)
        corr_matrix: torch.Tensor = None,  # (F, F) - factor correlation matrix
    ) -> torch.Tensor:              # (MAX_N,) scores
        """
        x:            (MAX_N, F)  - stock feature matrix (padded)
        padding_mask: (MAX_N,)    - True for padded positions
        x_macro:      (n_macro,)  - macro features (same for all stocks in month)
        corr_matrix:  (F, F)      - Spearman correlation matrix of stock features
        Returns:      (MAX_N,)    - scores; caller masks out padded positions
        """
        MAX_N = x.shape[0]

        # Stage 1: embed each stock's features independently
        tokens = self.tokenizer(x)                           # (MAX_N, F, d)

        # Macro FiLM: modulate feature tokens by macro state
        if self.use_macro_film and x_macro is not None:
            macro_emb = self.macro_encoder(x_macro)          # (d_macro,)
            tokens = self.macro_film(tokens, macro_emb)      # (MAX_N, F, d)

        cls_s1 = self.cls_token_s1.expand(MAX_N, -1, -1)    # (MAX_N, 1, d)
        tokens = torch.cat([cls_s1, tokens], dim=1)          # (MAX_N, F+1, d)

        if self.use_corr_bias and corr_matrix is not None:
            # Custom layers with correlation attention bias
            attn_bias = self.corr_bias(corr_matrix)          # (n_heads, F+1, F+1)
            for layer in self.s1_layers:
                tokens = layer(tokens, attn_bias=attn_bias)
            out_s1 = tokens
        else:
            out_s1 = self.transformer_s1(tokens)             # (MAX_N, F+1, d)

        stock_emb = out_s1[:, 0, :]                          # (MAX_N, d) - CLS output

        # Stage 2: cross-stock attention
        stock_seq = stock_emb.unsqueeze(0)                   # (1, MAX_N, d)
        mkt_cls   = self.market_cls.expand(1, -1, -1)        # (1, 1, d)
        stock_seq = torch.cat([mkt_cls, stock_seq], dim=1)   # (1, MAX_N+1, d)

        s2_mask = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool, device=x.device),
            padding_mask.unsqueeze(0),
        ], dim=1)

        out_s2    = self.transformer_s2(stock_seq, src_key_padding_mask=s2_mask)
        enriched  = out_s2[0, 1:, :]                         # (MAX_N, d)
        scores    = self.score_head(enriched).squeeze(-1)    # (MAX_N,)

        return scores

    def forward_enriched(
        self,
        x: torch.Tensor,            # (MAX_N, n_features)
        padding_mask: torch.Tensor,  # (MAX_N,) bool - True = padded
        x_macro: torch.Tensor = None,
        corr_matrix: torch.Tensor = None,
    ) -> tuple:
        """
        Same as forward() but also returns enriched embeddings (before score head).
        Used by RL fine-tuning to feed the noise head.
        Returns: (scores: (MAX_N,), enriched: (MAX_N, d_model))
        """
        MAX_N = x.shape[0]

        # Stage 1: per-stock feature attention
        tokens = self.tokenizer(x)

        if self.use_macro_film and x_macro is not None:
            macro_emb = self.macro_encoder(x_macro)
            tokens = self.macro_film(tokens, macro_emb)

        cls_s1 = self.cls_token_s1.expand(MAX_N, -1, -1)
        tokens = torch.cat([cls_s1, tokens], dim=1)

        if self.use_corr_bias and corr_matrix is not None:
            attn_bias = self.corr_bias(corr_matrix)
            for layer in self.s1_layers:
                tokens = layer(tokens, attn_bias=attn_bias)
            out_s1 = tokens
        else:
            out_s1 = self.transformer_s1(tokens)

        stock_emb = out_s1[:, 0, :]

        # Stage 2: cross-stock attention
        stock_seq = stock_emb.unsqueeze(0)
        mkt_cls   = self.market_cls.expand(1, -1, -1)
        stock_seq = torch.cat([mkt_cls, stock_seq], dim=1)
        s2_mask = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool, device=x.device),
            padding_mask.unsqueeze(0),
        ], dim=1)
        out_s2   = self.transformer_s2(stock_seq, src_key_padding_mask=s2_mask)
        enriched = out_s2[0, 1:, :]
        scores   = self.score_head(enriched).squeeze(-1)

        return scores, enriched


class ScoreNoiseHead(nn.Module):
    """
    Auxiliary noise head for stochastic policy during GRPO/DAPO training.

    Takes enriched stock embeddings (output of Stage 2) and produces
    per-stock log-std for Gaussian exploration noise on scores.
    Replaces the ad-hoc fixed noise in the old REINFORCE implementation.
    """
    LOG_STD_MIN = -10
    LOG_STD_MAX = 2

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, enriched: torch.Tensor) -> torch.Tensor:
        """enriched: (N, d_model) → log_std: (N,)"""
        log_std = self.net(enriched).squeeze(-1)
        return log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)


# DATA PREPARATION

def build_monthly_cross_sections(
    panel: pd.DataFrame,
    stock_feat_cols: list,
    max_stocks: int = 520,
    macro_cols: list = None,
) -> list:
    """
    Convert panel (long format) into list of monthly cross-section dicts.
    Each dict contains padded tensors ready for CrossSectionalTransformer.

    Args:
        stock_feat_cols: stock-level feature columns (cross-sectionally ranked)
        macro_cols: macro feature columns (same value for all stocks per month)

    Returns: list of dicts:
      { date, X:(MAX_N,F_stock), X_macro:(n_macro,), y:(MAX_N,),
        mask:(MAX_N,), tickers:[str], n_valid:int }
    """
    if macro_cols is None:
        macro_cols = []
    months = sorted(panel["date"].unique())
    cross_sections = []

    for m in months:
        sub = panel[panel["date"] == m].dropna(subset=["fwd_ret_1m"]).copy()
        n   = len(sub)

        if n == 0:
            continue

        if n > max_stocks:
            sub = sub.sort_values("ticker").iloc[:max_stocks]
            n   = max_stocks

        X_vals = sub[stock_feat_cols].fillna(0.0).values.astype(np.float32)   # (n, F_stock)
        y_vals = sub["fwd_ret_1m"].values.astype(np.float32)

        # Macro: take from first row (identical for all stocks in a month)
        if macro_cols:
            x_macro = sub[macro_cols].iloc[0].fillna(0.0).values.astype(np.float32)
        else:
            x_macro = np.zeros(0, dtype=np.float32)

        # Pad to max_stocks
        F_stock = len(stock_feat_cols)
        X_pad   = np.zeros((max_stocks, F_stock), dtype=np.float32)
        y_pad   = np.zeros(max_stocks, dtype=np.float32)
        mask    = np.ones(max_stocks, dtype=bool)

        X_pad[:n] = X_vals
        y_pad[:n] = y_vals
        mask[:n]  = False

        cross_sections.append({
            "date":    m,
            "X":       torch.from_numpy(X_pad),
            "X_macro": torch.from_numpy(x_macro),
            "y":       torch.from_numpy(y_pad),
            "mask":    torch.from_numpy(mask),
            "tickers": sub["ticker"].tolist(),
            "n_valid": n,
        })

    return cross_sections


def masked_mse_loss(
    pred:   torch.Tensor,   # (MAX_N,)
    target: torch.Tensor,   # (MAX_N,)
    mask:   torch.Tensor,   # (MAX_N,) bool: True = padded
) -> torch.Tensor:
    """MSE computed only over valid (non-padded) stocks."""
    valid_pred   = pred[~mask]
    valid_target = target[~mask]
    if len(valid_pred) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(valid_pred, valid_target)


def masked_ic_loss(
    pred:   torch.Tensor,   # (MAX_N,)
    target: torch.Tensor,   # (MAX_N,)
    mask:   torch.Tensor,   # (MAX_N,) bool: True = padded
) -> torch.Tensor:
    """
    Negative Pearson IC computed over valid stocks. Differentiable proxy for
    rank-IC - on monthly returns it tracks Spearman IC very closely without
    needing sorting networks.

    Minimising this objective = maximising IC. Same direction as "optimise
    ranking quality directly" but in a differentiable way.
    """
    valid_pred   = pred[~mask]
    valid_target = target[~mask]
    n = valid_pred.numel()
    if n < 5:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    p = valid_pred - valid_pred.mean()
    t = valid_target - valid_target.mean()
    p_norm = p.norm()
    t_norm = t.norm()
    if p_norm < 1e-8 or t_norm < 1e-8:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    ic = (p * t).sum() / (p_norm * t_norm)
    return -ic  # we minimise loss, so return negative IC


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Dispatch between MSE and IC loss based on config."""
    if loss_type == "ic":
        return masked_ic_loss(pred, target, mask)
    return masked_mse_loss(pred, target, mask)


# TRAINING

def train_cs_model(
    train_cs:   list,
    valid_cs:   list,
    n_features: int,
    n_macro: int = 0,
    corr_matrix: torch.Tensor = None,
) -> CrossSectionalTransformer:
    """
    Train CrossSectionalTransformer on monthly cross-sections.
    Shuffles at the month level each epoch.
    Early stopping on mean validation masked MSE.
    """
    p = TRANSFORMER_CS_PARAMS
    model = CrossSectionalTransformer(
        n_features     = n_features,
        d_model        = p["d_model"],
        n_heads_s1     = p["n_heads_s1"],
        n_layers_s1    = p["n_layers_s1"],
        n_heads_s2     = p["n_heads_s2"],
        n_layers_s2    = p["n_layers_s2"],
        dropout        = p["dropout"],
        max_stocks     = p["max_stocks"],
        n_macro        = n_macro,
        d_macro        = p.get("d_macro", 64),
        use_macro_film = p.get("use_macro_film", True),
        use_corr_bias  = p.get("use_corr_bias", False),
    ).to(DEVICE)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=p["epochs"])

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    rng            = np.random.default_rng(seed=42)

    loss_type = p.get("loss_type", "mse")  # "mse" or "ic"

    for epoch in range(p["epochs"]):
        # Train
        model.train()
        epoch_loss    = 0.0
        epoch_penalty = 0.0
        indices       = rng.permutation(len(train_cs))   # shuffle months

        for idx in indices:
            cs      = train_cs[idx]
            X       = cs["X"].to(DEVICE)
            y       = cs["y"].to(DEVICE)
            mask    = cs["mask"].to(DEVICE)
            x_macro = cs["X_macro"].to(DEVICE) if n_macro > 0 else None

            optimiser.zero_grad()
            scores  = model(X, mask, x_macro=x_macro, corr_matrix=corr_matrix)
            core    = masked_loss(scores, y, mask, loss_type=loss_type)
            penalty = model.tokenizer.feature_penalty(
                l1_lambda=p.get("l1_lambda", 1e-6),
                l2_lambda=p.get("l2_lambda", 1e-6),
            )
            loss = core + penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss    += core.item()
            epoch_penalty += penalty.item()

        scheduler.step()

        # Validate (always MSE - comparable across configs)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for cs in valid_cs:
                X       = cs["X"].to(DEVICE)
                y       = cs["y"].to(DEVICE)
                mask    = cs["mask"].to(DEVICE)
                x_macro = cs["X_macro"].to(DEVICE) if n_macro > 0 else None
                scores  = model(X, mask, x_macro=x_macro, corr_matrix=corr_matrix)
                val_losses.append(masked_mse_loss(scores, y, mask).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            n_batches = max(len(train_cs), 1)
            loss_label = "train_ic" if loss_type == "ic" else "train_mse"
            train_val = epoch_loss / n_batches
            # IC loss is stored as negative - flip sign for readable print
            if loss_type == "ic":
                train_val = -train_val
            print(f"    Epoch {epoch+1:3d} | "
                  f"{loss_label}={train_val:.5f} "
                  f"penalty={epoch_penalty/n_batches:.5f} "
                  f"| val_mse={val_loss:.5f} | patience={patience_count}")

        if patience_count >= p["patience"]:
            print(f"    Early stop at epoch {epoch+1} (best val MSE: {best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


# RL FINE-TUNING (Stage 2: portfolio-level reward)

def portfolio_reward(
    scores: torch.Tensor,   # (n_valid,)
    returns: torch.Tensor,  # (n_valid,)
    top_k: int = 100,
    bottom_k: int = 100,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Portfolio-level reward with differentiable top-K selection via sigmoid.

    Steep sigmoid approximates hard top-K/bottom-K while keeping gradients.
    Unlike softmax (which gives ALL stocks positive weight), this concentrates
    weight on top_k long and bottom_k short stocks.

    Returns:
        Scalar active return (long - short) for this month.
    """
    n = scores.shape[0]
    if n < top_k + bottom_k:
        top_k = max(1, n // 4)
        bottom_k = max(1, n // 4)

    # Sort scores descending - gradient flows through sorted_idx
    sorted_scores, sorted_idx = scores.sort(descending=True)
    sorted_returns = returns[sorted_idx]

    # Sigmoid weights: ~1 for top-k, ~0 for rest (soft transition at boundary)
    rank_pos = torch.arange(n, device=scores.device, dtype=scores.dtype)
    long_weights  = torch.sigmoid((top_k - 0.5 - rank_pos) / temperature)
    short_weights = torch.sigmoid((rank_pos - (n - bottom_k) + 0.5) / temperature)

    # Normalize to sum to 1
    long_weights  = long_weights / (long_weights.sum() + 1e-8)
    short_weights = short_weights / (short_weights.sum() + 1e-8)

    long_ret  = (long_weights * sorted_returns).sum()
    short_ret = (short_weights * sorted_returns).sum()

    return long_ret - short_ret


def _group_advantage(rewards: list) -> torch.Tensor:
    """Group-relative advantage: (r - mean) / std.  Matches 5e_dapo_agent.py."""
    r = np.array(rewards, dtype=np.float32)
    return torch.FloatTensor((r - r.mean()) / (r.std() + 1e-8))


def _gaussian_kl(
    model: CrossSectionalTransformer,
    noise_head: ScoreNoiseHead,
    ref_model: CrossSectionalTransformer,
    ref_noise_head: ScoreNoiseHead,
    X: torch.Tensor,
    mask: torch.Tensor,
    x_macro: torch.Tensor = None,
    corr_matrix: torch.Tensor = None,
) -> torch.Tensor:
    """KL(current || reference) for Gaussian score policy."""
    _, enriched = model.forward_enriched(X, mask, x_macro=x_macro, corr_matrix=corr_matrix)
    mu1 = model.score_head(enriched).squeeze(-1)
    s1  = noise_head(enriched).exp()

    with torch.no_grad():
        _, ref_enriched = ref_model.forward_enriched(X, mask, x_macro=x_macro, corr_matrix=corr_matrix)
        mu2 = ref_model.score_head(ref_enriched).squeeze(-1)
        s2  = ref_noise_head(ref_enriched).exp()

    n_valid = (~mask).sum()
    kl = (torch.log(s2 / (s1 + 1e-8))
          + (s1.pow(2) + (mu1 - mu2).pow(2)) / (2 * s2.pow(2) + 1e-8)
          - 0.5)
    return kl[:n_valid].mean()


def _sample_group_cs(
    model: CrossSectionalTransformer,
    noise_head: ScoreNoiseHead,
    X: torch.Tensor,
    mask: torch.Tensor,
    returns: torch.Tensor,
    n_valid: int,
    G: int,
    rp: dict,
    x_macro: torch.Tensor = None,
    corr_matrix: torch.Tensor = None,
) -> tuple:
    """
    Sample G noisy score vectors for one month's cross-section.
    Returns (log_probs_old: list[Tensor], rewards: list[float]).
    """
    scores, enriched = model.forward_enriched(X, mask, x_macro=x_macro, corr_matrix=corr_matrix)
    mu    = scores[:n_valid]                              # (n_valid,)

    # Guard: if the forward pass produced NaN/Inf on this month, skip it.
    # Safer than propagating garbage into the sampler and crashing training.
    if not torch.isfinite(mu).all():
        return [], []

    log_s = noise_head(enriched[:n_valid])                # (n_valid,)
    log_s = torch.nan_to_num(log_s, nan=0.0, posinf=2.0, neginf=-10.0)
    std   = log_s.exp().clamp(min=1e-4, max=10.0)         # numeric floor + ceiling

    if not torch.isfinite(std).all():
        return [], []

    dist = torch.distributions.Normal(mu, std)
    log_probs_old = []
    rewards = []

    for _ in range(G):
        noisy = dist.rsample()                            # (n_valid,)
        lp = dist.log_prob(noisy).sum()                   # scalar
        r  = portfolio_reward(
            noisy, returns,
            top_k=rp["top_k"], bottom_k=rp["bottom_k"],
            temperature=rp.get("sigmoid_temperature", 0.5),
        )
        log_probs_old.append(lp.detach())
        rewards.append(r.item())

    return log_probs_old, rewards


def _compute_val_rank_ic(
    model: CrossSectionalTransformer,
    valid_cs: list,
    n_macro: int = 0,
    corr_matrix: torch.Tensor = None,
) -> float:
    """
    Validation metric: mean Spearman rank IC across validation months.
    Uses rank correlation between predicted scores and realized returns -
    no return leakage since this is a read-only evaluation.
    """
    model.eval()
    ics = []
    with torch.no_grad():
        for cs in valid_cs:
            X = cs["X"].to(DEVICE)
            mask = cs["mask"].to(DEVICE)
            x_macro = cs["X_macro"].to(DEVICE) if n_macro > 0 else None
            n_valid = cs["n_valid"]

            scores = model(X, mask, x_macro=x_macro, corr_matrix=corr_matrix)[:n_valid].cpu().numpy()
            rets   = cs["y"][:n_valid].numpy()

            if len(scores) < 20:
                continue
            ic = sp_stats.spearmanr(scores, rets).statistic
            if np.isfinite(ic):
                ics.append(ic)

    return float(np.mean(ics)) if ics else 0.0


def grpo_finetune_cs_model(
    model: CrossSectionalTransformer,
    train_cs: list,
    valid_cs: list,
    method: str = "grpo",
    n_macro: int = 0,
    corr_matrix: torch.Tensor = None,
) -> CrossSectionalTransformer:
    """
    GRPO/DAPO/Hybrid RL fine-tuning of a pre-trained CS-Transformer.

    Stage 2 of two-stage training:
      1. MSE pre-train (already done) → model predicts returns
      2. RL fine-tune (this function) → model optimizes portfolio-level reward

    Methods (matching 5e_dapo_agent.py patterns):
      - "grpo": G=4 samples, group-relative advantage, PPO clipping + KL penalty
      - "dapo": asymmetric clipping (ε_low=0.20, ε_high=0.28), dynamic G (4→8), no KL
      - "hybrid": DAPO in risk-on regimes, GRPO in risk-off/transition

    Fixes over old REINFORCE implementation:
      1. ScoreNoiseHead for proper stochastic policy (not ad-hoc noise)
      2. No lookahead: trains on IS data only, validates via Rank IC
      3. Differentiable top-K via sigmoid (not softmax over all stocks)
      4. Group-relative advantage (not scalar EMA baseline)
      5. No entropy bonus (exploration via noise head)
      6. PPO-style clipping for stable updates
    """
    rp = RL_FINETUNE_PARAMS
    print(f"\n  RL Fine-Tuning ({method.upper()}): {rp['epochs']} epochs, "
          f"lr={rp['lr']}, top_k={rp['top_k']}")

    # Create noise head
    noise_head = ScoreNoiseHead(model.d_model).to(DEVICE)

    # Optimizer: optionally freeze backbone
    if rp.get("freeze_backbone", False):
        for p in model.parameters():
            p.requires_grad_(False)
        # Unfreeze score head
        for p in model.score_head.parameters():
            p.requires_grad_(True)
        # Unfreeze tokenizer (keep sparsity learning)
        for p in model.tokenizer.parameters():
            p.requires_grad_(True)
        opt_params = list(model.score_head.parameters()) + \
                     list(model.tokenizer.parameters()) + \
                     list(noise_head.parameters())
    else:
        opt_params = list(model.parameters()) + list(noise_head.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=rp["lr"], weight_decay=1e-5)

    # Reference model for GRPO KL penalty
    ref_model = None
    ref_noise_head = None
    if method in ("grpo", "hybrid"):
        ref_model = copy.deepcopy(model)
        ref_noise_head = copy.deepcopy(noise_head)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        for p in ref_noise_head.parameters():
            p.requires_grad_(False)

    # Training loop
    best_val_ic    = -float("inf")
    patience_count = 0
    best_state     = None
    rng = np.random.default_rng(seed=42)

    G_init = rp.get("grpo_G", 4)
    G_max  = rp.get("dapo_G_max", 8)
    clip_eps   = rp.get("grpo_clip_epsilon", 0.2)
    kl_beta    = rp.get("grpo_kl_beta", 0.01)
    dapo_low   = rp.get("dapo_clip_low", 0.20)
    dapo_high  = rp.get("dapo_clip_high", 0.28)
    var_thresh = 0.05

    for epoch in range(rp["epochs"]):
        model.train()
        noise_head.train()
        epoch_rewards = []
        indices = rng.permutation(len(train_cs))

        for idx in indices:
            cs = train_cs[idx]
            X = cs["X"].to(DEVICE)
            y = cs["y"].to(DEVICE)
            mask = cs["mask"].to(DEVICE)
            x_macro = cs["X_macro"].to(DEVICE) if n_macro > 0 else None
            n_valid = cs["n_valid"]
            returns = y[:n_valid]

            # Determine G and method for this month
            use_method = method
            G = G_init

            # Sample G candidates
            log_probs_old, rewards = _sample_group_cs(
                model, noise_head, X, mask, returns, n_valid, G, rp,
                x_macro=x_macro, corr_matrix=corr_matrix,
            )

            # DAPO: dynamic G expansion if high reward variance
            if use_method in ("dapo", "hybrid") and len(rewards) >= 2:
                if float(np.var(rewards)) > var_thresh and G < G_max:
                    extra_lp, extra_r = _sample_group_cs(
                        model, noise_head, X, mask, returns, n_valid,
                        G_max - len(rewards), rp,
                        x_macro=x_macro, corr_matrix=corr_matrix,
                    )
                    log_probs_old.extend(extra_lp)
                    rewards.extend(extra_r)

            if len(rewards) < 2:
                continue

            epoch_rewards.extend(rewards)

            # Group-relative advantage
            adv = _group_advantage(rewards).to(DEVICE)

            # Re-derive log-probs under current policy (for ratio)
            scores_now, enriched_now = model.forward_enriched(
                X, mask, x_macro=x_macro, corr_matrix=corr_matrix)
            mu_now    = scores_now[:n_valid]
            log_s_now = noise_head(enriched_now[:n_valid])
            std_now   = log_s_now.exp()
            dist_now  = torch.distributions.Normal(mu_now, std_now)

            # Re-sample same noisy scores (use old log_probs for ratio)
            # For PPO-style: ratio = exp(new_lp - old_lp)
            # We compute new log-probs for G samples
            log_probs_new = []
            for g in range(len(rewards)):
                # Sample a new noisy score to compute current log-prob
                noisy = dist_now.rsample()
                lp_new = dist_now.log_prob(noisy).sum()
                log_probs_new.append(lp_new)

            lp_new_t = torch.stack(log_probs_new)     # (G,)
            lp_old_t = torch.stack(log_probs_old).to(DEVICE)  # (G,)
            ratio = (lp_new_t - lp_old_t).exp()

            # Clipped policy gradient
            if use_method == "dapo":
                # Asymmetric clipping (5e_dapo_agent.py pattern)
                pos_mask  = adv > 0
                clip_high = torch.where(pos_mask,
                    torch.full_like(ratio, 1.0 + dapo_high),
                    torch.full_like(ratio, 1.0 + dapo_low))
                clip_low  = torch.full_like(ratio, 1.0 - dapo_low)
                ratio_clp = torch.max(torch.min(ratio, clip_high), clip_low)
                pg_loss = -torch.min(ratio * adv, ratio_clp * adv).mean()
            else:
                # GRPO / hybrid: symmetric clipping + KL
                ratio_clp = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                pg_loss = -torch.min(ratio * adv, ratio_clp * adv).mean()

            # KL penalty (GRPO only)
            kl_loss = torch.tensor(0.0, device=DEVICE)
            if use_method in ("grpo", "hybrid") and ref_model is not None:
                kl_loss = kl_beta * _gaussian_kl(
                    model, noise_head, ref_model, ref_noise_head, X, mask,
                    x_macro=x_macro, corr_matrix=corr_matrix,
                )

            # Feature penalty (keep sparsity pressure during RL)
            feat_penalty = model.tokenizer.feature_penalty(
                l1_lambda=TRANSFORMER_CS_PARAMS.get("l1_lambda", 1e-4),
                l2_lambda=TRANSFORMER_CS_PARAMS.get("l2_lambda", 1e-4),
            )

            total_loss = pg_loss + kl_loss + feat_penalty

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(opt_params, 0.5)
            optimizer.step()

        # Validate: Rank IC on validation months (no return leakage)
        val_ic = _compute_val_rank_ic(model, valid_cs, n_macro=n_macro,
                                       corr_matrix=corr_matrix)

        if val_ic > best_val_ic + 1e-6:
            best_val_ic    = val_ic
            patience_count = 0
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "noise_head": {k: v.cpu().clone() for k, v in noise_head.state_dict().items()},
            }
        else:
            patience_count += 1

        if (epoch + 1) % 5 == 0:
            mean_r = np.mean(epoch_rewards) if epoch_rewards else 0.0
            print(f"    RL Epoch {epoch+1:3d} | train_reward={mean_r:.5f} "
                  f"| val_IC={val_ic:.4f} | patience={patience_count}")

        if patience_count >= rp["patience"]:
            print(f"    RL early stop at epoch {epoch+1} (best val IC: {best_val_ic:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Report feature importance after RL
    importances = model.tokenizer.feature_importance()
    n_near_zero = (importances < 0.01).sum()
    print(f"    Feature importance: {n_near_zero}/{len(importances)} features near-zero after RL")

    # Unfreeze all params if backbone was frozen
    if rp.get("freeze_backbone", False):
        for p in model.parameters():
            p.requires_grad_(True)

    model.eval()
    return model


def predict_cs(
    model: CrossSectionalTransformer,
    cs:    dict,
    n_macro: int = 0,
    corr_matrix: torch.Tensor = None,
) -> np.ndarray:
    """
    Predict scores for one cross-section.
    Returns scores only for real (non-padded) stocks: shape (n_valid,).
    """
    model.eval()
    with torch.no_grad():
        X       = cs["X"].to(DEVICE)
        mask    = cs["mask"].to(DEVICE)
        x_macro = cs["X_macro"].to(DEVICE) if n_macro > 0 else None
        scores  = model(X, mask, x_macro=x_macro, corr_matrix=corr_matrix).cpu().numpy()
    return scores[:cs["n_valid"]]


# SEED ENSEMBLE

def train_ensemble(
    train_cs: list,
    valid_cs: list,
    n_features: int,
    n_macro: int = 0,
    corr_matrix: torch.Tensor = None,
    n_seeds: int = 1,
    rl_method: str = None,
) -> list:
    """
    Train a seed ensemble. If n_seeds=1, single model. If >1, train N models
    with different random seeds; later average their scores at prediction time.

    If rl_method is set, each model also goes through RL fine-tuning.

    Returns list of trained models.
    """
    if n_seeds <= 1:
        torch.manual_seed(42)
        np.random.seed(42)
        model = train_cs_model(train_cs, valid_cs, n_features,
                               n_macro=n_macro, corr_matrix=corr_matrix)
        if rl_method:
            model = grpo_finetune_cs_model(model, train_cs, valid_cs,
                                            method=rl_method, n_macro=n_macro,
                                            corr_matrix=corr_matrix)
        return [model]

    models = []
    for seed in range(n_seeds):
        print(f"\n  [Ensemble] Training seed {seed + 1}/{n_seeds}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = train_cs_model(train_cs, valid_cs, n_features,
                           n_macro=n_macro, corr_matrix=corr_matrix)
        if rl_method:
            m = grpo_finetune_cs_model(m, train_cs, valid_cs,
                                        method=rl_method, n_macro=n_macro,
                                        corr_matrix=corr_matrix)
        models.append(m)
    print(f"  [Ensemble] Trained {n_seeds} models; predictions will be averaged.")
    return models


def predict_cs_ensemble(
    models: list,
    cs: dict,
    n_macro: int = 0,
    corr_matrix: torch.Tensor = None,
) -> np.ndarray:
    """Average scores across ensemble. For a single model, same as predict_cs."""
    if len(models) == 1:
        return predict_cs(models[0], cs, n_macro=n_macro, corr_matrix=corr_matrix)
    scored = [predict_cs(m, cs, n_macro=n_macro, corr_matrix=corr_matrix)
              for m in models]
    return np.mean(np.stack(scored), axis=0)


# MAIN

def main():
    print(f"Device: {DEVICE}")
    print(f"Panel:  {PANEL_IN.name}")

    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= START_DATE].reset_index(drop=True)

    # Zombie-ticker filter: keep only real point-in-time S&P 500 constituents.
    #
    # This used to test membership of spx_weights.parquet by (date, ticker) and
    # dropped almost nothing, because 1c writes a row for every ticker in
    # prices.parquet regardless of whether it is still in the index. CPWR, which
    # delisted in 2014, has 24 rows after 2024 at weight 0.000000 and sailed
    # straight through. The model was training on its fabricated returns.
    #
    # Membership now requires a non-trivial index weight, not mere presence.
    spx_weights_path = DATA_DIR / "spx_weights.parquet"
    if spx_weights_path.exists():
        spx_w = pd.read_parquet(spx_weights_path)
        spx_w["date"] = pd.to_datetime(spx_w["date"]) + pd.offsets.MonthEnd(0)
        panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)

        spx_w = spx_w[spx_w["spx_weight"] >= MIN_SPX_WEIGHT]
        valid_keys = set(zip(spx_w["date"], spx_w["ticker"]))
        pre_rows = len(panel)
        mask = list(zip(panel["date"], panel["ticker"]))
        panel = panel[[k in valid_keys for k in mask]].reset_index(drop=True)

        # Impossible returns in names that clear the weight floor
        if "fwd_ret_1m" in panel.columns:
            bad = panel["fwd_ret_1m"].abs() > MAX_ABS_MONTHLY_RET
            if bad.any():
                print(f"Return gate: dropping {int(bad.sum()):,} rows with "
                      f"|fwd_ret_1m| > {MAX_ABS_MONTHLY_RET:.0%}")
                panel = panel[~bad].reset_index(drop=True)

        print(f"Zombie filter: {pre_rows:,} → {len(panel):,} rows "
              f"({pre_rows - len(panel):,} non-member pairs dropped, "
              f"weight floor {MIN_SPX_WEIGHT:.0e})")
    else:
        print(f"[WARN] {spx_weights_path} not found - skipping zombie filter")
        print( "       The model will train on delisted tickers. See Notes item 22.")

    # Separate stock vs macro features
    exclude   = {"date", "ticker", "fwd_ret_1m"}
    all_feat_cols = [c for c in panel.columns if c not in exclude]
    macro_cols = [c for c in MACRO_COLS if c in panel.columns]
    stock_feat_cols = [c for c in all_feat_cols if c not in macro_cols]
    n_stock_features = len(stock_feat_cols)
    n_macro = len(macro_cols)

    print(f"Stock features: {n_stock_features} | Macro features: {n_macro} | "
          f"Rows: {len(panel):,} | Tickers: {panel['ticker'].nunique()}")

    p = TRANSFORMER_CS_PARAMS

    # Time splits
    #
    # Scoring starts the month after OOS_START and runs to the end of the panel.
    # Everything at or before OOS_START is used to fit the initial model, split
    # into train and validation; the loop below then retrains every
    # RETRAIN_EVERY months on an expanding window, so no month is ever scored
    # by a model that saw it.
    #
    # Default OOS_START = VALID_END reproduces the original behaviour: a single
    # 17-month test window from 2024-07. Set it earlier to emit scores across
    # history, which is what the RL walk-forward in 5c needs - it wants ~143
    # months and cannot use a signal that only exists for the last 17:
    #
    #   ML_OOS_START=2013-12-31 python 3c_cs_transformer.py
    #
    # That trains five-plus times over, so it is materially slower. It is the
    # only way to put CS-Transformer scores and the RL overlay on the same
    # footing.
    months    = sorted(panel["date"].unique())
    oos_start = pd.Timestamp(os.environ.get("ML_OOS_START", VALID_END))

    pre_oos = [m for m in months if m <= oos_start]
    if len(pre_oos) < 24:
        raise SystemExit(f"Only {len(pre_oos)} months before {oos_start.date()}; "
                         "need at least 24 to fit an initial model.")

    n_val        = min(18, max(6, len(pre_oos) // 5))
    train_months = pre_oos[:-n_val]
    valid_months = pre_oos[-n_val:]
    test_months  = [m for m in months if m > oos_start]

    if not test_months:
        raise SystemExit(f"No months after {oos_start.date()} to score.")

    print(f"OOS starts after: {oos_start.date()}")
    print(f"Train: {train_months[0].date()} → {train_months[-1].date()} ({len(train_months)} months)")
    print(f"Valid: {valid_months[0].date()} → {valid_months[-1].date()} ({len(valid_months)} months)")
    print(f"Test:  {test_months[0].date()} → {test_months[-1].date()} ({len(test_months)} months)")

    # Build all monthly cross-sections once
    print("\nBuilding monthly cross-sections...")
    all_cs = build_monthly_cross_sections(
        panel, stock_feat_cols, max_stocks=p["max_stocks"], macro_cols=macro_cols,
    )
    cs_map = {cs["date"]: cs for cs in all_cs}

    train_cs = [cs_map[m] for m in train_months if m in cs_map]
    valid_cs = [cs_map[m] for m in valid_months if m in cs_map]
    print(f"  Train cross-sections: {len(train_cs)}")
    print(f"  Valid cross-sections: {len(valid_cs)}")
    print(f"  Max stocks per month (padded to): {p['max_stocks']}")
    if n_macro > 0:
        print(f"  Macro FiLM: {n_macro} macro features → per-feature modulation")

    # Correlation matrix (computed from IS training data only)
    corr_matrix = None
    if p.get("use_corr_bias", False):
        train_panel = panel[panel["date"] <= train_end]
        corr_np = train_panel[stock_feat_cols].corr(method="spearman").values
        corr_matrix = torch.FloatTensor(corr_np).to(DEVICE)
        print(f"  Correlation bias: {corr_matrix.shape[0]}x{corr_matrix.shape[1]} Spearman matrix")

    # Stages 1 + 2: train (optionally as a seed ensemble)
    n_seeds = p.get("ensemble_seeds", 1)
    rl_method = RL_FINETUNE_PARAMS["method"]
    loss_type = p.get("loss_type", "mse")
    if n_seeds > 1:
        print(f"\nTraining ensemble ({n_seeds} seeds) - loss={loss_type.upper()}, "
              f"RL={rl_method.upper()}")
    else:
        print(f"\nStage 1: pre-training (loss={loss_type.upper()})...")
        print(f"Stage 2 follows: RL fine-tuning ({rl_method.upper()})")
    models = train_ensemble(
        train_cs, valid_cs, n_stock_features,
        n_macro=n_macro, corr_matrix=corr_matrix,
        n_seeds=n_seeds, rl_method=rl_method,
    )
    model = models[0] if len(models) == 1 else None  # kept for RL retrain compat below

    # Walk-forward prediction
    all_scores = []

    for i, m in enumerate(test_months):
        # Retrain every RETRAIN_EVERY months with expanding window
        if i > 0 and i % RETRAIN_EVERY == 0:
            retrain_cutoff = m
            seen_months = [mo for mo in months if mo < retrain_cutoff]
            rl_val_n = min(18, max(6, len(seen_months) // 5))
            tr_cs_exp = [cs_map[mo] for mo in seen_months[:-rl_val_n] if mo in cs_map]
            va_cs_rl  = [cs_map[mo] for mo in seen_months[-rl_val_n:] if mo in cs_map]

            # Recompute correlation matrix on expanded training data
            retrain_corr = None
            if p.get("use_corr_bias", False):
                retrain_panel = panel[panel["date"] < retrain_cutoff]
                corr_np = retrain_panel[stock_feat_cols].corr(method="spearman").values
                retrain_corr = torch.FloatTensor(corr_np).to(DEVICE)

            if va_cs_rl:
                print(f"  Retraining at {m.date()} (train={len(tr_cs_exp)}, "
                      f"val={len(va_cs_rl)} months)...")
                models = train_ensemble(
                    tr_cs_exp, va_cs_rl, n_stock_features,
                    n_macro=n_macro, corr_matrix=retrain_corr,
                    n_seeds=n_seeds, rl_method=rl_method,
                )
                corr_matrix = retrain_corr  # use updated corr for predictions

        if m not in cs_map:
            continue

        cs     = cs_map[m]
        scores = predict_cs_ensemble(models, cs, n_macro=n_macro,
                                     corr_matrix=corr_matrix)

        # Match scores back to valid stock rows in the panel
        sub = (panel[panel["date"] == m]
               .dropna(subset=["fwd_ret_1m"])
               .head(cs["n_valid"])
               .copy())
        sub["score"] = scores
        all_scores.append(sub[["date", "ticker", "score", "fwd_ret_1m"]])

    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_df.to_parquet(OUT_SCORES, index=False)
    print(f"\nSaved scores: {OUT_SCORES} | rows={len(scores_df):,}")

    # Checkpoint the final ensemble alongside the scores.
    #
    # Without this a 90-minute run produces the project's headline artifact and
    # keeps nothing that generated it: you cannot audit which model produced a
    # given scores file, re-score without retraining, or reproduce a result
    # after config drifts. That gap is how stale scores ended up being compared
    # against current code for months without anyone noticing.
    #
    # Config is stored next to the weights so a checkpoint is self-describing.
    try:
        import torch as _torch
        ckpt_path = OUT_SCORES.with_name(OUT_SCORES.stem + "_model.pt")
        _torch.save({
            "state_dicts":      [m.state_dict() for m in models],
            "n_stock_features": n_stock_features,
            "n_macro":          n_macro,
            "stock_feat_cols":  stock_feat_cols,
            "macro_cols":       macro_cols,
            "params":           dict(p),
            "rl_method":        rl_method,
            "oos_start":        str(oos_start),
            "train_end":        str(train_months[-1]),
            "n_seeds":          len(models),
            "scores_rows":      len(scores_df),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
    except Exception as e:
        print(f"[WARN] checkpoint not saved: {e}")

    # Backtests
    lo_rets = scores_df.groupby("date").apply(long_only_ret, top_n=TOP_N).rename("port_ret")
    lo_rets = lo_rets.dropna().reset_index()
    lo_rets.to_csv(OUT_BT_LO, index=False)

    ls_rets = scores_df.groupby("date").apply(long_short_ret, frac=LONG_FRAC).rename("ls_ret")
    ls_rets = ls_rets.dropna().reset_index()
    ls_rets.to_csv(OUT_BT_LS, index=False)

    # Print results
    print("\n" + "=" * 65)
    print("CROSS-SECTIONAL TRANSFORMER - TEST PERIOD RESULTS")
    print("=" * 65)
    print_stats(f"Long-Only Top{TOP_N} (equal weight)", lo_rets["port_ret"])
    print_stats(f"Long-Short top/bot {int(LONG_FRAC*100)}%",  ls_rets["ls_ret"])

    # Compare to FT-Transformer if available
    ft_path = DATA_DIR / "scores_transformer.parquet"
    if ft_path.exists():
        ft_sc = pd.read_parquet(ft_path)
        ft_lo = ft_sc.groupby("date").apply(long_only_ret, top_n=TOP_N).dropna()
        ft_ls = ft_sc.groupby("date").apply(long_short_ret, frac=LONG_FRAC).dropna()

        ft_lo_s = perf_stats(ft_lo)
        ft_ls_s = perf_stats(ft_ls)
        cs_lo_s = perf_stats(lo_rets["port_ret"])
        cs_ls_s = perf_stats(ls_rets["ls_ret"])

        print("\n" + "=" * 65)
        print("COMPARISON: FT-Transformer  vs  Cross-Sectional Transformer")
        print("=" * 65)
        print(f"{'Strategy':<32} {'Sharpe':>8}  {'Ann Ret':>8}  {'MaxDD':>8}")
        print("-" * 65)
        print(f"{'FT-Transformer Long-Only':<32} {ft_lo_s['sharpe']:>8.2f}  "
              f"{ft_lo_s['ann']*100:>7.1f}%  {ft_lo_s['maxdd']*100:>7.1f}%")
        print(f"{'CS-Transformer Long-Only':<32} {cs_lo_s['sharpe']:>8.2f}  "
              f"{cs_lo_s['ann']*100:>7.1f}%  {cs_lo_s['maxdd']*100:>7.1f}%")
        print("-" * 65)
        print(f"{'FT-Transformer Long-Short':<32} {ft_ls_s['sharpe']:>8.2f}  "
              f"{ft_ls_s['ann']*100:>7.1f}%  {ft_ls_s['maxdd']*100:>7.1f}%")
        print(f"{'CS-Transformer Long-Short':<32} {cs_ls_s['sharpe']:>8.2f}  "
              f"{cs_ls_s['ann']*100:>7.1f}%  {cs_ls_s['maxdd']*100:>7.1f}%")
        print("=" * 65)


if __name__ == "__main__":
    main()
