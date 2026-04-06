"""
screen_autoencoder.py
=====================
Autoencoder-based latent factor discovery from rolling OHLCV windows.

Compresses 63-day rolling windows of (open, high, low, close, volume) into
K latent features per stock-month. The latent features capture price path
"shapes" that linear methods cannot.

Train on IS data only. Extract latent features for screening.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def build_rolling_windows(
    prices: pd.DataFrame,
    window: int = 63,
    features: List[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build rolling OHLCV windows for autoencoder input.

    For each (ticker, month-end date), extract the prior `window` days of
    normalized OHLCV data as a flat vector.

    Returns:
        X: array of shape (n_samples, window * n_features)
        meta: DataFrame with (date, ticker) for each sample
    """
    if features is None:
        features = ["open", "high", "low", "close", "volume"]
    features = [f for f in features if f in prices.columns]

    if not features:
        return np.array([]), pd.DataFrame()

    samples = []
    meta_rows = []

    for tk, grp in prices.groupby("ticker", sort=False):
        grp = grp.sort_values("date")

        # Normalize per-stock: divide by rolling mean for price, log for volume
        norm_data = {}
        for f in features:
            vals = grp[f].values.astype(float)
            if f == "volume":
                vals = np.log1p(np.maximum(vals, 0))
                # Z-score the log volume
                mu = pd.Series(vals).rolling(window, min_periods=window // 2).mean().values
                sig = pd.Series(vals).rolling(window, min_periods=window // 2).std().values
                vals = (vals - mu) / (sig + 1e-9)
            else:
                # Normalize price by rolling mean
                mu = pd.Series(vals).rolling(window, min_periods=window // 2).mean().values
                vals = vals / (mu + 1e-9) - 1  # percent deviation from rolling mean
            norm_data[f] = vals

        dates = grp["date"].values
        n = len(grp)

        # Sample at month-end dates
        month_ends = pd.Series(dates).groupby(
            pd.to_datetime(dates).to_period("M")
        ).transform("last")
        is_month_end = dates == month_ends.values

        for i in range(window, n):
            if not is_month_end[i]:
                continue
            window_data = []
            for f in features:
                w = norm_data[f][i - window:i]
                if np.any(np.isnan(w)):
                    window_data = None
                    break
                window_data.extend(w)
            if window_data is not None:
                samples.append(window_data)
                meta_rows.append({"date": dates[i], "ticker": tk})

    if not samples:
        return np.array([]), pd.DataFrame()

    X = np.array(samples, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, meta


def train_autoencoder(
    X_train: np.ndarray,
    latent_dim: int = 12,
    hidden_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> object:
    """
    Train a simple autoencoder on the rolling window data.

    Architecture: input_dim → hidden_dim → latent_dim → hidden_dim → input_dim

    Returns the trained encoder (input → latent).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  PyTorch not available — using PCA fallback for latent features")
        return _pca_fallback(X_train, latent_dim)

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    AE epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.6f}")

    model.eval()
    return model


def _pca_fallback(X: np.ndarray, n_components: int):
    """PCA fallback when PyTorch is unavailable."""
    from sklearn.decomposition import PCA

    class PCAEncoder:
        def __init__(self, pca):
            self.pca = pca
            self._is_pca = True

        def encode(self, X):
            return self.pca.transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    print(f"    PCA fallback: {n_components} components, "
          f"explained variance = {pca.explained_variance_ratio_.sum():.3f}")
    return PCAEncoder(pca)


def extract_latent_features(
    model,
    X: np.ndarray,
    meta: pd.DataFrame,
    prefix: str = "cand_ae_latent",
) -> pd.DataFrame:
    """
    Extract latent features from trained autoencoder.

    Returns DataFrame with (date, ticker, cand_ae_latent_0, ..., cand_ae_latent_K-1).
    """
    if hasattr(model, '_is_pca'):
        Z = model.encode(X)
    else:
        import torch
        model.eval()
        with torch.no_grad():
            _, Z = model(torch.FloatTensor(X))
            Z = Z.numpy()

    latent_cols = [f"{prefix}_{i}" for i in range(Z.shape[1])]
    latent_df = pd.DataFrame(Z, columns=latent_cols)
    result = pd.concat([meta.reset_index(drop=True), latent_df], axis=1)
    return result


def autoencoder_screen(
    prices: pd.DataFrame,
    end_date: str = "2022-12-31",
    window: int = 63,
    latent_dim: int = 12,
    features: List[str] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Full autoencoder pipeline: build windows → train on IS → extract latent features.

    Returns:
        (latent_df, model)
        latent_df: DataFrame with (date, ticker, cand_ae_latent_0..K-1)
    """
    print("  Building rolling OHLCV windows...")
    X, meta = build_rolling_windows(prices, window=window, features=features)

    if len(X) == 0:
        print("  No valid windows — skipping autoencoder")
        return pd.DataFrame(), None

    print(f"  Windows: {X.shape[0]} samples × {X.shape[1]} features")

    # Split IS/OOS
    meta["date"] = pd.to_datetime(meta["date"])
    is_mask = meta["date"] <= pd.Timestamp(end_date)
    X_is = X[is_mask.values]

    print(f"  Training autoencoder on {len(X_is)} IS samples (latent_dim={latent_dim})...")
    model = train_autoencoder(X_is, latent_dim=latent_dim)

    # Extract features for ALL data (IS + OOS)
    latent_df = extract_latent_features(model, X, meta)
    print(f"  Extracted {latent_dim} latent features for {len(latent_df)} stock-months")

    return latent_df, model
