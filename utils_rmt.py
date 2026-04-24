"""
utils_rmt.py
Random Matrix Theory (RMT) covariance / correlation matrix denoising.

Core function: rmt_denoise(matrix, q, return_as_corr)
  Applies the Marchenko-Pastur law to replace noise eigenvalues with their mean,
  leaving only genuine signal eigenvalues intact.

Key use cases in this pipeline:
  1. Return covariance cleaning in archive/3_pca_rp_backtest.py
     q = N_assets / T_days  (e.g. 50/252 ≈ 0.198)
     → fixes ill-conditioned covariance, resolves beta=2.07 bug

  2. Feature correlation diagnostics in 2f_factor_diagnostics.py
     q = N_features / T_obs  (e.g. 100/90000 ≈ 0.001)
     → q very small → λ+ ≈ 1.004 → most eigenvalues are "signal" → denoising is mild

References:
  Laloux et al. (1999) "Noise Dressing of Financial Correlation Matrices"
  Plerou et al. (2002) "Random Matrix Approach to Cross Correlations in Financial Data"
"""

import numpy as np
from typing import Tuple


def marchenko_pastur_upper(q: float, sigma_sq: float = 1.0) -> float:
    """
    Compute the Marchenko-Pastur upper eigenvalue bound λ+.

    λ+ = σ² · (1 + √q)²

    Eigenvalues BELOW λ+ are indistinguishable from a random (noise) matrix.
    Eigenvalues ABOVE λ+ contain genuine statistical signal.

    Parameters
    ----------
    q : float
        Ratio N/T (variables / observations). Must be > 0.
        For q >= 1 (more variables than observations) the MP distribution
        is degenerate - pass q < 1 for meaningful denoising.
    sigma_sq : float
        Population variance. For a correlation matrix σ² = 1.0.

    Returns
    -------
    float  Marchenko-Pastur upper bound λ+
    """
    return sigma_sq * (1.0 + np.sqrt(q)) ** 2


def rmt_denoise(
    matrix: np.ndarray,
    q: float,
    return_as_corr: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Denoise a symmetric PSD covariance or correlation matrix using Marchenko-Pastur.

    Algorithm
    ---------
    1. Convert input to correlation matrix C (save std devs for round-trip)
    2. Eigendecompose C = V D V^T  (ascending order, then flip to descending)
    3. Compute MP upper bound λ+ = (1 + √q)²  (for correlation matrix, σ²=1)
    4. Identify noise eigenvalues: those below λ+
    5. Replace ALL noise eigenvalues with their mean  (preserves trace; avoids rank deficiency)
    6. Reconstruct: C_clean = V D_clean V^T
    7. Re-normalise diagonal to 1.0 (numerical drift fix)
    8. If input was covariance: Σ_clean = diag(σ) @ C_clean @ diag(σ)

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Empirical covariance or correlation matrix. Assumed symmetric PSD.
    q : float
        Ratio N/T.  For covariance of 50 assets over 252 days: q = 50/252 ≈ 0.198
    return_as_corr : bool
        If True, return the denoised correlation matrix (diagonal = 1).
        If False (default), return in the original scale (covariance if input was covariance).

    Returns
    -------
    denoised : np.ndarray, shape (N, N)
        Denoised matrix.
    info : dict
        Diagnostics:
          lambda_plus   - MP upper bound
          n_signal      - number of eigenvalues above λ+ (genuine signal)
          n_noise       - number of eigenvalues below λ+ (replaced)
          noise_mean    - value used to replace noise eigenvalues
          signal_ratio  - fraction of total eigenvalue mass in signal components
    """
    matrix = np.array(matrix, dtype=float)
    N = matrix.shape[0]

    # Symmetrize for numerical stability
    matrix = 0.5 * (matrix + matrix.T)

    # Step 1: extract std devs and convert to correlation
    diag_var = np.diag(matrix)
    std_devs = np.sqrt(np.maximum(diag_var, 1e-14))
    D_inv = np.diag(1.0 / std_devs)
    corr = D_inv @ matrix @ D_inv
    np.fill_diagonal(corr, 1.0)   # enforce exact diagonal

    # Step 2: eigendecompose
    # np.linalg.eigh returns eigenvalues in ascending order; flip to descending
    eigenvalues_asc, eigenvectors_asc = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues_asc)[::-1]
    eigenvalues  = eigenvalues_asc[idx]
    eigenvectors = eigenvectors_asc[:, idx]

    # Step 3: MP upper bound (σ²=1 for correlation matrix)
    lambda_plus = marchenko_pastur_upper(q, sigma_sq=1.0)

    # Step 4: identify noise vs signal
    noise_mask  = eigenvalues < lambda_plus
    signal_mask = ~noise_mask

    n_signal = int(signal_mask.sum())
    n_noise  = int(noise_mask.sum())

    # Edge case: no noise eigenvalues - return unchanged
    if n_noise == 0:
        info = dict(lambda_plus=lambda_plus, n_signal=n_signal, n_noise=0,
                    noise_mean=float("nan"), signal_ratio=1.0)
        return (corr.copy() if return_as_corr else matrix.copy()), info

    # Step 5: replace noise eigenvalues with their mean
    noise_mean = float(eigenvalues[noise_mask].mean())
    eigenvalues_clean = eigenvalues.copy()
    eigenvalues_clean[noise_mask] = noise_mean

    # Step 6: reconstruct cleaned correlation matrix
    corr_clean = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T

    # Step 7: re-normalise diagonal to 1.0
    d = np.sqrt(np.maximum(np.diag(corr_clean), 1e-14))
    D_renorm = np.diag(1.0 / d)
    corr_clean = D_renorm @ corr_clean @ D_renorm
    np.fill_diagonal(corr_clean, 1.0)

    # Diagnostic: signal ratio
    total_trace  = float(eigenvalues.sum())
    signal_ratio = float(eigenvalues[signal_mask].sum() / max(total_trace, 1e-14))

    info = dict(
        lambda_plus=float(lambda_plus),
        n_signal=n_signal,
        n_noise=n_noise,
        noise_mean=noise_mean,
        signal_ratio=signal_ratio,
    )

    if return_as_corr:
        return corr_clean, info

    # Step 8: convert back to covariance
    D_std = np.diag(std_devs)
    cov_clean = D_std @ corr_clean @ D_std
    return cov_clean, info


def rmt_denoise_feature_corr(
    feature_matrix: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Convenience wrapper: denoise the correlation matrix of a feature panel.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (T_obs, N_features)
        All (date × ticker) observations stacked row-wise.
        N_features = number of factors (e.g. 100).
        T_obs      = total observations (e.g. 180 months × 500 stocks = 90 000).

    Returns
    -------
    corr_clean : np.ndarray, shape (N_features, N_features)
        Denoised feature correlation matrix.
    info : dict
        Diagnostic info (see rmt_denoise).
    """
    T, N = feature_matrix.shape
    q = N / T   # typically very small → λ+ ≈ 1.0 → most eigenvalues signal
    corr_empirical = np.corrcoef(feature_matrix, rowvar=False)
    corr_clean, info = rmt_denoise(corr_empirical, q=q, return_as_corr=True)
    if verbose:
        print(f"[RMT Feature Corr] T={T:,}, N={N}, q={q:.6f}")
        print(f"  MP upper bound λ+ = {info['lambda_plus']:.4f}")
        print(f"  Signal eigenvalues: {info['n_signal']} / {N}  "
              f"| Noise: {info['n_noise']}")
        print(f"  Signal variance ratio: {info['signal_ratio']:.4f}")
    return corr_clean, info
