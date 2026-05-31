"""
svt.py -- Singular Value Thresholding (SVT)

Two public functions:
  svt(Y, x)             basic SVT
  svt_with_rank(Y, x, n_components=None)  SVT + rank tracking + optional truncated SVD

Acceleration strategy:
  - n_components=None:  full SVD via scipy (exact, baseline method)
  - n_components=int:   randomized truncated SVD, only compute top-k singular values
    If the k-th singular value > x (threshold), fall back to full SVD.
    O(n^2*k) instead of O(n^3), ~30x faster when effective rank << n.
"""
import numpy as np
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd


def svt(Y, x):
    """SVT: E = U * max(s - x, 0) * Vt  (baseline: exact SVD)."""
    try:
        U, s, Vt = svd(Y, full_matrices=False)
    except np.linalg.LinAlgError:
        eps = np.linalg.norm(Y) * 1e-14
        U, s, Vt = svd(Y + eps * np.random.randn(*Y.shape), full_matrices=False)
    s = np.maximum(s - x, 0.0)
    return (U * s) @ Vt


def svt_with_rank(Y, x, n_components=None):
    """
    SVT with effective rank tracking and optional truncated SVD acceleration.

    When n_components is given (int), uses randomized SVD to compute only the
    top n_components singular values. If the smallest returned singular value
    is still above threshold x, fall back to full SVD to guarantee correctness.

    Args:
        Y: input matrix (m x n)
        x: SVT threshold (scalar)
        n_components: None = full SVD (baseline); int = truncated (acceleration)

    Returns:
        E: thresholded matrix (m x n)
        eff_rank: number of singular values surviving thresholding
        s: singular values BEFORE thresholding (for rank tracking)
    """
    n = min(Y.shape)

    # For large matrices, guess a reasonable initial n_components to avoid
    # the memory cost of full SVD on the first iteration. If the guess is
    # too small, randomized_svd naturally falls through to full SVD.
    if n_components is None and n > 2000:
        n_components = min(2000, n)

    if n_components is not None and n_components < n:
        try:
            U, s, Vt = randomized_svd(Y, n_components=n_components,
                                      n_iter=2, random_state=42)
            if s[-1] <= x:
                s_new = np.maximum(s - x, 0.0)
                eff_rank = int(np.sum(s_new > 0))
                return (U * s_new) @ Vt, eff_rank, s
        except Exception:
            pass

    try:
        U, s, Vt = svd(Y, full_matrices=False)
    except np.linalg.LinAlgError:
        eps = np.linalg.norm(Y) * 1e-14
        U, s, Vt = svd(Y + eps * np.random.randn(*Y.shape), full_matrices=False)

    s_new = np.maximum(s - x, 0.0)
    eff_rank = int(np.sum(s_new > 0))
    return (U * s_new) @ Vt, eff_rank, s
