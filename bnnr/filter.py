"""
Graph regularization utilities for GRMC (Graph-Regularized Matrix Completion).

Core components:
  normalised_laplacian(S) — symmetric normalised Laplacian L = I - D^{-1/2} S D^{-1/2}
  graph_filter(M, L_dis, L_drug, alpha) — bilateral exact Cholesky low-pass filter
  GF_BNNR(...) — [DEPRECATED] thin wrapper around GRMC for backward compatibility

The filter enforces smoothness on the drug and disease similarity manifolds.
alpha=0 recovers BNNR exactly; alpha in [0.1, 0.7] empirically improves
both AUROC and AUPR across datasets.
"""

import numpy as np


def sparsify_graph(S, k):
    """Keep only top-k neighbors per node, symmetrize, re-normalize diagonal.

    Dense similarity graphs contain weak/noisy edges between unrelated
    entities. Sparsification retains only the k strongest connections per
    node, producing a cleaner Laplacian for graph regularization.

    Args:
        S: (n, n) similarity matrix with 1 on diagonal
        k: number of neighbors to retain per node (excluding self)

    Returns:
        S_sparse: (n, n) sparsified similarity matrix, symmetric, diag=1
    """
    if k <= 0 or k >= S.shape[0] - 1:
        return S.copy()

    n = S.shape[0]
    S_sparse = np.zeros_like(S)

    # Zero out diagonal for neighbor selection
    S_nodiag = S.copy()
    S_nodiag.flat[::n + 1] = 0.0

    # For each node, keep top-k off-diagonal neighbors
    for i in range(n):
        top_k = np.argpartition(-S_nodiag[i], k)[:k]
        S_sparse[i, top_k] = S[i, top_k]

    # Symmetrize: edge exists if either direction selected it
    S_sparse = np.maximum(S_sparse, S_sparse.T)

    # Restore diagonal
    np.fill_diagonal(S_sparse, 1.0)

    return S_sparse


def normalised_laplacian(S):
    """L = I - D^{-1/2} S D^{-1/2}  (symmetric normalised Laplacian)."""
    n = S.shape[0]
    d = np.maximum(S.sum(axis=1), 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S_norm = d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    # In-place: L = I - S_norm, avoid allocating np.eye(n) for large matrices
    L = -S_norm
    L.flat[::n + 1] += 1.0
    return np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)


def graph_filter(M, L_dis, L_drug, alpha):
    """Bi-directional graph low-pass filter.

    M_filtered = (I + alpha*L_dis)^{-1} * M * (I + alpha*L_drug)^{-1}

    Uses exact Cholesky solve via np.linalg.solve.
    """
    n_dis, n_drug = M.shape
    # A_dis = I + alpha * L_dis — form in-place, avoid np.eye(n_dis) allocation
    A_dis = alpha * L_dis
    A_dis.flat[::n_dis + 1] += 1.0
    M_sm = np.linalg.solve(A_dis, M)
    # A_drug = I + alpha * L_drug
    A_drug = alpha * L_drug
    A_drug.flat[::n_drug + 1] += 1.0
    M_sm = np.linalg.solve(A_drug, M_sm.T).T
    return np.clip(M_sm, 0, 1)


def GF_BNNR(Wrr, Wdd, Wdr, alpha=1, beta=10,
            tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
            graph_alpha=0.5,
            S_drug=None, S_dis=None):
    """[DEPRECATED] Thin wrapper around GRMC for backward compatibility.

    GF-BNNR previously ran BNNR with GIP-fused similarities then applied
    a bilateral graph filter. GIP has been shown to be counterproductive
    under CVc — raw similarities consistently outperform GIP-fused graphs.

    This function now delegates to GRMC(). Use ``GRMC()`` directly for
    new code.

    Parameters
    ----------
    Wrr : ndarray (n_drug, n_drug)
    Wdd : ndarray (n_dis, n_dis)
    Wdr : ndarray (n_dis, n_drug)
    alpha, beta, tol1, tol2, maxiter, a, b : BNNR hyperparameters
    graph_alpha : float
        Graph filter strength (0 = no filtering, recovers BNNR).
    S_drug, S_dis : ndarray or None
        Pre-computed similarities (no data leakage).

    Returns
    -------
    M_filtered : ndarray (n_dis, n_drug)
        Graph-filtered association matrix.
    M_bnnr : ndarray (n_dis, n_drug)
        Same as M_filtered (kept for backward compat — previously raw BNNR).
    iter_num : int
        BNNR iterations.
    """
    from .sgrmc import GRMC

    M_final, history = GRMC(Wrr, Wdd, Wdr, alpha=alpha, beta=beta,
                            graph_alpha=graph_alpha,
                            tol1=tol1, tol2=tol2, maxiter=maxiter, a=a, b=b,
                            S_drug=S_drug, S_dis=S_dis)
    iter_num = history[0]["bnnr_iter"] if history else 0
    return M_final, M_final, int(iter_num)
