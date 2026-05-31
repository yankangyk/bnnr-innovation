"""
GF-BNNR: Graph-Filtered Bounded Nuclear Norm Regularization.

Applies bi-directional graph low-pass filtering to refine BNNR predictions:
  M_filtered = (I + alpha*L_dis)^{-1} * M_bnnr * (I + alpha*L_drug)^{-1}

The filter enforces smoothness on the drug and disease similarity manifolds.
alpha=0 recovers BNNR exactly; alpha in [0.1, 0.7] empirically improves
both AUROC and AUPR across datasets.
"""

import numpy as np
from .core import BNNR
from .gip import getGIPSim


def _normalised_laplacian(S):
    """L = I - D^{-1/2} S D^{-1/2}  (symmetric normalised Laplacian)."""
    n = S.shape[0]
    d = np.maximum(S.sum(axis=1), 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S_norm = d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    return np.eye(n) - S_norm


def _graph_filter(M, L_dis, L_drug, alpha):
    """Bi-directional graph low-pass filter."""
    n_dis, n_drug = M.shape
    M_sm = np.linalg.solve(np.eye(n_dis) + alpha * L_dis, M)
    M_sm = np.linalg.solve(np.eye(n_drug) + alpha * L_drug, M_sm.T).T
    return np.clip(M_sm, 0, 1)


def GF_BNNR(Wrr, Wdd, Wdr, alpha=1, beta=10,
            tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
            gamma_gip=1, w_gip=0.3,
            graph_alpha=0.5,
            S_drug=None, S_dis=None):
    """
    Graph-Filtered BNNR for drug repositioning.

    Runs BNNR first (optionally with GIP-fused similarities), then applies
    a bi-directional graph low-pass filter to enforce manifold smoothness.

    Parameters
    ----------
    Wrr : ndarray (n_drug, n_drug), drug similarity (original)
    Wdd : ndarray (n_dis, n_dis), disease similarity (original)
    Wdr : ndarray (n_dis, n_drug), disease-drug association (CV-masked)
    alpha, beta, tol1, tol2, maxiter, a, b : BNNR hyperparameters
    gamma_gip, w_gip : GIP kernel bandwidth and fusion weight
    graph_alpha : float, graph filter strength (0 = no filtering, recovers BNNR)
    S_drug, S_dis : ndarray or None, pre-computed similarities (no leakage)

    Returns
    -------
    M_filtered : ndarray (n_dis, n_drug), graph-filtered association matrix
    M_bnnr     : ndarray (n_dis, n_drug), raw BNNR output
    iter_num   : int, BNNR iterations
    """
    n_dis, n_drug = Wdr.shape

    if S_drug is not None and S_dis is not None:
        pass  # use pre-computed similarities directly (no leakage)
    else:
        G_dis, G_drug = getGIPSim(Wdr, gamma_gip, gamma_gip, 0, 0)
        S_dis = w_gip * G_dis + (1 - w_gip) * Wdd
        S_drug = w_gip * G_drug + (1 - w_gip) * Wrr

    # BNNR matrix completion
    T = np.block([[S_dis, Wdr],
                  [Wdr.T, S_drug]])
    trIndex = (T != 0).astype(np.float64)

    WW, iter_num = BNNR(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b,
                        adaptive_svd=False)
    M_bnnr = WW[:n_dis, -n_drug:]

    # Graph filtering
    L_dis = _normalised_laplacian(S_dis)
    L_drug = _normalised_laplacian(S_drug)
    M_filtered = _graph_filter(M_bnnr, L_dis, L_drug, graph_alpha)

    return M_filtered, M_bnnr, iter_num
