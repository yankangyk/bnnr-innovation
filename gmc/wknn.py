"""
WKNN — Weighted K-Nearest-Neighbours soft-label propagation.

Faithful Python port of Yang et al.'s multiGMF ``WKNN.m`` preprocessing:
the masked training association matrix is filled with soft scores propagated
from the K most similar drugs / diseases, weighted by a similarity decay.

    y_m(i, :) = sum_j r^(j-1) * sim_m(i,j) * MD(idx_m(j), :) / sum_j sim_m(i,j)
    y_d(:, i) = sum_j r^(j-1) * sim_d(i,j) * MD(:, idx_d(j)) / sum_j sim_d(i,j)
    MD_new     = max(MD, (y_m + y_d) / 2)

Only external similarity data and the *masked* training matrix are used, so
the soft scores introduce no label leakage under the CVc protocol — the same
reason the original method is accepted practice in this field.
"""

import numpy as np


def knn_network(network, k):
    """Standard top-k neighbour selection (self excluded) per row.

    This is the KNN used by WKNN (keeps the k strongest similarities per
    node) — distinct from ``sparsify_graph`` in ``gmc/filter.py``, which
    symmetrises and re-normalises for Laplacian construction.
    """
    n = network.shape[0]
    net = network.copy()
    np.fill_diagonal(net, 0.0)
    idx = np.argsort(-net, axis=1, kind="stable")[:, :k]
    knn = np.zeros_like(net)
    rows = np.arange(n)[:, None]
    knn[rows, idx] = net[rows, idx]
    return knn


def wknn(md_mat, mm_mat, dd_mat, k=10, r=0.9):
    """WKNN soft-label propagation.

    Parameters
    ----------
    md_mat : ndarray (n_drug, n_disease)
        Masked drug-disease association matrix (training matrix with held-out
        entries zeroed).
    mm_mat : ndarray (n_drug, n_drug)
        Drug similarity matrix (mean-fused, raw — not sparsified).
    dd_mat : ndarray (n_disease, n_disease)
        Disease similarity matrix (mean-fused, raw).
    k : int
        Number of neighbours (multiGMF uses k=10).
    r : float
        Similarity decay factor (multiGMF uses r=0.9).

    Returns
    -------
    md_new : ndarray (n_drug, n_disease)
        ``max(md_mat, soft_fill)`` — known entries preserved, unknown entries
        given neighbour-propagated soft scores in [0, 1].
    """
    n_drug, n_dis = md_mat.shape
    eps = 1e-12
    k = min(int(k), n_drug - 1, n_dis - 1)
    rpow = r ** np.arange(k)  # (k,)

    # ── drug side: y_m(i, c) = Σ_j r^(j-1)·sim_m(i,j)·MD(idx_m(i,j), c) ──
    knn_m = knn_network(mm_mat, k)                        # (n_drug, n_drug)
    sort_m = np.sort(knn_m, axis=1)[:, ::-1][:, :k]       # top-k sims, desc
    idx_m = np.argsort(knn_m, axis=1)[:, ::-1][:, :k]     # neighbour indices
    sum_m = sort_m.sum(axis=1, keepdims=True) + eps
    w_m = rpow[None, :] * sort_m                          # (n_drug, k)
    gathered_m = md_mat[idx_m]                            # (n_drug, k, n_disease)
    y_m = np.einsum("ij,ijc->ic", w_m, gathered_m) / sum_m

    # ── disease side: y_d(i, c) = Σ_j r^(j-1)·sim_d(c,j)·MD(i, idx_d(c,j)) ──
    knn_d = knn_network(dd_mat, k)                        # (n_dis, n_dis)
    sort_d = np.sort(knn_d, axis=1)[:, ::-1][:, :k]
    idx_d = np.argsort(knn_d, axis=1)[:, ::-1][:, :k]
    sum_d = sort_d.sum(axis=1) + eps                     # (n_dis,) — per-column
    w_d = rpow[None, :] * sort_d                          # (n_dis, k)
    gathered_d = md_mat[:, idx_d]                         # (n_drug, n_dis, k)
    y_d = np.einsum("cj,icj->ic", w_d, gathered_d) / sum_d[None, :]

    y_md = 0.5 * (y_m + y_d)
    return np.maximum(md_mat, y_md)
