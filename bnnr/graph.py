"""
BNNR_graph.py — Graph-Regularized BNNR variants.

Contains:
  - build_knn_graph(): confidence-weighted kNN graph construction
  - normalized_laplacian() / normalized_laplacian_sparse(): graph Laplacians
  - BNNR_graph(): basic graph-regularized BNNR (GBNNR)
  - BNNR_graph_enhanced_v3(): GBNNR-v3 with gamma confidence + block-wise weights
"""
import numpy as np
import warnings
from scipy import sparse
from .svt import svt, svt_with_rank


def build_knn_graph(S, k=10, sym=True, remove_diag=True, gamma=1.0):
    """
    Build confidence-weighted kNN graph from similarity matrix.

    Parameters
    ----------
    S : ndarray (n, n), similarity matrix
    k : int, number of neighbours
    sym : bool, symmetrise (G = max(G, G.T))
    remove_diag : bool, zero out self-loops
    gamma : float, confidence exponent (>1 strengthens strong edges)

    Returns
    -------
    G : ndarray (n, n), sparse kNN graph
    """
    S = np.array(S, dtype=np.float64).copy()
    n = S.shape[0]
    if remove_diag:
        np.fill_diagonal(S, 0.0)

    if k >= n:
        G = S ** gamma
    else:
        partition_idx = np.argpartition(S, -k, axis=1)[:, -k:]
        G = np.zeros_like(S)
        rows = np.arange(n)[:, None]
        G[rows, partition_idx] = S[rows, partition_idx] ** gamma

    if sym:
        G = np.maximum(G, G.T)

    return G


def normalized_laplacian(S, eps=1e-12):
    """Dense normalized graph Laplacian: L = I - D^{-1/2} S D^{-1/2}."""
    S = np.array(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix")
    d = np.sum(S, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    I = np.eye(S.shape[0], dtype=np.float64)
    L = I - D_inv_sqrt @ S @ D_inv_sqrt
    return L


def normalized_laplacian_sparse(S, eps=1e-8):
    """Sparse normalized graph Laplacian: L = I - D^{-1/2} S D^{-1/2}."""
    if not sparse.issparse(S):
        S = sparse.csr_matrix(S)
    d = np.array(S.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    I = sparse.eye(S.shape[0])
    L = I - D_inv_sqrt @ S @ D_inv_sqrt
    return L.tocsr()


def BNNR_graph(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b,
               L_r=None, L_d=None,
               lambda_r=1e-2, lambda_d=1e-2,
               inner_steps=10, lr=1e-2,
               X_init=None, verbose=0):
    """
    Graph-regularized BNNR (GBNNR).

    Objective:
        min ||X||_* + alpha/2 * ||P_Omega(W - T)||_F^2
            + lambda_r * Tr(W^T L_r W) + lambda_d * Tr(W L_d W^T)
        s.t. X = W, a <= W <= b

    The W-update uses gradient descent (inner_steps iterations) because the
    graph Laplacian terms prevent a closed-form solution.

    Parameters
    ----------
    alpha, beta, tol1, tol2, maxiter, a, b : standard BNNR params
    T : ndarray (m, n), augmented matrix
    trIndex : ndarray (m, n), observation mask
    L_r, L_d : sparse matrix or None, row/col graph Laplacians
    lambda_r, lambda_d : float, graph regularization strengths
    inner_steps : int, gradient steps per ADMM iteration
    lr : float, learning rate for inner gradient steps
    X_init : ndarray or None
    verbose : int, verbosity level

    Returns
    -------
    W : ndarray (m, n), completed matrix
    iter_num : int, ADMM iterations
    info : dict, convergence/rank history
    """
    T = np.array(T, dtype=np.float64)
    trIndex = np.array(trIndex, dtype=np.float64)
    if T.shape != trIndex.shape:
        raise ValueError("T and trIndex must have same shape")

    m, n = T.shape

    if L_r is None:
        L_r = sparse.csr_matrix((m, m), dtype=np.float64)
    elif not sparse.issparse(L_r):
        L_r = sparse.csr_matrix(L_r)

    if L_d is None:
        L_d = sparse.csr_matrix((n, n), dtype=np.float64)
    elif not sparse.issparse(L_d):
        L_d = sparse.csr_matrix(L_d)

    if L_r.shape != (m, m):
        raise ValueError(f"L_r shape {L_r.shape} != ({m}, {m})")
    if L_d.shape != (n, n):
        raise ValueError(f"L_d shape {L_d.shape} != ({n}, {n})")

    if X_init is None:
        X = T.copy()
        W = T.copy()
    else:
        X_init = np.array(X_init, dtype=np.float64)
        if X_init.shape != T.shape:
            raise ValueError(f"X_init shape {X_init.shape} != T shape {T.shape}")
        X = X_init.copy()
        W = X_init.copy()
        if verbose >= 1:
            print(f"[BNNR_graph] using external init, range: [{X.min():.4f}, {X.max():.4f}]")

    Y = T.copy()
    stop1 = 1.0
    stop2 = 1.0
    iter_num = 0
    beta_inv = 1.0 / beta
    rank_safe_factor = 1.5
    n_comp = None

    info = {"stop1_history": [], "stop2_history": [], "rank_history": []}

    while stop1 > tol1 or stop2 > tol2:
        Z = X + beta_inv * Y

        # W-update: inner gradient descent
        for _ in range(inner_steps):
            grad_data = alpha * ((W - T) * trIndex)
            grad_row = L_r @ W
            grad_col = W @ L_d
            grad_admm = beta * (W - Z)

            if sparse.issparse(grad_row):
                grad_row = np.asarray(grad_row.todense())
            if sparse.issparse(grad_col):
                grad_col = np.asarray(grad_col.todense())

            grad = (grad_data + 2.0 * lambda_r * grad_row
                    + 2.0 * lambda_d * grad_col + grad_admm)
            W -= lr * grad
            np.clip(W, a, b, out=W)

        # X-update (SVT)
        X_prev = X
        X, eff_rank, _ = svt_with_rank(W - beta_inv * Y, beta_inv,
                                        n_components=n_comp)
        n_comp = int(eff_rank * rank_safe_factor)
        n_comp = max(n_comp, 10)
        n_comp = min(n_comp, min(T.shape))

        # Y-update
        Y += beta * (X - W)

        # Convergence check
        stop1_prev = stop1
        norm_X_prev = np.linalg.norm(X_prev, 'fro')
        stop1 = np.linalg.norm(X - X_prev, 'fro') / norm_X_prev if norm_X_prev != 0 else 0.0
        stop2 = abs(stop1 - stop1_prev) / max(1.0, abs(stop1_prev))

        iter_num += 1
        info["stop1_history"].append(stop1)
        info["stop2_history"].append(stop2)
        info["rank_history"].append(eff_rank)

        if verbose >= 2:
            print(f"[BNNR_graph] iter={iter_num:03d}, "
                  f"stop1={stop1:.4e}, stop2={stop2:.4e}, rank={eff_rank}")

        if iter_num >= maxiter:
            warnings.warn("reach maximum iteration~~do not converge!!!")
            break

    info["stop1_history"] = np.array(info["stop1_history"])
    info["stop2_history"] = np.array(info["stop2_history"])
    info["rank_history"] = np.array(info["rank_history"])

    return W, iter_num, info


def BNNR_graph_enhanced_v3(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b,
                           Wrr_orig=None, Wdd_orig=None, n_drug=None,
                           knn_k=12, gamma_graph=2.0,
                           lambda_r=1e-3, lambda_d=1e-3,
                           lambda_diag_factor=0.2,
                           inner_steps=10, lr=1e-2,
                           X_init=None, verbose=0):
    """
    Enhanced GBNNR v3 — solves the AUROC-AUPR trade-off.

    Key mechanisms:
      1. Gamma > 1: suppresses weak-edge noise in kNN graph
         (e.g., 0.3^2 = 0.09 vs 0.9^2 = 0.81)
      2. Block-wise adaptive regularization: weaker on diagonal similarity
         blocks (already reliable), stronger on off-diagonal prediction blocks
      3. Adaptive truncated SVD for acceleration

    Parameters
    ----------
    alpha, beta, tol1, tol2, maxiter, a, b : standard BNNR params
    T : ndarray (m, n), augmented matrix [S_drug, Wdr.T; Wdr, S_dis]
    trIndex : ndarray (m, n), observation mask
    Wrr_orig, Wdd_orig : ndarray, original similarity matrices (for kNN graph)
    n_drug : int, number of drugs
    knn_k : int, k for kNN graph
    gamma_graph : float, confidence exponent (>1 suppresses weak edges)
    lambda_r, lambda_d : float, graph regularization strengths
    lambda_diag_factor : float, scaling factor for diagonal-block regularization
    inner_steps : int, gradient steps per ADMM iteration
    lr : float, learning rate
    X_init : ndarray or None
    verbose : int

    Returns
    -------
    W : ndarray (m, n), completed matrix
    iter_num : int, ADMM iterations
    info : dict, convergence/rank history
    """
    T = np.array(T, dtype=np.float64)
    trIndex = np.array(trIndex, dtype=np.float64)
    m, n = T.shape

    if n_drug is None:
        n_drug = m // 2

    # Build confidence-weighted kNN graphs
    G_drug = build_knn_graph(Wrr_orig, k=knn_k, sym=True,
                             remove_diag=True, gamma=gamma_graph)
    G_disease = build_knn_graph(Wdd_orig, k=knn_k, sym=True,
                                remove_diag=True, gamma=gamma_graph)

    L_drug = normalized_laplacian_sparse(G_drug)
    L_disease = normalized_laplacian_sparse(G_disease)
    L_aug = sparse.block_diag([L_drug, L_disease], format='csr')

    # Initialize
    if X_init is None:
        X = T.copy()
        W = T.copy()
    else:
        X = np.array(X_init, dtype=np.float64).copy()
        W = X.copy()

    Y = T.copy()
    stop1, stop2 = 1.0, 1.0
    iter_num = 0
    beta_inv = 1.0 / beta
    rank_safe_factor = 1.5
    n_comp = None

    # Precompute block-wise weight map
    mask_diag = np.zeros((m, n), dtype=float)
    mask_diag[:n_drug, :n_drug] = 1.0
    mask_diag[n_drug:, n_drug:] = 1.0
    weight_map = lambda_diag_factor * mask_diag + (1.0 - mask_diag)

    info = {"stop1_history": [], "rank_history": []}

    while stop1 > tol1 or stop2 > tol2:
        Z = X + beta_inv * Y

        # W-update: inner gradient descent with block-wise adaptive weights
        for _ in range(inner_steps):
            grad_data = alpha * ((W - T) * trIndex)

            grad_row = L_aug @ W
            grad_col = W @ L_aug

            if sparse.issparse(grad_row):
                grad_row = np.asarray(grad_row.todense())
            if sparse.issparse(grad_col):
                grad_col = np.asarray(grad_col.todense())

            grad_admm = beta * (W - Z)

            base_grad_reg = 2.0 * lambda_r * grad_row + 2.0 * lambda_d * grad_col
            adaptive_grad_reg = base_grad_reg * weight_map

            grad = grad_data + adaptive_grad_reg + grad_admm
            W -= lr * grad
            np.clip(W, a, b, out=W)

        # X-update (SVT)
        X_prev = X
        X, eff_rank, _ = svt_with_rank(W - beta_inv * Y, beta_inv,
                                        n_components=n_comp)
        n_comp = int(eff_rank * rank_safe_factor)
        n_comp = max(n_comp, 10)
        n_comp = min(n_comp, min(T.shape))

        # Y-update
        Y += beta * (X - W)

        # Convergence check
        stop1_prev = stop1
        norm_X_prev = np.linalg.norm(X_prev, 'fro')
        stop1 = np.linalg.norm(X - X_prev, 'fro') / norm_X_prev if norm_X_prev != 0 else 0.0
        stop2 = abs(stop1 - stop1_prev) / max(1.0, abs(stop1_prev))

        iter_num += 1
        info["rank_history"].append(eff_rank)

        if iter_num >= maxiter:
            warnings.warn("reach maximum iteration")
            break

    return W, iter_num, info
