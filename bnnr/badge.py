"""
SGRMC: Sparsified Graph-Regularized Matrix Completion.

Single-pass matrix completion with three components:
  1. Density-adaptive KNN graph sparsification (k=5 for ρ>0.1%, k=50 otherwise)
  2. Graph Laplacian regularization (Plan A: embedded Neumann ≤1000; Plan B: Cholesky >1000)
  3. Single-pass — no GIP fusion, no alternating iterations.

M-step: ADMM with embedded graph Laplacian regularization solves
    min ‖M‖_* + α/2‖P_Ω(M - A)‖² + γ·tr(MᵀL_dis M + M L_drug Mᵀ)
simultaneously — nuclear norm, data fidelity, and graph smoothness in one
Lagrangian.

Two solver paths selected automatically by matrix size:
  Plan A (≤1000): embedded graph regularization in ADMM via first-order
                  Neumann approximation — joint optimization.
  Plan B (>1000): two-step — standard ADMM then post-hoc bilateral exact
                  Cholesky graph filter for large matrices.

Note: The primary function is named ``GRMC()`` for code compatibility; the
method is referred to as **SGRMC** (Sparsified GRMC) in the accompanying paper.
"""
import numpy as np

from .core import BNNR, BNNR_graph_aware
from .filter import normalised_laplacian, graph_filter, sparsify_graph

# Threshold for switching to fallback: when max matrix dimension exceeds this,
# the embedded graph filter in BNNR_graph_aware becomes too expensive (dense
# Laplacian matmuls per ADMM iteration). We fall back to the efficient two-step
# approach: standard BNNR + post-hoc bilateral graph filter.
LARGE_MATRIX_THRESHOLD = 1000


def GRMC(Wrr, Wdd, Wdr, alpha=1, beta=10,
         graph_alpha=0.7,
         tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
         S_drug=None, S_dis=None,
         knn_k=None,
         verbose=0):
    """Graph-Regularized Matrix Completion — single-pass, no GIP.

    Uses raw structural similarity graphs directly (no GIP fusion) to
    regularize matrix completion via graph Laplacian smoothness.

    Parameters
    ----------
    Wrr : ndarray (n_drug, n_drug)
        Raw drug similarity matrix.
    Wdd : ndarray (n_dis, n_dis)
        Raw disease similarity matrix.
    Wdr : ndarray (n_dis, n_drug)
        CV-masked association matrix.
    alpha, beta : float
        BNNR data-fidelity and ADMM penalty.
    graph_alpha : float
        Graph filter strength (0 = BNNR, higher = more smoothing).
    tol1, tol2 : float
        BNNR convergence thresholds.
    maxiter : int
        Maximum BNNR iterations.
    a, b : float
        Predicted-value bounds [a, b].
    S_drug, S_dis : ndarray or None
        Pre-computed similarity matrices (no data leakage).
    knn_k : int or None
        KNN graph sparsification: keep only top-k neighbors per node.
        None (default): auto — k=5 for dense (>0.1%), k=50 for ultra-sparse.
        0: disable sparsification (full dense graph).
        >0: explicit k value.
    verbose : int
        0 = silent, 1 = summary.

    Returns
    -------
    M_final : ndarray (n_dis, n_drug)
        Completed association matrix.
    history : list of dict
        Single-entry list with diagnostics.
    """
    n_dis, n_drug = Wdr.shape
    known_mask = (Wdr != 0)
    density = np.count_nonzero(Wdr) / (n_dis * n_drug)

    # ── raw structural similarities (no GIP) ──
    if S_drug is not None and S_dis is not None:
        Sd_cur = S_drug.copy()
        St_cur = S_dis.copy()
    else:
        Sd_cur = Wrr.copy()
        St_cur = Wdd.copy()

    # ── KNN graph sparsification (adaptive by default) ──
    if knn_k is None:
        # Auto-select: aggressive (k=5) for dense graphs where weak edges
        # are noise; mild (k=50) for ultra-sparse matrices where similarity
        # edges are scarce information. Threshold: 0.1% association density.
        knn_k = 5 if density > 0.001 else 50
    if knn_k > 0:
        Sd_cur = sparsify_graph(Sd_cur, knn_k)
        St_cur = sparsify_graph(St_cur, knn_k)

    # ── build augmented matrix (drugs-first: matches build_augmented_matrix) ──
    T = np.block([[Sd_cur, Wdr.T],
                  [Wdr, St_cur]])
    trIndex = (T != 0)  # bool mask — saves ~250 MiB on DNdataset

    # ── compute graph Laplacians ──
    L_dis = normalised_laplacian(St_cur)
    L_drug = normalised_laplacian(Sd_cur)

    # ── M-step: completion with graph regularization ──
    is_large = max(n_dis, n_drug) > LARGE_MATRIX_THRESHOLD
    if is_large:
        if verbose >= 1:
            print(f"  [GRMC] large matrix ({n_dis}×{n_drug}), "
                  f"falling back to two-step BNNR+filter solver")
        # Plan B: two-step — standard BNNR → post-hoc bilateral Cholesky filter
        WW, bnnr_iter = BNNR(
            alpha=alpha, beta=beta, T=T, trIndex=trIndex,
            tol1=tol1, tol2=tol2, maxiter=maxiter, a=a, b=b)
        M_raw = WW[-n_dis:, :n_drug]
        M_filtered = graph_filter(M_raw, L_dis, L_drug, graph_alpha)
    else:
        # Plan A: embedded graph regularization in ADMM (joint optimization
        # via first-order Neumann approximation).
        WW, bnnr_iter = BNNR_graph_aware(
            alpha=alpha, beta=beta, T=T, trIndex=trIndex,
            tol1=tol1, tol2=tol2, maxiter=maxiter, a=a, b=b,
            L_dis=L_dis, L_drug=L_drug, alpha_f=graph_alpha,
            n_dis=n_dis, n_drug=n_drug)
        M_filtered = WW[-n_dis:, :n_drug]

    # ── preserve known entries ──
    M_final = np.where(known_mask, Wdr, M_filtered)

    diag = {
        'iteration': 1,
        'bnnr_iter': int(bnnr_iter),
        'density': float(density),
        'graph_alpha': graph_alpha,
        'knn_k': int(knn_k),
        'M_mean': float(M_filtered.mean()),
        'M_std': float(M_filtered.std()),
    }

    if verbose >= 1:
        print(f"  [GRMC] bnnr_iter={bnnr_iter}  "
              f"M_mean={diag['M_mean']:.4f}")

    return M_final, [diag]


def BADGE(Wrr, Wdd, Wdr, alpha=1, beta=10,
          graph_alpha=0.5, gamma_gip=1.0, w_gip=0.3,
          n_iter=2,
          tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
          S_drug=None, S_dis=None,
          verbose=0):
    """[DEPRECATED] Thin wrapper around GRMC for backward compatibility.

    BADGE (Bayesian Adaptive Drug-disease Graph Enhancement) previously
    implemented joint matrix-kernel estimation with GIP fusion and
    alternating minimization. GIP has been shown to be counterproductive
    under CVc — raw similarities consistently outperform GIP-fused graphs.

    This function now delegates to GRMC() (Graph-Regularized Matrix
    Completion). The gamma_gip, w_gip, and n_iter parameters are accepted
    for backward compatibility but **ignored**.

    Use ``GRMC()`` directly for new code.

    Parameters
    ----------
    Wrr : ndarray (n_drug, n_drug)
    Wdd : ndarray (n_dis, n_dis)
    Wdr : ndarray (n_dis, n_drug)
    alpha, beta : float
    graph_alpha : float
        Graph filter strength (the only regularization used).
    gamma_gip : float
        [IGNORED — GIP bandwidth, kept for backward compat]
    w_gip : float
        [IGNORED — GIP fusion weight, kept for backward compat]
    n_iter : int
        [IGNORED — alternating iterations, kept for backward compat]
    tol1, tol2 : float
    maxiter : int
    a, b : float
    S_drug, S_dis : ndarray or None
    verbose : int

    Returns
    -------
    M_final : ndarray (n_dis, n_drug)
    history : list of dict
    """
    import warnings
    warnings.warn(
        "BADGE is deprecated — use GRMC() instead. "
        "GIP fusion (w_gip, gamma_gip) and alternating iterations (n_iter) "
        "are ignored. Raw similarities are used directly.",
        DeprecationWarning, stacklevel=2)

    return GRMC(Wrr, Wdd, Wdr, alpha=alpha, beta=beta,
                graph_alpha=graph_alpha,
                tol1=tol1, tol2=tol2, maxiter=maxiter, a=a, b=b,
                S_drug=S_drug, S_dis=S_dis,
                verbose=verbose)
