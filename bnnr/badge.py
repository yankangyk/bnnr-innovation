"""
BADGE: Bayesian Adaptive Drug-disease Graph Enhancement.

Joint matrix-kernel estimation for drug repositioning. The similarity kernel is
treated as endogenous — jointly optimized with the completion matrix via
alternating minimization, rather than fixed a priori.

M-step: ADMM with embedded graph Laplacian regularization solves
    min ‖M‖_* + α/2‖P_Ω(M - A)‖² + γ·tr(MᵀL_dis M + M L_drug Mᵀ)
simultaneously — nuclear norm, data fidelity, and graph smoothness in one
Lagrangian.

S-step: GIP recomputed from completed M and fused with raw similarities:
    S_new = w_gip · GIP(M) + (1-w_gip) · S_raw

n_iter=1 → one-shot GIP+BNNR (Demo2 / GF-BNNR), N=1 special case.
n_iter=2 (default) → one GIP recomputation; fixed-point N=2 suffices.
"""
import numpy as np

from .core import BNNR, BNNR_graph_aware
from .filter import normalised_laplacian, graph_filter
from .gip import getGIPSim

# Threshold for switching to fallback: when max matrix dimension exceeds this,
# the embedded graph filter in BNNR_graph_aware becomes too expensive (dense
# Laplacian matmuls per ADMM iteration). We fall back to the efficient two-step
# approach: standard BNNR + post-hoc bilateral graph filter.
LARGE_MATRIX_THRESHOLD = 1000


def BADGE(Wrr, Wdd, Wdr, alpha=1, beta=10,
          graph_alpha=0.5, gamma_gip=1.0, w_gip=0.3,
          n_iter=2,
          tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
          S_drug=None, S_dis=None,
          verbose=0):
    """Joint matrix-kernel estimation via alternating minimization.

    M-step: ADMM with embedded graph Laplacian regularization, jointly
    optimizing nuclear norm + data fidelity + graph smoothness.
    S-step: recompute GIP from M and fuse with raw similarities.

    n_iter=1 → one-shot GIP+BNNR (Demo2/GF-BNNR), N=1 special case.
    n_iter=2 (default) → one GIP recomputation; fixed-point N=2 suffices.

    Parameters
    ----------
    Wrr : ndarray (n_drug, n_drug)
    Wdd : ndarray (n_dis, n_dis)
    Wdr : ndarray (n_dis, n_drug), CV-masked association matrix.
    alpha, beta : float, BNNR data-fidelity and ADMM penalty.
    graph_alpha : float, graph filter strength.
    gamma_gip : float, GIP kernel bandwidth.
    w_gip : float, GIP fusion weight in [0, 1].
    n_iter : int, refinement iterations (1 = GF-BNNR, ≥2 = BADGE).
    tol1, tol2 : float, BNNR convergence thresholds.
    maxiter : int, max BNNR iterations per outer loop.
    a, b : float, predicted-value bounds.
    S_drug, S_dis : ndarray or None, pre-fused similarities.
    verbose : int, 0=silent, 1=summary.

    Returns
    -------
    M_final : ndarray (n_dis, n_drug)
    history : list of dict, per-iteration diagnostics.
    """
    n_dis, n_drug = Wdr.shape
    known_mask = (Wdr != 0)
    density = np.count_nonzero(Wdr) / (n_dis * n_drug)

    # ── initial GIP from sparse associations ──
    if S_drug is not None and S_dis is not None:
        Sd_cur = S_drug.copy()
        St_cur = S_dis.copy()
    else:
        Gp_dis, Gp_drug = getGIPSim(Wdr, gamma_gip, gamma_gip, 0, 0)
        if Gp_dis is None or Gp_drug is None:
            Sd_cur = Wrr.copy()
            St_cur = Wdd.copy()
        else:
            Sd_cur = w_gip * Gp_drug + (1 - w_gip) * Wrr
            St_cur = w_gip * Gp_dis + (1 - w_gip) * Wdd

    history = []
    M_cur = Wdr.copy()

    # Detect large-matrix regime: embedded Laplacian matmuls per ADMM
    # iteration become prohibitively expensive → fall back to two-step.
    is_large = max(n_dis, n_drug) > LARGE_MATRIX_THRESHOLD
    if is_large and verbose >= 1:
        print(f"  [BADGE] large matrix ({n_dis}×{n_drug}), "
              f"falling back to two-step BNNR+filter solver")

    for t in range(n_iter):
        # ── build augmented matrix ──
        T = np.block([[St_cur, M_cur],
                       [M_cur.T, Sd_cur]])
        trIndex = (T != 0).astype(np.float64)

        # ── compute graph Laplacians ──
        L_dis = normalised_laplacian(St_cur)
        L_drug = normalised_laplacian(Sd_cur)

        # ── M-step: completion with graph regularization ──
        if is_large:
            # Two-step fallback: standard BNNR → post-hoc bilateral filter.
            # Avoids expensive per-iteration Laplacian matmuls on large matrices.
            WW, bnnr_iter = BNNR(
                alpha=alpha, beta=beta, T=T, trIndex=trIndex,
                tol1=tol1, tol2=tol2, maxiter=maxiter, a=a, b=b)
            M_raw = WW[:n_dis, -n_drug:]
            M_filtered = graph_filter(M_raw, L_dis, L_drug, graph_alpha)
        else:
            # Plan A: embedded graph regularization in ADMM (joint optimization).
            WW, bnnr_iter = BNNR_graph_aware(
                alpha=alpha, beta=beta, T=T, trIndex=trIndex,
                tol1=tol1, tol2=tol2, maxiter=maxiter, a=a, b=b,
                L_dis=L_dis, L_drug=L_drug, alpha_f=graph_alpha,
                n_dis=n_dis, n_drug=n_drug)
            M_filtered = WW[:n_dis, -n_drug:]

        # ── preserve known entries ──
        M_cur = np.where(known_mask, Wdr, M_filtered)

        diag = {
            'iteration': t + 1,
            'bnnr_iter': int(bnnr_iter),
            'density': float(density),
            'w_gip': w_gip,
            'graph_alpha': graph_alpha,
            'M_mean': float(M_filtered.mean()),
            'M_std': float(M_filtered.std()),
        }
        history.append(diag)

        if verbose >= 1:
            print(f"  [BADGE] iter={t+1}/{n_iter}  "
                  f"bnnr_iter={bnnr_iter}  "
                  f"M_mean={diag['M_mean']:.4f}")

        # ── recompute GIP from filtered completion ──
        if t < n_iter - 1:
            Ge_dis, Ge_drug = getGIPSim(M_cur, gamma_gip, gamma_gip, 0, 0)
            if Ge_dis is not None and Ge_drug is not None:
                Sd_cur = w_gip * Ge_drug + (1 - w_gip) * Wrr
                St_cur = w_gip * Ge_dis + (1 - w_gip) * Wdd

    if verbose >= 1 and len(history) > 1:
        delta_M = history[-1]['M_mean'] - history[0]['M_mean']
        print(f"  [BADGE] final: {n_iter} iterations, "
              f"delta_M_mean={delta_M:.6f}")

    return M_cur, history
