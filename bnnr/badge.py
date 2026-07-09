"""
BADGE: Bi-iterative Adaptive Drug-disease Graph Enhancement.

A drug repositioning method that iteratively refines GIP similarity graphs
from the completed association matrix. At each iteration, empirical GIP is
recomputed from the filtered completion and fused with raw chemical/clinical
similarities, enabling the manifold prior to adapt as the completion improves.

Core:  G_new = w_gip * GIP(M_filtered) + (1-w_gip) * S_raw
with  n_iter=2 (default), converging to an improved joint estimate.
"""
import numpy as np

from .core import BNNR
from .gip import getGIPSim


def _normalised_laplacian(S):
    n = S.shape[0]
    d = np.maximum(S.sum(axis=1), 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S_norm = d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    L = np.eye(n) - S_norm
    return np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)


def _graph_filter(M, L_dis, L_drug, alpha):
    n_dis, n_drug = M.shape
    M_sm = np.linalg.solve(np.eye(n_dis) + alpha * L_dis, M)
    M_sm = np.linalg.solve(np.eye(n_drug) + alpha * L_drug, M_sm.T).T
    return np.clip(M_sm, 0, 1)


def BADGE(Wrr, Wdd, Wdr, alpha=1, beta=10,
          graph_alpha=0.5, gamma_gip=1.0, w_gip=0.3,
          n_iter=2,
          tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
          S_drug=None, S_dis=None,
          verbose=0):
    """Bi-iterative Adaptive Drug-disease Graph Enhancement.

    Iteratively refines drug-disease association predictions by
    recomputing GIP similarity from the filtered completion at each
    iteration and fusing it with raw similarities:

        S_new = w_gip * GIP(M_filtered) + (1-w_gip) * S_raw

    n_iter=1 is equivalent to GF-BNNR.  n_iter=2 (default) provides
    one round of GIP recomputation, which yields most of the gain.

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

    for t in range(n_iter):
        # ── build augmented matrix ──
        T = np.block([[St_cur, M_cur],
                       [M_cur.T, Sd_cur]])
        trIndex = (T != 0).astype(np.float64)

        # ── BNNR completion ──
        WW, bnnr_iter = BNNR(alpha=alpha, beta=beta, T=T, trIndex=trIndex,
                             tol1=tol1, tol2=tol2, maxiter=maxiter, a=a, b=b)
        M_raw = WW[:n_dis, -n_drug:]

        # ── bi-directional graph filter ──
        L_dis = _normalised_laplacian(St_cur)
        L_drug = _normalised_laplacian(Sd_cur)
        M_filtered = _graph_filter(M_raw, L_dis, L_drug, graph_alpha)

        # ── preserve known entries ──
        M_cur = np.where(known_mask, Wdr, M_filtered)

        diag = {
            'iteration': t + 1,
            'bnnr_iter': int(bnnr_iter),
            'density': float(density),
            'w_gip': w_gip,
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
