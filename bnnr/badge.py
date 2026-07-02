"""
BADGE: Bayesian Adaptive Drug-disease Graph Enhancement.

A drug repositioning method that jointly estimates the association matrix
and similarity graphs through Bayesian shrinkage. Unlike prior methods
that treat similarities as fixed inputs or blindly replace them, BADGE
uses a density-adaptive confidence weight to optimally fuse prior GIP
(from sparse known associations) with empirical GIP (from the completed
matrix), achieving robustness across data-sparsity regimes.

Core innovation: G_new = lambda * G_empirical + (1-lambda) * G_prior
where lambda = sigma((log10(density) - mu) / tau) automatically adapts
to data sparsity.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .core import BNNR
from .gip import getGIPSim


def _normalised_laplacian(S):
    """L = I - D^{-1/2} S D^{-1/2}  (symmetric normalised Laplacian)."""
    n = S.shape[0]
    d = np.maximum(S.sum(axis=1), 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S_norm = d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    L = np.eye(n) - S_norm
    return np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)


def _prefactor_filter(L, alpha):
    """Cholesky factorisation of (I + alpha * L) for fast repeated solves."""
    A = np.eye(L.shape[0]) + alpha * L
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return cho_factor(A, lower=False)


def _apply_filter_prefactored(cho, V):
    """Solve (I + alpha*L) X = V using pre-computed Cholesky factor."""
    return cho_solve(cho, V)


def _graph_filter(M, L_dis, L_drug, alpha):
    """Bi-directional graph low-pass filter."""
    n_dis, n_drug = M.shape
    M_sm = np.linalg.solve(np.eye(n_dis) + alpha * L_dis, M)
    M_sm = np.linalg.solve(np.eye(n_drug) + alpha * L_drug, M_sm.T).T
    return np.clip(M_sm, 0, 1)


def _shrinkage_weight(density, mu=-3.5, tau=0.5):
    """Density-adaptive Bayesian shrinkage weight.

    lambda = sigmoid((log10(density) - mu) / tau)

    DNdataset (0.015%):     lambda ~ 0.06  (mostly trust prior -> GF-BNNR)
    Fdataset/Cdataset (1%): lambda ~ 0.97  (mostly trust empirical)

    Parameters
    ----------
    density : float, n_known / (n_drug * n_dis)
    mu : float, log10 transition centre (default -3.0 ~ 0.1% density)
    tau : float, transition sharpness (default 0.3)

    Returns
    -------
    lambda_val : float in [0, 1]
    """
    log_d = np.log10(max(density, 1e-8))
    return 1.0 / (1.0 + np.exp(-(log_d - mu) / tau))


def BADGE(Wrr, Wdd, Wdr, alpha=1, beta=10,
          graph_alpha=0.5, gamma_gip=1.0, w_gip=0.3,
          n_iter=2, shrinkage_mu=-3.0, shrinkage_tau=0.3,
          tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
          S_drug=None, S_dis=None,
          verbose=0):
    """Bayesian Adaptive Drug-disease Graph Enhancement.

    Iteratively refines drug-disease association predictions by
    treating similarity graphs as evolving parameters. At each
    iteration, the empirical GIP from the completed matrix is fused
    with the prior GIP via Bayesian shrinkage, where the shrinkage
    weight adapts to data density:

        G_new = lambda * G_emp + (1-lambda) * G_prior

    On ultra-sparse data, lambda -> 0 (trust prior), recovering
    GF-BNNR. On denser data, lambda -> 1 (trust empirical),
    extracting richer manifold structure from the completion.

    Parameters
    ----------
    Wrr : ndarray (n_drug, n_drug), drug-drug similarity matrix.
    Wdd : ndarray (n_dis, n_dis), disease-disease similarity matrix.
    Wdr : ndarray (n_dis, n_drug), disease-drug association (CV-masked).
    alpha : float, BNNR data-fidelity weight.
    beta : float, ADMM penalty parameter.
    graph_alpha : float, graph filter strength.
    gamma_gip : float, GIP kernel bandwidth.
    w_gip : float, GIP fusion weight in [0, 1].
    n_iter : int, max refinement iterations (1 = GF-BNNR, 2-3 = BADGE).
    shrinkage_mu : float, log10 density at transition centre.
    shrinkage_tau : float, transition sharpness.
    tol1, tol2 : float, BNNR convergence thresholds.
    maxiter : int, max BNNR iterations per outer loop.
    a, b : float, predicted-value bounds.
    S_drug, S_dis : ndarray or None, pre-computed similarities (no CV leakage).
    verbose : int, 0=silent, 1=summary, 2=detailed.

    Returns
    -------
    M_final : ndarray (n_dis, n_drug), predicted association matrix.
    history : list of dict, per-iteration diagnostics.
    """
    n_dis, n_drug = Wdr.shape
    known_mask = (Wdr != 0)
    density = np.count_nonzero(Wdr) / (n_dis * n_drug)
    lam = _shrinkage_weight(density, shrinkage_mu, shrinkage_tau)

    # ── prior GIP from sparse known associations ──
    if S_drug is not None and S_dis is not None:
        Sd_cur = S_drug.copy()
        St_cur = S_dis.copy()
        Gp_drug = S_drug  # pre-computed already includes GIP
        Gp_dis = S_dis
    else:
        Gp_dis, Gp_drug = getGIPSim(Wdr, gamma_gip, gamma_gip, 0, 0)
        if Gp_dis is None or Gp_drug is None:
            Sd_cur = Wrr.copy()
            St_cur = Wdd.copy()
            Gp_drug = Wrr.copy()
            Gp_dis = Wdd.copy()
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
                             tol1=tol1, tol2=tol2, maxiter=maxiter,
                             a=a, b=b)
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
            'shrinkage_lambda': float(lam),
            'density': float(density),
            'M_mean': float(M_filtered.mean()),
            'M_std': float(M_filtered.std()),
        }
        history.append(diag)

        if verbose >= 1:
            print(f"  [BADGE] iter={t+1}/{n_iter}  "
                  f"lambda={lam:.4f}  bnnr_iter={bnnr_iter}  "
                  f"M_mean={diag['M_mean']:.4f}")

        # ── Bayesian shrinkage of GIP ──
        if t < n_iter - 1 and lam > 0.01:
            Ge_dis, Ge_drug = getGIPSim(M_cur, gamma_gip, gamma_gip, 0, 0)
            if Ge_dis is not None and Ge_drug is not None:
                # Bayesian shrinkage: fuse prior and empirical GIP
                Gs_drug = lam * Ge_drug + (1 - lam) * Gp_drug
                Gs_dis = lam * Ge_dis + (1 - lam) * Gp_dis
                Sd_cur = w_gip * Gs_drug + (1 - w_gip) * Wrr
                St_cur = w_gip * Gs_dis + (1 - w_gip) * Wdd
            # if GIP fails (isolated nodes), retain previous similarities

    if verbose >= 1 and len(history) > 1:
        delta_M = history[-1]['M_mean'] - history[0]['M_mean']
        print(f"  [BADGE] final: {n_iter} iterations, "
              f"delta_M_mean={delta_M:.6f}")

    return M_cur, history
