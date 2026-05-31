"""
BNNR core algorithms: standard BNNR and rank-adaptive RA-BNNR.

Contains:
  - BNNR()            : baseline bounded nuclear norm regularization (ADMM + SVT)
  - BNNR_adaptive()   : rank-adaptive beta scheduling
  - infer_ra_params() : scale-aware hyperparameter auto-inference
"""
import numpy as np
import warnings
from .svt import svt, svt_with_rank


def BNNR(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b, X_init=None,
         adaptive_svd=True):
    """
    Bounded Nuclear Norm Regularization for matrix completion.
    Solves: min ||X||_*  s.t. P_Omega(X) = P_Omega(T), a <= X_ij <= b

    Parameters
    ----------
    alpha, beta: float, algorithm hyperparameters
    T: np.ndarray, target matrix, shape (m, n)
    trIndex: np.ndarray, 0-1 observation mask, same shape as T
    tol1, tol2: float, convergence thresholds
    maxiter: int, maximum iterations
    a, b: float, value bounds [a, b]
    X_init: np.ndarray or None, optional initial value for X and W
    adaptive_svd: bool, if True use adaptive truncated SVD (~30x faster, matches
        BNNR_master1). If False, use full SVD (exact, matches BNNR_baseline).

    Returns
    -------
    T_recovery: np.ndarray, completed matrix
    iter: int, actual iteration count
    """
    T = np.array(T, dtype=np.float64)
    trIndex = np.array(trIndex, dtype=np.float64)
    if T.shape != trIndex.shape:
        raise ValueError("T and trIndex must have same shape")

    if X_init is None:
        X = T.copy()
        W = T.copy()
    else:
        X_init = np.array(X_init, dtype=np.float64)
        if X_init.shape != T.shape:
            raise ValueError(f"X_init shape {X_init.shape} != T shape {T.shape}")
        X = X_init.copy()
        W = X_init.copy()

    Y = T.copy()
    i = 1
    stop1 = 1.0
    stop2 = 1.0
    iter_num = 0

    # Precompute constants
    beta_inv = 1.0 / beta
    alpha_over_beta = alpha / beta
    alpha_sum = alpha + beta
    alpha_ratio = alpha / alpha_sum

    # Adaptive truncation (only when adaptive_svd=True)
    rank_safe_factor = 2.0
    n_comp = None

    # Precompute T * trIndex (never changes)
    T_masked = T * trIndex

    while stop1 > tol1 or stop2 > tol2:
        # W-update: closed-form solution + clip to [a, b]
        tran = beta_inv * Y + alpha_over_beta * T_masked + X
        W = tran - alpha_ratio * (tran * trIndex)
        np.clip(W, a, b, out=W)

        # X-update: Singular Value Thresholding
        if adaptive_svd:
            X_new, eff_rank, _ = svt_with_rank(W - beta_inv * Y, beta_inv,
                                                n_components=n_comp)
            n_comp = int(eff_rank * rank_safe_factor)
            n_comp = max(n_comp, 10)
            n_comp = min(n_comp, min(T.shape))
        else:
            X_new = svt(W - beta_inv * Y, beta_inv)

        # Y-update: dual ascent
        Y += beta * (X_new - W)

        # Convergence check
        stop1_prev = stop1
        norm_X = np.linalg.norm(X, 'fro')
        stop1 = np.linalg.norm(X_new - X, 'fro') / norm_X if norm_X != 0 else 0.0
        stop2 = abs(stop1 - stop1_prev) / max(1.0, abs(stop1_prev))

        X = X_new  # swap reference, no copy
        i += 1

        if i <= maxiter:
            iter_num = i - 1
        else:
            iter_num = maxiter
            warnings.warn("reach maximum iteration~~do not converge!!!")
            break

    return W, iter_num


def infer_ra_params(T_shape, density):
    """
    Scale-Aware Hyperparameter Selection for Rank-Aware BNNR.

    Automatically infers RA hyperparameters from the augmented matrix T's
    shape and density, ensuring robustness across datasets of varying sizes.

    Calibration targets (from validated results):
      - Fdataset (density~0.55): eta~0.038, beta_max~35.5, ratio~0.99
      - Cdataset (density~0.58): eta~0.039, beta_max~38.4, ratio~0.99
      - DNdataset (T_density~0.34): eta~0.040, beta_max~50.0, ratio~0.99

    Parameters
    ----------
    T_shape : tuple, (n_rows, n_cols)
    density : float, observed entry density of the augmented matrix T

    Returns
    -------
    dict with keys: eta, beta_min, beta_max, warmup, target_rank_ratio
    """
    n_rows, n_cols = T_shape
    n_total = n_rows * n_cols
    min_dim = min(n_rows, n_cols)

    # Ultra-sparse regime safety net
    if density < 0.0005:
        return {
            'eta': 0.015,
            'beta_min': 1.0,
            'beta_max': 18.0,
            'warmup': 5,
            'target_rank_ratio': 0.88,
        }

    # Warmup: scales with log of matrix dimension
    warmup = int(5 + np.log(max(min_dim / 900.0, 1.0)) * 8)
    warmup = min(max(warmup, 5), 20)

    # Density-dependent scaling
    log_density = np.log10(max(density, 1e-6))
    log_ref = -2.0
    delta_log = log_density - log_ref

    # Beta_max: anchored at 25, scales with size and density
    size_factor = np.log10(max(n_total / 8e5, 0.5))
    beta_base = 25.0 + size_factor * 15.0
    beta_scale = np.exp(0.2 * delta_log)
    beta_max = beta_base * beta_scale
    beta_max = float(np.clip(beta_max, 15.0, 50.0))

    # Eta: learning rate calibrated for different densities
    eta = 0.02 * (density / 0.01) ** 0.15 * (min_dim / 500.0) ** 0.1
    eta = float(np.clip(eta, 0.01, 0.08))

    # Target rank ratio: sigmoid of log-density
    log_center = -2.5
    log_width = 0.8
    x = (log_density - log_center) / log_width
    target_rank_ratio = 0.88 + 0.12 / (1.0 + np.exp(-x))
    target_rank_ratio = float(np.clip(target_rank_ratio, 0.85, 1.0))

    return {
        'eta': round(eta, 4),
        'beta_min': 1.0,
        'beta_max': round(beta_max, 1),
        'warmup': warmup,
        'target_rank_ratio': round(target_rank_ratio, 2),
    }


def BNNR_adaptive(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b,
                  eta=0.05, beta_min=1.0, beta_max=50.0,
                  warmup=5, target_rank_ratio=1.0,
                  verbose=0):
    """
    RA-BNNR: Rank-Aware Adaptive Penalty Parameter for BNNR.

    Dynamically adjusts beta based on effective rank tracking to achieve
    better low-rank approximation than fixed-beta BNNR.

    Parameters
    ----------
    alpha, beta : float, BNNR hyperparameters (alpha fixed, beta is initial value)
    T : ndarray, target matrix
    trIndex : ndarray, observation mask (1=observed, 0=unobserved)
    tol1, tol2 : float, convergence thresholds
    maxiter : int, maximum iterations
    a, b : float, value bounds [a, b]
    eta : float, learning rate for beta adjustment
    beta_min, beta_max : float, bounds for beta
    warmup : int, warmup iterations to estimate target rank
    target_rank_ratio : float, target rank = ratio * mean(warmup ranks)
    verbose : int, 0=silent, 1=info, 2=per-adaptation

    Returns
    -------
    W : ndarray, recovered matrix
    iter_num : int, iteration count
    info : dict, diagnostics (beta_history, rank_history, etc.)
    """
    T = np.array(T, dtype=np.float64)
    trIndex = np.array(trIndex, dtype=np.float64)
    if T.shape != trIndex.shape:
        raise ValueError("T and trIndex must have same shape")

    X = T.copy()
    W = T.copy()
    Y = T.copy()

    stop1 = 1.0
    stop2 = 1.0
    iter_num = 0

    beta_cur = float(beta)
    beta_history = [beta_cur]
    rank_history = []
    warmup_ranks = []
    target_rank = None
    n_comp = None

    T_masked = T * trIndex

    while stop1 > tol1 or stop2 > tol2:
        beta_inv = 1.0 / beta_cur
        alpha_over_beta = alpha / beta_cur
        alpha_sum = alpha + beta_cur
        alpha_ratio = alpha / alpha_sum

        # W-update
        tran = beta_inv * Y + alpha_over_beta * T_masked + X
        W = tran - alpha_ratio * (tran * trIndex)
        np.clip(W, a, b, out=W)

        # X-update (SVT)
        Z = W - beta_inv * Y
        X_svt, eff_rank, _ = svt_with_rank(Z, beta_inv, n_components=n_comp)
        rank_history.append(eff_rank)

        n_comp = int(eff_rank * 2.0)
        n_comp = max(n_comp, 10)
        n_comp = min(n_comp, min(T.shape))

        # Y-update
        Y = Y + beta_cur * (X_svt - W)

        # Convergence check
        stop1_prev = stop1
        norm_X = np.linalg.norm(X, 'fro')
        stop1 = np.linalg.norm(X_svt - X, 'fro') / norm_X if norm_X != 0 else 0.0
        stop2 = abs(stop1 - stop1_prev) / max(1.0, abs(stop1_prev))

        X = X_svt
        iter_num += 1

        # Rank-aware beta adaptation
        if iter_num <= warmup:
            warmup_ranks.append(eff_rank)
        else:
            if target_rank is None and warmup_ranks:
                target_rank = target_rank_ratio * np.mean(warmup_ranks)
                if verbose >= 1:
                    print(f"    [rank_aware] target_rank={target_rank:.1f} "
                          f"(warmup_mean={np.mean(warmup_ranks):.1f}, "
                          f"ratio={target_rank_ratio})")

            if target_rank is not None and target_rank > 0:
                rank_ratio = (eff_rank - target_rank) / target_rank
                beta_new = beta_cur * np.exp(-eta * rank_ratio)
                beta_cur = np.clip(beta_new, beta_min, beta_max)

        beta_history.append(beta_cur)

        if stop1 <= tol1 and stop2 <= tol2:
            if verbose >= 1:
                print(f"    [converged] iter={iter_num}, "
                      f"beta_final={beta_cur:.3f}, rank={eff_rank}")
            break

        if iter_num >= maxiter:
            iter_num = maxiter
            warnings.warn(f"[BNNR_adaptive] max iter reached "
                          f"(beta={beta_cur:.2f}, rank={eff_rank})")
            break

    info = {
        'beta_final': float(beta_cur),
        'beta_history': np.array(beta_history),
        'rank_history': np.array(rank_history),
        'target_rank': target_rank,
        'converged': stop1 <= tol1 and stop2 <= tol2,
    }

    return W, iter_num, info
