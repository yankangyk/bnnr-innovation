"""Low-rank factorization cores ported from the winning CVa baselines.

These are clean Python re-implementations of the mechanisms that win under
CVa (random entry masking = matrix completion), used as building blocks for
the GMC model:

  graph_reg_nmf          — multiplicative NMF with graph-Laplacian regularized
                           factors (port of multiGMF fmultiGMF.m). The graph
                           smoothing acts as (μλw·I + λ1·L)⁻¹ on each factor,
                           exactly a low-pass filter on the factor manifold.
  bounded_nn_completion  — BNNR/SVT block nuclear-norm completion (View A).
  fitrpca                — ITRPCA tensor-RPCA completion (View C).
  deep_semi_nmf          — stacked semi-NMF with graph-Laplacian regularization
                           on the outer factor (port of DNMFDDA fDRDMF.m).

The GMC bilateral graph filter (gmc/filter.py) can be applied before/after
any of these.

Key optimization over the MATLAB originals: the graph matrices
(μλw·I + λ1·L) are constant across iterations, so their Cholesky/LU factors
are computed ONCE and reused (cho_solve per iteration instead of inv()).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .wknn import wknn as _wknn

EPS = 1e-10


# ────────────────────────────────────────────────────────────────────────
# multiGMF — graph-regularized multiplicative NMF
# ────────────────────────────────────────────────────────────────────────
def graph_reg_nmf(A_obs, drug_sims, dis_sims, rank, maxiter=300,
                  lambda_soft=1.0, lambda1=1e-4, lambda2=1e-4, lambda3=1.0,
                  mu1=1.0, mu2=1.0, tol1=2e-3, tol2=1e-4, seed=1,
                  max_rank=None):
    """Graph-regularized NMF of the WKNN-filled matrix.

    A_obs      : (n_dis, n_drug) masked/WKNN-filled association matrix (dn×dr)
    drug_sims  : list of (dr, dr) drug similarity matrices
    dis_sims   : list of (dn, dn) disease similarity matrices
    rank       : target latent rank
    Returns    : (n_dis, n_drug) predicted association scores
                 (equivalently H·Wᵀ where A ≈ W·Hᵀ, H·Wᵀ = (dn, dr)).
    """
    dn, dr = A_obs.shape
    X = A_obs.T.copy()                     # (dr, dn)
    rng = np.random.RandomState(seed)
    H = rng.rand(dn, rank)                 # (dn, rank)
    W = rng.rand(dr, rank)                 # (dr, rank)
    nG_r, nG_d = len(drug_sims), len(dis_sims)
    Rw = np.ones(nG_r) / nG_r
    Dw = np.ones(nG_d) / nG_d

    # Graph matrices are iteration-invariant → pre-factorize once.
    Lr_cf = []
    for k, A in enumerate(drug_sims):
        L = np.diag(A.sum(1)) - A
        mat = mu1 * lambda_soft * Rw[k] * np.eye(dr) + lambda1 * L
        Lr_cf.append(cho_factor(mat, lower=True))
    Ld_cf = []
    for k, B in enumerate(dis_sims):
        L = np.diag(B.sum(1)) - B
        mat = mu2 * lambda_soft * Dw[k] * np.eye(dn) + lambda2 * L
        Ld_cf.append(cho_factor(mat, lower=True))

    WH = X.copy()
    stop1, stop2 = 1.0, 1.0
    for _ in range(maxiter):
        # graph-smoothed factor auxiliaries:  (μλw·I + λ1·L)⁻¹ (μ·w·W)
        Sr = np.zeros_like(W)
        for k in range(nG_r):
            Sr += Rw[k] * cho_solve(Lr_cf[k], mu1 * Rw[k] * W)
        Sd = np.zeros_like(H)
        for k in range(nG_d):
            Sd += Dw[k] * cho_solve(Ld_cf[k], mu2 * Dw[k] * H)

        Sw = sum(Rw[k] * W for k in range(nG_r))      # == W (weights sum to 1)
        Sh = sum(Dw[k] * H for k in range(nG_d))      # == H

        num_w = X @ H + mu1 * lambda_soft * Sr
        den_w = W @ (H.T @ H) + mu1 * lambda_soft * Sw + lambda3 * W
        W = W * (num_w / np.maximum(den_w, EPS)) + 1e-8

        num_h = X.T @ W + mu2 * lambda_soft * Sd
        den_h = H @ (W.T @ W) + mu2 * lambda_soft * Sh + lambda3 * H
        H = H * (num_h / np.maximum(den_h, EPS)) + 1e-8

        stop1_0 = stop1
        stop1 = np.linalg.norm(W @ H.T - WH, "fro") / np.linalg.norm(WH, "fro")
        stop2 = abs(stop1 - stop1_0) / max(1.0, abs(stop1_0))
        WH = W @ H.T
        if stop1 < tol1 and stop2 < tol2:
            break

    return np.clip(H @ W.T, 0, 1)          # (dn, dr) disease-by-drug


# ────────────────────────────────────────────────────────────────────────
# BNNR — bounded nuclear-norm completion (ADMM + singular-value soft-threshold)
# ────────────────────────────────────────────────────────────────────────
def _svt(Z, tau, rank_cap=None):
    """Singular-value soft-thresholding (proximal of nuclear norm).

    Uses randomized SVD when rank_cap is set (fast on large block matrices,
    which are effectively low-rank).
    """
    if rank_cap and min(Z.shape) > rank_cap * 2:
        try:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vt = randomized_svd(Z, n_components=min(rank_cap, *Z.shape),
                                      n_iter=3, random_state=0)
            s = np.maximum(s - tau, 0)
            return (U * s) @ Vt
        except ImportError:
            pass
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    s = np.maximum(s - tau, 0)
    return (U * s) @ Vt


def bounded_nn_completion(T, trIndex, alpha=1.0, beta=10.0,
                          tol1=2e-3, maxiter=300, a=0.0, b=1.0,
                          rank_cap=None):
    """Bounded nuclear-norm regularized completion (port of BNNR.m).

    T        : (n, n) block matrix, observed entries set, unobserved = 0
    trIndex  : 1 where entry is observed, 0 elsewhere
    rank_cap : randomized-SVD rank cap for large matrices (fast approx.)
    Returns  : completed (n, n) block matrix.
    """
    X = T.copy()
    W = X.copy()
    Y = X.copy()
    stop1, stop2 = 1.0, 1.0
    i = 0
    while (stop1 > tol1 or stop2 > 2e-5) and i < maxiter:
        tran = (1 / beta) * (Y + alpha * (T * trIndex)) + X
        W = tran - (alpha / (alpha + beta)) * (tran * trIndex)
        W = np.clip(W, a, b)
        X_1 = _svt(W - (1 / beta) * Y, 1 / beta, rank_cap)
        Y = Y + beta * (X_1 - W)
        stop1_0 = stop1
        stop1 = np.linalg.norm(X_1 - X, "fro") / np.linalg.norm(X, "fro")
        stop2 = abs(stop1 - stop1_0) / max(1.0, abs(stop1_0))
        X = X_1
        i += 1
    return W


def bounded_nn_completion_graph(T, trIndex, L_dis, L_drug, n_dis, n_drug,
                                graph_alpha=0.7, alpha=1.0, beta=10.0,
                                tol1=2e-3, maxiter=300, a=0.0, b=1.0,
                                rank_cap=None):
    """Graph-regularized block completion (SGRMC ``BNNR_graph_aware`` port).

    Same ADMM as ``bounded_nn_completion`` but with graph Laplacian
    smoothness EMBEDDED in the objective instead of applied as a post-hoc
    filter:

        min ‖X‖_* + α/2·‖P_Ω(X−T)‖² + γ·tr(MᵀL_dis M + M L_drug Mᵀ)

    where M = T[0:n_dis, n_dis:n_dis+n_drug] is the association block. Each
    W-update applies a first-order Neumann smoothing step to the association
    block (M ← M − γ_iter·(L_dis @ M + M @ L_drug), with
    γ_iter = graph_alpha · β⁻¹ · 0.1) before the singular-value threshold, so
    the smoothing and the low-rank projection are interleaved and converge to
    a fixed point that is jointly low-rank and graph-smooth. graph_alpha=0
    disables the graph term and the solver reduces exactly to
    ``bounded_nn_completion``.

    T      : (n, n) block matrix in the GMC diseases-first layout
             [[Wdd, F], [Fᵀ, Wrr]]; the association block is the top-right
             block T[0:n_dis, n_dis:n_dis+n_drug].
    L_dis  : (n_dis, n_dis) normalised disease Laplacian (knn-sparsified).
    L_drug : (n_drug, n_drug) normalised drug Laplacian (knn-sparsified).
    n_dis, n_drug : dimensions of the association block (T is square).
    Returns the completed (n, n) block matrix.
    """
    X = T.copy()
    W = X.copy()
    Y = X.copy()
    stop1, stop2 = 1.0, 1.0
    i = 0
    gamma_iter = graph_alpha * (1.0 / beta) * 0.1
    r0, c0 = 0, n_dis
    r1, c1 = n_dis, n_dis + n_drug
    while (stop1 > tol1 or stop2 > 2e-5) and i < maxiter:
        tran = (1 / beta) * (Y + alpha * (T * trIndex)) + X
        W = tran - (alpha / (alpha + beta)) * (tran * trIndex)
        if gamma_iter != 0.0:
            # ── embedded graph smoothing (first-order Neumann approx.) ──
            M_block = W[r0:r1, c0:c1]
            M_sm = M_block - gamma_iter * (L_dis @ M_block + M_block @ L_drug)
            W[r0:r1, c0:c1] = M_sm
            W[c0:c1, r0:r1] = M_sm.T
        W = np.clip(W, a, b)
        X_1 = _svt(W - (1 / beta) * Y, 1 / beta, rank_cap)
        Y = Y + beta * (X_1 - W)
        stop1_0 = stop1
        stop1 = np.linalg.norm(X_1 - X, "fro") / np.linalg.norm(X, "fro")
        stop2 = abs(stop1 - stop1_0) / max(1.0, abs(stop1_0))
        X = X_1
        i += 1
    return W


# ────────────────────────────────────────────────────────────────────────
# ITRPCA — tensor robust PCA (FFT t-SVD, weighted Schatten-p prox)
# ────────────────────────────────────────────────────────────────────────
def solve_Lp_w(y, lam, p, J=4):
    """Generalized Lp-norm thresholding (port of solve_Lp_w.m).

    y   : singular values
    lam : threshold vector (weights/mu), same length as y
    """
    lam = np.broadcast_to(np.asarray(lam, dtype=float), np.shape(y))
    y = np.asarray(y, dtype=float)
    tau = (2 * lam * (1 - p)) ** (1 / (2 - p)) + \
        p * lam * (2 * (1 - p) * lam) ** ((p - 1) / (2 - p))
    x = np.zeros_like(y)
    i0 = np.abs(y) > tau
    if i0.any():
        y0 = y[i0]
        t = np.abs(y0)
        lam0 = lam[i0]
        for _ in range(J):
            t = np.abs(y0) - p * lam0 * t ** (p - 1)
        x[i0] = np.sign(y0) * t
    return x


def ffindw(A, ratio):
    """Small k such that top-k singular values exceed `ratio` of total energy."""
    s = np.linalg.svd(A, compute_uv=False)
    total = s.sum()
    for i in range(1, len(s) + 1):
        if s[:i].sum() > ratio * total:
            return i
    return len(s)


def _prox_tnn(Y, rho, p, rank_cap=None):
    """Tensor nuclear-norm proximal via FFT t-SVD (port of prox_tnn.m).

    Y        : (n1, n2, n3) tensor
    rho      : (n3,) per-slice threshold (weight/mu)
    rank_cap : randomized-SVD rank cap for speed (approx; full SVD if None)
    Returns (n1, n2, n3) low-rank tensor.
    """
    n1, n2, n3 = Y.shape
    Yf = np.fft.fft(Y, axis=2)
    Xf = np.zeros_like(Yf)
    for i in range(n3):
        Ys = Yf[:, :, i]
        if rank_cap and min(Ys.shape) > rank_cap * 2:
            Uu, s, Vvt = _rsvd_complex(Ys, rank_cap)
        else:
            Uu, s, Vvt = np.linalg.svd(Ys, full_matrices=False)
        s = solve_Lp_w(s, rho[i], p)
        Xf[:, :, i] = (Uu * s) @ Vvt
    return np.real(np.fft.ifft(Xf, axis=2))


def _rsvd_complex(A, k):
    """Randomized SVD for (possibly complex) A ≈ U s Vᵀ.  U (m,k), s (k,), Vt (k,n)."""
    m, n = A.shape
    p = min(k + 10, n)
    rng = np.random.RandomState(0)
    Omega = rng.randn(n, p) + 1j * rng.randn(n, p)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.conj().T @ A
    Uu, s, Vvt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uu
    kk = min(k, len(s))
    return U[:, :kk], s[:kk], Vvt[:kk]


def itrpca_tnn_lp(X, lam, weight, p, maxiter=5, rho_=1.1, mu0=1e-2,
                  rank_cap=None):
    """Tensor robust PCA via ALM (port of itrpca_tnn_lp_stop.m).

    X      : (n1, n2, n3) tensor
    lam    : l1 sparsity weight scalar
    weight : (n3,) per-slice Schatten weights
    Returns: low-rank tensor L (bounded to [0, 255]).
    """
    dim = X.shape
    n3 = dim[2]
    L = np.zeros(dim)
    S = np.zeros(dim)
    Y = np.zeros(dim)
    mu = mu0
    X_0 = X.copy()
    stop1 = 1.0
    for _ in range(maxiter):
        # update L
        L = _prox_tnn(-S + X - Y / mu, weight / mu, p, rank_cap)
        L = np.clip(L, 0, 255)
        # update S
        S = np.maximum(0, -L + X - Y / mu - lam / mu) + \
            np.minimum(0, -L + X - Y / mu + lam / mu)
        dY = L + S - X
        X_1 = L
        stop1_0 = stop1
        sum_norm = 0.0
        for j in range(n3):
            sum_norm += (np.linalg.norm(X_1[:, :, j] - X_0[:, :, j], "fro") /
                         np.linalg.norm(X_0[:, :, j], "fro"))
        stop1 = sum_norm
        stop2 = abs(stop1 - stop1_0) / max(1.0, abs(stop1_0))
        X_0 = X_1
        Y = Y + mu * dY
        mu = min(rho_ * mu, 1e10)
        if stop1 < 1e-3 and stop2 < 1e-4:
            break
    return L


def fitrpca(Trr, Tdd, P_TMat, p=0.9, K=30, rat1=0.1, rat2=0.2, rank_cap=None):
    """Incomplete tensor RPCA for DDA (port of fITRPCA.m).

    Trr    : (dr, dr, n_drug_sim) drug similarity tensor
    Tdd    : (dn, dn, n_dis_sim) disease similarity tensor
    P_TMat : (dn, dr) masked association matrix
    rank_cap : randomized-SVD cap for the tensor prox (speed)
    Returns: (dn, dr) prediction.
    """
    dn, dr = P_TMat.shape
    dr_num = Trr.shape[2]
    dd_num = Tdd.shape[2]
    Wrr = Trr.mean(axis=2)
    Wdd = Tdd.mean(axis=2)
    # WKNN fill (multiGMF WKNN, k=K, r=0.95)
    P_new = _wknn(P_TMat.T, Wrr, Wdd, k=K, r=0.95).T

    # ── drug tensor ──
    Tdr = np.tile(P_new[:, :, None], (1, 1, dr_num))       # (dn, dr, dr_num)
    R_ori = np.concatenate([Trr, Tdr], axis=0) * 255.0     # (dn+dr, dr, dr_num)
    n1, n2, n3 = R_ori.shape
    n = min(n1, n2)
    a1 = int(round(np.mean([ffindw(R_ori[:, :, e], rat1) for e in range(dr_num)])))
    a2 = -a1 + 2 + int(round(np.mean([ffindw(R_ori[:, :, e], rat2) for e in range(dr_num)])))
    w = np.concatenate([np.ones(a1), 2 * np.ones(a2), 4 * np.ones(n - a1 - a2)])
    kao = 1 / (5 * np.sqrt(n1 * n2 * n3))
    R_res = itrpca_tnn_lp(R_ori, kao, w, p, rank_cap=rank_cap) / 255.0
    R_Result = R_res[n1 - dn:n1, :dr, :].mean(axis=2)      # bottom (dn, dr), avg slices

    # ── disease tensor ──
    Tdr2 = np.tile(P_new[:, :, None], (1, 1, dd_num))      # (dn, dr, dd_num)
    D_ori = np.concatenate([Tdr2, Tdd], axis=1) * 255.0    # (dn, dr+dn, dd_num)
    nn1, nn2, nn3 = D_ori.shape
    nn = min(nn1, nn2)
    b1 = int(round(np.mean([ffindw(D_ori[:, :, e], rat1) for e in range(dd_num)])))
    b2 = -b1 + 2 + int(round(np.mean([ffindw(D_ori[:, :, e], rat2) for e in range(dd_num)])))
    w2 = np.concatenate([np.ones(b1), 2 * np.ones(b2), 4 * np.ones(nn - b1 - b2)])
    kao2 = 1 / (5 * np.sqrt(nn1 * nn2 * nn3))
    D_res = itrpca_tnn_lp(D_ori, kao2, w2, p, rank_cap=rank_cap) / 255.0
    D_Result = D_res[:dn, :dr, :].mean(axis=2)             # first (dn, dr), avg slices

    return np.clip(0.5 * (R_Result + D_Result), 0, 1)


# ────────────────────────────────────────────────────────────────────────
# DNMFDDA — deep semi-NMF (2 layers) with graph regularization
# ────────────────────────────────────────────────────────────────────────
def _seminmf_init(X, k):
    """Standard semi-NMF init (SVD-based): X ≈ W·Z, W signed, Z ≥ 0."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    W = U[:, :k] * np.sqrt(s[:k])
    Z = (Vt[:k].T * np.sqrt(s[:k])).T
    Z[Z < 0] = 0.0
    return W, Z


def _seminmf_solve(X, W0, Z0, maxiter=100):
    """Alternating semi-NMF: min |X - WZ|², Z≥0, W free.  X≈WZ, W (n,k), Z (k,m)."""
    W, Z = W0.copy(), Z0.copy()
    k = Z.shape[0]
    ZZt = Z @ Z.T
    for _ in range(maxiter):
        # W update (unconstrained least squares): W = (X Zᵀ)(Z Zᵀ)⁻¹
        W = np.linalg.solve(ZZt + 1e-9 * np.eye(k), (X @ Z.T).T).T
        # Z update (nonnegative multiplicative)
        num = W.T @ X
        den = W.T @ W @ Z
        Z = Z * np.sqrt(np.maximum(num, 0) / np.maximum(den, EPS))
    return W, Z


def _split(M):
    """Positive / negative parts of a (possibly signed) matrix."""
    Mp = (np.abs(M) + M) / 2
    Mn = (np.abs(M) - M) / 2
    return Mp, Mn


def deep_semi_nmf(X, layers, L_W, alpha=0.01, beta=1.0, lammad=1.0,
                  gamma=1.0, maxiter=400, tol1=5e-4, tol2=5e-7):
    """Deep (2-layer) semi-NMF of a view with graph Laplacian regularization
    on the factor matrices — port of DNMFDDA's fDRDMF for one view.

    X      : (n, m) view matrix (e.g. drug side [Wrr, A']).
    layers : [l1, l2] hidden dims.
    L_W    : (n, n) graph Laplacian on the outer factor (rows of X).
    Returns: (n, m) reconstruction.
    """
    n, m = X.shape
    m_layers = len(layers)

    # ── init: layer-by-layer semi-NMF ──
    W = [None] * m_layers
    Z = [None] * m_layers
    fac = X
    for j in range(m_layers):
        Wj, Zj = _seminmf_init(fac, layers[j])
        Wj, Zj = _seminmf_solve(fac, Wj, Zj, maxiter=60)
        W[j], Z[j] = Wj, Zj
        fac = Wj

    WH = X.copy()
    stop1, stop2 = 1.0, 1.0
    I2 = np.eye(layers[0])

    for _ in range(maxiter):
        for i in range(m_layers):
            # ── Z update (Zupdate.m) ──
            if i == 0:
                W_X = W[i].T @ X
                WX_p, WX_n = _split(W_X)
                WW_p, WW_n = _split(W[i].T @ W[i] @ Z[i])
                Zp, Zn = _split(Z[i])
                Z[i] = Z[i] * np.sqrt(
                    (WW_n + WX_p + beta * Zn) / np.maximum(WW_p + WX_n + beta * Zp, EPS))
            else:
                # i == 1: phi = Z[0]
                phi = Z[i - 1]
                if i - 1 >= 1:
                    for jp in range(i - 2, -1, -1):
                        phi = phi @ Z[jp]
                phi_phit = phi @ phi.T
                W_Z_phi = W[i].T @ W[i] @ Z[i] @ phi_phit
                WZ_p, WZ_n = _split(W_Z_phi)
                W_X_phi = W[i].T @ X @ phi.T
                WXP_p, WXP_n = _split(W_X_phi)
                W_W = W[i].T @ W[i - 1]
                WW_p, WW_n = _split(W_W)
                W_W_Z = W[i].T @ W[i] @ Z[i]
                WWZ_p, WWZ_n = _split(W_W_Z)
                Zp, Zn = _split(Z[i])
                Z[i] = Z[i] * np.sqrt(
                    (WZ_n + WXP_p + beta * Zn + gamma * WW_p + gamma * WWZ_n) /
                    np.maximum(WZ_p + WXP_n + beta * Zp + gamma * WW_n + gamma * WWZ_p, EPS))

            # ── W update (Wupdate.m) ──
            phi = np.eye(layers[i]) if i >= 1 else np.eye(n)
            X_phi = np.zeros_like(W[i])
            W_phi_phi = np.zeros_like(W[i])
            # accumulate over the single view (v=1)
            if i == 0:
                phi2 = Z[i]
                X_phi = X @ phi2.T
                W_phi_phi = W[i] @ phi2 @ phi2.T
            else:
                phi2 = Z[i]
                for jp in range(i - 1, -1, -1):
                    phi2 = phi2 @ Z[jp]
                X_phi = X @ phi2.T
                W_phi_phi = W[i] @ phi2 @ phi2.T

            Xp, Xn = _split(X_phi)
            Wpp, Wpn = _split(W_phi_phi)
            Wp, Wn = _split(W[i])
            GrL = L_W @ W[i]
            Gr_p, Gr_n = _split(GrL)

            if i == 0:
                # middle layer (i=1, m=2)
                W_Z = W[i + 1] @ Z[i + 1]
                WZ_p, WZ_n = _split(W_Z)
                num = (Xp + Wpn + alpha * Gr_n + (lammad + gamma) * Wn + gamma * WZ_p)
                den = (Xn + Wpp + alpha * Gr_p + (lammad + gamma) * Wp + gamma * WZ_n)
            else:
                # outermost layer (i==1==m)
                W_Z_Z = W[i] @ Z[i] @ Z[i].T
                WZZ_p, WZZ_n = _split(W_Z_Z)
                W_Z_t = W[i - 1] @ Z[i].T
                WZt_p, WZt_n = _split(W_Z_t)
                num = (Xp + Wpn + alpha * Gr_n + lammad * Wn
                       + gamma * WZt_p + gamma * WZZ_n)
                den = (Xn + Wpp + alpha * Gr_p + lammad * Wp
                       + gamma * WZt_n + gamma * WZZ_p)
            W[i] = W[i] * np.sqrt(num / np.maximum(den, EPS))

        # ── reconstruct + stopping ──
        Z_all = Z[-1]
        for j in range(m_layers - 2, -1, -1):
            Z_all = Z_all @ Z[j]
        M = W[-1] @ Z_all
        stop1_0 = stop1
        stop1 = np.linalg.norm(M - WH, "fro") / np.linalg.norm(WH, "fro")
        stop2 = abs(stop1 - stop1_0) / max(1.0, abs(stop1_0))
        WH = M
        if stop1 < tol1 and stop2 < tol2:
            break
    return M


def deep_semi_nmf_views(A_obs, Wrr, Wdd, drug_sims, dis_sims, K=10,
                        maxiter=100, alpha=0.01, beta=1.0, lammad=1.0,
                        gamma=1.0, seed=1):
    """DNMFDDA-style two-view deep semi-NMF of the association matrix.

    A_obs : (dn, dr) masked / KNN-cold-start-filled association matrix.
    Builds the drug-side view [R̄, A'] and disease-side view [D̄, A], runs
    deep_semi_nmf on each, extracts and averages the association blocks.
    Returns (dn, dr) prediction.
    """
    dn, dr = A_obs.shape
    Wrr_m = np.mean(drug_sims, axis=0)
    Wdd_m = np.mean(dis_sims, axis=0)
    L_r = np.diag(Wrr_m.sum(1)) - Wrr_m
    L_d = np.diag(Wdd_m.sum(1)) - Wdd_m

    # drug-side view
    X_drug = np.hstack([Wrr_m, A_obs.T])                 # (dr, dr+dn)
    R_min = min(X_drug.shape)
    R_layers = [int(R_min * 0.8), int(R_min * 0.6)]
    M_drug = deep_semi_nmf(X_drug, R_layers, L_r, alpha=alpha, beta=beta,
                           lammad=lammad, gamma=gamma, maxiter=maxiter)
    M_drug_A = M_drug[:, dr:].T                          # (dn, dr)

    # disease-side view
    X_dis = np.hstack([Wdd_m, A_obs])                    # (dn, dn+dr)
    D_min = min(X_dis.shape)
    D_layers = [int(D_min * 0.8), int(D_min * 0.6)]
    M_dis = deep_semi_nmf(X_dis, D_layers, L_d, alpha=alpha, beta=beta,
                          lammad=lammad, gamma=gamma, maxiter=maxiter)
    M_dis_A = M_dis[:, dn:]                              # (dn, dr)

    return np.clip(0.5 * (M_drug_A + M_dis_A), 0, 1)
