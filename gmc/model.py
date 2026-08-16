"""GMC — Graph multi-view low-rank Completion (the final proposed method).

Under CVa (random-entry masking = matrix completion) the winning paradigm is
global low-rank completion, not label propagation. GMC fuses that completion
capacity with graph machinery:

  1. WKNN soft-label fill (multiGMF WKNN.m port), cold-start restricted to
     entities with NO observed associations (OMC-style).
  2. View A — block nuclear-norm completion (BNNR/SVT) on the joint
     bipartite+similarity block [[Wdd, F], [Fᵀ, Wrr]]: the similarity
     structure regularizes the low-rank projection of the fill.
  3. View C — tensor-RPCA completion (ITRPCA) over the per-similarity
     tensors (5 drug + 2 disease slices kept separate).
  4. Fusion: raw weighted sum ("raw", filter-safe) or rank-normalized
     per-view averaging ("rank", scale-free — best for tensor+block).
  5. KNN-sparsified bilateral graph Laplacian low-pass filter (raw fusion
     only): (I + αL_dis)⁻¹ · M · (I + αL_drug)⁻¹, blended by β.
  6. Observed-entry re-imposition.

All solvers are re-implemented in Python in ``gmc/factorization.py``.
"""
import numpy as np

from .factorization import (graph_reg_nmf, bounded_nn_completion,
                            bounded_nn_completion_graph, fitrpca)
from .wknn import wknn
from .filter import sparsify_graph, normalised_laplacian, graph_filter


def rnorm01(M):
    """Scale-free [0,1] normalization by min-max rescaling (order-preserving).

    Maps each view's range onto [0,1] so views that live on different scales
    (block completion vs tensor RPCA) can be combined by a fixed weighted
    average. The map is strictly monotone, so it preserves each view's ranking
    without introducing ties. NOTE: the fused scores must NOT be run through
    the bilateral graph filter afterwards — post-hoc smoothing perturbs the
    ordering established here; the unified config (``fusion="rank"``) returns
    the fused scores directly without filtering.
    """
    lo, hi = M.min(), M.max()
    return (M - lo) / (hi - lo) if hi > lo else np.zeros_like(M)


def coldstart_fill(masked, soft):
    """Restrict the WKNN soft-fill to entities with NO observed associations.

    Parameters
    ----------
    masked : ndarray (n_dis, n_drug)
        Masked training association matrix.
    soft : ndarray (n_dis, n_drug)
        Full WKNN soft-label fill.

    Returns
    -------
    filled : ndarray (n_dis, n_drug)
        ``soft`` on all-zero rows/cols, ``masked`` elsewhere. Partially
        observed entities keep their sparse zeros so the completion / filter
        propagates from true labels only (OMC-style cold-start fill).
    """
    rows_cold = masked.sum(axis=1) == 0
    cols_cold = masked.sum(axis=0) == 0
    keep = rows_cold[:, None] | cols_cold[None, :]
    return np.where(keep, soft, masked)


def gmc_predict(masked, Wrr, Wdd, drug_sims, dis_sims,
                bnnr_alpha=0.5, bnnr_maxiter=40, bnnr_rank_cap=160,
                bnnr_beta=10.0, wknn_k=30, wknn_r=0.95,
                grnmf_rank=None, grnmf_maxiter=120, w_bnnr=1.0,
                w_graph=0.0, graph_alpha=0.7,
                w_grnmf=0.0, w_tensor=0.0, tensor_rank_cap=None,
                fusion="raw", filt_alpha=0.0, filt_beta=0.5,
                sparsify_k=0, coldstart=False, seed=1,
                trindex="all", fill="wknn", block="sym"):
    """Full GMC prediction for one fold. Returns (n_dis, n_drug) scores.

    sparsify_k>0   : KNN-sparsify the similarity blocks before completion.
    coldstart      : restrict the WKNN fill to all-zero rows/cols instead of
                     the full fill.
    w_graph>0      : View A uses the SGRMC-style graph-embedded completion
                     (bounded_nn_completion_graph): graph Laplacian smoothness
                     is embedded in the ADMM instead of a post-hoc filter.
                     The block view's fusion weight stays w_bnnr; graph_alpha
                     sets the embedding strength (γ).
    w_tensor>0     : add the tensor-RPCA view (ITRPCA mechanism) built from
                     the raw per-similarity tensors; fused with the block view.
    fusion         : "raw"  → weighted sum on raw ~[0,1] values (filter-safe);
                     "rank" → weighted sum after rnorm01 per view (scale-free,
                             best for tensor+block on CTD/Y; do NOT filter).
    tensor_rank_cap: randomized-SVD cap for the tensor t-SVD (200 = good
                     speed/quality tradeoff on CTD/Y; None = full SVD).
    trindex        : observation mask for the BNNR block completion:
                     "all"      → trI = ones (every entry treated as observed;
                                  WKNN soft-fill and zeros are hard-constrained)
                     "observed" → trI = (T != 0) (only non-zero entries are
                                  constrained; zeros are left free for the
                                  low-rank prior to fill).  On large/sparse
                                  datasets (Y) "observed" reduces over-fitting
                                  to the fill and improves AUPR (9/10 folds,
                                  p≈0.006); on F/C/CTD it is neutral-to-harmful,
                                  so keep "all" there.
    fill           : "wknn" (default) — WKNN soft-label fill, cold-start
                     restricted; "knn" — OMC-style KNN cold-start fill that
                     fills only all-zero rows/cols with similarity-weighted
                     neighbor averages (Y: lifts AUPR with the dual block +
                     observed mask; F/C keep "wknn"); "none" — no cold-start
                     fill (the masked matrix itself goes into the block), used
                     only as an ablation control to isolate the fill's
                     contribution to the block completion.
    block          : "sym" (default) — single symmetric block
                     [[Wdd,F],[Fᵀ,Wrr]]; "dual" — OMC-style dual asymmetric
                     blocks [Wrr;F] and [F,Wdd] completed separately then
                     averaged (Y: the winning View-A structure; F/C/CTD keep
                     "sym").
    """
    dn, dr = masked.shape
    if fill == "knn":
        from .knnfill import knn_fill_cold
        filled = knn_fill_cold(masked, Wdd, Wrr, K=wknn_k)
        coldstart = False
    elif fill == "none":
        filled = masked.copy()
    else:
        soft = wknn(masked.T, Wrr, Wdd, k=wknn_k, r=wknn_r).T
        if coldstart:
            filled = coldstart_fill(masked, soft)
        else:
            filled = soft

    if sparsify_k > 0:
        Wdd_b, Wrr_b = sparsify_graph(Wdd, sparsify_k), sparsify_graph(Wrr, sparsify_k)
    else:
        Wdd_b, Wrr_b = Wdd, Wrr

    # ── View A: block completion (plain, or SGRMC-style graph-embedded) ──
    if block == "dual" and w_graph == 0:
        # OMC-style dual asymmetric block: complete [Wrr; F] and [F, Wdd]
        # separately (each block constrains F against ONE similarity side),
        # then average the two association-block estimates.  trI is per-block
        # and the winning Y recipe uses the observed (non-zero) mask.
        T1 = np.vstack([Wrr_b, filled])          # (dr+dn, dr)
        tr1 = np.ones_like(T1) if trindex == "all" else (T1 != 0).astype(np.float64)
        W1 = bounded_nn_completion(T1, tr1, alpha=bnnr_alpha, beta=bnnr_beta,
                                   maxiter=bnnr_maxiter, rank_cap=bnnr_rank_cap)
        M1 = np.clip(W1[-dn:, :], 0, 1)          # bottom (dn, dr)
        T2 = np.hstack([filled, Wdd_b])          # (dn, dr+dn)
        tr2 = np.ones_like(T2) if trindex == "all" else (T2 != 0).astype(np.float64)
        W2 = bounded_nn_completion(T2, tr2, alpha=bnnr_alpha, beta=bnnr_beta,
                                   maxiter=bnnr_maxiter, rank_cap=bnnr_rank_cap)
        M2 = np.clip(W2[:, :dr], 0, 1)           # left (dn, dr)
        M_bnnr = 0.5 * (M1 + M2)
    else:
        T = np.block([[Wdd_b, filled], [filled.T, Wrr_b]])  # (dn+dr, dn+dr)
        if trindex == "observed":
            trI = (T != 0).astype(np.float64)
        else:
            trI = np.ones_like(T)
        if w_graph > 0:
            # SGRMC mechanism: graph Laplacian smoothness embedded in the ADMM,
            # so the completion itself is graph-aware (not a post-hoc filter).
            Ld = normalised_laplacian(sparsify_graph(Wdd, 5))
            Lr = normalised_laplacian(sparsify_graph(Wrr, 5))
            Wc = bounded_nn_completion_graph(T, trI, Ld, Lr, dn, dr,
                                             graph_alpha=graph_alpha,
                                             alpha=bnnr_alpha, beta=bnnr_beta,
                                             maxiter=bnnr_maxiter,
                                             rank_cap=bnnr_rank_cap)
        else:
            Wc = bounded_nn_completion(T, trI, alpha=bnnr_alpha, beta=bnnr_beta,
                                       maxiter=bnnr_maxiter, rank_cap=bnnr_rank_cap)
        M_bnnr = np.clip(Wc[0:dn, dn:dn + dr], 0, 1)   # top-right block, (dn, dr)

    # ── View B: graph-regularized NMF on the fill ──
    M_grnmf = np.zeros((dn, dr))
    if w_grnmf > 0:
        rank = grnmf_rank or int(np.floor(min(dn, dr) * 0.7))
        M_grnmf = graph_reg_nmf(filled, drug_sims, dis_sims, rank, seed=seed,
                                maxiter=grnmf_maxiter)

    # ── View C: tensor-RPCA completion (ITRPCA mechanism) ──
    # Built from the raw per-similarity tensors; the tensor does its own
    # WKNN fill (K=30,r=0.95) internally, so it takes the masked matrix.
    M_tensor = np.zeros((dn, dr))
    if w_tensor > 0:
        drn, ddn = len(drug_sims), len(dis_sims)
        Trr = np.zeros((dr, dr, drn)); Tdd = np.zeros((dn, dn, ddn))
        for i, s in enumerate(drug_sims):
            Trr[:, :, i] = s
        for i, s in enumerate(dis_sims):
            Tdd[:, :, i] = s
        M_tensor = fitrpca(Trr, Tdd, masked, p=0.9, K=30, rat1=0.1, rat2=0.2,
                           rank_cap=tensor_rank_cap)

    # ── fusion ──
    if fusion == "rank":
        # scale-free: rescale each view to [0,1] (order-preserving), then weight-sum.
        F = (w_bnnr * rnorm01(M_bnnr) + w_grnmf * rnorm01(M_grnmf)
             + w_tensor * rnorm01(M_tensor))
        Z = (w_bnnr + w_grnmf + w_tensor) or 1.0
        F = F / Z
        # rank-fused values must NOT be graph-filtered (rank ties break the
        # ordering); return directly with observed-entry re-imposition.
        return np.where(masked != 0, masked, np.clip(F, 0, 1))

    # raw-value fusion (views live on a common ~[0,1] scale)
    F = w_bnnr * M_bnnr + w_grnmf * M_grnmf + w_tensor * M_tensor

    # ── bilateral spectral post-smoothing, blended ──
    if filt_alpha > 0:
        Ld = normalised_laplacian(sparsify_graph(Wdd, 5))
        Lr = normalised_laplacian(sparsify_graph(Wrr, 5))
        G = graph_filter(F, Ld, Lr, filt_alpha)
        F = filt_beta * G + (1 - filt_beta) * F

    # ── observed-entry re-imposition ──
    return np.where(masked != 0, masked, np.clip(F, 0, 1))
