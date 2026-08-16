"""Ranking-objective completion solvers (AUPR/AUC-surrogate).

Under CVa the metric is AUPR, which is a *ranking* metric: the harness scores
test entries by their ORDER (rank-transformed), not by reconstruction error.
Every baseline (BNNR/SVT, ITRPCA, NMF, ...) optimizes fidelity (Frobenius /
reconstruction). That is a metric misalignment: a completion that directly
maximizes a smooth surrogate of the ranking metric can beat fidelity solvers
on AUPR even at equal reconstruction quality.

This module provides a projected-gradient low-rank completion whose objective
mixes a fidelity term (keep the low-rank denoising signal) with an AUC /
AUPR surrogate term (push observed-positive entries above observed-negative
entries).  The low-rank constraint is enforced by truncated-SVD projection,
which is exactly the "projected gradient descent based on truncated SVD"
framework used for pairwise-AUC low-rank estimation.

The AUC surrogate is a sampled pairwise hinge / logistic over (positive,
negative) entry pairs — the objective directly targets the discrimination
the metric measures, instead of only the pointwise error.

NOTE on the zero-floor lesson (memory: completion-objective-innovations-
falsified): fitting ONLY the positives is catastrophic; the zeros are the
ranking's structural signal.  This solver therefore keeps the full fidelity
term on ALL entries AND adds the ranking term — it perturbs the ordering, it
does not drop the reconstruction constraint.

Subsequent rank-based ideas that are monotone post-processing of scores do
NOT change AUPR (rank is invariant under monotone transforms) and are
therefore useless; the ranking signal must live in the completion itself.
"""
import numpy as np


def _proj_lowrank(M, rank):
    """Truncated-SVD projection onto rank<=`rank` matrices."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(rank, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]


def auc_surrogate_completion(filled, pos_mask, neg_mask, rank=120,
                             lam_auc=1.0, mu=1.0, lr=0.5, iters=60,
                             n_neg_batch=4000, margin=0.05, seed=1,
                             warm_start=None, verbose=False):
    """Ranking-aware low-rank completion (projected gradient, vectorized).

    Objective
    ---------
    min_M  mu/2 * ||M - filled||²                (fidelity: keep denoising signal)
         + lam_auc * Σ_{(i,j)∈pos} Σ_{(k,l)∈neg} h(M_kl - M_ij + margin)
                                                  (AUC surrogate: ordering)
         s.t.  rank(M) <= rank

    Parameters
    ----------
    filled       : (dn, dr) base completion / fill matrix
    pos_mask     : bool (dn, dr) — observed positive entries (training labels)
    neg_mask     : bool (dn, dr) — entries treated as negatives for ranking
    rank         : low-rank projection cap
    lam_auc      : weight of the ranking term
    mu           : weight of the fidelity term
    lr           : projected-gradient step
    iters        : outer iterations
    n_neg_batch  : negatives sampled per iteration (shared pool)
    margin       : AUC hinge margin
    warm_start   : initial M; defaults to ``filled``

    Returns
    -------
    M : (dn, dr) completed scores in [0, 1].
    """
    rng = np.random.RandomState(seed)
    dn, dr = filled.shape
    pos_idx = np.argwhere(pos_mask)
    neg_idx = np.argwhere(neg_mask)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return np.clip(np.asarray(filled, dtype=float), 0, 1)

    M = np.asarray(warm_start, dtype=float) if warm_start is not None \
        else np.clip(np.asarray(filled, dtype=float), 0, 1)
    M = _proj_lowrank(M, rank)
    n_neg_batch = min(n_neg_batch, n_neg)
    batch_pos = min(n_pos, 512)

    for it in range(iters):
        # ── sample this iteration's positives and negatives ──
        pi = pos_idx[rng.choice(n_pos, batch_pos, replace=False)]
        ni = neg_idx[rng.choice(n_neg, n_neg_batch, replace=False)]

        # ── AUC-surrogate gradient (vectorized pairwise hinge) ──
        #  deltas[b, k] = M[neg_k] - M[pos_b] + margin
        deltas = (M[ni[:, 0], ni[:, 1]][None, :]
                  - M[pi[:, 0], pi[:, 1]][:, None] + margin)
        active = deltas > 0.0                      # (B, N) bool
        col_cnt = active.sum(axis=0)               # per-negative weight
        row_cnt = active.sum(axis=1)               # per-positive weight

        grad_auc = np.zeros((dn, dr))
        np.add.at(grad_auc, (pi[:, 0], pi[:, 1]), -row_cnt)
        np.add.at(grad_auc, (ni[:, 0], ni[:, 1]), col_cnt)
        grad_auc *= lam_auc / (batch_pos * n_neg_batch)

        # ── fidelity gradient ──
        grad_fid = mu * (M - filled)

        # ── projected gradient step ──
        M = _proj_lowrank(M - lr * (grad_auc + grad_fid), rank)
        M = np.clip(M, 0, 1)

    return M


def rank_calibrate(scores, train_mask, labels, rank=120, lam=1.0, mu=1.0,
                   lr=0.5, iters=40, n_neg=2000, seed=1):
    """Ranking-calibrate a base completion's scores.

    Takes an existing completion ``scores`` and pushes its ordering toward
    the observed training labels with the same AUC-surrogate machinery, then
    re-projects to low rank.  The base completion supplies the reconstruction
    prior; this step reorders it to align with the metric.

    Parameters
    ----------
    scores     : (dn, dr) base completion scores
    train_mask : bool (dn, dr) — observed training entries (labels known)
    labels     : (dn, dr) 0/1 training labels (train_mask=True ⇒ observed)
    Returns calibrated (dn, dr) scores.
    """
    pos_mask = train_mask & (labels > 0)
    neg_mask = train_mask & (labels == 0)
    filled = np.asarray(scores, dtype=float)
    return auc_surrogate_completion(filled, pos_mask, neg_mask,
                                    rank=rank, lam_auc=lam, mu=mu, lr=lr,
                                    iters=iters, n_neg_batch=n_neg, seed=seed,
                                    warm_start=filled)
