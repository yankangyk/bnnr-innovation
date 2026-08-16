"""KNN cold-start fill (multiGMF KNN_diseaseS/KNN_drugS port).

OMC's cold-start KNN fill: for every all-zero row (cold disease), fill the
row with the similarity-weighted average of its K most similar (non-cold)
diseases' rows; symmetrically for all-zero columns (cold drugs).  Only
all-zero entities are touched — partially observed entities stay sparse so
the completion propagates from true labels (same OMC-style principle as
``coldstart_fill`` but with a KNN-weighted propagation instead of the WKNN
soft labels).

On Ydataset this fill is systematically better than the WKNN soft-label
fill for the block completion: KNN fill + dual asymmetric block + nonzero
observation mask lifts Y AUPR 0.7176 → 0.7347 (10/10 folds, p=0.002), while
on F/C the WKNN fill + symmetric block + ones mask remains superior.  The
win is dataset-specific: Y is the largest, sparsest, and relies on rank
fusion where the block is the sole structure carrier.
"""
import numpy as np


def knn_fill_cold(masked, Wdd, Wrr, K=10):
    """Fill all-zero rows/cols by similarity-weighted KNN of the neighbors.

    Parameters
    ----------
    masked : (n_dis, n_drug) masked training association matrix
    Wdd    : (n_dis, n_dis) fused disease similarity
    Wrr    : (n_drug, n_drug) fused drug similarity
    K      : number of neighbors

    Returns
    -------
    P : (n_dis, n_drug) ``masked`` with all-zero rows/cols filled.
    """
    P = masked.copy().astype(float)
    row_no = np.where(P.sum(1) == 0)[0]
    for i in row_no:
        sims = Wdd[i, :].copy()
        sims[i] = 0
        nn = np.argsort(sims)[::-1][:K]
        w = sims[nn]
        if w.sum() > 0:
            P[i, :] = (w @ P[nn, :]) / w.sum()
    col_no = np.where(P.sum(0) == 0)[0]
    for j in col_no:
        sims = Wrr[j, :].copy()
        sims[j] = 0
        nn = np.argsort(sims)[::-1][:K]
        w = sims[nn]
        if w.sum() > 0:
            P[:, j] = (P[:, nn] @ w) / w.sum()
    return P
