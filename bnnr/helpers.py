"""
Shared experiment utilities for BNNR benchmark scripts.

Convenience wrappers — not core algorithms. Eliminates duplication across
run_all.py, run_gfbnnr.py, and quick_demo.py.
"""
import os
import numpy as np
import scipy.io as sio
from .metrics import getPerfMetricROCcompute, compute_topk_metrics


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_dataset(path):
    """
    Load a .mat dataset file.

    Returns: (Wrr, Wdd, Wdr) — drug similarity, disease similarity, association matrix.
    """
    data = sio.loadmat(path)
    return (data["drug"].astype(np.float64),
            data["disease"].astype(np.float64),
            data["didr"].astype(np.float64))


def mask_test_entries(Wdr, Ind_test):
    """Zero out test-index entries in the association matrix (Fortran order)."""
    matDR = Wdr.copy()
    matDR_ravel = matDR.ravel(order="F")
    matDR_ravel[Ind_test] = 0
    return matDR_ravel.reshape(matDR.shape, order="F")


def build_augmented_matrix(S_rr, S_dd, Wdr_masked):
    """
    Build augmented block matrix T = [[S_rr, Wdr^T], [Wdr, S_dd]]
    and its observation mask trIndex.
    """
    T = np.block([[S_rr, Wdr_masked.T], [Wdr_masked, S_dd]])
    trIndex = (T != 0).astype(np.float64)
    return T, trIndex


def extract_recovery_block(WW, n_dis, n_drug):
    """Extract the predicted association block from the augmented recovery matrix."""
    return WW[-n_dis:, :n_drug]


def evaluate_fold(M_recovery, Wdr, Ind_test, ks=(10, 20)):
    """
    Evaluate a single fold's predictions.

    Returns dict with AUROC, AUPR, Acc, Sen, Spe, Pre, P@K, R@K, Hits@K.
    """
    labels = Wdr.ravel(order="F")[Ind_test]
    scores = M_recovery.ravel(order="F")[Ind_test]
    tbScalar, _tbVec, AUC, AUPR, Acc, Sen, Spe, Pre = getPerfMetricROCcompute(
        scores, labels, 1, 0)
    topk = compute_topk_metrics(scores, labels, ks=ks)
    result = {
        "AUROC": float(AUC), "AUPR": float(AUPR),
        "Acc": float(tbScalar["Acc"].values[0]),
        "Sen": float(tbScalar["Sen"].values[0]),
        "Spe": float(tbScalar["Spe"].values[0]),
        "Pre": float(tbScalar["Pre"].values[0]),
    }
    for k_val in ks:
        result[f"P@{k_val}"] = float(topk[f"P@{k_val}"])
        result[f"R@{k_val}"] = float(topk[f"R@{k_val}"])
        result[f"Hits@{k_val}"] = int(topk[f"Hits@{k_val}"])
    return result
