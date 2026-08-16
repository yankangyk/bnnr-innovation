"""
Shared experiment utilities for the GMC benchmark scripts: dataset loading,
CV masking, per-fold evaluation, and the project's Results/ paths.

All run scripts import FOLD_DIR / OUT_DIR / RESULT_DIR from here so the
Results layout lives in one place.
"""
import os

import numpy as np
import scipy.io as sio

from .metrics import getPerfMetricROCcompute, compute_topk_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT, "data")
FOLD_DIR = os.path.join(ROOT, "Results", "folds")
OUT_DIR = os.path.join(ROOT, "Results", "outputs")
RESULT_DIR = os.path.join(ROOT, "Results", "summaries")

DRUG_SIMS = ["drug_ChemS", "drug_AtcS", "drug_SideS", "drug_DDIS", "drug_TargetS"]
DIS_SIMS = ["disease_PhS", "disease_DoS"]


def load_dataset(path):
    """
    Load a .mat dataset file.

    Handles both formats:
      multiGMF 5+2 (F/C/CT/Y) — individual drug_*/disease_* similarity keys
                                → returns the mean-fused Wrr / Wdd
      legacy single-sim (DN)   — plain drug / disease keys → returned as-is

    Returns: (Wrr, Wdd, Wdr) — drug similarity, disease similarity, association.
    """
    data = sio.loadmat(path)
    if "drug_ChemS" in data:
        drugs = [0.5 * (data[k] + data[k].T) for k in DRUG_SIMS]
        diss = [0.5 * (data[k] + data[k].T) for k in DIS_SIMS]
        return (np.mean(drugs, axis=0).astype(np.float64),
                np.mean(diss, axis=0).astype(np.float64),
                np.asarray(data["didr"]).astype(np.float64))
    return (data["drug"].astype(np.float64),
            data["disease"].astype(np.float64),
            data["didr"].astype(np.float64))


def load_sim_lists(dataset):
    """Return ``([drug_sims...], [dis_sims...])`` raw per-similarity lists.

    Keeps each of the 5 drug + 2 disease similarity slices separate (as the
    tensor view needs), instead of the mean fusion used by ``load_dataset``.
    """
    d = sio.loadmat(os.path.join(DATA_DIR, f"{dataset}.mat"))
    if "drug_ChemS" in d:
        drugs = [0.5 * (d[k] + d[k].T) for k in DRUG_SIMS]
        diss = [0.5 * (d[k] + d[k].T) for k in DIS_SIMS]
    else:
        drugs = [d["drug"].astype(np.float64)]
        diss = [d["disease"].astype(np.float64)]
    return drugs, diss


def mask_test_entries(Wdr, Ind_test):
    """Zero out test-index entries in the association matrix (Fortran order)."""
    matDR = Wdr.copy()
    matDR_ravel = matDR.ravel(order="F")
    matDR_ravel[Ind_test] = 0
    return matDR_ravel.reshape(matDR.shape, order="F")


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
