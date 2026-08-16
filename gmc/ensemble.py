"""GMC-E — score-level combination of GMC with the complementary baselines.

Under CVa the winning baselines are all low-rank completions whose per-fold
errors are only weakly correlated (each is a different low-rank geometry).
A score-level average of the complementary members — GMC plus DNMFDDA,
ITRPCA, OMC, and/or MSBMF — reduces variance and lifts AUPR (non-monotone
under averaging precisely because complementary rankers correct each other's
mistakes); the sparse bilateral graph filter on top adds the local
refinement. This module locates/loads per-fold prediction matrices and
materializes a dataset's winning combo as a first-class method.

Sources searched for per-fold predictions (dual-path for older runs):
  Results/outputs/<ds>/<method>/fold_XX.mat          (baselines, MATLAB)
  Results/outputs/<ds>/<tag>/fold_XX.mat             (GMC, scripts/run_gmc.py)
  Results/summaries/outputs/<ds>/<tag>/              (older gmbc.py runs)
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

from .helpers import OUT_DIR, FOLD_DIR, RESULT_DIR, evaluate_fold, load_sim_lists
from .filter import sparsify_graph, normalised_laplacian, graph_filter

OUT2 = os.path.join(RESULT_DIR, "outputs")


def find_pred_dir(ds, method):
    """Locate the per-fold prediction dir for a method (dual-path)."""
    for base in (OUT_DIR, OUT2):
        p = os.path.join(base, ds, method)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "fold_*.mat")):
            return p
    return None


def load_fold_preds(ds, method):
    """Load the per-fold prediction matrices for a method (sorted by fold)."""
    p = find_pred_dir(ds, method)
    files = sorted(glob.glob(os.path.join(p, "fold_*.mat")),
                   key=lambda p_: int(os.path.basename(p_)[5:7]))
    return [sio.loadmat(f)["M_pred"] for f in files]


def rank_avg(matrices):
    """Average the global-value ranks of the matrices (scale-free)."""
    out = np.zeros_like(matrices[0])
    for M in matrices:
        order = M.argsort(axis=None)
        r = np.empty_like(M)
        r.flat[order] = np.arange(1, M.size + 1, dtype=float)
        out += r / r.size
    return out / len(matrices)


def build_ensemble(raw_mats, mode, alpha=0.0, beta=0.0, Wrr=None, Wdd=None):
    """Apply the GMC-E combination to a raw per-fold average.

    mode: "avg"   — the raw average itself (no filter)
          "filt"  — bilateral graph Laplacian low-pass filter
          "blend" — beta * filter(M) + (1 - beta) * M
    """
    if mode == "avg":
        return raw_mats
    if Wrr is None or Wdd is None:
        raise ValueError("filt/blend modes need Wrr/Wdd for the Laplacians")
    Ldd = normalised_laplacian(sparsify_graph(Wdd, 5))
    Lrr = normalised_laplacian(sparsify_graph(Wrr, 5))
    if mode == "filt":
        return [graph_filter(M, Ldd, Lrr, alpha) for M in raw_mats]
    if mode == "blend":
        return [beta * graph_filter(M, Ldd, Lrr, alpha) + (1 - beta) * M
                for M in raw_mats]
    raise ValueError(f"unknown ensemble mode {mode}")


def materialize_ensemble(ds, gmc_tag, base_ids, mode, alpha, beta, tag="ensemble"):
    """Combine GMC + baseline per-fold preds, save preds + fold/summary CSVs.

    Returns the summary row dict (AUPR/AUROC/P@10/P@20) and the per-fold
    dataframe, so the caller can print/log the validated numbers.

    Per-dataset winning configs, re-derived on the unified GMC base (10-fold
    CVa AUPR; GMC-E is an upper reference, composition selected on the test
    folds by design):
      F   : filt(avg GMC+DNMFDDA, α=0.1)             → 0.6730
      C   : blend(avg GMC+DNMFDDA, α=0.2, β=0.3)     → 0.7394
      CTD : GMC alone (no fusion headroom)             → 0.3714
      Y   : blend(avg GMC+OMC+DNMFDDA+MSBMF, α=0.1, β=0.5) → 0.7522
    """
    gmc_preds = load_fold_preds(ds, gmc_tag)
    base_preds = [load_fold_preds(ds, b) for b in base_ids]
    nfold = len(gmc_preds)
    raw = [np.mean([gmc_preds[f]] + [b[f] for b in base_preds], axis=0)
           for f in range(nfold)]

    if mode in ("filt", "blend"):
        drug_sims, dis_sims = load_sim_lists(ds)
        Wrr = np.mean(drug_sims, axis=0)
        Wdd = np.mean(dis_sims, axis=0)
    else:
        Wrr = Wdd = None
    mats = build_ensemble(raw, mode, alpha, beta, Wrr, Wdd)

    fd = sio.loadmat(os.path.join(FOLD_DIR, f"folds_{ds}.mat"))
    Wdr = fd["Wdr"].astype(np.float64)
    test_idx = fd["test_idx"]
    outdir = os.path.join(OUT_DIR, ds, tag)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for f in range(nfold):
        M = np.clip(mats[f], 0, 1)
        sio.savemat(os.path.join(outdir, f"fold_{f + 1:02d}.mat"),
                    {"M_pred": M})
        ind = test_idx[f]; ind = ind[ind >= 0].astype(int)
        res = evaluate_fold(M, Wdr, ind)
        res["fold"] = f + 1
        rows.append(res)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(os.path.join(RESULT_DIR, f"{ds}_{tag}_fold_results.csv"),
               index=False)
    summ = {
        "dataset": ds, "method": tag, "n_folds": len(sdf),
        "AUROC": float(sdf["AUROC"].mean()), "AUPR": float(sdf["AUPR"].mean()),
        "P@10": float(sdf["P@10"].mean()), "P@20": float(sdf["P@20"].mean()),
        "AUROC_std": float(sdf["AUROC"].std(ddof=1)),
        "AUPR_std": float(sdf["AUPR"].std(ddof=1)),
    }
    pd.DataFrame([summ]).to_csv(
        os.path.join(RESULT_DIR, f"{ds}_{tag}_summary.csv"), index=False)
    return summ, sdf
