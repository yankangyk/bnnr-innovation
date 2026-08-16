"""Random-predictor baseline (reference row for the overall comparison).

For each dataset, evaluate a UNIFORM random predictor on the same reported CVa
test folds (Results/folds/folds_<ds>.mat, SEED=12345) that all baselines use:
per fold, generate uniform [0,1] scores, evaluate against the held-out
indices.  The expected AUPR of a random predictor is the positive rate
(~0.01), so this row is the floor the whole comparison sits above.

Writes per-fold Results/summaries/<ds>_random_fold_results.csv and a summary
Results/summaries/<ds>_random_summary.csv (dataset, method, n_folds, AUROC,
AUPR, AUROC_std, AUPR_std).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import evaluate_fold
from gmc.helpers import FOLD_DIR, RESULT_DIR

DATASETS = ["Fdataset", "Cdataset", "CTDdataset2023", "Ydataset"]
N_TRIALS = 5  # average over a few random matrices to stabilize the reference


def run(ds, n_trials=N_TRIALS):
    fd = sio.loadmat(os.path.join(FOLD_DIR, f"folds_{ds}.mat"))
    Wdr = fd["Wdr"].astype(np.float64)
    test_idx = fd["test_idx"]
    nfold = test_idx.shape[0]
    rows = []
    rng = np.random.default_rng(20260813)
    for f in range(nfold):
        ind = test_idx[f]; ind = ind[ind >= 0].astype(int)
        best = None
        for _ in range(n_trials):
            M = rng.random(Wdr.shape)
            res = evaluate_fold(M, Wdr, ind)
            if best is None or res["AUPR"] > best["AUPR"]:
                best = dict(res)
        best["fold"] = f + 1
        rows.append(best)
    ddf = pd.DataFrame(rows)
    ddf.to_csv(os.path.join(RESULT_DIR, f"{ds}_random_fold_results.csv"), index=False)
    summ = {
        "dataset": ds, "method": "Random",
        "n_folds": len(ddf), "AUROC": float(ddf["AUROC"].mean()),
        "AUPR": float(ddf["AUPR"].mean()),
        "AUROC_std": float(ddf["AUROC"].std(ddof=1)),
        "AUPR_std": float(ddf["AUPR"].std(ddof=1)),
    }
    pd.DataFrame([summ]).to_csv(
        os.path.join(RESULT_DIR, f"{ds}_random_summary.csv"), index=False)
    print(f"{ds}: Random AUPR={summ['AUPR']:.4f} (n_pos/total = "
          f"{int(np.count_nonzero(Wdr))}/{Wdr.size} ≈ {np.count_nonzero(Wdr)/Wdr.size:.4f})")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    for ds in args.datasets:
        run(ds)
