"""Evaluate comparison-method predictions under the CVa protocol.

Reads the prediction matrices saved by the MATLAB drivers
(Results/outputs/<dataset>/<method>/fold_XX.mat), scores every fold with the
exact same metric code as the GMC harness (gmc.helpers.evaluate_fold →
gmc.metrics), and writes one summary CSV per method.

Usage: python evaluate.py [dataset ...]
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio

# Make the project root importable when this script is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import evaluate_fold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FOLD_DIR = os.path.join(ROOT, "Results", "folds")
OUT_DIR = os.path.join(ROOT, "Results", "outputs")
RESULT_DIR = os.path.join(ROOT, "Results", "summaries")
os.makedirs(RESULT_DIR, exist_ok=True)

METRIC_COLS = ["AUROC", "AUPR", "Acc", "Sen", "Spe", "Pre", "P@10", "P@20", "R@10", "R@20", "Hits@10", "Hits@20"]


def evaluate_method(dataset, method, outdir):
    fold_data = sio.loadmat(os.path.join(FOLD_DIR, f"folds_{dataset}.mat"))
    Wdr = fold_data["Wdr"].astype(np.float64)
    test_idx = fold_data["test_idx"]
    nfold = test_idx.shape[0]

    rows = []
    for f in range(nfold):
        pred_path = os.path.join(outdir, dataset, method, f"fold_{f + 1:02d}.mat")
        if not os.path.exists(pred_path):
            print(f"  [missing] {pred_path}")
            continue
        M_pred = sio.loadmat(pred_path)["M_pred"].astype(np.float64)
        assert M_pred.shape == Wdr.shape, f"{method} pred shape {M_pred.shape} != {Wdr.shape}"
        ind = test_idx[f]
        ind = ind[ind >= 0].astype(int)
        res = evaluate_fold(M_pred, Wdr, ind)
        res["fold"] = f + 1
        rows.append(res)

    if not rows:
        print(f"  {method}: NO folds evaluated")
        return None

    df = pd.DataFrame(rows)
    summary = {}
    for m in METRIC_COLS:
        if m in df.columns:
            summary[m] = float(df[m].mean())
            summary[f"{m}_std"] = float(df[m].std(ddof=1))
    summary["n_folds"] = len(df)
    summary["dataset"] = dataset
    summary["method"] = method
    sdf = pd.DataFrame([summary])
    out_csv = os.path.join(RESULT_DIR, f"{dataset}_{method}_summary.csv")
    sdf.to_csv(out_csv, index=False)
    print(f"  {dataset} {method}: AUROC={summary['AUROC']:.4f} AUPR={summary['AUPR']:.4f}")
    return sdf


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["Fdataset", "Cdataset"]
    for ds in datasets:
        ds_dir = os.path.join(OUT_DIR, ds)
        if not os.path.isdir(ds_dir):
            print(f"  [skip] no output dir for {ds}")
            continue
        print(f"== {ds} ==")
        for method in sorted(os.listdir(ds_dir)):
            evaluate_method(ds, method, OUT_DIR)


if __name__ == "__main__":
    main()
