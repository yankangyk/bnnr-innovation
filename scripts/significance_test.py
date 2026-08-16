"""Paired fold-level significance tests: GMC vs each comparison baseline.

CVa folds are paired by construction (same 10 held-out association splits), so
a fold-level paired test is the right comparison. For each dataset × metric we run
  - Wilcoxon signed-rank test (scipy.stats.wilcoxon)
  - paired Student's t-test (scipy.stats.ttest_rel)
on the 10 per-fold values, and report the mean difference with a 95% CI
(t-interval on the per-fold differences).

The model's per-fold metrics come from Results/summaries/<ds>_<tag>_fold_results.csv.
Baseline per-fold metrics are recomputed from the prediction matrices in
Results/outputs/<ds>/<method>/fold_XX.mat with the exact same metric code used
by the harness (gmc.helpers.evaluate_fold).

Usage: python scripts/significance_test.py [--model TAG] [dataset ...]
  TAG default: per-dataset GMC tag {Fdataset:gmc_cs_filt37,
  Cdataset:gmc_cs_filt37, CTDdataset2023:gmc_trrank, Ydataset:gmc_trrank}
Output: Results/summaries/SIGNIFICANCE_paired.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gmc import evaluate_fold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FOLD_DIR = os.path.join(ROOT, "Results", "folds")
OUT_DIR = os.path.join(ROOT, "Results", "outputs")
OUT2 = os.path.join(ROOT, "Results", "summaries", "outputs")
RESULT_DIR = os.path.join(ROOT, "Results", "summaries")

DEFAULT_MODEL = {
    "Fdataset": "gmc_cs_filt37", "Cdataset": "gmc_cs_filt37",
    "CTDdataset2023": "gmc_trrank", "Ydataset": "gmc_trrank",
}

METRICS = ["AUPR", "AUROC", "P@10"]


def load_model_folds(dataset, tag):
    df = pd.read_csv(os.path.join(RESULT_DIR, f"{dataset}_{tag}_fold_results.csv"))
    df = df.sort_values("fold")
    return df


def baseline_folds(dataset, method):
    fold_data = sio.loadmat(os.path.join(FOLD_DIR, f"folds_{dataset}.mat"))
    Wdr = fold_data["Wdr"].astype(np.float64)
    test_idx = fold_data["test_idx"]
    nfold = test_idx.shape[0]
    rows = []
    for f in range(nfold):
        p = os.path.join(OUT_DIR, dataset, method, f"fold_{f + 1:02d}.mat")
        if not os.path.exists(p):
            p = os.path.join(OUT2, dataset, method, f"fold_{f + 1:02d}.mat")
        if not os.path.exists(p):
            return None
        M_pred = sio.loadmat(p)["M_pred"].astype(np.float64)
        ind = test_idx[f]
        ind = ind[ind >= 0].astype(int)
        res = evaluate_fold(M_pred, Wdr, ind)
        rows.append(res)
    return pd.DataFrame(rows)


def ci95(diff):
    """95% CI for the mean of a small sample of differences (t-interval)."""
    n = len(diff)
    m, sd = diff.mean(), diff.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, h


def test_pair(x, y):
    """x = model per-fold, y = baseline per-fold. Return dict of stats."""
    d = np.asarray(x, float) - np.asarray(y, float)      # per-fold model - baseline
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.all(d == 0):
        return None
    try:
        w = stats.wilcoxon(d, zero_method="wilcox")
    except ValueError:
        w = None
    t = stats.ttest_1samp(d, 0.0)
    m, h = ci95(d)
    return {
        "mean_diff": float(m), "ci_lo": float(m - h), "ci_hi": float(m + h),
        "wilcoxon_p": float(w.pvalue) if w is not None else np.nan,
        "ttest_p": float(t.pvalue), "n_folds": int(len(d)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="per-fold CSV tag of the model to test (default: per-dataset GMC tag)")
    ap.add_argument("datasets", nargs="*",
                    default=["Fdataset", "Cdataset", "CTDdataset2023", "Ydataset"])
    args = ap.parse_args()
    rows = []
    for ds in args.datasets:
        tag = args.model or DEFAULT_MODEL.get(ds)
        if tag is None:
            print(f"[skip] no model tag for {ds}; pass --model")
            continue
        model_df = load_model_folds(ds, tag)
        ds_dir = os.path.join(OUT_DIR, ds)
        if not os.path.isdir(ds_dir):
            print(f"[skip] no outputs for {ds}")
            continue
        methods = sorted(os.listdir(ds_dir))
        print(f"== {ds}  (model {tag} folds: {len(model_df)}) ==")
        for method in methods:
            bdf = baseline_folds(ds, method)
            if bdf is None or len(bdf) != len(model_df):
                print(f"  [skip] {method}: missing folds")
                continue
            for metric in METRICS:
                if metric not in model_df.columns or metric not in bdf.columns:
                    continue
                r = test_pair(model_df[metric], bdf[metric])
                if r is None:
                    continue
                r.update({"dataset": ds, "method": method, "metric": metric,
                          "model_tag": tag,
                          "model": float(model_df[metric].mean()),
                          "base": float(bdf[metric].mean())})
                rows.append(r)
                sig = "sig" if r["wilcoxon_p"] < 0.05 else "n.s."
                print(f"  {method:<22} {metric:<6} {tag} {r['model']:.4f} vs {r['base']:.4f} "
                      f"Δ={r['mean_diff']:+.4f}  Wilcoxon p={r['wilcoxon_p']:.4f} [{sig}]  "
                      f"t p={r['ttest_p']:.4f}  95%CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(RESULT_DIR, "SIGNIFICANCE_paired.csv"), index=False)
        print(f"\n→ Results/summaries/SIGNIFICANCE_paired.csv  ({len(out)} rows)")
        # multiple-comparison note: report how many survive BH-FDR at q=0.05
        for ds in args.datasets:
            sub = out[(out["dataset"] == ds) & (out["metric"] == "AUPR")]
            if len(sub):
                pvals = sub["wilcoxon_p"].dropna().values
                from scipy.stats import false_discovery_control
                adj = false_discovery_control(pvals) if len(pvals) else []
                n_sig = int((adj < 0.05).sum()) if len(adj) else 0
                print(f"  {ds} AUPR: {len(pvals)} pairwise tests, {n_sig} survive BH-FDR q<0.05")


if __name__ == "__main__":
    main()
