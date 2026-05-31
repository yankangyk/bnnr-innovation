"""
GF-BNNR: Graph-Filtered Bounded Nuclear Norm Regularization
============================================================
Compares GF-BNNR against two baselines on all 3 benchmark datasets.

Methods:
  1. BNNR_raw   — original BNNR with raw similarities
  2. BNNR_GIP   — BNNR with GIP-enhanced similarities (w=0.3)
  3. GF_BNNR    — BNNR_GIP + bi-directional graph low-pass filter (alpha=0.5)

Usage:
    python scripts/run_gfbnnr.py
"""

import json, os, time, warnings
import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings('ignore')

from bnnr import (BNNR, GF_BNNR, getGIPSim,
                   getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   compute_topk_metrics,
                   ensure_dir, load_dataset, mask_test_entries,
                   build_augmented_matrix, extract_recovery_block,
                   evaluate_fold)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "Results", "GFBNNR")

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 12345
NFOLD = 10
CVTYPE = "CVa"

ALPHA, BETA = 1, 10
TOL1, TOL2 = 2e-3, 1e-5
MAXITER = 300
A_BOUND, B_BOUND = 0, 1

GAMMA_GIP = 1
W_GIP = 0.3
GRAPH_ALPHA = 0.5

DATASETS = ["Fdataset", "Cdataset", "DNdataset"]
METHODS = ["BNNR_raw", "BNNR_GIP", "GF_BNNR"]


# ── Script-specific helpers ──────────────────────────────────────────────────
def load_dataset(name):
    data = sio.loadmat(os.path.join(DATASET_DIR, f"{name}.mat"))
    return (data["drug"].astype(np.float64),
            data["disease"].astype(np.float64),
            data["didr"].astype(np.float64))


# ── Main loop ────────────────────────────────────────────────────────────────
def run_dataset(ds_name):
    Wrr, Wdd, Wdr = load_dataset(ds_name)
    n_dis, n_drug = Wdr.shape
    print(f"\n{'=' * 80}")
    print(f"Dataset: {ds_name}  |  {n_dis} diseases  {n_drug} drugs  "
          f"{int(np.count_nonzero(Wdr))} known  "
          f"({np.count_nonzero(Wdr)/(n_dis*n_drug)*100:.2f}%)")
    print(f"{'=' * 80}")

    np.random.seed(SEED + 1)
    CVdata = getKfoldCrossValidMatIndSet(Wdr, NFOLD, CVTYPE, "Unlabel", SEED + 1)
    method_folds = {m: [] for m in METHODS}

    for fold in range(NFOLD):
        Ind_test = np.union1d(CVdata["MatIndSet_pos_test"][fold],
                              CVdata["MatIndSet_neg_test"][fold])
        matDR = mask_test_entries(Wdr, Ind_test)
        t0 = time.time()

        # BNNR_raw: original similarities (adaptive_svd=False matches baseline)
        T, trIdx = build_augmented_matrix(Wrr, Wdd, matDR)
        WW, it_raw = BNNR(ALPHA, BETA, T, trIdx, TOL1, TOL2, MAXITER,
                          A_BOUND, B_BOUND, adaptive_svd=False)
        M_raw = extract_recovery_block(WW, n_dis, n_drug)
        r_raw = evaluate_fold(M_raw, Wdr, Ind_test)
        r_raw.update({"fold": fold + 1, "method": "BNNR_raw", "iter": it_raw})
        method_folds["BNNR_raw"].append(r_raw)

        # Shared GIP for BNNR_GIP and GF_BNNR
        G_dis, G_drug = getGIPSim(matDR, GAMMA_GIP, GAMMA_GIP, 0, 0)
        S_drug = W_GIP * G_drug + (1 - W_GIP) * Wrr
        S_dis = W_GIP * G_dis + (1 - W_GIP) * Wdd

        # BNNR_GIP (adaptive_svd=False matches baseline)
        T2, trIdx2 = build_augmented_matrix(S_drug, S_dis, matDR)
        WW2, it_gip = BNNR(ALPHA, BETA, T2, trIdx2, TOL1, TOL2, MAXITER,
                           A_BOUND, B_BOUND, adaptive_svd=False)
        M_gip = extract_recovery_block(WW2, n_dis, n_drug)
        r_gip = evaluate_fold(M_gip, Wdr, Ind_test)
        r_gip.update({"fold": fold + 1, "method": "BNNR_GIP", "iter": it_gip})
        method_folds["BNNR_GIP"].append(r_gip)

        # GF_BNNR
        M_gf, _, it_gf = GF_BNNR(Wrr, Wdd, matDR,
                                 alpha=ALPHA, beta=BETA,
                                 tol1=TOL1, tol2=TOL2,
                                 maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
                                 gamma_gip=GAMMA_GIP, w_gip=W_GIP,
                                 graph_alpha=GRAPH_ALPHA,
                                 S_drug=S_drug, S_dis=S_dis)
        r_gf = evaluate_fold(M_gf, Wdr, Ind_test)
        r_gf.update({"fold": fold + 1, "method": "GF_BNNR", "iter": it_gf})
        method_folds["GF_BNNR"].append(r_gf)

        elapsed = time.time() - t0
        print(f"Fold {fold+1:2d} [{elapsed:.0f}s]  "
              f"RAW={r_raw['AUROC']:.4f}  GIP={r_gip['AUROC']:.4f}  "
              f"GF={r_gf['AUROC']:.4f}")

    # Summary
    base_auc = np.mean([r["AUROC"] for r in method_folds["BNNR_raw"]])
    base_ap = np.mean([r["AUPR"] for r in method_folds["BNNR_raw"]])
    print(f"\n{'─' * 80}")
    print(f"RESULTS: {ds_name}  ({NFOLD}-fold CVa)")
    print(f"{'─' * 80}")
    header = (f"{'Method':<16s} {'AUROC':>14s} {'AUPR':>14s}  "
              f"{'P@10':>8s} {'P@20':>8s}  {'Δ AUC':>8s} {'Δ AP':>8s}")
    print(header)
    print("-" * 88)
    for m in METHODS:
        recs = method_folds[m]
        ma = np.mean([r["AUROC"] for r in recs])
        sa = np.std([r["AUROC"] for r in recs], ddof=1)
        mp = np.mean([r["AUPR"] for r in recs])
        sp = np.std([r["AUPR"] for r in recs], ddof=1)
        mp10 = np.mean([r["P@10"] for r in recs])
        mp20 = np.mean([r["P@20"] for r in recs])
        da = (ma - base_auc) * 100
        dp = (mp - base_ap) * 100
        print(f"{m:<16s} {ma:.4f} +/- {sa:.4f}  {mp:.4f} +/- {sp:.4f}  "
              f"{mp10:.4f}  {mp20:.4f}  {da:+.2f}%  {dp:+.2f}%")

    return method_folds


def main():
    ensure_dir(RESULT_DIR)
    all_rows = []

    for ds in DATASETS:
        ds_dir = os.path.join(RESULT_DIR, ds)
        ensure_dir(ds_dir)
        method_folds = run_dataset(ds)

        for m in METHODS:
            dfm = pd.DataFrame(method_folds[m])
            dfm.to_csv(os.path.join(ds_dir, f"{m}_folds.csv"),
                       index=False, encoding="utf-8-sig")
            all_rows.extend(method_folds[m])

        cols = ["AUROC", "AUPR", "P@10", "P@20", "R@10", "R@20",
                "Hits@10", "Hits@20"]
        summary = {}
        for m in METHODS:
            dfm = pd.DataFrame(method_folds[m])
            summary[m] = {c: {"mean": float(dfm[c].mean()),
                              "std": float(dfm[c].std(ddof=1))}
                          for c in cols}
        with open(os.path.join(ds_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(all_rows).to_csv(os.path.join(RESULT_DIR, "all_folds.csv"),
                                  index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
