"""
Run the missing control group: BNNR_raw + graph filter with raw similarities (no GIP).

Creates a clean 2x2 factorial design:
                     | No GIP (w_gip=0)  | GIP (w_gip=0.3)
    No filter (a=0)  | BNNR_raw          | BNNR_GIP
    Filter (a=0.5)   | BNNR_filter_raw   | GF_BNNR

Usage:
    python scripts/run_filter_only.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings('ignore')

from bnnr import (BNNR, GF_BNNR, getGIPSim,
                   getKfoldCrossValidMatIndSet,
                   ensure_dir, mask_test_entries,
                   build_augmented_matrix, extract_recovery_block,
                   evaluate_fold)
from bnnr.filter import _normalised_laplacian, _graph_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "Results", "GFBNNR")

# Same config as run_gfbnnr.py
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


def load_dataset(name):
    data = sio.loadmat(os.path.join(DATASET_DIR, f"{name}.mat"))
    return (data["drug"].astype(np.float64),
            data["disease"].astype(np.float64),
            data["didr"].astype(np.float64))


def run_dataset(ds_name):
    Wrr, Wdd, Wdr = load_dataset(ds_name)
    n_dis, n_drug = Wdr.shape
    n_known = int(np.count_nonzero(Wdr))
    print(f"\n{'='*80}")
    print(f"Dataset: {ds_name}  |  {n_dis} diseases  {n_drug} drugs  "
          f"{n_known} known  "
          f"({n_known/(n_dis*n_drug)*100:.2f}%)")
    print(f"{'='*80}")

    np.random.seed(SEED + 1)
    CVdata = getKfoldCrossValidMatIndSet(Wdr, NFOLD, CVTYPE, "Unlabel", SEED + 1)

    methods = ["BNNR_raw", "BNNR_filter_raw", "BNNR_GIP", "GF_BNNR"]
    method_folds = {m: [] for m in methods}

    for fold in range(NFOLD):
        Ind_test = np.union1d(CVdata["MatIndSet_pos_test"][fold],
                              CVdata["MatIndSet_neg_test"][fold])
        matDR = mask_test_entries(Wdr, Ind_test)
        t0 = time.time()

        # ── 1. BNNR_raw (standard baseline) ──
        T, trIdx = build_augmented_matrix(Wrr, Wdd, matDR)
        WW_raw, it_raw = BNNR(ALPHA, BETA, T, trIdx, TOL1, TOL2, MAXITER,
                              A_BOUND, B_BOUND, adaptive_svd=False)
        M_raw = extract_recovery_block(WW_raw, n_dis, n_drug)
        r_raw = evaluate_fold(M_raw, Wdr, Ind_test)
        r_raw.update({"fold": fold + 1, "method": "BNNR_raw", "iter": it_raw})
        method_folds["BNNR_raw"].append(r_raw)

        # ── 2. BNNR_filter_raw (NEW: raw filter, no GIP) ──
        # Apply graph filter to BNNR_raw output using RAW similarities only
        L_dis_raw = _normalised_laplacian(Wdd)
        L_drug_raw = _normalised_laplacian(Wrr)
        M_filter_raw = _graph_filter(M_raw, L_dis_raw, L_drug_raw, GRAPH_ALPHA)
        r_filt_raw = evaluate_fold(M_filter_raw, Wdr, Ind_test)
        r_filt_raw.update({"fold": fold + 1, "method": "BNNR_filter_raw",
                            "iter": it_raw})
        method_folds["BNNR_filter_raw"].append(r_filt_raw)

        # ── Shared GIP similarities for BNNR_GIP and GF_BNNR ──
        G_dis, G_drug = getGIPSim(matDR, GAMMA_GIP, GAMMA_GIP, 0, 0)
        S_drug = W_GIP * G_drug + (1 - W_GIP) * Wrr
        S_dis = W_GIP * G_dis + (1 - W_GIP) * Wdd

        # ── 3. BNNR_GIP ──
        T_gip, trIdx_gip = build_augmented_matrix(S_drug, S_dis, matDR)
        WW_gip, it_gip = BNNR(ALPHA, BETA, T_gip, trIdx_gip, TOL1, TOL2,
                              MAXITER, A_BOUND, B_BOUND, adaptive_svd=False)
        M_gip = extract_recovery_block(WW_gip, n_dis, n_drug)
        r_gip = evaluate_fold(M_gip, Wdr, Ind_test)
        r_gip.update({"fold": fold + 1, "method": "BNNR_GIP", "iter": it_gip})
        method_folds["BNNR_GIP"].append(r_gip)

        # ── 4. GF_BNNR (GIP + filter) ──
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
              f"RAW={r_raw['AUROC']:.4f}/{r_raw['AUPR']:.4f}  "
              f"RAW+F={r_filt_raw['AUROC']:.4f}/{r_filt_raw['AUPR']:.4f}  "
              f"GIP={r_gip['AUROC']:.4f}/{r_gip['AUPR']:.4f}  "
              f"GF={r_gf['AUROC']:.4f}/{r_gf['AUPR']:.4f}")

    # ── Summary ──
    base_auc = np.mean([r["AUROC"] for r in method_folds["BNNR_raw"]])
    base_ap = np.mean([r["AUPR"] for r in method_folds["BNNR_raw"]])
    print(f"\n{'─'*90}")
    print(f"RESULTS: {ds_name}  ({NFOLD}-fold CVa)")
    print(f"{'─'*90}")
    header = (f"{'Method':<20s} {'AUROC':>14s} {'AUPR':>14s}  "
              f"{'P@10':>8s} {'P@20':>8s}  {'Delta AUC':>10s} {'Delta AP':>10s}")
    print(header)
    print("-" * 90)
    for m in methods:
        recs = method_folds[m]
        ma = np.mean([r["AUROC"] for r in recs])
        sa = np.std([r["AUROC"] for r in recs], ddof=1)
        mp = np.mean([r["AUPR"] for r in recs])
        sp = np.std([r["AUPR"] for r in recs], ddof=1)
        mp10 = np.mean([r["P@10"] for r in recs])
        mp20 = np.mean([r["P@20"] for r in recs])
        da = (ma - base_auc) * 100
        dp = (mp - base_ap) * 100
        print(f"{m:<20s} {ma:.4f} +/- {sa:.4f}  {mp:.4f} +/- {sp:.4f}  "
              f"{mp10:.4f}  {mp20:.4f}  {da:+7.2f}%  {dp:+7.2f}%")
        print(f"  -> vs BNNR_raw:  "
              f"ΔAUROC={ma-base_auc:+.4f}  ΔAUPR={mp-base_ap:+.4f}")

    # ── Decomposition table ──
    raw_auc = np.mean([r["AUROC"] for r in method_folds["BNNR_raw"]])
    raw_ap = np.mean([r["AUPR"] for r in method_folds["BNNR_raw"]])
    gip_auc = np.mean([r["AUROC"] for r in method_folds["BNNR_GIP"]])
    gip_ap = np.mean([r["AUPR"] for r in method_folds["BNNR_GIP"]])
    filt_auc = np.mean([r["AUROC"] for r in method_folds["BNNR_filter_raw"]])
    filt_ap = np.mean([r["AUPR"] for r in method_folds["BNNR_filter_raw"]])
    gf_auc = np.mean([r["AUROC"] for r in method_folds["GF_BNNR"]])
    gf_ap = np.mean([r["AUPR"] for r in method_folds["GF_BNNR"]])

    print(f"\n{'─'*90}")
    print(f"DECOMPOSITION: {ds_name}")
    print(f"{'─'*90}")
    print(f"  GIP main effect (no filter):   "
          f"ΔAUROC={gip_auc-raw_auc:+.4f}  ΔAUPR={gip_ap-raw_ap:+.4f}")
    print(f"  Filter main effect (no GIP):    "
          f"ΔAUROC={filt_auc-raw_auc:+.4f}  ΔAUPR={filt_ap-raw_ap:+.4f}")
    print(f"  Combined (GIP + filter):        "
          f"ΔAUROC={gf_auc-raw_auc:+.4f}  ΔAUPR={gf_ap-raw_ap:+.4f}")
    print(f"  Interaction:                    "
          f"ΔAUROC={(gf_auc-raw_auc)-(gip_auc-raw_auc)-(filt_auc-raw_auc):+.4f}  "
          f"ΔAUPR={(gf_ap-raw_ap)-(gip_ap-raw_ap)-(filt_ap-raw_ap):+.4f}")

    return method_folds


def main():
    ensure_dir(RESULT_DIR)
    all_rows = []

    for ds in DATASETS:
        ds_dir = os.path.join(RESULT_DIR, ds)
        ensure_dir(ds_dir)
        method_folds = run_dataset(ds)

        for m in ["BNNR_raw", "BNNR_filter_raw", "BNNR_GIP", "GF_BNNR"]:
            dfm = pd.DataFrame(method_folds[m])
            dfm.to_csv(os.path.join(ds_dir, f"{m}_folds.csv"),
                       index=False, encoding="utf-8-sig")
            all_rows.extend(method_folds[m])

    pd.DataFrame(all_rows).to_csv(
        os.path.join(RESULT_DIR, "all_folds_with_filter_raw.csv"),
        index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
