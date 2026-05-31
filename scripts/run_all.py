"""
BNNR Innovation — Comprehensive Paper Suite
=============================================
Full ablation study across 5 methods on all 3 benchmark datasets.

Methods (in evolutionary order):
  1. BNNR          — baseline bounded nuclear norm regularization
  2. RA-BNNR       — rank-adaptive beta scheduling
  3. GBNNR         — graph-regularized BNNR (kNN graph + manifold regularization)
  4. GBNNR-v3      — GBNNR + gamma confidence weighting + block-wise adaptive reg
  5. GBNNR-v3-GIP  — PROPOSED: GIP-enhanced input similarities (plug-and-play)

Datasets: Fdataset (593x313), Cdataset (663x409), DNdataset (1490x4516)
CV strategy: 10-fold CVa (random split by association pairs)

Usage:
    python scripts/run_all.py              # fresh run
    python scripts/run_all.py --resume     # resume from last completed fold
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse

from bnnr import (getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   BNNR, BNNR_adaptive, infer_ra_params,
                   BNNR_graph, BNNR_graph_enhanced_v3,
                   build_knn_graph, normalized_laplacian_sparse, getGIPSim,
                   compute_topk_metrics,
                   ensure_dir, load_dataset,
                   build_augmented_matrix, mask_test_entries)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "Results")

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 12345
NCV = 1
NFOLD = 10
CVTYPE = "CVa"

ALPHA = 1
BETA = 10
BETA_BY_DATASET = {
    "Fdataset": 10,
    "Cdataset": 10,
    "DNdataset": 10,
}
TOL1 = 2e-3
TOL2 = 1e-5
MAXITER = 300
A_BOUND = 0
B_BOUND = 1

GRAPH_CFG = {
    "lambda_r": 1e-3,
    "lambda_d": 1e-3,
    "knn_k": 12,
    "inner_steps": 10,
    "lr": 1e-2,
    "gamma": 2.0,
    "lambda_diag_factor": 0.2,
}

GRAPH_CFG_GIP = {**GRAPH_CFG, "w_gip": 0.2, "gamma_gip": 1.0}

DATASETS = {
    "Fdataset": os.path.join(DATASET_DIR, "Fdataset.mat"),
    "Cdataset": os.path.join(DATASET_DIR, "Cdataset.mat"),
    "DNdataset": os.path.join(DATASET_DIR, "DNdataset.mat"),
}


# ── Script-specific helpers ──────────────────────────────────────────────────
def build_augmented_laplacian_sparse(Wrr, Wdd, knn_k, gamma=1.0):
    G_drug = build_knn_graph(Wrr, k=knn_k, sym=True, remove_diag=True, gamma=gamma)
    G_disease = build_knn_graph(Wdd, k=knn_k, sym=True, remove_diag=True, gamma=gamma)
    L_drug = normalized_laplacian_sparse(G_drug)
    L_disease = normalized_laplacian_sparse(G_disease)
    return sparse.block_diag([L_drug, L_disease], format='csr')


def extract_recovery_block(WW, Wdd, Wdr):
    return WW[-Wdd.shape[0]:, :Wdr.shape[1]]


def evaluate_fold(M_recovery, Wdr, Ind_test):
    labels = Wdr.ravel(order="F")[Ind_test]
    scores = M_recovery.ravel(order="F")[Ind_test]

    tbScalar, tbVec, AUC, AUPR, Acc, Sen, Spe, Pre = getPerfMetricROCcompute(
        scores, labels, 1, 0)
    topk = compute_topk_metrics(scores, labels, ks=(10, 20))

    return {
        "AUROC": float(AUC), "AUPR": float(AUPR),
        "Acc": float(tbScalar["Acc"].values[0]),
        "Sen": float(tbScalar["Sen"].values[0]),
        "Spe": float(tbScalar["Spe"].values[0]),
        "Pre": float(tbScalar["Pre"].values[0]),
        "P@10": float(topk["P@10"]), "P@20": float(topk["P@20"]),
        "R@10": float(topk["R@10"]), "R@20": float(topk["R@20"]),
        "Hits@10": int(topk["Hits@10"]), "Hits@20": int(topk["Hits@20"]),
    }


def run_method_on_fold(method_name, Wrr, Wdd, Wdr,
                       IndSet_pos_test, IndSet_neg_test, i_fold,
                       graph_cfg=None, beta=BETA):
    Ind_pos_test = IndSet_pos_test[i_fold]
    Ind_neg_test = IndSet_neg_test[i_fold]
    Ind_test = np.union1d(Ind_pos_test, Ind_neg_test)

    matDR = mask_test_entries(Wdr, Ind_test)
    T, trIndex = build_augmented_matrix(Wrr, Wdd, matDR)

    iter_num = None
    beta_final = None
    inferred_ra = None

    if method_name == "BNNR":
        WW, iter_num = BNNR(alpha=ALPHA, beta=beta, T=T, trIndex=trIndex,
                            tol1=TOL1, tol2=TOL2, maxiter=MAXITER,
                            a=A_BOUND, b=B_BOUND)
        M_recovery = extract_recovery_block(WW, Wdd, Wdr)

    elif method_name == "RA-BNNR":
        T_density = np.count_nonzero(T) / T.size
        ra_cfg = infer_ra_params(T.shape, T_density)
        WW, iter_num, info = BNNR_adaptive(
            alpha=ALPHA, beta=beta, T=T, trIndex=trIndex,
            tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
            eta=ra_cfg["eta"],
            beta_min=ra_cfg["beta_min"],
            beta_max=ra_cfg["beta_max"],
            warmup=ra_cfg["warmup"],
            target_rank_ratio=ra_cfg["target_rank_ratio"],
            verbose=0)
        M_recovery = extract_recovery_block(WW, Wdd, Wdr)
        beta_final = info.get("beta_final")

    elif method_name == "GBNNR":
        if graph_cfg is None:
            raise ValueError("graph_cfg required for GBNNR")
        L_aug = build_augmented_laplacian_sparse(
            Wrr, Wdd, graph_cfg["knn_k"], graph_cfg.get("gamma", 1.0))
        WW, iter_num, info = BNNR_graph(
            alpha=ALPHA, beta=beta, T=T, trIndex=trIndex,
            tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
            L_r=L_aug, L_d=L_aug,
            lambda_r=graph_cfg["lambda_r"], lambda_d=graph_cfg["lambda_d"],
            inner_steps=graph_cfg["inner_steps"], lr=graph_cfg["lr"], verbose=0)
        M_recovery = extract_recovery_block(WW, Wdd, Wdr)

    elif method_name == "GBNNR-v3":
        if graph_cfg is None:
            raise ValueError("graph_cfg required for GBNNR-v3")
        WW, iter_num, info = BNNR_graph_enhanced_v3(
            alpha=ALPHA, beta=beta, T=T, trIndex=trIndex,
            tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
            Wrr_orig=Wrr, Wdd_orig=Wdd, n_drug=Wrr.shape[0],
            knn_k=graph_cfg["knn_k"],
            gamma_graph=graph_cfg.get("gamma", 2.0),
            lambda_r=graph_cfg["lambda_r"], lambda_d=graph_cfg["lambda_d"],
            lambda_diag_factor=graph_cfg.get("lambda_diag_factor", 0.2),
            inner_steps=graph_cfg["inner_steps"], lr=graph_cfg["lr"], verbose=0)
        M_recovery = extract_recovery_block(WW, Wdd, Wdr)

    elif method_name == "GBNNR-v3-GIP":
        if graph_cfg is None:
            raise ValueError("graph_cfg required for GBNNR-v3-GIP")
        w_gip = graph_cfg.get("w_gip", 0.2)
        gamma_gip = graph_cfg.get("gamma_gip", 1.0)
        G1, G2 = getGIPSim(matDR, gamma_gip, gamma_gip, 0, 0)
        if G1.shape == Wrr.shape:
            G_drug, G_disease = G1, G2
        else:
            G_drug, G_disease = G2, G1
        S_rr = w_gip * G_drug + (1 - w_gip) * Wrr
        S_dd = w_gip * G_disease + (1 - w_gip) * Wdd
        T_gip = np.block([[S_rr, matDR.T], [matDR, S_dd]])
        trIndex_gip = (T_gip != 0).astype(np.float64)
        WW, iter_num, info = BNNR_graph_enhanced_v3(
            alpha=ALPHA, beta=beta, T=T_gip, trIndex=trIndex_gip,
            tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
            Wrr_orig=S_rr, Wdd_orig=S_dd, n_drug=Wrr.shape[0],
            knn_k=graph_cfg["knn_k"],
            gamma_graph=graph_cfg.get("gamma", 2.0),
            lambda_r=graph_cfg["lambda_r"], lambda_d=graph_cfg["lambda_d"],
            lambda_diag_factor=graph_cfg.get("lambda_diag_factor", 0.2),
            inner_steps=graph_cfg["inner_steps"], lr=graph_cfg["lr"], verbose=0)
        M_recovery = extract_recovery_block(WW, S_dd, Wdr)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    eval_result = evaluate_fold(M_recovery, Wdr, Ind_test)
    eval_result["iter_num"] = int(iter_num) if iter_num else 0
    eval_result["beta_final"] = beta_final
    eval_result["inferred_ra"] = inferred_ra
    return eval_result


def summarize_fold_results(df):
    metric_cols = ["AUROC", "AUPR", "Acc", "Sen", "Spe", "Pre",
                   "P@10", "P@20", "R@10", "R@20",
                   "Hits@10", "Hits@20", "iter_num"]
    summary = {}
    for col in metric_cols:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std(ddof=1) if len(df) > 1 else 0.0
    if "beta_final" in df.columns and df["beta_final"].notna().any():
        bf = df["beta_final"].dropna()
        summary["beta_final_mean"] = bf.mean()
        summary["beta_final_std"] = bf.std(ddof=1) if len(bf) > 1 else 0.0
    else:
        summary["beta_final_mean"] = np.nan
        summary["beta_final_std"] = np.nan
    return summary


# ── Resume helpers ────────────────────────────────────────────────────────────
def load_completed_folds(csv_path):
    """Return (list of completed fold rows, set of completed fold_ids)."""
    if not os.path.exists(csv_path):
        return [], set()
    df = pd.read_csv(csv_path)
    return df.to_dict("records"), set(df["fold_id"].unique())


# ── Main ─────────────────────────────────────────────────────────────────────
def run_suite(resume=False):
    ensure_dir(RESULT_DIR)
    global_start = time.time()
    all_fold_rows = []
    all_summary_rows = []

    method_plan = [
        "BNNR",
        "RA-BNNR",
        "GBNNR",
        "GBNNR-v3",
        "GBNNR-v3-GIP",
    ]

    for dataset_name, dataset_path in DATASETS.items():
        print("\n" + "=" * 100)
        print(f"Dataset: {dataset_name}")
        print("=" * 100)

        dataset_dir = os.path.join(RESULT_DIR, dataset_name)
        ensure_dir(dataset_dir)
        Wrr, Wdd, Wdr = load_dataset(dataset_path)
        dn, dr = Wdr.shape
        print(f"Shape: {dr} drugs x {dn} diseases  "
              f"({np.count_nonzero(Wdr)} known, "
              f"{np.count_nonzero(Wdr)/(dn*dr)*100:.2f}%)")

        for i_cv in range(NCV):
            current_seed = SEED + i_cv + 1
            np.random.seed(current_seed)
            CVdata = getKfoldCrossValidMatIndSet(
                Wdr, NFOLD, CVTYPE, "Unlabel", current_seed)
            IndSet_pos_test = CVdata["MatIndSet_pos_test"]
            IndSet_neg_test = CVdata["MatIndSet_neg_test"]

            for method_name in method_plan:
                model_tag = method_name

                if method_name == "GBNNR-v3-GIP":
                    current_cfg = GRAPH_CFG_GIP
                elif method_name in ("BNNR", "RA-BNNR"):
                    current_cfg = None
                else:
                    current_cfg = GRAPH_CFG

                csv_path = os.path.join(
                    dataset_dir, f"{model_tag}_fold_results.csv")
                completed_fold_rows, completed_fold_ids = \
                    load_completed_folds(csv_path) if resume else ([], set())

                if resume and completed_fold_ids:
                    n_done = len(completed_fold_ids)
                    print(f"\nResuming: {model_tag} "
                          f"({n_done}/10 folds already done)")
                else:
                    print(f"\nRunning: {model_tag}")

                fold_rows = list(completed_fold_rows)
                if completed_fold_rows:
                    all_fold_rows.extend(completed_fold_rows)

                tic = time.time()

                for i_fold in range(NFOLD):
                    fold_id = i_fold + 1
                    if resume and fold_id in completed_fold_ids:
                        print(f"Fold {fold_id:02d}: [SKIPPED]")
                        continue

                    ds_beta = BETA_BY_DATASET.get(dataset_name, BETA)
                    fold_result = run_method_on_fold(
                        method_name=method_name, Wrr=Wrr, Wdd=Wdd, Wdr=Wdr,
                        IndSet_pos_test=IndSet_pos_test,
                        IndSet_neg_test=IndSet_neg_test,
                        i_fold=i_fold, graph_cfg=current_cfg, beta=ds_beta)

                    row = {
                        "dataset": dataset_name, "cv_id": i_cv + 1,
                        "fold_id": fold_id, "method": method_name,
                        "model_tag": model_tag,
                        "ra_name": None,
                        "AUROC": fold_result["AUROC"],
                        "AUPR": fold_result["AUPR"],
                        "Acc": fold_result["Acc"],
                        "Sen": fold_result["Sen"],
                        "Spe": fold_result["Spe"],
                        "Pre": fold_result["Pre"],
                        "P@10": fold_result["P@10"],
                        "P@20": fold_result["P@20"],
                        "R@10": fold_result["R@10"],
                        "R@20": fold_result["R@20"],
                        "Hits@10": fold_result["Hits@10"],
                        "Hits@20": fold_result["Hits@20"],
                        "iter_num": fold_result["iter_num"],
                        "beta_final": fold_result["beta_final"],
                    }
                    if method_name == "RA-BNNR" and fold_result.get("inferred_ra"):
                        row["ra_name"] = "auto_inferred"
                        row["ra_eta"] = fold_result["inferred_ra"]["eta"]
                        row["ra_beta_max"] = fold_result["inferred_ra"]["beta_max"]
                        row["ra_ratio"] = fold_result["inferred_ra"]["target_rank_ratio"]
                    fold_rows.append(row)
                    all_fold_rows.append(row)

                    # Save incrementally after each fold (crash-safe)
                    pd.DataFrame(fold_rows).to_csv(
                        csv_path, index=False, encoding="utf-8-sig")

                    extra = ""
                    if method_name == "RA-BNNR" and row["ra_name"]:
                        extra = (f", eta={row['ra_eta']:.4f}, "
                                 f"beta_max={row['ra_beta_max']:.1f}, "
                                 f"ratio={row['ra_ratio']:.2f}")
                    print(f"Fold {fold_id:02d}: "
                          f"AUROC={row['AUROC']:.4f}, AUPR={row['AUPR']:.4f}, "
                          f"P@10={row['P@10']:.4f}, P@20={row['P@20']:.4f}{extra}")

                elapsed_min = (time.time() - tic) / 60.0
                fold_df = pd.DataFrame(fold_rows)
                summary = summarize_fold_results(fold_df)

                summary_row = {"dataset": dataset_name, "cv_id": i_cv + 1,
                               "method": method_name, "model_tag": model_tag,
                               "time_min": elapsed_min}
                summary_row.update(summary)
                all_summary_rows.append(summary_row)

                print(f"Summary: AUROC={summary['AUROC_mean']:.4f}"
                      f"±{summary['AUROC_std']:.4f}, "
                      f"AUPR={summary['AUPR_mean']:.4f}"
                      f"±{summary['AUPR_std']:.4f}")

                fold_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                with open(os.path.join(
                        dataset_dir, f"{model_tag}_config_and_summary.json"),
                        "w", encoding="utf-8") as f:
                    json.dump({"dataset": dataset_name, "method": method_name,
                               "model_tag": model_tag, "summary": summary_row},
                              f, ensure_ascii=False, indent=2)

    all_fold_df = pd.DataFrame(all_fold_rows)
    all_summary_df = pd.DataFrame(all_summary_rows)
    all_fold_df.to_csv(os.path.join(RESULT_DIR, "all_fold_results.csv"),
                       index=False, encoding="utf-8-sig")
    all_summary_df.to_csv(os.path.join(RESULT_DIR, "all_summary_results.csv"),
                          index=False, encoding="utf-8-sig")

    main_cols = ["dataset", "model_tag", "AUROC_mean", "AUROC_std",
                 "AUPR_mean", "AUPR_std", "P@10_mean", "P@20_mean", "time_min"]
    paper_table = all_summary_df[main_cols].copy()
    paper_table.to_csv(os.path.join(RESULT_DIR, "paper_main_table.csv"),
                       index=False, encoding="utf-8-sig")

    print(f"\nAll finished. "
          f"Total time: {(time.time() - global_start) / 3600:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BNNR Innovation — Comprehensive Paper Suite")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed fold")
    args = parser.parse_args()
    run_suite(resume=args.resume)
