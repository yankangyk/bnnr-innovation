"""
BADGE Experiment — Bi-iterative Adaptive Drug-disease Graph Enhancement
========================================================================

Compares BADGE against BNNR, GBNNR, and GF-BNNR on 3 benchmark datasets.

Experiments:
  E0  BNNR              baseline (raw similarities)
  E1  GBNNR             inside graph regularisation
  E2  GF-BNNR           outside post-hoc graph filter (= BADGE n_iter=1)
  E3  BADGE n_iter=2    PROPOSED — one round of Bayesian GIP refinement
  E4  BADGE n_iter=3    convergence check

Key validation criteria:
  (a) E3 >= E2 on moderate-density data (Fdataset, Cdataset)
  (b) E3 >= E2 on ultra-sparse data (DNdataset)
  (c) E4 ~ E3  — convergence within 2 iterations

Usage:
    python scripts/run_badge.py                  # fresh run, all datasets
    python scripts/run_badge.py --quick           # Fdataset only, 3 folds
    python scripts/run_badge.py --resume          # resume from last fold
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bnnr import (getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   BNNR, GBNNR,
                   GF_BNNR, BADGE,
                   getGIPSim,
                   compute_topk_metrics,
                   ensure_dir, load_dataset,
                   build_augmented_matrix, mask_test_entries)

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "Results", "BADGE")

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 12345
NFOLD = 10
CVTYPE = "CVa"

ALPHA = 1
BETA = 10
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

BADGE_CFG = {
    "graph_alpha": 0.5,
    "w_gip": 0.3,
    "gamma_gip": 1.0,
}

DATASETS = {
    "Fdataset": os.path.join(DATASET_DIR, "Fdataset.mat"),
    "Cdataset": os.path.join(DATASET_DIR, "Cdataset.mat"),
    "DNdataset": os.path.join(DATASET_DIR, "DNdataset.mat"),
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def extract_recovery_block(WW, Wdd, Wdr):
    return WW[-Wdd.shape[0]:, :Wdr.shape[1]]


def evaluate_fold(M_recovery, Wdr, Ind_test):
    labels = Wdr.ravel(order="F")[Ind_test]
    scores = M_recovery.ravel(order="F")[Ind_test]
    tbScalar, _tbVec, AUC, AUPR, Acc, Sen, Spe, Pre = getPerfMetricROCcompute(
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


def summarize_fold_results(df):
    metric_cols = ["AUROC", "AUPR", "Acc", "Sen", "Spe", "Pre",
                   "P@10", "P@20", "R@10", "R@20",
                   "Hits@10", "Hits@20", "iter_num"]
    summary = {}
    for col in metric_cols:
        if col in df.columns:
            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_std"] = df[col].std(ddof=1) if len(df) > 1 else 0.0
    if "total_bnnr_iter" in df.columns:
        summary["total_bnnr_iter_mean"] = df["total_bnnr_iter"].mean()
    return summary


def load_completed_folds(csv_path):
    if not os.path.exists(csv_path):
        return [], set()
    df = pd.read_csv(csv_path)
    return df.to_dict("records"), set(df["fold_id"].unique())


# ── Per-fold experiment functions ────────────────────────────────────────────

def run_E0_BNNR(Wrr, Wdd, Wdr, matDR):
    T, trIndex = build_augmented_matrix(Wrr, Wdd, matDR)
    WW, iter_num = BNNR(alpha=ALPHA, beta=BETA, T=T, trIndex=trIndex,
                         tol1=TOL1, tol2=TOL2, maxiter=MAXITER,
                         a=A_BOUND, b=B_BOUND)
    return extract_recovery_block(WW, Wdd, Wdr), iter_num, {}


def run_E1_GBNNR(Wrr, Wdd, Wdr, matDR):
    T, trIndex = build_augmented_matrix(Wrr, Wdd, matDR)
    WW, iter_num, _info = GBNNR(
        alpha=ALPHA, beta=BETA, T=T, trIndex=trIndex,
        tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
        Wrr_orig=Wrr, Wdd_orig=Wdd, n_drug=Wrr.shape[0],
        knn_k=GRAPH_CFG["knn_k"], gamma_graph=GRAPH_CFG["gamma"],
        lambda_r=GRAPH_CFG["lambda_r"], lambda_d=GRAPH_CFG["lambda_d"],
        lambda_diag_factor=GRAPH_CFG["lambda_diag_factor"],
        inner_steps=GRAPH_CFG["inner_steps"], lr=GRAPH_CFG["lr"], verbose=0)
    return extract_recovery_block(WW, Wdd, Wdr), iter_num, {}


def run_E2_GF_BNNR(Wrr, Wdd, Wdr, matDR):
    M_filtered, M_bnnr, iter_num = GF_BNNR(
        Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
        tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
        gamma_gip=BADGE_CFG["gamma_gip"], w_gip=BADGE_CFG["w_gip"],
        graph_alpha=BADGE_CFG["graph_alpha"])
    return M_filtered, iter_num, {}


def run_BADGE_n(Wrr, Wdd, Wdr, matDR, n_iter):
    M_final, history = BADGE(
        Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
        graph_alpha=BADGE_CFG["graph_alpha"],
        gamma_gip=BADGE_CFG["gamma_gip"], w_gip=BADGE_CFG["w_gip"],
        n_iter=n_iter,
        tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
        verbose=0)
    total_bnnr_iter = sum(h["bnnr_iter"] for h in history)
    extra = {
        "total_bnnr_iter": total_bnnr_iter,
        "n_iter": n_iter,
        "density": history[0]["density"],
        "n_completed": len(history),
    }
    return M_final, total_bnnr_iter, extra


def run_ABL_noGIP(Wrr, Wdd, Wdr, matDR):
    """A1: Ablate GIP fusion. w_gip=0, pure raw similarities."""
    M_final, history = BADGE(
        Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
        graph_alpha=BADGE_CFG["graph_alpha"],
        gamma_gip=BADGE_CFG["gamma_gip"], w_gip=0.0,
        n_iter=2,
        tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
        verbose=0)
    total_bnnr_iter = sum(h["bnnr_iter"] for h in history)
    extra = {"total_bnnr_iter": total_bnnr_iter, "n_iter": 2,
             "density": history[0]["density"]}
    return M_final, total_bnnr_iter, extra


def run_ABL_noFilter(Wrr, Wdd, Wdr, matDR):
    """A2: Ablate graph filter. graph_alpha=0, no Laplacian smoothing."""
    M_final, history = BADGE(
        Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
        graph_alpha=0.0,
        gamma_gip=BADGE_CFG["gamma_gip"], w_gip=BADGE_CFG["w_gip"],
        n_iter=2,
        tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
        verbose=0)
    total_bnnr_iter = sum(h["bnnr_iter"] for h in history)
    extra = {"total_bnnr_iter": total_bnnr_iter, "n_iter": 2,
             "density": history[0]["density"]}
    return M_final, total_bnnr_iter, extra


# ── Experiment registry ──────────────────────────────────────────────────────
EXPERIMENTS_QUICK = {
    "E0_BNNR":       lambda w: run_E0_BNNR(*w),
    "E2_GF-BNNR":    lambda w: run_E2_GF_BNNR(*w),
    "E3_BADGE_n2":   lambda w: run_BADGE_n(*w, n_iter=2),
    "A1_noGIP":      lambda w: run_ABL_noGIP(*w),
    "A2_noFilter":   lambda w: run_ABL_noFilter(*w),
}

EXPERIMENTS_FULL = {
    "E0_BNNR":       lambda w: run_E0_BNNR(*w),
    "E1_GBNNR":      lambda w: run_E1_GBNNR(*w),
    "E2_GF-BNNR":    lambda w: run_E2_GF_BNNR(*w),
    "E3_BADGE_n2":   lambda w: run_BADGE_n(*w, n_iter=2),
    "E4_BADGE_n3":   lambda w: run_BADGE_n(*w, n_iter=3),
    "A1_noGIP":      lambda w: run_ABL_noGIP(*w),
    "A2_noFilter":   lambda w: run_ABL_noFilter(*w),
}


# ── Main ─────────────────────────────────────────────────────────────────────
def run_experiments(resume=False, quick=False):
    ensure_dir(RESULT_DIR)
    global_start = time.time()
    all_fold_rows = []
    all_summary_rows = []

    datasets = {"Fdataset": DATASETS["Fdataset"]} if quick else DATASETS
    n_folds_to_run = 3 if quick else NFOLD
    exp_registry = EXPERIMENTS_QUICK if quick else EXPERIMENTS_FULL
    exp_list = list(exp_registry.keys())

    for dataset_name, dataset_path in datasets.items():
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)

        dataset_dir = os.path.join(RESULT_DIR, dataset_name)
        ensure_dir(dataset_dir)

        Wrr, Wdd, Wdr = load_dataset(dataset_path)
        dn, dr = Wdr.shape
        n_known = np.count_nonzero(Wdr)
        density = n_known / (dn * dr)
        print(f"Shape: {dr} drugs x {dn} diseases  "
              f"({n_known} known, {density*100:.3f}%)")

        np.random.seed(SEED)
        CVdata = getKfoldCrossValidMatIndSet(
            Wdr, NFOLD, CVTYPE, "Unlabel", SEED)
        IndSet_pos_test = CVdata["MatIndSet_pos_test"]
        IndSet_neg_test = CVdata["MatIndSet_neg_test"]

        for exp_name in exp_list:
            exp_func = exp_registry[exp_name]
            csv_path = os.path.join(dataset_dir, f"{exp_name}_fold_results.csv")
            completed_fold_rows, completed_fold_ids = (
                load_completed_folds(csv_path) if resume else ([], set()))

            if resume and completed_fold_ids:
                n_done = len(completed_fold_ids)
                print(f"\n  Resuming: {exp_name} ({n_done}/{n_folds_to_run} folds done)")
            else:
                print(f"\n  Running: {exp_name}")

            fold_rows = list(completed_fold_rows)
            if completed_fold_rows:
                all_fold_rows.extend(completed_fold_rows)

            tic = time.time()

            for i_fold in range(n_folds_to_run):
                fold_id = i_fold + 1
                if resume and fold_id in completed_fold_ids:
                    print(f"    Fold {fold_id:02d}: [SKIPPED]")
                    continue

                Ind_pos_test = IndSet_pos_test[i_fold]
                Ind_neg_test = IndSet_neg_test[i_fold]
                Ind_test = np.union1d(Ind_pos_test, Ind_neg_test)

                matDR = mask_test_entries(Wdr, Ind_test)

                M_recovery, iter_num, extra = exp_func(
                    (Wrr, Wdd, Wdr, matDR))

                eval_result = evaluate_fold(M_recovery, Wdr, Ind_test)
                eval_result["iter_num"] = int(iter_num) if iter_num else 0

                row = {
                    "dataset": dataset_name,
                    "fold_id": fold_id,
                    "experiment": exp_name,
                    "density": density,
                    **eval_result,
                }
                if extra:
                    for k, v in extra.items():
                        if isinstance(v, (int, float, str, bool)):
                            row[k] = v
                fold_rows.append(row)
                all_fold_rows.append(row)

                pd.DataFrame(fold_rows).to_csv(
                    csv_path, index=False, encoding="utf-8-sig")

                elapsed = time.time() - tic
                extra_str = ""
                if "n_completed" in row:
                    extra_str += f" n_iter={row['n_completed']}/{row['n_iter']}"
                print(f"    Fold {fold_id:02d}: AUROC={row['AUROC']:.4f}, "
                      f"AUPR={row['AUPR']:.4f}, "
                      f"P@10={row['P@10']:.4f}, "
                      f"iter={row['iter_num']}{extra_str}  "
                      f"[{elapsed:.0f}s]")

            elapsed_min = (time.time() - tic) / 60.0
            fold_df = pd.DataFrame(fold_rows)
            summary = summarize_fold_results(fold_df)

            summary_row = {
                "dataset": dataset_name, "experiment": exp_name,
                "time_min": elapsed_min, **summary}
            all_summary_rows.append(summary_row)

            print(f"    Summary: AUROC={summary['AUROC_mean']:.4f}"
                  f"+/-{summary['AUROC_std']:.4f}, "
                  f"AUPR={summary['AUPR_mean']:.4f}"
                  f"+/-{summary['AUPR_std']:.4f}")

            fold_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            with open(os.path.join(dataset_dir, f"{exp_name}_summary.json"),
                      "w", encoding="utf-8") as f:
                json.dump(summary_row, f, ensure_ascii=False, indent=2)

    # ── Save global results ──
    all_fold_df = pd.DataFrame(all_fold_rows)
    all_summary_df = pd.DataFrame(all_summary_rows)
    all_fold_df.to_csv(os.path.join(RESULT_DIR, "all_fold_results.csv"),
                       index=False, encoding="utf-8-sig")
    all_summary_df.to_csv(os.path.join(RESULT_DIR, "all_summary_results.csv"),
                          index=False, encoding="utf-8-sig")

    # ── Print comparison table ──
    print("\n" + "=" * 80)
    print("BADGE EXPERIMENT RESULTS")
    print("=" * 80)
    for ds_name in datasets:
        ds_rows = all_summary_df[all_summary_df["dataset"] == ds_name]
        if ds_rows.empty:
            continue
        print(f"\n{ds_name}:")
        header = (f"{'Experiment':<16s}  {'AUROC':>8s} {'+/-':>6s}  "
                  f"{'AUPR':>8s} {'+/-':>6s}  {'Time(min)':>9s}")
        print(header)
        print("-" * len(header))
        for _, r in ds_rows.iterrows():
            print(f"{r['experiment']:<16s}  {r['AUROC_mean']:8.4f} {r['AUROC_std']:6.4f}  "
                  f"{r['AUPR_mean']:8.4f} {r['AUPR_std']:6.4f}  {r['time_min']:9.1f}")

    total_time = (time.time() - global_start) / 60.0
    print(f"\nTotal time: {total_time:.1f} min")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BADGE Experiment Suite")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed fold")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: Fdataset only, 3 folds")
    args = parser.parse_args()
    run_experiments(resume=args.resume, quick=args.quick)
