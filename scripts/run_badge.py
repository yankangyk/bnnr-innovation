"""
GRMC Experiment — Graph-Regularized Matrix Completion
========================================================================

Compares GRMC against BNNR on 3 benchmark datasets under disease-centric
cross-validation (CVc). Includes filter strength (graph_alpha) ablation.

Experiments:
  E0  BNNR              baseline (raw similarities, no graph regularization)
  E1  GRMC α=0.5        single-pass graph-regularized completion
  E2  GRMC α=0.7        PROPOSED — stronger filter, improved default

Ablation:
  A1  α=0.1             weak filter
  A2  α=0.3             moderate filter
  A3  α=0.7             strong filter
  A4  α=0.0             no filter (≈ BNNR identity check)

Usage:
    python scripts/run_badge.py                  # benchmark: BNNR + GRMC
    python scripts/run_badge.py --quick           # Fdataset only, 3 folds
    python scripts/run_badge.py --experiments full  # benchmark + ablation
    python scripts/run_badge.py --datasets DNdataset  # specific dataset only
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
                   BNNR, GRMC,
                   compute_topk_metrics,
                   ensure_dir, load_dataset,
                   build_augmented_matrix, mask_test_entries)

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(BASE_DIR, "data")

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 12345
NFOLD = 10

ALPHA = 1
BETA = 10
TOL1 = 2e-3
TOL2 = 1e-5
MAXITER = 300
A_BOUND = 0
B_BOUND = 1

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
    """BNNR baseline — raw similarities, no graph regularization."""
    T, trIndex = build_augmented_matrix(Wrr, Wdd, matDR)
    WW, iter_num = BNNR(alpha=ALPHA, beta=BETA, T=T, trIndex=trIndex,
                         tol1=TOL1, tol2=TOL2, maxiter=MAXITER,
                         a=A_BOUND, b=B_BOUND)
    return extract_recovery_block(WW, Wdd, Wdr), iter_num, {}


def run_proposed(Wrr, Wdd, Wdr, matDR, graph_alpha=0.7, knn_k=None):
    """GRMC — single-pass graph-regularized matrix completion."""
    M_recovery, history = GRMC(Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
                               graph_alpha=graph_alpha,
                               knn_k=knn_k,
                               tol1=TOL1, tol2=TOL2, maxiter=MAXITER,
                               a=A_BOUND, b=B_BOUND)
    iter_num = history[0]["bnnr_iter"] if history else 0
    extra = {"graph_alpha": graph_alpha, "knn_k": history[0].get("knn_k", 0)}
    return M_recovery, iter_num, extra


# ── Experiment registries ────────────────────────────────────────────────────
EXPERIMENTS_BENCHMARK = {
    "E0_BNNR":       lambda w: run_E0_BNNR(*w),
    "E1_GRMC_a50":   lambda w: run_proposed(*w, graph_alpha=0.5, knn_k=0),
    "E2_GRMC_a70":   lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=0),
    "E3_GRMC":       lambda w: run_proposed(*w, graph_alpha=0.7),  # adaptive KNN
}

EXPERIMENTS_QUICK = {
    "E0_BNNR":       lambda w: run_E0_BNNR(*w),
    "E1_GRMC_a50":   lambda w: run_proposed(*w, graph_alpha=0.5, knn_k=0),
    "E2_GRMC_a70":   lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=0),
    "E3_GRMC":       lambda w: run_proposed(*w, graph_alpha=0.7),  # adaptive KNN
}

EXPERIMENTS_FULL = {
    "E0_BNNR":       lambda w: run_E0_BNNR(*w),
    "E1_GRMC_a50":   lambda w: run_proposed(*w, graph_alpha=0.5, knn_k=0),
    "E2_GRMC_a70":   lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=0),
    "E3_GRMC":       lambda w: run_proposed(*w, graph_alpha=0.7),  # adaptive KNN
    "A1_alpha_01":   lambda w: run_proposed(*w, graph_alpha=0.1, knn_k=0),
    "A2_alpha_03":   lambda w: run_proposed(*w, graph_alpha=0.3, knn_k=0),
    "A3_alpha_07":   lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=0),
    "A4_alpha_00":   lambda w: run_proposed(*w, graph_alpha=0.0, knn_k=0),
}

EXPERIMENTS_ABLATION = {
    "A1_alpha_01":   lambda w: run_proposed(*w, graph_alpha=0.1, knn_k=0),
    "A2_alpha_03":   lambda w: run_proposed(*w, graph_alpha=0.3, knn_k=0),
    "A3_alpha_07":   lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=0),
    "A4_alpha_00":   lambda w: run_proposed(*w, graph_alpha=0.0, knn_k=0),
}

EXPERIMENTS_KNN = {
    "E0_BNNR":       lambda w: run_E0_BNNR(*w),
    "E2_GRMC_a70":   lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=0),
    "K0_KNN_k05_a00": lambda w: run_proposed(*w, graph_alpha=0.0, knn_k=5),
    "K1_KNN_k05_a70": lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=5),
    "K2_KNN_k10_a70": lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=10),
    "K3_KNN_k20_a70": lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=20),
    "K4_KNN_k50_a70": lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=50),
    "K5_KNN_k100_a70": lambda w: run_proposed(*w, graph_alpha=0.7, knn_k=100),
}


# ── Main ─────────────────────────────────────────────────────────────────────
def run_experiments(resume=False, quick=False, cvtype="CVc",
                    experiments="benchmark", datasets_filter=None):
    result_dir = os.path.join(BASE_DIR, "Results",
                              f"GRMC_{cvtype}")
    ensure_dir(result_dir)
    global_start = time.time()
    all_fold_rows = []
    all_summary_rows = []

    if experiments == "benchmark":
        exp_registry = EXPERIMENTS_BENCHMARK
    elif experiments == "quick":
        exp_registry = EXPERIMENTS_QUICK
    elif experiments == "ablation":
        exp_registry = EXPERIMENTS_ABLATION
    elif experiments == "knn":
        exp_registry = EXPERIMENTS_KNN
    else:
        exp_registry = EXPERIMENTS_FULL

    if quick:
        datasets = {"Fdataset": DATASETS["Fdataset"]}
    elif datasets_filter:
        datasets = {k: DATASETS[k] for k in datasets_filter if k in DATASETS}
    else:
        datasets = DATASETS
    n_folds_to_run = 3 if quick else NFOLD
    exp_list = list(exp_registry.keys())

    neg_type = "Unlabel" if cvtype == "CVa" else None

    for dataset_name, dataset_path in datasets.items():
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)

        dataset_dir = os.path.join(result_dir, dataset_name)
        ensure_dir(dataset_dir)

        Wrr, Wdd, Wdr = load_dataset(dataset_path)
        dn, dr = Wdr.shape
        n_known = np.count_nonzero(Wdr)
        density = n_known / (dn * dr)
        print(f"Shape: {dr} drugs x {dn} diseases  "
              f"({n_known} known, {density*100:.3f}%)")

        np.random.seed(SEED)
        CVdata = getKfoldCrossValidMatIndSet(
            Wdr, NFOLD, cvtype, neg_type, SEED)
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
                if "graph_alpha" in row:
                    extra_str += f" α={row['graph_alpha']}"
                if row.get("knn_k", 0) > 0:
                    extra_str += f" k={row['knn_k']}"
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
    all_fold_df.to_csv(os.path.join(result_dir, "all_fold_results.csv"),
                       index=False, encoding="utf-8-sig")
    all_summary_df.to_csv(os.path.join(result_dir, "all_summary_results.csv"),
                          index=False, encoding="utf-8-sig")

    # ── Print comparison table ──
    print("\n" + "=" * 80)
    print("GRMC EXPERIMENT RESULTS")
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
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRMC Experiment Suite")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed fold")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: Fdataset only, 3 folds")
    parser.add_argument("--cvtype", type=str, default="CVc",
                        choices=["CVa", "CVc"],
                        help="CV protocol: CVa (random pair) or CVc (disease-centric)")
    parser.add_argument("--experiments", type=str, default="benchmark",
                        choices=["benchmark", "quick", "full", "ablation", "knn"],
                        help="Experiment set: benchmark (BNNR+GRMC α=0.5,0.7), "
                             "quick (Fdataset, 3-fold), full (benchmark+ablation), "
                             "ablation (filter strength variants), "
                             "knn (KNN graph sparsification sweep)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        choices=["Fdataset", "Cdataset", "DNdataset"],
                        help="Datasets to run (default: all)")
    args = parser.parse_args()
    run_experiments(resume=args.resume, quick=args.quick,
                    cvtype=args.cvtype, experiments=args.experiments,
                    datasets_filter=args.datasets)
