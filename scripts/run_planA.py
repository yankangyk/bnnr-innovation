"""
Run Plan A (embedded graph-regularized ADMM) on all datasets.
Only re-runs BADGE-dependent experiments (E3, E4, A1, A2).
Baselines (E0, E1, E2) are unchanged and skipped.
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bnnr import (getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   BADGE, getGIPSim, compute_topk_metrics,
                   ensure_dir, load_dataset, mask_test_entries)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "Results", "BADGE")

SEED = 12345
NFOLD = 10
CVTYPE = "CVa"

ALPHA = 1; BETA = 10; TOL1 = 2e-3; TOL2 = 1e-5; MAXITER = 300
A_BOUND = 0; B_BOUND = 1

BADGE_CFG = {"graph_alpha": 0.5, "w_gip": 0.3, "gamma_gip": 1.0}

DATASETS = {
    "Fdataset": os.path.join(DATASET_DIR, "Fdataset.mat"),
    "Cdataset": os.path.join(DATASET_DIR, "Cdataset.mat"),
    "DNdataset": os.path.join(DATASET_DIR, "DNdataset.mat"),
}

# Only re-run BADGE-based experiments
EXPERIMENTS = {
    "E3_BADGE_n2":   lambda w: run_BADGE_n(*w, n_iter=2),
    "E4_BADGE_n3":   lambda w: run_BADGE_n(*w, n_iter=3),
    "A1_noGIP":      lambda w: run_ABL_noGIP(*w),
    "A2_noFilter":   lambda w: run_ABL_noFilter(*w),
}


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


def run_experiment_on_dataset(dataset_name, dataset_path):
    dataset_dir = os.path.join(RESULT_DIR, dataset_name)
    ensure_dir(dataset_dir)

    Wrr, Wdd, Wdr = load_dataset(dataset_path)
    dn, dr = Wdr.shape
    n_known = np.count_nonzero(Wdr)
    density = n_known / (dn * dr)

    np.random.seed(SEED)
    CVdata = getKfoldCrossValidMatIndSet(Wdr, NFOLD, CVTYPE, "Unlabel", SEED)
    IndSet_pos_test = CVdata["MatIndSet_pos_test"]
    IndSet_neg_test = CVdata["MatIndSet_neg_test"]

    for exp_name, exp_func in EXPERIMENTS.items():
        csv_path = os.path.join(dataset_dir, f"{exp_name}_fold_results.csv")
        fold_rows = []
        tic = time.time()
        fold_times = []

        for i_fold in range(NFOLD):
            fold_id = i_fold + 1
            fold_tic = time.time()

            Ind_pos_test = IndSet_pos_test[i_fold]
            Ind_neg_test = IndSet_neg_test[i_fold]
            Ind_test = np.union1d(Ind_pos_test, Ind_neg_test)
            matDR = mask_test_entries(Wdr, Ind_test)

            M_recovery, iter_num, extra = exp_func((Wrr, Wdd, Wdr, matDR))
            eval_result = evaluate_fold(M_recovery, Wdr, Ind_test)
            eval_result["iter_num"] = int(iter_num) if iter_num else 0

            row = {
                "dataset": dataset_name, "fold_id": fold_id,
                "experiment": exp_name, "density": density, **eval_result,
            }
            if extra:
                for k, v in extra.items():
                    if isinstance(v, (int, float, str, bool)):
                        row[k] = v
            fold_rows.append(row)

            fold_elapsed = time.time() - fold_tic
            fold_times.append(fold_elapsed)

            extra_str = ""
            if "n_completed" in row:
                extra_str += f" n_iter={row['n_completed']}/{row['n_iter']}"
            print(f"    Fold {fold_id:02d}: AUROC={row['AUROC']:.4f}, "
                  f"AUPR={row['AUPR']:.4f}, "
                  f"P@10={row['P@10']:.4f}, "
                  f"iter={row['iter_num']}{extra_str}  "
                  f"[{fold_elapsed:.0f}s]")

            pd.DataFrame(fold_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

        elapsed_min = (time.time() - tic) / 60.0
        fold_df = pd.DataFrame(fold_rows)
        summary = summarize_fold_results(fold_df)
        summary_row = {"dataset": dataset_name, "experiment": exp_name,
                       "time_min": elapsed_min, **summary}

        print(f"    ── {exp_name} Summary ──")
        print(f"    AUROC={summary['AUROC_mean']:.4f}+/-{summary['AUROC_std']:.4f}  "
              f"AUPR={summary['AUPR_mean']:.4f}+/-{summary['AUPR_std']:.4f}  "
              f"Time={elapsed_min:.1f}min  "
              f"Fold_times: mean={np.mean(fold_times):.0f}s, max={np.max(fold_times):.0f}s")

        with open(os.path.join(dataset_dir, f"{exp_name}_summary.json"),
                  "w", encoding="utf-8") as f:
            json.dump(summary_row, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  {ds_name}")
        print(f"{'='*70}")
        run_experiment_on_dataset(ds_name, ds_path)
