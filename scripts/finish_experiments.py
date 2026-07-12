"""
Complete remaining experiments:
  Cdataset: E4_BADGE_n3 folds 8-10 (resume)
  DNdataset: re-run E3, E4, A1, A2 with current badge.py (fallback path)
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bnnr import (getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   BADGE, compute_topk_metrics,
                   ensure_dir, load_dataset, mask_test_entries)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE, "data")
RESULT_DIR = os.path.join(BASE, "Results", "BADGE")

SEED = 12345; NFOLD = 10
ALPHA = 1; BETA = 10; TOL1 = 2e-3; TOL2 = 1e-5; MAXITER = 300
A_BOUND = 0; B_BOUND = 1
BADGE_CFG = {"graph_alpha": 0.5, "w_gip": 0.3, "gamma_gip": 1.0}


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


def run_one_fold(Wrr, Wdd, Wdr, matDR, n_iter, graph_alpha, w_gip):
    M_final, history = BADGE(
        Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
        graph_alpha=graph_alpha,
        gamma_gip=BADGE_CFG["gamma_gip"], w_gip=w_gip,
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


def summarize(df):
    cols = ["AUROC","AUPR","Acc","Sen","Spe","Pre","P@10","P@20","R@10","R@20","Hits@10","Hits@20","iter_num"]
    s = {}
    for c in cols:
        if c in df.columns:
            s[f"{c}_mean"] = df[c].mean()
            s[f"{c}_std"] = df[c].std(ddof=1) if len(df) > 1 else 0.0
    return s


def run_experiment(dataset_name, dataset_path, exp_name, n_iter, graph_alpha, w_gip,
                   resume_csv=None):
    dataset_dir = os.path.join(RESULT_DIR, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, f"{exp_name}_fold_results.csv")

    # Resume logic
    completed_folds = set()
    if resume_csv and os.path.exists(resume_csv):
        existing = pd.read_csv(resume_csv)
        completed_folds = set(existing["fold_id"].unique())
        # Also load existing rows
        fold_rows = existing.to_dict("records")
    else:
        fold_rows = []

    Wrr, Wdd, Wdr = load_dataset(dataset_path)
    dn, dr = Wdr.shape
    n_known = np.count_nonzero(Wdr)
    density = n_known / (dn * dr)

    np.random.seed(SEED)
    CVdata = getKfoldCrossValidMatIndSet(Wdr, NFOLD, "CVa", "Unlabel", SEED)
    IndSet_pos_test = CVdata["MatIndSet_pos_test"]
    IndSet_neg_test = CVdata["MatIndSet_neg_test"]

    tic = time.time()
    fold_times = []

    for i_fold in range(NFOLD):
        fold_id = i_fold + 1
        if fold_id in completed_folds:
            print(f"    Fold {fold_id:02d}: [SKIPPED]")
            continue

        fold_tic = time.time()
        Ind_test = np.union1d(IndSet_pos_test[i_fold], IndSet_neg_test[i_fold])
        matDR = mask_test_entries(Wdr, Ind_test)

        M_recovery, iter_num, extra = run_one_fold(
            Wrr, Wdd, Wdr, matDR, n_iter, graph_alpha, w_gip)
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

        # Save incrementally
        pd.DataFrame(fold_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    elapsed_min = (time.time() - tic) / 60.0
    fold_df = pd.DataFrame(fold_rows)
    summary = summarize(fold_df)
    summary_row = {"dataset": dataset_name, "experiment": exp_name,
                   "time_min": elapsed_min, **summary}

    print(f"    ── {exp_name} ──")
    print(f"    AUROC={summary['AUROC_mean']:.4f}+/-{summary['AUROC_std']:.4f}  "
          f"AUPR={summary['AUPR_mean']:.4f}+/-{summary['AUPR_std']:.4f}  "
          f"Time={elapsed_min:.1f}min")

    fold_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(os.path.join(dataset_dir, f"{exp_name}_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary_row, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # ── 1. Cdataset E4 (resume: folds 8-10 missing) ──
    print("=" * 60)
    print("  Cdataset E4_BADGE_n3 (resume folds 8-10)")
    print("=" * 60)
    run_experiment(
        "Cdataset",
        os.path.join(DATASET_DIR, "Cdataset.mat"),
        "E4_BADGE_n3",
        n_iter=3,
        graph_alpha=BADGE_CFG["graph_alpha"],
        w_gip=BADGE_CFG["w_gip"],
        resume_csv=os.path.join(RESULT_DIR, "Cdataset", "E4_BADGE_n3_fold_results.csv"),
    )

    # ── 2. DNdataset: re-run all BADGE experiments with current code ──
    dn_path = os.path.join(DATASET_DIR, "DNdataset.mat")
    dn_exps = [
        ("E3_BADGE_n2", 2, BADGE_CFG["graph_alpha"], BADGE_CFG["w_gip"]),
        ("E4_BADGE_n3", 3, BADGE_CFG["graph_alpha"], BADGE_CFG["w_gip"]),
        ("A1_noGIP",    2, BADGE_CFG["graph_alpha"], 0.0),
        ("A2_noFilter", 2, 0.0, BADGE_CFG["w_gip"]),
    ]

    for exp_name, n_iter, ga, wg in dn_exps:
        print()
        print("=" * 60)
        print(f"  DNdataset {exp_name} (fresh run with fallback)")
        print("=" * 60)
        run_experiment("DNdataset", dn_path, exp_name,
                       n_iter=n_iter, graph_alpha=ga, w_gip=wg)
