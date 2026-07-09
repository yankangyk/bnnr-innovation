"""
w_gip parameter sweep.  5-fold CV on all 3 datasets.
Validates w_gip=0.3 as the default GIP fusion weight.
"""
import os, sys, time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bnnr import (getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   BADGE, compute_topk_metrics, ensure_dir, load_dataset,
                   mask_test_entries)

SEED = 12345; NFOLD = 10; NFOLD_SWEEP = 5; CVTYPE = "CVa"
ALPHA, BETA = 1, 10
TOL1, TOL2 = 2e-3, 1e-5; MAXITER = 300
A_BOUND, B_BOUND = 0, 1

BADGE_CFG = {"graph_alpha": 0.5, "w_gip": 0.3, "gamma_gip": 1.0}

DATASET_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "Results", "BADGE", "sweeps")
DATASETS = {
    "Fdataset": os.path.join(DATASET_DIR, "Fdataset.mat"),
    "Cdataset": os.path.join(DATASET_DIR, "Cdataset.mat"),
    "DNdataset": os.path.join(DATASET_DIR, "DNdataset.mat"),
}

WGIP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


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


def load_completed_folds(csv_path):
    if not os.path.exists(csv_path):
        return [], set()
    df = pd.read_csv(csv_path)
    return df.to_dict("records"), set(df["fold_id"].unique())


def run_fixed_wgip(Wrr, Wdd, Wdr, matDR, wgip):
    M_final, history = BADGE(
        Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
        graph_alpha=BADGE_CFG["graph_alpha"],
        gamma_gip=BADGE_CFG["gamma_gip"], w_gip=wgip,
        n_iter=2,
        tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND, verbose=0)
    total_bnnr_iter = sum(h["bnnr_iter"] for h in history)
    extra = {"total_bnnr_iter": total_bnnr_iter, "n_iter": 2,
             "w_gip": wgip, "density": history[0]["density"]}
    return M_final, total_bnnr_iter, extra


def main():
    ensure_dir(RESULT_DIR)
    global_start = time.time()

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        Wrr, Wdd, Wdr = load_dataset(ds_path)
        dn, dr = Wdr.shape
        density = np.count_nonzero(Wdr) / (dn * dr)
        print(f"Shape: {dr}x{dn}, density={density*100:.3f}%")

        np.random.seed(SEED)
        CVdata = getKfoldCrossValidMatIndSet(Wdr, NFOLD, CVTYPE, "Unlabel", SEED)
        IndSet_pos = CVdata["MatIndSet_pos_test"]
        IndSet_neg = CVdata["MatIndSet_neg_test"]

        ds_dir = os.path.join(RESULT_DIR, ds_name)
        ensure_dir(ds_dir)

        for wgip in WGIP_VALUES:
            exp_name = f"sweep_wgip_{wgip:.1f}"
            csv_path = os.path.join(ds_dir, f"{exp_name}.csv")
            completed_rows, completed_ids = load_completed_folds(csv_path)
            fold_rows = list(completed_rows)

            remaining = [f for f in range(1, NFOLD_SWEEP + 1) if f not in completed_ids]
            if not remaining:
                continue

            print(f"  w_gip={wgip:.1f}: {len(completed_ids)}/{NFOLD_SWEEP} done, "
                  f"running folds {remaining}", flush=True)
            tic = time.time()

            for fold_id in remaining:
                i_fold = fold_id - 1
                Ind_test = np.union1d(IndSet_pos[i_fold], IndSet_neg[i_fold])
                matDR = mask_test_entries(Wdr, Ind_test)

                M_recovery, iter_num, extra = run_fixed_wgip(
                    Wrr, Wdd, Wdr, matDR, wgip)

                eval_result = evaluate_fold(M_recovery, Wdr, Ind_test)
                eval_result["iter_num"] = int(iter_num) if iter_num else 0

                row = {
                    "dataset": ds_name, "fold_id": fold_id,
                    "experiment": exp_name, "sweep_type": "wgip",
                    "param_value": wgip, "density": density, **eval_result,
                }
                for k, v in extra.items():
                    if isinstance(v, (int, float, str, bool)):
                        row[k] = v
                fold_rows.append(row)

                pd.DataFrame(fold_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

                elapsed = time.time() - tic
                print(f"    Fold {fold_id}: AUROC={row['AUROC']:.4f} "
                      f"AUPR={row['AUPR']:.4f} [{elapsed:.0f}s]", flush=True)

            df = pd.DataFrame(fold_rows)
            if len(df) > 1:
                print(f"    -> AUROC={df['AUROC'].mean():.4f}+/-{df['AUROC'].std(ddof=1):.4f}  "
                      f"AUPR={df['AUPR'].mean():.4f}+/-{df['AUPR'].std(ddof=1):.4f}", flush=True)

    total_time = (time.time() - global_start) / 60.0
    print(f"\nTotal time: {total_time:.1f} min")


if __name__ == "__main__":
    main()
