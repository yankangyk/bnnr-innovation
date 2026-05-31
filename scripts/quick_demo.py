"""
BNNR Quick Demo — run a single method on Fdataset.

Usage:
    python scripts/quick_demo.py                  # BNNR (default)
    python scripts/quick_demo.py --method bnnr     # same
    python scripts/quick_demo.py --method ra-bnnr  # RA-BNNR (adaptive beta)
    python scripts/quick_demo.py --method gip      # BNNR + GIP fusion
    python scripts/quick_demo.py --method gf       # GF-BNNR (graph-filtered)
"""
import argparse, os, time
import numpy as np
import scipy.io as sio
from bnnr import (BNNR, BNNR_adaptive, GF_BNNR, getGIPSim,
                  getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                  compute_topk_metrics)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Fdataset.mat")

# Shared config
SEED = 12345
NFOLD = 10
ALPHA, BETA = 1, 10
TOL1, TOL2 = 2e-3, 1e-5
MAXITER, A, B = 300, 0, 1


def load_data():
    data = sio.loadmat(DATA_PATH)
    return (data["drug"].astype(np.float64),
            data["disease"].astype(np.float64),
            data["didr"].astype(np.float64))


def get_cv_folds(Wdr, nfold=NFOLD, cvtype="CVa"):
    np.random.seed(SEED + 1)
    CVdata = getKfoldCrossValidMatIndSet(Wdr, nfold, cvtype, "Unlabel", SEED + 1)
    return CVdata["MatIndSet_pos_test"], CVdata["MatIndSet_neg_test"]


def mask_test(Wdr, ind_test):
    mr = Wdr.ravel(order="F").copy()
    mr[ind_test] = 0
    return mr.reshape(Wdr.shape, order="F")


def evaluate(scores, labels):
    _, _, auc, aupr, _, _, _, _ = getPerfMetricROCcompute(scores, labels, 1, 0)
    topk = compute_topk_metrics(scores, labels, ks=(10, 20))
    return auc, aupr, topk


def run_bnnr(Wrr, Wdd, Wdr, pos_test, neg_test, nfold=NFOLD):
    results = []
    for fold in range(nfold):
        ind_test = np.union1d(pos_test[fold], neg_test[fold])
        matDR = mask_test(Wdr, ind_test)
        T = np.block([[Wrr, matDR.T], [matDR, Wdd]])
        trIndex = (T != 0).astype(np.float64)
        WW, it = BNNR(ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, A, B)
        M = WW[-Wdd.shape[0]:, :Wdr.shape[1]]
        labels = Wdr.ravel(order="F")[ind_test]
        scores = M.ravel(order="F")[ind_test]
        auc, aupr, topk = evaluate(scores, labels)
        results.append({"AUROC": auc, "AUPR": aupr, "iter": it, **topk})
        print(f"  Fold {fold+1:02d}: AUROC={auc:.4f}, AUPR={aupr:.4f}, "
              f"P@10={topk['P@10']:.4f}, iter={it}")
    return results


def run_ra_bnnr(Wrr, Wdd, Wdr, pos_test, neg_test, nfold=NFOLD):
    results = []
    ra = {"eta": 0.020, "beta_min": 1.0, "beta_max": 50.0,
          "warmup": 5, "target_rank_ratio": 0.95}
    for fold in range(nfold):
        ind_test = np.union1d(pos_test[fold], neg_test[fold])
        matDR = mask_test(Wdr, ind_test)
        T = np.block([[Wrr, matDR.T], [matDR, Wdd]])
        trIndex = (T != 0).astype(np.float64)
        WW, it, info = BNNR_adaptive(
            ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, A, B,
            eta=ra["eta"], beta_min=ra["beta_min"], beta_max=ra["beta_max"],
            warmup=ra["warmup"], target_rank_ratio=ra["target_rank_ratio"])
        M = WW[-Wdd.shape[0]:, :Wdr.shape[1]]
        labels = Wdr.ravel(order="F")[ind_test]
        scores = M.ravel(order="F")[ind_test]
        auc, aupr, topk = evaluate(scores, labels)
        results.append({"AUROC": auc, "AUPR": aupr, "iter": it, **topk})
        print(f"  Fold {fold+1:02d}: AUROC={auc:.4f}, AUPR={aupr:.4f}, "
              f"P@10={topk['P@10']:.4f}, iter={it}")
    return results


def run_gip(Wrr, Wdd, Wdr, pos_test, neg_test, nfold=NFOLD, w_gip=0.3, gamma_gip=1.0):
    results = []
    for fold in range(nfold):
        ind_test = np.union1d(pos_test[fold], neg_test[fold])
        matDR = mask_test(Wdr, ind_test)
        G_dis, G_drug = getGIPSim(matDR, gamma_gip, gamma_gip, 0, 0)
        S_drug = w_gip * G_drug + (1 - w_gip) * Wrr
        S_dis = w_gip * G_dis + (1 - w_gip) * Wdd
        T = np.block([[S_drug, matDR.T], [matDR, S_dis]])
        trIndex = (T != 0).astype(np.float64)
        WW, it = BNNR(ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, A, B)
        M = WW[-Wdd.shape[0]:, :Wdr.shape[1]]
        labels = Wdr.ravel(order="F")[ind_test]
        scores = M.ravel(order="F")[ind_test]
        auc, aupr, topk = evaluate(scores, labels)
        results.append({"AUROC": auc, "AUPR": aupr, "iter": it, **topk})
        print(f"  Fold {fold+1:02d}: AUROC={auc:.4f}, AUPR={aupr:.4f}, "
              f"P@10={topk['P@10']:.4f}, iter={it}")
    return results


def run_gf(Wrr, Wdd, Wdr, pos_test, neg_test, nfold=NFOLD, w_gip=0.3):
    results = []
    for fold in range(nfold):
        ind_test = np.union1d(pos_test[fold], neg_test[fold])
        matDR = mask_test(Wdr, ind_test)
        # Pre-compute GIP similarities
        G_dis, G_drug = getGIPSim(matDR, 1, 1, 0, 0)
        S_drug = w_gip * G_drug + (1 - w_gip) * Wrr
        S_dis = w_gip * G_dis + (1 - w_gip) * Wdd
        M_gf, _, it = GF_BNNR(Wrr, Wdd, matDR,
                               alpha=ALPHA, beta=BETA,
                               tol1=TOL1, tol2=TOL2,
                               maxiter=MAXITER, a=A, b=B,
                               gamma_gip=1, w_gip=w_gip,
                               graph_alpha=0.5,
                               S_drug=S_drug, S_dis=S_dis)
        labels = Wdr.ravel(order="F")[ind_test]
        scores = M_gf.ravel(order="F")[ind_test]
        auc, aupr, topk = evaluate(scores, labels)
        results.append({"AUROC": auc, "AUPR": aupr, "iter": it, **topk})
        print(f"  Fold {fold+1:02d}: AUROC={auc:.4f}, AUPR={aupr:.4f}, "
              f"P@10={topk['P@10']:.4f}, iter={it}")
    return results


def main():
    parser = argparse.ArgumentParser(description="BNNR quick demo on Fdataset")
    parser.add_argument("--method", default="bnnr",
                        choices=["bnnr", "ra-bnnr", "gip", "gf"],
                        help="Method to run (default: bnnr)")
    parser.add_argument("--folds", type=int, default=NFOLD,
                        help="Number of CV folds (default: 10)")
    parser.add_argument("--gip-weight", type=float, default=0.3,
                        help="GIP fusion weight for gip/gf methods (default: 0.3)")
    args = parser.parse_args()
    nfold = args.folds

    start = time.time()
    print("=" * 70)
    print(f"BNNR Quick Demo: {args.method.upper()} on Fdataset ({nfold}-fold CVa)")
    print("=" * 70)

    Wrr, Wdd, Wdr = load_data()
    dn, dr = Wdr.shape
    print(f"Dataset: {dr} drugs x {dn} diseases ({np.count_nonzero(Wdr)} known)")

    pos_test, neg_test = get_cv_folds(Wdr, nfold=nfold)

    dispatch = {
        "bnnr": run_bnnr,
        "ra-bnnr": run_ra_bnnr,
        "gip": run_gip,
        "gf": run_gf,
    }

    results = dispatch[args.method](Wrr, Wdd, Wdr, pos_test, neg_test, nfold=nfold)

    aurocs = [r["AUROC"] for r in results]
    auprs = [r["AUPR"] for r in results]
    p10s = [r["P@10"] for r in results]

    print(f"\n{'─' * 50}")
    print(f"Mean AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs, ddof=1):.4f}")
    print(f"Mean AUPR:  {np.mean(auprs):.4f} +/- {np.std(auprs, ddof=1):.4f}")
    print(f"Mean P@10:  {np.mean(p10s):.4f}")
    print(f"Total time: {(time.time() - start) / 60:.2f} min")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
