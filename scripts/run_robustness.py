"""Mask-fraction (sparsity) robustness sweep for the unified GMC config.

Reports GMC AUPR under the CVa protocol with the held-out fraction varied:
    0.05 / 0.10 / 0.20 / 0.30
The 0.10 case is the reported protocol itself (Results/folds/folds_<ds>.mat,
SEED=12345); the others use Results/folds/maskfolds_<frac>_<ds>.mat written by
scripts/gen_mask_folds.py.  Everything runs the SAME unified config
(scripts/run_gmc.py UNIFIED_CONFIG) — no per-fraction tuning — so the curve
isolates the model's sensitivity to sparsity.

AUPR is the primary metric; the sweep is evaluated per fold exactly like the
reported protocol: mask the held-out positives (negatives are already 0), run
gmc_predict, score against test_idx.

Usage:
    python scripts/gen_mask_folds.py --fractions 0.05 0.20 0.30   # once
    python scripts/run_robustness.py --datasets Fdataset Cdataset --quick

Writes Results/summaries/<ds>_robust_mask_summary.csv (columns: frac, AUPR,
AUROC, AUPR_std) and a fold-level CSV <ds>_robust_mask_fold_results.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import gmc_predict, evaluate_fold, mask_test_entries, load_sim_lists
from gmc.helpers import FOLD_DIR, RESULT_DIR
from run_gmc import UNIFIED_CONFIG  # noqa: E402

NFOLD = 10
FRACTIONS = [0.05, 0.10, 0.20, 0.30]


def _params():
    return {k: v for k, v in UNIFIED_CONFIG.items() if k != "tag"}


def run_dataset(ds, fractions, quick=False, folds=None):
    drug_sims, dis_sims = load_sim_lists(ds)
    Wrr = np.mean(drug_sims, axis=0)
    Wdd = np.mean(dis_sims, axis=0)
    params = _params()
    nfold = 2 if quick else NFOLD
    lo, hi = (folds if folds is not None else (1, nfold))
    fold_csv = os.path.join(RESULT_DIR, f"{ds}_robust_mask_fold_results.csv")
    rows = []
    have = set()
    if os.path.exists(fold_csv):
        old = pd.read_csv(fold_csv)
        have = set(zip(old["frac"], old["fold"]))
        rows = old.to_dict("records")
    else:
        old = None

    for frac in fractions:
        if frac == 0.10:
            # The 10% row IS the reported protocol — reuse the existing
            # gmc_unified test-fold results (exactly the main-table numbers)
            # instead of re-running. This guarantees the sanity check.
            old_u = pd.read_csv(os.path.join(RESULT_DIR, f"{ds}_gmc_unified_fold_results.csv"))
            for _, r in old_u.iterrows():
                fld = int(r["fold"])
                if fld < lo or fld > hi:
                    continue
                if (0.10, fld) in have:
                    continue
                rr = {k: r[k] for k in ("AUROC", "AUPR", "Acc", "Sen", "Spe", "Pre",
                                        "P@10", "R@10", "Hits@10", "P@20", "R@20", "Hits@20")}
                rr["fold"] = fld; rr["frac"] = 0.10
                rows.append(rr)
            print(f"  {ds} frac=0.10: reused gmc_unified test-fold results")
            continue
        fd = sio.loadmat(os.path.join(FOLD_DIR, f"maskfolds_{frac}_{ds}.mat"))
        Wdr = fd["Wdr"].astype(np.float64)
        pos_test_idx = fd["pos_test_idx"]
        test_idx = fd["test_idx"]
        for f in range(lo, hi + 1):
            if (frac, f) in have:
                continue
            ind = test_idx[f - 1]; ind = ind[ind >= 0].astype(int)
            p_idx = pos_test_idx[f - 1]; p_idx = p_idx[p_idx >= 0].astype(int)
            masked = mask_test_entries(Wdr, p_idx)
            M_pred = gmc_predict(masked, Wrr, Wdd, drug_sims, dis_sims,
                                 seed=1, **params)
            res = evaluate_fold(M_pred, Wdr, ind)
            res["fold"] = f
            res["frac"] = frac
            rows.append(res)
            print(f"  {ds} frac={frac} fold{f:02d}: AUROC={res['AUROC']:.4f} "
                  f"AUPR={res['AUPR']:.4f}", flush=True)

    ddf = pd.DataFrame(rows)
    if old is not None:
        ddf = pd.concat([old, ddf], ignore_index=True).drop_duplicates(
            ["frac", "fold"]).reset_index(drop=True)
    ddf.to_csv(fold_csv, index=False)
    summ = (ddf.groupby("frac").agg(AUPR=("AUPR", "mean"), AUROC=("AUROC", "mean"),
                                    AUPR_std=("AUPR", "std"))
            .reindex(fractions).reset_index())
    summ.to_csv(os.path.join(RESULT_DIR, f"{ds}_robust_mask_summary.csv"), index=False)
    print(f"\n== {ds} robustness (mask fraction sweep) ==")
    for _, r in summ.iterrows():
        print(f"  frac={r['frac']:.2f}  AUPR={r['AUPR']:.4f} ± {r['AUPR_std']:.4f}  "
              f"AUROC={r['AUROC']:.4f}")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Fdataset"])
    ap.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    ap.add_argument("--quick", action="store_true", help="2 folds only (smoke test)")
    args = ap.parse_args()
    for ds in args.datasets:
        run_dataset(ds, args.fractions, quick=args.quick)
