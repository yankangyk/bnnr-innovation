"""Parameter-sensitivity sweep around the unified GMC config (fresh folds).

Model-stability experiment: hold the unified config at its center and perturb
ONE hyperparameter at a time over an EQUIDISTANT 5-point grid, reporting AUPR
on the independent fresh folds (SEED_FRESH=24680, the folds the config was
selected on).  The center point (alpha=0.5, maxiter=40, rc400, k=10,
w_tensor=0.5) should reproduce the known gmc_unified fresh-fold numbers, which
is the sanity check.  All four datasets run all five axes.

Equidistant axes (each value = center with that ONE parameter changed):
    bnnr_alpha      {0.1, 0.3, 0.5, 0.7, 0.9}
    bnnr_maxiter    {20, 30, 40, 50, 60}
    bnnr_rank_cap   {200, 300, 400, 500, 600}
    wknn_k          {5, 10, 15, 20, 25}
    w_tensor        {0.1, 0.3, 0.5, 0.7, 0.9}   (w_bnnr = 1 - w_tensor)

Config names reuse the old convention and extend it (alpha01/03/07/09,
iter20/30/50/60, rc200/300/500/600, k5/15/20/25, wt01/03/07/09; the center of
each axis is the special "center" config), so already-computed (config, fold)
rows are resumed and only the new grid values are run.

Usage:
    python scripts/run_param_sweep.py --datasets Fdataset Cdataset
    python scripts/run_param_sweep.py --datasets CTDdataset2023 --folds 1-4   # chunked resume
    python scripts/run_param_sweep.py --datasets Ydataset --folds 5-7 --quick  # smoke

Writes Results/summaries/<ds>_param_sweep_summary.csv (columns: config, AUPR,
AUROC, AUPR_std) and a fold-level CSV.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_gmc_val import _load_val, _masked_for_val  # noqa: E402
from gmc import gmc_predict, evaluate_fold           # noqa: E402
from gmc.helpers import RESULT_DIR                   # noqa: E402

NFOLD = 10

# ── unified center (the reported gmc_unified config) ──────────────────────
CENTER = dict(fill="knn", block="sym", wknn_k=10, bnnr_alpha=0.5,
              bnnr_maxiter=40, bnnr_rank_cap=400, trindex="observed",
              w_bnnr=0.5, w_tensor=0.5, fusion="rank", seed=1)

# ── one-parameter-at-a-time EQUIDISTANT axes (5 points, center excluded) ────
# Config names reuse the old convention and extend it, so already-computed
# (config, fold) rows resume.  The center of each axis is the special "center"
# config (dict()) = CENTER unchanged.
AXES = {
    "alpha":   [("alpha01", dict(bnnr_alpha=0.1)), ("alpha03", dict(bnnr_alpha=0.3)),
                ("alpha07", dict(bnnr_alpha=0.7)), ("alpha09", dict(bnnr_alpha=0.9))],
    "maxiter": [("iter20", dict(bnnr_maxiter=20)), ("iter30", dict(bnnr_maxiter=30)),
                ("iter50", dict(bnnr_maxiter=50)), ("iter60", dict(bnnr_maxiter=60))],
    "rc":      [("rc200", dict(bnnr_rank_cap=200)), ("rc300", dict(bnnr_rank_cap=300)),
                ("rc500", dict(bnnr_rank_cap=500)), ("rc600", dict(bnnr_rank_cap=600))],
    "k":       [("k5", dict(wknn_k=5)), ("k15", dict(wknn_k=15)),
                ("k20", dict(wknn_k=20)), ("k25", dict(wknn_k=25))],
    "wt":      [("wt01", dict(w_bnnr=0.9, w_tensor=0.1)), ("wt03", dict(w_bnnr=0.7, w_tensor=0.3)),
                ("wt07", dict(w_bnnr=0.3, w_tensor=0.7)), ("wt09", dict(w_bnnr=0.1, w_tensor=0.9))],
}

# all five axes on every dataset (2026-08-16: equidistant grid, all datasets)
DEFAULT_AXES = {ds: ["alpha", "maxiter", "rc", "k", "wt"] for ds in
                ("Fdataset", "Cdataset", "CTDdataset2023", "Ydataset")}


def run(ds, axes=None, folds=(1, NFOLD), quick=False, reuse_center=True):
    lo, hi = folds
    axes = axes or DEFAULT_AXES[ds]
    cands = [("center", dict())]
    for a in axes:
        for name, over in AXES[a]:
            cands.append((name, over))
    nfold = 2 if quick else NFOLD
    csv = os.path.join(RESULT_DIR, f"{ds}_param_sweep_fold_results.csv")
    have = set()
    if os.path.exists(csv):
        old = pd.read_csv(csv)
        have = set(zip(old["config"], old["fold"]))
    else:
        old = None

    # center (= gmc_unified on the fresh folds) is already reported in
    # {ds}_unified_fresh_summary.csv as uni_obs_rc400_t50 — reuse those per-fold
    # rows instead of burning an expensive Y re-run.
    center_rows = []
    center_csv = os.path.join(RESULT_DIR, f"{ds}_unified_fresh_fold_results.csv")
    if reuse_center and os.path.exists(center_csv):
        cu = pd.read_csv(center_csv)
        cu = cu[cu["config"] == "uni_obs_rc400_t50"]
        if len(cu) >= nfold:
            for _, r in cu.iterrows():
                rr = {k: r[k] for k in ("AUROC", "AUPR", "Acc", "Sen", "Spe", "Pre",
                                        "P@10", "R@10", "Hits@10", "P@20", "R@20", "Hits@20")}
                rr["fold"] = int(r["fold"]); rr["config"] = "center"
                center_rows.append(rr)
            print(f"  {ds}: reuse center = uni_obs_rc400_t50 fresh-fold rows ({len(center_rows)})")

    Wdr, pos_test, val_pos, val_idx, Wrr, Wdd, drug_sims, dis_sims = _load_val(ds, fresh=True)
    rows = []
    for f in range(lo, min(hi, nfold) + 1):
        masked = _masked_for_val(Wdr, pos_test[f - 1], val_pos[f - 1])
        ind = val_idx[f - 1]; ind = ind[ind >= 0].astype(int)
        for name, over in cands:
            if (name, f) in have:
                continue
            params = dict(CENTER, **over)
            M = gmc_predict(masked, Wrr, Wdd, drug_sims, dis_sims, **params)
            res = evaluate_fold(M, Wdr, ind)
            res["fold"] = f
            res["config"] = name
            rows.append(res)
        this = {r['config']: r['AUPR'] for r in rows if r['fold'] == f}
        print(f"  {ds} fold{f:02d}: " + " ".join(f"{n}={v:.4f}" for n, v in sorted(this.items())),
              flush=True)

    ddf = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if old is not None else \
        pd.DataFrame(rows)
    ddf = pd.concat([ddf, pd.DataFrame(center_rows)], ignore_index=True)
    ddf = ddf.drop_duplicates(["config", "fold"]).reset_index(drop=True)
    ddf.to_csv(csv, index=False)
    summ = (ddf.groupby("config").agg(AUPR=("AUPR", "mean"), AUROC=("AUROC", "mean"),
                                      AUPR_std=("AUPR", "std"))
            .reindex([c for c, _ in cands]).reset_index())
    summ.to_csv(os.path.join(RESULT_DIR, f"{ds}_param_sweep_summary.csv"), index=False)
    print(f"\n== {ds} parameter sensitivity (fresh folds, center = gmc_unified) ==")
    for _, r in summ.iterrows():
        print(f"  {r['config']:<8s} AUPR={r['AUPR']:.4f} ± {r['AUPR_std']:.4f}  AUROC={r['AUROC']:.4f}")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Fdataset"])
    ap.add_argument("--axes", nargs="+", choices=list(AXES), default=None,
                    help="axes to sweep (default = all five, every dataset)")
    ap.add_argument("--folds", default="1-10", help="inclusive fold range, e.g. 1-3 (resumes)")
    ap.add_argument("--quick", action="store_true", help="2 folds only (smoke test)")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.folds.split("-"))
    for ds in args.datasets:
        run(ds, axes=args.axes, folds=(lo, hi), quick=args.quick)
