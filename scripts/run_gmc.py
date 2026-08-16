"""Run the GMC model (multi-view low-rank completion + graph filter) on CVa folds.

Since 2026-08-11 GMC is ONE config on all four datasets (tag: gmc_unified):
OMC-style KNN cold-start fill, symmetric block completion, tensor + block rank
fusion. It is a single genuinely-new model — no per-dataset parameter
strengthening (the shared completion core alpha=0.5/i40 is used everywhere).

  unified : fill=knn, block=sym, wknn_k=10, bnnr_alpha=0.5, bnnr_maxiter=40,
            bnnr_rank_cap=400, trindex=observed, w_bnnr=0.5, w_tensor=0.5,
            fusion=rank, no graph, no filter
  AUPR on the reported CVa test folds: F 0.6569 / C 0.7285 / CTD 0.3714 /
            Y 0.7404 — highest on all four (see COMPARISON_summary.csv).

The config is bundled as the DEFAULT_CONFIGS preset and selectable with
--preset <dataset> (explicit CLI flags override the preset). The config is a
structural hyperparameter validated on independent fresh folds (scripts/
run_unified_scope.py), not a data-driven rule; the run prints kappa(Wdd) only
as a diagnostic.

Usage:
    python scripts/run_gmc.py --datasets Fdataset --tag gmc_unified --preset Fdataset

Writes per-fold predictions to Results/outputs/<ds>/<tag>/fold_XX.mat (the same
convention as the MATLAB baselines, so ensemble/significance scripts find them)
and fold + summary CSVs to Results/summaries/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import gmc_predict, evaluate_fold, mask_test_entries, load_sim_lists
from gmc.helpers import FOLD_DIR, OUT_DIR, RESULT_DIR

# GMC configuration, selected on independent fresh validation folds
# (scripts/run_unified_scope.py) and reported in the manuscript. Since the
# 2026-08-11 unification ONE config is used on all four datasets — GMC is a
# single genuinely-new model (shared completion core alpha=0.5/i40, no
# per-dataset strengthening). Structural choices (fill, block, observation
# mask, fusion) are validated on fresh folds (SEED_FRESH=24680) and agree with
# the reported test folds to ±0.001–0.002. A causal check of a
# similarity-block-conditioning rationale for the older Y gain found it does
# not explain that gain (near-null Wdd directions lie beyond the rank cap) —
# do not cite conditioning as a mechanism. See the View A section of
# papers/gmc_manuscript.tex.
UNIFIED_CONFIG = dict(tag="gmc_unified", fill="knn", block="sym", wknn_k=10,
                      bnnr_alpha=0.5, bnnr_maxiter=40, bnnr_rank_cap=400,
                      trindex="observed", w_bnnr=0.5, w_tensor=0.5,
                      fusion="rank")
DEFAULT_CONFIGS = {ds: dict(UNIFIED_CONFIG) for ds in
                   ("Fdataset", "Cdataset", "CTDdataset2023", "Ydataset")}


def run_dataset(dataset, tag="gmc", params=None, quick=False, resume=True):
    fd = sio.loadmat(os.path.join(FOLD_DIR, f"folds_{dataset}.mat"))
    Wdr = fd["Wdr"].astype(np.float64)
    pos_test_idx = fd["pos_test_idx"]
    test_idx = fd["test_idx"]
    nfold = 2 if quick else test_idx.shape[0]
    drug_sims, dis_sims = load_sim_lists(dataset)
    Wrr = np.mean(drug_sims, axis=0)
    Wdd = np.mean(dis_sims, axis=0)

    # Diagnostic only (informational, not a selection rule): conditioning of
    # the disease-similarity block. See DEFAULT_CONFIGS for why this is not a
    # mechanism — the near-null directions lie beyond the rank cap.
    sdd = np.linalg.svd(Wdd, compute_uv=False)
    print(f"  {dataset}: kappa(Wdd)={sdd[0] / sdd[-1]:.2e}  (diagnostic only)")

    params = params or {}
    csv_path = os.path.join(RESULT_DIR, f"{dataset}_{tag}_fold_results.csv")
    rows = []
    done = set()
    if resume and os.path.exists(csv_path):
        ddf = pd.read_csv(csv_path)
        done = set(ddf["fold"].astype(int))
        rows = ddf.to_dict("records")
    outdir = os.path.join(OUT_DIR, dataset, tag)
    os.makedirs(outdir, exist_ok=True)

    for f in range(nfold):
        if f + 1 in done:
            continue
        ind = test_idx[f]; ind = ind[ind >= 0].astype(int)
        p_idx = pos_test_idx[f]; p_idx = p_idx[p_idx >= 0].astype(int)
        masked = mask_test_entries(Wdr, p_idx)
        M_pred = gmc_predict(masked, Wrr, Wdd, drug_sims, dis_sims,
                             seed=1, **params)
        sio.savemat(os.path.join(outdir, f"fold_{f + 1:02d}.mat"),
                    {"M_pred": M_pred})
        res = evaluate_fold(M_pred, Wdr, ind)
        res["fold"] = f + 1
        rows.append(res)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  {dataset} {tag} fold{f + 1:02d}: AUROC={res['AUROC']:.4f} "
              f"AUPR={res['AUPR']:.4f} P@10={res['P@10']:.4f}", flush=True)
    sdf = pd.DataFrame(rows)
    summ = {
        "dataset": dataset, "method": tag, "n_folds": len(sdf),
        "AUROC": float(sdf["AUROC"].mean()), "AUPR": float(sdf["AUPR"].mean()),
        "P@10": float(sdf["P@10"].mean()), "P@20": float(sdf["P@20"].mean()),
        "AUROC_std": float(sdf["AUROC"].std(ddof=1)),
        "AUPR_std": float(sdf["AUPR"].std(ddof=1)),
    }
    pd.DataFrame([summ]).to_csv(
        os.path.join(RESULT_DIR, f"{dataset}_{tag}_summary.csv"), index=False)
    print(f"== {dataset} {tag}: AUROC={summ['AUROC']:.4f} "
          f"AUPR={summ['AUPR']:.4f} P@10={summ['P@10']:.4f}")
    return sdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Fdataset"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="gmc")
    ap.add_argument("--bnnr-alpha", type=float, default=0.5)
    ap.add_argument("--bnnr-beta", type=float, default=10.0)
    ap.add_argument("--bnnr-maxiter", type=int, default=40)
    ap.add_argument("--bnnr-rank-cap", type=int, default=160)
    ap.add_argument("--wknn-k", type=int, default=30)
    ap.add_argument("--wknn-r", type=float, default=0.95)
    ap.add_argument("--w-bnnr", type=float, default=1.0)
    ap.add_argument("--w-graph", type=float, default=0.0,
                    help=">0: View A = SGRMC-style graph-embedded block completion")
    ap.add_argument("--graph-alpha", type=float, default=0.7,
                    help="graph embedding strength gamma for w_graph>0")
    ap.add_argument("--w-grnmf", type=float, default=0.0)
    ap.add_argument("--w-tensor", type=float, default=0.0)
    ap.add_argument("--tensor-rank-cap", type=int, default=None)
    ap.add_argument("--fusion", choices=["raw", "rank"], default="raw")
    ap.add_argument("--filt-alpha", type=float, default=0.0)
    ap.add_argument("--filt-beta", type=float, default=0.5)
    ap.add_argument("--sparsify-k", type=int, default=0)
    ap.add_argument("--coldstart", action="store_true")
    ap.add_argument("--trindex", choices=["all", "observed"], default="all",
                    help="BNNR observation mask: 'observed' constrains only "
                         "non-zero entries (Y: +0.003 AUPR, 9/10 folds); "
                         "'all' = default ones-mask")
    ap.add_argument("--fill", choices=["wknn", "knn"], default="wknn",
                    help="cold-start fill: 'wknn' = WKNN soft labels "
                         "(default, F/C); 'knn' = OMC-style KNN neighbor "
                         "average (Y: lifts AUPR with dual block)")
    ap.add_argument("--block", choices=["sym", "dual"], default="sym",
                    help="View-A block structure: 'sym' = single symmetric "
                         "block (default, F/C/CTD); 'dual' = OMC-style dual "
                         "asymmetric blocks (Y: +0.017 AUPR, 10/10 folds)")
    ap.add_argument("--preset", default=None, metavar="DATASET",
                    help="apply a validated per-dataset GMC config from "
                         "DEFAULT_CONFIGS as defaults (explicit CLI flags "
                         "override it). Datasets: Fdataset Cdataset "
                         "CTDdataset2023 Ydataset")
    args = ap.parse_args()

    param_fields = ("bnnr_alpha", "bnnr_beta", "bnnr_maxiter", "bnnr_rank_cap",
                    "wknn_k", "wknn_r", "w_bnnr", "w_graph", "graph_alpha",
                    "w_grnmf", "w_tensor", "tensor_rank_cap", "fusion",
                    "filt_alpha", "filt_beta", "sparsify_k", "coldstart",
                    "trindex", "fill", "block")
    preset = DEFAULT_CONFIGS.get(args.preset) if args.preset else None
    if preset is None:
        params = {f: getattr(args, f) for f in param_fields}
        tag = args.tag
    else:
        # documented per-dataset config; explicit CLI flags override it
        params = {k: v for k, v in preset.items() if k != "tag"}
        for f in param_fields:
            if getattr(args, f) != ap.get_default(f):
                params[f] = getattr(args, f)
        tag = args.tag if args.tag != "gmc" else preset.get("tag", "gmc")
    for ds in args.datasets:
        run_dataset(ds, tag=tag, params=params, quick=args.quick)
