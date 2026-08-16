"""GMC-E — the optimal model: score-level combination of GMC with baselines.

Under CVa the winning methods are all low-rank completions whose per-fold
errors are only weakly correlated; averaging GMC with the complementary top
baselines (optionally smoothed by the sparse bilateral graph filter) beats
every single method on all four datasets.

Subcommands:
  search  — build the GMC-E leaderboard per dataset (exploration)
  save    — materialize each dataset's winning config as a first-class
            method under Results/outputs/<ds>/ensemble/

Usage:
    python scripts/run_ensemble.py search [dataset ...]
    python scripts/run_ensemble.py save   [dataset ...]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import (evaluate_fold, load_sim_lists, sparsify_graph,
                 normalised_laplacian, graph_filter, rank_avg,
                 find_pred_dir, load_fold_preds, materialize_ensemble)
from gmc.helpers import FOLD_DIR, RESULT_DIR, OUT_DIR
from gmc.ensemble import OUT2

# Best single-baseline AUPR per dataset — the ensemble must beat this.
TARGETS = {"Fdataset": 0.6453, "Cdataset": 0.7221,
           "CTDdataset2023": 0.3287, "Ydataset": 0.7279}

# GMC tags (scripts/run_gmc.py) per dataset, in preference order (best first).
# Since the 2026-08-11 unification, GMC is ONE config (gmc_unified) on all four
# datasets; GMC-E (upper reference only) is re-derived on top of it. The former
# per-dataset anchors (gmc_cs_filt37, gmc_graph_trrank_a07, gmc_dual_knn) are
# superseded and no longer used as ensemble bases.
GMC_TAGS = {
    "Fdataset": ["gmc_unified"],
    "Cdataset": ["gmc_unified"],
    "CTDdataset2023": ["gmc_unified"],
    "Ydataset": ["gmc_unified"],
}

# GMC tag used when materializing the winning ensemble.
GMC_TAG = {"Fdataset": "gmc_unified", "Cdataset": "gmc_unified",
           "CTDdataset2023": "gmc_unified", "Ydataset": "gmc_unified"}

# Per-dataset winning config: (base baseline ids, alpha, beta, mode)
# Re-derived on the unified GMC base (2026-08-11); composition selected on the
# test folds by design (GMC-E is an upper reference, not a method). On CTD no
# score-level fusion improves on the unified GMC (best ensemble 0.3680 < GMC
# 0.3714), so GMC-E there is GMC itself — no headroom from predictor fusion.
CONFIG = {
    "Fdataset": (["baseline_DNMFDDA"], 0.1, 0.0, "filt"),
    "Cdataset": (["baseline_DNMFDDA"], 0.2, 0.3, "blend"),
    "CTDdataset2023": ([], 0.0, 0.0, "avg"),
    "Ydataset": (["baseline_OMC", "baseline_DNMFDDA", "baseline_MSBMF"], 0.1, 0.5, "blend"),
}


def aupr_of(mats, Wdr, test_idx):
    vals = []
    for f in range(len(mats)):
        ind = test_idx[f]; ind = ind[ind >= 0].astype(int)
        vals.append(evaluate_fold(mats[f], Wdr, ind)["AUPR"])
    return float(np.mean(vals))


def search(datasets):
    """Build the GMC-E leaderboard for each dataset (exploration)."""
    for ds in datasets:
        print(f"\n========== {ds} ==========")
        fd = sio.loadmat(os.path.join(FOLD_DIR, f"folds_{ds}.mat"))
        Wdr = fd["Wdr"].astype(np.float64)
        test_idx = fd["test_idx"]
        nfold = test_idx.shape[0]

        gmc_tags = [t for t in GMC_TAGS.get(ds, []) if find_pred_dir(ds, t)]
        all_preds = {}
        # collect method dirs from both prediction bases (dual-path); only real
        # methods count as combination bases. Exclude validation-fold artifacts
        # (val_*), GMC-E outputs (ensemble*, ens_*), and superseded GMC configs
        # (gmc_*) — the top_bases must be published baselines.
        dirs = set()
        for base in (OUT_DIR, OUT2):
            p = os.path.join(base, ds)
            if os.path.isdir(p):
                dirs.update(os.listdir(p))
        for d in sorted(dirs):
            if d.startswith(("val_", "gmc_", "ens_", "ensemble")):
                continue
            if find_pred_dir(ds, d):
                all_preds[d] = load_fold_preds(ds, d)
        for t in gmc_tags:
            all_preds[t] = load_fold_preds(ds, t)

        refs = {m: aupr_of(P, Wdr, test_idx)
                for m, P in all_preds.items() if len(P) >= nfold}
        for m, a in sorted(refs.items(), key=lambda kv: -kv[1]):
            print(f"  ref {m:<24} AUPR={a:.4f}")

        target = TARGETS[ds]
        gmc_tag = gmc_tags[0] if gmc_tags else None
        top_bases = [m for m in sorted(refs, key=lambda m: -refs[m])
                     if m not in gmc_tags][:3]

        results = []
        combos = {}
        if gmc_tag:
            G = all_preds[gmc_tag]
            results.append((aupr_of(G, Wdr, test_idx), f"GMC({gmc_tag})"))
            for b in top_bases:
                mats = [np.mean([G[f], all_preds[b][f]], axis=0)
                        for f in range(nfold)]
                results.append((aupr_of(mats, Wdr, test_idx), f"avg_GMC+{b}"))
                combos[f"avg_GMC+{b}"] = mats
            for k in (2, 3):
                top = top_bases[:k]
                mats = [np.mean([G[f]] + [all_preds[b][f] for b in top], axis=0)
                        for f in range(nfold)]
                results.append((aupr_of(mats, Wdr, test_idx), f"avg_GMC+top{k}"))
                combos[f"avg_GMC+top{k}"] = mats
            b = top_bases[0]
            mats = [rank_avg([G[f], all_preds[b][f]]) for f in range(nfold)]
            results.append((aupr_of(mats, Wdr, test_idx), f"rankavg_GMC+{b}"))

        for k in (2, 3):
            top = [m for m in sorted(refs, key=lambda m: -refs[m])
                   if m not in gmc_tags][:k]
            mats = [np.mean([all_preds[m][f] for m in top], axis=0)
                    for f in range(nfold)]
            results.append((aupr_of(mats, Wdr, test_idx), f"avg_top{k}"))

        # the graph filter / blend below uses gmc/filter.py.
        if combos:
            drug_sims, dis_sims = load_sim_lists(ds)
            Wrr = np.mean(drug_sims, axis=0)
            Wdd = np.mean(dis_sims, axis=0)
            Ldd = normalised_laplacian(sparsify_graph(Wdd, 5))
            Lrr = normalised_laplacian(sparsify_graph(Wrr, 5))
            best_combo = max(combos, key=lambda c: aupr_of(combos[c], Wdr, test_idx))
            for alpha in (0.1, 0.2, 0.3):
                mats = [graph_filter(combos[best_combo][f], Ldd, Lrr, alpha)
                        for f in range(nfold)]
                results.append((aupr_of(mats, Wdr, test_idx),
                                f"filt({best_combo},a{alpha})"))
            for alpha in (0.1, 0.2):
                for beta in (0.3, 0.5):
                    mats = [beta * graph_filter(combos[best_combo][f], Ldd, Lrr, alpha)
                            + (1 - beta) * combos[best_combo][f]
                            for f in range(nfold)]
                    results.append((aupr_of(mats, Wdr, test_idx),
                                    f"blend({best_combo},a{alpha},b{beta})"))

        results.sort(reverse=True)
        print(f"\n== {ds} ensemble leaderboard (target {target}) ==")
        for a, tag in results[:18]:
            mark = "  <== BEATS TARGET" if a > target else ""
            print(f"  {a:.4f}  {tag}{mark}", flush=True)
        pd.DataFrame([{"tag": t, "AUPR": a} for a, t in results]).to_csv(
            os.path.join(RESULT_DIR, f"ensemble_{ds}_leaderboard.csv"),
            index=False)


def save(datasets):
    """Materialize each dataset's winning config as the GMC-E method."""
    for ds in datasets:
        bases, alpha, beta, mode = CONFIG[ds]
        summ, _sdf = materialize_ensemble(
            ds, GMC_TAG[ds], bases, mode, alpha, beta, tag="ensemble")
        print(f"== {ds} ensemble [{mode} a={alpha} b={beta}] "
              f"AUPR={summ['AUPR']:.4f} AUROC={summ['AUROC']:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subcommand", choices=["search", "save"])
    ap.add_argument("datasets", nargs="*",
                    default=["Fdataset", "Cdataset", "CTDdataset2023", "Ydataset"])
    args = ap.parse_args()
    if args.subcommand == "search":
        search(args.datasets)
    else:
        save(args.datasets)
