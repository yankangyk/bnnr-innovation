"""Case study: drug–disease association prediction from GMC.

Two protocols (2026-08-16). The REPORTED case study (paper Sec.~case,
PROGRESS experiment 5) is ``--protocol loo`` (strict leave-all-known-out,
diagnostic control) plus ``--protocol full`` (full-data top-N novel
predictions, 65.5% direct / 86.2% literature-consistent hit rate).

--protocol loo            — STRICT leave-ALL-known-out (diagnostic control).
    The ENTIRE known matrix is masked (didr > 0 -> 0); the all-zero matrix
    makes the cold-start fill degenerate to zero, so the prediction is pure
    similarity-driven low-rank completion and hits@K ≈ chance (AUPR ≈ known
    density). The controlled demonstration that GMC completes from OBSERVED
    associations, not from similarity alone.  (Reported, Table tab:loo.)

--protocol full           — original full-data protocol: run GMC on the FULL
    (unmasked) matrix, then emit the top-N pairs with NO known association
    (didr == 0) as *novel* predictions. Manual PubMed/CTD check: 19/29
    unique top-10 pairs directly supported (65.5%), 25/29 literature-
    consistent (86.2%).  (Reported, Table tab:case.)

The .mat files carry only ID strings (no human-readable names / no name map
in the repo), so the emitted tables are meant for the user to look up in
PubMed/CTD/DrugBank themselves.

Usage:
    python scripts/case_study.py --datasets Fdataset --protocol loo --top 10
    python scripts/case_study.py --datasets Fdataset --protocol full --top 20

Writes (Results/summaries/):
    case_study_loo_summary.csv     — strict-LOO hits@K vs chance (diagnostic)
    case_study_loo_topN.csv        — strict-LOO top-K recovery table
    case_study_topN.csv            — novel predictions (full protocol)
    case_study_topN_readable.csv   — same, sorted for manual review
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import gmc_predict, load_sim_lists, evaluate_fold
from gmc.helpers import DATA_DIR, RESULT_DIR
from run_gmc import UNIFIED_CONFIG  # noqa: E402

DATASETS = ["Fdataset", "Cdataset", "CTDdataset2023", "Ydataset"]


def _load(ds):
    d = sio.loadmat(os.path.join(DATA_DIR, f"{ds}.mat"))
    didr = np.asarray(d["didr"])
    def _clean(x):
        return str(x).strip("[]'\" ")
    wdname = np.asarray([_clean(x) for x in np.asarray(d["Wdname"]).ravel()])
    wrname = np.asarray([_clean(x) for x in np.asarray(d["Wrname"]).ravel()])
    drug_sims, dis_sims = load_sim_lists(ds)
    Wrr = np.mean(drug_sims, axis=0)
    Wdd = np.mean(dis_sims, axis=0)
    return didr, wdname, wrname, Wrr, Wdd, drug_sims, dis_sims


def _params(fill="knn"):
    """Reported unified configuration. ``fill="knn"`` (default) is the reported
    cold-start KNN fill; ``fill="none"`` runs the pure low-rank completion on
    the masked matrix (used by the strict-LOO diagnostic).
    """
    p = {k: v for k, v in UNIFIED_CONFIG.items() if k != "tag"}
    p["fill"] = fill
    return p


# ── strict leave-all-known-out (diagnostic control) ─────────────────────────
def run_loo(ds, top=10, ks=(10, 20, 50)):
    """Strict leave-all-known-out: mask EVERYTHING, rank all dn×dr entries.

    Returns (rows, summ) where ``rows`` is the top-``top`` recovery table and
    ``summ`` holds per-dataset hits@K + the known-density random baseline.
    """
    didr, wdname, wrname, Wrr, Wdd, drug_sims, dis_sims = _load(ds)
    dn, dr = didr.shape
    masked = np.zeros_like(didr, dtype=np.float64)      # leave-all-known-out
    M_pred = gmc_predict(masked, Wrr, Wdd, drug_sims, dis_sims, seed=1,
                         fill="none", block="sym", wknn_k=10, bnnr_alpha=0.5,
                         bnnr_maxiter=40, bnnr_rank_cap=400, trindex="observed",
                         w_bnnr=0.5, w_tensor=0.5, fusion="rank")

    known_flat = (didr > 0).ravel(order="F")
    n_known = int(known_flat.sum())
    density = n_known / (dn * dr)
    flat = M_pred.ravel(order="F")

    # overall AUPR over ALL dn×dr entries (labels = the full known matrix):
    # at chance it equals the known density.
    full = evaluate_fold(np.clip(M_pred, 0, 1), didr.astype(np.float64),
                         np.arange(dn * dr))

    summ = {"dataset": ds, "n_dis": dn, "n_drug": dr, "n_known": n_known,
            "known_density": density, "aupr": float(full["AUPR"])}
    rows = []
    for K in ks:
        order = np.argsort(-flat)[:K]
        summ[f"hits@{K}"] = int(known_flat[order].sum())
        summ[f"hits_at_rate@{K}"] = float(known_flat[order].mean())
    # recovery table for the headline top-N
    order = np.argsort(-flat)[:top]
    for rank, k in enumerate(order, 1):
        d_i, r_j = np.unravel_index(k, (dn, dr), order="F")
        rows.append({"dataset": ds, "rank": rank,
                     "drug_id": wrname[r_j], "disease_id": wdname[d_i],
                     "score": float(flat[k]), "known": bool(known_flat[k])})
    print(f"\n== {ds} LOO top-{top} recovery (density={density:.4f}, "
          f"hits@10={summ['hits@10']}) ==", flush=True)
    for r in rows:
        tag = "TRUE " if r["known"] else "false"
        print(f"  {r['rank']:2d}. {tag} {r['drug_id']:10s} -> "
              f"{r['disease_id']:10s}  score={r['score']:.4f}", flush=True)
    return rows, summ


# ── full-data top-N novel predictions (reference protocol) ──────────────────
def run_full(ds, top=10):
    """Original full-data protocol: top-N NOVEL (didr==0) predictions."""
    didr, wdname, wrname, Wrr, Wdd, drug_sims, dis_sims = _load(ds)
    dn, dr = didr.shape
    params = _params()
    masked = didr.astype(np.float64)          # full matrix, no masking
    M_pred = gmc_predict(masked, Wrr, Wdd, drug_sims, dis_sims, seed=1, **params)

    known = didr > 0
    novel = ~known
    flat_pos = np.flatnonzero(novel.ravel(order="F"))
    order = np.argsort(-M_pred.ravel(order="F")[flat_pos])[:top]
    idx = flat_pos[order]
    rows = []
    for k in idx:
        d_i, r_j = np.unravel_index(k, (dn, dr), order="F")
        rows.append({"dataset": ds,
                     "drug_id": wrname[r_j], "disease_id": wdname[d_i],
                     "score": float(M_pred[d_i, r_j])})
    print(f"\n== {ds} top-{top} NOVEL drug-disease predictions (didr==0) ==")
    for r in rows:
        print(f"  {r['drug_id']:10s} -> {r['disease_id']:10s}  score={r['score']:.4f}")
    return rows


def main(datasets, top, protocol):
    if protocol == "loo":
        all_rows, sums = [], []
        for ds in datasets:
            rows, summ = run_loo(ds, top=top)
            all_rows.extend(rows)
            sums.append(summ)
        pd.DataFrame(all_rows).to_csv(
            os.path.join(RESULT_DIR, "case_study_loo_topN.csv"), index=False)
        pd.DataFrame(sums).to_csv(
            os.path.join(RESULT_DIR, "case_study_loo_summary.csv"), index=False)
        print(f"\nwrote case_study_loo_topN.csv / case_study_loo_summary.csv")
    else:
        all_rows = []
        for ds in datasets:
            all_rows.extend(run_full(ds, top=top))
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(RESULT_DIR, "case_study_topN.csv"), index=False)
        df["_s"] = -df["score"]
        df = df.sort_values(["dataset", "_s"]).drop(columns="_s")
        df.to_csv(os.path.join(RESULT_DIR, "case_study_topN_readable.csv"),
                  index=False)
        print(f"\nwrote case_study_topN.csv ({len(df)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--top", type=int, default=10,
                    help="top-N table rows (loo) / top-N novel pairs (full)")
    ap.add_argument("--protocol", choices=["loo", "full"], default="loo",
                    help="loo = strict leave-all-known-out (reported case study "
                         "diagnostic); full = top-N novel predictions on the "
                         "full matrix (reported)")
    args = ap.parse_args()
    main(args.datasets, args.top, args.protocol)
