"""Unified-config scoping (2026-08-10): can ONE GMC config clear the
"ahead of 5-6 baselines" bar on all four datasets?

Motivation (user decision): the four per-dataset GMC anchors look like module
stitching.  A single config applied identically to F/C/CTD/Y makes GMC a single
new model.  The per-dataset numbers show the bar is weak enough that one config
MIGHT clear it:
    F/C  : plain block completion already clears (0.6489 / 0.7197)
    CTD  : BINDS the tensor view  (block-only 0.2528 -> trrank 0.3352)
    Y    : BINDS the knn fill + observed mask + rc400 (trrank 0.7176 -> 0.7428)
so the unified candidate = sym block + KNN cold-start fill + observed-nonzero
mask + rc400 + tensor rank fusion + rank-normalized fusion (no filter).

This script runs the candidate grid on the INDEPENDENT validation folds
(de-leakage: same harness as scripts/run_gmc_val.py), reporting per-dataset
AUPR against the baseline bar.  It reuses run_gmc_val.py's fold loader.

Usage:
    python scripts/run_unified_scope.py --datasets Fdataset Cdataset CTDdataset2023
    python scripts/run_unified_scope.py --datasets Ydataset          # expensive, run last
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_gmc_val import _load_val, _masked_for_val  # noqa: E402  (same-dir harness)
from gmc import gmc_predict, evaluate_fold           # noqa: E402
from gmc.helpers import RESULT_DIR                   # noqa: E402

NFOLD = 10

# ── unified candidate base ──────────────────────────────────────────────────
# Every candidate shares this structural core (the "one model"):
#   sym block + KNN cold-start fill + rank fusion, no post-hoc filter,
#   with the SHARED completion core (alpha=0.5, maxiter=40 — F/C/CTD's
#   canonical defaults; Y's alpha=1.0/maxiter=100 is a Y-strengthening and is
#   only added back if the shared core fails on Y).
# Grid dimensions: trindex (observed|all), rank cap (200|400), tensor weight.
UNIFIED_BASE = dict(fill="knn", block="sym", wknn_k=10,
                    bnnr_alpha=0.5, bnnr_maxiter=40, fusion="rank", seed=1)

UNIFIED_CANDIDATES = [
    ("uni_obs_rc400_t80", dict(trindex="observed", bnnr_rank_cap=400,
                               w_bnnr=0.2, w_tensor=0.8)),   # primary: CTD-tuned 0.2/0.8
    ("uni_y_obs_rc400_t80", dict(trindex="observed", bnnr_rank_cap=400,
                                 bnnr_alpha=1.0, bnnr_maxiter=100,
                                 w_bnnr=0.2, w_tensor=0.8)),  # Y-strengthened core (slow)
    ("uni_obs_rc400_t50", dict(trindex="observed", bnnr_rank_cap=400,
                               w_bnnr=0.5, w_tensor=0.5)),   # balanced views
    ("uni_obs_rc400_t30", dict(trindex="observed", bnnr_rank_cap=400,
                               w_bnnr=0.3, w_tensor=0.7)),   # Y fold-1 suggestion 0.3/0.7
    ("uni_all_rc400_t80", dict(trindex="all",      bnnr_rank_cap=400,
                               w_bnnr=0.2, w_tensor=0.8)),   # all-mask control
    ("uni_obs_rc200_t80", dict(trindex="observed", bnnr_rank_cap=200,
                               w_bnnr=0.2, w_tensor=0.8)),   # does rc400 matter?
    ("uni_obs_rc400_nt",  dict(trindex="observed", bnnr_rank_cap=400,
                               w_bnnr=1.0, w_tensor=0.0)),   # no-tensor control
    ("uni_obs_rc400_nofill", dict(trindex="observed", bnnr_rank_cap=400,
                                 w_bnnr=1.0, w_tensor=0.0, fill="none")),
    # no-fill control: block completion on the bare masked matrix (no cold-start
    # fill); the gap vs uni_obs_rc400_nt quantifies the fill's contribution to
    # the block geometry. (Ablation control, not a config candidate.)
]

# ── Ablation candidates: remove-one-module from the FULL model ──────────────
# (2026-08-16) Standard leave-one-component-out ablation on the fresh folds.
# Each candidate is the full unified config (trindex=observed, rc400, w 0.5/0.5,
# rank fusion) with EXACTLY ONE architectural component removed:
#   ablate_fill   — mode A: remove the cold-start KNN fill      (fill="none")
#   ablate_block  — mode B: remove the block completion view    (tensor-only)
#   ablate_fusion — numerical-detail check (NOT a 4th component): raw weighted
#                   sum instead of scale-free rank fusion (fusion="raw"); neutral
#                   (|Δ|≤0.0003) and reported in prose only, not as an ablation bar
# mode C (remove the tensor view, w_tensor=0) reuses uni_obs_rc400_nt; the full
# model (mode 0) reuses uni_obs_rc400_t50.  The old uni_obs_rc400_nofill ladder
# control (block-only, removes BOTH fill and tensor) is superseded and kept
# only as a comparison note.
ABLATION_CANDIDATES = [
    ("ablate_fill", dict(trindex="observed", bnnr_rank_cap=400,
                         w_bnnr=0.5, w_tensor=0.5, fill="none")),
    ("ablate_block", dict(trindex="observed", bnnr_rank_cap=400,
                          w_bnnr=0.0, w_tensor=1.0)),
    ("ablate_fusion", dict(trindex="observed", bnnr_rank_cap=400,
                           w_bnnr=0.5, w_tensor=0.5, fusion="raw")),
]

# Baseline AUPR bars per dataset (from COMPARISON_summary.csv, excluding NMF-DR):
#   value = AUPR of the Nth-best real baseline; a config clears "ahead of K"
#   iff its AUPR is strictly above the (K+1)-th best.  Bars are for reference
#   in the report only — the decision is made on these validation-fold numbers.
BASELINE_BARS = {
    # dataset: (ahead-of-5 bar, ahead-of-6 bar)  = (6th-best, 7th-best) baseline AUPR
    "Fdataset": (0.61061, 0.62529),
    "Cdataset": (0.69000, 0.69314),
    "CTDdataset2023": (0.29063, 0.30509),
    "Ydataset": (0.72187, 0.72507),
}


def run(ds, folds=(1, NFOLD), candidates=None, fresh=True):
    """Run the unified grid on validation folds, resuming from any existing CSV.

    ``folds`` is an inclusive 1-based (lo, hi) range; per-fold rows are APPENDED
    to the fold-results CSV so runs can be chunked (foreground-friendly; the
    harness reaps long background jobs on Windows).  The summary is re-aggregated
    from the full CSV on every call.
    """
    lo, hi = folds
    all_cands = UNIFIED_CANDIDATES + ABLATION_CANDIDATES
    cands = [(n, o) for n, o in all_cands if n in (candidates or [])] \
        if candidates else UNIFIED_CANDIDATES
    tag = "fresh" if fresh else "nested"
    csv = os.path.join(RESULT_DIR, f"{ds}_unified_{tag}_fold_results.csv")
    have = set()
    if os.path.exists(csv):
        old = pd.read_csv(csv)
        have = set(zip(old["config"], old["fold"]))
    else:
        old = None

    Wdr, pos_test, val_pos, val_idx, Wrr, Wdd, drug_sims, dis_sims = _load_val(ds, fresh=fresh)
    rows = []
    for f in range(lo, hi + 1):
        masked = _masked_for_val(Wdr, pos_test[f - 1], val_pos[f - 1])
        ind = val_idx[f - 1]; ind = ind[ind >= 0].astype(int)
        for name, over in cands:
            if (name, f) in have:
                continue
            params = dict(UNIFIED_BASE, **over)
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
    ddf.to_csv(csv, index=False)
    summ = (ddf.groupby("config").agg(AUPR=("AUPR", "mean"), AUROC=("AUROC", "mean"),
                                      AUPR_std=("AUPR", "std")).sort_values("AUPR", ascending=False))
    summ.to_csv(os.path.join(RESULT_DIR, f"{ds}_unified_{tag}_summary.csv"))

    print(f"\n== {ds} UNIFIED {tag}-val leaderboard (mean AUPR over completed folds) ==")
    b5, b6 = BASELINE_BARS[ds]
    for name, row in summ.iterrows():
        v = row["AUPR"]
        mark = ("ahead-of-5" if v > b5 else "") + (" / ahead-of-6" if v > b6 else "")
        print(f"  {name:<22s} AUPR={v:.4f}  AUROC={row['AUROC']:.4f}  {mark}")
    print(f"  [bar] ahead-of-5={b5:.4f}  ahead-of-6={b6:.4f}   (best real baseline, excl NMF-DR)")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Fdataset"])
    ap.add_argument("--folds", default="1-10", help="inclusive fold range, e.g. 1-3 (resumes)")
    ap.add_argument("--candidates", nargs="+", default=None, help="subset of candidate names")
    ap.add_argument("--fresh", action="store_true", default=True)
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.folds.split("-"))
    for ds in args.datasets:
        run(ds, folds=(lo, hi), candidates=args.candidates, fresh=args.fresh)
