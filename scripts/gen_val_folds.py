"""Generate INDEPENDENT validation masks nested inside the CVa test folds.

Purpose (de-leakage / reviewer defence): the per-dataset GMC/GMC-E configurations
(view composition, fusion weights, graph-embedding switch, filter settings, GMC-E
member sets) were originally selected by inspecting the 10-fold CVa TEST results.
This script builds a separate hold-out so the configuration can instead be chosen
on folds that never touched the reported test indices.

For each dataset and each CVa fold f:
  * test_idx[f]   — the existing held-out test entries (10% pos + 10% neg).
  * training      — every entry NOT in test_idx[f].
  * val_pos[f]    — a fresh ~10% sample of the TRAINING positives.
  * val_neg[f]    — a fresh ~10% sample of the TRAINING negatives.
  * val_idx[f]    — val_pos ∪ val_neg, disjoint from test_idx[f] by construction.

Sampling is deterministic (SEED_VAL fixed, fresh default_rng), so every run of
this script reproduces the same masks.  Validation-training masking zeroes
(pos_test ∪ val_pos) — test negatives are already 0, so only positives need
masking; evaluation is on val_idx.

Output (per dataset): Results/folds/valfolds_<ds>.mat
    Wdr          (dn, dr) uint8      full disease-by-drug association matrix
    test_idx     (nfold, max_test)   held-out TEST entries (masked during val training)
    pos_test_idx (nfold, max_pos)    positive test entries (masked to zero)
    val_pos_idx  (nfold, max_valpos) positive VALIDATION entries (masked to zero)
    val_idx      (nfold, max_val)    all validation entries (evaluated for selection)
"""
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import load_dataset, getKfoldCrossValidMatIndSet
from gmc.helpers import FOLD_DIR

NFOLD = 10
SEED_VAL = 20260809          # NESTED validation sampling seed (independent of fold seed 12345)
SEED_FRESH = 24680           # FRESH independent CVa validation split seed
VAL_FRAC = 0.10              # ~10% of the per-fold training positives/negatives
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = {
    "Fdataset": os.path.join(ROOT, "data", "Fdataset.mat"),
    "Cdataset": os.path.join(ROOT, "data", "Cdataset.mat"),
    "CTDdataset2023": os.path.join(ROOT, "data", "CTDdataset2023.mat"),
    "Ydataset": os.path.join(ROOT, "data", "Ydataset.mat"),
}


def _pad_to(rows, width, pad=-1):
    mat = np.full((len(rows), width), pad, dtype=np.int64)
    for i, r in enumerate(rows):
        n = min(len(r), width)
        mat[i, :n] = np.asarray(r, dtype=np.int64)[:n]
    return mat


def main_fresh():
    """A FRESH, faithful 10-fold CVa validation split (SEED_FRESH, same ~10%
    masking as the test protocol — NOT nested, so training sparsity matches the
    reported scenario).  Used to adjudicate config choices where the nested
    validation folds and the test folds disagree (Y block structure, CTD w_bnnr).

    The fresh split reuses the CVa machinery directly: pos_test_idx/neg_test_idx
    are sampled from the FULL matrix exactly as in ``scripts/gen_folds.py``.
    Written to ``valfolds_fresh_<ds>.mat`` with the same schema as the nested
    val folds (val_pos_idx == pos_test_idx, val_idx == test_idx of the fresh
    split), so ``scripts/run_gmc_val.py --fresh`` can reuse the same code path.
    """
    for name, path in DATASETS.items():
        _wrr, _wdd, Wdr = load_dataset(path)
        dn, dr = Wdr.shape
        CVdata = getKfoldCrossValidMatIndSet(Wdr, NFOLD, "CVa", None, SEED_FRESH)
        pos_test = CVdata["MatIndSet_pos_test"]
        neg_test = CVdata["MatIndSet_neg_test"]

        pos_test_idx, test_idx = [], []
        for f in range(NFOLD):
            ind_pos = np.asarray(pos_test[f], dtype=np.int64)
            ind_neg = np.asarray(neg_test[f], dtype=np.int64)
            pos_test_idx.append(ind_pos)
            test_idx.append(np.union1d(ind_pos, ind_neg).astype(np.int64))

        max_pos = max(len(p) for p in pos_test_idx)
        max_test = max(len(t) for t in test_idx)
        out = {
            "Wdr": np.asarray(Wdr, dtype=np.uint8),
            "test_idx": _pad_to(test_idx, max_test),
            "pos_test_idx": _pad_to(pos_test_idx, max_pos),
            "val_pos_idx": _pad_to(pos_test_idx, max_pos),   # mask = fresh pos
            "val_idx": _pad_to(test_idx, max_test),           # eval = fresh test
        }
        out_path = os.path.join(FOLD_DIR, f"valfolds_fresh_{name}.mat")
        sio.savemat(out_path, out)
        n_pos = int(np.count_nonzero(Wdr))
        print(f"{name}: {dn}x{dr}  fresh pos/fold~{max_pos}  fresh test/fold~{max_test} "
              f"(n_pos={n_pos})  -> {out_path}")


def main():
    for name, path in DATASETS.items():
        _wrr, _wdd, Wdr = load_dataset(path)
        dn, dr = Wdr.shape
        flat = Wdr.ravel(order="F")
        all_pos = np.flatnonzero(flat != 0)          # true positives
        all_neg = np.flatnonzero(flat == 0)          # true negatives

        fd = sio.loadmat(os.path.join(FOLD_DIR, f"folds_{name}.mat"))
        pos_test = fd["pos_test_idx"]
        test_idx = fd["test_idx"]

        rng = np.random.default_rng(SEED_VAL)
        val_pos_idx, val_idx = [], []
        for f in range(NFOLD):
            pt = pos_test[f]; pt = pt[pt >= 0].astype(int)
            tt = test_idx[f]; tt = tt[tt >= 0].astype(int)
            neg_test = np.setdiff1d(tt, pt)          # negative test entries

            train_pos = np.setdiff1d(all_pos, pt)    # disjoint from test positives
            train_neg = np.setdiff1d(all_neg, neg_test)  # disjoint from test negatives

            n_vp = max(1, int(round(VAL_FRAC * len(train_pos))))
            vp = rng.choice(train_pos, size=n_vp, replace=False)
            n_vn = max(1, int(round(VAL_FRAC * len(train_neg))))
            vn = rng.choice(train_neg, size=n_vn, replace=False)
            vi = np.union1d(vp, vn).astype(np.int64)

            # disjointness by construction — assert it anyway
            assert not np.isin(vp, pt).any(), "val positives overlap test positives"
            assert not np.isin(vn, neg_test).any(), "val negatives overlap test negatives"
            assert not np.isin(vi, tt).any(), "val_idx overlaps test_idx"

            val_pos_idx.append(vp.astype(np.int64))
            val_idx.append(vi)

        max_vp = max(len(v) for v in val_pos_idx)
        max_vt = max(len(v) for v in val_idx)
        out = {
            "Wdr": np.asarray(Wdr, dtype=np.uint8),
            "test_idx": test_idx,
            "pos_test_idx": pos_test,
            "val_pos_idx": _pad_to(val_pos_idx, max_vp),
            "val_idx": _pad_to(val_idx, max_vt),
        }
        out_path = os.path.join(FOLD_DIR, f"valfolds_{name}.mat")
        sio.savemat(out_path, out)
        n_pos = int(np.count_nonzero(Wdr))
        print(f"{name}: {dn}x{dr}  val_pos/fold~{max_vp}  val/fold~{max_vt}  "
              f"(train_pos/fold~{len(train_pos)}, n_pos={n_pos})")
        print(f"  fold0: {len(val_idx[0])} val entries "
              f"({len(val_pos_idx[0])} positives) vs {len(test_idx[0])} test entries")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="write a FRESH independent 10-fold CVa validation split "
                         "(SEED_FRESH, same ~10% masking as the test protocol)")
    args = ap.parse_args()
    if args.fresh:
        main_fresh()
    else:
        main()
