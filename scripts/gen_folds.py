"""Generate CVa fold indices (random association-pair split, SEED=12345) and export to .mat.

The folds use getKfoldCrossValidMatIndSet with CVtype='CVa' and SEED=12345,
so GMC and every published baseline are evaluated on the SAME held-out
association pairs. CVa matches multiGMF's protocol (random split of
association entries, as opposed to whole-entity holdout). Do NOT change the
seed or CVtype here.

Output (per dataset): Results/folds/folds_<dataset>.mat with
    Wdr          (dn, dr) uint8      full disease-by-drug association matrix
    pos_test_idx (nfold, max_pos) int64 Fortran linear indices of held-out POSITIVE
                                      entries (0-based), -1 padded. Only positives
                                      need masking in training; negatives are already 0.
    test_idx     (nfold, max_test) int64 Fortran linear indices of ALL held-out entries
                                      (10% positives + 10% negatives), -1 padded,
                                      used for evaluation.
"""
import os
import sys
import numpy as np
import scipy.io as sio

# Make the project root importable when this script is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import getKfoldCrossValidMatIndSet, load_dataset

NFOLD = 10
SEED = 12345
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = {
    "Fdataset": os.path.join(ROOT, "data", "Fdataset.mat"),
    "Cdataset": os.path.join(ROOT, "data", "Cdataset.mat"),
    "CTDdataset2023": os.path.join(ROOT, "data", "CTDdataset2023.mat"),
    "Ydataset": os.path.join(ROOT, "data", "Ydataset.mat"),
}
OUT = os.path.join(ROOT, "Results", "folds")
os.makedirs(OUT, exist_ok=True)


def _pad_to(rows, width, pad=-1):
    """Pad a list of 1-D arrays to a common width with `pad`."""
    mat = np.full((len(rows), width), pad, dtype=np.int64)
    for i, r in enumerate(rows):
        n = min(len(r), width)
        mat[i, :n] = np.asarray(r, dtype=np.int64)[:n]
    return mat


def main():
    for name, path in DATASETS.items():
        _wrr, _wdd, Wdr = load_dataset(path)   # Wdr = didr (disease × drug)
        dn, dr = Wdr.shape

        CVdata = getKfoldCrossValidMatIndSet(Wdr, NFOLD, "CVa", None, SEED)
        pos_test = CVdata["MatIndSet_pos_test"]
        neg_test = CVdata["MatIndSet_neg_test"]

        pos_test_idx = []
        test_idx = []
        for f in range(NFOLD):
            ind_pos = np.asarray(pos_test[f], dtype=np.int64)
            ind_neg = np.asarray(neg_test[f], dtype=np.int64)
            pos_test_idx.append(ind_pos)
            test_idx.append(np.union1d(ind_pos, ind_neg).astype(np.int64))

        max_pos = max(len(p) for p in pos_test_idx)
        max_test = max(len(t) for t in test_idx)

        out = {
            "Wdr": np.asarray(Wdr, dtype=np.uint8),
            "pos_test_idx": _pad_to(pos_test_idx, max_pos),
            "test_idx": _pad_to(test_idx, max_test),
        }
        out_path = os.path.join(OUT, f"folds_{name}.mat")
        sio.savemat(out_path, out)
        n_pos = int(np.count_nonzero(Wdr))
        n_neg = dn * dr - n_pos
        print(f"{name}: {dn}x{dr}  pos/fold={max_pos}  test/fold={max_test}  "
              f"(n_pos={n_pos}, n_neg={n_neg})")
        # Sanity: CVa property — ~10% of positives and ~10% of negatives are
        # held out per fold, all pos_test entries are genuine positives, and
        # no whole disease row is zeroed by design (rows keep their other
        # positives).
        flat = Wdr.ravel(order="F")
        for f in range(2):
            ind_pos = pos_test_idx[f]
            ind_pos = ind_pos[ind_pos >= 0].astype(int)
            ind_test = test_idx[f]
            ind_test = ind_test[ind_test >= 0].astype(int)
            assert np.all(flat[ind_pos] == 1), "pos_test_idx must be positive entries"
            assert len(ind_pos) == n_pos // NFOLD or len(ind_pos) == n_pos // NFOLD + 1
            assert abs(len(ind_test) - (n_pos + n_neg) / NFOLD) <= 2
            # masking positives leaves every disease row that had >= 2 positives intact
            masked = Wdr.copy()
            mf = masked.ravel(order="F"); mf[ind_pos] = 0
            rows_ok = np.flatnonzero(np.asarray(Wdr.sum(axis=1)) >= 2)
            assert np.all(masked[rows_ok, :].sum(axis=1) > 0)
        print(f"  fold0: {len(pos_test_idx[0])} positives, "
              f"{len(test_idx[0]) - len(pos_test_idx[0])} negatives held out")


if __name__ == "__main__":
    main()
