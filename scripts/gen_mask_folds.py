"""Generate CVa folds at arbitrary mask fractions (sparsity-robustness sweep).

The reported protocol holds out ~10% of positives + ~10% of negatives
(SEED=12345, ``folds_<ds>.mat``).  A robustness sweep needs other mask
fractions (5% / 20% / 30%), which are NOT expressible as an integer fold count
in ``getKfoldCrossValidMatIndSet`` (mask = 1/nfold).  This script writes a
direct CVa sampler mirroring the CVa branch of ``gmc/cv.py`` (lines 48-65):
within each fold, ``frac`` of the positive entries and ``frac`` of the negative
entries are held out (Fortran linear indices), saved exactly like
``gen_folds.py`` so downstream runners can load them the same way.

Determinism: each fraction uses its own RNG seeded from SEED + fraction*1000
so the sweep is reproducible; 10 folds per fraction.  The 10% case is NOT
generated here — it is the reported ``folds_<ds>.mat`` (SEED=12345).

Output (per dataset, per fraction): Results/folds/maskfolds_<frac>_<dataset>.mat
    Wdr          (dn, dr) uint8            full disease-by-drug association matrix
    pos_test_idx (nfold, max_pos) int64    Fortran linear indices of held-out POSITIVE
                                           entries (0-based), -1 padded
    test_idx     (nfold, max_test) int64   ALL held-out entries (pos + neg), -1 padded

Usage:
    python scripts/gen_mask_folds.py --fractions 0.05 0.20 0.30
    python scripts/gen_mask_folds.py --datasets Ydataset   # all fractions
"""
import argparse
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmc import load_dataset  # noqa: E402

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


def _cva_masks(Wdr, frac, nfold=NFOLD, seed=SEED):
    """Direct CVa mask sampler: per fold, hold out `frac` of pos and `frac` of neg."""
    flat = np.asarray(Wdr).ravel(order="F")
    ind_pos = np.flatnonzero(flat != 0).astype(np.int64)
    ind_neg = np.flatnonzero(flat == 0).astype(np.int64)
    n_pos = int(round(len(ind_pos) * frac))
    n_neg = int(round(len(ind_neg) * frac))
    pos_test, test = [], []
    rng = np.random.default_rng(seed + int(round(frac * 1000)))
    for _ in range(nfold):
        p = rng.choice(ind_pos, size=min(n_pos, len(ind_pos)), replace=False)
        n = rng.choice(ind_neg, size=min(n_neg, len(ind_neg)), replace=False)
        pos_test.append(np.sort(p))
        test.append(np.sort(np.union1d(p, n)))
    return pos_test, test


def main(fractions, datasets):
    for name in datasets:
        path = DATASETS[name]
        _wrr, _wdd, Wdr = load_dataset(path)   # Wdr = didr (disease × drug)
        dn, dr = Wdr.shape
        for frac in fractions:
            pos_test, test_idx = _cva_masks(Wdr, frac)
            max_pos = max(len(p) for p in pos_test)
            max_test = max(len(t) for t in test_idx)
            out = {
                "Wdr": np.asarray(Wdr, dtype=np.uint8),
                "pos_test_idx": _pad_to(pos_test, max_pos),
                "test_idx": _pad_to(test_idx, max_test),
            }
            out_path = os.path.join(OUT, f"maskfolds_{frac}_{name}.mat")
            sio.savemat(out_path, out)
            n_pos = int(np.count_nonzero(Wdr))
            n_neg = dn * dr - n_pos
            print(f"{name} frac={frac}: {dn}x{dr}  pos/fold~{n_pos * frac:.0f}  "
                  f"neg/fold~{n_neg * frac:.0f}  -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.05, 0.20, 0.30],
                    help="mask fractions to generate (0.10 is the reported folds, skip)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    main(args.fractions, args.datasets)
