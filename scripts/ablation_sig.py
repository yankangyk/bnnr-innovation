"""Paired fold-level Wilcoxon + delta for the remove-one-module ablation.

Each ablation mode (A fill, B block, C tensor) is compared against the FULL
model (uni_obs_rc400_t50) on the SAME fresh validation folds (paired by
construction), per dataset. Prints delta (mode - full) and the two-sided paired
Wilcoxon signed-rank p-value (p=0.002 is the minimum for n=10 all-one-direction).

The rank-vs-raw fusion comparison (ablate_fusion) is a NUMERICAL-DETAIL check,
not a 4th architectural component: it is printed here for completeness but is
reported in the manuscript as a one-line prose note, not as an ablation mode.

Run:  python scripts/ablation_sig.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gmc.helpers import RESULT_DIR

DATASETS = ["Fdataset", "Cdataset", "CTDdataset2023", "Ydataset"]
FULL = "uni_obs_rc400_t50"
MODES = {  # config in the fresh fold CSV -> ablation label
    "ablate_fill": "A (-fill)",
    "ablate_block": "B (-block)",
    "uni_obs_rc400_nt": "C (-tensor)",
    "ablate_fusion": "fusion (num. detail)",
}


def main():
    print(f"{'dataset':<15s} {'mode':<12s} {'full':>7s} {'mode':>7s} {'delta':>7s}  p (paired Wilcoxon)")
    for ds in DATASETS:
        p = os.path.join(RESULT_DIR, f"{ds}_unified_fresh_fold_results.csv")
        df = pd.read_csv(p)
        full = df[df.config == FULL].set_index("fold")["AUPR"]
        for cfg, label in MODES.items():
            m = df[df.config == cfg].set_index("fold")["AUPR"]
            if len(m) < 10:
                print(f"{ds:<15s} {label:<12s} INCOMPLETE ({len(m)} folds)")
                continue
            common = full.index.intersection(m.index)
            d = (m - full).loc[common]
            try:
                stat, pv = wilcoxon(d, zero_method="wilcox")
                pv = max(pv, 2 / 2**10)  # n=10 two-sided floor = 0.002
            except ValueError:
                pv = float("nan")
            print(f"{ds:<15s} {label:<12s} {full.loc[common].mean():7.4f} "
                  f"{m.loc[common].mean():7.4f} {d.mean():+7.4f}  p={pv:.3f}")


if __name__ == "__main__":
    main()
