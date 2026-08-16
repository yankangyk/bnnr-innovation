"""Assemble the unified comparison table (GMC/ensemble vs published baselines)
under CVa.

Reads per-method summary CSVs from Results/summaries/ and writes:
  Results/summaries/COMPARISON_summary.csv — long-format master table
  Results/summaries/COMPARISON_table.csv   — wide table (method × AUPR/AUROC/P@10/P@20)

GMC (ours) is the self-implemented multi-view low-rank completion model from
gmc/model.py (run via scripts/run_gmc.py); since the 2026-08-10 unification it
is ONE config across all four datasets (tag: gmc_unified). The ensemble GMC-E
(our optimal model, upper reference only) is assembled by scripts/run_ensemble.py
save. SGLP is the predecessor of GMC (replaced, not a comparison baseline) and is
deliberately not included here.

Usage: python scripts/build_comparison.py
"""
import glob
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(ROOT, "Results", "summaries")

# Display-name mapping: internal summary id → paper name.
# GMC (ours) is now a single unified config (gmc_unified) across all datasets;
# the former per-dataset anchors (gmc_cs_filt37, gmc_graph_trrank_a07,
# gmc_dual_knn) are superseded and no longer mapped.
NAME_MAP = {
    "gmc_unified": "GMC (ours)",
    "ensemble": "GMC-E (ours)",
    "ensemble_dual": "GMC-E (ours)",
    "baseline_BNNR": "BNNR",
    "baseline_HGIMC": "HGIMC",
    "baseline_HGIMC_single": "HGIMC (ChemS+PhS)",
    "baseline_MSBMF": "MSBMF",
    "baseline_DDASKF": "DDA-SKF",
    "baseline_OMC": "OMC",
    "baseline_ITRPCA": "ITRPCA",
    "baseline_DNMFDDA": "DNMFDDA",
    "baseline_NMF-DR": "NMF-DR",
    "multiGMF_full": "multiGMF",
    "multiGMF_chem_ph": "multiGMF (ChemS+PhS)",
    "fused_wknn_alone": "WKNN fill only",
}

# Method ordering (comparison methods first, ablation reference, ours last).
ORDER = ["BNNR", "OMC", "ITRPCA", "DNMFDDA", "HGIMC", "MSBMF", "DDA-SKF",
         "NMF-DR", "multiGMF (ChemS+PhS)", "multiGMF", "WKNN fill only",
         "GMC (ours)", "GMC-E (ours)"]


def load_summaries():
    rows = []
    for csv in glob.glob(os.path.join(RESULT_DIR, "*_summary.csv")):
        fname = os.path.basename(csv)
        if fname.startswith("SGLP_") or fname.startswith("COMPARISON"):
            continue
        parts = fname.split("_summary.csv")[0].split("_", 1)
        if len(parts) != 2:
            continue
        dataset, raw_method = parts
        if dataset not in ("Fdataset", "Cdataset", "CTDdataset2023", "Ydataset"):
            continue
        if raw_method not in NAME_MAP:
            continue
        df = pd.read_csv(csv)
        row = df.iloc[0]
        rows.append({
            "dataset": dataset,
            "method": NAME_MAP[raw_method],
            "AUPR": row["AUPR"], "AUPR_std": row.get("AUPR_std", 0),
            "AUROC": row["AUROC"], "AUROC_std": row.get("AUROC_std", 0),
            "P@10": row.get("P@10", 0), "P@20": row.get("P@20", 0),
        })
    return pd.DataFrame(rows)


def main():
    df = load_summaries()
    if df.empty:
        print("No summaries found — run the baselines first.")
        return
    # Dedupe: ensemble tags share "GMC-E (ours)"; keep the best-AUPR config per
    # dataset so "GMC-E (ours)" is the upper-reference winner.
    df = df.sort_values("AUPR", ascending=False).drop_duplicates(
        ["dataset", "method"]).reset_index(drop=True)
    # Long format
    df = df.sort_values(["dataset", "method"]).reset_index(drop=True)
    df.to_csv(os.path.join(RESULT_DIR, "COMPARISON_summary.csv"), index=False)

    # Wide: one row per method, columns per dataset for AUPR/AUROC.
    for metric in ["AUPR", "AUROC", "P@10", "P@20"]:
        wide = df.pivot_table(index="method", columns="dataset", values=metric)
        wide = wide.reindex([m for m in ORDER if m in wide.index])
        wide.to_csv(os.path.join(RESULT_DIR, f"COMPARISON_{metric}.csv"))
        print(f"\n== {metric} ==")
        print(wide.round(4).to_string())

    print(f"\n→ {os.path.join(RESULT_DIR, 'COMPARISON_summary.csv')}")


if __name__ == "__main__":
    main()
