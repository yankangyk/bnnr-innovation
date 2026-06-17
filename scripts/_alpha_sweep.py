"""
GF-BNNR α-Sensitivity Experiment: filter strength sweep on all 3 datasets.

Replaces the "Estimated AUPR values" in _gen_figures.py Fig 2 with actual
experimental data. Runs on fold 1 of each dataset for parameter sensitivity
analysis (methodological experiment, not a performance benchmark).

Grid: α_f ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0}  ×  3 datasets
Total: 18 GF-BNNR runs

Usage:  python scripts/_alpha_sweep.py
"""
import csv, os, sys, time, io, numpy as np

# Fix Windows GBK encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bnnr import (BNNR, GF_BNNR, getGIPSim,
                   getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   compute_topk_metrics,
                   load_dataset, mask_test_entries,
                   build_augmented_matrix)

# ── Config ──────────────────────────────────────────────────────────────────────
ALPHA, BETA = 1, 10
TOL1, TOL2 = 2e-3, 1e-5
MAXITER = 300
SEED = 12345
GAMMA_GIP = 1.0
W_GIP = 0.3

ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
DATASETS = ["Fdataset", "Cdataset", "DNdataset"]

OUT_DIR = os.path.join(BASE_DIR, "Results", "alpha_sweep")
os.makedirs(OUT_DIR, exist_ok=True)


def eval_results(M_recovery, Wdr, Ind_test):
    labels = Wdr.ravel(order="F")[Ind_test]
    scores = M_recovery.ravel(order="F")[Ind_test]
    tbScalar, tbVec, AUC, AUPR, Acc, Sen, Spe, Pre = getPerfMetricROCcompute(
        scores, labels, 1, 0)
    topk = compute_topk_metrics(scores, labels, ks=(10, 20))
    return {
        "AUROC": float(AUC), "AUPR": float(AUPR),
        "P@10": float(topk["P@10"]), "P@20": float(topk["P@20"]),
    }


def run_dataset(ds_name):
    """Run α-sweep on a single dataset fold 1. Returns list of result dicts."""
    ds_path = os.path.join(BASE_DIR, "data", f"{ds_name}.mat")
    Wrr, Wdd, Wdr = load_dataset(ds_path)
    n_dis, n_drug = Wdr.shape
    density = np.count_nonzero(Wdr) / (n_dis * n_drug) * 100

    np.random.seed(SEED + 1)
    CVdata = getKfoldCrossValidMatIndSet(Wdr, 10, "CVa", "Unlabel", SEED + 1)
    Ind_test = np.union1d(CVdata["MatIndSet_pos_test"][0],
                          CVdata["MatIndSet_neg_test"][0])
    matDR = mask_test_entries(Wdr, Ind_test)

    # Pre-compute GIP similarities (shared across all α values)
    G_dis, G_drug = getGIPSim(matDR, GAMMA_GIP, GAMMA_GIP, 0, 0)
    S_drug = W_GIP * G_drug + (1 - W_GIP) * Wrr
    S_dis = W_GIP * G_dis + (1 - W_GIP) * Wdd

    # BNNR baseline (raw similarities, no GIP, no filter)
    T_raw, trIdx_raw = build_augmented_matrix(Wrr, Wdd, matDR)
    t0 = time.time()
    WW_bnnr, it_bnnr = BNNR(ALPHA, BETA, T_raw, trIdx_raw, TOL1, TOL2, MAXITER, 0, 1)
    M_bnnr = WW_bnnr[-n_dis:, :n_drug]
    r_bnnr = eval_results(M_bnnr, Wdr, Ind_test)

    results = []
    for af in ALPHAS:
        label = f"{ds_name} α_f={af:.1f}"
        print(f"  {label} ...", end=" ", flush=True)
        t0 = time.time()

        if af == 0.0:
            # α=0: filter is identity, M_filtered = M_raw → effectively BNNR+GIP
            # But we want the GF_BNNR pipeline output, so run it and take M_raw
            M_gf, M_raw, it_gf = GF_BNNR(
                Wrr, Wdd, matDR,
                alpha=ALPHA, beta=BETA,
                tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=0, b=1,
                gamma_gip=GAMMA_GIP, w_gip=W_GIP,
                graph_alpha=0.0,  # effectively disables filter
                S_drug=S_drug, S_dis=S_dis)
            M_out = M_gf
            it_used = it_gf
        else:
            M_gf, M_raw, it_gf = GF_BNNR(
                Wrr, Wdd, matDR,
                alpha=ALPHA, beta=BETA,
                tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=0, b=1,
                gamma_gip=GAMMA_GIP, w_gip=W_GIP,
                graph_alpha=af,
                S_drug=S_drug, S_dis=S_dis)
            M_out = M_gf
            it_used = it_gf

        r = eval_results(M_out, Wdr, Ind_test)
        elapsed = time.time() - t0

        row = {
            "dataset": ds_name,
            "density_pct": round(density, 4),
            "alpha_f": af,
            "AUROC": r["AUROC"], "AUPR": r["AUPR"],
            "P@10": r["P@10"], "P@20": r["P@20"],
            "iter_num": it_used, "time_s": round(elapsed, 1),
            "bnnr_aupr": r_bnnr["AUPR"],
            "delta_aupr": r["AUPR"] - r_bnnr["AUPR"],
        }
        results.append(row)

        print(f"AUROC={r['AUROC']:.4f}  AUPR={r['AUPR']:.4f}  "
              f"ΔAUPR={row['delta_aupr']:+.4f}  [{it_used} iters, {elapsed:.0f}s]")

    results.append({
        "dataset": ds_name, "density_pct": round(density, 4),
        "alpha_f": "BNNR_ref",
        "AUROC": r_bnnr["AUROC"], "AUPR": r_bnnr["AUPR"],
        "P@10": r_bnnr["P@10"], "P@20": r_bnnr["P@20"],
        "iter_num": it_bnnr, "time_s": round(time.time() - t0, 1),
        "bnnr_aupr": r_bnnr["AUPR"], "delta_aupr": 0.0,
    })

    return results


def main():
    print("=" * 80)
    print("GF-BNNR α-Sensitivity Experiment: Filter Strength Sweep")
    print("=" * 80)

    all_results = []

    for ds_name in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'─' * 60}")
        results = run_dataset(ds_name)
        all_results.extend(results)

    # ── Summary tables ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: AUPR by dataset and α_f")
    print("=" * 80)

    for ds_name in DATASETS:
        ds_results = [r for r in all_results
                      if r["dataset"] == ds_name and r["alpha_f"] != "BNNR_ref"]
        bnnr_ref = [r for r in all_results
                    if r["dataset"] == ds_name and r["alpha_f"] == "BNNR_ref"][0]

        print(f"\n{ds_name} (BNNR baseline AUPR={bnnr_ref['AUPR']:.4f}):")
        print(f"  {'α_f':>6s}  {'AUROC':>8s}  {'AUPR':>8s}  {'ΔAUPR':>8s}")
        print(f"  {'─' * 40}")
        for row in ds_results:
            af = row["alpha_f"]
            print(f"  {af:6.1f}  {row['AUROC']:8.4f}  {row['AUPR']:8.4f}  "
                  f"{row['delta_aupr']:+8.4f}")

        # Find best α
        best = max(ds_results, key=lambda r: r["AUPR"])
        print(f"  Best α_f = {best['alpha_f']:.1f} (AUPR = {best['AUPR']:.4f})")

    # ── Save ─────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "alpha_sweep_results.csv")
    fieldnames = ["dataset", "density_pct", "alpha_f", "AUROC", "AUPR",
                   "P@10", "P@20", "iter_num", "time_s",
                   "bnnr_aupr", "delta_aupr"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nResults saved to {csv_path}")

    # ── Compare with "estimated" values in _gen_figures.py ───────────────────
    print("\n" + "=" * 80)
    print("COMPARISON: Actual vs Estimated (from _gen_figures.py Fig 2)")
    print("=" * 80)
    old_f = [0.3071, 0.3134, 0.3179, 0.3198, 0.3172, 0.3145]
    old_c = [0.2970, 0.3320, 0.3453, 0.3479, 0.3438, 0.3380]
    old_dn = [0.2181, 0.2905, 0.3157, 0.3120, 0.3010, 0.2870]
    old_map = {"Fdataset": old_f, "Cdataset": old_c, "DNdataset": old_dn}

    for ds_name in DATASETS:
        ds_results = [r for r in all_results
                      if r["dataset"] == ds_name and r["alpha_f"] != "BNNR_ref"]
        actual = [r["AUPR"] for r in ds_results]
        estimated = old_map[ds_name]
        print(f"\n{ds_name}:")
        print(f"  α_f:     ", end="")
        for af in ALPHAS:
            print(f"  {af:6.1f}", end="")
        print()
        print(f"  Actual:  ", end="")
        for v in actual:
            print(f"  {v:6.4f}", end="")
        print()
        print(f"  Estimat: ", end="")
        for v in estimated:
            print(f"  {v:6.4f}", end="")
        print()
        max_diff = max(abs(a - e) for a, e in zip(actual, estimated))
        print(f"  Max diff: {max_diff:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
