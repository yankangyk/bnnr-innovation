"""
λ-Insensitivity Experiment: Full γ × λ grid sweep on Fdataset fold 1.

Reproduces Table 2 data. The hardcoded values in _gen_figures.py were from
an earlier manual run — this script systematically regenerates them.

Grid: γ ∈ {0.5, 1.0, 2.0, 3.0}  ×  λ ∈ {0, 1e-3, 1e-2, 1e-1}
Total: 16 GBNNR runs + 1 BNNR baseline = 17 runs

Usage:  python scripts/_lambda_sweep.py
"""
import csv, os, sys, time, io, numpy as np
from scipy import sparse

# Fix Windows GBK encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bnnr import (BNNR, BNNR_graph,
                   getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   compute_topk_metrics,
                   load_dataset, mask_test_entries,
                   build_augmented_matrix,
                   build_knn_graph, normalized_laplacian_sparse)

# ── Config ──────────────────────────────────────────────────────────────────────
DATASET = "Fdataset"
DATASET_PATH = os.path.join(BASE_DIR, "data", f"{DATASET}.mat")
ALPHA, BETA = 1, 10
TOL1, TOL2 = 2e-3, 1e-5
MAXITER = 300
SEED = 12345
KNN_K = 12
INNER_STEPS, LR = 10, 1e-2

GAMMAS = [0.5, 1.0, 2.0, 3.0]
LAMBDAS = [0.0, 1e-3, 1e-2, 1e-1]

OUT_DIR = os.path.join(BASE_DIR, "Results", "lambda_sweep")
os.makedirs(OUT_DIR, exist_ok=True)


def build_laplacian(Wrr, Wdd, gamma):
    """Build block-diagonal normalized Laplacian: L = blkdiag(L_dis, L_drug)."""
    G_drug = build_knn_graph(Wrr, k=KNN_K, sym=True, remove_diag=True, gamma=gamma)
    G_dis = build_knn_graph(Wdd, k=KNN_K, sym=True, remove_diag=True, gamma=gamma)
    L_drug = normalized_laplacian_sparse(G_drug)
    L_dis = normalized_laplacian_sparse(G_dis)
    return sparse.block_diag([L_dis, L_drug], format='csr')


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


def main():
    print("=" * 80)
    print("λ-Insensitivity Experiment: γ × λ Grid Sweep")
    print("=" * 80)

    # ── Load data and set up fold 1 ─────────────────────────────────────────
    Wrr, Wdd, Wdr = load_dataset(DATASET_PATH)
    n_dis, n_drug = Wdr.shape
    print(f"Dataset: {DATASET}  ({n_dis} diseases × {n_drug} drugs, "
          f"{np.count_nonzero(Wdr)} known)")

    np.random.seed(SEED + 1)
    CVdata = getKfoldCrossValidMatIndSet(Wdr, 10, "CVa", "Unlabel", SEED + 1)
    Ind_test = np.union1d(CVdata["MatIndSet_pos_test"][0],
                          CVdata["MatIndSet_neg_test"][0])
    matDR = mask_test_entries(Wdr, Ind_test)

    T, trIndex = build_augmented_matrix(Wrr, Wdd, matDR)

    # ── 1. BNNR baseline ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Running BNNR baseline...")
    t0 = time.time()
    WW_bnnr, it_bnnr = BNNR(ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, 0, 1)
    M_bnnr = WW_bnnr[-n_dis:, :n_drug]
    r_bnnr = eval_results(M_bnnr, Wdr, Ind_test)
    print(f"BNNR:  AUROC={r_bnnr['AUROC']:.4f}  AUPR={r_bnnr['AUPR']:.4f}  "
          f"P@10={r_bnnr['P@10']:.4f}  [{it_bnnr} iters, {time.time()-t0:.0f}s]")

    # ── 2. γ × λ grid ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Running γ × λ grid: {len(GAMMAS)} γ × {len(LAMBDAS)} λ = "
          f"{len(GAMMAS)*len(LAMBDAS)} experiments")
    print("=" * 80)

    results = []
    total = len(GAMMAS) * len(LAMBDAS)
    count = 0

    for gamma in GAMMAS:
        # Build Laplacian once per γ
        t_lap = time.time()
        L_aug = build_laplacian(Wrr, Wdd, gamma)
        lap_time = time.time() - t_lap

        for lam in LAMBDAS:
            count += 1
            label = f"γ={gamma:.1f}, λ={lam:.0e}"
            print(f"\n[{count}/{total}] {label} ...")

            t0 = time.time()
            WW, it, info = BNNR_graph(
                ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, 0, 1,
                L_r=L_aug, L_d=L_aug,
                lambda_r=lam, lambda_d=lam,
                inner_steps=INNER_STEPS, lr=LR, verbose=0)
            M = WW[-n_dis:, :n_drug]
            r = eval_results(M, Wdr, Ind_test)
            elapsed = time.time() - t0

            row = {
                "gamma": gamma, "lambda": lam,
                "AUROC": r["AUROC"], "AUPR": r["AUPR"],
                "P@10": r["P@10"], "P@20": r["P@20"],
                "iter_num": it, "time_s": elapsed,
                "bnnr_aupr": r_bnnr["AUPR"],
                "delta_aupr": r["AUPR"] - r_bnnr["AUPR"],
            }
            results.append(row)

            print(f"  AUROC={r['AUROC']:.4f}  AUPR={r['AUPR']:.4f}  "
                  f"ΔAUPR={row['delta_aupr']:+.4f}  "
                  f"[{it} iters, {elapsed:.0f}s]")

    # ── 3. Output table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS TABLE (γ × λ → AUPR)")
    print("=" * 80)

    # Header
    lambda_labels = ["0", "10⁻³", "10⁻²", "10⁻¹"]
    header = f"{'γ \\ λ':<8s}"
    for ll in lambda_labels:
        header += f" {ll:>10s}"
    print(header)
    print("-" * (8 + 10 * len(LAMBDAS)))

    # Build grid
    grid = {}
    for row in results:
        grid[(row["gamma"], row["lambda"])] = row["AUPR"]

    for gamma in GAMMAS:
        line = f"{gamma:<8.1f}"
        for lam in LAMBDAS:
            val = grid.get((gamma, lam), float("nan"))
            line += f" {val:10.4f}"
        print(line)

    print(f"\nBNNR baseline AUPR: {r_bnnr['AUPR']:.4f}")

    # ── 4. Save to CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "lambda_sweep_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # ── 5. Verify against chapter plan Table 2 ──────────────────────────────
    print("\n" + "=" * 80)
    print("VERIFICATION: Chapter Plan Table 2 Expected Values")
    print("=" * 80)
    expected = {
        (0.5, 0):       0.3258, (0.5, 1e-3): 0.3251, (0.5, 1e-2): 0.3251, (0.5, 1e-1): 0.3251,
        (1.0, 0):       0.3242, (1.0, 1e-3): 0.3242, (1.0, 1e-2): 0.3242, (1.0, 1e-1): 0.3242,
        (2.0, 0):       0.3273, (2.0, 1e-3): 0.3274, (2.0, 1e-2): 0.3274, (2.0, 1e-1): 0.3274,
        (3.0, 0):       0.3245, (3.0, 1e-3): 0.3244, (3.0, 1e-2): 0.3244, (3.0, 1e-1): 0.3244,
    }
    all_match = True
    for (gamma, lam), exp_val in expected.items():
        actual = grid.get((gamma, lam), float("nan"))
        match = abs(actual - exp_val) < 0.001
        if not match:
            print(f"  ⚠ γ={gamma}, λ={lam}: expected {exp_val:.4f}, got {actual:.4f}")
            all_match = False
    if all_match:
        print("  All values match chapter plan within 0.001 tolerance.")
    else:
        print("  NOTE: Some values differ. Update chapter plan and _gen_figures.py"
              " with the new values above.")

    print("\nDone.")


if __name__ == "__main__":
    main()
