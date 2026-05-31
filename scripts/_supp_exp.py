"""
Supplementary experiments for paper revision:
  (1) GBNNR lambda=0 → verify degenerates to BNNR
  (2) GBNNR + GF-BNNR combined pipeline
Runs on Fdataset fold 1 for quick turnaround.
"""
import sys, time, numpy as np
print = lambda *a, **kw: [sys.stdout.write(" ".join(map(str, a)) + "\n"), sys.stdout.flush()]

from bnnr import (BNNR, BNNR_graph, GF_BNNR, getGIPSim,
                   getKfoldCrossValidMatIndSet, getPerfMetricROCcompute,
                   compute_topk_metrics,
                   load_dataset, mask_test_entries,
                   build_augmented_matrix, extract_recovery_block,
                   build_knn_graph, normalized_laplacian_sparse)
from scipy import sparse

# ── Config ──────────────────────────────────────────────────────────────────────
DATASET = "Fdataset"
DATASET_PATH = f"data/{DATASET}.mat"
ALPHA, BETA = 1, 10
TOL1, TOL2 = 2e-3, 1e-5
MAXITER = 300
SEED = 12345
KNN_K = 12
GAMMA_GRAPH = 2.0
LAMBDA_VAL = 1e-3
INNER_STEPS, LR = 10, 1e-2
GAMMA_GIP, W_GIP, GRAPH_ALPHA = 1.0, 0.3, 0.5


# ── Helpers ─────────────────────────────────────────────────────────────────────
def build_laplacian(Wrr, Wdd, gamma):
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
        "Hits@10": int(topk["Hits@10"]), "Hits@20": int(topk["Hits@20"]),
    }


def apply_gf_filter(M_raw, Wrr, Wdd, Wdr, w_gip=W_GIP, gamma_gip=GAMMA_GIP,
                    graph_alpha=GRAPH_ALPHA):
    """Apply GF filter to an arbitrary completed matrix (no BNNR re-run)."""
    # Reconstruct GIP-fused similarities (same as GF_BNNR)
    G_dis, G_drug = getGIPSim(Wdr, gamma_gip, gamma_gip, 0, 0)
    S_dis = w_gip * G_dis + (1 - w_gip) * Wdd
    S_drug = w_gip * G_drug + (1 - w_gip) * Wrr

    # Normalised Laplacian
    from bnnr.filter import _normalised_laplacian, _graph_filter
    L_dis = _normalised_laplacian(S_dis)
    L_drug = _normalised_laplacian(S_drug)
    M_filtered = _graph_filter(M_raw, L_dis, L_drug, graph_alpha)
    return M_filtered


# ── Main ────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("Supplementary Experiments: lambda=0 + GBNNR+GF-BNNR stack")
print("=" * 80)

Wrr, Wdd, Wdr = load_dataset(DATASET_PATH)
n_dis, n_drug = Wdr.shape
print(f"Dataset: {DATASET}  ({n_dis} diseases, {n_drug} drugs, "
      f"{np.count_nonzero(Wdr)} known)")

np.random.seed(SEED + 1)
CVdata = getKfoldCrossValidMatIndSet(Wdr, 10, "CVa", "Unlabel", SEED + 1)
Ind_test = np.union1d(CVdata["MatIndSet_pos_test"][0],
                      CVdata["MatIndSet_neg_test"][0])
matDR = mask_test_entries(Wdr, Ind_test)
n_pos_test = len(CVdata["MatIndSet_pos_test"][0])
n_neg_test = len(CVdata["MatIndSet_neg_test"][0])
print(f"Fold 1: {n_pos_test} pos + {n_neg_test} neg test pairs")

# ── 1. BNNR baseline ──────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Experiment 1: BNNR baseline")
T, trIndex = build_augmented_matrix(Wrr, Wdd, matDR)
t0 = time.time()
WW, it_bnnr = BNNR(ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, 0, 1)
M_bnnr = WW[-n_dis:, :n_drug]
r_bnnr = eval_results(M_bnnr, Wdr, Ind_test)
print(f"BNNR:        AUROC={r_bnnr['AUROC']:.4f}  AUPR={r_bnnr['AUPR']:.4f}  "
      f"P@10={r_bnnr['P@10']:.4f}  P@20={r_bnnr['P@20']:.4f}  "
      f"[{it_bnnr} iters, {time.time()-t0:.0f}s]")

# ── 2. GBNNR lambda = 0 ──────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Experiment 2: GBNNR lambda=0 (should ~= BNNR)")
L_aug = build_laplacian(Wrr, Wdd, GAMMA_GRAPH)
t0 = time.time()
WW0, it_g0, info0 = BNNR_graph(
    ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, 0, 1,
    L_r=L_aug, L_d=L_aug, lambda_r=0.0, lambda_d=0.0,
    inner_steps=INNER_STEPS, lr=LR, verbose=0)
M_g0 = WW0[-n_dis:, :n_drug]
r_g0 = eval_results(M_g0, Wdr, Ind_test)
print(f"GBNNR λ=0:   AUROC={r_g0['AUROC']:.4f}  AUPR={r_g0['AUPR']:.4f}  "
      f"P@10={r_g0['P@10']:.4f}  P@20={r_g0['P@20']:.4f}  "
      f"[{it_g0} iters, {time.time()-t0:.0f}s]")
print(f"  vs BNNR:  ΔAUROC={r_g0['AUROC']-r_bnnr['AUROC']:+.4f}  "
      f"ΔAUPR={r_g0['AUPR']-r_bnnr['AUPR']:+.4f}")

# ── 3. GBNNR lambda = 1e-3 (standard) ────────────────────────────────────────
print("\n" + "-" * 60)
print("Experiment 3: GBNNR lambda=1e-3 (standard, for reference)")
t0 = time.time()
WW1, it_g1, info1 = BNNR_graph(
    ALPHA, BETA, T, trIndex, TOL1, TOL2, MAXITER, 0, 1,
    L_r=L_aug, L_d=L_aug, lambda_r=LAMBDA_VAL, lambda_d=LAMBDA_VAL,
    inner_steps=INNER_STEPS, lr=LR, verbose=0)
M_g1 = WW1[-n_dis:, :n_drug]
r_g1 = eval_results(M_g1, Wdr, Ind_test)
print(f"GBNNR λ=1e-3: AUROC={r_g1['AUROC']:.4f}  AUPR={r_g1['AUPR']:.4f}  "
      f"P@10={r_g1['P@10']:.4f}  P@20={r_g1['P@20']:.4f}  "
      f"[{it_g1} iters, {time.time()-t0:.0f}s]")
print(f"  vs BNNR:  ΔAUROC={r_g1['AUROC']-r_bnnr['AUROC']:+.4f}  "
      f"ΔAUPR={r_g1['AUPR']-r_bnnr['AUPR']:+.4f}")

# ── 4. GF-BNNR ───────────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Experiment 4: GF-BNNR (standard, for reference)")
t0 = time.time()
M_gf, M_gf_raw, it_gf = GF_BNNR(
    Wrr, Wdd, matDR, alpha=ALPHA, beta=BETA,
    tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=0, b=1,
    gamma_gip=GAMMA_GIP, w_gip=W_GIP, graph_alpha=GRAPH_ALPHA)
r_gf = eval_results(M_gf, Wdr, Ind_test)
print(f"GF-BNNR:     AUROC={r_gf['AUROC']:.4f}  AUPR={r_gf['AUPR']:.4f}  "
      f"P@10={r_gf['P@10']:.4f}  P@20={r_gf['P@20']:.4f}  "
      f"[{it_gf} BNNR iters, {time.time()-t0:.0f}s]")
print(f"  vs BNNR:  ΔAUROC={r_gf['AUROC']-r_bnnr['AUROC']:+.4f}  "
      f"ΔAUPR={r_gf['AUPR']-r_bnnr['AUPR']:+.4f}")

# ── 5. GBNNR + GF filter stack ───────────────────────────────────────────────
print("\n" + "-" * 60)
print("Experiment 5: GBNNR(λ=1e-3) + GF filter stack")
t0 = time.time()
M_stack = apply_gf_filter(M_g1, Wrr, Wdd, matDR)
r_stack = eval_results(M_stack, Wdr, Ind_test)
print(f"GBNNR+GF:    AUROC={r_stack['AUROC']:.4f}  AUPR={r_stack['AUPR']:.4f}  "
      f"P@10={r_stack['P@10']:.4f}  P@20={r_stack['P@20']:.4f}  "
      f"[{time.time()-t0:.0f}s for filter]")
print(f"  vs GBNNR alone:  ΔAUROC={r_stack['AUROC']-r_g1['AUROC']:+.4f}  "
      f"ΔAUPR={r_stack['AUPR']-r_g1['AUPR']:+.4f}")
print(f"  vs GF-BNNR alone: ΔAUROC={r_stack['AUROC']-r_gf['AUROC']:+.4f}  "
      f"ΔAUPR={r_stack['AUPR']-r_gf['AUPR']:+.4f}")
print(f"  vs BNNR:          ΔAUROC={r_stack['AUROC']-r_bnnr['AUROC']:+.4f}  "
      f"ΔAUPR={r_stack['AUPR']-r_bnnr['AUPR']:+.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Method':<25s} {'AUROC':>8s} {'AUPR':>8s} {'P@10':>8s} {'P@20':>8s}")
print("-" * 60)
for name, r in [("BNNR", r_bnnr), ("GBNNR λ=0", r_g0), ("GBNNR λ=1e-3", r_g1),
                 ("GF-BNNR", r_gf), ("GBNNR+GF stack", r_stack)]:
    print(f"{name:<25s} {r['AUROC']:8.4f} {r['AUPR']:8.4f} "
          f"{r['P@10']:8.4f} {r['P@20']:8.4f}")
