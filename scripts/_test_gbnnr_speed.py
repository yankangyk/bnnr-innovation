"""Quick test: BNNR_graph on DNdataset fold 1 with timing."""
import time, sys, numpy as np
print = lambda *a, **kw: [sys.stdout.write(" ".join(map(str, a)) + "\n"), sys.stdout.flush()]

from bnnr import (getKfoldCrossValidMatIndSet, BNNR_graph,
                   build_knn_graph, normalized_laplacian_sparse,
                   load_dataset, mask_test_entries, build_augmented_matrix)
from scipy import sparse

Wrr, Wdd, Wdr = load_dataset('data/DNdataset.mat')
print(f'Loaded: Wrr={Wrr.shape}, Wdd={Wdd.shape}, Wdr={Wdr.shape}')
print(f'Nonzeros: {np.count_nonzero(Wdr)}')

SEED = 12345
np.random.seed(SEED + 1)
CVdata = getKfoldCrossValidMatIndSet(Wdr, 10, 'CVa', 'Unlabel', SEED + 1)
Ind_test = np.union1d(CVdata['MatIndSet_pos_test'][0], CVdata['MatIndSet_neg_test'][0])
matDR = mask_test_entries(Wdr, Ind_test)

t0 = time.time()
G_drug = build_knn_graph(Wrr, k=12, sym=True, remove_diag=True, gamma=2.0)
G_disease = build_knn_graph(Wdd, k=12, sym=True, remove_diag=True, gamma=2.0)
L_drug = normalized_laplacian_sparse(G_drug)
L_disease = normalized_laplacian_sparse(G_disease)
L_aug = sparse.block_diag([L_drug, L_disease], format='csr')
print(f'Laplacian built in {time.time()-t0:.1f}s, nnz={L_aug.nnz}')

t0 = time.time()
T, trIndex = build_augmented_matrix(Wrr, Wdd, matDR)
print(f'Augmented T: {T.shape}, built in {time.time()-t0:.1f}s')

print('Starting BNNR_graph (verbose=2)...')
t0 = time.time()
WW, iter_num, info = BNNR_graph(
    alpha=1, beta=10, T=T, trIndex=trIndex,
    tol1=2e-3, tol2=1e-5, maxiter=300, a=0, b=1,
    L_r=L_aug, L_d=L_aug,
    lambda_r=1e-3, lambda_d=1e-3,
    inner_steps=10, lr=1e-2, verbose=2)
print(f'BNNR_graph done: {iter_num} iters in {(time.time()-t0)/60:.1f}min')
