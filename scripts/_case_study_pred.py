"""
Quick case-study prediction: GF-BNNR on full DNdataset (no CV masking).
Identifies top novel drug predictions for Parkinson's disease.
"""
import sys, time, numpy as np
print = lambda *a, **kw: [sys.stdout.write(" ".join(map(str, a)) + "\n"), sys.stdout.flush()]

from bnnr import BNNR, GF_BNNR, getGIPSim, load_dataset
from bnnr.filter import _normalised_laplacian, _graph_filter

DATASET_PATH = "data/DNdataset.mat"
ALPHA, BETA = 1, 10
TOL1, TOL2 = 2e-3, 1e-5
MAXITER = 300
A_BOUND, B_BOUND = 0, 1
GAMMA_GIP, W_GIP, GRAPH_ALPHA = 1.0, 0.3, 0.3  # α=0.3 for DNdataset

# Target diseases (0-indexed in DNdataset)
PARKINSON_IDX = 23
SCHIZOPHRENIA_IDX = 2019  # SCHIZOPHRENIFORM DISORDER
PARANOID_SCHIZ_IDX = 3790  # PARANOID SCHIZOPHRENIA

Wrr, Wdd, Wdr = load_dataset(DATASET_PATH)
n_dis, n_drug = Wdr.shape

# Load names
import scipy.io as sio
data = sio.loadmat(DATASET_PATH)
dnames = [str(d[0][0]) for d in data['Wdname']]
rnames = [str(r[0][0]) for r in data['Wrname']]

print(f"Running GF-BNNR on full {DATASET_PATH} ({n_dis}×{n_drug})...")
t0 = time.time()

# GF-BNNR full inference
M_filtered, M_raw, it_num = GF_BNNR(
    Wrr, Wdd, Wdr, alpha=ALPHA, beta=BETA,
    tol1=TOL1, tol2=TOL2, maxiter=MAXITER, a=A_BOUND, b=B_BOUND,
    gamma_gip=GAMMA_GIP, w_gip=W_GIP, graph_alpha=GRAPH_ALPHA)

elapsed = (time.time() - t0) / 60
print(f"Done in {elapsed:.1f} min, {it_num} BNNR iterations")

def top_novel_predictions(M, disease_idx, known_matrix, drug_names, disease_name, N=15):
    """Get top-N novel (not in known) drug predictions for a disease."""
    scores = M[disease_idx, :]
    known_drugs = set(np.where(known_matrix[disease_idx, :] > 0)[0])

    # Sort by score descending, exclude known
    ranked = np.argsort(scores)[::-1]
    novel = [(i, scores[i]) for i in ranked if i not in known_drugs]

    print(f"\n{'='*60}")
    print(f"Top-{N} novel predictions for: {disease_name} (idx={disease_idx})")
    print(f"Known associations: {len(known_drugs)}")
    print(f"{'='*60}")
    for rank, (drug_idx, score) in enumerate(novel[:N], 1):
        print(f"  {rank:2d}. {drug_names[drug_idx]:30s}  score={score:.6f}")
    return novel[:N]

parkinson_novel = top_novel_predictions(M_filtered, PARKINSON_IDX, Wdr, rnames,
                                         dnames[PARKINSON_IDX])
schiz_novel = top_novel_predictions(M_filtered, SCHIZOPHRENIA_IDX, Wdr, rnames,
                                     dnames[SCHIZOPHRENIA_IDX])
paranoid_novel = top_novel_predictions(M_filtered, PARANOID_SCHIZ_IDX, Wdr, rnames,
                                        dnames[PARANOID_SCHIZ_IDX])

print(f"\nTotal runtime: {elapsed:.1f} min")
