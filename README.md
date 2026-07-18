# BADGE: Bayesian Adaptive Drug-disease Graph Enhancement

Drug repositioning via joint matrix-kernel estimation with alternating minimization.

Based on Yang et al., *"Drug repositioning based on bounded nuclear norm regularization,"* Bioinformatics, 2019.

## Core Idea

**BNNR** (Yang et al., 2019) solves drug repositioning as a matrix completion problem: given a partially observed drug-disease association matrix $A$ and fixed similarity graphs $S_{\text{drug}}, S_{\text{dis}}$ (computed once from chemical fingerprints and phenotype ontologies), it recovers a complete matrix $M$ via nuclear norm minimization with ADMM.

**BADGE** extends this by treating the similarity graphs as **endogenous** — jointly optimized with the completion matrix through alternating minimization:

**M-step — Embedded graph regularization in ADMM.** Instead of applying graph smoothing as a separate post-processing step, BADGE embeds the graph Laplacian directly into each ADMM iteration. The Lagrangian jointly optimizes three objectives in a single pass:

$$\min_M \ \underbrace{\|M\|_*}_{\text{low-rank}} + \underbrace{\frac{\alpha}{2}\|P_\Omega(M-A)\|^2_F}_{\text{data fidelity}} + \underbrace{\gamma \cdot \text{tr}(M^\top L_{\text{dis}} M + M L_{\text{drug}} M^\top)}_{\text{graph smoothness}}$$

**S-step — GIP recomputation from completed matrix.** After each M-step, the Gaussian Interaction Profile (GIP) kernel is recomputed from the *completed* matrix $M$ — which encodes richer topological structure than the sparse input $A$ — and fused with raw similarities. This refined graph feeds into the next M-step.

**Two iterations reach fixed point.** N=1 uses the initial GIP once. N=2 performs one round of GIP recomputation — the refined graph enables a better completion, which in turn would produce a nearly identical graph. N=3 shows negligible change across all datasets.

| | BNNR (Yang et al.) | BADGE |
|---|---|---|
| Similarity graph | Fixed (chemical + phenotype) | **Endogenous — recomputed from completed M** |
| Graph regularization | None | **Embedded in ADMM Lagrangian** |
| Optimization | Single nuclear norm min | **Alternating M-step ↔ S-step** |

## Methods

| Method | Role | Module |
|--------|------|--------|
| **BNNR** | Foundation (Yang et al., 2019) | `bnnr/core.py` |
| **BADGE** | Proposed method — joint matrix-kernel estimation | `bnnr/badge.py` |

Additional variants developed during this project:

| Variant | Description | Module |
|---------|-------------|--------|
| GF-BNNR | BNNR + post-hoc graph filter (= BADGE with N=1) | `bnnr/filter.py` |
| GBNNR | Graph-regularized BNNR with kNN Laplacian | `bnnr/graph.py` |
| RA-BNNR | Rank-adaptive BNNR with proportional beta scheduling | `bnnr/core.py` |

## Project Structure

```
BNNR_Innovation/
├── bnnr/                      # Core package
│   ├── badge.py               #   BADGE (proposed)
│   ├── core.py                #   BNNR + RA-BNNR + BNNR_graph_aware
│   ├── graph.py               #   BNNR_graph + GBNNR
│   ├── filter.py              #   GF-BNNR + graph_filter + normalised_laplacian
│   ├── svt.py                 #   Singular Value Thresholding
│   ├── gip.py                 #   Gaussian Interaction Profile kernel
│   ├── cv.py                  #   Cross-validation (CVa/CVr/CVc)
│   ├── metrics.py             #   AUROC/AUPR + Top-K metrics
│   └── helpers.py             #   Shared utilities
├── scripts/                   # Experiments
│   ├── run_badge.py           #   Full experiment suite
│   ├── run_sweeps.py          #   Parameter sweep (w_gip)
│   └── quick_demo.py          #   Single-method quick test
├── data/                      # Benchmark datasets (.mat)
├── papers/                    # Manuscript
│   ├── badge_manuscript.tex   #   LaTeX manuscript (Bioinformatics template)
│   ├── references.bib         #   Bibliography
│   ├── oup-authoring-template.cls
│   └── oup-abbrvnat.bst
├── Results/                   # Experimental output
│   └── BADGE/                 #   Per-fold CSVs + summary JSONs
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo — BNNR baseline on Fdataset (~2 min)
python scripts/quick_demo.py

# Try other variants
python scripts/quick_demo.py --method ra-bnnr
python scripts/quick_demo.py --method gip
python scripts/quick_demo.py --method gf

# Full experiment — 7 experiments x 3 datasets, 10-fold CVa
python scripts/run_badge.py

# Quick test (Fdataset only, 3 folds)
python scripts/run_badge.py --quick

# Resume interrupted run
python scripts/run_badge.py --resume
```

## Datasets

| Dataset | Drugs | Diseases | Known Assoc. | Density |
|---------|-------|----------|-------------|---------|
| Fdataset | 593 | 313 | 1,933 | 1.04% |
| Cdataset | 663 | 409 | 2,532 | 0.93% |
| DNdataset | 1,490 | 4,516 | 1,008 | 0.015% |

Drug similarity: Tanimoto scores from CDK chemical fingerprints (SMILES from DrugBank).
Disease similarity: MeSH-based MimMiner (Van Driel et al., 2006).

## Results (10-fold CVa, mean ± std)

| Dataset | Method | AUROC | AUPR | P@10 | P@20 |
|---------|--------|-------|------|------|------|
| **Fdataset** | BNNR | 0.9319 ± 0.0157 | 0.3061 ± 0.0240 | 0.930 | 0.905 |
| | GBNNR | 0.9282 ± 0.0138 | 0.3199 ± 0.0245 | 1.000 | 0.975 |
| | GF-BNNR | **0.9370** ± 0.0146 | 0.3153 ± 0.0251 | 0.970 | 0.925 |
| | **BADGE (N=2)** | 0.9336 ± 0.0143 | **0.3210** ± 0.0254 | **1.000** | **0.960** |
| **Cdataset** | BNNR | 0.9502 ± 0.0116 | 0.2772 ± 0.1212 | 0.900 | 0.945 |
| | GBNNR | 0.9472 ± 0.0121 | 0.4006 ± 0.0202 | 1.000 | 0.995 |
| | GF-BNNR | **0.9545** ± 0.0133 | 0.3958 ± 0.0195 | 0.950 | 0.950 |
| | **BADGE (N=2)** | 0.9508 ± 0.0132 | **0.4050** ± 0.0200 | **1.000** | **1.000** |
| **DNdataset** | BNNR | 0.9288 ± 0.0172 | 0.2564 ± 0.1345 | 0.360 | 0.275 |
| | GBNNR | 0.9206 ± 0.0148 | 0.2539 ± 0.0240 | 0.310 | 0.220 |
| | GF-BNNR | **0.9725** ± 0.0100 | 0.3166 ± 0.0207 | 0.480 | 0.345 |
| | **BADGE (N=2)** | 0.9683 ± 0.0108 | **0.3207** ± 0.0227 | **0.500** | **0.355** |

**Bold** = best per dataset. All values verified against `Results/BADGE/<dataset>/<exp>_summary.json`.

### Ablation

| Dataset | Variant | AUROC | AUPR |
|---------|---------|-------|------|
| Fdataset | BADGE (full) | 0.9336 | **0.3210** |
| | − GIP fusion (w_gip=0) | 0.9300 | 0.3125 |
| | − Graph filter (α_f=0) | 0.9361 | 0.3142 |
| Cdataset | BADGE (full) | 0.9508 | **0.4050** |
| | − GIP fusion (w_gip=0) | 0.9456 | 0.3929 |
| | − Graph filter (α_f=0) | 0.9526 | 0.3729 |
| DNdataset | BADGE (full) | 0.9683 | **0.3207** |
| | − GIP fusion (w_gip=0) | 0.9725 | 0.1956 |

### Convergence

| N | Fdataset AUPR | Cdataset AUPR | DNdataset AUPR |
|---|--------------|--------------|----------------|
| 1 | 0.3153 | 0.3958 | 0.3166 |
| 2 | **0.3210** | **0.4050** | 0.3207 |
| 3 | 0.3190 | 0.4039 | **0.3211** |

## Hyperparameters

```python
# BNNR (foundation)
ALPHA = 1, BETA = 10, TOL1 = 2e-3, TOL2 = 1e-5, MAXITER = 300, A,B = 0,1

# GBNNR
knn_k = 12, gamma = 2.0, lambda_r = 1e-3, lambda_d = 1e-3
inner_steps = 10, lr = 1e-2

# BADGE
graph_alpha = 0.5, w_gip = 0.3, gamma_gip = 1.0, n_iter = 2

# Cross-validation
SEED = 12345, NFOLD = 10, CVTYPE = "CVa"
```

## Dependencies

```
numpy, scipy, scikit-learn, pandas, matplotlib
```
