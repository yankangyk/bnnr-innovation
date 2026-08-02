# SGRMC: Sparsified Graph-Regularized Matrix Completion

Drug repositioning via single-pass matrix completion with density-adaptive KNN graph sparsification and graph Laplacian regularization.

## Core Idea

Drug repositioning is framed as a matrix completion problem: given a partially observed drug-disease association matrix $A$ and similarity graphs $S_{\text{drug}}, S_{\text{dis}}$ (computed from chemical fingerprints and phenotype ontologies), recover the complete matrix $M$ via nuclear norm minimization with ADMM.

**SGRMC** improves on this with three components, all applied in a single pass:

1. **Density-adaptive KNN graph sparsification.** Each similarity graph is sparsified to its $k$-nearest neighbors per node before constructing the Laplacian. Weak similarity edges (e.g., Tanimoto similarity $\approx 0.1$ between unrelated drugs) introduce noise into the graph Laplacian eigenbasis; removing them yields substantially cleaner regularization. The level $k$ is set adaptively by association density: aggressive ($k=5$) for moderate-density matrices, conservative ($k=50$) for ultra-sparse matrices.

2. **Graph Laplacian regularization.** Graph smoothness is enforced by embedding the Laplacian regularizer directly into ADMM:
   $$\min_M \ \underbrace{\|M\|_*}_{\text{low-rank}} + \underbrace{\frac{\alpha_{\text{mc}}}{2}\|P_\Omega(M-A)\|^2_F}_{\text{data fidelity}} + \underbrace{\alpha \cdot \text{tr}(M^\top L_{\text{dis}} M + M L_{\text{drug}} M^\top)}_{\text{graph smoothness}}$$

3. **Two solver paths, auto-selected.** Plan A ($\le 1000$ rows/cols) embeds the graph Laplacian in each ADMM iteration via first-order Neumann approximation; Plan B ($> 1000$) runs standard completion then applies a post-hoc bilateral exact Cholesky filter $(I+\alpha L_{\text{dis}})^{-1} M (I+\alpha L_{\text{drug}})^{-1}$.

## Methods

| Method | Role | Module |
|--------|------|--------|
| **BNNR** | Baseline (Yang et al., 2019) — nuclear norm matrix completion | `bnnr/core.py` |
| **SGRMC** | Proposed method — sparsified graph-regularized completion | `bnnr/sgrmc.py` |

## Project Structure

```
BNNR_Innovation/
├── bnnr/                      # Core package
│   ├── sgrmc.py               #   SGRMC (proposed) + deprecated BADGE wrapper
│   ├── core.py                #   BNNR + ADMM solvers
│   ├── filter.py              #   graph_filter + normalised_laplacian + sparsify_graph
│   ├── svt.py                 #   Singular Value Thresholding
│   ├── gip.py                 #   [DEPRECATED] GIP kernel, kept for reference
│   ├── graph.py               #   [LEGACY] GBNNR / BNNR_graph, backward compat
│   ├── cv.py                  #   Cross-validation (CVa/CVr/CVc)
│   ├── metrics.py             #   AUROC/AUPR + Top-K metrics
│   └── helpers.py             #   Shared utilities
├── scripts/                   # Experiments
│   ├── run_sgrmc.py           #   Full experiment suite
│   └── paper_figures.py       #   Publication figure generation
├── data/                      # Benchmark datasets (.mat)
├── papers/                    # Manuscript
│   ├── sgrmc_manuscript.tex   #   LaTeX manuscript (Bioinformatics template)
│   ├── references.bib         #   Bibliography
│   └── figures/               #   Generated figures (not tracked)
├── Results/                   # Experimental output (not tracked)
│   └── GRMC_CVc/              #   Per-fold CSVs + summary JSONs
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Full benchmark — BNNR vs SGRMC, 10-fold CVc, all datasets
python scripts/run_sgrmc.py

# Quick test (Fdataset only, 3 folds)
python scripts/run_sgrmc.py --quick

# KNN sparsification sweep
python scripts/run_sgrmc.py --experiments knn

# Resume interrupted run
python scripts/run_sgrmc.py --resume

# Regenerate publication figures
python scripts/paper_figures.py
```

## Datasets

| Dataset | Drugs | Diseases | Known Assoc. | Density |
|---------|-------|----------|-------------|---------|
| Fdataset | 593 | 313 | 1,933 | 1.04% |
| Cdataset | 663 | 409 | 2,532 | 0.93% |
| DNdataset | 1,490 | 4,516 | 1,008 | 0.015% |

Drug similarity: Tanimoto scores from CDK chemical fingerprints (SMILES from DrugBank).
Disease similarity: MeSH-based MimMiner (Van Driel et al., 2006).

## Results (10-fold CVc, mean ± std)

Disease-centric cross-validation (CVc): all unlabeled entries in held-out disease columns serve as negatives. AUPR is the primary metric given extreme class imbalance.

| Dataset | Method | AUROC | AUPR | P@10 | P@20 |
|---------|--------|-------|------|------|------|
| **Fdataset** | BNNR | 0.7820 ± 0.0369 | 0.1398 ± 0.0442 | 0.43 | 0.43 |
| | **SGRMC** (k=5, α=0.7) | 0.7867 ± 0.0340 | **0.1686** ± 0.0538 | 0.51 | 0.54 |
| **Cdataset** | BNNR | 0.7943 ± 0.0415 | 0.1326 ± 0.0477 | 0.53 | 0.52 |
| | **SGRMC** (k=5, α=0.7) | 0.8014 ± 0.0461 | **0.1829** ± 0.0604 | 0.67 | 0.64 |
| **DNdataset** | BNNR | 0.8391 ± 0.0399 | 0.2942 ± 0.1162 | 0.72 | 0.59 |
| | **SGRMC** (k=50, α=0.7) | 0.9493 ± 0.0180 | **0.3724** ± 0.0369 | 0.79 | 0.70 |

Relative AUPR improvement: +20.6% (Fdataset), +37.9% (Cdataset), +26.6% (DNdataset).

### KNN sparsification sensitivity (AUPR)

| k | Fdataset | Cdataset | DNdataset |
|---|----------|----------|-----------|
| 0 (dense) | 0.1419 | 0.1414 | 0.3661 |
| 5 | **0.1686** | **0.1829** | 0.2667 |
| 10 | 0.1665 | 0.1818 | 0.3215 |
| 20 | 0.1521 | 0.1761 | 0.3704 |
| 50 | 0.1416 | 0.1569 | **0.3724** |
| 100 | 0.1439 | 0.1503 | 0.3663 |

Density-dependent optimum: aggressive sparsification (k=5) helps moderate-density datasets (F, C); mild sparsification (k=50) is optimal for ultra-sparse DNdataset.

## Hyperparameters

```python
# SGRMC
ALPHA = 1, BETA = 10, TOL1 = 2e-3, TOL2 = 1e-5, MAXITER = 300, A, B = 0, 1
graph_alpha = 0.7       # graph filter strength (fixed)
knn_k = None            # auto: k=5 if density > 0.1%, else k=50

# Cross-validation
SEED = 12345, NFOLD = 10, CVTYPE = "CVc"
```

## Dependencies

```
numpy, scipy, scikit-learn, pandas, matplotlib
```
