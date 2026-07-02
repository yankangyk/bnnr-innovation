# BADGE: Bayesian Adaptive Drug-disease Graph Enhancement

Drug repositioning via density-adaptive iterative graph refinement for matrix completion.

Based on: Yang et al., *"Drug repositioning based on bounded nuclear norm regularization,"* Bioinformatics, 2019.

## Core Idea

Matrix completion methods treat similarity graphs as fixed inputs. BADGE allows graphs to evolve with the completion through **density-adaptive shrinkage**:

$$G_{new} = \lambda(\rho) \cdot G_{empirical} + (1-\lambda(\rho)) \cdot G_{prior}$$

where $\lambda(\rho) = \text{sigmoid}((\log_{10}\rho - \mu) / \tau)$ automatically adapts to data sparsity.

| Regime | Density | $\lambda$ | Behavior |
|--------|---------|-----------|----------|
| Ultra-sparse | $\ll$ 0.1% | $\to$ 0 | Trust prior GIP (safe, like GF-BNNR) |
| Moderate | $\gtrsim$ 0.5% | $\to$ 1 | Trust empirical GIP (extract richer structure) |

## Methods

| Method | Module | Description |
|--------|--------|-------------|
| **BADGE** | `bnnr/badge.py` | Proposed: iterative Bayesian GIP refinement with density-adaptive shrinkage |
| **BNNR** | `bnnr/core.py` | Baseline: bounded nuclear norm regularization (Yang et al., 2019) |
| **GBNNR** | `bnnr/graph.py` | Graph-regularized BNNR with kNN Laplacian + inner gradient descent |
| **GF-BNNR** | `bnnr/filter.py` | Graph-filtered BNNR with bi-directional Laplacian smoothing |

## Project Structure

```
BNNR_Innovation/
├── bnnr/                      # Core algorithms
│   ├── badge.py               #   BADGE (proposed method)
│   ├── core.py                #   BNNR + RA-BNNR
│   ├── graph.py               #   GBNNR + GBNNR-v3
│   ├── filter.py              #   GF-BNNR
│   ├── svt.py                 #   Singular Value Thresholding
│   ├── gip.py                 #   GIP similarity kernel
│   ├── cv.py                  #   Cross-validation
│   ├── metrics.py             #   AUROC / AUPR + Top-K metrics
│   └── helpers.py             #   Shared utilities
├── scripts/                   # Experiments
│   ├── run_badge.py           #   BADGE experiment suite (5 methods × 3 datasets)
│   ├── _gen_badge_figures.py  #   Publication figures
│   ├── _case_study_pred.py    #   Case study: top-N novel drug predictions
│   └── quick_demo.py          #   Quick single-method test
├── data/                      # Benchmark datasets (.mat)
├── papers/                    # Manuscript + figures + references
│   ├── badge_manuscript.md    #   Markdown manuscript
│   ├── badge_manuscript.tex   #   LaTeX manuscript
│   ├── references.bib         #   Bibliography
│   └── figures/               #   Publication figures (PDF + PNG)
├── Results/                   # Experimental output
│   └── BADGE/                 #   BADGE experiment results (3 datasets × 5 methods)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Quick demo — BNNR baseline on Fdataset (~2 min)
python scripts/quick_demo.py

# Try other methods
python scripts/quick_demo.py --method ra-bnnr
python scripts/quick_demo.py --method gip
python scripts/quick_demo.py --method gf

# Full BADGE experiment — 5 methods × 3 datasets, 10-fold CVa
python scripts/run_badge.py

# Quick test (Fdataset only, 3 folds)
python scripts/run_badge.py --quick

# Resume interrupted run
python scripts/run_badge.py --resume

# Generate figures
python scripts/_gen_badge_figures.py
```

## Datasets

| Dataset | Drugs | Diseases | Known Associations | Density |
|---------|-------|----------|-------------------|---------|
| Fdataset | 593 | 313 | 1,933 | 1.04% |
| Cdataset | 663 | 409 | 2,532 | 0.93% |
| DNdataset | 1,490 | 4,516 | 1,008 | 0.015% |

Drug similarity: Tanimoto scores from chemical fingerprints (CDK, SMILES from DrugBank).
Disease similarity: MeSH-based MimMiner (Van Driel et al., 2006).

## Results (10-fold CVa)

| Dataset | Method | AUROC | AUPR | P@10 | P@20 |
|---------|--------|-------|------|------|------|
| **Fdataset** | BNNR | 0.9319 | 0.3061 | 0.93 | 0.905 |
| | GBNNR | 0.9282 | 0.3199 | 1.00 | 0.975 |
| | GF-BNNR | **0.9370** | 0.3153 | 0.97 | 0.925 |
| | **BADGE (N=2)** | 0.9339 | **0.3233** | 0.99 | 0.930 |
| **Cdataset** | BNNR | 0.9502 | 0.2772 | 0.90 | 0.945 |
| | GBNNR | 0.9472 | 0.4006 | 1.00 | 0.995 |
| | GF-BNNR | **0.9545** | 0.3958 | 0.95 | 0.950 |
| | **BADGE (N=2)** | 0.9517 | **0.4051** | **1.00** | **1.00** |
| **DNdataset** | BNNR | 0.9288 | 0.2564 | 0.36 | 0.275 |
| | GBNNR | 0.9206 | 0.2539 | 0.31 | 0.220 |
| | GF-BNNR | **0.9725** | 0.3166 | 0.48 | 0.345 |
| | **BADGE (N=2)** | 0.9683 | **0.3207** | **0.50** | **0.355** |

BADGE achieves the highest AUPR on all three datasets. Best AUROC and top-K per dataset in **bold**.

## Hyperparameters

```python
# BNNR (all methods)
ALPHA = 1, BETA = 10
TOL1 = 2e-3, TOL2 = 1e-5
MAXITER = 300
A, B = 0, 1

# GBNNR
knn_k = 12, gamma = 2.0
lambda_r = 1e-3, lambda_d = 1e-3
inner_steps = 10, lr = 1e-2

# GF-BNNR / BADGE
graph_alpha = 0.5       # filter strength
w_gip = 0.3             # GIP fusion weight
gamma_gip = 1.0         # GIP kernel bandwidth

# BADGE-specific
shrinkage_mu = -3.0     # transition center (0.1% density)
shrinkage_tau = 0.3     # transition sharpness
n_iter = 2              # refinement iterations

# Cross-validation
SEED = 12345, NFOLD = 10, CVTYPE = "CVa"
```

## Citation

Yang, M., Luo, H., Li, Y., & Wang, J. (2019). Drug repositioning based on bounded nuclear norm regularization. *Bioinformatics*, 35(14), i455-i463.
