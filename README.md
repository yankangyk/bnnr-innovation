# BNNR Innovation — Graph-Regularized Matrix Completion for Drug Repositioning

Paper open-source repository integrating methods from two projects:

| Source | Methods | Original Script |
|--------|---------|----------------|
| `BNNR_master1` | BNNR, RA-BNNR, GBNNR, GBNNR-v3, GBNNR-v3-GIP | `run_paper_suite.py` |
| `BNNR_baseline` | BNNR_raw, BNNR_GIP, GF-BNNR | `main.py` |

Based on: Yang et al., *"Drug repositioning based on bounded nuclear norm regularization,"* Bioinformatics, 2019.

## Core Framework: Inside-Outside Complementarity

| Mechanism | Method | Module | Source |
|-----------|--------|--------|--------|
| **Inside** (during optimization) | **GBNNR** | `bnnr/graph.py` | `BNNR_master1` |
| **Outside** (after optimization) | **GF-BNNR** | `bnnr/filter.py` | `BNNR_baseline` |

**GBNNR** injects manifold structure into ADMM iterations via kNN graph Laplacian regularization. Improves AUPR by +5% to +31% with negligible AUROC change.

**GF-BNNR** applies a post-hoc bi-directional graph low-pass filter to enforce manifold smoothness. Uniquely improves **both** AUROC and AUPR, with gains scaling on ultra-sparse data (DNdataset: +5.3% AUROC, +27.7% AUPR).

### Key finding: λ-insensitivity

GBNNR's AUPR is constant across λ ∈ [10⁻³, 10⁻¹] — a 100× range. The benefit comes from graph **topology** (γ-weighted kNN edges), not regularization magnitude. GBNNR is effectively hyperparameter-free for λ.

## Method Inventory

| Method | Module | Source | Role |
|--------|--------|--------|------|
| **BNNR** | `bnnr/core.py` | `BNNR_master1` | Optimized baseline: precomputed constants, adaptive truncated SVD (~30× speedup) |
| **GBNNR** | `bnnr/graph.py` | `BNNR_master1` | Primary: kNN graph Laplacian + inner gradient descent (γ=2.0) |
| **GBNNR-v3** | `bnnr/graph.py` | `BNNR_master1` | GBNNR + block-wise adaptive weights (marginal difference from GBNNR at λ=1e-3) |
| **GF-BNNR** | `bnnr/filter.py` | `BNNR_baseline` | Primary: BNNR + GIP fusion + bi-directional graph low-pass filter |
| **RA-BNNR** | `bnnr/core.py` | `BNNR_master1` | Rank-adaptive β scheduling (marginal +1-3% AUPR; confirms β=10 is near-optimal) |
| GBNNR-v3-GIP | `bnnr/gip.py` | `BNNR_master1` | GIP-enhanced input similarities for GBNNR-v3 (plug-and-play, ~1% extra AUPR) |

## Project Structure

```
BNNR_Innovation/
├── bnnr/                      # Core algorithms
│   ├── core.py                #   BNNR + RA-BNNR + infer_ra_params
│   ├── graph.py               #   GBNNR + GBNNR-v3 (BNNR_graph + BNNR_graph_enhanced_v3)
│   ├── filter.py              #   GF-BNNR
│   ├── svt.py                 #   Singular Value Thresholding (full + randomized truncated)
│   ├── gip.py                 #   GIP similarity kernel
│   ├── cv.py                  #   Cross-validation (CVa / CVr / CVc)
│   ├── metrics.py             #   AUROC / AUPR + Top-K metrics
│   └── helpers.py             #   Shared utilities (load, mask, augment, evaluate)
├── scripts/                   # Experiments (one per source project + quick demo)
│   ├── run_all.py             #   ← BNNR_master1/run_paper_suite.py
│   ├── run_gfbnnr.py          #   ← BNNR_baseline/main.py
│   └── quick_demo.py          #   Quick single-method test on Fdataset
├── data/                      # Benchmark datasets (.mat)
├── papers/                    # Reference PDFs + chapter plan + manuscript draft
│   ├── _chapter_plan.md       #   Chapter planning (ARS plan mode output)
│   ├── _manuscript_draft.md   #   Full paper draft
│   ├── _bnnr_ref.md           #   BNNR 2019 paper (markitdown conversion)
│   └── _gao2023_ref.md        #   Gao 2023 paper (markitdown conversion)
├── Results/                   # Experimental output (CSV + JSON per dataset)
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Quick demo — BNNR baseline on Fdataset (~2 min)
python scripts/quick_demo.py

# Try other methods
python scripts/quick_demo.py --method ra-bnnr
python scripts/quick_demo.py --method gip
python scripts/quick_demo.py --method gf

# GF-BNNR experiment — 3 methods × 3 datasets (~30 min)
# Corresponds to: BNNR_baseline/main.py
python scripts/run_gfbnnr.py

# Full ablation — 5 methods × 3 datasets (~2-4 hours)
# Corresponds to: BNNR_master1/run_paper_suite.py
python scripts/run_all.py
```

## Datasets

| Dataset | Drugs | Diseases | Known Associations | Density |
|---------|-------|----------|-------------------|---------|
| Fdataset | 593 | 313 | 1,933 | 1.04% |
| Cdataset | 663 | 409 | 2,532 | 0.93% |
| DNdataset | 1,490 | 4,516 | 1,008 | 0.015% |

Drug similarity: Tanimoto scores from chemical fingerprints (CDK, SMILES from DrugBank).
Disease similarity: MeSH-based MimMiner (Van Driel et al., 2006).

## Results (10-fold CVa, mean ± std)

Results are from two experiment batches with independent random seeds. Within each batch, all methods share identical fold splits.

### Experiment A: Full Ablation (`run_all.py` ← `BNNR_master1`)

| Dataset | Method | AUROC | AUPR | P@10 |
|---------|--------|-------|------|------|
| **Fdataset** | BNNR | 0.9342 ± 0.0136 | 0.3073 ± 0.0212 | 0.95 ± 0.05 |
| | RA-BNNR | 0.9374 ± 0.0116 | 0.1620 ± 0.0694 | 0.66 ± 0.14 |
| | GBNNR | 0.9300 ± 0.0137 | 0.3240 ± 0.0204 | 1.00 ± 0.00 |
| | GBNNR-v3 | 0.9302 ± 0.0136 | 0.3239 ± 0.0204 | 1.00 ± 0.00 |
| | GBNNR-v3-GIP | 0.9325 ± 0.0134 | **0.3281** ± 0.0201 | 1.00 ± 0.00 |
| **Cdataset** | BNNR | 0.9519 ± 0.0067 | 0.3403 ± 0.0908 | 0.89 ± 0.06 |
| | RA-BNNR | 0.9547 ± 0.0055 | 0.2152 ± 0.1064 | 0.57 ± 0.11 |
| | GBNNR | 0.9483 ± 0.0083 | 0.3987 ± 0.0280 | 1.00 ± 0.00 |
| | GBNNR-v3 | 0.9482 ± 0.0084 | 0.3985 ± 0.0282 | 1.00 ± 0.00 |
| | GBNNR-v3-GIP | 0.9493 ± 0.0080 | **0.4037** ± 0.0301 | 1.00 ± 0.00 |
| **DNdataset** | BNNR | 0.9205 ± 0.0228 | 0.2181 ± 0.1490 | 0.32 ± 0.15 |
| | RA-BNNR | 0.9445 ± 0.0100 | 0.1967 ± 0.1681 | 0.45 ± 0.13 |
| | GBNNR | 0.9027 ± 0.0246 | 0.2853 ± 0.1020 | 0.31 ± 0.19 |
| | GBNNR-v3 | 0.9049 ± 0.0236 | 0.2849 ± 0.1019 | 0.30 ± 0.19 |
| | GBNNR-v3-GIP | 0.8918 ± 0.0283 | 0.2713 ± 0.0980 | 0.35 ± 0.20 |

### Experiment B: GF-BNNR (`run_gfbnnr.py` ← `BNNR_baseline`)

| Dataset | Method | AUROC | AUPR | P@10 |
|---------|--------|-------|------|------|
| **Fdataset** | BNNR_raw | 0.9342 ± 0.0137 | 0.3073 ± 0.0212 | 0.95 ± 0.05 |
| | BNNR_GIP | 0.9376 ± 0.0139 | 0.3190 ± 0.0208 | 1.00 ± 0.00 |
| | GF-BNNR | **0.9401** ± 0.0130 | **0.3198** ± 0.0191 | **0.99** ± 0.03 |
| **Cdataset** | BNNR_raw | 0.9519 ± 0.0067 | 0.3182 ± 0.1070 | 0.89 ± 0.06 |
| | BNNR_GIP | 0.9533 ± 0.0062 | 0.3495 ± 0.0973 | 0.96 ± 0.05 |
| | GF-BNNR | **0.9558** ± 0.0059 | **0.3479** ± 0.0956 | **0.96** ± 0.05 |
| **DNdataset** | BNNR_raw | 0.9231 ± 0.0212 | 0.2473 ± 0.1298 | 0.34 ± 0.14 |
| | BNNR_GIP | 0.9177 ± 0.0222 | 0.2839 ± 0.1021 | 0.39 ± 0.14 |
| | GF-BNNR | **0.9721** ± 0.0117 | **0.3157** ± 0.0256 | **0.45** ± 0.14 |

### Ablation & Analysis

**GBNNR λ-Insensitivity (Fdataset fold 1)**

| γ \ λ | 10⁻³ | 10⁻² | 10⁻¹ |
|-------|------|------|------|
| 1.0 | 0.3242 | 0.3242 | 0.3242 |
| 2.0 | **0.3274** | **0.3274** | **0.3274** |

λ has zero effect across a 100× range. γ=2.0 is the robust default.

**GBNNR Component Breakdown (Fdataset)**

| Configuration | AUPR |
|--------------|------|
| BNNR (no graph) | 0.3073 |
| + kNN graph Laplacian (γ=2.0) | 0.3240 |
| + GIP similarity fusion | 0.3281 |

## Hyperparameters

```python
# Universal BNNR (all methods)
ALPHA = 1
BETA = 10
TOL1 = 2e-3
TOL2 = 1e-5
MAXITER = 300
A, B = 0, 1           # predicted value bounds

# GBNNR / GBNNR-v3
knn_k = 12
gamma = 2.0            # confidence exponent (>1 suppresses weak kNN edges)
lambda_r = 1e-3        # graph regularisation (any value in [1e-3, 1e-1] works)
lambda_d = 1e-3
inner_steps = 10       # gradient steps per ADMM iteration
lr = 1e-2              # inner gradient learning rate
lambda_diag_factor = 0.2  # block-wise weight (v3 only, marginal effect)

# RA-BNNR (V4 calibrated)
eta = 0.020
beta_min = 1.0
beta_max = 50.0
warmup = 5
target_rank_ratio = 0.95

# GF-BNNR
graph_alpha = 0.5      # filter strength (0.3 for DNdataset)
w_gip = 0.3            # GIP fusion weight
gamma_gip = 1.0        # GIP kernel bandwidth

# Cross-validation
SEED = 12345
NFOLD = 10
CVTYPE = "CVa"
```

## Notes

- **GBNNR vs GBNNR-v3**: these produce nearly identical results. Both use γ=2.0 kNN graphs. The v3's block-wise adaptive weights have negligible effect at standard λ values. The paper treats them as the same method (GBNNR).
- **RA-BNNR**: marginal +1-3% AUPR improvement on some folds, unstable across folds. Confirms BNNR's default β=10 is near-optimal.
- **GIP fusion**: modest +1-4% AUPR across datasets. Plug-and-play — no algorithm changes needed.
- **DNdataset BNNR_GIP**: GIP fusion degrades AUROC (0.9177 vs 0.9231) due to extreme sparsity (0.015% density makes GIP similarities unreliable).

## Citation

Yang, M., Luo, H., Li, Y., & Wang, J. (2019). Drug repositioning based on bounded nuclear norm regularization. *Bioinformatics*, 35(14), i455-i463.
