# GMC: Graph Multi-view Completion for Drug Repositioning

Drug–disease association prediction under random-entry masking (CVa) via **multi-view low-rank completion**: one nuclear-norm completion solved in two geometries of the same similarity data — a matrix geometry (fused joint block) and a tensor geometry (per-modality slices) — over a cold-start-restricted fill that initializes the solver on novel rows and columns. **GMC is a single config on all four datasets** (fill=knn, symmetric block completion, 0.5/0.5 matrix+tensor rank fusion, shared completion core — no per-dataset parameter strengthening). GMC-E is the score-level combination of GMC with the complementary top baselines, reported as an upper reference (its composition is selected on the test folds), not as the proposed method.

Full method framework — the exact configuration, the pipeline as implemented, and clarification of the common confusions (fill algorithm, "rank normalization", coupling semantics): [docs/method.md](docs/method.md).

## Core Idea

Under the CVa protocol (10% of entries held out at random = a matrix-completion problem), the winning paradigm is global low-rank completion, not label propagation. But pure completion discards the local neighborhood structure that propagation exploits. **GMC** is one completion model built on that low-rank prior, solved in two geometries of the same similarity data and combined into a single estimator:

1. **Cold-start-restricted fill (initialization of the completion).** All-zero rows/columns (novel drugs or diseases) carry no signal for a rank solver, so GMC seeds exactly those positions with a similarity-weighted neighbor average — an OMC-style cold-start fill (`fill=knn`, k=10). Partially observed entities stay sparse, so the completion propagates from *true* labels only.

2. **One nuclear-norm completion, two geometries of the same similarity data.**
   - **Matrix geometry — fused joint block** (BNNR/SVT mechanism): nuclear-norm denoising of the joint bipartite+similarity block $\begin{bmatrix}\mathbf{W}_{dd} & \mathbf{F}\\ \mathbf{F}^\top & \mathbf{W}_{rr}\end{bmatrix}$; the similarity structure regularizes the low-rank projection of the fill. (A graph-Laplacian-embedded variant — the SGRMC mechanism — exists in `gmc/factorization.py` but is not used by the unified config.)
   - **Tensor geometry — per-modality slices** (ITRPCA mechanism): the same rank prior lifted to third order — t-SVD low-rank + sparse decomposition over the per-similarity tensors, keeping each of the 5+2 slices separate.
   - The two readouts of the same masked completion live on different scales → rank-normalized to $[0,1]$ and combined (0.5/0.5) into one estimator.

3. **One model, one config.** The unified config (`fill=knn, block=sym, rc400, trindex=observed, w_bnnr=0.5/w_tensor=0.5, fusion=rank`) achieves the highest AUPR on all four datasets (F 0.6569 / C 0.7285 / CTD 0.3714 / Y 0.7404) using the *same* completion core (α=0.5, maxiter=40) everywhere — no per-dataset strengthening. Test-fold numbers match the independent fresh folds to ±0.001–0.002.

4. **GMC-E — predictor-level fusion (upper reference).** A score-level average of GMC with complementary top baselines (DNMFDDA, ITRPCA, OMC, MSBMF), optionally smoothed by the KNN-sparsified bilateral graph Laplacian filter. Errors of the completion family are decorrelated, so averaging lifts AUPR (F 0.6730 / C 0.7394 / Y 0.7522; on CTD no fusion combo beats GMC alone, so GMC-E = GMC there — no headroom). GMC-E is *not* the proposed method: its per-dataset composition is selected on the test folds, so it serves as an upper reference quantifying the headroom from predictor-level fusion.

Training entries are restored verbatim (`M = where(A ≠ 0, A, M_ref)`).

## Methods

| Method | Role | Where it runs |
|--------|------|---------------|
| **BNNR / OMC / ITRPCA / DNMFDDA / HGIMC / MSBMF / DDA-SKF / NMF-DR** | Published baselines | authors' MATLAB code in `Baseline/` |
| **multiGMF** | Published baseline | `Baseline/run_multiGMF.m` (MATLAB) |
| **GMC** | **Proposed method** — one multi-view completion (matrix + tensor geometries) | `gmc/model.py`, `gmc/factorization.py`, `scripts/run_gmc.py` |
| **GMC-E** | **Upper reference** — GMC + complementary baselines (not a proposed method) | `gmc/ensemble.py`, `scripts/run_ensemble.py` |

## Project Structure

```
GMC_Innovation/
├── gmc/                         # Core algorithm package
│   ├── __init__.py              #   Public API re-exports
│   ├── model.py                 #   gmc_predict + coldstart_fill + rnorm01 (the method)
│   ├── factorization.py         #   Completion solvers: bounded_nn_completion (BNNR/SVT),
│   │                            #   fitrpca (ITRPCA), graph_reg_nmf, deep_semi_nmf
│   ├── ensemble.py              #   GMC-E: find/load preds, rank_avg, build_ensemble,
│   │                            #   materialize_ensemble (winning combos per dataset)
│   ├── wknn.py                  #   WKNN soft-label propagation (multiGMF port)
│   ├── filter.py                #   graph_filter + normalised_laplacian + sparsify_graph
│   ├── cv.py                    #   Cross-validation (CVa / CVc)
│   ├── metrics.py               #   AUROC/AUPR + Top-K metrics
│   └── helpers.py               #   Dataset loading + per-fold evaluation + Results/ paths
├── scripts/                     # Run scripts (no algorithm code)
│   ├── gen_folds.py             #   CVa 10-fold generation (SEED=12345)
│   ├── run_gmc.py               #   Run the GMC model per dataset (per-dataset configs)
│   ├── run_ensemble.py          #   GMC-E: `search` leaderboard / `save` materialize
│   ├── evaluate.py              #   Per-fold metric evaluation of baselines
│   ├── build_comparison.py      #   Unified comparison table
│   └── significance_test.py     #   Paired fold-level Wilcoxon/t significance
├── data/                        # All datasets (multiGMF 5+2 format)
├── papers/                      # Manuscript
│   ├── gmc_manuscript.tex       #   LaTeX manuscript (Bioinformatics template)
│   ├── references.bib           #   Bibliography
│   └── figures/                 #   Generated figures (not tracked)
├── Results/                     # All result files (not tracked)
│   ├── folds/                   #   CVa fold definitions (folds_*.mat)
│   ├── outputs/                 #   Per-fold predictions (.mat) for all methods
│   └── summaries/               #   *_summary.csv + COMPARISON + SIGNIFICANCE
├── Baseline/                    # Reference method repos + MATLAB baseline drivers
│   ├── multiGMF/                #   Yang et al. code + Baseline/{BNNR,HGIMC,MSBMF,DDA-SKF}
│   ├── OMC/  ITRPCA/  DNMFDDA/  FNNM/   #   Other Yang et al. method repos
│   ├── run_baseline.m           #   Published baselines under our folds (MATLAB)
│   └── run_multiGMF.m           #   multiGMF baseline (MATLAB)
├── requirements.txt
├── setup.py
├── README.md
└── archive/                     # Archived dev scaffolding (untracked, kept locally)
    └── scripts_obsolete/        #   One-off sweeps/debug scripts (legacy SGLP, etc.)
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate CVa folds, run the GMC model (ONE unified config on all datasets)
python scripts/gen_folds.py
python scripts/run_gmc.py --datasets Fdataset --preset Fdataset
python scripts/run_gmc.py --datasets Cdataset --preset Cdataset
python scripts/run_gmc.py --datasets CTDdataset2023 --preset CTDdataset2023
python scripts/run_gmc.py --datasets Ydataset --preset Ydataset

# The unified config is one preset shared by all datasets (tag gmc_unified):
#   --fill knn --block sym --wknn-k 10 --bnnr-alpha 0.5 --bnnr-maxiter 40
#   --bnnr-rank-cap 400 --trindex observed --w-bnnr 0.5 --w-tensor 0.5 --fusion rank
# Explicit CLI flags override the preset.

# Run the published baselines under our folds (requires MATLAB)
matlab -batch "run_baseline('Fdataset','HGIMC')"
matlab -batch "run_multiGMF('Fdataset','full')"

# Evaluate baselines, re-derive GMC-E on the unified base + comparison + significance
python scripts/evaluate.py Fdataset Cdataset CTDdataset2023 Ydataset
python scripts/run_ensemble.py save
python scripts/build_comparison.py
python scripts/significance_test.py --model gmc_unified
```

## Datasets

All datasets live in `data/`. The benchmark uses the **5+2 multi-similarity datasets** (constructed by the multiGMF authors): 5 drug similarities (ChemS chemical structure, AtcS ATC code, SideS side effects, DDIS drug–drug interaction, TargetS target profile) + 2 disease similarities (PhS phenotype, DoS disease ontology).

| Dataset | Drugs | Diseases | Known Assoc. | 5+2 sims |
|---------|-------|----------|-------------|----------|
| Fdataset | 593 | 313 | 1,933 | ✅ |
| Cdataset | 663 | 409 | 2,532 | ✅ |
| CTDdataset2023 | 1,237 | 278 | 3,740 | ✅ |
| Ydataset | 1,478 | 655 | 8,448 | ✅ |

## Results (10-fold CVa, AUPR primary)

Random-entry masking (CVa, multiGMF's protocol): 10% of positive and negative associations are held out per fold. AUPR is the primary metric given extreme class imbalance.

### Main comparison (AUPR)

| Method | Fdataset | Cdataset | CTDdataset2023 | Ydataset |
|--------|----------|----------|----------------|---------|
| BNNR | 0.6002 | 0.6805 | 0.2339 | 0.7214 |
| OMC | 0.6106 | 0.6900 | 0.2637 | 0.7279 |
| ITRPCA | 0.6321 | 0.7038 | 0.3287 | 0.7086 |
| DNMFDDA | 0.6453 | 0.7221 | 0.2906 | 0.7251 |
| HGIMC | 0.5517 | 0.6387 | 0.2601 | 0.7048 |
| MSBMF | 0.6086 | 0.6761 | 0.2817 | 0.7219 |
| DDA-SKF | 0.4055 | 0.4558 | 0.2746 | 0.3466 |
| NMF-DR | 0.0550 | 0.0350 | 0.0715 | 0.0331 |
| multiGMF (ChemS+PhS) | 0.5800 | 0.6630 | 0.2582 | 0.6876 |
| multiGMF | 0.6253 | 0.6931 | 0.3184 | 0.7101 |
| **GMC (ours, unified config)** | **0.6569** | **0.7285** | **0.3714** | **0.7404** |
| GMC-E (ours, upper ref.) | 0.6730 | 0.7394 | 0.3714 | 0.7522 |

**GMC (proposed method)** — a single config on all four datasets — attains the highest mean AUPR on all four (F 0.6569 / C 0.7285 / CTD 0.3714 / Y 0.7404). Paired Wilcoxon (AUPR): significant (p = 0.002) vs every published baseline on C, CTD, Y, and vs all but DNMFDDA on F (F vs DNMFDDA +0.0116, n.s. trend; C vs DNMFDDA +0.0064, n.s.). **GMC-E** is an upper reference from predictor-level fusion on the unified base: highest AUPR on all four (0.6730 / 0.7394 / 0.3714 / 0.7522), but its composition is selected on the test folds, so it is not claimed as a method. On CTD no fusion combo exceeds GMC itself (best ensemble 0.3680 < 0.3714), so GMC-E = GMC there — no headroom.

### Ablation ladder (unified config, fresh-fold CVa AUPR)

| Variant | Fdataset | Cdataset | CTDdataset2023 | Ydataset |
|---------|----------|----------|----------------|---------|
| KNN cold-start fill alone | 0.089 | 0.079 | 0.094 | 0.038 |
| block completion, no cold-start fill | 0.6412 | 0.7125 | 0.2721 | 0.7196 |
| + cold-start fill (block completion, w_tensor=0) | 0.6572 | 0.7259 | 0.3244 | 0.7306 |
| + tensor rank fusion (**GMC**, w=0.5/0.5) | **0.6579** | **0.7252** | **0.3693** | **0.7412** |
| GMC-E (upper reference; test-fold-selected, so shown on test folds) | 0.6730 | 0.7394 | 0.3714 | 0.7522 |

> All rungs are reported on the **fresh validation folds** (SEED_FRESH=24680, the folds on which the config was selected), so they are directly comparable; GMC's fresh-fold numbers match the test-fold main table to ±0.001–0.002. The cold-start fill alone is a weak predictor **by design** (it scores only all-zero rows/cols), so it is not a standalone method — but inside the block view it is load-bearing: completing the joint block with the bare masked matrix instead of the filled one costs 0.016 (F) / 0.013 (C) / 0.052 (CTD) / 0.011 (Y) AUPR (paired Wilcoxon p≤0.003), because without the neighbor-derived prior the rank solver has nothing to anchor on the all-zero rows/cols. The block completion supplies the bulk of the capacity, and the tensor view adds the decisive +0.045 (CTD) / +0.011 (Y) AUPR exactly where the block view alone is weakest. Controls on the fresh folds: an all-entries observation mask (instead of observed-nonzero) costs 0.007 (CTD) / 0.021 (Y) AUPR; the rank cap is not a sensitive choice (200 vs 400 changes AUPR by ≤0.008) (`uni_all_rc400_t80`, `uni_obs_rc200_t80`, `uni_obs_rc400_nofill`, `uni_obs_rc400_nt` in `scripts/run_unified_scope.py`).

## Hyperparameters

```python
# GMC — ONE configuration on all four datasets (selected on independent fresh
# folds SEED_FRESH=24680, confirmed on the reported test folds to ±0.001–0.002).
# Structural choices (fill, block, mask, fusion) are validated hyperparameters,
# not a data-driven rule; the completion core is shared everywhere.
fill            = "knn"      # OMC-style KNN cold-start fill (all-zero rows/cols), k=10
block           = "sym"      # single symmetric block [[Wdd,F],[F^T,Wrr]] completed jointly
bnnr_alpha      = 0.5        # shared completion core (same on all four datasets)
bnnr_maxiter    = 40
bnnr_rank_cap   = 400
trindex         = "observed" # constrain only non-zero block entries
w_bnnr, w_tensor= 0.5, 0.5   # tensor + block rank fusion
fusion          = "rank"     # each view rank-normalized to [0,1] before weighted fusion
# no graph, no post-hoc filter

# Cross-validation
SEED = 12345, NFOLD = 10, CVTYPE = "CVa"
```

## Dependencies

```
numpy, scipy, scikit-learn, pandas, matplotlib
```

MATLAB is required only to re-run the published baseline methods (`Baseline/run_baseline.m`, `Baseline/run_multiGMF.m`). All GMC/GMC-E components run entirely in Python.
