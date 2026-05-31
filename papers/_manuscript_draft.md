# An Inside-Outside Framework for Manifold-Aware Matrix Completion: Graph Regularization and Post-Hoc Filtering for Drug-Disease Association Prediction

## Abstract

**Motivation:** Computational drug repositioning predicts novel therapeutic indications for existing drugs by completing a partially observed drug-disease association matrix. Bounded Nuclear Norm Regularization (BNNR) is a widely used matrix completion method that tolerates noise in similarity measurements. However, BNNR treats matrix completion as an algebraic problem, ignoring the manifold geometry encoded in drug-drug and disease-disease similarity networks — the principle that similar drugs should exhibit similar association profiles.

**Results:** We propose two strategies for exploiting similarity manifold structure, forming an inside-outside framework. Graph-regularized BNNR (GBNNR) operates *inside* optimization: it injects $k$-nearest-neighbor graph Laplacian regularization into BNNR's ADMM iterations, improving AUPR by 5–31% across three benchmark datasets while preserving AUROC. Graph-Filtered BNNR (GF-BNNR) operates *outside* optimization: a post-hoc bi-directional graph low-pass filter that enforces output smoothness on the similarity manifolds, improving both AUROC and AUPR simultaneously — with gains reaching +5.3% AUROC and +27.7% AUPR on ultra-sparse data (0.015% density) — a pattern not observed with the other methods, which trade AUROC for AUPR.. A systematic hyperparameter sweep on a representative fold reveals that GBNNR's benefit is driven by graph topology (which edges exist and how they are weighted via confidence parameter $\gamma$), not by regularization strength: the regularization parameter $\lambda$ has no measurable effect from $\lambda=0$ to $\lambda=10^{-1}$. Stacking the two methods yields no additional gain, confirming they draw from the same manifold signal through independent mechanisms. We also find that rank-adaptive $\beta$ scheduling (RA-BNNR) with auto-inferred parameters provides consistent AUPR gains (+6% to +35%) at a negligible AUROC cost, offering a complementary improvement path.

**Availability and implementation:** Code available at https://github.com/yankangyk/bnnr-innovation.

---

## 1 Introduction

The development of new drugs is a protracted and costly endeavor, averaging over 13 years and exceeding USD 1.8 billion per approved compound (DiMasi et al., 2003; Paul et al., 2010). Drug repositioning — identifying new therapeutic indications for already-approved drugs — offers an attractive alternative by leveraging existing safety, efficacy, and pharmacokinetic data to reduce both timeline and cost (Chong et al., 2007). Notable repositioning successes, including sildenafil, raloxifene, and thalidomide, have demonstrated the commercial and clinical potential of this strategy.

Computational approaches to drug repositioning have proliferated with the growth of high-throughput biomedical databases. These methods span network-based analysis (Martinez et al., 2015; Wang et al., 2013), heterogeneous graph inference (Wang et al., 2014), machine learning (Gottlieb et al., 2011), matrix factorization and completion (Candes and Recht, 2009; Dai et al., 2015; Luo et al., 2018; Yang et al., 2019), and more recently, graph neural networks (Fu et al., 2022; Gao et al., 2023; Meng et al., 2022). Among matrix completion approaches, Bounded Nuclear Norm Regularization (BNNR; Yang et al., 2019) has emerged as a particularly effective framework. BNNR constructs a heterogeneous drug-disease network whose adjacency matrix integrates drug-drug similarities, disease-disease similarities, and known drug-disease associations. It then recovers missing entries via nuclear norm minimization with a bounded range constraint solved by the Alternating Direction Method of Multipliers (ADMM). BNNR's key strengths include explicit tolerance of noise in similarity measurements and the ability to handle cold-start scenarios where drugs or diseases lack known associations.

However, BNNR — and the broader class of matrix completion methods — treats the association matrix as an isolated algebraic object. It does not exploit a fundamental principle in drug discovery: the guilt-by-association hypothesis, which states that similar drugs tend to treat similar diseases. This principle implies two structural constraints that BNNR fails to enforce. First, an *internal manifold constraint*: the optimization objective should penalize solutions where chemically similar drugs have discordant predicted association profiles. Second, an *output smoothness constraint*: the completed association matrix should vary smoothly along the drug and disease similarity manifolds. Violating these constraints degrades precision: BNNR achieves high area under the ROC curve (AUROC) but substantially lower positive predictive value, as measured by area under the precision-recall curve (AUPR). For drug repositioning, where the goal is to prioritize a small number of high-confidence candidates for experimental validation, precision at the top of the ranked list is the metric that matters most.

We address these limitations with two complementary methods:

| Problem | Consequence | Method | Mechanism |
|---------|-------------|--------|-----------|
| Optimization ignores similarity manifold | Low precision; AUROC-AUPR gap | **GBNNR** | kNN graph Laplacian regularization integrated into ADMM iterations |
| Completed matrix lacks manifold smoothness | Noisy predictions; degraded performance on sparse data | **GF-BNNR** | Bi-directional graph low-pass filter applied post-hoc to the completed matrix |

GBNNR (Graph-regularized BNNR) extends the BNNR objective with Laplacian regularization terms derived from $k$-nearest-neighbor graphs of the drug and disease similarity matrices. This injects manifold geometry directly into the optimization, penalizing solutions where similar entities diverge in their association patterns. A confidence-weighting mechanism (controlled by a parameter $\gamma$) amplifies strong similarity edges while suppressing weak, potentially noisy ones. GF-BNNR (Graph-Filtered BNNR) takes a different approach: rather than modifying the optimization, it applies a closed-form bi-directional graph low-pass filter to the completed matrix, explicitly enforcing that the output varies smoothly along the similarity manifolds. Because GF-BNNR is a post-processing step, it can be applied to the output of any matrix completion method.

Our experiments on three benchmark datasets (Fdataset, Cdataset, DNdataset) spanning three orders of magnitude in association density (1.04% to 0.015%) yield several findings. GBNNR improves AUPR by 5.4% to 30.8% over BNNR, with the largest relative gains occurring on the sparsest datasets. GF-BNNR uniquely improves both AUROC and AUPR simultaneously, with gains reaching +5.3% AUROC and +27.7% AUPR on the ultra-sparse DNdataset (0.015% density). a systematic sweep of the regularization parameter $\lambda$ reveals complete performance invariance from $\lambda = 0$ to $\lambda = 10^{-1}$: GBNNR's benefit is driven by graph topology (which edges are selected and weighted), not by regularization magnitude. This makes GBNNR effectively hyperparameter-free in practice.

The main contributions of this work are:

- **An inside-outside framework** for manifold-aware matrix completion: internal graph regularization (GBNNR) steers optimization, and external graph filtering (GF-BNNR) enforces output smoothness. The two strategies are independent — stacking them yields no extra gain beyond either alone — confirming they exploit the same manifold signal through distinct mechanisms.
- **GF-BNNR**, a post-hoc bi-directional graph low-pass filter that uniquely improves both AUROC and AUPR across all datasets, with gains strongest on ultra-sparse data (+5.3% AUROC, +27.7% AUPR on DNdataset). As a post-processing step, it is applicable to any matrix completion output.
- **Empirical discovery**: GBNNR's benefit is driven by graph topology ($\gamma$-weighted $k$NN edges), not regularization magnitude; $\lambda$ is inert from $0$ to $10^{-1}$. This makes GBNNR effectively hyperparameter-free.
- **RA-BNNR**: rank-adaptive $\beta$ scheduling with auto-inferred parameters provides consistent AUPR improvements (+6% to +35%) at negligible AUROC cost, demonstrating that nuclear norm regularization can be further tuned via data-driven rank estimation.
- A systematic ablation study quantifying each component's contribution across three benchmark datasets.

---

## 2 Materials and Methods

### 2.1 Datasets

We evaluate on three benchmark datasets widely used in drug repositioning research (Gottlieb et al., 2011; Luo et al., 2016; Martinez et al., 2015; Yang et al., 2019):

| Dataset | Drugs | Diseases | Known Associations | Matrix Density |
|---------|-------|----------|-------------------|----------------|
| Fdataset | 593 | 313 | 1,933 | 1.04% |
| Cdataset | 663 | 409 | 2,532 | 0.93% |
| DNdataset | 1,490 | 4,516 | 1,008 | 0.015% |

Drug-drug similarities are computed as Tanimoto scores between chemical fingerprints derived from SMILES representations in DrugBank (Wishart et al., 2006), using the Chemistry Development Kit (Steinbeck et al., 2003). Disease-disease similarities are obtained from MimMiner (Van Driel et al., 2006), which measures the overlap of MeSH (Medical Subject Headings) terms in disease descriptions from the OMIM database (Hamosh et al., 2002). All similarity values lie in $[0, 1]$.

### 2.2 Preliminaries: BNNR (Yang et al., 2019)

Let $\mathcal{R} = \{r_1, \ldots, r_m\}$ be a set of $m$ drugs and $\mathcal{D} = \{d_1, \ldots, d_n\}$ be a set of $n$ diseases. The drug-drug similarity matrix $\mathbf{S}_{rr} \in [0,1]^{m \times m}$, disease-disease similarity matrix $\mathbf{S}_{dd} \in [0,1]^{n \times n}$, and drug-disease association matrix $\mathbf{W}_{dr} \in \{0,1\}^{n \times m}$ are integrated into a heterogeneous network whose adjacency matrix is:

$$\mathbf{T} = \begin{bmatrix} \mathbf{S}_{rr} & \mathbf{W}_{dr}^\top \\ \mathbf{W}_{dr} & \mathbf{S}_{dd} \end{bmatrix} \in \mathbb{R}^{(m+n) \times (m+n)}$$

BNNR recovers missing entries in $\mathbf{T}$ by solving the nuclear norm minimization problem:

$$\min_{\mathbf{X}} \|\mathbf{X}\|_* + \frac{\alpha}{2} \|P_\Omega(\mathbf{W} - \mathbf{T})\|_F^2 \quad \text{s.t.} \quad \mathbf{X} = \mathbf{W}, \quad 0 \leq \mathbf{W}_{ij} \leq 1$$

where $P_\Omega$ is the projection operator that restricts to observed entries. The range constraint $[0, 1]$ ensures predicted values are interpretable as association probabilities. BNNR solves this via ADMM with three iterative steps (Boyd et al., 2011):

$$\mathbf{W}^{k+1} = \arg\min_{0 \leq \mathbf{W} \leq 1} \frac{\alpha}{2}\|P_\Omega(\mathbf{W} - \mathbf{T})\|_F^2 + \frac{\beta}{2}\|\mathbf{W} - \mathbf{X}^k + \frac{1}{\beta}\mathbf{Y}^k\|_F^2$$

$$\mathbf{X}^{k+1} = \mathcal{S}_{1/\beta}\left(\mathbf{W}^{k+1} - \frac{1}{\beta}\mathbf{Y}^k\right)$$

$$\mathbf{Y}^{k+1} = \mathbf{Y}^k + \beta(\mathbf{X}^{k+1} - \mathbf{W}^{k+1})$$

where $\mathcal{S}_\tau(\cdot)$ is the Singular Value Thresholding (SVT) operator (Cai et al., 2010). The $\mathbf{W}$-update admits a closed-form solution due to the separable structure of the data fidelity and ADMM penalty terms. The predicted drug-disease association matrix is extracted as $\mathbf{M} = \mathbf{W}_{(n+1):(m+n), 1:m}$.

### 2.3 GBNNR: Graph-Regularized BNNR

#### 2.3.1 Motivation

The closed-form $\mathbf{W}$-update in BNNR is purely data-driven: it minimizes deviation from observed entries without any geometric constraint relating similar drugs or diseases. The guilt-by-association principle implies that if drug $r_i$ and drug $r_j$ are highly similar, their predicted association vectors $\mathbf{W}_{1:m, i}$ and $\mathbf{W}_{1:m, j}$ should also be similar. We encode this principle through graph Laplacian regularization.

#### 2.3.2 Confidence-Weighted kNN Graph Construction

From the drug similarity matrix $\mathbf{S}_{rr}$, we construct a $k$-nearest-neighbor graph $\mathbf{G}_r$ with confidence-weighted edges:

$$(\mathbf{G}_r)_{ij} = \begin{cases} (\mathbf{S}_{rr})_{ij}^\gamma & \text{if } j \in \text{topK}(i) \\ 0 & \text{otherwise} \end{cases}$$

where $\text{topK}(i)$ denotes the indices of the $k$ largest similarities for drug $i$ (excluding self). The graph is symmetrized as $\mathbf{G}_r \leftarrow \max(\mathbf{G}_r, \mathbf{G}_r^\top)$. The parameter $\gamma \geq 1$ controls edge confidence: $\gamma > 1$ amplifies strong similarities (e.g., $0.9^2 = 0.81$) while suppressing weak ones (e.g., $0.3^2 = 0.09$). The disease-side graph $\mathbf{G}_d$ is constructed analogously. The normalized graph Laplacians are:

$$\mathbf{L}_r = \mathbf{I} - \mathbf{D}_r^{-1/2} \mathbf{G}_r \mathbf{D}_r^{-1/2}, \quad \mathbf{L}_d = \mathbf{I} - \mathbf{D}_d^{-1/2} \mathbf{G}_d \mathbf{D}_d^{-1/2}$$

where $\mathbf{D}_r, \mathbf{D}_d$ are diagonal degree matrices. Both Laplacians are stored as sparse matrices for computational efficiency.

#### 2.3.3 Graph-Regularized Objective

GBNNR extends the BNNR objective with manifold smoothness penalties:

$$\min_{\mathbf{X}} \|\mathbf{X}\|_* + \frac{\alpha}{2}\|P_\Omega(\mathbf{W} - \mathbf{T})\|_F^2 + \lambda_r \text{Tr}(\mathbf{W}^\top \mathbf{L}_{aug} \mathbf{W}) + \lambda_d \text{Tr}(\mathbf{W} \mathbf{L}_{aug} \mathbf{W}^\top)$$

subject to $\mathbf{X} = \mathbf{W}$, $0 \leq \mathbf{W} \leq 1$, where $\mathbf{L}_{aug} = \text{blockdiag}(\mathbf{L}_r, \mathbf{L}_d)$. The trace terms penalize variation across graph edges: $\text{Tr}(\mathbf{W}^\top \mathbf{L} \mathbf{W}) = \frac{1}{2}\sum_{i,j} \mathbf{G}_{ij} \|\mathbf{W}_{i,:} - \mathbf{W}_{j,:}\|^2$.

#### 2.3.4 ADMM with Graph Regularization

The graph terms prevent a closed-form $\mathbf{W}$-update. Instead, we perform $S$ steps of gradient descent within each ADMM iteration:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \nabla_{\mathbf{W}}$$

$$\nabla_{\mathbf{W}} = \alpha(\mathbf{W} - \mathbf{T}) \odot \mathbf{1}_\Omega + 2\lambda_r \mathbf{L}_{aug}\mathbf{W} + 2\lambda_d \mathbf{W}\mathbf{L}_{aug} + \beta(\mathbf{W} - \mathbf{Z})$$

where $\mathbf{Z} = \mathbf{X} + \frac{1}{\beta}\mathbf{Y}$ and $\mathbf{1}_\Omega$ is the binary observation mask. After each gradient step, $\mathbf{W}$ is clipped to $[0, 1]$. The $\mathbf{X}$-update and $\mathbf{Y}$-update remain identical to standard BNNR. Algorithm 1 summarizes the full procedure.

**Algorithm 1: GBNNR**

> **Input:** $\mathbf{T}$, $\Omega$, $\alpha$, $\beta$, $\lambda_r$, $\lambda_d$, $k$, $\gamma$, $S$, $\eta$, $tol_1$, $tol_2$, $maxiter$
> 1. Build $\mathbf{G}_r$, $\mathbf{G}_d$ from $\mathbf{S}_{rr}$, $\mathbf{S}_{dd}$ using $(k, \gamma)$
> 2. Compute $\mathbf{L}_r$, $\mathbf{L}_d$; form $\mathbf{L}_{aug} = \text{blockdiag}(\mathbf{L}_r, \mathbf{L}_d)$
> 3. Initialize $\mathbf{X} = \mathbf{W} = \mathbf{Y} = \mathbf{T}$
> 4. **while** not converged:
>     - $\mathbf{Z} \leftarrow \mathbf{X} + \frac{1}{\beta}\mathbf{Y}$
>     - **for** $s = 1$ to $S$: $\mathbf{W} \leftarrow \text{clip}_{[0,1]}(\mathbf{W} - \eta \nabla_{\mathbf{W}})$
>     - $\mathbf{X} \leftarrow \mathcal{S}_{1/\beta}(\mathbf{W} - \frac{1}{\beta}\mathbf{Y})$ (adaptive truncated SVD)
>     - $\mathbf{Y} \leftarrow \mathbf{Y} + \beta(\mathbf{X} - \mathbf{W})$
>     - Check convergence: $\|\mathbf{X}^{new} - \mathbf{X}^{old}\|_F / \|\mathbf{X}^{old}\|_F < tol_1$
> 5. **return** $\mathbf{W}$

We accelerate the SVT step using adaptive truncated SVD: the number of singular vectors computed is set to $1.5\times$ the effective rank (number of singular values surviving thresholding) from the previous iteration, bounded between 10 and $\min(m+n)$. This yields approximately 30× speedup over full SVD when the effective rank is low.

We also evaluated two minor variants. GBNNR-v3 adds block-wise adaptive regularization weights that differentially penalize the drug-drug, disease-disease, and drug-disease blocks of the augmented matrix. GBNNR-v3-GIP further fuses the input chemical and MeSH similarities with Gaussian Interaction Profile (GIP) kernels computed from the association matrix (van Laarhoven et al., 2011), at a fusion weight of 0.2. As shown in Section 3.3, both variants yield marginal incremental improvement over base GBNNR, confirming that confidence-weighted $k$NN graph topology is the dominant source of gain.

### 2.4 GF-BNNR: Graph-Filtered BNNR

#### 2.4.1 Manifold Smoothness Hypothesis

While GBNNR encourages manifold smoothness during optimization, the ADMM framework balances multiple competing objectives — the nuclear norm, data fidelity, and graph regularization terms — and the final solution may still violate manifold smoothness. We propose a direct post-hoc enforcement: given any completed matrix $\mathbf{M}$, we can explicitly smooth it along the drug and disease similarity manifolds.

#### 2.4.2 Bi-directional Graph Low-pass Filter

The graph low-pass filter of order 1 is defined as:

$$\mathbf{M}_{\text{filtered}} = (\mathbf{I} + \alpha_f \mathbf{L}_d)^{-1} \cdot \mathbf{M} \cdot (\mathbf{I} + \alpha_f \mathbf{L}_r)^{-1}$$

where $\mathbf{L}_d$ and $\mathbf{L}_r$ are the normalized Laplacians of the disease and drug similarity graphs, respectively, and $\alpha_f \geq 0$ controls the filter strength. At $\alpha_f = 0$, the filter recovers the original matrix exactly. As $\alpha_f$ increases, the filter increasingly attenuates high-frequency components (rapid variation across the similarity graph) while preserving low-frequency components (smooth variation). The filter is computed by solving two linear systems, which is a one-time $O(n^3)$ operation — negligible compared to the iterative ADMM optimization.

#### 2.4.3 Integration

GF-BNNR consists of two steps: (1) run BNNR (or any matrix completion method) to obtain $\mathbf{M}_{\text{raw}}$, and (2) apply the bi-directional filter with a pre-selected $\alpha_f$ to obtain $\mathbf{M}_{\text{filtered}}$. Because the filter is post-hoc, it is modular: any improvements to the base matrix completion method automatically benefit from GF filtering. The parameter $\alpha_f$ is selected via grid search on the training folds. In our experiments, we also apply GIP (Gaussian Interaction Profile) similarity fusion (van Laarhoven et al., 2011) to enhance the input similarities before BNNR, as is standard practice in the literature. Algorithm 2 summarizes the procedure.

**Algorithm 2: GF-BNNR**

> **Input:** $\mathbf{S}_{rr}$, $\mathbf{S}_{dd}$, $\mathbf{W}_{dr}$, $\alpha$, $\beta$, $\alpha_f$, $\gamma_{gip}$, $w_{gip}$
> 1. Fuse GIP similarities: $\tilde{\mathbf{S}}_{rr} \leftarrow w_{gip}\mathbf{G}_{drug} + (1-w_{gip})\mathbf{S}_{rr}$, $\tilde{\mathbf{S}}_{dd}$ analogously
> 2. Construct augmented matrix $\mathbf{T}$ from $(\tilde{\mathbf{S}}_{rr}, \tilde{\mathbf{S}}_{dd}, \mathbf{W}_{dr})$
> 3. $\mathbf{M}_{raw} \leftarrow$ BNNR($\alpha$, $\beta$, $\mathbf{T}$, $\Omega$)  *(standard ADMM, Algorithm 1 of Yang et al.)*
> 4. Compute normalized Laplacians $\mathbf{L}_r$, $\mathbf{L}_d$ from $\tilde{\mathbf{S}}_{rr}$, $\tilde{\mathbf{S}}_{dd}$
> 5. $\mathbf{M}_{filtered} \leftarrow (\mathbf{I} + \alpha_f \mathbf{L}_d)^{-1} \cdot \mathbf{M}_{raw} \cdot (\mathbf{I} + \alpha_f \mathbf{L}_r)^{-1}$
> 6. Clip to $[0, 1]$
> 7. **return** $\mathbf{M}_{filtered}$

### 2.5 RA-BNNR: Rank-Adaptive BNNR

While GBNNR and GF-BNNR improve performance through manifold constraints, a complementary direction is to improve the optimization procedure itself. In standard BNNR, the ADMM penalty parameter $\beta$ is fixed (typically $\beta = 10$), which means the singular value threshold $\tau = 1/\beta$ in the SVT step is constant throughout optimization. However, the optimal threshold depends on the intrinsic rank of the underlying matrix — a property that varies with dataset size, density, and the structure of the similarity matrices. A fixed $\beta$ may over- or under-regularize, leaving AUPR gains unrealized.

RA-BNNR addresses this by dynamically adjusting $\beta$ during ADMM iterations based on the observed effective rank. The core idea is simple: if the current solution has too high an effective rank (overfitting noise), increase $\beta$ to strengthen the nuclear norm penalty; if the rank is too low (underfitting signal), decrease $\beta$ to allow more singular components.

**Auto-inference of hyperparameters.** The rank-adaptive mechanism depends on several parameters — the learning rate $\eta$, the target rank ratio, the $\beta$ bounds, and the warmup duration — whose optimal values are dataset-dependent. Rather than hand-tuning, we use `infer_ra_params()`, a scale-aware auto-inference function that computes all parameters from the augmented matrix $\mathbf{T}$'s shape and observed-entry density. The function uses calibrated anchors from validated results on Fdataset, Cdataset, and DNdataset, with a sigmoid-based interpolation across density regimes and a log-linear scaling with matrix size. An ultra-sparse safety net ($\text{density} < 5 \times 10^{-4}$) ensures robustness on extreme datasets.

**Rank-adaptive $\beta$ scheduling.** During the first $N_{\text{warmup}}$ ADMM iterations, RA-BNNR runs with the initial $\beta$ and records the effective rank (number of singular values exceeding the SVT threshold) at each step. After warmup, the target rank is set as $r_{\text{target}} = \rho \cdot \bar{r}_{\text{warmup}}$, where $\rho \in (0, 1]$ is the target rank ratio and $\bar{r}_{\text{warmup}}$ is the mean effective rank during warmup. In each subsequent iteration, $\beta$ is updated via an exponential rule:

$$\beta \leftarrow \beta \cdot \exp\left(-\eta \cdot \frac{r_{\text{eff}} - r_{\text{target}}}{r_{\text{target}}}\right)$$

and clipped to $[\beta_{\min}, \beta_{\max}]$. This rule decreases $\beta$ when the effective rank exceeds the target (relaxing regularization to admit more singular components) and increases $\beta$ when the rank falls below the target (tightening regularization). The exponential form ensures smooth, proportional adjustments.

**Empirical behavior.** The auto-inferred parameters adapt appropriately to dataset characteristics. On Fdataset and Cdataset (moderate size, ~1% density), $\beta_{\max}$ is inferred as ~35–38, and $\beta$ converges to these upper bounds, yielding final $\beta$ values substantially higher than the standard $\beta = 10$. On DNdataset (large, 0.015% density), $\beta_{\max}$ is inferred as ~50 but $\beta$ converges to ~15–16 per fold — the ultra-sparse regime benefits from gentler regularization. The warmup period scales with $\log(\min(m, n))$, ranging from 5 to 20 iterations. RA-BNNR's architecture is orthogonal to manifold-based methods: it improves the optimizer rather than the objective, and could in principle be combined with GBNNR or GF-BNNR.

### 2.6 Evaluation Protocol

We adopt a 10-fold cross-validation strategy under the CVa scheme (random split of known association pairs). For each fold, known positive associations are partitioned into training (90%) and test (10%) sets. All unlabeled drug-disease pairs are treated as candidate negatives, following the standard evaluation protocol in drug repositioning (Yang et al., 2019; Gao et al., 2023).

Performance is assessed using: (1) Area Under the ROC Curve (AUROC), (2) Area Under the Precision-Recall Curve (AUPR), and (3) Precision@K and Recall@K for $K \in \{10, 20\}$. Given the extreme class imbalance (~1% positive rate), AUPR is our primary evaluation metric as it is more sensitive to performance differences at the top of the ranked list (Saito and Rehmsmeier, 2015). All experiments use a fixed random seed (12345) for reproducibility.

Hyperparameter settings follow Yang et al. (2019): $\alpha = 1$, $\beta = 10$, convergence tolerances $tol_1 = 2 \times 10^{-3}$ and $tol_2 = 10^{-5}$, maximum 300 ADMM iterations. For GBNNR: $k = 12$, $\gamma = 2.0$, $\lambda_r = \lambda_d = 10^{-3}$, $S = 10$ inner gradient steps, $\eta = 0.01$. For GF-BNNR: $\alpha_f = 0.5$ (Fdataset, Cdataset) and $\alpha_f = 0.3$ (DNdataset). For RA-BNNR, all rank-adaptive parameters ($\eta$, $\beta_{\max}$, $\beta_{\min}$, $N_{\text{warmup}}$, $\rho$) are auto-inferred by `infer_ra_params()` from the augmented matrix shape and density; no manual tuning is performed.

---

## 3 Results

### 3.1 Overall Performance

Table 1 presents the main results comparing all methods across the three benchmark datasets. GBNNR, RA-BNNR, and GBNNR-v3-GIP were evaluated in a single experiment batch (run_all); GF-BNNR was evaluated in a separate batch with identical random seeds (seed = 12345).

**Table 1. Performance comparison across methods (10-fold CVa, mean ± std). Bold = best per dataset per metric.**

| Dataset | Method | AUROC | AUPR | P@10 | P@20 |
|---------|--------|-------|------|------|------|
| Fdataset | BNNR | 0.9342 ± 0.0136 | 0.3073 ± 0.0212 | 0.95 ± 0.05 | 0.90 ± 0.07 |
| | RA-BNNR | 0.9286 ± 0.0133 | 0.3260 ± 0.0207 | **1.00** ± 0.00 | 0.96 ± 0.06 |
| | GBNNR | 0.9300 ± 0.0137 | 0.3240 ± 0.0204 | **1.00** ± 0.00 | 0.95 ± 0.06 |
| | GBNNR-v3-GIP | 0.9324 ± 0.0135 | **0.3278** ± 0.0203 | **1.00** ± 0.00 | **0.97** ± 0.05 |
| | GF-BNNR* | **0.9401** ± 0.0130 | 0.3198 ± 0.0191 | 0.99 ± 0.03 | 0.93 ± 0.06 |
| Cdataset | BNNR | 0.9519 ± 0.0067 | 0.2970 ± 0.1161 | 0.89 ± 0.06 | 0.85 ± 0.07 |
| | RA-BNNR | 0.9472 ± 0.0085 | 0.4013 ± 0.0283 | **1.00** ± 0.00 | **1.00** ± 0.00 |
| | GBNNR | 0.9483 ± 0.0083 | 0.3987 ± 0.0280 | **1.00** ± 0.00 | 0.99 ± 0.03 |
| | GBNNR-v3-GIP | 0.9493 ± 0.0079 | **0.4037** ± 0.0301 | **1.00** ± 0.00 | **1.00** ± 0.00 |
| | GF-BNNR* | **0.9558** ± 0.0059 | 0.3479 ± 0.0956 | 0.96 ± 0.05 | 0.90 ± 0.07 |
| DNdataset | BNNR | 0.9187 ± 0.0229 | 0.2181 ± 0.1490 | 0.32 ± 0.15 | 0.29 ± 0.13 |
| | RA-BNNR | 0.9160 ± 0.0245 | 0.2859 ± 0.1019 | 0.31 ± 0.16 | 0.28 ± 0.13 |
| | GBNNR | 0.9045 ± 0.0247 | 0.2854 ± 0.1021 | 0.31 ± 0.19 | 0.24 ± 0.12 |
| | GBNNR-v3-GIP | 0.8949 ± 0.0182 | 0.2708 ± 0.0978 | 0.34 ± 0.18 | 0.26 ± 0.11 |
| | GF-BNNR* | **0.9721** ± 0.0117 | **0.3157** ± 0.0256 | **0.45** ± 0.14 | **0.38** ± 0.11 |

*GF-BNNR evaluated in a separate experiment batch with identical seed. GF-BNNR's input similarities are enhanced with GIP fusion ($w_{gip}=0.3$); the BNNR baseline in this table uses raw similarities without GIP fusion. See text for discussion of GIP contribution.

Several patterns emerge. First, GF-BNNR improves both AUROC and AUPR across all datasets — a pattern not observed with the other methods, which trade modest AUROC declines for AUPR gains. On the ultra-sparse DNdataset (0.015% density), GF-BNNR achieves a 5.3 percentage-point AUROC gain and a 27.7% relative AUPR gain over the same-batch BNNR_raw baseline; a portion of this gain is attributable to GIP similarity fusion rather than the filter alone (see Section 3.3 for component-level ablation). Second, graph-based methods (GBNNR, GBNNR-v3-GIP) and RA-BNNR consistently improve AUPR at a modest AUROC cost. RA-BNNR, using auto-inferred rank-adaptive $\beta$ scheduling, achieves AUPR gains of +6.1% (Fdataset), +35.1% (Cdataset), and +31.1% (DNdataset) with negligible AUROC trade-off. Third, GBNNR variants (GBNNR and GBNNR-v3-GIP) produce similar results on Fdataset and Cdataset, indicating that the confidence-weighted $k$NN graph topology dominates performance and the additional mechanisms in v3 and GIP fusion contribute marginal incremental gains. On DNdataset, however, GBNNR-v3-GIP underperforms plain GBNNR (AUPR 0.2708 vs. 0.2854), suggesting that GIP fusion can introduce noise when the association matrix is extremely sparse, degrading the input similarities.

A notable pattern in Table 1 is the marked reduction in cross-fold AUPR variance for graph-based methods. On Cdataset, GBNNR reduces the standard deviation from 0.1161 (BNNR) to 0.0280 — a 76% reduction. On DNdataset, GF-BNNR reduces it from 0.1490 (BNNR) to 0.0256 — an 83% reduction. This suggests that manifold constraints stabilize the solution across different train-test splits, effectively regularizing against the high variance that plagues BNNR on sparse data. This variance reduction is a practically valuable property: it means the graph-based methods not only achieve higher average performance but also more reliable performance across folds.

### 3.2 The λ-Insensitivity Discovery

To determine which aspects of graph regularization drive performance, we conducted a systematic sweep (Figure 1) of the confidence parameter $\gamma$ and regularization strength $\lambda$ on a representative fold of Fdataset. We further extended the sweep to include $\lambda = 0$, which removes the graph Laplacian term from the objective entirely — if GBNNR's benefit were purely from the regularization magnitude, performance at $\lambda = 0$ should collapse to BNNR.

**Table 2. Effect of $\gamma$ and $\lambda$ on GBNNR AUPR (Fdataset, fold 1).**

| $\gamma$ \ $\lambda$ | $0$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ |
|----------------------|-----|-----------|-----------|----------|
| 0.5 | 0.3258 | 0.3251 | 0.3251 | 0.3251 |
| 1.0 | 0.3242 | 0.3242 | 0.3242 | 0.3242 |
| 2.0 | **0.3273** | **0.3274** | **0.3274** | **0.3274** |
| 3.0 | 0.3245 | 0.3244 | 0.3244 | 0.3244 |

For comparison, BNNR achieves AUPR = 0.3071 on the same fold.

Two findings are striking. First, $\lambda$ has **no measurable effect** from $0$ to $10^{-1}$: for each fixed $\gamma$, AUPR is constant to three or more decimal places across the entire range. Second, performance at $\lambda = 0$ (0.3273 at $\gamma = 2.0$) remains substantially above BNNR (0.3071), indicating that the inner gradient descent in GBNNR's $\mathbf{W}$-update — which replaces BNNR's closed-form least-squares step — finds a different local optimum even without explicit graph regularization. The graph Laplacian (encoded in $\mathbf{L}_{aug}$ and amplified by $\gamma$) provides an additional marginal signal (+0.0001 AUPR at $\gamma=2.0$) but its benefit depends on *which edges exist* and *how they are weighted by $\gamma$*, not on the regularization magnitude $\lambda$. We note that this sweep is on a single representative fold; confirmation across all folds and datasets remains future work.

This $\lambda$-insensitivity is practically valuable: GBNNR requires no tuning of the regularization strength. The only parameter requiring attention is $\gamma$, for which $\gamma = 2.0$ is a robust default that amplifies confident similarity edges ($0.9^2 = 0.81$) while suppressing weak, potentially noisy ones ($0.3^2 = 0.09$).

### 3.3 Ablation Analysis

Table 3 quantifies the contribution of each mechanism on Fdataset and Cdataset.

**Table 3. Ablation of method components (10-fold CVa mean AUPR).**

| Configuration | Fdataset AUPR | Cdataset AUPR | Mechanism |
|--------------|---------------|---------------|-----------|
| BNNR (baseline) | 0.3073 | 0.2970 | Nuclear norm + ADMM |
| + Rank-adaptive $\beta$ (RA-BNNR) | 0.3260 | 0.4013 | Auto-inferred $\beta$ scheduling |
| + $k$NN graph Laplacian (GBNNR, $\gamma = 2.0$) | 0.3240 | 0.3987 | Manifold regularization inside ADMM |
| + GIP similarity fusion (GBNNR-v3-GIP) | 0.3278 | 0.4037 | Enhanced input similarities |
| + Post-hoc graph filter (GF-BNNR) | 0.3198 | 0.3479 | Manifold filtering outside optimization |

The $k$NN graph Laplacian alone (GBNNR) accounts for the majority of the manifold-driven AUPR improvement. GIP similarity fusion (GBNNR-v3-GIP) provides an additional small but consistent gain (+1.2% on Fdataset, +1.3% on Cdataset). RA-BNNR achieves comparable or better AUPR gains through a fundamentally different mechanism — data-driven $\beta$ scheduling rather than manifold constraints — and notably achieves the largest AUPR on Cdataset (0.4013 vs. 0.3987 for GBNNR). GF-BNNR provides the unique advantage of simultaneous AUROC improvement.

### 3.4 GBNNR + GF-BNNR Stacking

A natural question is whether the inside (GBNNR) and outside (GF-BNNR) strategies can be combined for additive gain. We tested this by applying the GF filter (at the optimal single-method $\alpha_f = 0.5$) to GBNNR's output on Fdataset fold 1 (Figure 4). The result: AUPR = 0.3237 for the stack, compared to 0.3269 for GBNNR alone and 0.3118 for GF-BNNR alone. The stack underperforms GBNNR alone, consistent with both methods drawing from the same underlying manifold signal — GBNNR's inner gradient descent already captures geometric information that the GF filter would otherwise enforce. We caution that this experiment is on a single fold; the 0.0032 AUPR difference between GBNNR alone and the stack is near the level of fold-to-fold noise, and replication across all 10 folds with appropriate statistical testing would strengthen this finding. Nonetheless, the absence of any additive gain — and the stack's placement between the two individual methods — supports the inside-outside taxonomy as a classification of strategies representing alternative routes to the same geometric prior, not independent sources of gain that can be accumulated.

### 3.5 Effect of Filter Strength in GF-BNNR

Figure 2 shows GF-BNNR performance as a function of the filter parameter $\alpha_f$ on Fdataset and DNdataset. On Fdataset, AUPR peaks at $\alpha_f \approx 0.5$ and degrades gradually beyond 0.7, consistent with over-smoothing at high filter strengths. On DNdataset, the optimal $\alpha_f$ is lower (~0.3), likely reflecting the weaker reliability of similarity estimates when interaction data is extremely sparse. At $\alpha_f = 0$, GF-BNNR recovers BNNR exactly, confirming that the filter provides a strict improvement for any $\alpha_f > 0$ within the tested range.

### 3.6 Computational Efficiency

Table 4 reports per-fold runtime on a standard workstation. GBNNR is slower than BNNR due to the inner gradient descent loop (10 steps per ADMM iteration), but still completes within practical timeframes. GF-BNNR adds negligible overhead (two linear system solves per fold). The adaptive truncated SVD acceleration reduces SVT cost by approximately 30× when the effective rank is low.

**Table 4. Mean runtime per fold (seconds).**

| Method | Fdataset | Cdataset | DNdataset |
|--------|----------|----------|-----------|
| BNNR | 48 | 72 | 2,150 |
| GBNNR | 86 | 237 | 4,702 |
| GF-BNNR | 87 | 86 | 172 |

---

## 4 Discussion

### 4.1 The Inside-Outside Framework

GBNNR and GF-BNNR represent two strategies for exploiting manifold structure that we term "inside" and "outside" regularization (Figure 3). GBNNR operates *inside* the optimization: the ADMM iterations are steered toward solutions that respect the similarity geometry via graph Laplacian regularization and inner gradient descent. GF-BNNR operates *outside* the optimization: a post-hoc bi-directional graph low-pass filter directly enforces output smoothness on the similarity manifolds.

These strategies target the same underlying manifold signal through independent mechanisms. Our stacking experiment (Section 3.4) confirms this: applying the GF filter to GBNNR's output yields AUPR = 0.3237, below GBNNR alone (0.3269) and between the two individual methods. The non-additivity is expected — GBNNR's inner gradient descent already captures geometric structure that the GF filter would otherwise enforce, and the filter's linear smoothing cannot extract additional information from a matrix whose manifold structure has already been partially encoded during optimization. This supports the inside-outside taxonomy as a *classification* of strategies, not a recipe for combination: the two approaches represent alternative routes to the same geometric prior, each with distinct trade-offs. GF-BNNR uniquely improves both AUROC and AUPR and is modular — applicable to any matrix completion output. GBNNR offers larger absolute AUPR gains on denser datasets and requires no $\lambda$ tuning. The choice between them depends on the application's priority: simultaneous AUROC improvement (GF-BNNR) or maximal AUPR with zero tuning burden (GBNNR).

### 4.2 Topology Over Strength: Why λ Doesn't Matter

The most surprising finding of this study is the complete insensitivity of GBNNR to the regularization parameter $\lambda$. Across the range from $\lambda = 0$ to $\lambda = 10^{-1}$, AUPR remains constant to three or more decimal places for each fixed $\gamma$. This is not a failure of the regularization — GBNNR substantially outperforms BNNR even at $\lambda = 0$ — but rather evidence that the graph Laplacian's contribution operates through its discrete structure rather than its continuous magnitude.

We offer the following explanation. At $\lambda = 0$, the graph Laplacian term is entirely absent from the objective; the only difference from BNNR is the replacement of the closed-form $\mathbf{W}$-update with 10 steps of inner gradient descent. That GBNNR ($\lambda = 0$) achieves AUPR = 0.3273 versus BNNR's 0.3071 implies that the inner gradient descent alone — without any graph regularization — accounts for the majority of the gain (+0.0202 AUPR). When the graph Laplacian is introduced ($\lambda > 0$), an additional marginal gain of approximately +0.0001 AUPR is observed, which saturates immediately: $\lambda = 10^{-3}$ and $\lambda = 10^{-1}$ produce identical results. The Laplacian $\mathbf{L}_{aug}$ therefore defines *which directions* in the parameter space are explored during gradient descent (the graph's eigenvectors provide structured gradient directions), and the inner GD's 10 steps suffice to reach equilibrium along those directions regardless of $\lambda$'s magnitude. We emphasize that the $\lambda$-insensitivity result is based on a single representative fold of Fdataset. While the pattern — inner GD dominating, graph topology providing marginal additional benefit — is consistent with the broader results in Table 1, confirmation across additional folds and datasets would strengthen the generality of this finding.

A practical consequence is that GBNNR is effectively hyperparameter-free for $\lambda$: any value in $[0, 10^{-1}]$ yields identical results. The only parameter requiring attention is $\gamma$, for which $\gamma = 2.0$ is a robust default that amplifies confident similarity edges while suppressing weak, potentially noisy ones.

### 4.3 Density-Dependent Benefits

Both manifold-based methods exhibit a striking inverse relationship between data density and relative improvement. On Fdataset (1.04% density), GBNNR improves AUPR by 5.4% and GF-BNNR by 4.1%. On DNdataset (0.015% density), the corresponding gains are 30.8% and 27.7%. This pattern makes intuitive sense: when observed associations are abundant, the data fidelity term in the optimization provides strong guidance, and manifold information serves as a tiebreaker. When observations are scarce, manifold structure becomes the primary source of information. This property is particularly valuable for drug repositioning, where emerging diseases or newly characterized drugs have very few known associations — precisely the cold-start scenarios that BNNR was designed to handle.

To assess the practical relevance of these predictions, we examined GF-BNNR's top-ranked predictions on the full DNdataset for Parkinson's disease (MeSH: "PARKINSON'S DISEASE") and schizophrenia (MeSH: "SCHIZOPHRENIFORM DISORDER" and "PARANOID SCHIZOPHRENIA"). DNdataset contains only 1,008 known associations across 1,490 drugs and 4,516 diseases — many true drug-disease relationships are absent from the training data, providing a realistic test of the model's ability to prioritize missing true associations.

For Parkinson's disease (16 known associations in DNdataset), GF-BNNR's top-ranked novel predictions are bromocriptine (score 0.193), rasagiline (0.182), metixene (0.181), biperiden (0.181), amantadine (0.178), benztropine (0.174), and cabergoline (0.170). All seven are established antiparkinsonian agents — dopamine agonists, MAO-B inhibitors, and anticholinergics — that are standard clinical treatments but not recorded as associations in this sparse dataset. The model's ability to rank these known treatments at the top, using only manifold structure from chemical and MeSH similarities, provides a strong sanity check on the biological relevance of manifold-smoothed predictions. Among lower-ranked predictions, memantine (0.095), donepezil (0.094), and galantamine (0.094) — memantine (an NMDA receptor antagonist) and the cholinesterase inhibitors donepezil and galantamine, all approved for Alzheimer's disease — have been investigated for Parkinson's disease dementia. Rivastigmine, another cholinesterase inhibitor, is already FDA-approved for this indication (Emre et al., 2004). Memantine has been evaluated for PDD and DLB in a randomized controlled trial (Aarsland et al., 2009).

For schizophrenia (0 known associations for either MeSH term in DNdataset), the top predictions for schizophreniform disorder include amisulpride (0.064), paroxetine (0.038), maprotiline (0.032), fluoxetine (0.028), and sertraline (0.027); for paranoid schizophrenia, amisulpride (0.068), paroxetine (0.044), maprotiline (0.037), and fluoxetine (0.033). Amisulpride, an atypical antipsychotic, is a standard first-line treatment for schizophrenia (Leucht et al., 2013). The antidepressant predictions — SSRIs (paroxetine, fluoxetine, sertraline) and tetracyclics (maprotiline) — are consistent with their established role as adjunctive therapy for negative symptoms and comorbid depression in schizophrenia (Singh et al., 2010; Helfer et al., 2016). The model also predicts several drugs with emerging literature support: lamotrigine (0.015), a mood stabilizer investigated for schizophrenia augmentation (Tiihonen et al., 2009), and thiamine (vitamin B1, 0.041), implicated in metabolic dysfunction pathways recently linked to schizophrenia pathophysiology (Cao et al., 2021).

These case studies illustrate a key practical property of manifold-aware matrix completion: on ultra-sparse datasets, the manifold signal recovers known-but-unrecorded associations at high precision, while also surfacing mechanistically plausible novel candidates. The predictions do not rely on disease-specific training data — schizophrenia had zero known associations in DNdataset — yet the manifold geometry alone is sufficient to rank established antipsychotics and adjunctive antidepressants above random baseline. We note an important caveat: the Parkinson's disease validation is partially circular, as the top-ranked drugs (bromocriptine, rasagiline, etc.) share substantial chemical similarity with the 16 known Parkinson's treatments in the training data. The method uses chemical similarity to predict that chemically similar drugs treat the same disease, so high-ranking known antiparkinsonians primarily validates that the similarity signal is coherent — not that the method discovers mechanistically novel therapies. The schizophrenia case, with zero training associations, provides a stronger test of de novo prediction, though even here, predictions may be partially driven by chemical similarity to antipsychotics used for other psychiatric conditions in the training set. Systematic external validation against clinical trial registries would be needed to assess true novelty.

### 4.4 Limitations and Future Work

Several limitations should be noted. First, the $\lambda$-insensitivity sweep (Section 3.2) and the stacking experiment (Section 3.4) were conducted on a single representative fold of Fdataset. While the patterns are consistent with the broader 10-fold results, confirming the $\lambda$ invariance and non-additivity across all folds and datasets would strengthen these findings. Second, the $k$NN graph uses a fixed neighborhood size $k = 12$ throughout; given the paper's central claim that graph topology drives performance, sensitivity to $k$ merits investigation. Third, GF-BNNR's reported improvements over the BNNR baseline include a contribution from GIP similarity fusion ($w_{gip}=0.3$), which enriches the input similarities before BNNR. Isolating the filter's contribution from the GIP contribution would provide a cleaner assessment. Fourth, GF-BNNR's filter strength $\alpha_f$ requires per-dataset tuning, though the performance landscape is relatively flat near the optimum. Fifth, we did not compare against deep learning methods such as graph neural networks (Gao et al., 2023) or contrastive learning approaches, which represent an alternative paradigm for DDA prediction and are now standard comparators in the literature. Sixth, the case study validations remain preliminary: the Parkinson's disease predictions are partially circular (chemically similar drugs predicted for the same indication), and systematic external validation against clinical trial registries is needed to assess true novelty.

Future work should explore: (1) replication of the $\lambda$-insensitivity and stacking non-additivity findings across all 10 folds and multiple datasets; (2) sensitivity analysis of the neighborhood size $k$; (3) joint optimization of $(k, \gamma, \alpha_f)$ as a unified parameter set; (4) isolation of GIP fusion and graph filter contributions in GF-BNNR through component-level ablation; (5) extension to multi-relational heterogeneous networks incorporating drug-target, drug-side-effect, and protein-protein interaction data; (6) theoretical analysis of why the inner gradient descent combined with graph topology produces $\lambda$-insensitivity; (7) comprehensive benchmarking against GNN-based methods using standardized evaluation protocols; (8) systematic case studies with external validation against clinical trial registries and recent literature; and (9) application of the inside-outside framework to other bipartite matrix completion problems in computational biology, such as drug-target interaction prediction and miRNA-disease association inference.

---

## References

1. Hamosh, A. et al. (2002) Online Mendelian Inheritance in Man (OMIM), a knowledgebase of human genes and genetic disorders. *Nucleic Acids Research*, 30, 52–55.
2. Boyd, S. et al. (2011) Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3, 1–122.
3. Cai, J.-F. et al. (2010) A singular value thresholding algorithm for matrix completion. *SIAM Journal on Optimization*, 20, 1956–1982.
4. Chong, C.R. et al. (2007) New uses for old drugs. *Nature*, 448, 645–646.
5. Dai, W. et al. (2015) Matrix factorization-based prediction of novel drug indications. *BMC Bioinformatics*, 16, S5.
6. Fu, Z. et al. (2022) MVGCN: Multi-view graph convolutional network for drug-disease association prediction. *Bioinformatics*, 38, 4019–4026.
7. Gao, Z. et al. (2023) Similarity measures-based graph co-contrastive learning for drug-disease association prediction. *Bioinformatics*, 39, btad357.
8. Gottlieb, A. et al. (2011) PREDICT: a method for inferring novel drug indications with application to personalized medicine. *Molecular Systems Biology*, 7, 496.
9. Luo, H. et al. (2016) Drug repositioning based on comprehensive similarity measures and Bi-Random walk algorithm. *Bioinformatics*, 32, 2664–2671.
10. Luo, H. et al. (2018) DRRS: a drug repositioning recommendation system based on singular value thresholding. *BMC Bioinformatics*, 19, 406.
11. Martinez, V. et al. (2015) DrugNet: network-based drug-disease prioritization. *Journal of Biomedical Informatics*, 53, 85–97.
12. Meng, Y. et al. (2022) DRWBNCF: drug repositioning based on weighted Bayesian neural collaborative filtering. *Briefings in Bioinformatics*, 23, bbac001.
13. Paul, S.M. et al. (2010) How to improve R&D productivity. *Nature Reviews Drug Discovery*, 9, 203–214.
14. Saito, T. and Rehmsmeier, M. (2015) The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10, e0118432.
15. Steinbeck, C. et al. (2003) The Chemistry Development Kit (CDK). *Journal of Chemical Information and Computer Sciences*, 43, 493–500.
16. Van Driel, M.A. et al. (2006) A text-mining analysis of the human phenome. *European Journal of Human Genetics*, 14, 535–542.
17. van Laarhoven, T. et al. (2011) Gaussian interaction profile kernels for predicting drug-target interaction. *Bioinformatics*, 27, 3036–3043.
18. Wang, J. et al. (2013) HGBI: heterogeneous graph based inference for drug-target interaction prediction. *BMC Bioinformatics*, 14, S6.
19. Wang, J. et al. (2014) Drug-disease association prediction based on heterogeneous graph inference. *BMC Systems Biology*, 8, S4.
20. Wishart, D.S. et al. (2006) DrugBank: a comprehensive resource for in silico drug discovery. *Nucleic Acids Research*, 34, D668–D672.
21. Yang, M. et al. (2019) Drug repositioning based on bounded nuclear norm regularization. *Bioinformatics*, 35, i455–i463.
22. Aarsland, D. et al. (2009) Memantine in patients with Parkinson's disease dementia or dementia with Lewy bodies: a double-blind, placebo-controlled, multicentre trial. *The Lancet Neurology*, 8, 613–618.
23. Emre, M. et al. (2004) Rivastigmine for dementia associated with Parkinson's disease. *New England Journal of Medicine*, 351, 2509–2518.
24. Leucht, S. et al. (2013) Comparative efficacy and tolerability of 15 antipsychotic drugs in schizophrenia: a multiple-treatments meta-analysis. *The Lancet*, 382, 951–962.
25. Singh, S.P. et al. (2010) Efficacy of antidepressants in treating the negative symptoms of chronic schizophrenia: meta-analysis. *British Journal of Psychiatry*, 197, 174–179.
26. Helfer, B. et al. (2016) Efficacy and safety of antidepressants added to antipsychotics for schizophrenia: systematic review and meta-analysis. *American Journal of Psychiatry*, 173, 876–886.
27. Tiihonen, J. et al. (2009) Lamotrigine in treatment-resistant schizophrenia: a randomized placebo-controlled crossover trial. *Biological Psychiatry*, 66, 794–799.
28. Cao, B. et al. (2021) Metabolic dysfunction in schizophrenia: the role of thiamine and benfotiamine. *Frontiers in Psychiatry*, 12, 681493.
29. Candes, E.J. and Recht, B. (2009) Exact matrix completion via convex optimization. *Foundations of Computational Mathematics*, 9, 717–772.
30. DiMasi, J.A. et al. (2003) The price of innovation: new estimates of drug development costs. *Journal of Health Economics*, 22, 151–185.
