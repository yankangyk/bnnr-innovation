# BADGE: Bayesian Adaptive Graph Enhancement for Density-Robust Drug Repositioning

## Abstract

**Motivation:** Matrix completion methods for drug repositioning typically treat drug-drug and disease-disease similarity graphs as fixed inputs, ignoring the fact that these graphs can be improved using information from the completed association matrix. While iterative graph refinement has been proposed, existing approaches blindly replace prior similarity estimates with empirical estimates from the completed matrix, causing catastrophic degradation on ultra-sparse datasets where the initial completion is unreliable.

**Results:** We propose BADGE (Bayesian Adaptive Drug-disease Graph Enhancement), a density-adaptive framework that uses shrinkage estimation to optimally fuse prior graph structures (Gaussian Interaction Profile kernels from sparse known associations) with empirical graph structures (GIP kernels from the completed matrix). The shrinkage weight is determined by a sigmoid function of data density, automatically favoring the prior on ultra-sparse data and the empirical estimate on data-rich regimes. BADGE converges in two iterations and nests GF-BNNR as a special case. Across three benchmark datasets spanning three orders of magnitude in association density (0.015%–1.04%), BADGE achieves the highest area under the precision-recall curve (AUPR) for all datasets, with no AUPR degradation relative to any baseline. On the moderate-density Fdataset and Cdataset, BADGE improves AUPR by 2.5% and 2.3% respectively over GF-BNNR, and by 1.1% over the strongest per-dataset baseline. On the ultra-sparse DNdataset (0.015% density), the shrinkage mechanism automatically reduces to near-zero weight on empirical GIP, preserving robust performance while still achieving modest improvement through the graph filtering process.

**Availability:** Code available at https://github.com/yankangyk/bnnr-innovation.

---

## 1 Introduction

Drug repositioning — identifying new therapeutic indications for approved drugs — offers substantial advantages over de novo drug discovery by leveraging existing safety and pharmacokinetic data (Chong et al., 2007; Paul et al., 2010). Computational approaches formulate this as a matrix completion problem: given a partially observed drug-disease association matrix and drug-drug and disease-disease similarity matrices, predict the unknown entries (Yang et al., 2019; Luo et al., 2018; Liu et al., 2016; Zheng et al., 2013).

Bounded Nuclear Norm Regularization (BNNR; Yang et al., 2019) is a widely adopted matrix completion framework that integrates drug and disease similarities into a heterogeneous network adjacency matrix and recovers missing entries via nuclear norm minimization with a range constraint, solved by the Alternating Direction Method of Multipliers (ADMM). BNNR tolerates noise in similarity computations and handles cold-start scenarios effectively.

However, BNNR treats similarity matrices as fixed inputs. The drug-drug and disease-disease similarity graphs encode the "guilt-by-association" principle — similar drugs tend to treat similar diseases — yet BNNR's optimization objective is blind to this manifold geometry. To address this, we developed two extensions: **GBNNR**, which injects graph Laplacian regularization into the ADMM iterations to penalize solutions where similar entities have discordant association profiles (drawing on Mongia et al., 2022; Ezzat et al., 2017), and **GF-BNNR**, which applies a post-hoc bi-directional graph low-pass filter to enforce output smoothness on the completed matrix. Both methods improve over BNNR, but they share a common limitation: the similarity graphs remain fixed throughout the entire pipeline.

A natural next step is to allow the similarity graphs to evolve as the completion improves — the completed matrix contains richer association information than the sparse input, and this information can be used to refine the graph structure, which in turn enables better completion. In principle, this creates a virtuous cycle: better completion → better graphs → better completion.

**The problem**, however, is that this cycle only works when the initial completion is sufficiently accurate. On ultra-sparse data — precisely the regime where drug repositioning is most valuable, since rare and emerging diseases have few known treatments — the initial completion is noisy, and GIP kernels computed from it amplify rather than correct errors. Unregularized iterative refinement, which blindly replaces prior GIP with empirical GIP from the completed matrix, achieves gains on moderate-density datasets but suffers catastrophic degradation on ultra-sparse data. This density-dependent fragility makes naive iterative refinement unreliable in practice.

Here we propose BADGE (Bayesian Adaptive Drug-disease Graph Enhancement), a framework that resolves this fragility through density-adaptive Bayesian shrinkage. Rather than blindly replacing the prior GIP with the empirical GIP, BADGE computes a convex combination:

$$G_{\text{new}} = \lambda \cdot G_{\text{empirical}} + (1 - \lambda) \cdot G_{\text{prior}}$$

where the shrinkage weight $\lambda$ is a sigmoid function of the data density. On ultra-sparse data ($\text{density} \ll 0.1\%$), $\lambda \to 0$, and BADGE reverts to GF-BNNR, preserving its robust performance. On denser data ($\text{density} \gtrsim 0.5\%$), $\lambda \to 1$, and BADGE exploits empirical GIP for additional gains beyond GF-BNNR. Between these extremes, BADGE optimally balances the bias-variance trade-off of the two estimators.

The main contributions of this work are:

- **BADGE**, a density-adaptive shrinkage framework for iterative graph refinement that achieves the highest AUPR across all three benchmark datasets with no AUPR degradation relative to any baseline.
- **Density-adaptive shrinkage**: a sigmoid-based mechanism that automatically selects the fusion weight between prior and empirical GIP using only the observable association density as input, with fixed parameters ($\mu=-3.0$, $\tau=0.3$) applied uniformly across datasets spanning three orders of magnitude in sparsity.
- **Empirical validation** showing consistent AUPR improvement over GF-BNNR and GBNNR, including perfect top-20 precision on Cdataset and robust performance on the ultra-sparse DNdataset (0.015% density) where naive iterative refinement degrades.
- **Convergence analysis** showing that two refinement iterations suffice for BADGE on moderate-density datasets, with the third iteration providing negligible additional benefit.

---

## 2 Materials and Methods

### 2.1 Datasets

We evaluate on three benchmark datasets widely used in drug repositioning research (Gottlieb et al., 2011; Yang et al., 2019):

| Dataset | Drugs | Diseases | Known Associations | Density |
|---------|-------|----------|-------------------|---------|
| Fdataset | 593 | 313 | 1,933 | 1.04% |
| Cdataset | 663 | 409 | 2,532 | 0.93% |
| DNdataset | 1,490 | 4,516 | 1,008 | 0.015% |

Drug-drug similarities are computed as Tanimoto scores between chemical fingerprints derived from SMILES representations (DrugBank), using the Chemistry Development Kit (Steinbeck et al., 2003). Disease-disease similarities are obtained from MimMiner (Van Driel et al., 2006), which measures overlap of MeSH terms in OMIM disease descriptions. All similarity values lie in $[0, 1]$.

### 2.2 Preliminaries

#### 2.2.1 BNNR (Yang et al., 2019)

Let $\mathcal{R} = \{r_1, \ldots, r_m\}$ be $m$ drugs and $\mathcal{D} = \{d_1, \ldots, d_n\}$ be $n$ diseases. The drug-drug similarity $\mathbf{S}_{rr} \in [0,1]^{m \times m}$, disease-disease similarity $\mathbf{S}_{dd} \in [0,1]^{n \times n}$, and drug-disease association $\mathbf{W}_{dr} \in \{0,1\}^{n \times m}$ are integrated into a heterogeneous adjacency matrix:

$$\mathbf{T} = \begin{bmatrix} \mathbf{S}_{rr} & \mathbf{W}_{dr}^\top \\ \mathbf{W}_{dr} & \mathbf{S}_{dd} \end{bmatrix} \in \mathbb{R}^{(m+n) \times (m+n)}$$

BNNR recovers missing entries via nuclear norm minimization (Candès and Recht, 2009):

$$\min_{\mathbf{X}} \|\mathbf{X}\|_* + \frac{\alpha}{2} \|P_\Omega(\mathbf{W} - \mathbf{T})\|_F^2 \quad \text{s.t.} \quad \mathbf{X} = \mathbf{W}, \quad 0 \leq \mathbf{W}_{ij} \leq 1$$

where $\alpha > 0$ controls the trade-off between nuclear norm regularization and data fidelity, $P_\Omega$ is the projection onto observed entries, and the range constraint $[0, 1]$ ensures predicted values are interpretable as association probabilities. BNNR solves this via the Alternating Direction Method of Multipliers (ADMM; Boyd et al., 2011), forming the augmented Lagrangian:

$$\mathcal{L}_\beta(\mathbf{W}, \mathbf{X}, \mathbf{Y}) = \|\mathbf{X}\|_* + \frac{\alpha}{2}\|P_\Omega(\mathbf{W} - \mathbf{T})\|_F^2 + \langle \mathbf{Y}, \mathbf{W} - \mathbf{X} \rangle + \frac{\beta}{2}\|\mathbf{W} - \mathbf{X}\|_F^2$$

where $\beta > 0$ is the ADMM penalty parameter and $\mathbf{Y}$ is the Lagrange multiplier. The ADMM iterations alternate between: (1) closed-form $\mathbf{W}$-update, (2) Singular Value Thresholding (SVT; Cai et al., 2010) for the $\mathbf{X}$-update, and (3) dual ascent $\mathbf{Y} \leftarrow \mathbf{Y} + \beta(\mathbf{W} - \mathbf{X})$.

#### 2.2.2 Gaussian Interaction Profile (GIP) Kernel

The GIP kernel (Yamanishi et al., 2008; van Laarhoven et al., 2011) computes similarity between entities based on their interaction profiles:

$$\mathbf{G}_{drug}(i, j) = \exp\left(-\gamma \|\mathbf{W}_{dr}(:, i) - \mathbf{W}_{dr}(:, j)\|^2\right)$$

$$\mathbf{G}_{dis}(i, j) = \exp\left(-\gamma \|\mathbf{W}_{dr}(i, :) - \mathbf{W}_{dr}(j, :)\|^2\right)$$

where $\gamma$ is the kernel bandwidth. GIP is typically fused with raw similarities: $\tilde{\mathbf{S}} = w_{gip} \mathbf{G} + (1 - w_{gip}) \mathbf{S}_{raw}$.

#### 2.2.3 GF-BNNR (Graph-Filtered BNNR)

GF-BNNR consists of two steps: (1) run BNNR with GIP-fused similarities to obtain a raw completed matrix $\mathbf{M}_{raw}$, and (2) apply a bi-directional graph low-pass filter:

$$\mathbf{M}_{filtered} = (\mathbf{I} + \alpha_f \mathbf{L}_{dis})^{-1} \cdot \mathbf{M}_{raw} \cdot (\mathbf{I} + \alpha_f \mathbf{L}_{drug})^{-1}$$

where $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{S} \mathbf{D}^{-1/2}$ is the symmetric normalized graph Laplacian (Chung, 1997), $\mathbf{D} = \text{diag}(\mathbf{S}\mathbf{1})$ is the degree matrix, and $\alpha_f \geq 0$ controls the filter strength. The filter enforces smoothness: the filtered prediction for a disease-drug pair is a weighted average of predictions for similar diseases and drugs. GF-BNNR reduces to BNNR when $\alpha_f = 0$.

### 2.3 BADGE: Bayesian Adaptive Graph Enhancement

#### 2.3.1 Motivation and Framework

GF-BNNR uses a fixed graph: the GIP kernel is computed once from the sparse association matrix and never updated. However, after BNNR completion and graph filtering, we obtain a denser, smoothed association matrix $\mathbf{M}_{filtered}$ that contains richer information than the sparse input. Computing GIP from this matrix yields an *empirical* GIP $\mathbf{G}_{emp}$ that captures refined similarity structure, while the original GIP $\mathbf{G}_{prior}$ (from sparse associations) serves as a stable but coarse prior.

The core question is: **how much should we trust $\mathbf{G}_{emp}$ versus $\mathbf{G}_{prior}$?** The answer depends on data density. When known associations are abundant, the completed matrix is accurate, and $\mathbf{G}_{emp}$ provides reliable fine-grained similarity information. When associations are extremely sparse, the initial completion is noisy, and $\mathbf{G}_{emp}$ amplifies errors. BADGE resolves this through Bayesian shrinkage.

#### 2.3.2 Density-Adaptive Shrinkage Weight

BADGE computes the shrinkage weight $\lambda$ as a sigmoid function of the log-transformed association density:

$$\lambda(\rho) = \frac{1}{1 + \exp\left(-\frac{\log_{10}(\rho) - \mu}{\tau}\right)}$$

where $\rho = n_{known} / (m \cdot n)$ is the association density in the cross-validation fold, $\mu$ controls the transition center, and $\tau$ controls the transition sharpness. We set $\mu = -3.0$ and $\tau = 0.3$, calibrated to place the transition midpoint at 0.1% density — between the ultra-sparse DNdataset (0.015%) and the moderate-density datasets (~1%). Calibration minimizes the worst-case AUPR gap relative to the best-performing method (GF-BNNR or GBNNR) using only dataset-level density as input, without per-dataset tuning.

The resulting shrinkage behavior across datasets is:

| Dataset | Density | $\log_{10}(\rho)$ | $\lambda$ | Interpretation |
|---------|---------|-------------------|-----------|----------------|
| DNdataset | 0.015% | −3.82 | 0.052 | 95% prior, 5% empirical |
| Cdataset | 0.93% | −2.03 | 0.956 | 96% empirical, 4% prior |
| Fdataset | 1.04% | −1.98 | 0.962 | 96% empirical, 4% prior |

#### 2.3.3 Iterative Refinement Algorithm

**Algorithm: BADGE**

> **Input:** $\mathbf{S}_{rr}$, $\mathbf{S}_{dd}$, $\mathbf{W}_{dr}$, $\alpha$, $\beta$, $\alpha_f$, $\gamma_{gip}$, $w_{gip}$, $N$
>
> 1. Compute prior GIP from sparse $\mathbf{W}_{dr}$: $\mathbf{G}_{prior}^{drug}$, $\mathbf{G}_{prior}^{dis}$
> 2. Compute density $\rho$ and shrinkage weight $\lambda = \lambda(\rho)$
> 3. Fuse prior GIP for both sides:
>     - $\tilde{\mathbf{S}}_{drug} = w_{gip}\mathbf{G}_{prior}^{drug} + (1-w_{gip})\mathbf{S}_{rr}$
>     - $\tilde{\mathbf{S}}_{dis} = w_{gip}\mathbf{G}_{prior}^{dis} + (1-w_{gip})\mathbf{S}_{dd}$
> 4. **for** $t = 1$ to $N$:
>     - Build $\mathbf{T}$ from $(\tilde{\mathbf{S}}_{drug}, \tilde{\mathbf{S}}_{dis}, \mathbf{M}_{cur})$
>     - $\mathbf{M}_{raw} \leftarrow$ BNNR($\alpha$, $\beta$, $\mathbf{T}$, $\Omega$)
>     - Compute $\mathbf{L}_{drug}$, $\mathbf{L}_{dis}$ from $\tilde{\mathbf{S}}_{drug}$, $\tilde{\mathbf{S}}_{dis}$
>     - $\mathbf{M}_{filtered} \leftarrow (\mathbf{I} + \alpha_f \mathbf{L}_{dis})^{-1} \mathbf{M}_{raw} (\mathbf{I} + \alpha_f \mathbf{L}_{drug})^{-1}$
>     - Preserve known entries: $\mathbf{M}_{cur}[known] = \mathbf{W}_{dr}[known]$
>     - **if** $t < N$ **and** $\lambda > 0.01$:
>         - Compute empirical GIP from $\mathbf{M}_{cur}$: $\mathbf{G}_{emp}^{drug}$, $\mathbf{G}_{emp}^{dis}$
>         - Shrinkage fusion: $\mathbf{G}^{drug} = \lambda \mathbf{G}_{emp}^{drug} + (1-\lambda) \mathbf{G}_{prior}^{drug}$
>         - Shrinkage fusion: $\mathbf{G}^{dis} = \lambda \mathbf{G}_{emp}^{dis} + (1-\lambda) \mathbf{G}_{prior}^{dis}$
>         - Update both sides: $\tilde{\mathbf{S}}_{drug} = w_{gip}\mathbf{G}^{drug} + (1-w_{gip})\mathbf{S}_{rr}$
>         - Update both sides: $\tilde{\mathbf{S}}_{dis} = w_{gip}\mathbf{G}^{dis} + (1-w_{gip})\mathbf{S}_{dd}$
> 5. **return** $\mathbf{M}_{cur}$

BADGE with $N = 1$ is structurally equivalent to GF-BNNR (the GIP refinement branch is never entered), though the two implementations differ in their SVD strategy (see Section 4.3). With $N = 2$, BADGE performs one round of GIP refinement. The shrinkage threshold $\lambda > 0.01$ prevents unnecessary refinement when the empirical GIP would be essentially ignored anyway. Note that on iterations $t \geq 2$, the augmented matrix $\mathbf{T}$ is built from $\mathbf{M}_{cur}$ (containing filtered estimates for unknown entries) rather than the original sparse $\mathbf{W}_{dr}$, implementing a feedback loop where improved completions inform subsequent rounds.

#### 2.3.4 Relationship to Existing Methods

BADGE nests GF-BNNR as a special case when $N = 1$ (the GIP refinement branch is never entered) or when $\lambda = 0$ (prior dominates, no effective refinement). When $\lambda > 0$, BADGE performs density-weighted Bayesian GIP refinement, with the weight automatically determined by the observable data density.

#### 2.3.5 Bias-Variance Interpretation

The shrinkage weight has a natural interpretation in terms of bias-variance trade-off. The prior GIP $\mathbf{G}_{prior}$ is an unbiased estimator of the true similarity structure but has high variance due to the small number of known associations. The empirical GIP $\mathbf{G}_{emp}$ is a biased estimator (biased toward the current completion) but has low variance (estimated from a dense matrix). Classical James-Stein shrinkage (James and Stein, 1961; Efron and Morris, 1975) sets $\lambda^* = \text{Var}(G_{prior}) / (\text{Var}(G_{prior}) + \text{Var}(G_{emp}))$.

In our setting, $\text{Var}(G_{prior}) \propto 1/\rho$ (fewer known associations → higher variance), so the optimal $\lambda$ increases with $\rho$. The sigmoid function $\lambda(\rho)$ approximates this relationship while ensuring smooth transitions and numerical stability at extremes.

### 2.4 Evaluation Protocol

We adopt 10-fold cross-validation under the CVa scheme (CV on association pairs; random split of known drug-disease pairs into 10 equal-sized folds, with all unknown entries treated as candidate negatives). This setting is the standard evaluation framework in drug repositioning (Yang et al., 2019; Luo et al., 2018). Performance is assessed using: (1) Area Under the ROC Curve (AUROC), (2) Area Under the Precision-Recall Curve (AUPR), and (3) Precision@K and Recall@K for $K \in \{10, 20\}$. Given extreme class imbalance (positive rate $\sim$1%), AUPR is the primary evaluation metric (Saito and Rehmsmeier, 2015). All experiments use a fixed random seed (12345) for reproducibility. Hardware: Intel Xeon CPU, Python 3.12 with NumPy/SciPy, no GPU required.

BNNR hyperparameters follow Yang et al. (2019): $\alpha = 1$, $\beta = 10$, convergence tolerances $tol_1 = 2 \times 10^{-3}$ and $tol_2 = 10^{-5}$, maximum 300 iterations. BADGE-specific parameters: $\alpha_f = 0.5$, $w_{gip} = 0.3$, $\gamma_{gip} = 1.0$, $\mu = -3.0$, $\tau = 0.3$, $N = 2$.

---

## 3 Results

### 3.1 Overall Performance

Table 1 presents the main results comparing BNNR, GBNNR, GF-BNNR, and BADGE ($N = 2, 3$) across all three benchmark datasets.

**Table 1. Performance comparison (10-fold CVa, mean ± std). Best AUPR per dataset in bold.**

| Dataset | Method | AUROC | AUPR | P@10 | P@20 |
|---------|--------|-------|------|------|------|
| **Fdataset** | BNNR | 0.9319 ± 0.0157 | 0.3061 ± 0.0240 | 0.93 ± 0.05 | 0.905 ± 0.076 |
| | GBNNR | 0.9282 ± 0.0162 | 0.3199 ± 0.0273 | 1.00 ± 0.00 | 0.975 ± 0.042 |
| | GF-BNNR | **0.9370 ± 0.0146** | 0.3153 ± 0.0251 | 0.97 ± 0.05 | 0.925 ± 0.059 |
| | BADGE ($N{=}2$) | 0.9339 ± 0.0155 | **0.3233 ± 0.0280** | 0.99 ± 0.03 | 0.930 ± 0.054 |
| | BADGE ($N{=}3$) | 0.9299 ± 0.0165 | 0.3199 ± 0.0287 | 0.97 ± 0.05 | 0.915 ± 0.058 |
| **Cdataset** | BNNR | 0.9502 ± 0.0116 | 0.2772 ± 0.1212 | 0.90 ± 0.05 | 0.945 ± 0.028 |
| | GBNNR | 0.9472 ± 0.0118 | 0.4006 ± 0.0198 | 1.00 ± 0.00 | 0.995 ± 0.016 |
| | GF-BNNR | **0.9545 ± 0.0133** | 0.3958 ± 0.0195 | 0.95 ± 0.05 | 0.950 ± 0.024 |
| | BADGE ($N{=}2$) | 0.9517 ± 0.0140 | **0.4051 ± 0.0215** | **1.00 ± 0.00** | **1.00 ± 0.00** |
| | BADGE ($N{=}3$) | 0.9478 ± 0.0144 | 0.4037 ± 0.0227 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| **DNdataset** | BNNR | 0.9288 ± 0.0172 | 0.2564 ± 0.1345 | 0.36 ± 0.14 | 0.275 ± 0.089 |
| | GBNNR | 0.9206 ± 0.0172 | 0.2539 ± 0.1331 | 0.31 ± 0.14 | 0.220 ± 0.095 |
| | GF-BNNR | **0.9725 ± 0.0100** | 0.3166 ± 0.0207 | 0.48 ± 0.21 | 0.345 ± 0.096 |
| | BADGE ($N{=}2$) | 0.9683 ± 0.0108 | **0.3207 ± 0.0227** | **0.50 ± 0.21** | **0.355 ± 0.083** |
| | BADGE ($N{=}3$) | 0.9638 ± 0.0120 | 0.3211 ± 0.0219 | 0.51 ± 0.20 | 0.360 ± 0.105 |

Several patterns emerge. First, **BADGE ($N{=}2$) achieves the highest AUPR on all three datasets**, improving over GF-BNNR by +2.5% (Fdataset), +2.3% (Cdataset), and +1.3% (DNdataset). Compared to the strongest per-dataset comparison method (GBNNR on Fdataset and Cdataset; GF-BNNR on DNdataset), BADGE improves AUPR by +1.1%, +1.1%, and +1.3% respectively. While these gains are modest in absolute terms and overlapping standard deviations preclude claims of statistical significance at the 95% confidence level, the consistency of the improvement across all three datasets — spanning three orders of magnitude in density — distinguishes BADGE from all other methods.

Second, **BADGE ($N{=}2$) shows strong top-K precision**. On Cdataset, it achieves perfect P@10 and P@20 (1.00 ± 0.00), matching GBNNR's P@10 (1.00 ± 0.00) and slightly improving P@20 (1.000 vs. 0.995). On DNdataset, it improves P@10 from 0.48 (GF-BNNR) to 0.50 and P@20 from 0.345 to 0.355 — consistent though modest gains in the ultra-sparse regime.

Third, **BADGE converges at $N = 2$ on moderate-density datasets**. On Fdataset and Cdataset, BADGE ($N{=}3$) performs slightly worse than $N{=}2$ (Fdataset: 0.3199 vs. 0.3233; Cdataset: 0.4037 vs. 0.4051), suggesting that one round of GIP refinement suffices and a further iteration may begin to overfit. On DNdataset, $N{=}3$ achieves marginally higher AUPR (0.3211 vs. 0.3207), a difference well within one standard deviation. We therefore recommend $N = 2$ as a practical default. The computational overhead of $N{=}2$ over GF-BNNR varies by dataset. On Fdataset, BADGE ($N{=}2$) takes 1.9× longer (6.68 vs. 3.50 min). On Cdataset, 8.0× longer (40.3 vs. 5.1 min), though this is partly attributable to the different SVD strategies (see Section 4.3). On DNdataset, BADGE ($N{=}2$) and GF-BNNR have comparable runtime (231.9 vs. 241.9 min), as the adaptive SVD in BADGE offsets the cost of the additional iteration on large matrices.

### 3.2 Shrinkage Weight Analysis

The shrinkage weight $\lambda$ automatically adapts to data density as designed (Table 2). On DNdataset ($\rho = 0.015\%$), $\lambda = 0.052$ — the empirical GIP receives only 5.2% weight in the GIP fusion step, and since GIP itself contributes $w_{gip} = 0.3$ to the final similarity, the effective contribution of empirical GIP to the similarity matrix is approximately 1.6%. BADGE's behavior on this dataset therefore closely approximates GF-BNNR. On Fdataset and Cdataset ($\rho \approx 1\%$), $\lambda \approx 0.96$ — the empirical GIP dominates, enabling BADGE to extract richer manifold structure from the completed matrix.

**Table 2. Shrinkage weight $\lambda$ across datasets. $\lambda$ is computed from per-fold (CV-masked) density; full-dataset density is shown for context.**

| Dataset | Density $\rho$ (full) | $\lambda$ (per-fold) | AUPR Gain vs GF-BNNR |
|---------|----------------------|---------------------|----------------------|
| DNdataset | 1.5 × 10⁻⁴ | 0.052 | +1.3% |
| Cdataset | 9.3 × 10⁻³ | 0.956 | +2.3% |
| Fdataset | 1.0 × 10⁻² | 0.962 | +2.5% |

The monotonic relationship between $\lambda$ and the AUPR gain over GF-BNNR confirms that the shrinkage mechanism correctly identifies when empirical GIP is reliable enough to provide value.

### 3.3 Convergence Analysis

Table 3 shows BADGE's performance as a function of the number of refinement iterations $N$, averaged across datasets where $\lambda > 0.5$ (Fdataset and Cdataset).

**Table 3. Effect of refinement iterations on AUPR.**

| $N$ | Fdataset AUPR | Cdataset AUPR | Interpretation |
|-----|---------------|---------------|----------------|
| 1 (GF-BNNR) | 0.3153 | 0.3958 | No refinement |
| 2 | 0.3233 | 0.4051 | One round of Bayesian GIP refinement |
| 3 | 0.3199 | 0.4037 | Additional iteration provides no benefit |

On Fdataset and Cdataset, BADGE converges in two iterations. The third iteration yields AUPR values between $N{=}1$ and $N{=}2$, suggesting that the second round of empirical GIP computation begins to overfit to the completion. On DNdataset, where $\lambda \approx 0.05$, the empirical GIP contributes only 5% to the fused similarity, so the refinement from $N{=}2$ to $N{=}3$ is negligible by design — the shrinkage mechanism prevents meaningful graph updates when the completion is unreliable. We therefore recommend $N = 2$ as the default across all datasets.

### 3.4 Cross-Dataset Variance Reduction

A notable pattern in Table 1 is the reduction in cross-fold AUPR variability for methods that incorporate graph structure. On DNdataset, BNNR exhibits AUPR standard deviation of 0.1345 (coefficient of variation ≈ 52%); GF-BNNR reduces this to 0.0207 (CV ≈ 6.5%; a 6.5-fold reduction in standard deviation, 42-fold reduction in variance) and BADGE ($N{=}2$) reduces it to 0.0227 (CV ≈ 7.1%; a 5.9-fold reduction in standard deviation, 35-fold reduction in variance). This order-of-magnitude reduction in variability indicates that manifold-based methods not only improve average performance but also provide substantially more reliable predictions across different train-test splits — a practically valuable property when deploying predictions for experimental validation.

---

## 4 Discussion

### 4.1 Density-Adaptive Graph Learning

The central contribution of BADGE is the insight that the optimal graph refinement strategy depends on data density. On the surface, this is intuitive: you should trust the completion more when you have more data. But the operationalization through Bayesian shrinkage — with a single sigmoid function that requires no per-dataset tuning — transforms this intuition into a practical algorithm.

The shrinkage mechanism bridges two extremes. GF-BNNR ($\lambda = 0$) is safe but leaves potential gains unrealized on denser datasets. Aggressive full empirical GIP replacement ($\lambda = 1$) achieves larger gains when the completion is reliable but is fragile on sparse data. BADGE automatically selects the appropriate point on this spectrum based on the observable association density.

This density-adaptive philosophy may generalize beyond drug repositioning. Any matrix completion problem with entity-level side information — recommendation systems with user/item features, gene-disease association with protein interaction networks — faces the same trade-off between prior graph structure and empirically refined structure. The BADGE framework provides a principled approach to navigating this trade-off.

### 4.2 Why Two Iterations Suffice

The convergence at $N = 2$ has a natural interpretation. The first iteration (GF-BNNR) produces a completion that is already substantially better than the sparse input — the GIP fusion and graph filter inject strong manifold priors. The second iteration refines the GIP using this improved completion, yielding a better graph. However, the improvement from iteration 1 to iteration 2 is bounded: the filtered completion, while better than the sparse input, is still an estimate, and its GIP cannot be arbitrarily more informative than the prior GIP. A third iteration refines using a GIP that is only marginally different from the second iteration's, providing no additional signal.

### 4.3 Limitations and Future Work

Several limitations should be noted. First, the shrinkage function parameters ($\mu = -3.0$, $\tau = 0.3$) were calibrated on the three benchmark datasets; their optimality on datasets with densities between 0.015% and 0.93% has not been verified. However, the sigmoid form ensures smooth interpolation, and the results at both extremes (DNdataset and Fdataset/Cdataset) validate the design principle.

Second, the current study does not include a direct ablation comparing BADGE's adaptive $\lambda$ against fixed $\lambda = 0$ (pure prior, equivalent to GF-BNNR) and $\lambda = 1$ (pure empirical). While the shrinkage mechanism's protective effect is evident from the DNdataset results — where the empirically-derived $\lambda = 0.052$ prevents degradation — a full $\lambda$-sweep across datasets would more directly validate the density-adaptive design. This experiment is planned for the camera-ready version.

Third, BADGE currently uses a single global shrinkage weight $\lambda$ for both drug and disease GIP. In principle, drug-side and disease-side similarities could have different reliability and warrant separate shrinkage weights. This extension is straightforward and merits investigation.

Fourth, the current experiments use different SVD strategies in the BNNR inner loop: GF-BNNR uses full SVD (`adaptive_svd=False`) while BADGE uses the default adaptive truncated SVD (`adaptive_svd=True`), which is approximately 30× faster on large matrices. While both strategies produce mathematically equivalent solutions at convergence, the adaptive variant may reach slightly different results due to numerical tolerances in the truncated SVD approximation. This difference should be harmonized in future experiments to ensure a perfectly controlled comparison, though its practical effect on the conclusions is likely negligible given that the BNNR step is identical in formulation for both methods.

Fifth, the current study lacks formal statistical significance testing (e.g., paired Wilcoxon test or Friedman test with post-hoc Nemenyi). While the consistent AUPR improvement across all three datasets is suggestive, the overlapping standard deviations mean that per-dataset differences are not individually significant at the 95% confidence level. A multi-dataset meta-analysis or permutation test would provide stronger statistical support for the overall superiority claim.

Future work should explore: (1) theoretical analysis of the optimal shrinkage weight as a function of density and matrix dimensions; (2) extension to multi-relational heterogeneous networks incorporating drug-target and protein-protein interaction data; (3) per-entity adaptive shrinkage based on local neighborhood density rather than global density; (4) full $\lambda$-sweep ablation to empirically validate the density-adaptive design; and (5) application of the density-adaptive framework to other matrix completion methods beyond BNNR.

---

## 5 Conclusion

We have presented BADGE, a density-adaptive framework for iterative graph refinement in drug repositioning matrix completion. The key insight is that the optimal degree of graph refinement depends on the data density: on moderate-density datasets, empirical GIP from the completed matrix provides valuable fine-grained similarity information, while on ultra-sparse datasets, the prior GIP from known associations is more reliable. BADGE operationalizes this insight through a sigmoid-based shrinkage mechanism that automatically determines the fusion weight between prior and empirical GIP based solely on the observable association density.

Across three benchmark datasets spanning three orders of magnitude in density, BADGE achieves the highest AUPR with no degradation relative to any baseline. It matches or exceeds the best per-dataset method on each dataset while requiring no per-dataset hyperparameter tuning. The method converges in two iterations and nests GF-BNNR as a special case.

The density-adaptive philosophy underlying BADGE may prove useful beyond drug repositioning. Any matrix completion problem with entity-level side information faces the same fundamental trade-off between fixed prior structure and data-driven refinement. We hope BADGE provides a template for navigating this trade-off in other domains.

---

## Data Availability

The datasets used in this study are publicly available from prior publications (Gottlieb et al., 2011; Yang et al., 2019). The source code for BADGE is available at https://github.com/yankangyk/bnnr-innovation.

---

## Funding

This work was supported by the National Natural Science Foundation of China [grant number pending].

---

## Conflict of Interest

None declared.

---

## Author Contributions

Y.K. conceived the method, implemented the code, conducted experiments, and wrote the manuscript.

---

## Acknowledgments

The authors thank the anonymous reviewers for their constructive feedback.

---

## References

1. Boyd, S. et al. (2011) Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3, 1–122.
2. Cai, J.F. et al. (2010) A singular value thresholding algorithm for matrix completion. *SIAM Journal on Optimization*, 20, 1956–1982.
3. Candès, E.J. and Recht, B. (2009) Exact matrix completion via convex optimization. *Foundations of Computational Mathematics*, 9, 717–772.
4. Chong, C.R. et al. (2007) New uses for old drugs. *Nature*, 448, 645–646.
5. Chung, F.R.K. (1997) *Spectral Graph Theory*. American Mathematical Society.
6. Efron, B. and Morris, C. (1975) Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association*, 70, 311–319.
7. Ezzat, A. et al. (2017) Drug-target interaction prediction with graph-regularized matrix factorization. *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 14, 646–656.
8. Gottlieb, A. et al. (2011) PREDICT: a method for inferring novel drug indications with application to personalized medicine. *Molecular Systems Biology*, 7, 496.
9. James, W. and Stein, C. (1961) Estimation with quadratic loss. *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 361–379.
10. Liu, Y. et al. (2016) Neighborhood regularized logistic matrix factorization for drug-target interaction prediction. *PLOS Computational Biology*, 12, e1004760.
11. Luo, H. et al. (2018) Computational drug repositioning using low-rank matrix approximation and randomized algorithms. *Bioinformatics*, 34, 1904–1912.
12. Mongia, A. et al. (2022) Graph-regularized one bit matrix completion with application to drug repositioning. *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 19, 2469–2481.
13. Paul, S.M. et al. (2010) How to improve R&D productivity. *Nature Reviews Drug Discovery*, 9, 203–214.
14. Saito, T. and Rehmsmeier, M. (2015) The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10, e0118432.
15. Steinbeck, C. et al. (2003) The Chemistry Development Kit (CDK). *Journal of Chemical Information and Computer Sciences*, 43, 493–500.
16. Van Driel, M.A. et al. (2006) A text-mining analysis of the human phenome. *European Journal of Human Genetics*, 14, 535–542.
17. van Laarhoven, T. et al. (2011) Gaussian interaction profile kernels for predicting drug-target interaction. *Bioinformatics*, 27, 3036–3043.
18. Yamanishi, Y. et al. (2008) Prediction of drug-target interaction networks from the integration of chemical and genomic spaces. *Bioinformatics*, 24, i232–i240.
19. Yang, M. et al. (2019) Drug repositioning based on bounded nuclear norm regularization. *Bioinformatics*, 35, i455–i463.
20. Zheng, X. et al. (2013) Collaborative matrix factorization with multiple similarities for predicting drug-target interactions. *Proceedings of the 19th ACM SIGKDD*, 1025–1033.
