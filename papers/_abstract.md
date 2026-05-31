# Bilingual Abstract — BNNR Innovation Paper

**Target Journal**: Bioinformatics
**Title**: *An Inside-Outside Framework for Manifold-Aware Matrix Completion: Graph Regularization and Post-Hoc Filtering for Drug-Disease Association Prediction*

---

## Abstract (English)

**Motivation:** Bounded Nuclear Norm Regularization (BNNR) is a widely used matrix completion method for computational drug repositioning that tolerates noise in similarity measurements. However, BNNR treats the drug-disease association matrix as an isolated algebraic object, ignoring the manifold geometry encoded in drug-drug and disease-disease similarity networks — the principle that similar drugs should exhibit similar association profiles. This geometric blindness degrades precision, leaving a substantial gap between AUROC and AUPR.

**Results:** We propose an inside-outside framework for manifold-aware matrix completion with two complementary strategies. Graph-Regularized BNNR (GBNNR) operates *inside* optimization: it injects $k$-nearest-neighbor graph Laplacian regularization into BNNR's ADMM iterations, replacing the closed-form $\mathbf{W}$-update with inner gradient descent. This improves AUPR by 5.4–30.8% across three benchmark datasets while preserving AUROC. Graph-Filtered BNNR (GF-BNNR) operates *outside* optimization: a post-hoc bi-directional graph low-pass filter, $\mathbf{M}_{\text{filtered}} = (\mathbf{I} + \alpha_f \mathbf{L}_d)^{-1} \mathbf{M}_{\text{raw}} (\mathbf{I} + \alpha_f \mathbf{L}_r)^{-1}$, enforces output smoothness along the similarity manifolds. GF-BNNR improves both AUROC and AUPR simultaneously — a pattern not observed with the other methods — with gains reaching +5.3% AUROC and +27.7% AUPR on ultra-sparse data (DNdataset, 0.015% density); a portion of this improvement is attributable to GIP similarity fusion. A systematic hyperparameter sweep on a representative fold reveals that GBNNR's AUPR benefit is dominated by inner gradient descent, with graph topology providing a marginal additional gain: the regularization parameter $\lambda$ has no measurable effect from $\lambda = 0$ to $\lambda = 10^{-1}$, and at $\lambda = 0$, GBNNR still substantially outperforms BNNR. Stacking GBNNR and GF-BNNR yields no additive gain (AUPR 0.3237 vs. 0.3269 for GBNNR alone), confirming both methods draw from the same manifold signal through independent mechanisms. We also find that Rank-Adaptive BNNR (RA-BNNR), which auto-infers a data-dependent $\beta$ schedule from matrix shape and density, provides consistent AUPR gains (+6% to +35%) at negligible AUROC cost, offering a complementary, non-manifold improvement path. Case-study predictions on the full DNdataset validate the biological relevance of manifold-smoothed predictions: for Parkinson's disease, seven established antiparkinsonian agents (bromocriptine, rasagiline, amantadine, etc.) are ranked at the top despite being absent from the training data; for schizophrenia subtypes with zero known associations, the model ranks amisulpride — a first-line antipsychotic — highest.

**Availability and implementation:** Code available at https://github.com/yankangyk/bnnr-innovation.

**Keywords:** drug repositioning, matrix completion, manifold regularization, graph filtering, drug-disease association, nuclear norm minimization

---

## 摘要（繁體中文）

**研究動機：** 有界核範數正則化（BNNR）是一種廣泛應用於計算藥物重定位的矩陣補全方法，能容忍相似性測量中的噪聲。然而，BNNR 將藥物-疾病關聯矩陣視為孤立的代數對象，忽略了編碼在藥物-藥物和疾病-疾病相似性網絡中的流形幾何結構——即相似藥物應展現相似關聯模式的基本原則。這種幾何盲區降低了預測精度，導致 AUROC 與 AUPR 之間存在顯著差距。

**研究結果：** 我們提出了一個用於流形感知矩陣補全的「內-外」框架，包含兩種互補策略。圖正則化 BNNR（GBNNR）在優化*內部*運行：將 $k$ 最近鄰圖拉普拉斯正則項注入 BNNR 的 ADMM 迭代中，以內層梯度下降取代閉式 $\mathbf{W}$ 更新。在三個基準資料集上，GBNNR 將 AUPR 提升了 5.4–30.8%，同時保持 AUROC。圖濾波 BNNR（GF-BNNR）在優化*外部*運行：一個事後雙向圖低通濾波器 $\mathbf{M}_{\text{filtered}} = (\mathbf{I} + \alpha_f \mathbf{L}_d)^{-1} \mathbf{M}_{\text{raw}} (\mathbf{I} + \alpha_f \mathbf{L}_r)^{-1}$ 強制輸出在相似性流形上平滑變化。GF-BNNR 能同時改善 AUROC 和 AUPR——此模式未見於其他方法——在超稀疏資料集（DNdataset，密度 0.015%）上增益達 +5.3% AUROC 和 +27.7% AUPR，其中部分改善可歸因於 GIP 相似性融合。在一個代表性折疊上的系統性超參數掃描揭示，GBNNR 的 AUPR 增益主要由內層梯度下降貢獻，圖拓撲結構提供額外的邊際增益：正則化參數 $\lambda$ 在 $0$ 到 $10^{-1}$ 範圍內無任何可測量影響，且在 $\lambda = 0$ 時，GBNNR 仍顯著優於 BNNR。將 GBNNR 與 GF-BNNR 疊加不產生額外增益（AUPR 0.3237 vs. 單獨 GBNNR 的 0.3269），確認兩者透過獨立機制利用了相同的流形訊號。我們還發現，秩自適應 BNNR（RA-BNNR）根據矩陣形狀和密度自動推斷依賴資料的 $\beta$ 調度，以微不足道的 AUROC 代價提供了持續的 AUPR 增益（+6% 至 +35%），提供了一條互補的非流形改進路徑。在完整 DNdataset 上的案例研究驗證了流形平滑預測的生物學相關性：對於帕金森氏症，七種已確立的抗帕金森藥物（溴隱亭、雷沙吉蘭、金剛烷胺等）儘管未出現在訓練資料中，仍被排在預測前列；對於已知關聯數為零的精神分裂症亞型，模型將一線抗精神病藥物氨磺必利排至最高。

**可用性與實作：** 程式碼可於 https://github.com/yankangyk/bnnr-innovation 取得。

**關鍵詞：** 藥物重定位、矩陣補全、流形正則化、圖濾波、藥物-疾病關聯、核範數最小化

---

## Supplementary Information

**Data availability:** The three benchmark datasets (Fdataset, Cdataset, DNdataset) are publicly available from prior studies (Gottlieb et al., 2011; Yang et al., 2019). Drug similarities are computed as Tanimoto scores from DrugBank SMILES via CDK. Disease similarities are from MimMiner (Van Driel et al., 2006).

**Funding:** [To be added]

**Conflict of Interest:** None declared.
