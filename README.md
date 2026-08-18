# GMC：图多视角补全——药物重定位

随机条目掩码（CVa）协议下的药物-疾病关联预测，基于**多视角低秩补全**：在相同相似性数据上求解一个核范数补全问题，分别用矩阵（融合联合块）和张量（每模态切片）两种几何视角，叠加冷启动受限 KNN 填充，通过无量纲秩融合组合。

GMC 在四个基准上均达到最高 AUPR（F 0.6569 / C 0.7285 / CTD 0.3714 / Y 0.7404）。GMC-E 是 GMC 与互补基线在分数层面的融合——作为上界参考，**不是**所提出的方法（其组成在测试折上选择）。

## 快速开始

```bash
pip install -r requirements.txt

# 生成 CVa 折叠，运行 GMC（四个数据集同一配置）
python scripts/gen_folds.py
python scripts/run_gmc.py --datasets Fdataset Cdataset CTDdataset2023 Ydataset

# 评估基线、构建对比表
python scripts/evaluate.py Fdataset Cdataset CTDdataset2023 Ydataset
python scripts/build_comparison.py
python scripts/significance_test.py --model gmc_unified
```

MATLAB 仅用于运行已发表基线：`Baseline/run_baseline.m`、`Baseline/run_multiGMF.m`。

## 主要结果（10 折 CVa，AUPR）

| 方法 | F | C | CTD | Y |
|--------|---|---|-----|---|
| 最佳基线 | 0.6453 (DNMFDDA) | 0.7221 (DNMFDDA) | 0.3287 (ITRPCA) | 0.7279 (OMC) |
| **GMC（本文）** | **0.6569** | **0.7285** | **0.3714** | **0.7404** |
| GMC-E（上界参考） | 0.6730 | 0.7394 | 0.3714 | 0.7522 |

GMC 在 CTD/Y 上显著优于所有基线（p=0.002），在 F/C 上领先 DNMFDDA 但未达显著（p=0.131）。CTD 上无融合组合可提升 GMC，故 GMC-E = GMC。

## 项目结构

```
gmc/               # 核心算法包
  model.py         #   gmc_predict + coldstart_fill + rnorm01
  factorization.py #   补全求解器（BNNR、ITRPCA、graph_reg_nmf、deep_semi_nmf）
  ensemble.py      #   GMC-E：查找/加载预测、秩平均、物化
  wknn.py          #   WKNN 软标签传播
  filter.py        #   graph_filter + laplacian + sparsify_graph
  cv.py            #   交叉验证（CVa / CVc）
  metrics.py       #   AUROC/AUPR + Top-K
  helpers.py       #   数据集加载、逐折评估、结果路径
scripts/           # 运行脚本（不含算法代码）
data/              # 数据集（multiGMF 5+2 格式）
papers/            # 论文手稿（gmc_manuscript.tex） + 图表
Baseline/          # 参考方法仓库 + MATLAB 基线驱动
```

## 统一配置

```
fill=knn, block=sym, wknn_k=10, alpha=0.5, maxiter=40, rank_cap=400,
trindex=observed, w_bnnr=0.5, w_tensor=0.5, fusion=rank
```

无图滤波、无后处理滤波。四个数据集共享同一补全核心。

## 依赖

Python：numpy、scipy、scikit-learn、pandas、matplotlib。基线仅需 MATLAB。