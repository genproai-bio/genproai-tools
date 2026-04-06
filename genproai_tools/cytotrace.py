"""
细胞分化潜能评分 (CytoTRACE)

核心算法源自 CytoTRACE (Gulati et al. 2020 Science)
重实现原因: 原版是 R 包 + 需要下载 R 脚本，Python 生态无官方版

算法 (极简):
    1. gene_counts = 每个细胞中表达 >0 的基因数 (proxy for stemness)
    2. 对每个基因，计算其表达与 gene_counts 的 Pearson 相关
    3. 取 top 200 正相关基因的表达均值作为精炼评分
    4. 高分 = 更多基因表达 = 更高分化潜能 (less differentiated / more stem-like)

依赖: numpy, scipy (全部已在 bioinfo env)
"""

import numpy as np
from scipy.stats import pearsonr, rankdata
from typing import Optional


def cytotrace(
    X,
    gene_names: Optional[np.ndarray] = None,
    top_genes: int = 200,
) -> dict:
    """计算 CytoTRACE 分化潜能评分。

    Parameters
    ----------
    X : (n_cells, n_genes) array-like, 表达矩阵 (raw counts 或 normalized 均可，推荐 raw counts)
    gene_names : (n_genes,) array, 基因名 (可选，用于返回 top 相关基因)
    top_genes : 用于精炼评分的 top 正相关基因数

    Returns
    -------
    dict:
        score : (n_cells,) 精炼 CytoTRACE 评分 (0-1, 高=更未分化/干性)
        gene_counts : (n_cells,) 原始表达基因数
        gene_correlations : (n_genes,) 每个基因与 gene_counts 的 Pearson 相关
        top_gene_indices : (top_genes,) top 正相关基因的索引
    """
    X = np.asarray(X, dtype=float)
    if hasattr(X, "toarray"):
        X = X.toarray()
    n_cells, n_genes = X.shape

    # 1. Gene counts: 每个细胞有多少基因 expression > 0
    gene_counts = (X > 0).sum(axis=1).astype(float)

    # 2. 每个基因与 gene_counts 的 Pearson 相关
    correlations = np.zeros(n_genes)
    for j in range(n_genes):
        col = X[:, j]
        if col.std() > 0:
            correlations[j], _ = pearsonr(col, gene_counts)
        else:
            correlations[j] = 0.0

    # 3. Top 正相关基因
    top_k = min(top_genes, (correlations > 0).sum())
    top_idx = np.argsort(correlations)[::-1][:top_k]

    # 4. 精炼评分: top 基因表达均值 → rank normalize to [0, 1]
    if top_k > 0:
        refined = X[:, top_idx].mean(axis=1)
    else:
        refined = gene_counts.copy()

    # Rank normalize to [0, 1]
    score = rankdata(refined) / len(refined)

    result = {
        "score": score,
        "gene_counts": gene_counts,
        "gene_correlations": correlations,
        "top_gene_indices": top_idx,
    }
    if gene_names is not None:
        result["top_genes"] = np.asarray(gene_names)[top_idx]

    return result


def cytotrace_adata(adata, layer=None, key_added="cytotrace", **kwargs):
    """AnnData 接口。

    Parameters
    ----------
    adata : AnnData
    layer : 使用的 layer (None = .X)
    key_added : 存入 adata.obs 的前缀

    Modifies
    --------
    adata.obs[key_added] : 精炼评分
    adata.obs[key_added + '_gene_counts'] : 原始基因数
    adata.var[key_added + '_correlation'] : 基因与 gene_counts 的相关
    """
    X = adata.X if layer is None else adata.layers[layer]
    res = cytotrace(X, gene_names=np.asarray(adata.var_names), **kwargs)
    adata.obs[key_added] = res["score"]
    adata.obs[key_added + "_gene_counts"] = res["gene_counts"]
    adata.var[key_added + "_correlation"] = res["gene_correlations"]
    return res
