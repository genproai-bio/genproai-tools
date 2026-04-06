"""
单细胞药物敏感性预测 (Drug Sensitivity)

核心算法源自 DREEP (Chen et al. 2023 BMC Medicine, PMID: 37990218)
重实现原因: 原版 R+C++ 依赖 gf-icf 包(同作者)，无 Python 版

算法:
    1. gf-icf 标准化 (类 NLP tf-idf，自然处理 scRNA dropout):
       - gf(i,j) = expr(i,j) / sum_j expr(i,j)  (gene frequency per cell)
       - icf(j) = log2(N / n_j)  (inverse cell frequency)
       - score(i,j) = gf(i,j) × icf(j)
    2. 每个细胞取 top-K 高 gf-icf 基因 → "cell identity gene set"
    3. GPDS (Gene-Perturbation Drug Signature):
       预计算: Spearman(IC50, gene_expr) across cell lines → 排序基因列表
       基因按 correlation 从正(耐药)到负(敏感)排列
    4. Rank-based enrichment: cell gene set vs GPDS
       正分 = 敏感 (细胞特征基因在 GPDS 敏感端富集)
       负分 = 耐药

性能注意:
    - AUC ~0.73 (GDSC2), ~0.69 (CTRP2), ~0.66 (PRISM)
    - 这是"闭卷"场景 (无药物处理数据) 的 SOTA
    - 输出定位为优先级排序，非精确预测

依赖: numpy, scipy, pandas (全部已在 bioinfo env)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. gf-icf normalization
# ---------------------------------------------------------------------------

def gf_icf(X):
    """gf-icf normalization for scRNA-seq (analogous to tf-idf in NLP).

    Up-weights cell-specific genes, down-weights housekeeping genes.
    Naturally handles dropout by penalizing ubiquitously expressed genes.

    Parameters
    ----------
    X : (n_cells, n_genes) array-like
        Expression matrix. Must NOT be log-transformed (use TPM, CPM, or raw counts).
        Sparse matrices are densified.

    Returns
    -------
    np.ndarray : (n_cells, n_genes) gf-icf scores
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    n_cells, n_genes = X.shape

    # Gene frequency: expression fraction per cell
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    gf = X / row_sums

    # Inverse cell frequency: penalize genes expressed in many cells
    n_expressing = (X > 0).sum(axis=0).astype(np.float64)
    n_expressing[n_expressing == 0] = 1.0
    icf = np.log2(n_cells / n_expressing)

    return gf * icf


# ---------------------------------------------------------------------------
# 2. Cell identity gene extraction
# ---------------------------------------------------------------------------

def cell_identity_genes(gf_icf_matrix, top_k=250):
    """Extract top-K identity genes per cell by gf-icf score.

    Parameters
    ----------
    gf_icf_matrix : (n_cells, n_genes) array
    top_k : int, number of top genes per cell

    Returns
    -------
    np.ndarray : (n_cells, top_k) integer indices of top genes
    """
    # argpartition is O(n) vs O(n log n) for argsort — much faster
    n_cells, n_genes = gf_icf_matrix.shape
    k = min(top_k, n_genes)

    top_indices = np.empty((n_cells, k), dtype=np.intp)
    for i in range(n_cells):
        row = gf_icf_matrix[i]
        # Get top-k indices (unordered), then sort them by value
        idx = np.argpartition(row, -k)[-k:]
        top_indices[i] = idx[np.argsort(row[idx])[::-1]]

    return top_indices


# ---------------------------------------------------------------------------
# 3. Rank-based enrichment scoring
# ---------------------------------------------------------------------------

def rank_enrichment(cell_gene_indices, gpds_ranks, batch_size=100):
    """Vectorized mean-rank enrichment scoring.

    For each cell-drug pair, compute normalized mean rank of cell identity
    genes in the drug's GPDS ranking.

    Score = 2 * (mean_rank / (n_genes - 1)) - 1
    Range [-1, +1]:  +1 = sensitive,  -1 = resistant

    Parameters
    ----------
    cell_gene_indices : (n_cells, top_k) int array
        Gene indices from cell_identity_genes().
    gpds_ranks : (n_drugs, n_genes) float array
        Pre-computed GPDS. Each row: genes ranked from resistance (0)
        to sensitivity (n_genes-1).
    batch_size : int
        Process this many drugs at a time to control memory.

    Returns
    -------
    np.ndarray : (n_cells, n_drugs) enrichment scores
    """
    n_cells, top_k = cell_gene_indices.shape
    n_drugs, n_genes = gpds_ranks.shape

    scores = np.empty((n_cells, n_drugs), dtype=np.float32)

    for d_start in range(0, n_drugs, batch_size):
        d_end = min(d_start + batch_size, n_drugs)
        batch = gpds_ranks[d_start:d_end]  # (batch, n_genes)

        # For each drug in batch, gather ranks of each cell's identity genes
        # batch[:, cell_gene_indices] → (batch, n_cells, top_k)
        batch_ranks = batch[:, cell_gene_indices]
        # Mean rank per cell per drug, then normalize to [-1, +1]
        mean_ranks = batch_ranks.mean(axis=2)  # (batch, n_cells)
        scores[:, d_start:d_end] = (
            2.0 * mean_ranks / max(n_genes - 1, 1) - 1.0
        ).T

    return scores


# ---------------------------------------------------------------------------
# 4. Main scoring function
# ---------------------------------------------------------------------------

def drug_sensitivity_score(
    X,
    gene_names: np.ndarray,
    gpds: pd.DataFrame,
    top_k: int = 250,
) -> pd.DataFrame:
    """Predict per-cell drug sensitivity from scRNA-seq expression.

    Parameters
    ----------
    X : (n_cells, n_genes) array-like
        Expression matrix. NOT log-transformed (TPM/CPM/counts).
    gene_names : (n_genes,) array
        Gene symbols matching columns of X.
    gpds : pd.DataFrame
        Pre-built GPDS signatures. Index = drug names, columns = gene symbols,
        values = ranks (0 = most resistance-associated, max = most sensitivity).
    top_k : int
        Number of top gf-icf genes per cell (default 250).

    Returns
    -------
    pd.DataFrame : (n_cells, n_drugs) enrichment scores.
        Positive = predicted sensitive, negative = predicted resistant.
    """
    gene_names = np.asarray(gene_names)

    # Intersect genes between expression and GPDS
    common_genes = np.intersect1d(gene_names, gpds.columns)
    coverage = len(common_genes) / len(gpds.columns)
    logger.info(
        f"Gene overlap: {len(common_genes)}/{len(gpds.columns)} "
        f"GPDS genes ({coverage:.1%})"
    )
    if coverage < 0.5:
        logger.warning(
            f"Gene coverage {coverage:.1%} < 50%. Results may be unreliable. "
            "Check gene symbol format (Hugo symbols expected)."
        )

    # Subset and align
    expr_mask = np.isin(gene_names, common_genes)
    expr_sub = np.asarray(X)[:, expr_mask]
    if hasattr(expr_sub, "toarray"):
        expr_sub = expr_sub.toarray()
    gene_sub = gene_names[expr_mask]

    gpds_sub = gpds[gene_sub].values  # (n_drugs, n_common_genes)

    # Re-rank GPDS on common genes (ranks must be contiguous)
    gpds_ranks = np.empty_like(gpds_sub)
    for d in range(gpds_sub.shape[0]):
        order = np.argsort(gpds_sub[d])
        gpds_ranks[d, order] = np.arange(len(order))

    # gf-icf normalization
    gficf = gf_icf(expr_sub.astype(np.float64))

    # Extract cell identity genes
    cell_genes = cell_identity_genes(gficf, top_k=top_k)

    # Score
    scores = rank_enrichment(cell_genes, gpds_ranks.astype(np.float32))

    return pd.DataFrame(
        scores,
        columns=gpds.index,  # drug names
    )


# ---------------------------------------------------------------------------
# 5. AnnData interface
# ---------------------------------------------------------------------------

def drug_sensitivity_adata(
    adata,
    gpds: pd.DataFrame,
    layer: Optional[str] = None,
    top_k: int = 250,
    key_added: str = "drug_sensitivity",
) -> pd.DataFrame:
    """AnnData interface for drug sensitivity scoring.

    Parameters
    ----------
    adata : AnnData
        Must contain non-log expression (TPM/CPM/counts).
        If .X is log-transformed, specify a raw layer.
    gpds : pd.DataFrame
        Pre-built GPDS signatures.
    layer : str or None
        Layer to use (None = .X). Use if .X is log-transformed.
    top_k : int
        Top gf-icf genes per cell.
    key_added : str
        Prefix for storing top-drug results in adata.obs.

    Returns
    -------
    pd.DataFrame : (n_cells, n_drugs) full score matrix.

    Modifies
    --------
    adata.obs[key_added + '_top_drug'] : most sensitive drug per cell
    adata.obs[key_added + '_top_score'] : score of the most sensitive drug
    adata.obsm[key_added] : full (n_cells, n_drugs) score matrix
    """
    X = adata.X if layer is None else adata.layers[layer]
    gene_names = np.asarray(adata.var_names)

    scores_df = drug_sensitivity_score(X, gene_names, gpds, top_k=top_k)
    scores_df.index = adata.obs_names

    # Store top drug per cell
    adata.obs[key_added + "_top_drug"] = scores_df.idxmax(axis=1).values
    adata.obs[key_added + "_top_score"] = scores_df.max(axis=1).values

    # Store full matrix
    adata.obsm[key_added] = scores_df.values

    return scores_df


# ---------------------------------------------------------------------------
# 6. GPDS construction utilities
# ---------------------------------------------------------------------------

def build_gpds(
    expr_df: pd.DataFrame,
    ic50_df: pd.DataFrame,
    min_cell_lines: int = 10,
) -> pd.DataFrame:
    """Build Gene-Perturbation Drug Signatures from pharmacogenomics data.

    For each drug, compute Spearman correlation between IC50 and gene
    expression across cell lines. Genes are ranked from most resistance-
    associated (rank 0) to most sensitivity-associated (rank max).

    Parameters
    ----------
    expr_df : pd.DataFrame
        Cell line expression. Index = cell line IDs, columns = gene symbols.
        Should be TPM or log2(TPM+1).
    ic50_df : pd.DataFrame
        Drug sensitivity. Index = cell line IDs, columns = drug names.
        Values = IC50 or ln(IC50). Higher = more resistant.
    min_cell_lines : int
        Minimum cell lines with both expression and IC50 for a drug.

    Returns
    -------
    pd.DataFrame : (n_drugs, n_genes) GPDS rank matrix.
        Index = drug names, columns = gene symbols.
    """
    common_cls = expr_df.index.intersection(ic50_df.index)
    logger.info(f"Common cell lines: {len(common_cls)}")

    expr = expr_df.loc[common_cls]
    ic50 = ic50_df.loc[common_cls]

    # Filter genes with zero variance
    gene_std = expr.std()
    valid_genes = gene_std[gene_std > 0].index
    expr = expr[valid_genes]
    logger.info(f"Valid genes (non-zero variance): {len(valid_genes)}")

    gpds_dict = {}
    n_drugs = ic50.shape[1]

    for i, drug in enumerate(ic50.columns):
        drug_ic50 = ic50[drug].dropna()
        if len(drug_ic50) < min_cell_lines:
            continue

        cls = drug_ic50.index.intersection(expr.index)
        if len(cls) < min_cell_lines:
            continue

        # Spearman correlation: IC50 vs expression for each gene
        ic50_vals = drug_ic50.loc[cls].values
        correlations = np.array([
            spearmanr(ic50_vals, expr.loc[cls, g].values).statistic
            for g in valid_genes
        ])

        # Rank: 0 = most positive correlation (resistant) → max = most negative (sensitive)
        # argsort of correlations descending: highest corr gets rank 0
        order = np.argsort(-correlations)  # descending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))

        gpds_dict[drug] = ranks

        if (i + 1) % 100 == 0:
            logger.info(f"GPDS built: {i + 1}/{n_drugs} drugs")

    gpds = pd.DataFrame(gpds_dict, index=valid_genes).T
    logger.info(f"Final GPDS: {gpds.shape[0]} drugs × {gpds.shape[1]} genes")
    return gpds


def load_gpds(path: str) -> pd.DataFrame:
    """Load pre-built GPDS from parquet file.

    Parameters
    ----------
    path : str
        Path to .parquet file (index=drug names, columns=gene symbols, values=ranks).
    """
    return pd.read_parquet(path)


def save_gpds(gpds: pd.DataFrame, path: str):
    """Save GPDS to parquet (compact, fast I/O).

    Parameters
    ----------
    gpds : pd.DataFrame from build_gpds()
    path : str, must end with .parquet
    """
    # Convert ranks to uint16 to save space (max 65535 genes, more than enough)
    gpds.astype(np.uint16).to_parquet(path, engine="pyarrow")
    logger.info(f"GPDS saved to {path}")
