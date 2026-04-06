"""
Co-varying Neighborhood Analysis (CNA)

核心算法源自 rcna (Reshef & Rumker, Wagner 2024 Nat Biotechnol)
重实现原因: 原版 R 包绑定 Seurat，Python 版可直接在 scanpy/AnnData 生态中使用

算法:
    1. KNN 图上做 diffusion 构建 Neighborhood Abundance Matrix (NAM)
    2. 去除 batch/covariate 效应 (ridge residualization)
    3. SVD 分解 NAM
    4. 计算每个细胞的 neighborhood correlation score (ncorrs) 与表型的关联
    5. Conditional permutation FDR 控制

适用: 检测 scRNA-seq 中与协变量（治疗/时间/疾病状态）关联的细胞状态变化

依赖: numpy, scipy, sklearn (全部已在 bioinfo env)
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from typing import Optional
import warnings


def _diffuse_on_graph(adj: csr_matrix, s: np.ndarray, n_steps: int = 5) -> np.ndarray:
    """KNN 图上扩散 sample assignment matrix。

    Parameters
    ----------
    adj : (n_cells, n_cells) sparse adjacency matrix (binary KNN graph)
    s : (n_cells, n_samples) one-hot sample assignment
    n_steps : diffusion steps

    Returns
    -------
    (n_cells, n_samples) diffused neighborhood abundance matrix
    """
    degree = np.array(adj.sum(axis=1)).flatten() + 1  # +1 for self
    for _ in range(n_steps):
        s_norm = s / degree[:, None]
        s = adj.dot(s_norm) + s_norm
    # Column-normalize to proportions
    col_sums = s.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    return s / col_sums


def _residualize(X: np.ndarray, covariates: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression residualization: 去除协变量效应。"""
    model = Ridge(alpha=alpha)
    model.fit(covariates, X)
    return X - model.predict(covariates)


def covarying_neighborhoods(
    connectivities,
    sample_labels: np.ndarray,
    phenotype: np.ndarray,
    batch: Optional[np.ndarray] = None,
    covariates: Optional[np.ndarray] = None,
    n_steps: int = 5,
    n_components: int = 10,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Co-varying Neighborhood Analysis。

    Parameters
    ----------
    connectivities : (n_cells, n_cells) sparse KNN connectivity matrix
        (from scanpy: adata.obsp['connectivities'])
    sample_labels : (n_cells,) sample/patient ID per cell
    phenotype : (n_samples,) phenotype vector (continuous or binary),
        indexed to match unique(sample_labels) sorted
    batch : (n_samples,) batch labels (optional, for residualization)
    covariates : (n_samples, p) additional covariates matrix (optional)
    n_steps : diffusion steps on KNN graph
    n_components : number of SVD components to use
    n_perm : permutation test iterations
    seed : random seed

    Returns
    -------
    dict:
        ncorrs : (n_cells,) neighborhood correlation scores
        global_pvalue : overall significance (permutation-based)
        fdr_threshold_5pct : ncorrs threshold for 5% FDR
        significant_cells : (n_cells,) bool mask at 5% FDR
    """
    rng = np.random.RandomState(seed)

    # Convert adjacency to sparse if needed
    if not issparse(connectivities):
        connectivities = csr_matrix(connectivities)
    adj = (connectivities > 0).astype(float)

    # Unique samples
    unique_samples = np.unique(sample_labels)
    n_samples = len(unique_samples)
    n_cells = len(sample_labels)
    sample_to_idx = {s: i for i, s in enumerate(unique_samples)}

    # One-hot sample assignment
    s = np.zeros((n_cells, n_samples))
    for i, sl in enumerate(sample_labels):
        s[i, sample_to_idx[sl]] = 1

    # 1. Diffusion
    nam = _diffuse_on_graph(adj, s, n_steps=n_steps)  # (n_cells, n_samples)

    # 2. Residualize batch/covariates from NAM (sample-level)
    if batch is not None or covariates is not None:
        cov_list = []
        if batch is not None:
            # One-hot encode batch
            batch = np.asarray(batch)
            unique_batches = np.unique(batch)
            batch_onehot = np.zeros((n_samples, len(unique_batches)))
            for i, b in enumerate(batch):
                batch_onehot[i, np.where(unique_batches == b)[0][0]] = 1
            cov_list.append(batch_onehot)
        if covariates is not None:
            cov_list.append(np.asarray(covariates).reshape(n_samples, -1))
        cov_mat = np.hstack(cov_list)
        nam = _residualize(nam.T, cov_mat, alpha=1.0).T  # residualize columns

    # 3. SVD on NAM
    k = min(n_components, min(nam.shape) - 1)
    U, sv, Vt = svds(csr_matrix(nam.T), k=k)  # nam.T = (n_samples, n_cells)
    V = Vt.T  # (n_cells, k) — neighborhood loadings

    # 4. Compute ncorrs
    y = np.asarray(phenotype, dtype=float)
    if batch is not None or covariates is not None:
        y_resid = _residualize(y.reshape(-1, 1), cov_mat).flatten()
    else:
        y_resid = y - y.mean()
    y_resid = y_resid / (y_resid.std() + 1e-10)

    beta = U.T @ y_resid  # (k,)
    ncorrs = V @ (np.sqrt(sv) * beta / n_samples)  # (n_cells,)

    # 5. Permutation test for global significance
    null_max_ncorrs = []
    for _ in range(n_perm):
        if batch is not None:
            # Conditional permutation within batch
            y_perm = y.copy()
            for b in np.unique(batch):
                mask = batch == b
                y_perm[mask] = rng.permutation(y_perm[mask])
        else:
            y_perm = rng.permutation(y)

        if batch is not None or covariates is not None:
            y_perm_resid = _residualize(y_perm.reshape(-1, 1), cov_mat).flatten()
        else:
            y_perm_resid = y_perm - y_perm.mean()
        y_perm_resid = y_perm_resid / (y_perm_resid.std() + 1e-10)

        beta_null = U.T @ y_perm_resid
        ncorrs_null = V @ (np.sqrt(sv) * beta_null / n_samples)
        null_max_ncorrs.append(np.max(np.abs(ncorrs_null)))

    obs_max = np.max(np.abs(ncorrs))
    global_pvalue = (np.sum(np.array(null_max_ncorrs) >= obs_max) + 1) / (n_perm + 1)

    # 6. Empirical FDR threshold
    # Stack all null ncorrs for FDR estimation
    null_all = np.concatenate([
        V @ (np.sqrt(sv) * (U.T @ rng.permutation(y_resid)) / n_samples)
        for _ in range(min(100, n_perm))
    ])

    thresholds = np.percentile(np.abs(ncorrs), np.arange(90, 100, 0.5))
    fdr_5pct = np.max(np.abs(ncorrs))  # default: nothing passes
    for thr in thresholds:
        n_obs = (np.abs(ncorrs) >= thr).sum()
        n_null = (np.abs(null_all) >= thr).sum() / min(100, n_perm)
        if n_obs > 0:
            fdr = n_null / n_obs
            if fdr <= 0.05:
                fdr_5pct = thr
                break

    return {
        "ncorrs": ncorrs,
        "global_pvalue": global_pvalue,
        "fdr_threshold_5pct": fdr_5pct,
        "significant_cells": np.abs(ncorrs) >= fdr_5pct,
        "n_significant": (np.abs(ncorrs) >= fdr_5pct).sum(),
        "svd_components_used": k,
    }


def cna_adata(
    adata,
    sample_col: str,
    phenotype_col: str,
    batch_col: Optional[str] = None,
    use_rep: str = "connectivities",
    **kwargs,
) -> dict:
    """AnnData 接口。

    Parameters
    ----------
    adata : AnnData with KNN graph computed (sc.pp.neighbors)
    sample_col : .obs column for sample/patient ID
    phenotype_col : .obs column for phenotype (sample-level; will be averaged per sample)
    batch_col : .obs column for batch (optional)
    """
    conn = adata.obsp[use_rep]
    sample_labels = adata.obs[sample_col].values

    # Aggregate phenotype to sample level
    sample_pheno = adata.obs.groupby(sample_col)[phenotype_col].mean()
    unique_samples = np.unique(sample_labels)
    phenotype = np.array([sample_pheno[s] for s in unique_samples])

    batch = None
    if batch_col:
        sample_batch = adata.obs.groupby(sample_col)[batch_col].first()
        batch = np.array([sample_batch[s] for s in unique_samples])

    result = covarying_neighborhoods(conn, sample_labels, phenotype, batch=batch, **kwargs)

    adata.obs["cna_ncorrs"] = result["ncorrs"]
    adata.obs["cna_significant"] = result["significant_cells"]

    return result
