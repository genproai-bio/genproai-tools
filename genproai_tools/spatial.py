"""
空间蛋白组/转录组交互分析

核心算法源自 Scimap (Nirmal et al., JOSS 2024; Harvard LSP)
重实现原因: 原包依赖 scipy<=1.12/zarr==2.10.3/numpy<2，与 bioinfo env 不兼容

算法:
    spatial_interaction: k-NN + permutation test 检验细胞类型空间交互
    spatial_pscore: 邻域共现评分 (Proximity Volume / Density)

依赖: numpy, pandas, scipy, joblib (全部已在 bioinfo env)
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from scipy.stats import norm
from typing import Optional, List, Union
import warnings


def _build_neighbors(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 30,
    radius: float = 50.0,
) -> list:
    """构建空间邻域。

    Parameters
    ----------
    coords : (n_cells, 2 or 3) array of spatial coordinates
    method : 'knn' or 'radius'
    k : number of neighbors for knn mode
    radius : search radius for radius mode

    Returns
    -------
    list of arrays, each containing neighbor indices for one cell
    """
    tree = cKDTree(coords)
    if method == "knn":
        _, indices = tree.query(coords, k=k + 1)
        # 第 0 列是自身，去掉
        return [row[1:] for row in indices]
    elif method == "radius":
        all_neighbors = tree.query_ball_point(coords, r=radius)
        # 去掉自身
        return [np.array([j for j in nb if j != i]) for i, nb in enumerate(all_neighbors)]
    else:
        raise ValueError(f"method must be 'knn' or 'radius', got '{method}'")


def spatial_interaction(
    coords: np.ndarray,
    phenotype: np.ndarray,
    method: str = "knn",
    k: int = 30,
    radius: float = 50.0,
    n_perm: int = 1000,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """细胞类型空间交互分析 (permutation-based)。

    对每对细胞类型 (A, B)，检验 A 的邻域中 B 的出现频率是否显著
    高于（interaction）或低于（avoidance）随机预期。

    Parameters
    ----------
    coords : (n_cells, 2) array, 细胞空间坐标 (X, Y)
    phenotype : (n_cells,) array, 细胞类型标签
    method : 'knn' or 'radius'
    k : knn 邻居数
    radius : radius 模式搜索半径
    n_perm : 排列检验次数
    n_jobs : 并行核数 (-1 = 全部)
    seed : 随机种子

    Returns
    -------
    pd.DataFrame with columns:
        center_type, neighbor_type, observed, expected, zscore, pvalue, direction
    """
    phenotype = np.asarray(phenotype)
    categories = np.unique(phenotype)
    n_cats = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    pheno_idx = np.array([cat_to_idx[p] for p in phenotype])

    # 1. 构建邻域
    neighbors = _build_neighbors(coords, method=method, k=k, radius=radius)

    # 2. 计算观测共现矩阵 (center_type x neighbor_type)
    observed = np.zeros((n_cats, n_cats), dtype=float)
    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            continue
        ci = pheno_idx[i]
        for nj in pheno_idx[nb]:
            observed[ci, nj] += 1

    # 按列归一化 (除以 neighbor_type 的总细胞数)
    type_counts = np.bincount(pheno_idx, minlength=n_cats).astype(float)
    type_counts[type_counts == 0] = 1  # 防除零
    observed_norm = observed / type_counts[np.newaxis, :]

    # 3. 排列检验
    rng = np.random.RandomState(seed)

    def _one_perm(_):
        perm_idx = rng.permutation(pheno_idx)
        mat = np.zeros((n_cats, n_cats), dtype=float)
        for i, nb in enumerate(neighbors):
            if len(nb) == 0:
                continue
            ci = pheno_idx[i]  # center 不变
            for nj in perm_idx[nb]:  # neighbor 标签置换
                mat[ci, nj] += 1
        return mat / type_counts[np.newaxis, :]

    perm_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_perm)(i) for i in range(n_perm)
    )
    perm_stack = np.stack(perm_results)  # (n_perm, n_cats, n_cats)

    # 4. Z-score 和 p-value
    perm_mean = perm_stack.mean(axis=0)
    perm_std = perm_stack.std(axis=0)
    perm_std[perm_std == 0] = 1  # 防除零

    zscore = (observed_norm - perm_mean) / perm_std
    pvalue = 2 * norm.sf(np.abs(zscore))  # 双侧
    direction = np.sign(observed_norm - perm_mean)

    # 5. 组装结果
    rows = []
    for i, ci in enumerate(categories):
        for j, cj in enumerate(categories):
            rows.append({
                "center_type": ci,
                "neighbor_type": cj,
                "observed": observed_norm[i, j],
                "expected": perm_mean[i, j],
                "zscore": zscore[i, j],
                "pvalue": pvalue[i, j],
                "direction": "interaction" if direction[i, j] > 0 else "avoidance",
            })
    return pd.DataFrame(rows)


def spatial_pscore(
    coords: np.ndarray,
    phenotype: np.ndarray,
    proximity: List[str],
    method: str = "knn",
    k: int = 30,
    radius: float = 50.0,
) -> dict:
    """邻域共现评分 (Proximity Score)。

    对指定的细胞类型组合，计算它们在空间上共现的密度。

    Parameters
    ----------
    coords : (n_cells, 2) array
    phenotype : (n_cells,) array
    proximity : list of cell type names that must co-occur in a neighborhood
    method : 'knn' or 'radius'
    k, radius : neighbor parameters

    Returns
    -------
    dict with keys:
        proximity_volume : 交互邻域数 / 总细胞数
        proximity_density : 交互邻域数 / proximity 类型的总细胞数
        interaction_mask : (n_cells,) bool array, 标记交互位点细胞
    """
    phenotype = np.asarray(phenotype)
    neighbors = _build_neighbors(coords, method=method, k=k, radius=radius)

    # 对每个细胞，检查其邻域是否同时包含所有 proximity 类型
    interaction_mask = np.zeros(len(phenotype), dtype=bool)
    n_qualifying = 0

    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            continue
        nb_types = set(phenotype[nb])
        # 包含自身
        nb_types.add(phenotype[i])
        if all(p in nb_types for p in proximity):
            n_qualifying += 1
            # 标记邻域中属于 proximity 类型的细胞
            interaction_mask[i] = True
            for j in nb:
                if phenotype[j] in proximity:
                    interaction_mask[j] = True

    n_total = len(phenotype)
    n_proximity_cells = sum(phenotype[i] in proximity for i in range(n_total))

    return {
        "proximity_volume": n_qualifying / n_total if n_total > 0 else 0,
        "proximity_density": n_qualifying / n_proximity_cells if n_proximity_cells > 0 else 0,
        "interaction_mask": interaction_mask,
    }


def spatial_interaction_adata(
    adata,
    phenotype_col: str = "cell_type",
    x_col: str = "X_centroid",
    y_col: str = "Y_centroid",
    image_col: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """AnnData 接口: 从 adata.obs 提取坐标和表型，运行 spatial_interaction。

    Parameters
    ----------
    adata : AnnData with spatial coordinates in .obs
    phenotype_col : column name for cell type labels
    x_col, y_col : column names for spatial coordinates
    image_col : if provided, run per-image and concatenate results
    **kwargs : passed to spatial_interaction()

    Returns
    -------
    pd.DataFrame, 结果同 spatial_interaction()，多图像时多一列 'image_id'
    """
    if image_col is not None and image_col in adata.obs.columns:
        results = []
        for img_id, sub in adata.obs.groupby(image_col):
            idx = sub.index
            coords = np.column_stack([sub[x_col].values, sub[y_col].values])
            pheno = sub[phenotype_col].values
            res = spatial_interaction(coords, pheno, **kwargs)
            res["image_id"] = img_id
            results.append(res)
        return pd.concat(results, ignore_index=True)
    else:
        coords = np.column_stack([adata.obs[x_col].values, adata.obs[y_col].values])
        pheno = adata.obs[phenotype_col].values
        return spatial_interaction(coords, pheno, **kwargs)
