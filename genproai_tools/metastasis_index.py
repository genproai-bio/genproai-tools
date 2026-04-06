"""
原发灶 vs 转移灶 TME 重塑指标 (Metastasis TME Remodeling Indices)

核心算法源自 PBMA (Xing 2025 Cancer Cell)
重实现原因: 原版是 R pipeline 绑定 Seurat，我们只需两个公式

算法:
    RI (Remodeling Index): Soergel distance between PT and Met cell composition vectors
    TI (Transformation Index): signed Soergel distance of functional state scores

依赖: numpy, pandas (已在 bioinfo env)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict


def soergel_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Soergel distance (= 1 - Ruzicka similarity).

    d = sum(|x_i - y_i|) / sum(max(x_i, y_i))
    Range: [0, 1]. 0 = identical, 1 = maximally different.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.sum(np.maximum(x, y))
    if denom == 0:
        return 0.0
    return np.sum(np.abs(x - y)) / denom


def remodeling_index(
    cell_types: np.ndarray,
    condition: np.ndarray,
    primary_label: str = "PT",
    met_label: str = "Met",
    categories: Optional[List[str]] = None,
) -> dict:
    """Remodeling Index: 原发灶→转移灶的细胞组成整体变化。

    Parameters
    ----------
    cell_types : (n_cells,) 细胞类型标签 (e.g., Tumor/Immune/Stromal, 或更细分)
    condition : (n_cells,) 原发灶/转移灶标签
    primary_label, met_label : condition 中的标签
    categories : 用于计算的细胞类型列表。None 则自动使用所有类型

    Returns
    -------
    dict: ri (float), pt_composition (Series), met_composition (Series)
    """
    df = pd.DataFrame({"cell_type": cell_types, "condition": condition})

    if categories is None:
        categories = sorted(df["cell_type"].unique())

    pt_counts = df[df["condition"] == primary_label]["cell_type"].value_counts()
    met_counts = df[df["condition"] == met_label]["cell_type"].value_counts()

    pt_comp = np.array([pt_counts.get(c, 0) for c in categories], dtype=float)
    met_comp = np.array([met_counts.get(c, 0) for c in categories], dtype=float)

    # Normalize to proportions
    pt_comp = pt_comp / pt_comp.sum() if pt_comp.sum() > 0 else pt_comp
    met_comp = met_comp / met_comp.sum() if met_comp.sum() > 0 else met_comp

    ri = soergel_distance(pt_comp, met_comp)

    return {
        "ri": ri,
        "pt_composition": pd.Series(pt_comp, index=categories),
        "met_composition": pd.Series(met_comp, index=categories),
    }


def transformation_index(
    scores_pt: np.ndarray,
    scores_met: np.ndarray,
) -> dict:
    """Transformation Index: 功能状态从原发灶→转移灶的有向变化。

    Parameters
    ----------
    scores_pt : (n_pt_cells,) 原发灶中某功能评分 (e.g., M1 score, 已 rescale 到 [0,1])
    scores_met : (n_met_cells,) 转移灶中同一评分

    Returns
    -------
    dict: ti (float), direction (str), distance (float)
    """
    scores_pt = np.asarray(scores_pt, dtype=float)
    scores_met = np.asarray(scores_met, dtype=float)

    # Soergel distance on mean score vectors (simplified: 1D case)
    pt_mean = np.median(scores_pt)
    met_mean = np.median(scores_met)

    # For multi-dimensional case, use full Soergel; for 1D, simplify
    distance = abs(met_mean - pt_mean) / max(abs(met_mean), abs(pt_mean), 1e-10)
    sign = 1 if met_mean >= pt_mean else -1
    ti = sign * distance

    return {
        "ti": ti,
        "direction": "up_in_met" if sign > 0 else "down_in_met",
        "distance": distance,
        "pt_median": pt_mean,
        "met_median": met_mean,
    }


def compare_pt_vs_met(
    adata,
    condition_col: str,
    celltype_col: str,
    primary_label: str = "PT",
    met_label: str = "Met",
    score_cols: Optional[List[str]] = None,
) -> dict:
    """AnnData 接口: 一次性计算 RI + 多个功能维度的 TI。

    Parameters
    ----------
    adata : AnnData with cell type and PT/Met labels in .obs
    condition_col : .obs column for PT/Met
    celltype_col : .obs column for cell type
    score_cols : .obs columns for functional scores (TI will be computed for each)

    Returns
    -------
    dict: ri_result, ti_results (per score_col per cell_type)
    """
    ri_result = remodeling_index(
        adata.obs[celltype_col].values,
        adata.obs[condition_col].values,
        primary_label=primary_label,
        met_label=met_label,
    )

    ti_results = []
    if score_cols:
        for col in score_cols:
            if col not in adata.obs.columns:
                continue
            for ct in sorted(adata.obs[celltype_col].unique()):
                mask_pt = (adata.obs[condition_col] == primary_label) & (adata.obs[celltype_col] == ct)
                mask_met = (adata.obs[condition_col] == met_label) & (adata.obs[celltype_col] == ct)
                if mask_pt.sum() >= 5 and mask_met.sum() >= 5:
                    ti = transformation_index(
                        adata.obs.loc[mask_pt, col].values,
                        adata.obs.loc[mask_met, col].values,
                    )
                    ti["score_col"] = col
                    ti["cell_type"] = ct
                    ti_results.append(ti)

    return {"ri": ri_result, "ti": pd.DataFrame(ti_results) if ti_results else pd.DataFrame()}
