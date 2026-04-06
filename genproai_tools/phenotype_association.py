"""
bulk-scRNA 表型关联分析 (Phenotype Association)

核心算法源自 Shears (Marteau 2026 Cancer Cell; icbi-lab Innsbruck)
重实现原因: 原版有 6 个已知 bug，且 API 设计有多处陷阱

修复的 bug:
    1. recipe 不返回新对象导致 KeyError → 本版直接原地修改
    2. family 必须是 statsmodels 对象 → 本版接受字符串
    3. batch_key 必填 → 本版可选
    4. cell_weights inplace=False return 错误变量名 → 本版修复
    5. Cox ConvergenceError 未捕获 → 本版 try-except 返回 NaN
    6. init_kwargs 解析 bug → 本版不走原始路径

依赖: numpy, pandas, sklearn, statsmodels, lifelines, joblib (全部已在 bioinfo env)
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.linear_model import Ridge
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from typing import Optional
import warnings


def preprocess(adata_sc, adata_bulk, n_top_genes=2000):
    """预处理: 交集基因 → HVG → quantile normalization。

    直接修改 adata_sc 和 adata_bulk (原版 bug#1: 返回新对象但用户不知道要接)。

    Parameters
    ----------
    adata_sc : AnnData, scRNA-seq (raw counts)
    adata_bulk : AnnData, bulk RNA-seq (TPM/FPKM)
    n_top_genes : HVG 数量
    """
    import scanpy as sc

    # 交集基因
    common = adata_sc.var_names.intersection(adata_bulk.var_names)
    adata_sc._inplace_subset_var(common)
    adata_bulk._inplace_subset_var(common)

    # HVG (seurat_v3 支持 raw counts)
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=n_top_genes, flavor="seurat_v3", subset=True)
    adata_bulk._inplace_subset_var(adata_sc.var_names)

    # Quantile normalization
    adata_sc.layers["qn"] = sklearn.preprocessing.quantile_transform(
        adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else adata_sc.X, copy=True
    )
    adata_bulk.layers["qn"] = sklearn.preprocessing.quantile_transform(
        adata_bulk.X.toarray() if hasattr(adata_bulk.X, "toarray") else adata_bulk.X, copy=True
    )

    print(f"  Preprocessed: {len(common)} common genes → {adata_sc.shape[1]} HVGs")


def compute_cell_weights(
    adata_sc,
    adata_bulk,
    layer="qn",
    alpha=None,
    n_jobs=None,
    key_added="cell_weights",
):
    """Ridge 反卷积: 对每个 bulk sample，用 scRNA 矩阵做 Ridge 回归得到 cell weights。

    Parameters
    ----------
    adata_sc, adata_bulk : preprocessed AnnData (需有 layer)
    layer : quantile-normalized layer name
    alpha : Ridge 正则化参数。None 则自动设为 10 * n_cells
    n_jobs : 并行核数
    key_added : 存入 adata_sc.obsm 的 key

    Returns
    -------
    pd.DataFrame, (n_cells, n_bulk_samples) weight matrix (也存入 adata_sc.obsm)
    """
    sc_mat = adata_sc.layers[layer]
    if hasattr(sc_mat, "toarray"):
        sc_mat = sc_mat.toarray()
    bulk_mat = adata_bulk.layers[layer]
    if hasattr(bulk_mat, "toarray"):
        bulk_mat = bulk_mat.toarray()

    if alpha is None:
        alpha = 10 * adata_sc.shape[0]

    def _fit_one(bulk_sample):
        model = Ridge(alpha=alpha, positive=True, random_state=0)
        model.fit(sc_mat.T, bulk_sample)
        return model.coef_

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one)(bulk_mat[i, :]) for i in range(adata_bulk.shape[0])
    )

    weights = pd.DataFrame(
        np.array(results).T,
        index=adata_sc.obs_names,
        columns=adata_bulk.obs_names,
    )
    adata_sc.obsm[key_added] = weights
    print(f"  Cell weights: {weights.shape}")
    return weights  # Bug#4 修复: 原版 return res_df (未定义变量)


def association_glm(
    adata_sc,
    adata_bulk,
    dep_var,
    family="binomial",
    covariates=None,
    cell_weights_key="cell_weights",
    n_jobs=None,
):
    """GLM per-cell 关联检验。

    Parameters
    ----------
    dep_var : adata_bulk.obs 中的表型列名
    family : 'binomial', 'gaussian', or 'poisson' (Bug#2 修复: 原版只接受 statsmodels 对象)
    covariates : 额外协变量列名列表 (可选)
    """
    import statsmodels.api as sm

    # Bug#2 修复: 接受字符串
    family_map = {
        "binomial": sm.families.Binomial(),
        "gaussian": sm.families.Gaussian(),
        "poisson": sm.families.Poisson(),
    }
    if isinstance(family, str):
        family = family_map.get(family.lower(), sm.families.Binomial())

    weights = adata_sc.obsm[cell_weights_key]
    bulk_obs = adata_bulk.obs[[dep_var]].copy()
    if covariates:
        for c in covariates:
            if c in adata_bulk.obs.columns:
                bulk_obs[c] = adata_bulk.obs[c]

    def _test_cell(cell_idx):
        cell_w = weights.iloc[cell_idx].values
        df = bulk_obs.copy()
        df["cell_weight"] = cell_w
        X = sm.add_constant(df.drop(columns=[dep_var]))
        y = df[dep_var].values.astype(float)
        try:
            res = sm.GLM(y, X, family=family).fit()
            return res.pvalues["cell_weight"], res.params["cell_weight"]
        except Exception:
            return np.nan, np.nan

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_test_cell)(i) for i in range(adata_sc.shape[0])
    )

    pvals, coefs = zip(*results)
    return pd.DataFrame({
        "pvalue": pvals,
        "coef": coefs,
    }, index=adata_sc.obs_names)


def association_cox(
    adata_sc,
    adata_bulk,
    duration_col,
    event_col,
    covariates=None,
    cell_weights_key="cell_weights",
    penalizer=0.1,
    n_jobs=None,
):
    """Cox PH per-cell 生存关联检验。

    Bug#5 修复: 原版不 catch ConvergenceError → 整个进程崩溃
    Bug#6 修复: 原版 init_kwargs 解析 bug → 本版直接在 CoxPHFitter 构造函数传 penalizer
    """
    from lifelines import CoxPHFitter

    weights = adata_sc.obsm[cell_weights_key]
    bulk_obs = adata_bulk.obs[[duration_col, event_col]].copy().astype(float)
    if covariates:
        for c in covariates:
            if c in adata_bulk.obs.columns:
                bulk_obs[c] = adata_bulk.obs[c].astype(float)

    def _test_cell(cell_idx):
        cell_w = weights.iloc[cell_idx].values
        df = bulk_obs.copy()
        df["cell_weight"] = cell_w
        try:
            # Bug#5 修复: try-except
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Bug#6 修复: penalizer 直接传构造函数
                cph = CoxPHFitter(penalizer=penalizer)
                cph.fit(df, duration_col=duration_col, event_col=event_col,
                        formula="cell_weight" + ((" + " + " + ".join(covariates)) if covariates else ""))
                return cph.summary.at["cell_weight", "p"], cph.summary.at["cell_weight", "coef"]
        except Exception:
            return np.nan, np.nan

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_test_cell)(i) for i in range(adata_sc.shape[0])
    )

    pvals, coefs = zip(*results)
    return pd.DataFrame({
        "pvalue": pvals,
        "coef": coefs,
    }, index=adata_sc.obs_names)


def aggregate_by_celltype(
    adata_sc,
    result_df,
    groupby="cell_type",
    batch_key=None,
):
    """按细胞类型聚合统计。

    Bug#3 修复: batch_key 可选 (原版必填)。
    """
    ct = adata_sc.obs[groupby]
    result_df = result_df.copy()
    result_df["cell_type"] = ct.values
    if batch_key and batch_key in adata_sc.obs.columns:
        result_df["batch"] = adata_sc.obs[batch_key].values

    rows = []
    for ct_name in sorted(ct.unique()):
        mask = result_df["cell_type"] == ct_name
        sub = result_df[mask]
        coefs = sub["coef"].dropna()

        row = {
            "cell_type": ct_name,
            "n_cells": mask.sum(),
            "mean_coef": coefs.mean(),
            "median_coef": coefs.median(),
            "pct_significant": (sub["pvalue"].dropna() < 0.05).mean() * 100,
            "direction": "risk" if coefs.mean() > 0 else "protective",
        }

        # Wilcoxon test: coefs vs 0
        if len(coefs) >= 5:
            try:
                _, wp = wilcoxon(coefs)
                row["wilcoxon_p"] = wp
            except Exception:
                row["wilcoxon_p"] = np.nan
        else:
            row["wilcoxon_p"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        valid = ~df["wilcoxon_p"].isna()
        if valid.sum() > 1:
            _, fdr, _, _ = multipletests(df.loc[valid, "wilcoxon_p"], method="fdr_bh")
            df.loc[valid, "wilcoxon_fdr"] = fdr
    return df
