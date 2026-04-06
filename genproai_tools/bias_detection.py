"""
生物标志物预测偏差检测 (Biomarker Prediction Bias Detection)

核心算法源自 HistBiases (Dawood 2026 Nat Biomed Eng)
重实现原因: 原版与 TIAToolbox/特定 DL 模型绑定，我们只需统计检验框架

算法:
    1. Co-dependence test: Fisher exact + log2 OR 检测 biomarker 间共依赖
    2. Stratified permutation test: 检测混杂因素 (grade/TMB) 对预测性能的偏差影响
       - 在 null 假设下，混杂因素与预测能力无关
       - 随机置换混杂因素标签 → 各子组 AUROC 应趋近整体 AUROC
       - 观测 AUROC 显著偏离 null → 存在 bias

适用: 任何模型预测 + biomarker 标签 + 混杂因素的 bias 审计 (不限于 WSI)

依赖: numpy, pandas, scipy, sklearn (全部已在 bioinfo env)
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from typing import Optional, List, Dict


def codependence_test(
    labels: pd.DataFrame,
    pairs: Optional[List[tuple]] = None,
    pseudocount: float = 1e-2,
) -> pd.DataFrame:
    """Biomarker 共依赖检验 (Fisher exact + log2 Odds Ratio)。

    Parameters
    ----------
    labels : pd.DataFrame, 每列是一个二值 biomarker (0/1)
    pairs : 待检验的 biomarker 对列表。None 则检验所有对
    pseudocount : LOR 计算的伪计数

    Returns
    -------
    pd.DataFrame with columns: marker1, marker2, pvalue, log2_OR, fdr
    """
    if pairs is None:
        cols = labels.columns.tolist()
        pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    results = []
    for m1, m2 in pairs:
        ct = pd.crosstab(labels[m1], labels[m2])
        if ct.shape != (2, 2):
            continue
        a, b, c, d = ct.iloc[0, 0], ct.iloc[0, 1], ct.iloc[1, 0], ct.iloc[1, 1]

        _, pval = fisher_exact([[a, b], [c, d]])
        lor = np.log2(((a + pseudocount) * (d + pseudocount)) /
                      ((b + pseudocount) * (c + pseudocount)))
        results.append({"marker1": m1, "marker2": m2, "pvalue": pval, "log2_OR": lor})

    df = pd.DataFrame(results)
    if len(df) > 0:
        _, fdr, _, _ = multipletests(df["pvalue"], method="fdr_bh")
        df["fdr"] = fdr
    return df


def codependence_matrix(labels: pd.DataFrame, **kwargs) -> tuple:
    """构建完整的 LOR 共依赖矩阵。

    Returns
    -------
    (lor_matrix, pval_matrix) : 对称的 pd.DataFrame
    """
    res = codependence_test(labels, **kwargs)
    markers = labels.columns.tolist()
    lor_mat = pd.DataFrame(0.0, index=markers, columns=markers)
    pval_mat = pd.DataFrame(1.0, index=markers, columns=markers)

    for _, row in res.iterrows():
        lor_mat.loc[row["marker1"], row["marker2"]] = row["log2_OR"]
        lor_mat.loc[row["marker2"], row["marker1"]] = row["log2_OR"]
        pval_mat.loc[row["marker1"], row["marker2"]] = row["fdr"]
        pval_mat.loc[row["marker2"], row["marker1"]] = row["fdr"]

    return lor_mat, pval_mat


def stratified_permutation_test(
    score: np.ndarray,
    label: np.ndarray,
    confounder: np.ndarray,
    n_perm: int = 10000,
    n_jobs: int = -1,
    seed: int = 42,
) -> dict:
    """分层置换检验: 检测混杂因素对预测性能的偏差影响。

    Parameters
    ----------
    score : (n_samples,) 模型预测概率 [0, 1]
    label : (n_samples,) 真实标签 (0/1)
    confounder : (n_samples,) 混杂因素分层变量 (e.g., grade: 0/1/2)
    n_perm : 置换次数
    n_jobs : 并行核数
    seed : 随机种子

    Returns
    -------
    dict:
        overall_auroc : 全样本 AUROC
        stratum_aurocs : {stratum: AUROC} 各子组 AUROC
        stratum_pvalues : {stratum: p-value} 各子组双侧置换 p 值
        stratum_fdr : {stratum: FDR-BH p-value}
        null_distributions : {stratum: (n_perm,) array} null AUROCs
        bias_detected : bool, 是否检测到显著偏差 (any FDR < 0.05)
    """
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    confounder = np.asarray(confounder)

    strata = np.unique(confounder)
    rng = np.random.RandomState(seed)

    # 全样本 AUROC
    overall_auroc = roc_auc_score(label, score) if len(np.unique(label)) > 1 else 0.5

    # 各子组观测 AUROC
    obs_aurocs = {}
    for s in strata:
        mask = confounder == s
        if len(np.unique(label[mask])) > 1 and mask.sum() >= 5:
            obs_aurocs[s] = roc_auc_score(label[mask], score[mask])
        else:
            obs_aurocs[s] = np.nan

    # 置换: 打乱 confounder 标签
    def _one_perm(_):
        perm_conf = rng.permutation(confounder)
        null_aurocs = {}
        for s in strata:
            mask = perm_conf == s
            if len(np.unique(label[mask])) > 1 and mask.sum() >= 5:
                null_aurocs[s] = roc_auc_score(label[mask], score[mask])
            else:
                null_aurocs[s] = np.nan
        return null_aurocs

    perm_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_perm)(i) for i in range(n_perm)
    )

    # 汇总 null distribution 和 p-value
    null_dists = {s: np.array([r.get(s, np.nan) for r in perm_results]) for s in strata}
    raw_pvals = {}
    for s in strata:
        if np.isnan(obs_aurocs.get(s, np.nan)):
            raw_pvals[s] = np.nan
            continue
        null = null_dists[s][~np.isnan(null_dists[s])]
        if len(null) == 0:
            raw_pvals[s] = np.nan
            continue
        obs = obs_aurocs[s]
        p_upper = np.mean(obs <= null)
        p_lower = np.mean(obs >= null)
        raw_pvals[s] = 2 * min(p_upper, p_lower)  # 双侧

    # FDR 校正
    valid_strata = [s for s in strata if not np.isnan(raw_pvals.get(s, np.nan))]
    fdr_pvals = {}
    if len(valid_strata) > 1:
        pvals_arr = np.array([raw_pvals[s] for s in valid_strata])
        _, fdr_arr, _, _ = multipletests(pvals_arr, method="fdr_bh")
        for s, fdr in zip(valid_strata, fdr_arr):
            fdr_pvals[s] = fdr
    else:
        for s in valid_strata:
            fdr_pvals[s] = raw_pvals[s]

    bias_detected = any(v < 0.05 for v in fdr_pvals.values() if not np.isnan(v))

    return {
        "overall_auroc": overall_auroc,
        "stratum_aurocs": obs_aurocs,
        "stratum_pvalues": raw_pvals,
        "stratum_fdr": fdr_pvals,
        "null_distributions": null_dists,
        "bias_detected": bias_detected,
    }


def audit_biomarker_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    confounders: pd.DataFrame,
    n_perm: int = 10000,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """批量审计多个 biomarker 预测的偏差。

    Parameters
    ----------
    predictions : pd.DataFrame, 每列是一个 biomarker 的预测概率
    labels : pd.DataFrame, 每列是对应 biomarker 的真实标签 (0/1)
    confounders : pd.DataFrame, 每列是一个混杂因素

    Returns
    -------
    pd.DataFrame, 每行一个 (biomarker, confounder) 组合
    """
    results = []
    for bio in predictions.columns:
        if bio not in labels.columns:
            continue
        score = predictions[bio].values
        label = labels[bio].values
        valid = ~(np.isnan(score) | np.isnan(label))

        for conf in confounders.columns:
            conf_vals = confounders[conf].values
            valid_mask = valid & ~pd.isna(confounders[conf])
            if valid_mask.sum() < 20:
                continue

            res = stratified_permutation_test(
                score[valid_mask], label[valid_mask].astype(int),
                conf_vals[valid_mask], n_perm=n_perm, n_jobs=n_jobs
            )

            for stratum, auroc in res["stratum_aurocs"].items():
                results.append({
                    "biomarker": bio,
                    "confounder": conf,
                    "stratum": stratum,
                    "overall_auroc": res["overall_auroc"],
                    "stratum_auroc": auroc,
                    "auroc_diff": auroc - res["overall_auroc"] if not np.isnan(auroc) else np.nan,
                    "pvalue": res["stratum_pvalues"].get(stratum, np.nan),
                    "fdr": res["stratum_fdr"].get(stratum, np.nan),
                    "bias_detected": res["bias_detected"],
                })

    return pd.DataFrame(results)
