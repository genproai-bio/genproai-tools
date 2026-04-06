"""
代谢通量推断 (Metabolic Flux Inference)

核心算法源自 METAFlux (Huang 2023 Nat Commun; KChen-lab, MD Anderson)
重实现原因: 原包是 R，且依赖 Seurat V4 不兼容 V5；Python 版可直接在 bioinfo env 运行

算法:
    1. Gene-Protein-Reaction (GPR) 规则: 基因表达 → 13082 个反应活性 (MRAS)
       - iso (OR): sum(expr / gene_participation_count)
       - simple_comp (AND): min(expr / gene_participation_count)
       - multi_comp (AND+OR): 递归组合
    2. QP 优化: MRAS 作为通量边界，maximize biomass，subject to Sv=0
       - 使用 osqp 求解器 (Python 原生支持)

依赖: numpy, scipy, pandas, osqp
参考数据: human_gem_metaflux.pkl (从 R sysdata.rda 提取)
"""

import numpy as np
import pandas as pd
from scipy.sparse import eye as speye, vstack as spvstack
from typing import Optional, Dict, List
import pickle
import os
import re

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_ref():
    """加载 Human-GEM 参考数据。"""
    path = os.path.join(_DATA_DIR, "human_gem_metaflux.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_media():
    """加载培养基/血液营养物配置。"""
    path = os.path.join(_DATA_DIR, "metaflux_media.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _calc_iso_score(reaction_genes, expr_df, gene_num):
    """Isoenzyme (OR): sum(expr / participation_count)"""
    available = [g for g in reaction_genes if g in expr_df.index]
    if not available:
        return None
    weights = np.array([gene_num.get(g, 1) for g in available], dtype=float)
    vals = expr_df.loc[available].values  # (n_genes, n_samples)
    return (vals / weights[:, None]).sum(axis=0)


def _calc_complex_score(reaction_genes, expr_df, gene_num):
    """Simple complex (AND): min(expr / participation_count)"""
    available = [g for g in reaction_genes if g in expr_df.index]
    if not available:
        return None
    weights = np.array([gene_num.get(g, 1) for g in available], dtype=float)
    vals = expr_df.loc[available].values
    return (vals / weights[:, None]).min(axis=0)


def _parse_multi_comp(formula, expr_df, gene_num):
    """Multi-complex (AND+OR mixed): 解析括号公式递归计算。

    Examples:
        "DLD and PDHX and (PDHA1 or PDHA2) and DLAT"
        "(PDHA1 and PDHB) or (PDHA2 and PDHB)"
    """
    # 提取括号内的子表达式
    bracket_contents = re.findall(r"\(([^()]+)\)", formula)
    if not bracket_contents:
        # 无括号，纯 and 或纯 or
        if " and " in formula:
            genes = [g.strip() for g in formula.split(" and ")]
            return _calc_complex_score(genes, expr_df, gene_num)
        elif " or " in formula:
            genes = [g.strip() for g in formula.split(" or ")]
            return _calc_iso_score(genes, expr_df, gene_num)
        else:
            # 单基因
            g = formula.strip()
            if g in expr_df.index:
                w = gene_num.get(g, 1)
                return expr_df.loc[g].values / w
            return None

    # 判断括号内是 or 还是 and
    first_bracket = bracket_contents[0]
    bracket_is_or = " or " in first_bracket

    sub_scores = []
    for bc in bracket_contents:
        genes = [g.strip() for g in re.split(r"\s+(?:and|or)\s+", bc)]
        if bracket_is_or:
            s = _calc_iso_score(genes, expr_df, gene_num)
        else:
            s = _calc_complex_score(genes, expr_df, gene_num)
        if s is not None:
            sub_scores.append(s)

    # 提取括号外的独立基因
    outer = re.sub(r"\([^()]+\)", "", formula)
    if bracket_is_or:
        # 括号内 OR → 外层是 AND
        outer_genes = [g.strip() for g in re.split(r"\s+and\s+", outer) if g.strip()]
    else:
        # 括号内 AND → 外层是 OR
        outer_genes = [g.strip() for g in re.split(r"\s+or\s+", outer) if g.strip()]

    for g in outer_genes:
        if g in expr_df.index:
            w = gene_num.get(g, 1)
            sub_scores.append(expr_df.loc[g].values / w)

    if not sub_scores:
        return None

    stacked = np.stack(sub_scores)
    if bracket_is_or:
        # 括号内 OR, 外层 AND → 取 min
        return stacked.min(axis=0)
    else:
        # 括号内 AND, 外层 OR → 取 sum
        return stacked.sum(axis=0)


def calculate_reaction_score(expr_df: pd.DataFrame) -> pd.DataFrame:
    """计算代谢反应活性评分 (MRAS)。

    Parameters
    ----------
    expr_df : pd.DataFrame, 基因表达矩阵 (gene_symbol × samples)
              值应为 log-transformed 且非负

    Returns
    -------
    pd.DataFrame, (13082 reactions × n_samples)
    """
    if (expr_df < 0).any().any():
        raise ValueError("Expression data must be non-negative (log-transformed)")

    ref = _load_ref()
    gene_num = ref["gene_num"]
    reactions = ref["Reaction"]

    found = sum(g in expr_df.index for g in gene_num)
    print(f"  Metabolic genes found: {found}/{len(gene_num)} ({found/len(gene_num)*100:.1f}%)")

    # 1. 计算三种 GPR 类型
    n_samples = expr_df.shape[1]
    scores = {}

    for rxn_id, genes in ref["iso"].items():
        s = _calc_iso_score(genes, expr_df, gene_num)
        if s is not None:
            scores[rxn_id] = s

    for rxn_id, genes in ref["simple_comp"].items():
        s = _calc_complex_score(genes, expr_df, gene_num)
        if s is not None:
            scores[rxn_id] = s

    for rxn_id, formula in ref["multi_comp"].items():
        s = _parse_multi_comp(formula, expr_df, gene_num)
        if s is not None:
            scores[rxn_id] = s

    print(f"  Reactions scored: {len(scores)}/{len(ref['iso'])+len(ref['simple_comp'])+len(ref['multi_comp'])}")

    # 2. 归一化: 每个样本除以该样本最大值 → [0, 1]
    score_df = pd.DataFrame(scores, index=expr_df.columns).T
    for col in score_df.columns:
        col_max = score_df[col].max()
        if col_max > 0:
            score_df[col] = score_df[col] / col_max

    # 3. 对齐到 13082 反应，无 GPR 的反应填 1
    result = pd.DataFrame(index=reactions, columns=expr_df.columns, dtype=float)
    result.loc[:, :] = 1.0  # 默认 1
    for rxn in score_df.index:
        if rxn in result.index:
            result.loc[rxn] = score_df.loc[rxn]
    result = result.fillna(1.0)

    # Biomass 相关反应 (LB=0, UB=0) 设为 0
    lb = ref["LB"]
    ub = ref["UB"]
    zero_mask = (lb == 0) & (ub == 0)
    result.iloc[zero_mask] = 0

    return result


def compute_flux(
    mras: pd.DataFrame,
    medium: str = "human_blood",
    verbose: bool = True,
) -> pd.DataFrame:
    """QP 优化计算代谢通量。

    Parameters
    ----------
    mras : pd.DataFrame, (13082 reactions × n_samples), output of calculate_reaction_score()
    medium : 'human_blood' or 'cell_medium'
    verbose : print progress

    Returns
    -------
    pd.DataFrame, (13082 reactions × n_samples), 通量值
    """
    try:
        import osqp
    except ImportError:
        raise ImportError("osqp is required: pip install osqp")

    from scipy.sparse import csc_matrix

    ref = _load_ref()
    media = _load_media()

    S = ref["S"]  # (8378, 13082) sparse
    rev = ref["rev"]
    obj = ref["Obj"]
    reactions = ref["Reaction"]
    pathway = ref["pathway"]

    n_reactions = len(reactions)
    n_metabolites = S.shape[0]

    # 营养物质列表
    medium_reactions = media[medium]["reaction_name"]
    exchange_mask = np.array([p == "Exchange/demand reactions" for p in pathway])

    # QP 矩阵 (固定部分)
    P = speye(n_reactions, format="csc")
    q = np.zeros(n_reactions)
    biomass_idx = np.where(obj == 1)[0]
    q[biomass_idx] = -10000  # maximize biomass

    A = spvstack([S, speye(n_reactions, format="csc")], format="csc")

    results = {}
    for i, sample in enumerate(mras.columns):
        mras_vals = mras[sample].values.astype(float)

        # 下界
        lb = np.zeros(n_reactions)
        lb[rev == 1] = -mras_vals[rev == 1]  # 可逆反应
        lb[exchange_mask] = 0  # exchange 默认禁止摄取
        for rxn in medium_reactions:
            if rxn in reactions:
                lb[reactions.index(rxn)] = -1  # medium 允许摄取

        # 上界
        ub = mras_vals.copy()

        # 约束向量
        l = np.concatenate([np.zeros(n_metabolites), lb])
        u = np.concatenate([np.zeros(n_metabolites), ub])

        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u,
                     max_iter=1000000, eps_abs=1e-4, eps_rel=1e-4,
                     adaptive_rho_interval=50, verbose=False)
        res = solver.solve()

        if res.info.status == "solved" or res.info.status == "solved_inaccurate":
            results[sample] = res.x
        else:
            if verbose:
                print(f"  Warning: sample {sample} solver status: {res.info.status}")
            results[sample] = np.full(n_reactions, np.nan)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(mras.columns)} samples")

    flux_df = pd.DataFrame(results, index=reactions)
    if verbose:
        biomass_flux = flux_df.iloc[biomass_idx[0]]
        print(f"  Biomass flux range: [{biomass_flux.min():.4f}, {biomass_flux.max():.4f}]")

    return flux_df


def metaflux_pipeline(
    expr_df: pd.DataFrame,
    medium: str = "human_blood",
) -> dict:
    """端到端代谢通量分析 pipeline。

    Parameters
    ----------
    expr_df : pd.DataFrame, 基因表达矩阵 (gene_symbol × samples), log-transformed

    Returns
    -------
    dict with keys:
        mras : pd.DataFrame, 反应活性评分
        flux : pd.DataFrame, 代谢通量
        pathway_flux : pd.DataFrame, 通路级汇总通量
    """
    print("[1/3] Calculating reaction scores (MRAS)...")
    mras = calculate_reaction_score(expr_df)

    print("[2/3] Computing metabolic flux (QP optimization)...")
    flux = compute_flux(mras, medium=medium)

    print("[3/3] Aggregating pathway-level flux...")
    ref = _load_ref()
    pathways = ref["pathway"]
    flux_with_pathway = flux.copy()
    flux_with_pathway["pathway"] = pathways
    pathway_flux = flux_with_pathway.groupby("pathway").mean()
    pathway_flux = pathway_flux.loc[pathway_flux.abs().sum(axis=1) > 0]

    print(f"  Active pathways: {len(pathway_flux)}")
    return {"mras": mras, "flux": flux, "pathway_flux": pathway_flux}
