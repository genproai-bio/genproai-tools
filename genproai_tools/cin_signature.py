"""
染色体不稳定性 (CIN) Signature 量化

核心算法源自 CINSignatureQuantification (Drews 2022 Nature; Thompson 2025 Nat Genet)
重实现原因: 原版 R 包依赖 limSolve + Biobase，Python 版可直接在 bioinfo env 运行

算法:
    1. 输入 copy number segment table (chr, start, end, segVal, sample)
    2. 平滑合并 → 提取 5 种 CN 特征 → 43 个 mixture model component
    3. Posterior probability → sample-by-component matrix
    4. NNLS 分解为 17 个 CIN signatures (CX1-CX17)

CIN Signatures 临床意义:
    - CX1-CX3: 全基因组倍体化相关
    - CX5, CX8: 铂类化疗耐药 (Thompson 2025)
    - CX13, CX14: HRD 相关

依赖: numpy, pandas, scipy (全部已在 bioinfo env)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from scipy.optimize import nnls
from typing import Optional, Tuple
import warnings

# ============================================================
# Reference Data: Mixture Model Parameters (Drews 2022 TCGA)
# ============================================================
MIXTURE_MODELS = {
    "segsize": {
        "Mean": [38420.77, 86319.11, 171234.26, 427454.0, 918939.41, 1494667.9,
                 2496039.63, 3898217.97, 4908317.82, 6371179.0, 8780945.0, 13435586.0,
                 18665397.0, 22913723.21, 30753807.02, 36812267.4, 44866787.95,
                 57831002.6, 68086495.85, 81015695.88, 93268268.02, 154120036.5],
        "SD": [15438.88, 30243.07, 55096.05, 174237.0, 260126.65, 404499.0,
               608098.85, 583841.6, 398896.48, 985163.0, 1086234.0, 1439442.0,
               1664701.0, 1028090.2, 2370747.45, 1799660.32, 4250506.18,
               3574456.9, 6023604.95, 3149494.22, 10492824.57, 27901372.67],
        "type": "gaussian", "n_components": 22,
    },
    "changepoint": {
        "Mean": [0.1059458, 0.342835, 1.0081859, 1.4075413, 2.2026688,
                 3.2015574, 4.114426, 5.1385089, 6.9335776, 9.9634177],
        "SD": [0.06312023, 0.1572276, 0.2528096, 0.3988254, 0.4656683,
               0.32865165, 0.49429328, 1.16026304, 1.76527488, 2.61954362],
        "type": "gaussian", "n_components": 10,
    },
    "bp10MB": {
        "Mean": [3.897146e-08, 1.049718, 5.439898],
        "type": "poisson", "n_components": 3,
    },
    "bpchrarm": {
        "Mean": [0.05683922, 1.9260777, 7.06398355, 18.15045958, 46.66514448],
        "type": "poisson", "n_components": 5,
    },
    "osCN": {
        "Mean": [0.2454004, 2.4253259, 9.3645396],
        "type": "poisson", "n_components": 3,
    },
}

# 17 CIN Signature weights matrix (43 components × 17 signatures)
# Loaded from reference data on first use
_SIGNATURE_MATRIX = None
_THRESHOLDS = None
_SCALING = None


def _get_reference_data():
    """Load or return cached signature matrix and parameters."""
    global _SIGNATURE_MATRIX, _THRESHOLDS, _SCALING
    if _SIGNATURE_MATRIX is not None:
        return _SIGNATURE_MATRIX, _THRESHOLDS, _SCALING

    import os, pickle
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ref_path = os.path.join(data_dir, "cin_signature_reference.pkl")

    if os.path.exists(ref_path):
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)
        _SIGNATURE_MATRIX = ref["W"]
        _THRESHOLDS = ref.get("thresholds")
        _SCALING = ref.get("scaling")
    else:
        # Fallback: generate identity matrix (user must provide reference)
        warnings.warn("CIN signature reference data not found. Run extract_cin_reference() first.")
        _SIGNATURE_MATRIX = np.eye(43, 17)
        _THRESHOLDS = np.zeros(17)
        _SCALING = {"mean": np.zeros(17), "scale": np.ones(17)}

    return _SIGNATURE_MATRIX, _THRESHOLDS, _SCALING


# ============================================================
# Feature Extraction
# ============================================================
def _smooth_segments(segs: pd.DataFrame, wiggle: float = 0.1) -> pd.DataFrame:
    """Smooth near-diploid segments and merge adjacent identical segments."""
    segs = segs.copy()
    # Near-diploid → exactly diploid
    near_diploid = (segs["segVal"] >= 2 - wiggle) & (segs["segVal"] <= 2 + wiggle)
    segs.loc[near_diploid, "segVal"] = 2.0
    # Negative → 0
    segs.loc[segs["segVal"] < 0, "segVal"] = 0.0

    # Merge adjacent segments with same segVal on same chromosome
    merged = []
    for (sample, chrom), group in segs.groupby(["sample", "chromosome"]):
        group = group.sort_values("start")
        rows = group.values.tolist()
        result = [rows[0]]
        for row in rows[1:]:
            if abs(row[3] - result[-1][3]) < 1e-10:  # same segVal
                result[-1][2] = row[2]  # extend end
            else:
                result.append(row)
        merged.extend(result)

    return pd.DataFrame(merged, columns=segs.columns)


def _extract_segsize(segs: pd.DataFrame) -> list:
    """Segment size feature: length of non-diploid segments."""
    non_diploid = segs[segs["segVal"] != 2.0]
    return [(s, e - st) for s, st, e in zip(non_diploid["sample"], non_diploid["start"], non_diploid["end"])]


def _extract_changepoint(segs: pd.DataFrame) -> list:
    """Changepoint feature: absolute difference between adjacent segments."""
    results = []
    for (sample, chrom), group in segs.groupby(["sample", "chromosome"]):
        group = group.sort_values("start")
        vals = group["segVal"].values
        # If first segment is not diploid, count distance from diploid
        if len(vals) > 0 and abs(vals[0] - 2) > 0.1:
            results.append((sample, abs(2 - vals[0])))
        # Adjacent differences (excluding diploid segments)
        non_dip_idx = np.where(vals != 2.0)[0]
        for i in range(1, len(non_dip_idx)):
            results.append((sample, abs(vals[non_dip_idx[i]] - vals[non_dip_idx[i-1]])))
    return results


def _extract_bp10mb(segs: pd.DataFrame) -> list:
    """Breakpoints per 10MB window."""
    results = []
    for (sample, chrom), group in segs.groupby(["sample", "chromosome"]):
        group = group.sort_values("start")
        bp_positions = group["end"].values[:-1]  # breakpoints = segment ends (except last)
        if len(bp_positions) == 0:
            continue
        max_pos = group["end"].max()
        n_windows = max(1, int(np.ceil(max_pos / 1e7)))
        counts, _ = np.histogram(bp_positions, bins=n_windows, range=(0, n_windows * 1e7))
        for c in counts:
            results.append((sample, int(c)))
    return results


def _extract_oscn(segs: pd.DataFrame) -> list:
    """Copy number oscillation: A-B-A patterns."""
    results = []
    for (sample, chrom), group in segs.groupby(["sample", "chromosome"]):
        vals = np.round(group.sort_values("start")["segVal"].values).astype(int)
        if len(vals) < 4:
            continue
        osc_count = 0
        for j in range(2, len(vals)):
            if vals[j] == vals[j-2] and vals[j] != vals[j-1]:
                osc_count += 1
        results.append((sample, osc_count))
    return results


def _extract_bpchrarm(segs: pd.DataFrame) -> list:
    """Breakpoints relative to centromere (p-arm vs q-arm count per chromosome)."""
    # Simplified: count total breakpoints per chromosome (without arm distinction)
    results = []
    for (sample, chrom), group in segs.groupby(["sample", "chromosome"]):
        n_bp = max(0, len(group) - 1)
        results.append((sample, n_bp))
    return results


def extract_features(segs: pd.DataFrame) -> dict:
    """Extract all 5 CN features from segment table.

    Parameters
    ----------
    segs : pd.DataFrame with columns: chromosome, start, end, segVal, sample

    Returns
    -------
    dict: {feature_name: [(sample, value), ...]}
    """
    segs = segs.copy()
    segs.columns = ["chromosome", "start", "end", "segVal", "sample"]
    segs["chromosome"] = segs["chromosome"].astype(str).str.replace("chr", "")
    segs = segs[~segs["chromosome"].isin(["Y", "M", "MT"])]

    # Smooth
    segs = _smooth_segments(segs)

    # Check quiet samples
    sample_non_diploid = segs[segs["segVal"] != 2.0].groupby("sample").size()
    quiet = sample_non_diploid[sample_non_diploid < 20].index.tolist()
    if quiet:
        warnings.warn(f"Removed {len(quiet)} quiet samples (< 20 non-diploid segments)")
        segs = segs[~segs["sample"].isin(quiet)]

    features = {
        "segsize": _extract_segsize(segs),
        "changepoint": _extract_changepoint(segs),
        "bp10MB": _extract_bp10mb(segs),
        "bpchrarm": _extract_bpchrarm(segs),
        "osCN": _extract_oscn(segs),
    }
    return features


# ============================================================
# Sample-by-Component Matrix
# ============================================================
def _compute_posterior(values: np.ndarray, model: dict) -> np.ndarray:
    """Compute mixture model posterior probabilities (uninformative prior)."""
    means = np.array(model["Mean"])
    n_comp = len(means)

    if model["type"] == "gaussian":
        sds = np.array(model["SD"])
        densities = np.column_stack([
            norm.pdf(values, loc=means[k], scale=sds[k]) for k in range(n_comp)
        ])
    else:  # poisson
        densities = np.column_stack([
            poisson.pmf(values.astype(int), mu=max(means[k], 1e-10)) for k in range(n_comp)
        ])

    # Normalize rows to probabilities
    row_sums = densities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return densities / row_sums


def compute_sample_by_component(features: dict, samples: list) -> pd.DataFrame:
    """Convert raw features to sample-by-component matrix via mixture model posteriors.

    Returns
    -------
    pd.DataFrame: (n_samples, 43) — columns named segsize1..22, changepoint1..10, etc.
    """
    all_components = []
    col_names = []

    for feat_name in ["segsize", "changepoint", "bp10MB", "bpchrarm", "osCN"]:
        model = MIXTURE_MODELS[feat_name]
        n_comp = model["n_components"]
        obs = features[feat_name]

        if not obs:
            # No observations → zeros
            mat = pd.DataFrame(0.0, index=samples, columns=[f"{feat_name}{i+1}" for i in range(n_comp)])
        else:
            obs_df = pd.DataFrame(obs, columns=["sample", "value"])
            # Posterior probabilities
            posteriors = _compute_posterior(obs_df["value"].values, model)
            # Aggregate by sample (sum)
            post_df = pd.DataFrame(posteriors, columns=[f"{feat_name}{i+1}" for i in range(n_comp)])
            post_df["sample"] = obs_df["sample"].values
            mat = post_df.groupby("sample").sum()
            # Ensure all samples present
            mat = mat.reindex(samples, fill_value=0.0)

        all_components.append(mat)
        col_names.extend([f"{feat_name}{i+1}" for i in range(n_comp)])

    result = pd.concat(all_components, axis=1)
    result = result[col_names]  # ensure column order
    return result


# ============================================================
# NNLS Signature Decomposition
# ============================================================
def quantify_signatures(
    segs: pd.DataFrame,
    output_type: str = "threshold",
) -> pd.DataFrame:
    """End-to-end CIN signature quantification.

    Parameters
    ----------
    segs : pd.DataFrame with columns: chromosome, start, end, segVal, sample
    output_type : 'raw', 'normalized', 'threshold', or 'scaled'

    Returns
    -------
    pd.DataFrame: (n_samples, 17) — CX1 to CX17 signature activities
    """
    W, thresholds, scaling = _get_reference_data()

    # 1. Extract features
    print("  Extracting CN features...")
    features = extract_features(segs)
    samples = sorted(segs["sample"].unique())
    n_valid = len(set(s for feat in features.values() for s, _ in feat))
    print(f"  {n_valid} samples with features")

    # 2. Sample-by-component matrix
    print("  Computing sample-by-component matrix (43 components)...")
    sxc = compute_sample_by_component(features, samples)

    # 3. NNLS decomposition
    print("  NNLS decomposition into 17 CIN signatures...")
    sig_names = [f"CX{i+1}" for i in range(W.shape[1])]
    activities = np.zeros((len(samples), W.shape[1]))

    for i in range(len(samples)):
        v = sxc.iloc[i].values
        h, _ = nnls(W, v)
        activities[i] = h

    result = pd.DataFrame(activities, index=samples, columns=sig_names)

    # 4. Post-processing
    if output_type == "raw":
        return result

    # Normalize: row sums to 1
    row_sums = result.sum(axis=1)
    row_sums[row_sums == 0] = 1
    normalized = result.div(row_sums, axis=0)

    if output_type == "normalized":
        return normalized

    # Threshold: set below-threshold values to 0
    if thresholds is not None:
        for j, sig in enumerate(sig_names):
            normalized.loc[normalized[sig] < thresholds[j], sig] = 0

    if output_type == "threshold":
        return normalized

    # Scaled: z-score using TCGA reference
    if scaling is not None:
        scaled = (normalized - scaling["mean"]) / scaling["scale"]
        return scaled

    return normalized
