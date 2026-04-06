"""
稳定特征选择 (Stable Feature Selection with FDR Control)

核心算法源自 Stabl (Hedou 2024 Nat Biotech)
重实现原因: 原包依赖复杂(knockpy等)，我们只需核心的 noise injection + FDR 控制

算法:
    1. 在真实特征旁注入等量随机噪声特征（打乱行顺序的真实特征列）
    2. 对 lambda_grid 中的每个正则化参数，做 n_bootstraps 次 subsampling + LASSO
    3. 记录每个特征（真实+噪声）被选中的频率
    4. 噪声特征的最大选择频率 → FDP 上界 → 自动确定阈值
    5. 真实特征中选择频率超过阈值的 = 稳定特征

vs 普通 LASSO:
    - 普通 LASSO 选的特征不稳定（换个 random seed 结果就变）
    - Stabl 通过 noise injection 给出 FDR 保证，减少 cohort-specific noise

依赖: numpy, sklearn (全部已在 bioinfo env)
"""

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from joblib import Parallel, delayed
from typing import Optional, Union, List
import warnings


def _inject_noise(X, proportion=1.0, rng=None):
    """注入随机噪声特征：从 X 中随机选列，打乱行顺序。"""
    if rng is None:
        rng = np.random.RandomState(42)
    n_samples, n_features = X.shape
    n_noise = int(n_features * proportion)
    # 随机选列，对每列打乱行顺序
    col_idx = rng.choice(n_features, size=n_noise, replace=n_noise > n_features)
    noise = np.empty((n_samples, n_noise))
    for i, ci in enumerate(col_idx):
        noise[:, i] = rng.permutation(X[:, ci])
    return noise


def _fit_one_bootstrap(X_aug, y, subsample_idx, estimator_cls, estimator_params, threshold=1e-5):
    """单次 bootstrap: 子采样 → 拟合 → 返回被选中的特征 mask。"""
    X_sub = X_aug[subsample_idx]
    y_sub = y[subsample_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimator_cls(**estimator_params)
        est.fit(X_sub, y_sub)
    coef = est.coef_.ravel() if hasattr(est, "coef_") else np.zeros(X_aug.shape[1])
    return np.abs(coef) > threshold


def stable_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[np.ndarray] = None,
    task: str = "regression",
    n_bootstraps: int = 500,
    sample_fraction: float = 0.5,
    noise_proportion: float = 1.0,
    lambda_grid: Optional[list] = None,
    n_jobs: int = -1,
    seed: int = 42,
    hard_threshold: Optional[float] = None,
) -> dict:
    """稳定特征选择 with noise-injection FDR control。

    Parameters
    ----------
    X : (n_samples, n_features) 特征矩阵
    y : (n_samples,) 目标变量 (连续 for regression, 0/1 for classification)
    feature_names : (n_features,) 特征名
    task : 'regression' or 'classification'
    n_bootstraps : 每个 lambda 的 bootstrap 次数
    sample_fraction : 每次子采样的比例
    noise_proportion : 噪声特征数 / 真实特征数
    lambda_grid : 正则化参数列表。None 则自动生成
    n_jobs : 并行核数
    seed : 随机种子
    hard_threshold : 若指定，绕过 FDR 直接用此阈值

    Returns
    -------
    dict:
        selected_features : bool mask (n_features,)
        selected_names : 选中的特征名列表 (if feature_names provided)
        stability_scores : (n_features,) 每个特征的最大选择频率
        fdr_threshold : 自动确定的阈值
        fdp_curve : (thresholds, fdp_values)
        n_selected : 选中的特征数
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    n_subsample = int(n_samples * sample_fraction)

    # 1. 注入噪声
    noise = _inject_noise(X, proportion=noise_proportion, rng=rng)
    n_noise = noise.shape[1]
    X_aug = np.hstack([X, noise])  # (n_samples, n_features + n_noise)

    # 2. 设置估计器和 lambda grid
    if task == "regression":
        est_cls = Lasso
        if lambda_grid is None:
            # 自动: alpha from alpha_max/30 to alpha_max
            alpha_max = np.abs(X.T @ y).max() / n_samples
            lambda_grid = np.logspace(np.log10(alpha_max / 30), np.log10(alpha_max), 10)
        param_key = "alpha"
    elif task == "classification":
        est_cls = LogisticRegression
        if lambda_grid is None:
            lambda_grid = np.linspace(0.01, 1.0, 10)
        param_key = "C"
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

    # 3. Bootstrap 循环
    n_total = n_features + n_noise
    all_scores = np.zeros((n_total, len(lambda_grid)))

    for li, lam in enumerate(lambda_grid):
        if task == "regression":
            params = {"alpha": lam, "max_iter": 5000}
        else:
            params = {"C": lam, "penalty": "l1", "solver": "liblinear",
                      "class_weight": "balanced", "max_iter": 5000}

        # 生成所有 bootstrap 的子采样索引
        bootstrap_indices = [rng.choice(n_samples, size=n_subsample, replace=False)
                            for _ in range(n_bootstraps)]

        selections = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_one_bootstrap)(X_aug, y, idx, est_cls, params)
            for idx in bootstrap_indices
        )

        # 选择频率
        sel_matrix = np.stack(selections)  # (n_bootstraps, n_total)
        all_scores[:, li] = sel_matrix.mean(axis=0)

    # 4. 每个特征在所有 lambda 下的最大选择频率
    max_scores_real = all_scores[:n_features].max(axis=1)
    max_scores_noise = all_scores[n_features:].max(axis=1)

    # 5. FDP 计算
    thresholds = np.arange(0.01, 1.0, 0.01)
    fdp_values = np.ones(len(thresholds))
    for ti, tau in enumerate(thresholds):
        n_noise_above = (max_scores_noise > tau).sum() / noise_proportion + 1
        n_real_above = max((max_scores_real > tau).sum(), 1)
        fdp_values[ti] = n_noise_above / n_real_above

    # 6. 最优阈值
    if hard_threshold is not None:
        fdr_threshold = hard_threshold
    else:
        best_idx = np.argmin(fdp_values)
        fdr_threshold = thresholds[best_idx]

    selected = max_scores_real > fdr_threshold

    # 兜底: 如果零特征被选中，取 top 5
    if selected.sum() == 0:
        top5_cutoff = np.sort(max_scores_real)[-min(5, len(max_scores_real))] - 0.01
        selected = max_scores_real > top5_cutoff

    result = {
        "selected_features": selected,
        "stability_scores": max_scores_real,
        "fdr_threshold": fdr_threshold,
        "fdp_curve": (thresholds, fdp_values),
        "n_selected": selected.sum(),
        "noise_max_scores": max_scores_noise,
    }
    if feature_names is not None:
        result["selected_names"] = list(np.asarray(feature_names)[selected])

    print(f"  Stabl: {selected.sum()}/{n_features} features selected "
          f"(FDR threshold={fdr_threshold:.2f}, "
          f"noise ceiling={max_scores_noise.max():.3f})")

    return result
