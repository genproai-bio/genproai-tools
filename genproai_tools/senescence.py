"""
衰老细胞检测 (Senescent Cell Detection)

核心算法源自 DeepScence (Qu et al. 2025 Cell Genomics)
重实现原因: 原包要求 Python 3.8 + torch 2.2.2 + tensorflow 2.12，与 bioinfo env 不兼容

算法:
    1. 子集到 CoreScence 39 基因 → normalize → scale
    2. ZINB autoencoder (5303 params) 学习 latent representation
    3. Bottleneck pre-activation score 作为衰老评分
    4. CDKN1A (p21) 锚定方向
    5. GMM 二值化 (SnC vs Normal)

依赖: torch, scanpy, numpy, sklearn (全部已在 bioinfo env)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from typing import Optional, List
import warnings

# ============================================================
# CoreScence Gene Set (occurrence >= 5, from 2966 candidate genes)
# 上调 23 + 下调 16 = 39 consensus senescence markers
# ============================================================
CORESCENCE_GENES_HUMAN = {
    # gene_symbol: (occurrence, direction)  direction: 1=up, -1=down
    "IL6": (7, 1), "IGFBP3": (7, 1), "EGFR": (7, -1),
    "SERPINE1": (6, 1), "IGFBP1": (6, 1), "IGFBP7": (6, 1),
    "FAS": (6, 1), "FGF2": (6, 1), "VEGFA": (6, 1),
    "CDKN1A": (6, 1), "CDKN2A": (6, 1),
    "STAT1": (5, 1), "TNFRSF10C": (5, 1), "CXCL8": (5, 1),
    "IL1A": (5, 1), "CXCL1": (5, 1), "ICAM1": (5, 1),
    "CCL2": (5, 1), "IGFBP2": (5, 1), "IGFBP5": (5, 1),
    "GDF15": (5, 1), "CDKN2B": (5, 1), "IGF1": (5, 1),
    "TGFB1": (5, 1),
    "PARP1": (5, -1), "AXL": (5, -1), "WNT2": (5, -1),
    "HMGB2": (5, -1), "HMGB1": (5, -1), "MDM2": (5, -1),
    "CCNA2": (5, -1), "CDK1": (5, -1), "HELLS": (5, -1),
    "FOXM1": (5, -1), "BUB1B": (5, -1), "LMNB1": (5, -1),
    "BRCA1": (5, -1), "JUN": (5, -1), "MIF": (5, -1),
}

# 人-鼠同源基因映射 (常用的 CoreScence 基因)
HUMAN_TO_MOUSE = {
    "IL6": "Il6", "IGFBP3": "Igfbp3", "EGFR": "Egfr",
    "SERPINE1": "Serpine1", "IGFBP7": "Igfbp7", "FAS": "Fas",
    "FGF2": "Fgf2", "VEGFA": "Vegfa", "CDKN1A": "Cdkn1a",
    "CDKN2A": "Cdkn2a", "STAT1": "Stat1", "CXCL1": "Cxcl1",
    "ICAM1": "Icam1", "CCL2": "Ccl2", "IGFBP2": "Igfbp2",
    "IGFBP5": "Igfbp5", "GDF15": "Gdf15", "CDKN2B": "Cdkn2b",
    "IGF1": "Igf1", "TGFB1": "Tgfb1", "PARP1": "Parp1",
    "AXL": "Axl", "WNT2": "Wnt2", "HMGB2": "Hmgb2",
    "HMGB1": "Hmgb1", "MDM2": "Mdm2", "CCNA2": "Ccna2",
    "CDK1": "Cdk1", "HELLS": "Hells", "FOXM1": "Foxm1",
    "BUB1B": "Bub1b", "LMNB1": "Lmnb1", "BRCA1": "Brca1",
    "JUN": "Jun", "MIF": "Mif", "IGFBP1": "Igfbp1",
    "TNFRSF10C": "Tnfrsf10c", "CXCL8": "Cxcl15",  # mouse ortholog
    "IL1A": "Il1a",
}


# ============================================================
# ZINB Autoencoder
# ============================================================
class ZINBAutoencoder(nn.Module):
    """Zero-Inflated Negative Binomial Autoencoder.

    Architecture: input -> Linear(h) -> ReLU -> Linear(2) -> Tanh
                  -> Linear(h) -> ReLU -> Linear(3*input) for pi/mu/theta
    """

    def __init__(self, n_genes: int, hidden_size: int = 32):
        super().__init__()
        self.n_genes = n_genes

        # Encoder
        self.enc1 = nn.Linear(n_genes, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.enc_act = nn.ReLU()

        # Bottleneck (2 neurons for orthogonality regularization)
        self.bottleneck = nn.Linear(hidden_size, 2)
        self.bn_act = nn.Tanh()

        # Decoder
        self.dec1 = nn.Linear(2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dec_act = nn.ReLU()
        self.dec_out = nn.Linear(hidden_size, n_genes * 3)  # pi, mu, theta

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.enc_act(self.bn1(self.enc1(x)))
        z_pre = self.bottleneck(h)  # pre-activation (for scoring)
        z = self.bn_act(z_pre)
        return z, z_pre

    def decode(self, z, size_factor):
        h = self.dec_act(self.bn2(self.dec1(z)))
        out = self.dec_out(h)
        pi, mu, theta = torch.split(out, self.n_genes, dim=1)
        pi = torch.sigmoid(pi)
        mu = torch.clamp(torch.exp(mu), 1e-5, 1e6) * size_factor
        theta = torch.clamp(torch.exp(theta), 1e-4, 1e4)
        return pi, mu, theta

    def forward(self, x, size_factor):
        z, z_pre = self.encode(x)
        pi, mu, theta = self.decode(z, size_factor)
        return pi, mu, theta, z_pre

    def predict_scores(self, x):
        """返回 bottleneck pre-activation scores (不经 Tanh)。"""
        self.eval()
        with torch.no_grad():
            h = self.enc_act(self.bn1(self.enc1(x)))
            return self.bottleneck(h)


def _zinb_loss(y, pi, mu, theta):
    """ZINB negative log-likelihood."""
    eps = 1e-10
    # NB part
    t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1) - torch.lgamma(y + theta + eps)
    t2 = (theta + y) * torch.log1p(mu / (theta + eps)) + y * (torch.log(theta + eps) - torch.log(mu + eps))
    nb_case = t1 + t2 - torch.log(1 - pi + eps)

    # Zero-inflated part
    zero_nb = torch.pow(theta / (theta + mu + eps), theta)
    zero_case = -torch.log(pi + (1 - pi) * zero_nb + eps)

    loss = torch.where(y < 1e-8, zero_case, nb_case)
    return loss.mean()


def _ortho_penalty(z_pre):
    """正交性惩罚: pearson_cor(s1, s2)^2"""
    s1 = z_pre[:, 0]
    s2 = z_pre[:, 1]
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    cor = (s1_c * s2_c).sum() / (torch.sqrt((s1_c ** 2).sum() * (s2_c ** 2).sum()) + 1e-8)
    return cor ** 2


def _mmd_penalty(z_pre, batch_labels):
    """简化版 MMD: 惩罚不同 batch 在 latent space 中的均值差异。"""
    unique_batches = torch.unique(batch_labels)
    if len(unique_batches) < 2:
        return torch.tensor(0.0, device=z_pre.device)
    means = []
    for b in unique_batches:
        mask = batch_labels == b
        if mask.sum() > 0:
            means.append(z_pre[mask].mean(dim=0))
    mmd = 0.0
    count = 0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            mmd += ((means[i] - means[j]) ** 2).sum()
            count += 1
    return mmd / max(count, 1)


# ============================================================
# Main API
# ============================================================
def score_senescence(
    adata,
    species: str = "human",
    anchor_gene: Optional[str] = None,
    batch_key: Optional[str] = None,
    lambda_ortho: float = 1.0,
    lambda_mmd: float = 1.0,
    hidden_size: int = 32,
    lr: float = 0.005,
    weight_decay: float = 1e-3,
    max_epochs: int = 500,
    patience: int = 35,
    seed: int = 42,
    device: str = "auto",
    key_added: str = "senescence_score",
) -> None:
    """计算每个细胞的衰老评分。

    Parameters
    ----------
    adata : AnnData, 建议已去噪 (DCA 或其他)。.X 可以是 raw counts 或 normalized。
    species : 'human' or 'mouse'
    anchor_gene : 方向锚定基因, 默认 CDKN1A (human) / Cdkn1a (mouse)
    batch_key : adata.obs 中的 batch/celltype 列名, 用于 MMD 正则化
    lambda_ortho, lambda_mmd : 正则化权重
    hidden_size : encoder hidden layer size
    lr, weight_decay, max_epochs, patience : 训练参数
    seed : 随机种子
    device : 'cpu', 'cuda', or 'auto'
    key_added : 存入 adata.obs 的列名

    Modifies
    --------
    adata.obs[key_added] : float, 衰老评分 (高 = 更衰老)
    adata.obs[key_added + '_binary'] : str, 'SnC' or 'Normal' (GMM 二值化)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Gene set subsetting
    if species == "human":
        gene_set = list(CORESCENCE_GENES_HUMAN.keys())
        if anchor_gene is None:
            anchor_gene = "CDKN1A"
    elif species == "mouse":
        gene_set = [HUMAN_TO_MOUSE.get(g, g) for g in CORESCENCE_GENES_HUMAN.keys()]
        if anchor_gene is None:
            anchor_gene = "Cdkn1a"
    else:
        raise ValueError(f"species must be 'human' or 'mouse', got '{species}'")

    available = [g for g in gene_set if g in adata.var_names]
    if len(available) < 10:
        raise ValueError(
            f"Only {len(available)}/{len(gene_set)} CoreScence genes found. "
            f"Check gene names match {species} symbols."
        )

    adata_sub = adata[:, available].copy()

    # 2. Preprocessing: normalize → log1p → scale; keep raw counts
    if adata_sub.X.max() > 50:  # likely raw counts
        raw_X = adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X.copy()
        sc.pp.normalize_total(adata_sub)
        size_factors = adata_sub.obs["n_counts"].values if "n_counts" in adata_sub.obs else np.ones(adata_sub.n_obs)
        median_sf = np.median(size_factors[size_factors > 0])
        size_factors = size_factors / median_sf
        sc.pp.log1p(adata_sub)
        sc.pp.scale(adata_sub)
    else:
        # Already normalized/log-transformed
        raw_X = np.expm1(adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X.copy())
        raw_X = np.clip(raw_X, 0, None)
        size_factors = np.ones(adata_sub.n_obs)
        sc.pp.scale(adata_sub)

    X_scaled = adata_sub.X if isinstance(adata_sub.X, np.ndarray) else adata_sub.X.toarray()
    X_scaled = X_scaled.astype(np.float32)
    raw_X = raw_X.astype(np.float32)
    sf = size_factors.astype(np.float32).reshape(-1, 1)

    n_genes = X_scaled.shape[1]
    print(f"  CoreScence genes found: {len(available)}/{len(gene_set)}")

    # 3. Build model
    model = ZINBAutoencoder(n_genes, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)

    X_tensor = torch.tensor(X_scaled, device=device)
    raw_tensor = torch.tensor(raw_X, device=device)
    sf_tensor = torch.tensor(sf, device=device)

    batch_tensor = None
    if batch_key and batch_key in adata.obs.columns:
        batch_codes = pd.Categorical(adata.obs[batch_key]).codes
        batch_tensor = torch.tensor(batch_codes, device=device, dtype=torch.long)

    # 4. Train
    best_loss = float("inf")
    epochs_no_improve = 0

    model.train()
    for epoch in range(max_epochs):
        pi, mu, theta, z_pre = model(X_tensor, sf_tensor)
        loss = _zinb_loss(raw_tensor, pi, mu, theta)
        loss += lambda_ortho * _ortho_penalty(z_pre)
        if batch_tensor is not None:
            loss += lambda_mmd * _mmd_penalty(z_pre, batch_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach().item())

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    print(f"  Training: {epoch + 1} epochs, best loss = {best_loss:.4f}")

    # 5. Score extraction
    scores = model.predict_scores(X_tensor).cpu().numpy()  # (n_cells, 2)

    # Direction correction using anchor gene
    anchor_idx = available.index(anchor_gene) if anchor_gene in available else None
    gene_expr = X_scaled[:, anchor_idx] if anchor_idx is not None else None

    best_neuron = 0
    best_abs_cor = 0
    for ni in range(2):
        s = scores[:, ni]
        if gene_expr is not None:
            cor, _ = pearsonr(s, gene_expr)
            if cor < 0:
                scores[:, ni] = -s
                s = -s
        # Select neuron with strongest average correlation to all genes
        avg_abs_cor = np.mean([abs(pearsonr(s, X_scaled[:, gi])[0]) for gi in range(n_genes)])
        if avg_abs_cor > best_abs_cor:
            best_abs_cor = avg_abs_cor
            best_neuron = ni

    final_score = scores[:, best_neuron]

    # 6. GMM binarization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = final_score.reshape(-1, 1)
        means_init = [[np.mean(data)], [np.percentile(data, 90)]]
        gmm = GaussianMixture(n_components=2, covariance_type="full",
                              random_state=0, means_init=means_init)
        gmm.fit(data)
        assignments = gmm.predict(data)

        # Find threshold: scan from highest score down
        sorted_idx = np.argsort(final_score)[::-1]
        max_assignment = assignments[sorted_idx[0]]
        threshold = final_score.min()
        for idx in sorted_idx:
            if assignments[idx] != max_assignment:
                threshold = final_score[idx]
                break

    binary = np.where(final_score > threshold, "SnC", "Normal")
    n_snc = (binary == "SnC").sum()
    print(f"  Result: {n_snc}/{len(binary)} cells classified as senescent ({n_snc/len(binary)*100:.1f}%)")

    # 7. Store results
    adata.obs[key_added] = final_score
    adata.obs[key_added + "_binary"] = binary
