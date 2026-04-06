"""
Benchmark v2: Shears vs genproai phenotype_association
Each tool uses its own preprocessing pipeline, but Ridge alpha is matched.
"""

import sys, time, os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/yuezong.bai/projects/genproai_tools")

SCRNA_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/gse132465_scrna/gse132465_crc.h5ad"
BULK_EXPR_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/tcga_coadread/TCGA_COAD_HiSeqV2_xena.gz"
CLINICAL_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/tcga_coadread/TCGA-COAD_clinical_survival.csv"
N_CELLS = 1000
N_BULK = 100
RESULTS_DIR = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("BENCHMARK v2: Shears vs genproai (matched alpha)")
print("=" * 60)

# ── 1. Load data ──
print("\n[1/7] Loading data...")
t0 = time.time()

adata_sc_full = sc.read_h5ad(SCRNA_PATH)
np.random.seed(42)
ct_col = "cell_type" if "cell_type" in adata_sc_full.obs.columns else None
if ct_col:
    idx = []
    cts = adata_sc_full.obs[ct_col].value_counts()
    for ct, n in cts.items():
        ct_idx = np.where(adata_sc_full.obs[ct_col] == ct)[0]
        n_sample = max(1, int(N_CELLS * n / len(adata_sc_full)))
        idx.extend(np.random.choice(ct_idx, min(n_sample, len(ct_idx)), replace=False))
    idx = np.array(sorted(idx))[:N_CELLS]
else:
    idx = np.random.choice(adata_sc_full.shape[0], N_CELLS, replace=False)

adata_sc_raw = adata_sc_full[idx].copy()
del adata_sc_full

bulk_expr = pd.read_csv(BULK_EXPR_PATH, sep="\t", index_col=0)
clin = pd.read_csv(CLINICAL_PATH)
clin = clin[["bcr_patient_barcode", "OS", "OS.time"]].rename(columns={"bcr_patient_barcode": "sample_id"})
clin["OS"] = pd.to_numeric(clin["OS"], errors="coerce")
clin["OS.time"] = pd.to_numeric(clin["OS.time"], errors="coerce")
clin = clin.dropna(subset=["OS", "OS.time"])
clin = clin[clin["OS.time"] > 0]

sample_map = {}
for s in bulk_expr.columns:
    short = "-".join(s.split("-")[:3])
    if short in clin["sample_id"].values:
        sample_map[s] = short

matched_bulk = list(sample_map.keys())
np.random.seed(42)
if len(matched_bulk) > N_BULK:
    matched_bulk = list(np.random.choice(matched_bulk, N_BULK, replace=False))

bulk_sub = bulk_expr[matched_bulk].T
adata_bulk_raw = ad.AnnData(
    X=bulk_sub.values.astype(np.float32),
    obs=pd.DataFrame(index=bulk_sub.index),
    var=pd.DataFrame(index=bulk_sub.columns),
)
for s in adata_bulk_raw.obs_names:
    short = sample_map.get(s, s)
    row = clin[clin["sample_id"] == short]
    if len(row) > 0:
        adata_bulk_raw.obs.loc[s, "OS_time"] = float(row.iloc[0]["OS.time"])
        adata_bulk_raw.obs.loc[s, "OS_event"] = float(row.iloc[0]["OS"])
adata_bulk_raw = adata_bulk_raw[adata_bulk_raw.obs["OS_time"].notna()].copy()
print(f"  {adata_sc_raw.shape[0]} cells × {adata_bulk_raw.shape[0]} bulk, {time.time()-t0:.1f}s")

ALPHA = adata_sc_raw.shape[0]  # matched: n_cells

# ── 2. Original Shears pipeline ──
print(f"\n[2/7] Original Shears (alpha={ALPHA})...")
import shears

adata_sc_o = adata_sc_raw.copy()
adata_bulk_o = adata_bulk_raw.copy()

t1 = time.time()
adata_sc_o, adata_bulk_o = shears.pp.recipe_shears(adata_sc_o, adata_bulk_o)
t_orig_pp = time.time() - t1
print(f"  Preprocess: {adata_sc_o.shape[1]} HVGs, {t_orig_pp:.2f}s")

t1 = time.time()
shears.pp.cell_weights(adata_sc_o, adata_bulk_o, alpha_callback=lambda x: ALPHA)
t_orig_w = time.time() - t1
orig_w = adata_sc_o.obsm["cell_weights"].copy()
if not isinstance(orig_w, pd.DataFrame):
    orig_w = pd.DataFrame(orig_w, index=adata_sc_o.obs_names, columns=adata_bulk_o.obs_names)
print(f"  Cell weights: {orig_w.shape}, {t_orig_w:.2f}s")

# Cox (expect crash)
t1 = time.time()
try:
    orig_cox = shears.tl.shears_cox(
        adata_sc_o, adata_bulk_o,
        duration_col="OS_time", event_col="OS_event",
    )
    t_orig_cox = time.time() - t1
    orig_cox_ok = orig_cox["pvalue"].notna().sum()
    print(f"  Cox: {orig_cox_ok}/{len(orig_cox)} OK, {t_orig_cox:.2f}s")
except Exception as e:
    print(f"  Cox CRASHED: {type(e).__name__}: {str(e)[:80]}")
    orig_cox = None
    t_orig_cox = None

# ── 3. genproai pipeline ──
print(f"\n[3/7] genproai phenotype_association (alpha={ALPHA})...")
from genproai_tools import phenotype_association as pa

adata_sc_g = adata_sc_raw.copy()
adata_bulk_g = adata_bulk_raw.copy()

t1 = time.time()
pa.preprocess(adata_sc_g, adata_bulk_g)
t_gp_pp = time.time() - t1
print(f"  Preprocess: {adata_sc_g.shape[1]} HVGs, {t_gp_pp:.2f}s")

t1 = time.time()
gp_w = pa.compute_cell_weights(adata_sc_g, adata_bulk_g, alpha=ALPHA)
t_gp_w = time.time() - t1
print(f"  Cell weights: {gp_w.shape}, {t_gp_w:.2f}s")

t1 = time.time()
gp_cox = pa.association_cox(
    adata_sc_g, adata_bulk_g,
    duration_col="OS_time", event_col="OS_event",
    penalizer=0.1,
)
t_gp_cox = time.time() - t1
gp_cox_ok = gp_cox["pvalue"].notna().sum()
print(f"  Cox: {gp_cox_ok}/{len(gp_cox)} OK, {t_gp_cox:.2f}s")

# ── 4. Check HVG overlap ──
print("\n[4/7] HVG overlap...")
orig_genes = set(adata_sc_o.var_names)
gp_genes = set(adata_sc_g.var_names)
overlap = orig_genes & gp_genes
print(f"  Original: {len(orig_genes)} genes")
print(f"  genproai: {len(gp_genes)} genes")
print(f"  Overlap:  {len(overlap)} ({100*len(overlap)/max(len(orig_genes),len(gp_genes)):.1f}%)")

# ── 5. Compare cell_weights ──
print("\n[5/7] Cell weights comparison...")

# Both have same cell indices (same raw data), but gene sets may differ
# The cell_weights matrix is cells × bulk_samples
# Compare per-sample across cells
common_cells = orig_w.index.intersection(gp_w.index)
common_bulk = orig_w.columns.intersection(gp_w.columns) if hasattr(orig_w, 'columns') and hasattr(gp_w, 'columns') else None

if common_bulk is not None and len(common_bulk) > 0:
    ow = orig_w.loc[common_cells, common_bulk]
    gw = gp_w.loc[common_cells, common_bulk]
else:
    ow = orig_w.loc[common_cells]
    gw = gp_w.loc[common_cells]

ow_flat = ow.values.flatten()
gw_flat = gw.values.flatten()
mask = np.isfinite(ow_flat) & np.isfinite(gw_flat)

r_all, _ = pearsonr(ow_flat[mask], gw_flat[mask])
rho_all, _ = spearmanr(ow_flat[mask], gw_flat[mask])
rmse = np.sqrt(np.mean((ow_flat[mask] - gw_flat[mask]) ** 2))

# Per-sample r
per_sample_r = []
for i in range(ow.shape[1]):
    o = ow.iloc[:, i].values
    g = gw.iloc[:, i].values
    m = np.isfinite(o) & np.isfinite(g)
    if m.sum() > 10:
        r, _ = pearsonr(o[m], g[m])
        per_sample_r.append(r)

# Per-cell r (across bulk samples)
per_cell_r = []
for i in range(min(ow.shape[0], 200)):  # sample 200 cells
    o = ow.iloc[i, :].values
    g = gw.iloc[i, :].values
    m = np.isfinite(o) & np.isfinite(g)
    if m.sum() > 5:
        r, _ = pearsonr(o[m], g[m])
        per_cell_r.append(r)

print(f"  Global Pearson r:     {r_all:.6f}")
print(f"  Global Spearman rho:  {rho_all:.6f}")
print(f"  RMSE:                 {rmse:.2e}")
print(f"  Per-sample r:  mean={np.mean(per_sample_r):.4f}  min={np.min(per_sample_r):.4f}")
print(f"  Per-cell r:    mean={np.mean(per_cell_r):.4f}  min={np.min(per_cell_r):.4f}")

# ── 6. Cox comparison ──
print("\n[6/7] Cox comparison...")
if orig_cox is not None:
    common = orig_cox.index.intersection(gp_cox.index)
    valid = orig_cox.loc[common, "pvalue"].notna() & gp_cox.loc[common, "pvalue"].notna()
    if valid.sum() > 10:
        cox_coef_r, _ = pearsonr(orig_cox.loc[common][valid]["coef"], gp_cox.loc[common][valid]["coef"])
        cox_p_r, _ = pearsonr(
            -np.log10(orig_cox.loc[common][valid]["pvalue"].clip(1e-300)),
            -np.log10(gp_cox.loc[common][valid]["pvalue"].clip(1e-300))
        )
        rescued = orig_cox.loc[common, "pvalue"].isna().sum()
        print(f"  Coef Pearson r:   {cox_coef_r:.6f}")
        print(f"  -log10(p) r:      {cox_p_r:.6f}")
        print(f"  Cells rescued by bug#5 fix: {rescued}")
    else:
        cox_coef_r = cox_p_r = None
        print(f"  Too few valid pairs ({valid.sum()})")
else:
    cox_coef_r = cox_p_r = None
    print(f"  Original crashed → bug#5 confirmed")
    print(f"  genproai: {gp_cox_ok}/{len(gp_cox)} cells converged")

# ── 7. Summary ──
print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

summary = {
    "Module": "phenotype_association (Shears)",
    "Dataset": f"GSE132465 ({N_CELLS} cells) + TCGA-COAD ({adata_bulk_raw.shape[0]} bulk)",
    "HVG overlap": f"{len(overlap)}/{max(len(orig_genes),len(gp_genes))} ({100*len(overlap)/max(len(orig_genes),len(gp_genes)):.0f}%)",
    "Ridge alpha": ALPHA,
    "Weights global r": f"{r_all:.4f}",
    "Weights per-sample r (mean)": f"{np.mean(per_sample_r):.4f}",
    "Weights RMSE": f"{rmse:.2e}",
    "Cox orig status": f"{orig_cox_ok if orig_cox is not None else 'CRASHED'}/{N_CELLS}",
    "Cox genproai status": f"{gp_cox_ok}/{len(gp_cox)}",
    "Cox coef r": f"{cox_coef_r:.4f}" if cox_coef_r else "N/A",
    "Time weights orig": f"{t_orig_w:.2f}s",
    "Time weights genproai": f"{t_gp_w:.2f}s",
    "Time cox genproai": f"{t_gp_cox:.2f}s",
}

for k, v in summary.items():
    print(f"  {k:30s}: {v}")
print("=" * 60)

# Save
pd.Series(summary).to_csv(f"{RESULTS_DIR}/benchmark_shears_v2.csv")
print(f"\nSaved to {RESULTS_DIR}/benchmark_shears_v2.csv")
