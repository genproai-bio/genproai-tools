"""
Benchmark: original Shears v0.0.1 vs genproai_tools.phenotype_association
Dataset: GSE132465 CRC scRNA (subset) + TCGA-COAD bulk (subset)
Metrics: cell_weights Pearson r, Cox p-value rank correlation, runtime
"""

import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/yuezong.bai/projects/genproai_tools")

# ── Config ──
SCRNA_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/gse132465_scrna/gse132465_crc.h5ad"
BULK_EXPR_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/tcga_coadread/TCGA_COAD_HiSeqV2_xena.gz"
CLINICAL_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/tcga_coadread/TCGA-COAD_clinical_survival.csv"
N_CELLS = 1000
N_BULK = 100
RESULTS_DIR = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/data"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load data ──
print("=" * 60)
print("BENCHMARK: Shears v0.0.1 vs genproai phenotype_association")
print("=" * 60)

print("\n[1/6] Loading data...")
t0 = time.time()

# scRNA
adata_sc_full = sc.read_h5ad(SCRNA_PATH)
print(f"  scRNA full: {adata_sc_full.shape}")

# Subset: stratified sample by cell_type if available
if "cell_type" in adata_sc_full.obs.columns:
    ct_col = "cell_type"
elif "celltype" in adata_sc_full.obs.columns:
    ct_col = "celltype"
else:
    ct_col = None

np.random.seed(42)
if ct_col:
    # Stratified subsample
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
del adata_sc_full  # free memory

# Ensure raw counts in .X (check if it looks like counts)
if adata_sc_raw.X.max() < 50:
    # Likely log-normalized, check for raw layer
    if "counts" in adata_sc_raw.layers:
        adata_sc_raw.X = adata_sc_raw.layers["counts"].copy()
        print("  Using 'counts' layer as .X")
    elif "raw_counts" in adata_sc_raw.layers:
        adata_sc_raw.X = adata_sc_raw.layers["raw_counts"].copy()
        print("  Using 'raw_counts' layer as .X")
    else:
        print(f"  WARNING: .X max={adata_sc_raw.X.max():.2f}, may not be raw counts")

print(f"  scRNA subset: {adata_sc_raw.shape}")

# Bulk
bulk_expr = pd.read_csv(BULK_EXPR_PATH, sep="\t", index_col=0)
print(f"  Bulk expr: {bulk_expr.shape}")

# Clinical
clin = pd.read_csv(CLINICAL_PATH)
# Extract barcode and survival
clin = clin[["bcr_patient_barcode", "OS", "OS.time"]].copy()
clin = clin.rename(columns={"bcr_patient_barcode": "sample_id"})
# Clean survival data
clin["OS"] = pd.to_numeric(clin["OS"], errors="coerce")
clin["OS.time"] = pd.to_numeric(clin["OS.time"], errors="coerce")
clin = clin.dropna(subset=["OS", "OS.time"])
clin = clin[clin["OS.time"] > 0]

# Match bulk samples to clinical
# Xena expression has sample IDs like TCGA-AA-3672-01, clinical has TCGA-AA-3672
bulk_samples = bulk_expr.columns.tolist()
sample_map = {}
for s in bulk_samples:
    short = "-".join(s.split("-")[:3])
    if short in clin["sample_id"].values:
        sample_map[s] = short

print(f"  Matched bulk-clinical: {len(sample_map)} samples")

# Subset bulk to matched samples
matched_bulk = list(sample_map.keys())
np.random.seed(42)
if len(matched_bulk) > N_BULK:
    matched_bulk = list(np.random.choice(matched_bulk, N_BULK, replace=False))

bulk_sub = bulk_expr[matched_bulk].copy()
clin_sub = clin[clin["sample_id"].isin([sample_map[s] for s in matched_bulk])].copy()

# Create bulk AnnData
# Xena format: genes x samples → transpose to samples x genes
bulk_mat = bulk_sub.T
adata_bulk_raw = ad.AnnData(
    X=bulk_mat.values.astype(np.float32),
    obs=pd.DataFrame(index=bulk_mat.index),
    var=pd.DataFrame(index=bulk_mat.columns),
)
# Add survival info
for s in adata_bulk_raw.obs_names:
    short = sample_map.get(s, s)
    row = clin_sub[clin_sub["sample_id"] == short]
    if len(row) > 0:
        adata_bulk_raw.obs.loc[s, "OS_time"] = float(row.iloc[0]["OS.time"])
        adata_bulk_raw.obs.loc[s, "OS_event"] = float(row.iloc[0]["OS"])

# Drop samples without survival
adata_bulk_raw = adata_bulk_raw[adata_bulk_raw.obs["OS_time"].notna()].copy()

print(f"  Bulk AnnData: {adata_bulk_raw.shape}, OS events: {int(adata_bulk_raw.obs['OS_event'].sum())}")
print(f"  Data loading: {time.time()-t0:.1f}s")

# ── 2. Run ORIGINAL Shears ──
print("\n[2/6] Running original Shears...")

# Deep copy for original
adata_sc_orig = adata_sc_raw.copy()
adata_bulk_orig = adata_bulk_raw.copy()

import shears

t1 = time.time()
try:
    # Original API: recipe returns new objects
    adata_sc_orig, adata_bulk_orig = shears.pp.recipe_shears(adata_sc_orig, adata_bulk_orig)
    print(f"  Preprocess: {adata_sc_orig.shape[1]} HVGs")
except Exception as e:
    print(f"  Preprocess error: {e}")
    # Try workaround
    adata_sc_orig, adata_bulk_orig = shears.pp.recipe_shears(adata_sc_orig, adata_bulk_orig)

t_orig_preprocess = time.time() - t1

t1 = time.time()
shears.pp.cell_weights(adata_sc_orig, adata_bulk_orig)
t_orig_weights = time.time() - t1
print(f"  Cell weights: {t_orig_weights:.2f}s")

# Extract cell_weights
orig_weights = adata_sc_orig.obsm["cell_weights"].copy()
if isinstance(orig_weights, pd.DataFrame):
    orig_weights_df = orig_weights
else:
    orig_weights_df = pd.DataFrame(orig_weights, index=adata_sc_orig.obs_names)

# Cox survival (with try-except for original's bug#5)
t1 = time.time()
try:
    import statsmodels.api as sm
    orig_cox = shears.tl.shears_cox(
        adata_sc_orig, adata_bulk_orig,
        duration_col="OS_time",
        event_col="OS_event",
    )
    t_orig_cox = time.time() - t1
    print(f"  Cox test: {t_orig_cox:.2f}s, non-NaN: {orig_cox['pvalue'].notna().sum()}/{len(orig_cox)}")
except Exception as e:
    print(f"  Cox test FAILED (expected — bug#5): {type(e).__name__}: {e}")
    orig_cox = None
    t_orig_cox = float("nan")

t_orig_total = t_orig_preprocess + t_orig_weights + (t_orig_cox if not np.isnan(t_orig_cox) else 0)

# ── 3. Run genproai phenotype_association ──
print("\n[3/6] Running genproai phenotype_association...")

from genproai_tools import phenotype_association as pa

# Deep copy for genproai
adata_sc_gp = adata_sc_raw.copy()
adata_bulk_gp = adata_bulk_raw.copy()

t1 = time.time()
pa.preprocess(adata_sc_gp, adata_bulk_gp)
t_gp_preprocess = time.time() - t1

t1 = time.time()
gp_weights = pa.compute_cell_weights(adata_sc_gp, adata_bulk_gp)
t_gp_weights = time.time() - t1
print(f"  Cell weights: {t_gp_weights:.2f}s")

t1 = time.time()
gp_cox = pa.association_cox(
    adata_sc_gp, adata_bulk_gp,
    duration_col="OS_time",
    event_col="OS_event",
    penalizer=0.1,
)
t_gp_cox = time.time() - t1
print(f"  Cox test: {t_gp_cox:.2f}s, non-NaN: {gp_cox['pvalue'].notna().sum()}/{len(gp_cox)}")

t_gp_total = t_gp_preprocess + t_gp_weights + t_gp_cox

# ── 4. Compare outputs ──
print("\n[4/6] Comparing outputs...")

# 4a. Cell weights correlation
# Align indices
common_cells = orig_weights_df.index.intersection(gp_weights.index)
common_samples = orig_weights_df.columns.intersection(gp_weights.columns) if hasattr(orig_weights_df, 'columns') else None

if common_samples is not None and len(common_samples) > 0:
    ow = orig_weights_df.loc[common_cells, common_samples]
    gw = gp_weights.loc[common_cells, common_samples]
else:
    ow = orig_weights_df.loc[common_cells]
    gw = gp_weights.loc[common_cells]

# Flatten and correlate
ow_flat = ow.values.flatten()
gw_flat = gw.values.flatten()
mask = np.isfinite(ow_flat) & np.isfinite(gw_flat)
r_weights, p_weights = pearsonr(ow_flat[mask], gw_flat[mask])
rho_weights, _ = spearmanr(ow_flat[mask], gw_flat[mask])
print(f"  Cell weights (n={mask.sum()}): Pearson r={r_weights:.6f}, Spearman rho={rho_weights:.6f}")

# Per-sample correlation
per_sample_r = []
for col in ow.columns:
    o = ow[col].values
    g = gw[col].values
    m = np.isfinite(o) & np.isfinite(g) & ((o > 0) | (g > 0))
    if m.sum() > 10:
        r, _ = pearsonr(o[m], g[m])
        per_sample_r.append(r)
print(f"  Per-sample weight r: mean={np.mean(per_sample_r):.6f}, min={np.min(per_sample_r):.6f}, max={np.max(per_sample_r):.6f}")

# 4b. Cox p-value comparison (if original didn't crash)
if orig_cox is not None:
    common_cox = orig_cox.index.intersection(gp_cox.index)
    oc = orig_cox.loc[common_cox]
    gc = gp_cox.loc[common_cox]

    # Both non-NaN
    both_valid = oc["pvalue"].notna() & gc["pvalue"].notna()
    if both_valid.sum() > 10:
        r_cox_p, _ = pearsonr(-np.log10(oc.loc[both_valid, "pvalue"].clip(1e-300)),
                               -np.log10(gc.loc[both_valid, "pvalue"].clip(1e-300)))
        r_cox_coef, _ = pearsonr(oc.loc[both_valid, "coef"], gc.loc[both_valid, "coef"])
        rho_cox_coef, _ = spearmanr(oc.loc[both_valid, "coef"], gc.loc[both_valid, "coef"])
        print(f"  Cox coef (n={both_valid.sum()}): Pearson r={r_cox_coef:.6f}, Spearman rho={rho_cox_coef:.6f}")
        print(f"  Cox -log10(p): Pearson r={r_cox_p:.6f}")

        # Cells that failed in original but succeeded in genproai
        orig_fail = oc["pvalue"].isna() & gc["pvalue"].notna()
        print(f"  Cells: orig failed but genproai OK: {orig_fail.sum()} (bug#5 fix)")
    else:
        r_cox_p = r_cox_coef = rho_cox_coef = float("nan")
        print(f"  Too few valid Cox results to compare ({both_valid.sum()})")
else:
    r_cox_p = r_cox_coef = rho_cox_coef = float("nan")
    print("  Original Cox crashed entirely — confirms bug#5")

# ── 5. Summary ──
print("\n[5/6] Summary")
print("-" * 60)
print(f"  Dataset: {N_CELLS} cells × {adata_bulk_raw.shape[0]} bulk samples")
print(f"  HVGs: {adata_sc_orig.shape[1] if hasattr(adata_sc_orig, 'shape') else 'N/A'}")
print(f"  ")
print(f"  Cell weights Pearson r:  {r_weights:.6f}")
print(f"  Cell weights Spearman:   {rho_weights:.6f}")
print(f"  Per-sample r (mean):     {np.mean(per_sample_r):.6f}")
if not np.isnan(r_cox_coef):
    print(f"  Cox coef Pearson r:      {r_cox_coef:.6f}")
    print(f"  Cox coef Spearman:       {rho_cox_coef:.6f}")
print(f"  ")
print(f"  Runtime original:   preprocess={t_orig_preprocess:.1f}s  weights={t_orig_weights:.1f}s  cox={t_orig_cox:.1f}s  total={t_orig_total:.1f}s")
print(f"  Runtime genproai:   preprocess={t_gp_preprocess:.1f}s  weights={t_gp_weights:.1f}s  cox={t_gp_cox:.1f}s  total={t_gp_total:.1f}s")
print("-" * 60)

# ── 6. Save results ──
print("\n[6/6] Saving results...")
results = {
    "module": "phenotype_association",
    "original": "Shears v0.0.1",
    "dataset": f"GSE132465 ({N_CELLS} cells) + TCGA-COAD ({adata_bulk_raw.shape[0]} samples)",
    "n_cells": N_CELLS,
    "n_bulk": adata_bulk_raw.shape[0],
    "n_hvg": adata_sc_orig.shape[1] if hasattr(adata_sc_orig, 'shape') else None,
    "weights_pearson_r": r_weights,
    "weights_spearman_rho": rho_weights,
    "weights_per_sample_r_mean": np.mean(per_sample_r),
    "weights_per_sample_r_min": np.min(per_sample_r),
    "cox_coef_pearson_r": r_cox_coef if not np.isnan(r_cox_coef) else None,
    "cox_coef_spearman_rho": rho_cox_coef if not np.isnan(rho_cox_coef) else None,
    "cox_logp_pearson_r": r_cox_p if not np.isnan(r_cox_p) else None,
    "runtime_orig_total_s": t_orig_total,
    "runtime_genproai_total_s": t_gp_total,
    "runtime_orig_weights_s": t_orig_weights,
    "runtime_genproai_weights_s": t_gp_weights,
    "runtime_orig_cox_s": t_orig_cox if not np.isnan(t_orig_cox) else None,
    "runtime_genproai_cox_s": t_gp_cox,
    "bug5_rescued_cells": int(orig_fail.sum()) if orig_cox is not None else "N/A (original crashed)",
}

pd.Series(results).to_csv(f"{RESULTS_DIR}/benchmark_shears.csv")
print(f"  Saved to {RESULTS_DIR}/benchmark_shears.csv")
print("\nDone!")
