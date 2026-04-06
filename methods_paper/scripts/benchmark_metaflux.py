"""
Benchmark: METAFlux R (original) vs genproai_tools.metabolic (Python)
Dataset: TCGA-OV (subset 50 samples for speed)
Metric: MRAS (metabolic reaction activity scores) correlation
"""

import sys, time, os, subprocess, tempfile
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/yuezong.bai/projects/genproai_tools")

BULK_EXPR_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/tcga_coadread/TCGA_COAD_HiSeqV2_xena.gz"
R_SCRIPT = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/scripts/metaflux_reference.R"
N_SAMPLES = 50
RESULTS_DIR = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("BENCHMARK: METAFlux R vs genproai metabolic (Python)")
print("=" * 60)

# ── 1. Load data ──
print("\n[1/5] Loading TCGA-COAD expression data...")
# Using TCGA-COAD (both R and Python tested on this)
# Format: gene_symbol × samples, log2(norm_count+1)
t0 = time.time()
expr = pd.read_csv(BULK_EXPR_PATH, sep="\t", index_col=0)
print(f"  Full: {expr.shape}")

# Subset samples
np.random.seed(42)
cols = np.random.choice(expr.columns, N_SAMPLES, replace=False)
expr_sub = expr[cols].copy()
del expr
print(f"  Subset: {expr_sub.shape}, range=[{expr_sub.min().min():.2f}, {expr_sub.max().max():.2f}]")
print(f"  Load time: {time.time()-t0:.1f}s")

# ── 2. Export for R ──
print("\n[2/5] Exporting for R...")
tmpdir = tempfile.mkdtemp()
input_csv = os.path.join(tmpdir, "expr_matrix.csv")
output_r_csv = os.path.join(tmpdir, "mras_r.csv")

expr_sub.to_csv(input_csv)
print(f"  Exported: {os.path.getsize(input_csv)/1e6:.1f} MB")

# ── 3. Run R METAFlux ──
print("\n[3/5] Running R METAFlux calculate_reaction_score...")
t1 = time.time()
result = subprocess.run(
    ["Rscript", R_SCRIPT, input_csv, output_r_csv],
    capture_output=True, text=True, timeout=600,
)
t_r = time.time() - t1

if result.stdout:
    for line in result.stdout.strip().split("\n"):
        print(f"  {line}")
if result.returncode != 0:
    print(f"  R ERROR: {result.stderr[:500]}")
    sys.exit(1)
print(f"  R total time: {t_r:.2f}s")

# Read R results
mras_r = pd.read_csv(output_r_csv, index_col=0)
print(f"  R MRAS: {mras_r.shape}")

# ── 4. Run Python genproai ──
print("\n[4/5] Running genproai metabolic...")
from genproai_tools.metabolic import calculate_reaction_score

t1 = time.time()
mras_py = calculate_reaction_score(expr_sub)
t_py = time.time() - t1
print(f"  Python MRAS: {mras_py.shape}")
print(f"  Python time: {t_py:.2f}s")

# ── 5. Compare MRAS ──
print("\n[5/5] Comparing MRAS outputs...")

# Align reactions and samples
common_rxn = mras_r.index.intersection(mras_py.index)
common_samp = mras_r.columns.intersection(mras_py.columns)
print(f"  Common reactions: {len(common_rxn)}/{max(len(mras_r), len(mras_py))}")
print(f"  Common samples: {len(common_samp)}/{N_SAMPLES}")

mr = mras_r.loc[common_rxn, common_samp]
mp = mras_py.loc[common_rxn, common_samp]

# Global correlation
mr_flat = mr.values.flatten()
mp_flat = mp.values.flatten()
mask = np.isfinite(mr_flat) & np.isfinite(mp_flat)

r_global, _ = pearsonr(mr_flat[mask], mp_flat[mask])
rho_global, _ = spearmanr(mr_flat[mask], mp_flat[mask])
rmse = np.sqrt(np.mean((mr_flat[mask] - mp_flat[mask]) ** 2))
max_diff = np.max(np.abs(mr_flat[mask] - mp_flat[mask]))

print(f"\n  Global Pearson r:    {r_global:.6f}")
print(f"  Global Spearman:     {rho_global:.6f}")
print(f"  RMSE:                {rmse:.6e}")
print(f"  Max abs diff:        {max_diff:.6e}")

# Per-sample correlation
per_sample_r = []
for col in common_samp:
    o = mr[col].values
    g = mp[col].values
    m = np.isfinite(o) & np.isfinite(g)
    if m.sum() > 100:
        r, _ = pearsonr(o[m], g[m])
        per_sample_r.append(r)
print(f"  Per-sample r:  mean={np.mean(per_sample_r):.6f}  min={np.min(per_sample_r):.6f}")

# Per-reaction correlation (across samples)
per_rxn_r = []
for rxn in common_rxn:
    o = mr.loc[rxn].values
    g = mp.loc[rxn].values
    m = np.isfinite(o) & np.isfinite(g)
    if m.sum() > 5 and np.std(o[m]) > 0 and np.std(g[m]) > 0:
        r, _ = pearsonr(o[m], g[m])
        per_rxn_r.append(r)
print(f"  Per-reaction r: mean={np.mean(per_rxn_r):.6f}  min={np.min(per_rxn_r):.6f}  (n={len(per_rxn_r)})")

# Check value distributions
r_unique = np.unique(mr_flat[mask])
p_unique = np.unique(mp_flat[mask])
print(f"\n  R unique values:  {len(r_unique)}, {np.sum(mr_flat[mask]==0)} zeros, {np.sum(mr_flat[mask]==1)} ones")
print(f"  Py unique values: {len(p_unique)}, {np.sum(mp_flat[mask]==0)} zeros, {np.sum(mp_flat[mask]==1)} ones")

# Detailed breakdown of discrepancies
if max_diff > 1e-10:
    diff = np.abs(mr.values - mp.values)
    n_exact = np.sum(diff < 1e-10)
    n_close = np.sum(diff < 1e-3)
    n_total = diff.size
    print(f"\n  Exact match (<1e-10): {n_exact}/{n_total} ({100*n_exact/n_total:.1f}%)")
    print(f"  Close match (<1e-3):  {n_close}/{n_total} ({100*n_close/n_total:.1f}%)")

    # Find worst discrepancies
    worst_idx = np.unravel_index(np.argsort(diff.ravel())[-5:], diff.shape)
    print(f"  Worst 5 discrepancies:")
    for i in range(5):
        rxn = common_rxn[worst_idx[0][i]]
        samp = common_samp[worst_idx[1][i]]
        rv = mr.loc[rxn, samp]
        pv = mp.loc[rxn, samp]
        print(f"    {rxn}: R={rv:.6f} vs Py={pv:.6f} (diff={abs(rv-pv):.6e})")

# Summary
print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

summary = {
    "Module": "metabolic (METAFlux MRAS)",
    "Original": "METAFlux R v0.0.0.9000",
    "Dataset": f"TCGA-COAD ({N_SAMPLES} samples, {expr_sub.shape[0]} genes)",
    "Common reactions": f"{len(common_rxn)}",
    "MRAS global Pearson r": f"{r_global:.6f}",
    "MRAS global Spearman": f"{rho_global:.6f}",
    "MRAS RMSE": f"{rmse:.2e}",
    "MRAS max diff": f"{max_diff:.2e}",
    "Per-sample r (mean)": f"{np.mean(per_sample_r):.6f}",
    "Per-sample r (min)": f"{np.min(per_sample_r):.6f}",
    "Per-reaction r (mean)": f"{np.mean(per_rxn_r):.6f}",
    "Per-reaction r (n)": f"{len(per_rxn_r)}",
    "Runtime R": f"{t_r:.2f}s",
    "Runtime Python": f"{t_py:.2f}s",
    "Speedup": f"{t_r/t_py:.1f}x" if t_py > 0 else "N/A",
}

for k, v in summary.items():
    print(f"  {k:30s}: {v}")
print("=" * 60)

pd.Series(summary).to_csv(f"{RESULTS_DIR}/benchmark_metaflux.csv")
print(f"\nSaved to {RESULTS_DIR}/benchmark_metaflux.csv")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
