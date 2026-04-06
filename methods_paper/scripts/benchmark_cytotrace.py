"""
Benchmark: CytoTRACE R reference vs genproai_tools.cytotrace (Python)
Dataset: STAD-1 scRNA (GSE183904, subset 5000 cells)
"""

import sys, time, os, subprocess, tempfile
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/yuezong.bai/projects/genproai_tools")

SCRNA_PATH = "/Users/yuezong.bai/Downloads/客户科研项目/小项目/边总文章数据/projects/STAD-1_senescence/data/gse183904_qc_annotated.h5ad"
R_SCRIPT = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/scripts/cytotrace_reference.R"
N_CELLS = 5000
RESULTS_DIR = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("BENCHMARK: CytoTRACE R (reference) vs genproai (Python)")
print("=" * 60)

# ── 1. Load and subset data ──
print("\n[1/5] Loading scRNA data...")
t0 = time.time()
adata = sc.read_h5ad(SCRNA_PATH)
print(f"  Full: {adata.shape}")

# Subset
np.random.seed(42)
idx = np.random.choice(adata.shape[0], N_CELLS, replace=False)
adata_sub = adata[idx].copy()
del adata

# Get dense matrix (raw counts or .X)
X = adata_sub.X
if sparse.issparse(X):
    X = X.toarray()
X = np.asarray(X, dtype=np.float64)

gene_names = np.array(adata_sub.var_names)
cell_names = np.array(adata_sub.obs_names)
print(f"  Subset: {X.shape}, max={X.max():.1f}, nnz_rate={np.mean(X>0):.3f}")
print(f"  Load time: {time.time()-t0:.1f}s")

# ── 2. Export for R ──
print("\n[2/5] Exporting matrix for R...")
tmpdir = tempfile.mkdtemp()
input_csv = os.path.join(tmpdir, "expr_matrix.csv")
output_r_csv = os.path.join(tmpdir, "cytotrace_r_results.csv")

# Save as cells x genes CSV
df = pd.DataFrame(X, index=cell_names, columns=gene_names)
df.to_csv(input_csv)
print(f"  Exported: {df.shape} to {input_csv} ({os.path.getsize(input_csv)/1e6:.1f} MB)")

# ── 3. Run R reference ──
print("\n[3/5] Running R CytoTRACE reference...")
t1 = time.time()
result = subprocess.run(
    ["Rscript", R_SCRIPT, input_csv, output_r_csv],
    capture_output=True, text=True, timeout=600,
)
t_r = time.time() - t1
print(f"  R stdout: {result.stdout.strip()}")
if result.returncode != 0:
    print(f"  R stderr: {result.stderr[:500]}")
    sys.exit(1)
print(f"  R runtime: {t_r:.2f}s")

# Read R results
r_results = pd.read_csv(output_r_csv)
r_results = r_results.set_index("cell")
print(f"  R results: {r_results.shape}")

# ── 4. Run genproai CytoTRACE ──
print("\n[4/5] Running genproai CytoTRACE (Python)...")
from genproai_tools.cytotrace import cytotrace

t1 = time.time()
py_results = cytotrace(X, gene_names=gene_names, top_genes=200)
t_py = time.time() - t1
print(f"  Python runtime: {t_py:.2f}s")

# ── 5. Compare ──
print("\n[5/5] Comparing outputs...")

# Align
r_score = r_results.loc[cell_names, "score"].values
r_gc = r_results.loc[cell_names, "gene_counts"].values
py_score = py_results["score"]
py_gc = py_results["gene_counts"]

# Gene counts should be IDENTICAL
gc_match = np.allclose(r_gc, py_gc)
gc_r, _ = pearsonr(r_gc, py_gc)
print(f"  Gene counts identical: {gc_match} (r={gc_r:.6f})")

# Score correlation
r_score_corr, _ = pearsonr(r_score, py_score)
rho_score, _ = spearmanr(r_score, py_score)
rmse = np.sqrt(np.mean((r_score - py_score) ** 2))
max_diff = np.max(np.abs(r_score - py_score))

print(f"  Score Pearson r:   {r_score_corr:.6f}")
print(f"  Score Spearman:    {rho_score:.6f}")
print(f"  Score RMSE:        {rmse:.6e}")
print(f"  Score max diff:    {max_diff:.6e}")

# Gene correlations comparison
r_gene_corr = np.zeros(len(gene_names))
for j in range(len(gene_names)):
    col = X[:, j]
    if col.std() > 0:
        r_gene_corr[j], _ = pearsonr(col, r_gc)

py_gene_corr = py_results["gene_correlations"]
gc_corr_r, _ = pearsonr(r_gene_corr, py_gene_corr)
print(f"  Gene correlations r: {gc_corr_r:.6f}")

# Summary
print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

summary = {
    "Module": "cytotrace",
    "Original": "CytoTRACE R (Gulati 2020 Science)",
    "Dataset": f"GSE183904 STAD scRNA ({N_CELLS} cells, {X.shape[1]} genes)",
    "Gene counts identical": str(gc_match),
    "Score Pearson r": f"{r_score_corr:.6f}",
    "Score Spearman rho": f"{rho_score:.6f}",
    "Score RMSE": f"{rmse:.2e}",
    "Score max diff": f"{max_diff:.2e}",
    "Gene correlations r": f"{gc_corr_r:.6f}",
    "Runtime R": f"{t_r:.2f}s",
    "Runtime Python": f"{t_py:.2f}s",
    "Speedup": f"{t_r/t_py:.1f}x",
}

for k, v in summary.items():
    print(f"  {k:30s}: {v}")
print("=" * 60)

pd.Series(summary).to_csv(f"{RESULTS_DIR}/benchmark_cytotrace.csv")
print(f"\nSaved to {RESULTS_DIR}/benchmark_cytotrace.csv")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
