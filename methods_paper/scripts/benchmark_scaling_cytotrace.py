"""
Scaling benchmark: CytoTRACE R vs Python across cell counts
Sizes: 1K, 2K, 5K, 10K, 20K, 50K cells
Fix: convert to dense AFTER subsetting to avoid OOM on full matrix
"""

import sys, time, os, subprocess, tempfile
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/yuezong.bai/projects/genproai_tools")

SCRNA_PATH = "/Users/yuezong.bai/Downloads/客户科研项目/小项目/边总文章数据/projects/STAD-1_senescence/data/gse183904_qc_annotated.h5ad"
R_SCRIPT = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/scripts/cytotrace_reference.R"
RESULTS_DIR = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZES = [1000, 2000, 5000, 10000, 20000, 50000]

print("=" * 60)
print("SCALING BENCHMARK: CytoTRACE R vs Python")
print("=" * 60)

# Load full data once (keep sparse!)
print("\nLoading scRNA data...")
t0 = time.time()
adata = sc.read_h5ad(SCRNA_PATH)
n_total = adata.shape[0]
n_genes = adata.shape[1]
gene_names = np.array(adata.var_names)
cell_names = np.array(adata.obs_names)
print(f"  Full: {n_total} cells × {n_genes} genes, {time.time()-t0:.1f}s")
print(f"  Matrix type: {type(adata.X)}")

from genproai_tools.cytotrace import cytotrace

results = []

for n_cells in SIZES:
    if n_cells > n_total:
        print(f"\n--- Skipping {n_cells} (only {n_total} cells available) ---")
        continue

    print(f"\n--- n_cells = {n_cells} ---")

    # Subset FIRST, then convert to dense
    np.random.seed(42)
    idx = np.random.choice(n_total, n_cells, replace=False)
    X_sub = adata.X[idx]
    if sparse.issparse(X_sub):
        X_sub = X_sub.toarray()
    X = np.asarray(X_sub, dtype=np.float64)
    cells = cell_names[idx]
    print(f"  Dense matrix: {X.shape}, {X.nbytes / 1e6:.0f} MB")

    # --- Python ---
    t1 = time.time()
    py_res = cytotrace(X, gene_names=gene_names, top_genes=200)
    t_py = time.time() - t1
    print(f"  Python: {t_py:.3f}s")

    # --- R ---
    # For large matrices, use feather-like approach: save as binary
    # But for compatibility, use CSV with reduced precision
    tmpdir = tempfile.mkdtemp()
    input_csv = os.path.join(tmpdir, "expr.csv")
    output_csv = os.path.join(tmpdir, "result.csv")

    # Export: for >20K cells, this will be large
    csv_size_est = n_cells * n_genes * 6 / 1e9  # ~6 bytes per value in CSV
    print(f"  CSV export est: {csv_size_est:.1f} GB")

    if csv_size_est > 5.0:
        print(f"  SKIP R: CSV too large ({csv_size_est:.1f} GB), extrapolating from smaller sizes")
        t_r = None
        corr = None
    else:
        t_export_start = time.time()
        df = pd.DataFrame(X, index=cells, columns=gene_names)
        df.to_csv(input_csv)
        t_export = time.time() - t_export_start
        actual_size = os.path.getsize(input_csv) / 1e9
        print(f"  CSV exported: {actual_size:.2f} GB in {t_export:.1f}s")

        t1 = time.time()
        result = subprocess.run(
            ["Rscript", R_SCRIPT, input_csv, output_csv],
            capture_output=True, text=True, timeout=1200,
        )
        t_r = time.time() - t1

        if result.returncode != 0:
            print(f"  R FAILED: {result.stderr[:200]}")
            t_r = None
            corr = None
        else:
            print(f"  R: {t_r:.3f}s (+ export {t_export:.1f}s)")

            # Verify equivalence
            r_res = pd.read_csv(output_csv, index_col="cell")
            from scipy.stats import pearsonr
            r_score = r_res.loc[cells, "score"].values
            py_score = py_res["score"]
            corr, _ = pearsonr(r_score, py_score)
            print(f"  Correlation: r={corr:.6f}")

    results.append({
        "n_cells": n_cells,
        "n_genes": n_genes,
        "runtime_python_s": round(t_py, 4),
        "runtime_r_s": round(t_r, 4) if t_r is not None else None,
        "speedup": round(t_r / t_py, 1) if t_r is not None else None,
        "correlation": round(corr, 6) if corr is not None else None,
    })

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    # Free memory
    del X, X_sub
    if 'df' in dir():
        del df

# Summary
print("\n" + "=" * 60)
print("SCALING SUMMARY: CytoTRACE")
print("=" * 60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
df_results.to_csv(f"{RESULTS_DIR}/scaling_cytotrace.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/scaling_cytotrace.csv")

# Extrapolate for skipped sizes
measured = df_results.dropna(subset=["runtime_r_s"])
if len(measured) >= 2:
    from numpy.polynomial import polynomial as P
    x = np.log10(measured["n_cells"].values)
    y_r = np.log10(measured["runtime_r_s"].values)
    y_py = np.log10(measured["runtime_python_s"].values)
    coef_r = P.polyfit(x, y_r, 1)
    coef_py = P.polyfit(x, y_py, 1)
    print(f"\nScaling exponent (log-log slope):")
    print(f"  R:      {coef_r[1]:.2f} (runtime ∝ n^{coef_r[1]:.2f})")
    print(f"  Python: {coef_py[1]:.2f} (runtime ∝ n^{coef_py[1]:.2f})")

    for n in SIZES:
        if n not in measured["n_cells"].values:
            t_r_pred = 10 ** P.polyval(np.log10(n), coef_r)
            t_py_pred = 10 ** P.polyval(np.log10(n), coef_py)
            print(f"  Extrapolated {n}: R ~{t_r_pred:.1f}s, Python ~{t_py_pred:.1f}s, speedup ~{t_r_pred/t_py_pred:.0f}x")
