"""
Scaling benchmark: METAFlux R vs Python across sample counts
Sizes: 10, 25, 50, 100, 200, 329 samples
Output: CSV with runtime per size per implementation
"""

import sys, time, os, subprocess, tempfile
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/yuezong.bai/projects/genproai_tools")

BULK_EXPR_PATH = "/Users/yuezong.bai/Data/biomedical_data_assets/colorectal_cancer/tcga_coadread/TCGA_COAD_HiSeqV2_xena.gz"
R_SCRIPT = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/scripts/metaflux_reference.R"
RESULTS_DIR = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZES = [10, 25, 50, 100, 200, 329]

print("=" * 60)
print("SCALING BENCHMARK: METAFlux R vs Python")
print("=" * 60)

# Load full data once
print("\nLoading TCGA-COAD expression data...")
expr_full = pd.read_csv(BULK_EXPR_PATH, sep="\t", index_col=0)
n_total_samples = expr_full.shape[1]
n_genes = expr_full.shape[0]
print(f"  Full: {n_genes} genes × {n_total_samples} samples")

from genproai_tools.metabolic import calculate_reaction_score

results = []

for n_samples in SIZES:
    if n_samples > n_total_samples:
        print(f"\n--- Skipping {n_samples} (only {n_total_samples} available) ---")
        continue

    print(f"\n--- n_samples = {n_samples} ---")

    # Subset
    np.random.seed(42)
    cols = np.random.choice(expr_full.columns, n_samples, replace=False)
    expr_sub = expr_full[cols].copy()

    # --- Python ---
    t1 = time.time()
    mras_py = calculate_reaction_score(expr_sub)
    t_py = time.time() - t1
    print(f"  Python: {t_py:.3f}s")

    # --- R ---
    tmpdir = tempfile.mkdtemp()
    input_csv = os.path.join(tmpdir, "expr.csv")
    output_csv = os.path.join(tmpdir, "mras.csv")

    expr_sub.to_csv(input_csv)

    t1 = time.time()
    result = subprocess.run(
        ["Rscript", R_SCRIPT, input_csv, output_csv],
        capture_output=True, text=True, timeout=600,
    )
    t_r = time.time() - t1

    if result.returncode != 0:
        print(f"  R FAILED: {result.stderr[:200]}")
        t_r = None
        corr = None
    else:
        print(f"  R: {t_r:.3f}s")

        # Verify equivalence
        mras_r = pd.read_csv(output_csv, index_col=0)
        common_rxn = mras_r.index.intersection(mras_py.index)
        common_col = mras_r.columns.intersection(mras_py.columns)
        mr = mras_r.loc[common_rxn, common_col].values.flatten()
        mp = mras_py.loc[common_rxn, common_col].values.flatten()
        mask = np.isfinite(mr) & np.isfinite(mp)
        from scipy.stats import pearsonr
        corr, _ = pearsonr(mr[mask], mp[mask])
        print(f"  Correlation: r={corr:.6f}")

    results.append({
        "n_samples": n_samples,
        "n_genes": n_genes,
        "runtime_python_s": round(t_py, 4),
        "runtime_r_s": round(t_r, 4) if t_r else None,
        "speedup": round(t_r / t_py, 1) if t_r else None,
        "correlation": round(corr, 6) if corr else None,
    })

    import shutil
    shutil.rmtree(tmpdir)

# Summary
print("\n" + "=" * 60)
print("SCALING SUMMARY: METAFlux")
print("=" * 60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
df_results.to_csv(f"{RESULTS_DIR}/scaling_metaflux.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/scaling_metaflux.csv")
