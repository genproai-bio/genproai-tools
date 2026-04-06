# GenProAI Tools

A unified Python toolkit for multi-dimensional tumor microenvironment (TME) characterization. Reimplements the core algorithms of 11 published bioinformatics methods in a single conda environment, with bug fixes and 5-17x speedups over original implementations.

## Modules

| Module | Source method | What it does |
|--------|-------------|--------------|
| `spatial` | Scimap | Spatial cell-cell interaction testing via k-NN permutation |
| `senescence` | DeepScence | Senescent cell detection using ZINB autoencoder |
| `metabolic` | METAFlux | Metabolic flux inference via GPR rules + QP optimization |
| `cytotrace` | CytoTRACE | Cell differentiation potential scoring |
| `stable_selection` | Stabl | Noise-injection FDR-controlled feature selection |
| `bias_detection` | HistBiases | Biomarker prediction bias detection |
| `phenotype_association` | Shears | Bulk-scRNA phenotype association (6 bug fixes) |
| `drug_sensitivity` | DREEP | Single-cell drug sensitivity prediction |
| `cin_signature` | CINSignatureQuantification | Chromosomal instability signature decomposition |
| `metastasis_index` | PBMA | Primary-to-metastasis TME remodeling indices |
| `covarying_neighborhoods` | rcna | Co-varying neighborhood analysis |

## Installation

```bash
# Requires Python >= 3.10 with scanpy, PyTorch, scikit-learn, osqp
git clone https://github.com/genproai-bio/genproai-tools.git
cd genproai-tools

# Add to your Python path
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Quick start

```python
from genproai_tools.cytotrace import cytotrace_adata
from genproai_tools.metabolic import metaflux_pipeline

# Differentiation potential scoring
cytotrace_adata(adata)  # adds adata.obs["cytotrace"]

# Metabolic flux inference
results = metaflux_pipeline(expr_df)  # returns {mras, flux, pathway_flux}
```

## Benchmark

Validated against original implementations on real-world datasets:

| Module | Original | Pearson *r* | Speedup |
|--------|----------|:-----------:|:-------:|
| `phenotype_association` | Shears v0.0.1 | 1.000 | 1.1x |
| `cytotrace` | CytoTRACE (R) | 1.000 | 5-17x |
| `metabolic` | METAFlux (R) | 0.986 | 4-10x |

## Citation

Manuscript in preparation. If you use GenProAI Tools in your research, please cite this repository.

## License

MIT
