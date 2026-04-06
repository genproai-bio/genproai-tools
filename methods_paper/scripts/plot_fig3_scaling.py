"""
Fig 3: Speed & Scaling Benchmark (4 panels)
a) CytoTRACE scaling (log-log)
b) METAFlux scaling (log-log)
c) Speedup vs data size
d) Summary speedup at max scale
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/data"
FIGURES = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/figures"
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_GREY = "#999999"

# ── Load data ──
cyto = pd.read_csv(f"{RESULTS}/scaling_cytotrace.csv")
meta = pd.read_csv(f"{RESULTS}/scaling_metaflux.csv")

# Add extrapolated 50K for CytoTRACE R
cyto_measured = cyto.dropna(subset=["runtime_r_s"])
from numpy.polynomial import polynomial as P
x_log = np.log10(cyto_measured["n_cells"].values)
y_r_log = np.log10(cyto_measured["runtime_r_s"].values)
coef_r = P.polyfit(x_log, y_r_log, 1)
cyto_50k_r = 10 ** P.polyval(np.log10(50000), coef_r)

fig = plt.figure(figsize=(7.08, 4.5))
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)
panels = "abcd"

# ── Panel a: CytoTRACE scaling ──
ax = fig.add_subplot(gs[0, 0])
# R points
cyto_r = cyto.dropna(subset=["runtime_r_s"])
ax.loglog(cyto_r["n_cells"], cyto_r["runtime_r_s"], "o-", color=C_RED,
          ms=4, lw=1.2, label="CytoTRACE R", zorder=3)
# R extrapolated point
ax.loglog(50000, cyto_50k_r, "o", color=C_RED, ms=4, mfc="white", mew=1.2, zorder=3)
# Python points
ax.loglog(cyto["n_cells"], cyto["runtime_python_s"], "s-", color=C_BLUE,
          ms=4, lw=1.2, label="GenProAI (Python)", zorder=3)

# Fit lines
x_fit = np.logspace(3, 4.7, 50)
y_r_fit = 10 ** P.polyval(np.log10(x_fit), coef_r)
y_py_log = np.log10(cyto["runtime_python_s"].values)
x_py_log = np.log10(cyto["n_cells"].values)
coef_py = P.polyfit(x_py_log, y_py_log, 1)
y_py_fit = 10 ** P.polyval(np.log10(x_fit), coef_py)
ax.loglog(x_fit, y_r_fit, "--", color=C_RED, lw=0.6, alpha=0.5)
ax.loglog(x_fit, y_py_fit, "--", color=C_BLUE, lw=0.6, alpha=0.5)

ax.set_xlabel("Number of cells")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("CytoTRACE", fontsize=8, fontstyle="italic")
ax.legend(frameon=False, loc="upper left")

# Annotate scaling exponents
ax.text(0.65, 0.35, f"R: n^{coef_r[1]:.2f}", transform=ax.transAxes,
        fontsize=6, color=C_RED)
ax.text(0.65, 0.22, f"Py: n^{coef_py[1]:.2f}", transform=ax.transAxes,
        fontsize=6, color=C_BLUE)
ax.text(-0.15, 1.08, panels[0], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel b: METAFlux scaling ──
ax = fig.add_subplot(gs[0, 1])
ax.loglog(meta["n_samples"], meta["runtime_r_s"], "o-", color=C_RED,
          ms=4, lw=1.2, label="METAFlux R")
ax.loglog(meta["n_samples"], meta["runtime_python_s"], "s-", color=C_BLUE,
          ms=4, lw=1.2, label="GenProAI (Python)")

ax.set_xlabel("Number of samples")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("METAFlux (MRAS)", fontsize=8, fontstyle="italic")
ax.legend(frameon=False, loc="upper left")
ax.text(-0.15, 1.08, panels[1], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel c: Speedup vs data size ──
ax = fig.add_subplot(gs[1, 0])
# CytoTRACE speedup
cyto_sp = cyto.dropna(subset=["speedup"])
ax.plot(cyto_sp["n_cells"], cyto_sp["speedup"], "o-", color=C_GREEN,
        ms=5, lw=1.5, label="CytoTRACE")
# Add extrapolated 50K
ax.plot(50000, cyto_50k_r / cyto.loc[cyto["n_cells"]==50000, "runtime_python_s"].values[0],
        "o", color=C_GREEN, ms=5, mfc="white", mew=1.5)

ax.set_xlabel("Number of cells")
ax.set_ylabel("Speedup (R / Python)")
ax.set_title("Speedup scaling", fontsize=8, fontstyle="italic")

# Add METAFlux on secondary x-axis (inset or twin)
ax2 = ax.twinx()
ax2.plot(meta["n_samples"] * 100, meta["speedup"], "^-", color=C_ORANGE,
         ms=5, lw=1.5, label="METAFlux")
ax2.set_ylabel("METAFlux speedup", color=C_ORANGE, fontsize=7)
ax2.tick_params(axis="y", labelcolor=C_ORANGE, labelsize=6)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
ax.text(-0.15, 1.08, panels[2], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel d: Summary speedup at max scale ──
ax = fig.add_subplot(gs[1, 1])
modules = ["CytoTRACE\n(20K cells)", "METAFlux\n(329 samples)", "CytoTRACE\n(50K, extrap.)"]
speedups = [11.3, 9.8, 17.1]
colors = [C_GREEN, C_ORANGE, C_GREEN]
hatches = ["", "", "//"]
bars = ax.bar(range(len(modules)), speedups, color=colors, width=0.6,
              edgecolor="white", linewidth=0.5)
for bar, h in zip(bars, hatches):
    if h:
        bar.set_hatch(h)
        bar.set_edgecolor(C_GREY)
ax.set_xticks(range(len(modules)))
ax.set_xticklabels(modules, fontsize=6)
ax.set_ylabel("Speedup (×)")
ax.set_title("Maximum observed speedup", fontsize=8, fontstyle="italic")
for bar, val in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}×", ha="center", va="bottom", fontsize=7, fontweight="bold")
ax.axhline(1, color=C_GREY, lw=0.5, ls=":")
ax.text(-0.15, 1.08, panels[3], transform=ax.transAxes, fontsize=11, fontweight="bold")

plt.savefig(f"{FIGURES}/fig3_scaling.pdf", format="pdf")
plt.savefig(f"{FIGURES}/fig3_scaling.png", format="png", dpi=300)
print(f"Saved fig3_scaling.pdf/png to {FIGURES}/")
plt.close()
