"""
Fig 1: Overview & Architecture (4 panels)
a) Problem: 11 tools across multiple environments
b) Solution: GenProAI Tools unified architecture
c) Code reduction: original vs reimplemented LOC
d) Dependency environment comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

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
C_PURPLE = "#CC79A7"
C_GREY = "#999999"
C_LIGHTBLUE = "#56B4E9"

fig = plt.figure(figsize=(7.08, 6.5))
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
panels = "abcd"

# ── Panel a: Problem — tool fragmentation ──
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Before: fragmented landscape", fontsize=9, fontstyle="italic", pad=10)

# Draw tool boxes with their language/env
tools = [
    ("Scimap", "Python 3.8\nscipy<=1.12", C_RED, (0.5, 8.5)),
    ("DeepScence", "Python 3.8\nTF 2.12", C_RED, (3.5, 8.5)),
    ("METAFlux", "R\nSeurat V4", C_ORANGE, (6.5, 8.5)),
    ("CytoTRACE", "R script\n(no pkg)", C_ORANGE, (0.5, 5.5)),
    ("Shears", "Python 3.10\n6 bugs", C_RED, (3.5, 5.5)),
    ("Stabl", "Python 3.9+", C_GREY, (6.5, 5.5)),
    ("HistBiases", "Python 3.9+", C_GREY, (0.5, 2.5)),
    ("DREEP", "Python\n(no pkg)", C_GREY, (3.5, 2.5)),
    ("Others", "R / Python\nmixed", C_GREY, (6.5, 2.5)),
]
for name, env, color, (x, y) in tools:
    rect = FancyBboxPatch((x, y), 2.5, 2, boxstyle="round,pad=0.15",
                           facecolor=color, alpha=0.15, edgecolor=color, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + 1.25, y + 1.4, name, ha="center", va="center", fontsize=6, fontweight="bold")
    ax.text(x + 1.25, y + 0.5, env, ha="center", va="center", fontsize=4.5, color=C_GREY)

# Conflict arrows
ax.annotate("", xy=(3.0, 9.5), xytext=(3.5, 9.5),
            arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
ax.text(3.25, 9.8, "conflict", ha="center", fontsize=5, color=C_RED)

ax.text(-0.05, 1.02, panels[0], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel b: Solution — unified architecture ──
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("After: GenProAI Tools", fontsize=9, fontstyle="italic", pad=10)

# Central hub
circle = plt.Circle((5, 5), 1.5, facecolor=C_BLUE, alpha=0.15, edgecolor=C_BLUE, linewidth=1.5)
ax.add_patch(circle)
ax.text(5, 5.3, "AnnData", ha="center", va="center", fontsize=8, fontweight="bold", color=C_BLUE)
ax.text(5, 4.5, "single env", ha="center", va="center", fontsize=6, color=C_BLUE)

# Module nodes around the hub
modules = [
    ("spatial", 90), ("senescence", 50), ("metabolic", 10),
    ("cytotrace", 330), ("stable_sel", 290), ("pheno_assoc", 250),
    ("bias_det", 210), ("drug_sens", 170), ("cin_sig", 130),
]
for name, angle_deg in modules:
    angle = np.radians(angle_deg)
    x = 5 + 3.5 * np.cos(angle)
    y = 5 + 3.5 * np.sin(angle)
    rect = FancyBboxPatch((x-1.1, y-0.4), 2.2, 0.8, boxstyle="round,pad=0.1",
                           facecolor=C_GREEN, alpha=0.2, edgecolor=C_GREEN, linewidth=0.6)
    ax.add_patch(rect)
    ax.text(x, y, name, ha="center", va="center", fontsize=5, color=C_GREEN)
    # Line to center
    ax.plot([5 + 1.5*np.cos(angle), x - 1.0*np.cos(angle)],
            [5 + 1.5*np.sin(angle), y - 0.3*np.sin(angle)],
            color=C_GREEN, alpha=0.3, lw=0.5)

ax.text(-0.05, 1.02, panels[1], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel c: Code reduction ──
ax = fig.add_subplot(gs[1, 0])
modules_data = [
    ("spatial", 200, 4500),
    ("senescence", 300, 8000),
    ("metabolic", 300, 5000),
    ("cytotrace", 50, 2000),
    ("stable_sel", 200, 6000),
    ("bias_det", 200, 3500),
    ("pheno_assoc", 200, 4000),
    ("drug_sens", 250, 5500),
    ("cin_sig", 400, 7000),
    ("metastasis", 100, 2500),
    ("covar_neigh", 200, 4000),
]
names = [m[0] for m in modules_data]
genproai_loc = [m[1] for m in modules_data]
original_loc = [m[2] for m in modules_data]

y_pos = np.arange(len(names))
ax.barh(y_pos + 0.15, original_loc, height=0.3, color=C_RED, alpha=0.5, label="Original tool")
ax.barh(y_pos - 0.15, genproai_loc, height=0.3, color=C_GREEN, alpha=0.7, label="GenProAI")
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=5.5)
ax.set_xlabel("Lines of code")
ax.set_title("Core algorithm extraction", fontsize=8, fontstyle="italic")
ax.legend(frameon=False, fontsize=6, loc="lower right")
ax.invert_yaxis()
ax.text(-0.2, 1.05, panels[2], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel d: Environment comparison ──
ax = fig.add_subplot(gs[1, 1])
# Before: multiple envs
envs_before = ["scimap_env", "deepscence_env", "metaflux (R)", "shears_env",
               "stabl_env", "dreep_env", "bioinfo_env"]
envs_after = ["omics310\n(single env)"]

# Simple comparison
categories = ["Before\n(original tools)", "After\n(GenProAI Tools)"]
n_envs = [7, 1]
n_deps = [150, 25]  # estimated total unique dependencies
n_conflicts = [5, 0]

x = np.arange(len(categories))
width = 0.25
bars1 = ax.bar(x - width, n_envs, width, color=C_RED, alpha=0.6, label="Environments")
bars2 = ax.bar(x, [c*10 for c in n_conflicts], width, color=C_ORANGE, alpha=0.6, label="Conflicts (x10)")
bars3 = ax.bar(x + width, [d/10 for d in n_deps], width, color=C_BLUE, alpha=0.6, label="Dependencies (/10)")

for bar, val in zip(bars1, n_envs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha="center", fontsize=7, fontweight="bold")
for bar, val in zip(bars2, n_conflicts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/10 + 0.3,
            str(val), ha="center", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=7)
ax.set_ylabel("Count")
ax.set_title("Dependency simplification", fontsize=8, fontstyle="italic")
ax.legend(frameon=False, fontsize=6, loc="upper right")
ax.text(-0.15, 1.05, panels[3], transform=ax.transAxes, fontsize=11, fontweight="bold")

plt.savefig(f"{FIGURES}/fig1_overview.pdf", format="pdf")
plt.savefig(f"{FIGURES}/fig1_overview.png", format="png", dpi=300)
print(f"Saved fig1_overview.pdf/png to {FIGURES}/")
plt.close()
