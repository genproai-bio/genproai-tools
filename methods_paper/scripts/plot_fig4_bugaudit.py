"""
Fig 4: Bug Audit & Documentation (4 panels)
a) Issue category breakdown (horizontal bar)
b) Severity distribution by tool (stacked bar)
c) Tool × issue type heatmap
d) Shears bug#5 flowchart comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

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
C_PURPLE = "#CC79A7"
C_GREY = "#999999"
C_YELLOW = "#F0E442"

# ── Load bug audit data ──
bugs = pd.read_csv(f"{RESULTS}/bug_audit_table.csv")

fig = plt.figure(figsize=(7.08, 5.0))
gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.45)
panels = "abcd"

# ── Panel a: Category breakdown (horizontal bar) ──
ax = fig.add_subplot(gs[0, 0])
cat_counts = bugs["category"].value_counts()
cat_order = ["Bug", "Dependency", "API Design", "Ecosystem", "Performance", "Implementation", "Documentation", "Design"]
cat_counts = cat_counts.reindex([c for c in cat_order if c in cat_counts.index])
cat_colors = {
    "Bug": C_RED, "Dependency": C_ORANGE, "API Design": C_PURPLE,
    "Ecosystem": C_BLUE, "Performance": C_GREEN, "Implementation": C_GREY,
    "Documentation": C_YELLOW, "Design": "#56B4E9"
}
colors = [cat_colors.get(c, C_GREY) for c in cat_counts.index]
bars = ax.barh(range(len(cat_counts)), cat_counts.values, color=colors,
               height=0.6, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(cat_counts)))
ax.set_yticklabels(cat_counts.index, fontsize=7)
ax.set_xlabel("Number of issues")
ax.set_title("Issue categories", fontsize=8, fontstyle="italic")
ax.invert_yaxis()
for bar, val in zip(bars, cat_counts.values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(val), ha="left", va="center", fontsize=6)
ax.text(-0.2, 1.08, panels[0], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel b: Severity by tool (stacked bar) ──
ax = fig.add_subplot(gs[0, 1])
tools = ["Shears", "Scimap", "DeepScence", "METAFlux", "CytoTRACE"]
severity_levels = ["Fatal", "High", "Medium", "Low"]
severity_colors = {"Fatal": C_RED, "High": C_ORANGE, "Medium": C_YELLOW, "Low": C_GREY}

# Count severity per tool
sev_data = {}
for sev in severity_levels:
    sev_data[sev] = []
    for tool in tools:
        count = len(bugs[(bugs["tool"] == tool) & (bugs["severity"] == sev)])
        sev_data[sev].append(count)

bottom = np.zeros(len(tools))
for sev in severity_levels:
    vals = sev_data[sev]
    ax.bar(range(len(tools)), vals, bottom=bottom, color=severity_colors[sev],
           width=0.6, label=sev, edgecolor="white", linewidth=0.3)
    bottom += np.array(vals)

ax.set_xticks(range(len(tools)))
ax.set_xticklabels(tools, fontsize=6, rotation=30, ha="right")
ax.set_ylabel("Number of issues")
ax.set_title("Severity by tool", fontsize=8, fontstyle="italic")
ax.legend(frameon=False, fontsize=6, loc="upper right", ncol=2)
ax.text(-0.2, 1.08, panels[1], transform=ax.transAxes, fontsize=11, fontweight="bold")

# ── Panel c: Detailed issue table (text-based) ──
ax = fig.add_subplot(gs[1, :])
ax.axis("off")

# Create a formatted table
table_data = []
for _, row in bugs.iterrows():
    sev_symbol = {"Fatal": "●", "High": "▲", "Medium": "■", "Low": "○"}
    table_data.append([
        row["tool"],
        row["issue_id"],
        sev_symbol.get(row["severity"], "?") + " " + row["severity"],
        row["description"][:65] + ("..." if len(row["description"]) > 65 else ""),
        "✓" if row["fixed_in_genproai"] == "Yes" else "—",
    ])

col_labels = ["Tool", "ID", "Severity", "Description", "Fixed"]
col_widths = [0.08, 0.05, 0.09, 0.68, 0.05]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    colWidths=col_widths,
    loc="center",
    cellLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(5.5)
table.scale(1, 1.15)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#E8E8E8")
    cell.set_text_props(fontweight="bold", fontsize=6)

# Color severity cells
sev_bg = {"Fatal": "#FFD0D0", "High": "#FFE8CC", "Medium": "#FFFFF0", "Low": "#F0F0F0"}
for i, row in enumerate(table_data):
    sev_text = row[2].split(" ")[-1]
    table[i+1, 2].set_facecolor(sev_bg.get(sev_text, "white"))

ax.set_title("Complete issue audit (14 issues across 5 tools)", fontsize=9,
             fontstyle="italic", pad=12)
ax.text(-0.02, 1.02, panels[2], transform=ax.transAxes, fontsize=11, fontweight="bold")

plt.savefig(f"{FIGURES}/fig4_bugaudit.pdf", format="pdf")
plt.savefig(f"{FIGURES}/fig4_bugaudit.png", format="png", dpi=300)
print(f"Saved fig4_bugaudit.pdf/png to {FIGURES}/")
plt.close()
