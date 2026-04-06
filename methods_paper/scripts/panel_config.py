"""
Panel configuration for GenProAI Tools methods paper.
Target journal: Briefings in Bioinformatics (BIB, Oxford Academic)

Reverse dimension workflow:
  Journal page width → figure width → gap allocation → panel sizes
  All panels rendered at EXACT final size (no rescaling).
"""

# ── Journal Specification ──
JOURNAL = "Briefings in Bioinformatics"
PAGE_WIDTH_MM = 170      # full-width figure
COLUMN_WIDTH_MM = 80     # single-column figure
MIN_FONT_PT = 6
DPI = 600                # line art
FORMAT = "pdf"

# ── Gaps ──
H_GAP_MM = 3             # horizontal gap between panels
V_GAP_MM = 4             # vertical gap between rows
MARGIN_MM = 1            # outer margin

# ── Conversion ──
MM_TO_INCH = 1 / 25.4

# ── Typography (matched to journal min font) ──
BASE_SIZE = 7            # base font size (pt)
TITLE_SIZE = 8           # panel title
LABEL_SIZE = 10          # panel label (a, b, c)
AXIS_TEXT_SIZE = 6       # axis tick labels
LEGEND_TEXT_SIZE = 6     # legend text
ANNOT_SIZE = 6           # in-plot annotations

# ── Colorblind-safe palette (Okabe-Ito) ──
PAL = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
    "yellow":  "#F0E442",
    "skyblue": "#56B4E9",
}

# Semantic colors
COL_ORIGINAL = PAL["red"]       # original tool (R/buggy)
COL_GENPROAI = PAL["blue"]      # GenProAI (Python/fixed)
COL_SHEARS   = PAL["blue"]
COL_CYTOTRACE = PAL["green"]
COL_METAFLUX = PAL["orange"]
COL_BUG_FATAL = PAL["red"]
COL_BUG_HIGH = PAL["orange"]
COL_BUG_MED  = PAL["yellow"]
COL_BUG_LOW  = PAL["grey"]
COL_NS       = "#CCCCCC"

# ── Figure Layouts (reverse dimension) ──
def _compute_panels(total_w_mm, n_cols, n_rows, panel_h_mm):
    """Compute panel dimensions from figure width."""
    pw = (total_w_mm - (n_cols - 1) * H_GAP_MM) / n_cols
    total_h = panel_h_mm * n_rows + (n_rows - 1) * V_GAP_MM
    return {
        "fig_w_in": total_w_mm * MM_TO_INCH,
        "fig_h_in": total_h * MM_TO_INCH,
        "panel_w_in": pw * MM_TO_INCH,
        "panel_h_in": panel_h_mm * MM_TO_INCH,
        "panel_w_mm": pw,
        "panel_h_mm": panel_h_mm,
        "total_h_mm": total_h,
    }

# Fig 1: Overview (2×2 uniform grid)
FIG1 = _compute_panels(PAGE_WIDTH_MM, n_cols=2, n_rows=2, panel_h_mm=62)
# → panels 83.5mm × 62mm each, figure 170mm × 128mm

# Fig 2: Equivalence (3×2 uniform grid)
FIG2 = _compute_panels(PAGE_WIDTH_MM, n_cols=3, n_rows=2, panel_h_mm=52)
# → panels 54.7mm × 52mm each, figure 170mm × 108mm

# Fig 3: Scaling (2×2 uniform grid)
FIG3 = _compute_panels(PAGE_WIDTH_MM, n_cols=2, n_rows=2, panel_h_mm=58)
# → panels 83.5mm × 58mm each, figure 170mm × 120mm

# Fig 4: Bug Audit — mixed layout
# Row 1: 2 panels (a, b) each 83.5mm × 52mm
# Row 2: 1 panel (c) full width 170mm × 68mm (table)
FIG4_ROW1 = _compute_panels(PAGE_WIDTH_MM, n_cols=2, n_rows=1, panel_h_mm=52)
FIG4_TABLE_H_MM = 68
FIG4 = {
    "fig_w_in": PAGE_WIDTH_MM * MM_TO_INCH,
    "fig_h_in": (52 + V_GAP_MM + FIG4_TABLE_H_MM) * MM_TO_INCH,
    "row1_panel_w_in": FIG4_ROW1["panel_w_in"],
    "row1_panel_h_in": FIG4_ROW1["panel_h_in"],
    "table_w_in": PAGE_WIDTH_MM * MM_TO_INCH,
    "table_h_in": FIG4_TABLE_H_MM * MM_TO_INCH,
}

# Fig 5: Combination (3×2 uniform grid)
FIG5 = _compute_panels(PAGE_WIDTH_MM, n_cols=3, n_rows=2, panel_h_mm=55)
# → panels 54.7mm × 55mm each, figure 170mm × 114mm

# ── matplotlib rcParams helper ──
def apply_style():
    """Apply journal-matched matplotlib style globally."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": BASE_SIZE,
        "axes.linewidth": 0.6,
        "axes.labelsize": BASE_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "xtick.labelsize": AXIS_TEXT_SIZE,
        "ytick.labelsize": AXIS_TEXT_SIZE,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "legend.fontsize": LEGEND_TEXT_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,       # TrueType (editable in Illustrator)
        "ps.fonttype": 42,
    })

def save_panel(fig, name, figures_dir=None):
    """Save figure as PDF + PNG at journal DPI."""
    if figures_dir is None:
        figures_dir = "/Users/yuezong.bai/projects/genproai_tools/methods_paper/figures"
    import os
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(f"{figures_dir}/{name}.pdf", format="pdf")
    fig.savefig(f"{figures_dir}/{name}.png", format="png", dpi=300)  # PNG at 300 for preview
    print(f"  Saved {name}.pdf + .png")


# ── Summary ──
if __name__ == "__main__":
    print("GenProAI Tools — Figure Dimensions (BIB)")
    print("=" * 50)
    for name, spec in [("Fig1", FIG1), ("Fig2", FIG2), ("Fig3", FIG3), ("Fig5", FIG5)]:
        print(f"{name}: {spec['fig_w_in']:.2f}\" × {spec['fig_h_in']:.2f}\" "
              f"(panels {spec['panel_w_mm']:.1f}mm × {spec['panel_h_mm']}mm)")
    print(f"Fig4: {FIG4['fig_w_in']:.2f}\" × {FIG4['fig_h_in']:.2f}\" (mixed layout)")
    print(f"\nBase font: {BASE_SIZE}pt, DPI: {DPI}, format: {FORMAT}")
