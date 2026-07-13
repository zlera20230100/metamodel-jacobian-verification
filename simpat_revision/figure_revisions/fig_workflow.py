# -*- coding: utf-8 -*-
"""Generic component-level metamodel-Jacobian V&V workflow (Figure 1)."""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
BLUE = "#4B86B4"
BLUE_DARK = "#2F6690"
BLUE_PALE = "#EAF2F8"
GREY = "#5F6368"
GREY_PALE = "#F3F4F5"
ORANGE = "#C46A35"
ORANGE_PALE = "#FBEFE8"
INK = "#202124"


def box(ax, xy, wh, title, detail, fc, ec, title_color=INK, align="center"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.15, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(patch)
    ha = "left" if align == "left" else "center"
    tx = x + 0.028 * w if align == "left" else x + w / 2
    ax.text(tx, y + 0.64 * h, title, ha=ha, va="center",
            fontsize=9.0, fontweight="bold", color=title_color)
    ax.text(tx, y + 0.31 * h, detail, ha=ha, va="center",
            fontsize=7.6, color=GREY, linespacing=1.18)
    return patch


def arrow(ax, p0, p1, color=GREY, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
        connectionstyle="arc3,rad=0",
    ))


fig, ax = plt.subplots(figsize=(10.8, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.02, 0.965, "Component-level verification of a metamodel design Jacobian",
        ha="left", va="top", fontsize=12.0, fontweight="bold", color=INK)
ax.text(0.02, 0.908,
        "Reference labels calibrate an intended-use decision; deployment spends simulator calls only on selected components.",
        ha="left", va="top", fontsize=8.5, color=GREY)

# Main sequence
box(ax, (0.025, 0.54), (0.16, 0.25), "Independent retrains",
    r"$M$ fitted metamodels" + "\n" + r"$\widehat J_k^{(m)}=\partial\widehat F^{(m)}/\partial g_k$",
    BLUE_PALE, BLUE_DARK)
box(ax, (0.225, 0.54), (0.16, 0.25), "Component scores",
    "sign agreement, SNR,\nor magnitude", BLUE_PALE, BLUE_DARK)
box(ax, (0.425, 0.54), (0.18, 0.25), "Reference calibration",
    "verified gradients select score,\nthreshold, risk, and coverage", GREY_PALE, GREY)

# Decision diamond
cx, cy, dw, dh = 0.700, 0.665, 0.075, 0.125
diamond = Polygon(
    [[cx, cy + dh], [cx + dw, cy], [cx, cy - dh], [cx - dw, cy]],
    closed=True, facecolor="white", edgecolor=BLUE_DARK, linewidth=1.25,
)
ax.add_patch(diamond)
ax.text(cx, cy, "score meets\noperating rule?", ha="center", va="center",
        fontsize=8.1, color=INK, linespacing=1.15)

box(ax, (0.825, 0.66), (0.145, 0.15), "ACCEPT",
    r"use $\overline{J}_k$", BLUE_PALE, BLUE_DARK, title_color=BLUE_DARK)
box(ax, (0.825, 0.43), (0.145, 0.15), "VERIFY",
    "central FD\n2 simulator calls", ORANGE_PALE, ORANGE, title_color=ORANGE)

arrow(ax, (0.185, 0.665), (0.225, 0.665), BLUE_DARK)
arrow(ax, (0.385, 0.665), (0.425, 0.665), BLUE_DARK)
arrow(ax, (0.605, 0.665), (0.625, 0.665), BLUE_DARK)
arrow(ax, (0.775, 0.695), (0.825, 0.735), BLUE_DARK)
arrow(ax, (0.745, 0.57), (0.825, 0.505), ORANGE)
ax.text(0.792, 0.747, "yes", fontsize=7.3, color=BLUE_DARK, ha="center")
ax.text(0.780, 0.535, "no", fontsize=7.3, color=ORANGE, ha="center")

# Calibration/reference source and numerical screen
box(ax, (0.225, 0.18), (0.18, 0.19), "Reference simulator",
    "representative calibration queries\nand selective component checks",
    GREY_PALE, GREY)
box(ax, (0.445, 0.18), (0.18, 0.19), "Numerical derivative screen",
    r"step/grid/independent check" + "\n" + r"before assigning $J_k^\star$",
    GREY_PALE, GREY)
arrow(ax, (0.405, 0.275), (0.445, 0.275), GREY)
arrow(ax, (0.535, 0.37), (0.515, 0.54), GREY)

# Failure branch and outputs
box(ax, (0.665, 0.18), (0.18, 0.19), "ABSTAIN / refine reference",
    "used when the numerical\nderivative screen fails", ORANGE_PALE, ORANGE,
    title_color=ORANGE)
arrow(ax, (0.625, 0.275), (0.665, 0.275), ORANGE)
ax.text(0.645, 0.292, "fail", fontsize=7.3, color=ORANGE, ha="center")

box(ax, (0.865, 0.18), (0.105, 0.19), "Outputs",
    "hybrid Jacobian\nrisk / coverage\nsimulator cost", BLUE_PALE, BLUE_DARK,
    align="left")
arrow(ax, (0.898, 0.43), (0.918, 0.37), BLUE_DARK)
arrow(ax, (0.898, 0.66), (0.932, 0.37), BLUE_DARK)

ax.text(0.025, 0.065,
        "Score computation at a deployment query adds no reference call; calibration, verification, and failed-reference refinement do.",
        ha="left", va="center", fontsize=8.0, color=GREY)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_workflow.{ext}", dpi=360 if ext == "png" else None,
                bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("wrote fig_workflow.pdf/.png")
