#!/usr/bin/env python3
"""Publication-style graphical abstract for the SIMPAT submission.

The layout is deliberately data-led: no rounded cards, shadows, gradients, or
decorative icons.  PDF and SVG retain vector geometry and editable text.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent

# Okabe--Ito-derived, print-safe palette.
BLUE = "#0072B2"
BLUE_LIGHT = "#DCECF4"
ORANGE = "#D55E00"
ORANGE_LIGHT = "#F7E6D8"
GREY_1 = "#F2F2F2"
GREY_2 = "#B8B8B8"
GREY_3 = "#666666"
INK = "#1B1B1B"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def text(ax, x, y, value, *, size=9.0, weight="normal", color=INK,
         ha="left", va="center", **kwargs):
    """Consistent text helper in normalized figure coordinates."""
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        **kwargs,
    )


def arrow(ax, start, end, *, color=INK, lw=1.05, scale=10):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
        )
    )


def panel_heading(ax, x, y, letter, heading):
    text(ax, x, y, letter, size=12.5, weight="bold", va="top")
    text(ax, x + 0.025, y - 0.002, heading, size=11.5, weight="bold", va="top")


fig, ax = plt.subplots(figsize=(13.2, 5.25), facecolor="white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ---------------------------------------------------------------------------
# Header: modest, left aligned, and separated from the content by one rule.
text(
    ax,
    0.035,
    0.956,
    "Reference-calibrated selective verification of surrogate Jacobians",
    size=16.2,
    weight="bold",
    va="top",
)
text(
    ax,
    0.035,
    0.904,
    "Rank components after training; spend reference simulations only where calibrated evidence is insufficient.",
    size=9.7,
    color=GREY_3,
    va="top",
)
ax.plot([0.035, 0.965], [0.862, 0.862], color=INK, lw=0.85)

# Three restrained columns, separated by hairlines instead of cards.
ax.plot([0.327, 0.327], [0.265, 0.825], color=GREY_2, lw=0.65)
ax.plot([0.698, 0.698], [0.265, 0.825], color=GREY_2, lw=0.65)

# ---------------------------------------------------------------------------
# (a) Cross-retrain evidence: a Jacobian sign matrix and explicit score set.
panel_heading(ax, 0.035, 0.824, "a", "Cross-retrain evidence")
text(ax, 0.060, 0.758, r"Autodiff components from $M$ independently trained models", size=8.8)
text(ax, 0.293, 0.758, r"$k$", size=8.4, color=GREY_3, ha="right")

signs = [
    [1, 1, 1, -1, 1, -1],
    [1, 1, -1, -1, 1, 1],
    [1, 1, 1, -1, -1, -1],
    [1, 1, -1, -1, 1, -1],
    [1, 1, 1, -1, 1, 1],
]
x0, y0 = 0.083, 0.465
cw, ch, gx, gy = 0.028, 0.043, 0.006, 0.009
for row, values in enumerate(signs):
    yy = y0 + (len(signs) - 1 - row) * (ch + gy)
    text(ax, 0.067, yy + ch / 2, rf"{row + 1}", size=7.4, color=GREY_3, ha="right")
    for col, value in enumerate(values):
        xx = x0 + col * (cw + gx)
        face = BLUE_LIGHT if value > 0 else ORANGE_LIGHT
        edge = BLUE if value > 0 else ORANGE
        ax.add_patch(Rectangle((xx, yy), cw, ch, facecolor=face, edgecolor=edge, lw=0.75))
        text(
            ax,
            xx + cw / 2,
            yy + ch / 2,
            "+" if value > 0 else "−",
            size=8.7,
            weight="bold",
            color=edge,
            ha="center",
        )

for col in range(6):
    xx = x0 + col * (cw + gx) + cw / 2
    text(ax, xx, y0 - 0.025, str(col + 1), size=7.4, color=GREY_3, ha="center")
text(ax, 0.047, y0 + 2.5 * (ch + gy) + 0.008, r"$m$", size=8.4, color=GREY_3, ha="right")

ax.plot([0.060, 0.292], [0.420, 0.420], color=GREY_2, lw=0.55)
text(ax, 0.060, 0.386, "Candidate component scores", size=8.6, weight="bold")
text(ax, 0.060, 0.344, r"mean-aligned sign   $a_k$", size=8.7, color=BLUE)
text(ax, 0.060, 0.306, r"signal-to-noise       $u_k=|\overline{J}_k|/\widehat{\sigma}_k$", size=8.7)
text(ax, 0.060, 0.268, r"magnitude               $m_k=|\overline{J}_k|$", size=8.7)

# ---------------------------------------------------------------------------
# (b) Calibration and deployment: threshold axis, branch, and hybrid vector.
panel_heading(ax, 0.350, 0.824, "b", "Calibrate, then deploy")
text(ax, 0.375, 0.758, "Reference-labelled calibration queries", size=9.2, weight="bold")
text(
    ax,
    0.375,
    0.720,
    r"Choose score and $\tau_s$ for the intended loss; abstain if the risk screen fails.",
    size=8.4,
    color=GREY_3,
)

# Score axis for a new design query.
axis_y, axis_x0, axis_x1 = 0.615, 0.385, 0.655
ax.plot([axis_x0, axis_x1], [axis_y, axis_y], color=INK, lw=0.8)
for xx, lab in [(axis_x0, "low"), (axis_x1, "high")]:
    ax.plot([xx, xx], [axis_y - 0.010, axis_y + 0.010], color=INK, lw=0.75)
    text(ax, xx, axis_y - 0.032, lab, size=7.2, color=GREY_3, ha="center", va="top")
tau_x = 0.550
ax.plot([tau_x, tau_x], [axis_y - 0.037, axis_y + 0.047], color=ORANGE, lw=1.15)
text(ax, tau_x, axis_y + 0.060, r"$\tau_s$", size=9.4, color=ORANGE, ha="center")
for xx, marker, face in [
    (0.410, "o", "white"),
    (0.455, "o", ORANGE),
    (0.505, "o", "white"),
    (0.578, "o", BLUE),
    (0.620, "o", BLUE),
    (0.646, "o", "white"),
]:
    ax.scatter([xx], [axis_y], s=30, marker=marker, facecolor=face, edgecolor=INK, linewidth=0.65, zorder=4)
text(ax, 0.500, 0.675, r"deployment scores $s_k$", size=8.2, color=GREY_3, ha="center")

# Straight branch: below threshold is checked; above threshold is retained.
text(ax, 0.424, 0.525, r"$s_k<\tau_s$", size=8.8, weight="bold", color=ORANGE, ha="center")
text(ax, 0.612, 0.525, r"$s_k\geq\tau_s$", size=8.8, weight="bold", color=BLUE, ha="center")
arrow(ax, (0.455, 0.586), (0.430, 0.505), color=ORANGE)
arrow(ax, (0.610, 0.586), (0.610, 0.505), color=BLUE)

ax.add_patch(Rectangle((0.374, 0.405), 0.112, 0.073, facecolor="white", edgecolor=ORANGE, lw=1.0))
text(ax, 0.430, 0.447, "central reference FD", size=9.0, weight="bold", color=ORANGE, ha="center")
text(ax, 0.430, 0.421, "2 simulator evaluations", size=7.4, color=GREY_3, ha="center")
ax.add_patch(Rectangle((0.554, 0.405), 0.112, 0.073, facecolor="white", edgecolor=BLUE, lw=1.0))
text(ax, 0.610, 0.447, r"retain $\overline{J}_k$", size=9.2, weight="bold", color=BLUE, ha="center")
text(ax, 0.610, 0.421, "no simulator call", size=7.4, color=GREY_3, ha="center")

arrow(ax, (0.430, 0.398), (0.486, 0.345), color=ORANGE)
arrow(ax, (0.610, 0.398), (0.553, 0.345), color=BLUE)
text(ax, 0.520, 0.350, "assembled hybrid Jacobian", size=8.7, weight="bold", ha="center", va="bottom")
vx0, vy, vw, vh = 0.437, 0.282, 0.028, 0.047
hybrid = [BLUE, ORANGE, ORANGE, BLUE, BLUE, ORANGE]
for i, col in enumerate(hybrid):
    xx = vx0 + i * (vw + 0.006)
    ax.add_patch(Rectangle((xx, vy), vw, vh, facecolor="white", edgecolor=col, lw=1.2))
    text(ax, xx + vw / 2, vy + vh / 2, r"$\bar J$" if col == BLUE else r"$J^r$",
         size=7.2, color=col, ha="center")

# ---------------------------------------------------------------------------
# (c) Auditable operating point: actual held-out values, not an icon panel.
panel_heading(ax, 0.721, 0.824, "c", r"Measured operating point ($a_k\geq0.9$)")
text(ax, 0.746, 0.752, "benchmark", size=7.7, weight="bold", color=GREY_3)
text(ax, 0.863, 0.752, "coverage", size=7.3, weight="bold", color=GREY_3, ha="center")
text(ax, 0.913, 0.752, "risk", size=7.3, weight="bold", color=GREY_3, ha="center")
text(ax, 0.957, 0.752, "evals/query", size=6.8, weight="bold", color=GREY_3, ha="right")
ax.plot([0.746, 0.958], [0.730, 0.730], color=INK, lw=0.65)

rows = [
    ("TMM near-saturated", "95.9%", "1.5%", "0.90/22"),
    ("TMM stressed", "74.0%", "13.4%", "5.72/22"),
    ("Heat/Poisson stressed", "79.9%", "4.9%", "6.44/32"),
]
for i, (name, cov, risk, cost) in enumerate(rows):
    yy = 0.686 - i * 0.080
    text(ax, 0.746, yy, name, size=8.0)
    text(ax, 0.863, yy, cov, size=8.1, color=BLUE, weight="bold", ha="center")
    text(ax, 0.913, yy, risk, size=8.1, color=ORANGE, weight="bold", ha="center")
    text(ax, 0.957, yy, cost, size=8.1, ha="right")
    ax.plot([0.746, 0.958], [yy - 0.035, yy - 0.035], color=GREY_1, lw=1.0)

text(ax, 0.746, 0.438, "Score choice depends on the loss", size=9.2, weight="bold")
ax.plot([0.746, 0.958], [0.418, 0.418], color=GREY_2, lw=0.55)
text(ax, 0.746, 0.382, "ROC AUC", size=8.0, color=GREY_3)
text(ax, 0.958, 0.382, "SNR: both TMM; magnitude: heat", size=7.8, color=BLUE, ha="right")
text(ax, 0.746, 0.340, "fixed-budget error", size=8.0, color=GREY_3)
text(ax, 0.958, 0.340, "sign: lowest in all three", size=8.0, color=ORANGE, ha="right")
text(ax, 0.746, 0.296, r"$R_{\rm FT}=P(\mathrm{sign\ wrong}\mid\mathrm{accepted})$", size=7.5, color=GREY_3)
text(ax, 0.958, 0.267, "cost column: selective / complete central FD", size=6.8, color=GREY_3, ha="right")

# ---------------------------------------------------------------------------
# One compact scope boundary replaces the former decorative warning banner.
ax.plot([0.035, 0.965], [0.215, 0.215], color=ORANGE, lw=1.15)
text(ax, 0.035, 0.170, "Scope boundary", size=9.2, weight="bold", color=ORANGE)
text(
    ax,
    0.138,
    0.170,
    r"common positive rescaling of all members at fixed $k$ $\Rightarrow$ unchanged sign agreement and SNR; $m_k$ changes",
    size=8.8,
)
text(ax, 0.680, 0.170, r"shared bias $\Rightarrow$ high agreement can still be wrong", size=8.8)
text(
    ax,
    0.035,
    0.105,
    "Evidence screen: 3,580/3,580 external labels passed step/grid checks; unresolved antenna derivatives were excluded.",
    size=8.4,
    color=GREY_3,
)
text(ax, 0.965, 0.105, "calibrated triage  ≠  correctness certificate", size=8.4,
     color=GREY_3, ha="right")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
for ext in ("pdf", "svg", "png"):
    fig.savefig(
        HERE / f"graphical_abstract_simpat.{ext}",
        dpi=600 if ext == "png" else None,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
print("wrote graphical_abstract_simpat.pdf/svg/png")
