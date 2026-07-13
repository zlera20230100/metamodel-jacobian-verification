# -*- coding: utf-8 -*-
"""Risk and intraclass-correlation diagnostics under shared seed bias."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.9,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#3775BA"
ORANGE = "#C76B3C"
MID_GREY = "#9AA3AA"
GREY = "#555555"

d = np.load(os.path.join(HERE, "shared_bias_screen.npz"))
rho = d["rho"]
ft_plain = d["ft_plain"]
ft_global = d["ft_global"]
ft_percomp = d["ft_percomp"]
pearson = float(d["pearson"])
rhohat = d["rhohat_mean"]
rhohat_sd = d["rhohat_sd"]
intercept = float(d["fit_intercept"])

fig, (ax_risk, ax_icc) = plt.subplots(
    1,
    2,
    figsize=(11.0, 4.1),
    gridspec_kw={"width_ratios": [1.18, 1.0]},
)

# (a) Accepted-set sign risk.
ax_risk.plot(
    rho,
    ft_plain,
    "-o",
    color=ORANGE,
    ms=5.0,
    lw=1.8,
    label="plain gate",
)
ax_risk.plot(
    rho,
    ft_global,
    "-^",
    color=MID_GREY,
    ms=5.4,
    lw=1.6,
    label="global-abstain variant",
)
ax_risk.plot(
    rho,
    ft_percomp,
    "-o",
    color=BLUE,
    ms=5.0,
    lw=2.0,
    label="per-component spread screen",
)

for values, colour in (
    (ft_global, MID_GREY),
    (ft_plain, ORANGE),
    (ft_percomp, BLUE),
):
    ax_risk.text(
        rho[-1] + 0.012,
        values[-1],
        f"{values[-1]:.3f}",
        ha="left",
        va="center",
        fontsize=8.0,
        color=colour,
    )

ax_risk.set_xlabel("shared-bias fraction $\\rho$  (seed correlation / ICC)")
ax_risk.set_ylabel("false-trust risk  $P(\\mathrm{sign\\ wrong}\\mid\\mathrm{accepted})$")
ax_risk.set_xlim(-0.02, 0.88)
ax_risk.set_ylim(-0.004, 0.118)
ax_risk.legend(loc="upper left", frameon=False)
ax_risk.set_title("(a) Accepted-set risk under shared bias", loc="left", fontsize=9.4)

# (b) Global ICC as a relative meter.
ax_icc.plot([0, 0.85], [0, 0.85], color=GREY, ls=(0, (5, 4)), lw=1.35, zorder=1)
ax_icc.fill_between(
    rho,
    rhohat - rhohat_sd,
    rhohat + rhohat_sd,
    color=BLUE,
    alpha=0.08,
    zorder=2,
    label="$\\hat{\\rho}$ across-trial $\\pm$1 s.d.",
)
ax_icc.plot(
    rho,
    rhohat,
    "-o",
    color=BLUE,
    ms=5.0,
    lw=2.0,
    zorder=4,
    label="$\\hat{\\rho}$ (global ICC)",
)
ax_icc.text(
    0.03,
    0.965,
    f"Pearson $r={pearson:.3f}$; Spearman $=1.000$\nintercept $={intercept:.2f}$",
    transform=ax_icc.transAxes,
    fontsize=8.1,
    va="top",
    ha="left",
    color="0.25",
)
ax_icc.set_xlabel("true shared-bias fraction $\\rho$")
ax_icc.set_ylabel("estimated $\\hat{\\rho}$  (one-way ensemble ICC)")
ax_icc.set_xlim(-0.02, 0.82)
ax_icc.set_ylim(0.0, 1.0)
ax_icc.legend(loc="lower right", frameon=False)
ax_icc.set_title("(b) Global ICC as a relative meter", loc="left", fontsize=9.4)

fig.tight_layout(pad=0.7)

# Orient the identity label in screen coordinates after layout is fixed.
fig.canvas.draw()
p0 = ax_icc.transData.transform((0.05, 0.05))
p1 = ax_icc.transData.transform((0.78, 0.78))
angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
theta = np.radians(angle)
label_transform = offset_copy(
    ax_icc.transData,
    fig=fig,
    x=3.6 * np.sin(theta),
    y=-3.6 * np.cos(theta),
    units="points",
)
ax_icc.text(
    0.74,
    0.74,
    "identity ($\\hat{\\rho}=\\rho$)",
    transform=label_transform,
    rotation=angle,
    rotation_mode="anchor",
    ha="center",
    va="top",
    fontsize=7.6,
    color=GREY,
    style="italic",
)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_shared_bias_screen.{ext}"), dpi=400, bbox_inches="tight")

print(
    "wrote fig_shared_bias_screen.pdf/.png; "
    f"plain risk={ft_plain[0]:.4f}->{ft_plain[-1]:.4f}; "
    f"spread-screen risk={ft_percomp[0]:.4f}->{ft_percomp[-1]:.4f}; "
    f"Pearson={pearson:.3f}"
)
