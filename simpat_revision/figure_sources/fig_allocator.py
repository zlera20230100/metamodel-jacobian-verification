# -*- coding: utf-8 -*-
"""Fixed- and variable-budget gradient-correction frontiers."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        "legend.fontsize": 8.3,
        "axes.linewidth": 0.9,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#3775BA"
ORANGE = "#C76B3C"
GREEN = "#6B9AC4"
GREY = "#555555"

d = np.load(os.path.join(HERE, "rank_allocator.npz"))
budgets = d["Bs"]
k = int(d["K"])
err_rank = d["err_rank_sa"]
err_random = d["err_random"]
err_oracle = d["err_oracle"]
gate_cost = d["gate_cost_curve"]
gate_error = d["gate_err_curve"]
op_cost = float(d["op_cost_tau"])
op_error = float(d["op_err_tau"])
op_cost_sd = float(d["op_cost_std"])
tau = float(d["tau"])

area_rank = float(d["auc_rank_sa"])
area_random = float(d["auc_random"])
area_oracle = float(d["auc_oracle"])

fig, ax = plt.subplots(figsize=(6.6, 4.35))

# The shaded interval is the observed random-to-oracle gap.
ax.fill_between(budgets, err_oracle, err_random, color="#E5E5E5", alpha=0.65, zorder=0)
ax.plot(
    budgets,
    err_random,
    "-s",
    color=ORANGE,
    ms=5.0,
    lw=1.6,
    label="random ordering",
    zorder=3,
)

# Repeated costs occur in the threshold sweep; show one point per realised cost.
keep = np.concatenate([[True], np.abs(np.diff(gate_cost)) > 1e-9])
ax.plot(
    gate_cost[keep],
    gate_error[keep],
    ":D",
    color=GREEN,
    ms=5.2,
    lw=1.35,
    label="fixed-$\\tau$ rule (variable budget)",
    zorder=4,
)
ax.plot(
    budgets,
    err_rank,
    "-o",
    color=BLUE,
    ms=5.7,
    lw=2.0,
    label="rank-and-allocate (fixed budget)",
    zorder=5,
)
ax.plot(
    budgets,
    err_oracle,
    "--*",
    color=GREY,
    ms=8.5,
    lw=1.35,
    label="oracle lower bound (diagnostic)",
    zorder=4,
)

# Only the threshold rule has a variable realised budget; the other curves use fixed B.
ax.errorbar(
    [op_cost],
    [op_error],
    xerr=[[op_cost_sd], [op_cost_sd]],
    fmt="D",
    color=GREEN,
    mfc="white",
    mec=GREEN,
    mew=1.2,
    ms=6.0,
    ecolor=GREEN,
    elinewidth=1.2,
    capsize=3.5,
    zorder=7,
)
# Explain the hollow operating-point marker in an unused corner.  A leader line
# or an adjacent label would cross the surrounding frontiers at this location.
ax.text(
    0.02,
    0.055,
    f"hollow diamond: $\\tau={tau}$ operating point\n"
    f"horizontal bar: {op_cost:.1f} $\\pm$ {op_cost_sd:.1f} realised checks",
    transform=ax.transAxes,
    color=GREEN,
    fontsize=7.8,
    ha="left",
    va="bottom",
    zorder=8,
)

ax.set_xlabel("central-FD component checks $B$  (2 reference evaluations each)")
ax.set_ylabel("relative gradient error  $\\|g-a\\|/\\|a\\|$")
ax.set_xlim(-0.15, k + 0.15)
ax.set_ylim(-0.02, 0.55)
ax.set_xticks(range(k + 1))
ax.legend(loc="upper right", frameon=False)

fig.tight_layout(pad=0.7)
for ext in ("pdf", "svg", "png", "tiff"):
    dpi = 300 if ext == "png" else (600 if ext == "tiff" else None)
    fig.savefig(os.path.join(HERE, f"fig_allocator.{ext}"), dpi=dpi, bbox_inches="tight")

print(
    "wrote fig_allocator.pdf/.png; "
    f"areas rank/random/oracle={area_rank:.3f}/{area_random:.3f}/{area_oracle:.3f}; "
    f"gate cost={op_cost:.2f}+-{op_cost_sd:.2f}"
)
