# -*- coding: utf-8 -*-
"""Publication figure: external score discrimination versus budgeted-gradient loss."""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AUC_CSV = HERE / "external_baseline_auc.csv"
AREA_CSV = HERE / "external_baseline_allocator_summary.csv"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

datasets = [
    "TMM easy (near-saturated)",
    "TMM stressed",
    "Heat/Poisson stressed",
]
dataset_labels = ["TMM near", "TMM stressed", "Heat/Poisson"]
methods = ["mean_aligned_sign", "snr", "magnitude"]
method_labels = ["mean-aligned sign", "SNR", "magnitude"]
colors = ["#0F4D92", "#8BCF8B", "#E9A6A1"]
hatches = ["///", "...", "\\\\"]

auc = pd.read_csv(AUC_CSV).set_index(["dataset", "method"])
area = pd.read_csv(AREA_CSV).set_index(["dataset", "method"])

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.15), gridspec_kw={"wspace": 0.28})
x = np.arange(len(datasets), dtype=float)
width = 0.24

ax = axes[0]
for j, (method, label, color, hatch) in enumerate(zip(methods, method_labels, colors, hatches)):
    vals = np.array([auc.loc[(d, method), "auc"] for d in datasets], dtype=float)
    lo = np.array([auc.loc[(d, method), "auc_cluster_low"] for d in datasets], dtype=float)
    hi = np.array([auc.loc[(d, method), "auc_cluster_high"] for d in datasets], dtype=float)
    pos = x + (j - 1) * width
    ax.bar(pos, vals - 0.5, bottom=0.5, width=width, color=color, edgecolor="black",
           linewidth=0.7, hatch=hatch, label=label, zorder=3)
    ax.errorbar(pos, vals, yerr=np.vstack([vals - lo, hi - vals]), fmt="none", color="black",
                elinewidth=0.9, capsize=2.5, zorder=4)
    for px, value in zip(pos, vals):
        ax.text(px, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5,
                rotation=90)
ax.axhline(0.5, color="#555555", lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(0.01, 0.505, "chance", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=7.7, color="#555555")
ax.set_ylim(0.5, 1.03)
ax.set_ylabel("sign-correctness AUC (cluster 95% CI)")
ax.set_xticks(x, dataset_labels)
ax.set_title("(a) Classification ranking favors SNR or magnitude", loc="left", fontweight="bold")

ax = axes[1]
for j, (method, label, color, hatch) in enumerate(zip(methods, method_labels, colors, hatches)):
    vals = np.array([area.loc[(d, method), "allocator_relerr_area"] for d in datasets], dtype=float)
    lo = np.array([area.loc[(d, method), "allocator_relerr_area_cluster_low"] for d in datasets], dtype=float)
    hi = np.array([area.loc[(d, method), "allocator_relerr_area_cluster_high"] for d in datasets], dtype=float)
    pos = x + (j - 1) * width
    ax.bar(pos, vals, width=width, color=color, edgecolor="black", linewidth=0.7,
           hatch=hatch, zorder=3)
    ax.errorbar(pos, vals, yerr=np.vstack([vals - lo, hi - vals]), fmt="none", color="black",
                elinewidth=0.9, capsize=2.5, zorder=4)
    for px, value in zip(pos, vals):
        ax.text(px, value + 0.022, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5,
                rotation=90)
ax.set_ylim(0.0, 0.90)
ax.set_ylabel("normalized error-frontier area\n(lower is better; cluster 95% CI)")
ax.set_xticks(x, dataset_labels)
ax.set_title("(b) Fixed-budget gradient loss favors sign ordering", loc="left", fontweight="bold")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
           columnspacing=1.8, handlelength=2.0)
fig.tight_layout(rect=(0, 0, 1, 0.93), pad=1.2)

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_external_score_comparison.{ext}", dpi=400, bbox_inches="tight")
print("wrote fig_external_score_comparison.pdf/.png")
