# -*- coding: utf-8 -*-
"""Regenerate Fig. Jacobian with the fixed-step openEMS result labelled as unresolved."""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
NEU = '#444444'
SIG = '#3775BA'
ACC = '#C76B3C'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
})


def panel_label(ax, text, x=-0.02, y=1.06):
    ax.set_title('')
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=12,
        fontweight='bold', va='bottom', ha='left'
    )


ms = np.load(os.path.join(DIR, 'zones_multiseed.npz'), allow_pickle=True)
fw = np.load(os.path.join(DIR, 'grad_fullwave.npz'), allow_pickle=True)
nf = ms['norm_feed']
na = ms['norm_ap']
seeds = ms['seeds']

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
x = np.arange(len(seeds))
w = 0.36
a.bar(x - w / 2, nf, w, color=NEU, edgecolor='black', linewidth=0.6,
      hatch='///', label=r'$\Vert J^{\mathrm{feed}}\Vert$ (impedance)')
a.bar(x + w / 2, na, w, color=SIG, edgecolor='black', linewidth=0.6,
      hatch='...', label=r'$\Vert J^{\mathrm{ap}}\Vert$ (aperture)')
a.set_yscale('log')
a.set_xticks(x)
a.set_xticklabels([f'seed {int(seed)}' for seed in seeds])
a.set_ylabel(r'$\Vert J\Vert$')
a.legend(
    frameon=False, fontsize=8.5, loc='lower center',
    bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.5, handletextpad=0.4
)
a.grid(alpha=0.25, axis='y', which='both')
a.text(
    0.03, 0.92, f'ratio {ms["ratio"].mean():.0f}$\\times$ (mean over seeds)',
    transform=a.transAxes, fontsize=9, color=SIG
)
panel_label(a, '(a)')

zones = fw['zones']
fd = fw['fd_grad']
surrogate = fw['surrogate_rela']
xb = np.arange(len(zones))
b.bar(xb - w / 2, surrogate, w, color=SIG, edgecolor='black', linewidth=0.6,
      hatch='...', label='surrogate', zorder=2)
b.bar(
    xb + w / 2, fd, w, color=ACC, edgecolor='black', linewidth=0.6, hatch='\\\\',
    label=r'openEMS FD ($h=0.05$; unresolved)', zorder=2
)
b.axhline(0, ls=(0, (5, 4)), color='#404040', lw=1.4, zorder=1)
b.set_xticks(xb)
b.set_xticklabels([f'zone {int(zone)}' for zone in zones])
b.set_ylabel(r'$\partial\ln Q_{\mathrm{ap}}/\partial g_k$')
b.legend(frameon=False, fontsize=8.5)
b.grid(alpha=0.25, axis='y')
panel_label(b, '(b)')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_jacobian.{ext}'), dpi=320, bbox_inches='tight')
print('wrote fig_jacobian.pdf/.png with openEMS h=0.05 labelled unresolved')
