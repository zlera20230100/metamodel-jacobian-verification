# -*- coding: utf-8 -*-
# Controlled-spectrum cost--accuracy frontier for the selectively corrected design gradient.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'; ORG = '#e67e22'; NEU = '#404040'
d = np.load(os.path.join(DIR, 'hybrid_gradient.npz'))
K = int(d['K']); tau = float(d['tau'])

fig, a = plt.subplots(figsize=(6.8, 4.3))

# (a) cost-accuracy frontier
c = d['cost_curve']; e = d['err_curve']
# keep distinct (cost,err) points along the frontier
keep = np.concatenate([[True], np.abs(np.diff(c)) > 1e-9])
a.plot(c[keep], e[keep], '-o', color=SIG, ms=5, lw=1.6, zorder=3, label='gate frontier (sweep $\\tau$)')
# endpoints and references
a.scatter([0.0], [float(d['err_allad'])], marker='X', s=110, c=ACC, edgecolors='k', linewidths=0.5, zorder=5,
          label='all-autodiff (0 reference evaluations)')
a.scatter([K], [0.0], marker='s', s=70, c='0.25', edgecolors='k', linewidths=0.5, zorder=5,
          label=f'all-FD ({K} checks = {2*K} evaluations)')
a.scatter([float(d['cost_oracle'])], [float(d['err_oracle'])], marker='*', s=190, c=GRN, edgecolors='k',
          linewidths=0.5, zorder=6, label='reference-informed hybrid')
a.scatter([float(d['op_cost'])], [float(d['op_err'])], marker='o', s=95, facecolors='none', edgecolors=ORG,
          linewidths=2.0, zorder=7, label=f'gate @ $\\tau={tau}$')
a.annotate(f"{100*(1-float(d['op_cost'])/K):.0f}% fewer reference evaluations\nthan all-FD,\n"
           f"{float(d['err_allad'])/float(d['op_err']):.1f}$\\times$ lower error\nthan all-autodiff",
           xy=(float(d['op_cost']), float(d['op_err'])), xytext=(3.05, 0.33), fontsize=8.2, color='0.2',
           arrowprops=dict(arrowstyle='->', color='0.45', lw=1.0,
                           shrinkA=2, shrinkB=4, connectionstyle='arc3,rad=-0.12'))
a.set_xlabel('central-FD component checks (2 reference evaluations each)')
a.set_ylabel('relative error of the design gradient')
a.set_xlim(-0.3, K + 0.3); a.set_ylim(-0.03, 0.58)
a.legend(loc='upper right', frameon=False, fontsize=7.9)
a.set_title('Controlled selective-correction frontier', loc='left', fontsize=9.4, fontweight='bold')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_hybrid_gradient.{ext}'), dpi=320, bbox_inches='tight')
print('wrote fig_hybrid_gradient.pdf/.png')
