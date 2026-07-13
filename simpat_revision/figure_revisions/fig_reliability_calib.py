# -*- coding: utf-8 -*-
# fig_reliability_calib (2 panels) from a Monte-Carlo over a controlled reliability spectrum.
# (a) AUC for predicting per-zone reliability vs ensemble size M (paper uses M=10), for sign-agreement and SNR.
# (b) calibration at M=10: ensemble sign-agreement vs the empirical fraction of reliable gradients.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#A9C5DF'; SIG = '#3775BA'; ACC = '#C76B3C'
d = np.load(os.path.join(DIR, 'reliability_calib.npz'))
M = d['M_list']; auc_sign = d['auc_sign']; auc_snr = d['auc_snr']
cc = d['calib_centers']; cf = d['calib_fracs']

fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0))

# (a) AUC vs ensemble size
GUIDE = '#404040'; GUIDE_DASH = (0, (5, 4))
a.plot(M, auc_snr, 'o-', color=GRN, lw=1.9, ms=6, label='ensemble SNR')
a.plot(M, auc_sign, 's--', color=SIG, lw=1.8, ms=6, label='ensemble sign-agreement')
a.axvline(10, color=GUIDE, ls=GUIDE_DASH, lw=1.4)
tr_a = offset_copy(a.get_xaxis_transform(), fig=fig, x=-3, y=2, units='points')
a.text(10, 0.03, 'paper uses 10 seeds', transform=tr_a, rotation=90,
       ha='right', va='bottom', fontsize=7.6, color=GUIDE, style='italic')
a.set_xscale('log')
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
a.xaxis.set_major_locator(FixedLocator(list(M))); a.xaxis.set_minor_locator(NullLocator())
a.xaxis.set_minor_formatter(NullFormatter()); a.set_xticklabels([int(m) for m in M])
a.set_xlabel('ensemble size $M$ (retrained seeds)'); a.set_ylabel('reliability-prediction AUC')
a.set_ylim(0.75, 1.01); a.legend(frameon=False, loc='lower right', fontsize=8.6); a.grid(alpha=0.25, which='both')
a.set_title('(a) controlled-model discrimination versus ensemble size', loc='left', fontsize=9.6, fontweight='bold')

# (b) calibration: sign-agreement vs empirical reliable fraction
b.plot(cc, cf, 'o-', color=GRN, lw=1.9, ms=6)
b.axvline(0.9, color=GUIDE, ls=GUIDE_DASH, lw=1.4)
tr_b = offset_copy(b.get_xaxis_transform(), fig=fig, x=-3, y=2, units='points')
b.text(0.9, 0.03, 'study threshold', transform=tr_b, rotation=90,
       ha='right', va='bottom', fontsize=7.6, color=GUIDE, style='italic')
b.set_xlabel('ensemble sign-agreement ($M{=}10$)'); b.set_ylabel('empirical fraction truly reliable')
b.set_xlim(0.5, 1.0); b.set_ylim(-0.03, 1.03); b.grid(alpha=0.25)
b.set_title('(b) model-specific score--outcome relationship', loc='left', fontsize=9.6, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(DIR, 'fig_reliability_calib.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_reliability_calib.pdf'), bbox_inches='tight')
print('saved fig_reliability_calib; AUC(sign) M=10 =', float(auc_sign[list(M).index(10)]),
      'AUC(snr) M=10 =', float(auc_snr[list(M).index(10)]))
