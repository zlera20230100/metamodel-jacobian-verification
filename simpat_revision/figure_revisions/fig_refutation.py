# -*- coding: utf-8 -*-
# Core figure: DIRECTION is one fixed trust criterion across 3 physics families; MAGNITUDE is not
# (which magnitude regime is untrustworthy flips: large on the antenna, small on TMM and heat).
# (a) antenna: the dangerous radiation gradients are large yet sign-unstable -> a magnitude rule trusts them,
#     the direction gate rejects them. (b) direction works on all three oracles; a "trust large gradients"
#     magnitude rule works only where sign-wrong gradients happen to be small (TMM, heat) and inverts on the
#     antenna. All numbers are honest manuscript values; antenna direction AUC=1.00 is the degenerate n=18
#     small-sample value (labelled). No fabricated numbers.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
})
NEU = '#4D4D4D'; SIG = '#3775BA'; ACC = '#C76B3C'; GRN = '#A9C5DF'

R = np.load(os.path.join(DIR, 'reliability.npz'))
mag = R['magnitude']; sa = R['sign_agree']
auc_mag_ant = float(R['auc_mag']); auc_dir_ant = float(R['auc_sign']); thr = float(R['thr'])
resp, null, rad = slice(0, 6), slice(6, 12), slice(12, 18)
dir_auc = {'antenna': auc_dir_ant, 'TMM': 0.906, 'heat': 0.761}  # cross-seed direction gate, 3 oracles

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.15))
from matplotlib.transforms import offset_copy

# ---- Panel (a): antenna scatter, magnitude vs sign-agreement ----
axA.axhline(thr, ls=(0, (5, 4)), lw=1.4, color='#404040', zorder=1)
_trA = offset_copy(axA.get_yaxis_transform(), fig=fig, x=-2, y=3, units='points')
axA.text(1.0, thr, r'acceptance threshold $\tau=0.9$', transform=_trA,
         ha='right', va='bottom', fontsize=7.6, color='#404040', style='italic')
axA.scatter(mag[resp], sa[resp], s=58, marker='o', facecolor=GRN, edgecolor='k', lw=0.6,
            label='reliable (responsive)', zorder=3)
axA.scatter(mag[null], sa[null], s=46, marker='s', facecolor='#c2c2c2', edgecolor='k', lw=0.5,
            label='unreliable (null)', zorder=3)
axA.scatter(mag[rad], sa[rad], s=78, marker='X', facecolor=ACC, edgecolor='k', lw=0.6,
            label='unreliable (radiation)', zorder=4)
axA.set_xscale('log'); axA.set_ylim(0.42, 1.06); axA.set_xlim(5e-3, 30)
axA.set_xlabel(r'per-seed gradient magnitude $|\partial \mathbf{E}/\partial g|$')
axA.set_ylabel('cross-seed sign-agreement')
axA.set_title('(a) 24-GHz antenna: the dangerous gradient is large, yet sign-unstable', fontsize=9.5, loc='left')
axA.annotate('radiation gradients: large magnitude (overlapping the\nresponsive controls) yet sign-unstable, so a\nmagnitude rule accepts them; sign agreement rejects them',
             xy=(float(mag[rad].min()), float(sa[rad].min()) + 0.03), xytext=(0.0075, 0.485),
             ha='left', va='center', fontsize=7.4, color=ACC,
             arrowprops=dict(arrowstyle='->', color=ACC, lw=1.0,
                             shrinkA=4, shrinkB=3, connectionstyle='arc3,rad=-0.12'))
axA.text(0.025, 0.975, f'magnitude AUC {auc_mag_ant:.3f}\ndirection AUC {auc_dir_ant:.2f} (n=18, illustrative)',
         transform=axA.transAxes, ha='left', va='top', fontsize=8.0, color=NEU)
axA.legend(fontsize=7.6, loc='upper center', frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.16),
           handletextpad=0.3, columnspacing=1.0)

# ---- Panel (b): direction is one fixed rule across 3 oracles; magnitude's regime flips ----
oracles = ['24-GHz\nantenna', 'thin-film\nphotonics', '1-D\nheat']
dvals = [dir_auc['antenna'], dir_auc['TMM'], dir_auc['heat']]
x = np.arange(3); w = 0.52
axB.axhline(0.5, ls=(0, (5, 4)), lw=1.4, color='#404040', zorder=1)
_trB = offset_copy(axB.get_yaxis_transform(), fig=fig, x=-2, y=3, units='points')
axB.text(1.0, 0.5, 'chance', transform=_trB,
         ha='right', va='bottom', fontsize=7.6, color='#404040', style='italic')
bars = axB.bar(x, dvals, w, color=SIG, label='cross-retrain sign-agreement score')
axB.set_xticks(x); axB.set_xticklabels(oracles)
axB.set_xlim(-0.6, 2.72)  # extend right so right-aligned 'chance' label clears the 1-D heat bar
axB.set_ylim(0.42, 1.08)
axB.set_ylabel('sign-agreement AUC (predict sign-correctness)')
axB.set_title('(b) one scale-insensitive score; domain-specific risk', fontsize=9.5, loc='left')
for r, v, lab in zip(bars, dvals, ['1.00*', '0.906', '0.761']):
    axB.text(r.get_x() + r.get_width() / 2, v + 0.013, lab, ha='center', fontsize=8.4, color=SIG, fontweight='bold')
# magnitude "trust-large" rule verdict per oracle (the regime flip), below the axis
mag_verdict = [('magnitude inverts', ACC), ('magnitude works', GRN), ('magnitude works', GRN)]
for xi, (txt, col) in zip(x, mag_verdict):
    axB.text(xi, -0.135, txt, ha='center', va='top', fontsize=7.5, color=col, style='italic',
             transform=axB.get_xaxis_transform())
axB.legend(fontsize=7.8, loc='upper right', frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(DIR, 'fig_refutation.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_refutation.png'), dpi=200, bbox_inches='tight')
try:
    from PIL import Image
    w0, h0 = Image.open(os.path.join(DIR, 'fig_refutation.png')).size
    print(f'saved fig_refutation: PNG {w0}x{h0}px | direction AUC antenna/TMM/heat = {dvals} | antenna mag-AUC {auc_mag_ant}')
except Exception:
    print('saved')
