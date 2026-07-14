# -*- coding: utf-8 -*-
"""External-domain ROC comparison using the mean-aligned sign score."""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from sklearn.metrics import roc_curve, roc_auc_score

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.9,
})

DIR = os.path.dirname(os.path.abspath(__file__))
GRN = '#6B9AC4'
SIG = '#3775BA'
REF = '#404040'
REF_DASH = (0, (5, 4))


def find_member_archive(filename):
    candidates = [
        os.path.join(DIR, filename),
        os.path.join(DIR, '..', filename),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(filename)
    return path


def operational_arrays(filename):
    """Return mean-aligned agreement and correctness for one benchmark."""
    d = np.load(find_member_archive(filename))
    g_members = d['member_gradients']
    g_mean = g_members.mean(axis=1)
    g_ref = d['reference_gradients']
    score = (np.sign(g_members) == np.sign(g_mean)[:, None, :]).mean(axis=1)
    correct = (np.sign(g_mean) == np.sign(g_ref)).astype(int)
    return score.ravel(), correct.ravel()


sa_t, cor_t = operational_arrays('external_members_tmm_easy.npz')
sa_p, cor_p = operational_arrays('external_members_heat_poisson_stressed.npz')
auc_t = float(roc_auc_score(cor_t, sa_t))
auc_p = float(roc_auc_score(cor_p, sa_p))
tau = 0.9
nw = int((cor_p == 0).sum())
nc = int((cor_p == 1).sum())

# Query-clustered 95% CIs reported in the manuscript.
lo_t, hi_t = 0.841, 0.963
lo_p, hi_p = 0.729, 0.793

fig, (a, b) = plt.subplots(
    1, 2, figsize=(10.6, 4.2), gridspec_kw={'width_ratios': [1.05, 0.95]}
)

# (a) ROC curves.
fpr_t, tpr_t, _ = roc_curve(cor_t, sa_t)
fpr_p, tpr_p, thr_p = roc_curve(cor_p, sa_p)
a.plot([0, 1], [0, 1], color=REF, linestyle=REF_DASH, lw=1.4, zorder=2)
a.plot(
    fpr_t, tpr_t, '-', color=SIG, lw=2.3, zorder=3,
    solid_capstyle='round', label=f'resonance (TMM): AUC {auc_t:.3f}'
)
a.fill_between(fpr_p, tpr_p, color=GRN, alpha=0.10, zorder=1)
a.plot(
    fpr_p, tpr_p, '-', color=GRN, lw=2.3, zorder=4,
    solid_capstyle='round', label=f'diffusion (heat): AUC {auc_p:.3f}'
)
j = int(np.argmin(np.abs(thr_p - tau)))
a.plot(fpr_p[j], tpr_p[j], 'o', ms=7.5, mfc='white', mec=GRN, mew=1.8, zorder=5)
a.annotate(
    rf'$\tau={tau}$', xy=(fpr_p[j], tpr_p[j]),
    xytext=(fpr_p[j] + 0.17, tpr_p[j] - 0.16), fontsize=8.5,
    color=GRN, va='center', arrowprops=dict(arrowstyle='-', color=GRN, lw=0.8)
)
a.set_xlabel('false positive rate')
a.set_ylabel('true positive rate')
a.set_xlim(0, 1)
a.set_ylim(0, 1.02)
a.set_xticks([0, 0.5, 1])
a.set_yticks([0, 0.5, 1])
a.legend(loc='lower right', frameon=False, fontsize=8.2)
a.set_title(
    '(a) above-chance ranking in diffusion, weaker than resonance',
    loc='left', fontsize=9.0, fontweight='bold'
)

# (b) query-clustered intervals.
labels = ['diffusion\n(heat)', 'resonance\n(TMM)']
aucs = [auc_p, auc_t]
los = [lo_p, lo_t]
his = [hi_p, hi_t]
cols = [GRN, SIG]
xb = np.arange(2)
b.bar(xb, aucs, color=cols, edgecolor='k', lw=0.6, width=0.58, zorder=3)
b.errorbar(
    xb, aucs,
    yerr=[np.array(aucs) - np.array(los), np.array(his) - np.array(aucs)],
    fmt='none', ecolor='k', elinewidth=1.1, capsize=5, zorder=5
)
for i, (value, lo, hi) in enumerate(zip(aucs, los, his)):
    b.text(
        i, hi + 0.02, f'{value:.3f}\n[{lo:.3f}, {hi:.3f}]',
        ha='center', va='bottom', fontsize=8.2
    )
b.axhline(0.5, color=REF, linestyle=REF_DASH, lw=1.4, zorder=2)
trb = offset_copy(b.get_yaxis_transform(), fig=fig, x=-2, y=3, units='points')
b.text(
    1.0, 0.5, 'chance', transform=trb, ha='right', va='bottom',
    fontsize=7.6, color=REF, style='italic'
)
b.set_xticks(xb)
b.set_xticklabels(labels, fontsize=8.6)
b.set_ylabel(r'reliability AUC  (mean-aligned sign score $\to$ sign-correct)')
b.set_ylim(0.0, 1.15)
b.set_xlim(-0.6, 1.6)
b.set_title(
    '(b) above chance in both, but diffusion is weaker',
    loc='left', fontsize=9.0, fontweight='bold'
)

fig.text(
    0.74, -0.02,
    f'diffusion class balance: {nw} sign-wrong / {nc} correct;  '
    f'both CIs exclude chance.\nThe ranking remains informative, with physics-dependent strength; '
    f'the verification payoff is physics-dependent.',
    fontsize=7.6, va='top', ha='center', color='0.3'
)

fig.tight_layout()
fig.canvas.draw()
p0 = a.transData.transform((0.0, 0.0))
p1 = a.transData.transform((1.0, 1.0))
ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
rad = np.radians(ang)
gap = 4.0
chk = offset_copy(
    a.transData, fig=fig, x=gap * np.sin(rad), y=-gap * np.cos(rad), units='points'
)
a.text(
    0.78, 0.78, 'chance', transform=chk, rotation=ang,
    rotation_mode='anchor', ha='center', va='top', fontsize=7.6,
    color=REF, fontstyle='italic', zorder=6
)

for ext in ('pdf', 'svg', 'png', 'tiff'):
    dpi = 300 if ext == 'png' else (600 if ext == 'tiff' else None)
    fig.savefig(os.path.join(DIR, f'fig_extdomain.{ext}'), dpi=dpi, bbox_inches='tight')
print(
    f'wrote fig_extdomain.pdf/.png ; heat AUC {auc_p:.3f} '
    f'[{lo_p:.3f}, {hi_p:.3f}] ({nw} wrong / {nc} correct) vs '
    f'TMM AUC {auc_t:.3f} [{lo_t:.3f}, {hi_t:.3f}]'
)
