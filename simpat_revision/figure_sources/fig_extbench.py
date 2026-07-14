# -*- coding: utf-8 -*-
# fig_extbench (2 panels) for the thin-film transfer-matrix (TMM) benchmark.
# (a) ROC of the sign-agreement gate predicting whether each surrogate gradient component
#     is sign-correct against the TMM gradient, pooled over query points.
# (b) sign-agreement separated by sign-correct vs sign-wrong components, with the trust threshold.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from sklearn.metrics import roc_curve, roc_auc_score
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#3775BA'; ACC = '#C76B3C'; SIG = '#3775BA'; NEU = '#4D4D4D'
REF = '#404040'; REF_DASH = (0, (5, 4)); REF_LW = 1.4  # reference-line style for chance and threshold
d = np.load(os.path.join(DIR, 'extbench_tmm.npz'))

# Recompute the operational, mean-aligned sign score used in the manuscript.
# The frozen extbench archive stores the modal-vote score; the rebuilt
# member-gradient archive permits a direct modal/mean comparison.
member_candidates = [
    os.path.join(DIR, 'external_members_tmm_easy.npz'),
    os.path.join(DIR, '..', 'external_members_tmm_easy.npz'),
]
member_path = next((p for p in member_candidates if os.path.exists(p)), None)
if member_path is None:
    raise FileNotFoundError('external_members_tmm_easy.npz is required')
dm = np.load(member_path)
g_members = dm['member_gradients']
g_mean = g_members.mean(axis=1)
g_ref = dm['reference_gradients']
sa = (np.sign(g_members) == np.sign(g_mean)[:, None, :]).mean(axis=1).ravel()
corr = (np.sign(g_mean) == np.sign(g_ref)).astype(int).ravel()

auc = float(roc_auc_score(corr, sa)); tau = float(d['tau'])
ntr = float(d['n_trust_mean']); K = int(d['K']); cos_ad = float(d['cos_ad'])

fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={'width_ratios': [1.0, 1.05]})

# (a) ROC -- shaded area shows the AUC; the operating point sits at the trust threshold tau
fpr, tpr, thr = roc_curve(corr, sa)
a.fill_between(fpr, tpr, color=SIG, alpha=0.12, zorder=1)
a.plot(fpr, tpr, '-', color=SIG, lw=2.3, zorder=3, solid_capstyle='round')
a.plot([0, 1], [0, 1], color=REF, ls=REF_DASH, lw=REF_LW, zorder=2)
j = int(np.argmin(np.abs(thr - tau)))
a.plot(fpr[j], tpr[j], 'o', ms=7.5, mfc='white', mec=SIG, mew=1.8, zorder=4)
a.annotate(rf'$\tau={tau}$', xy=(fpr[j], tpr[j]),
           xytext=(fpr[j] + 0.155, tpr[j] - 0.155),
           fontsize=9.0, color=SIG, va='center', ha='left',
           arrowprops=dict(arrowstyle='->', color=SIG, lw=0.9,
                           shrinkA=1, shrinkB=4))
a.text(0.96, 0.07, f'AUC $=$ {auc:.3f}', ha='right', va='bottom', fontsize=12,
       color=SIG, fontweight='bold')
a.set_xlabel('false positive rate'); a.set_ylabel('true positive rate')
a.set_xlim(0, 1); a.set_ylim(0, 1.02)
a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
a.set_title('(a) ROC on the external TMM benchmark',
            loc='left', fontsize=9.0, fontweight='bold')

# (b) separation -- trusted band, jittered points, and a coloured mean line per group
b.axhspan(tau, 1.03, color=GRN, alpha=0.06, zorder=0)
b.axhline(tau, color=REF, ls=REF_DASH, lw=REF_LW, zorder=2)
# horizontal guide-line convention: right-aligned, hugging the line ~3pt above
b_thr_tr = offset_copy(b.get_yaxis_transform(), fig=fig, x=-2, y=3, units='points')
b.text(1.0, tau, rf'acceptance threshold $\tau={tau}$', transform=b_thr_tr,
       ha='right', va='bottom', fontsize=7.8, fontstyle='italic', color=REF, zorder=6)
rng = np.random.default_rng(0)
grp = [('sign-correct', sa[corr == 1], GRN), ('sign-wrong', sa[corr == 0], ACC)]
for i, (lab, vals, col) in enumerate(grp):
    if not len(vals):
        continue
    b.scatter(i + 0.085 * rng.standard_normal(len(vals)), vals, s=18, c=col,
              edgecolors='none', alpha=0.5, zorder=3)
    m = float(vals.mean())
    b.plot([i - 0.24, i + 0.24], [m, m], '-', color=col, lw=3.0, zorder=5, solid_capstyle='round')
    b.annotate(f'mean {m:.2f}', xy=(i + 0.24, m), xytext=(i + 0.30, m),
               va='center', ha='left', fontsize=8.3, color=col, fontweight='bold')
b.set_xticks([0, 1]); b.set_xticklabels([g[0] for g in grp]); b.set_xlim(-0.5, 1.55)
b.set_ylabel('mean-aligned sign score (10 seeds)'); b.set_ylim(0.42, 1.03)
b.text(0.5, 0.028,
       rf'$\tau={tau}$: ${ntr:.1f}/{K}$ accepted; correcting rejects costs 0.90 evaluations/query '
       rf'(cosine {cos_ad:.3f})',
       transform=b.transAxes, fontsize=7.4, va='bottom', ha='center', color=NEU)
b.set_title('(b) reliable vs. unreliable components', loc='left',
            fontsize=9.0, fontweight='bold')

fig.tight_layout()

# diagonal ref-line label: rotate to match the rendered y=x slope, sit just below it,
# ~half-char gap, near the upper-right (consistent with Fig 9).
fig.canvas.draw()  # finalise layout so transData reflects the rendered axes
_p0 = a.transData.transform((0.0, 0.0)); _p1 = a.transData.transform((1.0, 1.0))
_ang = float(np.degrees(np.arctan2(_p1[1] - _p0[1], _p1[0] - _p0[0])))
_rad = np.radians(_ang); _gap = 4.0  # ~half-char gap, perpendicular below the line (pts)
_chk = offset_copy(a.transData, fig=fig, x=_gap * np.sin(_rad), y=-_gap * np.cos(_rad),
                   units='points')
XM = 0.78
a.text(XM, XM, 'chance', transform=_chk, rotation=_ang, rotation_mode='anchor',
       ha='center', va='top', fontsize=7.6, color=REF, fontstyle='italic', zorder=6)

for ext in ('pdf', 'svg', 'png', 'tiff'):
    dpi = 300 if ext == 'png' else (600 if ext == 'tiff' else None)
    fig.savefig(os.path.join(DIR, f'fig_extbench.{ext}'), dpi=dpi, bbox_inches='tight')
print('wrote fig_extbench.pdf/.png ; AUC', round(auc, 3), 'trusted', round(ntr, 1), '/', K)
