# -*- coding: utf-8 -*-
# fig_allocator (single panel) -- the cost-accuracy FRONTIER of the rank-and-allocate solver-budget
# allocator on the controlled spectrum (20000 trials). Relative gradient error vs the number of
# full-wave (FD) solves B = 0..K. Curves: rank-and-allocate (new, by sign-agreement), the fixed-tau
# threshold gate (its operating points -- a VARIABLE-budget special case of the allocator), random
# ordering, and the oracle lower bound. Rank-allocate dominates random (~43% smaller area) and
# closes ~70% of the gap to the oracle; the new value is controllability + the explicit frontier.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'; ORG = '#e67e22'
d = np.load(os.path.join(DIR, 'rank_allocator.npz'))
Bs = d['Bs']; K = int(d['K'])
err_rank = d['err_rank_sa']; err_rand = d['err_random']; err_orac = d['err_oracle']
# threshold-gate operating points (sweep tau): mean cost vs mean error
gcost = d['gate_cost_curve']; gerr = d['gate_err_curve']
op_c = float(d['op_cost_tau']); op_e = float(d['op_err_tau']); op_cstd = float(d['op_cost_std']); tau = float(d['tau'])
# areas under the frontier (lower better)
A_rank = float(d['auc_rank_sa']); A_rand = float(d['auc_random']); A_orac = float(d['auc_oracle'])
area_vs_random = (1.0 - A_rank / A_rand) * 100.0
gap_closed = float(d['mean_closeness']) * 100.0

fig, ax = plt.subplots(figsize=(6.6, 4.6))

# shade the achievable region between oracle (best) and random (worst) for visual reference
ax.fill_between(Bs, err_orac, err_rand, color='0.85', alpha=0.45, zorder=0)

ax.plot(Bs, err_rand, '-s', color=ACC, ms=5, lw=1.6, zorder=3, label='random ordering')
# threshold gate: dedupe the staircase (tau-sweep repeats x), show as discrete operating points
keep = np.concatenate([[True], np.abs(np.diff(gcost)) > 1e-9])
ax.plot(gcost[keep], gerr[keep], ':D', color=SIG, ms=5.5, lw=1.4, zorder=4,
        label='fixed-$\\tau$ accept/verify rule (variable budget)')
ax.plot(Bs, err_rank, '-o', color=GRN, ms=6, lw=2.2, zorder=5, label='rank-and-allocate (proposed)')
ax.plot(Bs, err_orac, '--*', color='0.30', ms=9, lw=1.4, zorder=4, label='reference-informed lower bound')

# Mark the gate's realized operating point and its HORIZONTAL (budget-direction) spread. This is the
# ONLY point with a bar because every other series is plotted at a FIXED, user-chosen budget B (no
# x-spread), whereas the tau-gate's realized budget is a random, data-dependent quantity -- the
# uncontrolled budget the allocator replaces with an exact B. The bar is in x (solves), not y (error).
ax.errorbar([op_c], [op_e], xerr=[[op_cstd], [op_cstd]], fmt='D', color=SIG, mfc=SIG, mec='white',
            mew=0.8, ms=6.5, ecolor=SIG, elinewidth=1.4, capsize=4, zorder=7, alpha=0.95)
ax.annotate(f'gate @ $\\tau={tau}$: realized\n'
            f'budget uncontrolled (bar $=$\n'
            f'$\\pm$std of component checks, $B$):\n'
            f'mean {op_c:.2f} $\\pm$ {op_cstd:.2f} checks',
            xy=(op_c, op_e), xytext=(0.05, 0.075), fontsize=7.3, color=SIG, ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color=SIG, lw=1.1,
                            connectionstyle='arc3,rad=-0.18'))

ax.set_xlabel('central-FD component checks $B$ (each = 2 reference evaluations)')
ax.set_ylabel('relative gradient error  $\\|g-a\\|/\\|a\\|$')
ax.set_xlim(-0.15, K + 0.15); ax.set_ylim(-0.02, 0.55)
ax.set_xticks(range(0, K + 1))
ax.legend(loc='upper right', frameon=False, fontsize=8.2)
ax.text(0.975, 0.77,
        f'rank-allocate: {area_vs_random:.0f}% smaller frontier area than random;\n'
        f'closes $\\sim${gap_closed:.0f}% of the gap to the reference-informed bound.\n'
        f'the threshold gate is its variable-budget special case\n'
        f'(equal at matched budget) $\\Rightarrow$ added value = controllability.',
        transform=ax.transAxes, fontsize=7.5, va='top', ha='right', color='0.2',
        bbox=dict(boxstyle='round,pad=0.4', fc='#eef6f0', ec=GRN, lw=0.7))
ax.set_title('Rank-and-allocate exposes a controllable cost-accuracy frontier',
             loc='left', fontsize=9.6, fontweight='bold')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_allocator.{ext}'), dpi=320, bbox_inches='tight')
print(f'wrote fig_allocator.pdf/.png ; area vs random -{area_vs_random:.1f}% '
      f'(rank {A_rank:.3f} / rand {A_rand:.3f} / oracle {A_orac:.3f}), gap closed ~{gap_closed:.0f}%, '
      f'gate cost {op_c:.2f}+-{op_cstd:.2f}')
