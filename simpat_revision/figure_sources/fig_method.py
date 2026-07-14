# -*- coding: utf-8 -*-
# Regenerate the paper figures. Output filenames match the LaTeX source.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle

DIR = os.path.dirname(os.path.abspath(__file__))

# Shared style settings and helpers.
NEU = '#4D4D4D'   # neutral/reference
SIG = '#3775BA'   # Nature-family primary blue
ACC = '#C76B3C'   # single muted warm accent
LIGHT_BLUE = '#A9C5DF'

def apply_style():
    plt.rcParams.update({
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

def panel_label(ax, txt, x=-0.02, y=1.06):
    ax.set_title('')
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', ha='left')

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, name + '.png'), dpi=300)
    fig.savefig(os.path.join(DIR, name + '.pdf'))
    fig.savefig(os.path.join(DIR, name + '.svg'))
    fig.savefig(os.path.join(DIR, name + '.tiff'), dpi=600)
    plt.close(fig)

apply_style()

# fig_method: device stack (a) and PINN input/output schematic (b).
def make_method():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                               gridspec_kw={'width_ratios': [1.0, 1.08]})

    # (a) FPC device stack -------------------------------------------------
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis('off')
    x0, w = 1.1, 6.2
    layers = [
        (0.8, 0.55, NEU,       'ground plane',                   NEU),
        (1.35, 1.5, '#d9c3a0', 'FR4 substrate ($\\varepsilon_r=4.4$)', NEU),
        (2.85, 2.4, '#e6eef7', 'air cavity',                     NEU),
    ]
    for y, h, c, lab, lc in layers:
        a.add_patch(Rectangle((x0, y), w, h, facecolor=c, edgecolor=NEU, lw=1.0))
    # coded metasurface cells (heights encode the per-cell code g_k)
    ytop = 5.35
    ncell = 12
    cw = w / ncell
    hs = 0.40 + 0.50 * (0.5 + 0.5 * np.sin(np.linspace(0, 1, ncell) * np.pi * 1.4))
    for i, hh in enumerate(hs):
        cx = x0 + i * cw + cw * 0.14
        a.add_patch(Rectangle((cx, ytop), cw * 0.72, hh,
                              facecolor=SIG, edgecolor='white', lw=0.5))

    def leader(yc, text, color=NEU):
        # short pointer with an arrowhead aimed back at the layer edge
        a.annotate('', xy=(x0 + w + 0.08, yc), xytext=(x0 + w + 0.5, yc),
                   arrowprops=dict(arrowstyle='-|>', color='#888888', lw=0.9,
                                   shrinkA=0, shrinkB=0,
                                   mutation_scale=8))
        a.text(x0 + w + 0.6, yc, text, va='center', ha='left', fontsize=8.6, color=color)
    leader(0.8 + 0.275, 'ground plane')
    leader(1.35 + 0.75, 'FR4 substrate ($\\varepsilon_r=4.4$)')
    leader(2.85 + 1.2, 'air cavity')
    leader(ytop + 0.55, 'coded 96-cell\nmetasurface ($g_k$)', SIG)
    # probe feed
    xf = x0 + w * 0.5
    a.add_patch(Rectangle((xf - 0.045, 0.0), 0.09, 1.35, facecolor=ACC, edgecolor=ACC))
    a.annotate('probe feed', xy=(xf - 0.05, 0.55), xytext=(x0 - 0.15, 0.55),
               fontsize=8.6, color=ACC, va='center', ha='right',
               arrowprops=dict(arrowstyle='->', color=ACC, lw=1.0))
    # broadside radiation arrow
    a.annotate('', xy=(xf, 9.5), xytext=(xf, 6.7),
               arrowprops=dict(arrowstyle='-|>', color=NEU, lw=1.9))
    a.text(xf + 0.28, 8.2, 'broadside', fontsize=8.8, color=NEU, va='center')
    panel_label(a, '(a)', x=0.0, y=0.99)

    # (b) geometry-conditioned PINN I/O schematic --------------------------
    b.set_xlim(0, 10); b.set_ylim(0, 10); b.axis('off')
    inputs = ['$x$', '$y$', '$z$', '$f$', '$g$']
    iy = np.linspace(7.7, 2.3, len(inputs))
    for s, y in zip(inputs, iy):
        b.add_patch(Rectangle((0.5, y - 0.36), 0.95, 0.72,
                    facecolor='#eef3f9', edgecolor=SIG, lw=1.0))
        b.text(0.975, y, s, ha='center', va='center', fontsize=11)
    b.text(0.975, 8.55, 'inputs', ha='center', fontsize=8.2, color='#777')
    # collect inputs onto a vertical bus, then a single arrow into the PINN.
    # Use one flow style for stubs, the bus, lead-ins, and directional arrows.
    # share the same dark colour, linewidth and arrowhead size, so the whole
    # path reads as a single clean structural flow (no grey-vs-black mismatch).
    FLOW = '#404040'           # one structural-connector colour (matches guides)
    FLW = 1.5                  # one structural-connector linewidth
    FMS = 14                   # one arrowhead size for every -|> connector
    busx = 2.35
    boxr = 0.5 + 0.95          # right edge of the input boxes
    ymid = 5.0                 # bus midpoint (== mean of iy)
    # per-input stubs meet the bus exactly
    for y in iy:
        b.plot([boxr, busx], [y, y], color=FLOW, lw=FLW,
               solid_capstyle='round', zorder=1)
    # vertical bus, drawn slightly past the end stubs so the T-junctions are solid
    b.plot([busx, busx], [iy.min(), iy.max()], color=FLOW, lw=FLW,
           solid_capstyle='round', zorder=1)
    # the lead-in continues the bus toward the PINN as the SAME dark line, then
    # the arrowhead terminates exactly at the PINN box boundary.
    b.plot([busx, 3.0], [ymid, ymid], color=FLOW, lw=FLW,
           solid_capstyle='round', zorder=1)
    b.add_patch(FancyArrowPatch((3.0, ymid), (3.5, ymid),
                arrowstyle='-|>', color=FLOW, lw=FLW, mutation_scale=FMS,
                shrinkA=0, shrinkB=0, zorder=2))
    # network box
    b.add_patch(Rectangle((3.5, 3.0), 3.0, 4.0,
                facecolor='#f5f5f5', edgecolor=NEU, lw=1.3))
    b.text(5.0, 6.15, 'geometry-\nconditioned', ha='center', va='center',
           fontsize=9, color='#555555')
    b.text(5.0, 5.0, 'PINN', ha='center', va='center', fontsize=15, fontweight='bold')
    b.text(5.0, 3.85, r'$\Theta$', ha='center', va='center', fontsize=12, color=SIG)
    # field output
    b.add_patch(Rectangle((7.4, 5.25), 2.35, 1.2,
                facecolor='#eef3f9', edgecolor=SIG, lw=1.0))
    b.text(8.575, 5.85, r'$\mathbf{E}(x,y,z,f;g)$', ha='center', va='center', fontsize=9.5)
    b.plot([6.5, 6.95, 6.95], [5.6, 5.6, 5.85], color=FLOW, lw=FLW,
           solid_capstyle='butt', solid_joinstyle='miter')
    b.add_patch(FancyArrowPatch((6.95, 5.85), (7.4, 5.85), arrowstyle='-|>',
                color=FLOW, lw=FLW, mutation_scale=FMS, shrinkA=0, shrinkB=0))
    # autodiff gradient output
    b.add_patch(Rectangle((7.4, 2.65), 2.35, 1.2,
                facecolor='#fbeceb', edgecolor=ACC, lw=1.0))
    b.text(8.575, 3.25, r'$\partial\mathbf{E}/\partial g$', ha='center', va='center',
           fontsize=11, color=ACC)
    # the single semantic warm accent, with the same linewidth/arrowhead as E
    b.plot([6.5, 6.95, 6.95], [4.4, 4.4, 3.25], color=ACC, lw=FLW,
           solid_capstyle='butt', solid_joinstyle='miter')
    b.add_patch(FancyArrowPatch((6.95, 3.25), (7.4, 3.25), arrowstyle='-|>',
                color=ACC, lw=FLW, mutation_scale=FMS, shrinkA=0, shrinkB=0))
    b.text(8.575, 2.25, 'autodiff (one backward pass)', ha='center', va='top',
           fontsize=7.4, color=ACC)
    panel_label(b, '(b)', x=0.0, y=0.99)

    save(fig, 'fig_method')
    print('fig_method: schematic (no data file); device stack + PINN I/O')

# fig_forward: (a) E-plane pattern, PINN vs openEMS; (b) ten-design S11.
def make_forward():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))

    oe = np.load(os.path.join(DIR, 'fpc_result.npz'), allow_pickle=True)
    th = oe['theta'].copy(); gE = oe['gE']
    if th.min() >= 0 and th.max() > 120:
        th = th - 90.0
    a.plot(th, gE, '-', color=NEU, lw=2.0,
           label=f"openEMS ($D$={float(oe['peakD']):.1f} dBi)")
    hq = np.load(os.path.join(DIR, 'hq_pattern.npz'), allow_pickle=True)
    gp = hq['g']; tp = hq['theta']; pp = hq['phi']
    i0 = int(np.argmin(np.abs(pp - 0))); i180 = int(np.argmin(np.abs(pp - 180)))
    tt = np.concatenate([-tp[::-1], tp])
    gg = np.concatenate([gp[i180][::-1], gp[i0]])
    a.plot(tt, gg, '--', color=SIG, lw=1.8,
           label=f"PINN surrogate (max {float(hq['max_gain']):.1f} dBi)")
    a.set_xlim(-90, 90); a.set_xlabel(r'$\theta$ (deg)')
    a.set_ylabel('directivity (dBi)')
    a.grid(alpha=0.25)
    a.set_ylim(-11, 14)  # headroom so the legend clears the main-beam peak
    a.legend(frameon=False, loc='upper right', fontsize=8.5, borderaxespad=0.15)
    panel_label(a, '(a)')

    cd = np.load(os.path.join(DIR, 'closure_directive.npz'), allow_pickle=True)
    order = ['uniform', 'grad10', 'grad20', 'grad30', 'gstrong20', 'gstrong30',
             'surr_broadside', 'surr_steer10', 'surr_steer20', 'surr_steerm15']
    order = [o for o in order if o + '_s11_24' in cd.files]
    s11 = [float(cd[o + '_s11_24']) for o in order]
    grp = [NEU] + [SIG] * 5 + [ACC] * 4
    grp = grp[:len(order)]
    bars = b.bar(range(len(order)), s11, color=grp, edgecolor='black', linewidth=0.6, width=0.7)
    group_hatches = ['///'] + ['...'] * 5 + ['\\\\'] * 4
    for patch, hatch in zip(bars, group_hatches[:len(order)]):
        patch.set_hatch(hatch)
    mean = np.mean(s11)
    b.axhline(mean, ls=(0, (5, 4)), color='#404040', lw=1.4, zorder=1)
    # label hugs its guide line: right-aligned at the line's right end, ~half a
    # char above it (offset is in screen points, so +3pt is visually ABOVE even
    # though this S11 axis is inverted). ref-line label convention v2.
    from matplotlib.transforms import offset_copy
    tr = offset_copy(b.get_yaxis_transform(), fig=fig, x=-2, y=3, units='points')
    b.text(1.0, mean, f'mean {mean:.1f} dB', transform=tr, ha='right',
           va='bottom', fontsize=7.6, color='#404040', style='italic')
    b.legend(handles=[mpatches.Patch(facecolor=NEU, edgecolor='black', hatch='///', label='uniform'),
                      mpatches.Patch(facecolor=SIG, edgecolor='black', hatch='...', label='phase-gradient codes'),
                      mpatches.Patch(facecolor=ACC, edgecolor='black', hatch='\\\\', label='surrogate codes')],
             fontsize=8, frameon=False, loc='lower center',
             bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.3, handletextpad=0.4)
    b.set_xticks(range(len(order)))
    b.set_xticklabels([str(i + 1) for i in range(len(order))], fontsize=9)
    b.set_xlabel('re-coding index')
    b.set_ylabel('$|S_{11}|$ at 24 GHz (dB)')
    b.set_ylim(min(s11) - 1.2, 0); b.invert_yaxis()
    b.grid(alpha=0.25, axis='y')
    panel_label(b, '(b)')

    save(fig, 'fig_forward')
    print('fig_forward: (a) fpc_result.npz{theta,gE,peakD} + hq_pattern.npz{g,theta,phi,max_gain};',
          '(b) closure_directive.npz{*_s11_24}; s11=', np.round(s11, 2),
          'span', round(max(s11) - min(s11), 2))

# fig_jacobian: (a) ||J_ap|| vs ||J_feed|| per seed; (b) FD vs surrogate gradient.
def make_jacobian():
    ms = np.load(os.path.join(DIR, 'zones_multiseed.npz'), allow_pickle=True)
    fw = np.load(os.path.join(DIR, 'grad_fullwave.npz'), allow_pickle=True)
    nf = ms['norm_feed']; na = ms['norm_ap']; seeds = ms['seeds']
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    x = np.arange(len(seeds)); w = 0.36
    a.bar(x - w / 2, nf, w, color=NEU, label=r'$\Vert J^{\mathrm{feed}}\Vert$ (impedance)')
    a.bar(x + w / 2, na, w, color=SIG, label=r'$\Vert J^{\mathrm{ap}}\Vert$ (aperture)')
    a.set_yscale('log')
    a.set_xticks(x); a.set_xticklabels([f'seed {int(s)}' for s in seeds])
    a.set_ylabel(r'$\Vert J\Vert$')
    a.legend(frameon=False, fontsize=8.5, loc='lower center',
             bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.5, handletextpad=0.4)
    a.grid(alpha=0.25, axis='y', which='both')
    a.text(0.03, 0.92, f'ratio {ms["ratio"].mean():.0f}$\\times$ (mean over seeds)',
           transform=a.transAxes, fontsize=9, color=SIG)
    panel_label(a, '(a)')

    zt = fw['zones']; fd = fw['fd_grad']; sr = fw['surrogate_rela']
    xb = np.arange(len(zt))
    b.bar(xb - w / 2, sr, w, color=SIG, label='surrogate', zorder=2)
    b.bar(xb + w / 2, fd, w, color=ACC, label='full-wave FD', zorder=2)
    b.axhline(0, ls=(0, (5, 4)), color='#404040', lw=1.4, zorder=1)
    b.set_xticks(xb); b.set_xticklabels([f'zone {int(z)}' for z in zt])
    b.set_ylabel(r'$\partial\ln Q_{\mathrm{ap}}/\partial g_k$')
    b.legend(frameon=False, fontsize=8.5)
    b.grid(alpha=0.25, axis='y')
    panel_label(b, '(b)')

    save(fig, 'fig_jacobian')
    print('fig_jacobian: (a) zones_multiseed.npz{norm_ap,norm_feed,seeds};',
          '(b) grad_fullwave.npz{zones,fd_grad,surrogate_rela}; ratio',
          np.round(ms['ratio'], 1), 'sign_stable', bool(ms['sign_stable']))

# fig_inert: overlaid D(f) for a global cell-scale sweep.
# reads movable.npz{FBAND, g0.70__Df ... g1.30__Df}.
def make_inert():
    mv = np.load(os.path.join(DIR, 'movable.npz'), allow_pickle=True)
    fHz = mv['FBAND']
    fGHz = fHz / 1e9
    gs = ['g0.70', 'g0.85', 'g1.00', 'g1.15', 'g1.30']
    gvals = [0.70, 0.85, 1.00, 1.15, 1.30]
    # manuscript blue-neutral-warm sequence across codes
    cols = [LIGHT_BLUE, '#6B9AC4', NEU, '#DCE8F1', ACC]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for g, gv, c in zip(gs, gvals, cols):
        Df = mv[g + '__Df']
        ax.plot(fGHz, Df, '-', color=c, lw=1.8, marker='o', ms=3.2,
                label=f'$g$ = {gv:.2f}')
    ax.axvline(24.0, ls=':', color='#222', lw=1.2)
    ax.annotate('24 GHz', xy=(24.0, ax.get_ylim()[1]), xytext=(24.4, ax.get_ylim()[1]),
                ha='left', va='top', fontsize=9, color='#222')
    ax.set_xlabel('frequency (GHz)')
    ax.set_ylabel('broadside directivity $D(f)$ (dBi)')
    ax.legend(frameon=False, fontsize=8.5, title='cell-scale code',
              title_fontsize=8.5, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, 'fig_inert')
    # peak frequencies per code
    fpks = [float(mv[g + '__fpk']) / 1e9 for g in gs]
    print('fig_inert: movable.npz{FBAND, g0.70..g1.30 __Df}; peak freqs (GHz)=',
          np.round(fpks, 3), '-> all ~24 GHz (near-inert)')

# fig_remedy: horizontal bars of resonance-region PINN phase shift per remedy
# vs the full-wave reference (-193 deg). values taken from the remedy table.
def make_remedy():
    remedies = [
        ('Base (multi-scale Fourier)', 1, SIG),
        ('+ frequency curriculum', 7, SIG),
        ('+ grad-norm adaptive', -10, SIG),
        ('+ curriculum & adaptive', 24, SIG),
        ('+ L-BFGS (unstable)', None, NEU),       # unstable
        ('numerical continuation', 0, SIG),
        ('FBPINN', 3, SIG),
    ]
    ref = -193.0
    labels = [r[0] for r in remedies]
    vals = [r[1] for r in remedies]
    y = np.arange(len(remedies))[::-1]  # top-to-bottom

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for yi, (lab, v, c) in zip(y, remedies):
        if v is None:
            ax.plot(0, yi, 'x', color=NEU, ms=7)
            ax.text(4.5, yi, 'unstable (omitted)', va='center', ha='left',
                    fontsize=9, color=NEU, style='italic')
        else:
            ax.barh(yi, v, color=ACC if abs(v) > 50 else SIG, height=0.6)
            ax.text(v + (1.5 if v >= 0 else -1.5), yi, f'{v:+d}',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
    # full-wave reference
    ax.axvline(ref, color=ACC, ls='--', lw=1.6)
    ax.text(ref + 4, y.max(), f'full-wave reference\n{ref:.0f}$^\\circ$',
            color=ACC, fontsize=9, va='top', ha='left')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('resonance-region PINN phase shift (deg)')
    ax.set_xlim(-210, 60)
    ax.grid(alpha=0.25, axis='x')
    save(fig, 'fig_remedy')
    print('fig_remedy: hard-coded remedy-table values vs full-wave ref',
          ref, 'deg; no remedy approaches the resonance wall')

if __name__ == '__main__':
    make_method()
    make_forward()
    print('DONE: fig_method and fig_forward')
