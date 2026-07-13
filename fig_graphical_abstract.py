# -*- coding: utf-8 -*-
# Elsevier graphical abstract: a "same magnitude, opposite trust" hero contrast over a slim
# solver-free pipeline band.  One cohesive editorial banner.
#
#   TITLE   : bold "Direction, not magnitude" + one-line subtitle.
#   HERO    : "SAME MAGNITUDE, OPPOSITE TRUST" -- each design gradient drawn as a BUNDLE of M
#             equal-length re-trained-seed arrows from one common tail (left TRUST/green: a tidy
#             near-parallel rightward bundle -> signs agree; right VERIFY/red: same-length but
#             splayed, with 2-3 flipped to point left -> signs disagree); check / cross verdict
#             discs; a centre "=" badge proving the two bundles are the SAME size (same arrow
#             length); punch line "same size -- opposite verdict".
#   PIPELINE: a compact left-to-right band -- differentiable surrogate -> design Jacobian
#             dr/dg (free autodiff) -> sign-agreement GATE (tau = 0.9) -> {signs agree: free
#             autodiff, 0 solves | signs disagree: one full-wave solve} -> AUC 0.91 (external) ,
#             SOLVER-FREE payoff.
#
# CORE MESSAGE: which design gradients of a differentiable surrogate to TRUST is set by
# cross-seed SIGN AGREEMENT, not by gradient MAGNITUDE.  Solver-free where seeds agree;
# spend ONE full-wave solve only where they disagree.
#
# Values below are taken from the archived analyses; provenance is documented.
#   hero/payoff = AUC 0.91 predicting per-component gradient SIGN-correctness on an EXTERNAL
#                 transfer-matrix (thin-film TMM) benchmark [0.906, 95% CI [0.833, 0.965];
#                 880 pooled comps -- see AUC_CONFIDENCE_INTERVALS.md].
#   tag         = "solver-free".
#   gate        = split at tau = 0.9 (reliability.npz: thr = 0.9).
#   verdict     = "trusts the impedance gradient, rejects the seed-unstable radiation gradient"
#                 (reliability.npz: sa_resp = 1.0 for impedance/responsive comps -> trusted;
#                  sa_rad = 0.5-0.7 < tau = 0.9 for radiation comps -> flagged. Verified.)
#   We deliberately do NOT headline "63% fewer solves" (scoped to a controlled spectrum).
#
# Self-contained: needs only matplotlib (numbers are quoted constants with provenance above).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle

DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- palette (cohesive, refined -- matched to references) ----------------
INK = '#2c3e50'    # slate ink -- all text
GRN = '#1e7a45'    # trust  (signs agree)
RED = '#d1495b'    # verify (signs disagree) -- soft rose-red
BLU = '#1f5fa6'    # surrogate / neutral nodes
MUT = '#8a97a3'    # muted grey -- secondary marks / connectors
PANEL = '#f5f7f9'  # soft card fill
EDGE = '#dce1e6'   # soft card edge


def tint(c, a):
    """Soft tint: colour c at alpha a (returns rgba tuple)."""
    r = list(mcolors.to_rgba(c)); r[3] = a; return tuple(r)


plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.0,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

# ---- taller canvas so the hero contrast and the pipeline band each get room ----
FW, FH = 11.5, 7.2
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor('white')
# single full-canvas axes in 0..1 coords -> everything placed by hand (no clipping)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# visual aspect: x-units per y-unit (a true circle has width = height / ASP)
ASP = FW / FH


# ============================================================ helpers (drawn marks, nodes)
def draw_check(cx, cy, s, color, lw=2.6, z=14):
    """Hand-drawn check mark (robust; no glyph-tofu risk). s in y-units; x widened by ASP."""
    ax.plot([cx - 0.60 * s / ASP, cx - 0.14 * s / ASP, cx + 0.78 * s / ASP],
            [cy - 0.02 * s, cy - 0.46 * s, cy + 0.50 * s],
            color=color, lw=lw, solid_capstyle='round', solid_joinstyle='round',
            zorder=z, clip_on=False)


def draw_cross(cx, cy, s, color, lw=2.6, z=14):
    """Hand-drawn x mark (robust). s in y-units; x widened by ASP for a true square cross."""
    ax.plot([cx - 0.50 * s / ASP, cx + 0.50 * s / ASP], [cy - 0.50 * s, cy + 0.50 * s],
            color=color, lw=lw, solid_capstyle='round', zorder=z, clip_on=False)
    ax.plot([cx - 0.50 * s / ASP, cx + 0.50 * s / ASP], [cy + 0.50 * s, cy - 0.50 * s],
            color=color, lw=lw, solid_capstyle='round', zorder=z, clip_on=False)


def rounded(x, y, w, h, fc, ec, lw=1.6, rad=0.10, z=2, ls='-'):
    """Rounded card. Transparency lives in `fc` (rgba) only -- no patch-level alpha."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.0,rounding_size={rad}',
                       facecolor=fc, edgecolor=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p); return p


def connect(p0, p1, color=MUT, lw=2.4, z=5, rad=0.0, mut=15, sA=2, sB=2):
    a = FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}',
                        arrowstyle='-|>', mutation_scale=mut, lw=lw, color=color,
                        shrinkA=sA, shrinkB=sB, zorder=z, joinstyle='round', capstyle='round')
    ax.add_patch(a); return a


# ============================================================ TITLE BLOCK
ax.text(0.5, 0.957, 'Direction, not magnitude', ha='center', va='center',
        fontsize=31, fontweight='bold', color=INK)
ax.text(0.5, 0.911,
        'which design gradients of a differentiable surrogate to trust is set by '
        'cross-seed sign agreement, not gradient size',
        ha='center', va='center', fontsize=12.5, color=MUT, style='italic')
# two-stream accent rule under the title (trust-green -> verify-red)
ax.plot([0.300, 0.500], [0.885, 0.885], color=GRN, lw=3.2, solid_capstyle='round', zorder=3)
ax.plot([0.500, 0.700], [0.885, 0.885], color=RED, lw=3.2, solid_capstyle='round', zorder=3)

# ============================================================ HERO BLOCK (dominant focal point)
#   two contrast panels (left = TRUST / green, right = VERIFY / red)
PY0, PY1 = 0.470, 0.855          # panel vertical extent
LPX0, LPX1 = 0.055, 0.487        # left card x-extent
RPX0, RPX1 = 0.513, 0.945        # right card x-extent

for x0, x1, col, al in [(LPX0, LPX1, GRN, 0.085), (RPX0, RPX1, RED, 0.075)]:
    ax.add_patch(FancyBboxPatch((x0, PY0), x1 - x0, PY1 - PY0,
                 boxstyle='round,pad=0.006,rounding_size=0.020',
                 facecolor=tint(col, al), edgecolor=tint(col, 0.95), lw=1.8, zorder=1))

LCx = 0.5 * (LPX0 + LPX1)        # left panel centre x
RCx = 0.5 * (RPX0 + RPX1)        # right panel centre x

# --- geometry of the hero bundles: each gradient = a BUNDLE of M equal-length seed arrows
#     sharing ONE common tail.  Same arrow length on both sides => SAME MAGNITUDE. ---
M = 7
ARR_Y = 0.598                    # baseline y the bundles are centred on
SEED_LEN = 0.150                 # common arrow length (identical on both sides)
TAILx = lambda cx: cx - 0.5 * SEED_LEN   # common tail x (bundle centred in panel)


def seed_bundle(cx, angles_deg, col, clip):
    """A tidy bundle of M equal-length arrows from one common tail, splayed by `angles_deg`
    (deg, 0 = +x = 'positive sign'; ~180 = flipped/'negative sign'). All arrows identical
    length => the bundle's reach encodes the (shared) gradient MAGNITUDE."""
    tail = np.array([TAILx(cx), ARR_Y])
    for ang in angles_deg:
        th = np.deg2rad(ang)
        tip = tail + np.array([SEED_LEN * np.cos(th), SEED_LEN * np.sin(th) * ASP])
        a = FancyArrowPatch(tuple(tail), tuple(tip),
                            arrowstyle='-|>,head_width=0.62,head_length=0.9',
                            mutation_scale=15, lw=2.6, color=col, zorder=6,
                            capstyle='round', joinstyle='round', alpha=0.92)
        a.set_clip_path(clip)
        ax.add_patch(a)
    # common-tail hub disc
    ax.add_patch(Ellipse((tail[0], tail[1]), width=0.016 / ASP, height=0.016,
                 facecolor=col, edgecolor='white', lw=1.4, zorder=8))


# clip the bundles to their own panel card
left_card = ax.patches[0]
right_card = ax.patches[1]

# LEFT: signs agree -> a tight, near-parallel rightward bundle (all arrows point right)
left_ang = np.array([6.0, 3.0, 1.0, -1.0, -3.0, -6.0, 0.0])
seed_bundle(LCx, left_ang, GRN, left_card)
# RIGHT: signs disagree -> same length, splayed, with 3 of 7 clearly FLIPPED to point left
#   flipped arrows kept near-horizontal (~175 deg) so they do NOT shoot up into the text
right_ang = np.array([10.0, 4.0, -6.0, 178.0, -178.0, 172.0, -12.0])
seed_bundle(RCx, right_ang, RED, right_card)

# small explanatory tags under each bundle
ax.text(LCx, ARR_Y - 0.110, r'$M$ re-trained seeds  $\rightarrow$  signs agree',
        ha='center', va='center', fontsize=10.0, color=GRN, style='italic')
ax.text(RCx, ARR_Y - 0.110, r'$M$ re-trained seeds  $\rightarrow$  some signs flip',
        ha='center', va='center', fontsize=10.0, color=RED, style='italic')


# --- verdict badges (check / cross) inside coloured discs, top of each panel ---
def verdict(cx, cy, col, mark):
    ax.add_patch(Ellipse((cx, cy), width=0.060 / ASP, height=0.060,
                 facecolor='white', edgecolor=col, lw=2.6, zorder=9))
    if mark == 'check':
        draw_check(cx, cy, 0.020, col, lw=3.4, z=11)
    else:
        draw_cross(cx, cy, 0.018, col, lw=3.4, z=11)


VY = 0.806
verdict(LCx, VY, GRN, 'check')
verdict(RCx, VY, RED, 'cross')

# --- panel headline labels ---
ax.text(LCx, 0.760, 'TRUST', ha='center', va='center', fontsize=21,
        fontweight='bold', color=GRN, zorder=9)
ax.text(LCx, 0.726, 'use the free autodiff gradient', ha='center', va='center',
        fontsize=11.0, color=INK, zorder=9)
ax.text(LCx, 0.696, r'$\mathbf{0}$  full-wave solves', ha='center', va='center',
        fontsize=11.5, color=GRN, zorder=9, fontweight='bold')

ax.text(RCx, 0.760, 'VERIFY', ha='center', va='center', fontsize=21,
        fontweight='bold', color=RED, zorder=9)
ax.text(RCx, 0.726, 'the sign is unstable', ha='center', va='center',
        fontsize=11.0, color=INK, zorder=9)
ax.text(RCx, 0.696, r'$\mathbf{1}$  full-wave solve', ha='center', va='center',
        fontsize=11.5, color=RED, zorder=9, fontweight='bold')

# central "=" on the equator -- proof the two design gradients have identical size
ax.text(0.5, ARR_Y, r'$=$', ha='center', va='center', fontsize=32, color=INK,
        fontweight='bold', zorder=7,
        bbox=dict(boxstyle='circle,pad=0.16', facecolor='white',
                  edgecolor=tint(INK, 0.25), lw=1.0))

# --- punch line strip just under the hero panels ---
ax.text(0.5, 0.438, 'same size  —  opposite verdict', ha='center', va='center',
        fontsize=15.5, fontweight='bold', color=INK)
ax.text(0.5, 0.404,
        r'trust the $\mathbf{sign\ agreement\ across\ seeds}$, not the magnitude',
        ha='center', va='center', fontsize=11.5, color=INK)

# ============================================================ DIVIDER (hero | pipeline)
ax.plot([0.055, 0.945], [0.368, 0.368], color=tint(INK, 0.18), lw=1.0,
        solid_capstyle='round', zorder=1)

# ============================================================ PIPELINE BAND (slim supporting)
#  tinted strip background so the band reads as a distinct supporting unit
SBY0, SBY1 = 0.038, 0.346
ax.add_patch(FancyBboxPatch((0.040, SBY0), 0.920, SBY1 - SBY0,
             boxstyle='round,pad=0.0,rounding_size=0.014',
             facecolor=PANEL, edgecolor=EDGE, lw=1.2, zorder=0))

# tiny stage caption (top-left of the band)
ax.text(0.058, SBY1 - 0.026, 'THE SOLVER-FREE PIPELINE', ha='left', va='center',
        fontsize=9.5, color=MUT, fontweight='bold')

yc = 0.190                       # pipeline lane centre y

# ---- node 1: differentiable surrogate (with autodiff Jacobian chip) ----
n1x, n1w = 0.062, 0.205
n1h = 0.190; n1y = yc - n1h / 2
rounded(n1x, n1y, n1w, n1h, tint(BLU, 0.09), BLU, lw=1.8, rad=0.022, z=2)
ax.text(n1x + n1w / 2, n1y + n1h - 0.030, 'DIFFERENTIABLE', ha='center', va='center',
        fontsize=9.8, color=BLU, fontweight='bold')
ax.text(n1x + n1w / 2, n1y + n1h - 0.056, 'SURROGATE', ha='center', va='center',
        fontsize=9.8, color=BLU, fontweight='bold')
ax.text(n1x + n1w / 2, n1y + n1h - 0.082, 'PINN / neural operator', ha='center', va='center',
        fontsize=8.6, color=INK)
# Jacobian chip inside
jx, jw = n1x + 0.018, n1w - 0.036
jh = 0.066; jy = n1y + 0.016
rounded(jx, jy, jw, jh, 'white', BLU, lw=1.3, rad=0.018, z=3)
ax.text(jx + jw / 2, jy + jh * 0.70, 'design Jacobian', ha='center', va='center',
        fontsize=8.4, color=INK)
ax.text(jx + jw / 2, jy + jh * 0.28,
        r'$\partial \mathbf{r}/\partial \mathbf{g}$  free  (autodiff)', ha='center',
        va='center', fontsize=9.4, color=BLU, fontweight='bold')

# ---- node 2: the GATE ----
n2x, n2w = 0.330, 0.215
n2h = 0.226; n2y = yc - n2h / 2
rounded(n2x, n2y, n2w, n2h, tint(MUT, 0.13), INK, lw=2.0, rad=0.022, z=2)
ax.text(n2x + n2w / 2, n2y + n2h - 0.028, 'SIGN-AGREEMENT GATE', ha='center', va='center',
        fontsize=10.0, color=INK, fontweight='bold')
# two tidy rows of per-seed SIGN CHIPS: all "+" (agree, green) / alternating "+/-" (disagree)
def sign_chip(cx, cy, sgn, col):
    """One sign rendered inside a small rounded square chip (deliberate, evenly spaced)."""
    cs = 0.026                                   # chip side (y-units); width widened by ASP
    cw = cs / ASP
    ax.add_patch(FancyBboxPatch((cx - cw / 2, cy - cs / 2), cw, cs,
                 boxstyle='round,pad=0.0,rounding_size=0.006',
                 facecolor=tint(col, 0.14), edgecolor=tint(col, 0.85), lw=1.2, zorder=4))
    # draw the glyph as short strokes (consistent weight; minus never looks thin)
    hl = 0.0072                                  # half stroke length (y-units)
    ax.plot([cx - hl / ASP, cx + hl / ASP], [cy, cy], color=col, lw=2.6,
            solid_capstyle='round', zorder=5)    # horizontal bar (both + and -)
    if sgn == '+':
        ax.plot([cx, cx], [cy - hl, cy + hl], color=col, lw=2.6,
                solid_capstyle='round', zorder=5)  # vertical bar -> plus
chip_dx = 0.040                                  # even chip spacing
gxc = n2x + 0.052
ry1 = n2y + n2h - 0.072
ry2 = n2y + n2h - 0.120
for j, sgn in enumerate(['+', '+', '+', '+']):
    sign_chip(gxc + j * chip_dx, ry1, sgn, GRN)
for j, sgn in enumerate(['+', '-', '+', '-']):
    sign_chip(gxc + j * chip_dx, ry2, sgn, RED)
# split node at tau = 0.9
nx, ny = n2x + n2w / 2, n2y + 0.052
ax.add_patch(Ellipse((nx, ny), width=0.040 / ASP, height=0.040,
             facecolor='white', edgecolor=INK, lw=1.8, zorder=6))
ax.text(nx, ny, r'$\tau$', ha='center', va='center', fontsize=11, color=INK,
        fontweight='bold', zorder=7)
ax.text(nx, n2y + 0.012, r'split at  $\tau = 0.9$', ha='center', va='center',
        fontsize=8.8, color=INK, fontweight='bold')

# ---- node 3: two outcomes (stacked) ----
n3x, n3w = 0.598, 0.232
o_h = 0.094
oTy = yc + 0.012                 # TRUST outcome (top)
rounded(n3x, oTy, n3w, o_h, tint(GRN, 0.12), GRN, lw=1.8, rad=0.024, z=2)
draw_check(n3x + 0.030, oTy + o_h / 2 + 0.004, 0.020, GRN, lw=2.8)
ax.text(n3x + 0.062, oTy + o_h - 0.026, 'signs agree  ', ha='left', va='center',
        fontsize=9.8, color=GRN, fontweight='bold')
ax.text(n3x + n3w - 0.012, oTy + o_h - 0.026, 'free autodiff', ha='right', va='center',
        fontsize=9.6, color=GRN, fontweight='bold')
ax.text(n3x + 0.062, oTy + 0.024, r'$\mathbf{0}$  full-wave solves', ha='left', va='center',
        fontsize=9.2, color=INK)

oVy = yc - 0.012 - o_h           # VERIFY outcome (bottom)
rounded(n3x, oVy, n3w, o_h, tint(RED, 0.11), RED, lw=1.8, rad=0.024, z=2)
draw_cross(n3x + 0.030, oVy + o_h / 2 + 0.002, 0.018, RED, lw=2.8)
ax.text(n3x + 0.062, oVy + o_h - 0.026, 'signs disagree', ha='left', va='center',
        fontsize=9.8, color=RED, fontweight='bold')
ax.text(n3x + n3w - 0.012, oVy + o_h - 0.026, 'verify', ha='right', va='center',
        fontsize=9.6, color=RED, fontweight='bold')
ax.text(n3x + 0.062, oVy + 0.024, r'$\mathbf{1}$  full-wave solve', ha='left', va='center',
        fontsize=9.2, color=INK)

# ---- connectors between the three nodes ----
connect((n1x + n1w + 0.004, yc), (n2x - 0.004, yc), color=INK, lw=2.6, mut=16)
connect((n2x + n2w + 0.004, yc + 0.006), (n3x - 0.004, oTy + o_h / 2),
        color=GRN, lw=2.4, rad=-0.18, mut=15)
connect((n2x + n2w + 0.004, yc - 0.006), (n3x - 0.004, oVy + o_h / 2),
        color=RED, lw=2.4, rad=0.18, mut=15)
# branch-arrow labels (agree on the green branch -> TRUST; disagree on the red -> VERIFY)
brx = 0.5 * (n2x + n2w + n3x)                     # midpoint x between gate and outcomes
ax.text(brx, oTy + o_h / 2 + 0.034, 'agree', ha='center', va='center',
        fontsize=8.8, color=GRN, fontweight='bold', style='italic', zorder=8)
ax.text(brx, oVy + o_h / 2 - 0.034, 'disagree', ha='center', va='center',
        fontsize=8.8, color=RED, fontweight='bold', style='italic', zorder=8)

# ---- payoff panel (right end of the band): AUC 0.91 + SOLVER-FREE ----
pbx, pbw = 0.846, 0.100
pbh = 0.270; pby = yc - pbh / 2
rounded(pbx, pby, pbw, pbh, INK, 'none', lw=0, rad=0.024, z=3)
connect((n3x + n3w + 0.004, yc), (pbx - 0.004, yc), color=MUT, lw=2.2, mut=14)
ax.text(pbx + pbw / 2, pby + pbh - 0.044, 'AUC', ha='center', va='center',
        fontsize=11, color='white', fontweight='bold', zorder=5)
ax.text(pbx + pbw / 2, pby + pbh - 0.094, '0.91', ha='center', va='center',
        fontsize=24, color='white', fontweight='bold', zorder=5)
ax.text(pbx + pbw / 2, pby + pbh - 0.142, 'external', ha='center', va='center',
        fontsize=8.2, color=tint('white', 0.78), style='italic', zorder=5)
ax.plot([pbx + 0.014, pbx + pbw - 0.014], [pby + 0.092, pby + 0.092],
        color=tint('white', 0.30), lw=1.0, zorder=5)
ax.text(pbx + pbw / 2, pby + 0.060, 'SOLVER', ha='center', va='center',
        fontsize=10.5, color='white', fontweight='bold', zorder=5)
ax.text(pbx + pbw / 2, pby + 0.030, 'FREE', ha='center', va='center',
        fontsize=10.5, color='white', fontweight='bold', zorder=5)

# ---- Result line at the bottom of the band; colour identifies the case. ----
vy_line = SBY0 + 0.026
ax.text(0.330, vy_line, 'trusts the impedance gradient', ha='left', va='center',
        fontsize=9.6, color=GRN, fontweight='bold')
ax.text(0.604, vy_line, 'rejects the seed-unstable radiation gradient',
        ha='left', va='center', fontsize=9.6, color=RED, fontweight='bold')

# ============================================================ save
out_pdf = os.path.join(DIR, 'fig_graphical_abstract.pdf')
out_png = os.path.join(DIR, 'fig_graphical_abstract.png')
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=150)
plt.close(fig)
try:
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'fig_graphical_abstract (hybrid hero + pipeline): PNG {w}x{h} px -> {DIR}')
except Exception:
    print('fig_graphical_abstract: written')
