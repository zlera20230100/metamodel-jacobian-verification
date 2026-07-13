# -*- coding: utf-8 -*-
# Second EXTERNAL benchmark for the gradient-reliability gate, in a STRUCTURALLY DIFFERENT physics
# family from the antenna (resonant EM) and thin-film (Fabry-Perot) cases: 1-D steady-state heat
# conduction -- an elliptic DIFFUSION / Poisson boundary-value problem. No resonance, no photonics;
# the operator is self-adjoint, positive-definite and non-oscillatory, the opposite character to a
# Fabry-Perot flank. This tests whether the deep-ensemble sign-agreement gate generalises across
# physics families, not just across instances of the same (resonant) family.
#
# Design problem (inverse conductivity design):
#   Physics: steady 1-D Fourier conduction on x in [0,1] with an internal source q(x):
#       -d/dx( g(x) dT/dx ) = q(x),   T(0)=T_L, T(1)=T_R   (Dirichlet)
#   The bar is split into K segments; the design vector is the per-segment conductivity g in R^K
#   (g_k > 0). The ORACLE is an EXACT tridiagonal BVP solve on a fine grid (finite volume with
#   harmonic-mean face conductivity -> SPD tridiagonal system -> Thomas algorithm). One "solve" =
#   one such BVP solve, the analogue of one TMM / full-wave evaluation in the other benchmarks.
#   Objective (MINIMISED by the designer) = profile-matching MSE against a target field:
#       J(g) = mean_i ( T_i(g) - T_target_i )^2 .
#   Design gradient dJ/dg in R^K. GROUND TRUTH = central finite difference of J through the oracle
#   BVP solve (relative step; 2K solves). The surrogate predicts J(g) directly; its autodiff dJ/dg
#   is what the gate must vet.
#
# REGIME NOTE (honest): a smooth, well-conditioned conduction design is *too easy* for this test --
# a cheap MLP already gets ~100% of gradient-component SIGNS right, so there are no sign-wrong
# components, the gate AUC is undefined, and the allocator has nothing to save. (We log this easy
# regime for the record.) The MAIN benchmark therefore deliberately stresses the problem into a
# discriminating regime where a non-trivial fraction of components are sign-wrong, exactly as the
# companion extbench_tmm_hard.py does for the photonics case: high-contrast (log-normal)
# conductivities, a sharp sign-CHANGING internal source, MANY segments (K=16) so each segment's
# influence is small/sign-fragile, FEW training samples (NS=70), and query points that roam to
# +-25% (mild extrapolation). The surrogate recipe (tiny MLP K->128^3->1 SiLU), deep ensemble
# (M=10 fixed seeds), per-component sign-agreement gate sa=max(frac>0,frac<0), pooled AUC of
# "sign-agreement predicts AD-component-sign-correct vs the oracle", and the least-trusted-first
# allocator (cosine-to-truth vs #solves for gate / random / oracle ordering, solves-saved) are all
# reused verbatim from extbench_tmm.py. Writes extbench_poisson.npz.  CPU-only, fixed seeds.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(6)
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------------------------
# Oracle: exact 1-D conduction BVP solve (finite volume, harmonic-mean face conductivity, Thomas).
# ----------------------------------------------------------------------------------------------
NGRID = 160                                       # interior unknowns of the fine FV grid (oracle)
TL, TR = 1.0, 0.0                                 # Dirichlet end temperatures
XC = (np.arange(NGRID) + 0.5) / NGRID             # cell centres in (0,1)
H = 1.0 / NGRID                                    # cell width


def build_problem(K, q_sharp, gtrue_seed):
    """Return (seg, Q, G_TRUE, T_TARGET) for a K-segment conduction design problem.
    q_sharp>1 sharpens a sign-CHANGING internal source (heater near x=0.3, sink near x=0.7);
    G_TRUE is a high-contrast log-normal conductivity profile that generates the target field."""
    seg = np.minimum((XC * K).astype(int), K - 1)          # fine cell -> coarse segment index
    Q = (40.0 * np.exp(-((XC - 0.30) / (0.06 + 0.14 / q_sharp)) ** 2)
         - 25.0 * np.exp(-((XC - 0.70) / 0.08) ** 2))      # sign-changing source
    rng = np.random.default_rng(gtrue_seed)
    g_true = np.exp(rng.normal(0.0, 0.8, K))               # high-contrast (log-normal) conductivities
    return seg, Q, g_true


def make_solver(seg, Q):
    def conduction_solve(gseg):
        """Exact steady 1-D conduction solve. gseg: (K,) segment conductivities (>0).
        Returns T at the NGRID interior cell centres. SPD tridiagonal -> Thomas algorithm."""
        gcell = np.asarray(gseg, dtype=float)[seg]                       # per-cell conductivity
        gf = 2.0 * gcell[:-1] * gcell[1:] / (gcell[:-1] + gcell[1:])     # harmonic-mean faces
        inv_h2 = 1.0 / (H * H)
        n = Q.size
        lower = np.zeros(n); diag = np.zeros(n); upper = np.zeros(n); b = Q.copy()
        lower[1:] = -gf * inv_h2; diag[1:] += gf * inv_h2
        upper[:-1] = -gf * inv_h2; diag[:-1] += gf * inv_h2
        diag[0] += 2.0 * gcell[0] * inv_h2;  b[0] += 2.0 * gcell[0] * inv_h2 * TL
        diag[-1] += 2.0 * gcell[-1] * inv_h2; b[-1] += 2.0 * gcell[-1] * inv_h2 * TR
        cp = upper.copy(); dp = b.copy(); cp[0] /= diag[0]; dp[0] /= diag[0]
        for i in range(1, n):
            m = diag[i] - lower[i] * cp[i - 1]
            cp[i] = upper[i] / m
            dp[i] = (b[i] - lower[i] * dp[i - 1]) / m
        T = np.empty(n); T[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            T[i] = dp[i] - cp[i] * T[i + 1]
        return T
    return conduction_solve


def run_benchmark(tag, K, q_sharp, NS, span, roam, NQ=100, gtrue_seed=3, verbose=True):
    """Full pipeline for one conduction-design regime. Returns a results dict."""
    seg, Q, G_TRUE = build_problem(K, q_sharp, gtrue_seed)
    solve = make_solver(seg, Q)
    T_TARGET = solve(G_TRUE)
    NOMINAL = np.full(K, 1.0)                                   # working-point design

    def objective(g):
        T = solve(g); return float(np.mean((T - T_TARGET) ** 2))

    def obj_grad_fd(g, rel=1e-3):                               # ground-truth dJ/dg, 2K solves
        g = np.asarray(g, dtype=float); grad = np.zeros(K)
        for k in range(K):
            h = rel * max(abs(g[k]), 1e-2)
            gp = g.copy(); gp[k] += h; gm = g.copy(); gm[k] -= h
            grad[k] = (objective(gp) - objective(gm)) / (2.0 * h)
        return grad

    if verbose:
        print(f"\n=== regime '{tag}': 1-D conduction BVP, K={K} segments, oracle grid={NGRID}, "
              f"q_sharp={q_sharp}, NS={NS}, roam=+-{roam} ===")
        print(f"  working point: nominal uniform g=1.0, J(nominal)={objective(NOMINAL):.4e}; "
              f"J(G_TRUE)={objective(G_TRUE):.2e} (~0, perfect design exists)")

    # --- surrogate training set: J(g) over a conductivity box around the working point ---
    RNG = np.random.default_rng(7)
    X = NOMINAL[None, :] * (1.0 + span * (2 * RNG.random((NS, K)) - 1)); X = np.clip(X, 0.05, None)
    Y = np.array([objective(x) for x in X])
    xm, xs = X.mean(0), X.std(0); ym, ysd = Y.mean(), Y.std()
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32)
    Yt = torch.tensor((Y - ym) / ysd, dtype=torch.float32).view(-1, 1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(K, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))

        def forward(self, x):
            return self.net(x)

    def train_one(seed):                                       # identical recipe to extbench_tmm.py
        torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = nn.MSELoss()
        n = Xt.size(0); idx = torch.randperm(n, generator=g); tr = idx[:int(0.9 * n)]
        for ep in range(400):
            perm = tr[torch.randperm(tr.numel(), generator=g)]
            for b in perm.split(256):
                opt.zero_grad(); lossf(net(Xt[b]), Yt[b]).backward(); opt.step()
        return net

    M = 10
    nets = [train_one(s) for s in range(M)]
    if verbose:
        print(f"  ensemble trained (M={M}, NS={NS}, K={K})")

    def ad_grad(net, xphys):
        xn = torch.tensor(((xphys - xm) / xs), dtype=torch.float32).view(1, -1).requires_grad_(True)
        y = net(xn); y.backward()
        return (xn.grad.detach().numpy().ravel()) * (ysd / xs)

    # --- query points near the working point; gate + allocator (verbatim logic) ---
    rngq = np.random.default_rng(11)
    QP = NOMINAL[None, :] * (1.0 + roam * (2 * rngq.random((NQ, K)) - 1)); QP = np.clip(QP, 0.05, None)
    TAU = 0.9
    all_sa, all_correct = [], []
    cos_gate = np.zeros((NQ, K + 1)); cos_rand = np.zeros((NQ, K + 1)); cos_orac = np.zeros((NQ, K + 1))

    def cossim(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    rng2 = np.random.default_rng(123)
    n_trust_at_tau = []
    for q, xq in enumerate(QP):
        Gm = np.array([ad_grad(net, xq) for net in nets])      # (M,K)
        ad = Gm.mean(0); sa = np.maximum((Gm > 0).mean(0), (Gm < 0).mean(0))
        fd = obj_grad_fd(xq)
        all_sa.append(sa); all_correct.append((np.sign(ad) == np.sign(fd)).astype(int))
        n_trust_at_tau.append(int((sa >= TAU).sum()))
        order_gate = np.argsort(sa)                              # least reliable first (gate-driven)
        order_orac = np.argsort(-np.abs(ad - fd))               # worst first (oracle upper bound)
        for n in range(K + 1):
            g = ad.copy(); g[order_gate[:n]] = fd[order_gate[:n]]; cos_gate[q, n] = cossim(g, fd)
            g = ad.copy(); g[order_orac[:n]] = fd[order_orac[:n]]; cos_orac[q, n] = cossim(g, fd)
            acc = 0.0; R = 8
            for _ in range(R):
                perm = rng2.permutation(K); g = ad.copy(); g[perm[:n]] = fd[perm[:n]]; acc += cossim(g, fd)
            cos_rand[q, n] = acc / R

    all_sa = np.concatenate(all_sa); all_correct = np.concatenate(all_correct)
    n_pos = int(all_correct.sum()); n_neg = int(all_correct.size - n_pos)
    auc = roc_auc_score(all_correct, all_sa) if 0 < n_pos < all_correct.size else float('nan')
    cg, cr, co = cos_gate.mean(0), cos_rand.mean(0), cos_orac.mean(0)
    frac_wrong = 1.0 - all_correct.mean()

    if verbose:
        print(f"  pooled over {NQ} query points x {K} comps = {all_correct.size} samples:")
        print(f"    class balance: #sign-correct={n_pos}  #sign-wrong={n_neg}  "
              f"(sign-wrong fraction={frac_wrong:.3f})")
        print(f"    gate AUC (sign-agreement predicts AD-component-sign-correct): {auc:.3f}")
        print(f"    mean #components trusted at tau={TAU}: {np.mean(n_trust_at_tau):.1f}/{K}")
        print("  cosine-to-truth vs #FD(BVP) solves (descent-direction recovery):")
        print(f"  {'#FD':>4} {'gate-order':>11} {'random-order':>13} {'oracle-order':>13} {'gate-rand':>10}")
        for n in range(K + 1):
            print(f"  {n:>4} {cg[n]:>11.3f} {cr[n]:>13.3f} {co[n]:>13.3f} {cg[n]-cr[n]:>+10.3f}")

    def solves_to(curve, target):
        idx = np.where(curve >= target)[0]
        return int(idx[0]) if len(idx) else K

    save_extra = {}
    if verbose:
        print("  ALLOCATOR -- BVP solves to reach a target descent-direction cosine:")
        print(f"    base cosine at 0 FD solves (pure surrogate gradient): {cg[0]:.3f}; "
              f"all-FD baseline = {K} solves (cosine=1.000)")
        print(f"    {'target':>7} {'gate':>5} {'random':>7} {'saved-vs-random':>16}")
    for tgt in (0.90, 0.95, 0.99):
        sg, sr = solves_to(cg, tgt), solves_to(cr, tgt)
        saved = sr - sg; pct = 100 * saved / max(sr, 1)
        if verbose:
            print(f"    {tgt:>7.2f} {sg:>5} {sr:>7} {f'{saved} ({pct:.0f}%)':>16}")
        save_extra[f'solves_gate_{tgt}'] = sg
        save_extra[f'solves_rand_{tgt}'] = sr

    return dict(tag=tag, auc=auc, frac_correct=float(all_correct.mean()), frac_wrong=float(frac_wrong),
                n_sign_correct=n_pos, n_sign_wrong=n_neg, cos_gate=cg, cos_rand=cr, cos_orac=co,
                K=K, tau=TAU, NS=NS, NQ=NQ, M=M, q_sharp=q_sharp, roam=roam, span=span,
                n_trust_mean=float(np.mean(n_trust_at_tau)), all_sa=all_sa, all_correct=all_correct,
                cos_ad=float(cg[0]), **save_extra)


# ----------------------------------------------------------------------------------------------
# (A) MAIN benchmark: discriminating (stressed) regime -- the one reported in the manuscript.
# ----------------------------------------------------------------------------------------------
main = run_benchmark(tag='main', K=16, q_sharp=3.0, NS=70, span=0.8, roam=0.25, NQ=100)

# ----------------------------------------------------------------------------------------------
# (B) Easy regime, logged for honesty: well-conditioned conduction is near-saturated (signs already
#     ~100% correct, gate AUC undefined, nothing for the allocator to save).
# ----------------------------------------------------------------------------------------------
print("\n----- honesty check: easy/well-conditioned conduction regime (for the record) -----")
easy = run_benchmark(tag='easy', K=8, q_sharp=1.0, NS=400, span=0.45, roam=0.10, NQ=60)

# ----------------------------------------------------------------------------------------------
# Save (MAIN regime arrays at top level so it mirrors extbench_tmm.npz; easy regime under easy_*).
# ----------------------------------------------------------------------------------------------
# top-level keys mirror extbench_tmm.npz (MAIN regime); plus the per-target solve counts and the
# easy-regime summary. Built as a plain dict because some keys contain '.' (not valid kwargs).
out = {k: main[k] for k in ('auc', 'frac_correct', 'frac_wrong', 'n_sign_correct', 'n_sign_wrong',
                            'cos_gate', 'cos_rand', 'cos_orac', 'K', 'tau', 'NS', 'NQ', 'M',
                            'q_sharp', 'roam', 'span', 'n_trust_mean', 'all_sa', 'all_correct',
                            'cos_ad', 'solves_gate_0.9', 'solves_rand_0.9', 'solves_gate_0.95',
                            'solves_rand_0.95', 'solves_gate_0.99', 'solves_rand_0.99')}
out['ngrid'] = NGRID
out['easy_auc'] = easy['auc']
out['easy_frac_wrong'] = easy['frac_wrong']
out['easy_n_sign_wrong'] = easy['n_sign_wrong']
out['easy_cos_ad'] = easy['cos_ad']
out['easy_K'] = easy['K']
np.savez('extbench_poisson.npz', **out)
print("\nsaved extbench_poisson.npz")
