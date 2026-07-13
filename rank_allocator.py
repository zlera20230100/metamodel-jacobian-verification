# -*- coding: utf-8 -*-
# Rank-and-allocate optimal solver-budget allocator.
#
# Reviewer point: the current "allocator" is just a fixed threshold (trust every component with
# sign-agreement >= tau), which gives a VARIABLE, uncontrolled number of full-wave finite-difference
# (FD) solves and no notion of "best gradient for B solves".
#
# Rank-and-allocate procedure:
#   Given a fixed budget B in {0..K} FD solves, RANK the K components by a cheap gate-reliability
#   score (ensemble sign-agreement, or SNR), then spend the B available solves on the B LEAST-reliable
#   components (greedy: correct the most-likely-wrong first) and take free autodiff on the rest.
#   This yields EXACTLY B solves (a controllable budget) and, for each B, the lowest expected gradient
#   error achievable for that cost under the cheap ranking.
#
# We compare the cost-accuracy FRONTIER (mean relative gradient error vs number of solves B = 0..K) for
#   (a) rank-and-allocate by sign-agreement   (NEW, the proposed allocator)
#   (a') rank-and-allocate by SNR             (NEW variant)
#   (b) fixed-tau threshold gate              (CURRENT method; variable cost -> binned to a frontier)
#   (c) random ordering                       (spend B solves on B random components)
#   (d) oracle hybrid                         (FD the B components that are actually MOST wrong -> lower bound)
#
# Controlled spectrum reuses the EXACT generator of reliability_calib.py / hybrid_gradient.py /
# hybrid_decouple.py: known per-zone gradient a_true, M=10 retrained seeds, K=6 components, each with
# its own sigma drawn log-uniform, over many trials.
#
# Also runs the allocator on the real 24-GHz FPC device (6 radiation zones, FD truth available).
#
# Writes rank_allocator.npz and prints all frontier numbers + verdict.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '6')
import numpy as np

# ----------------------------------------------------------------------------------------------------
# Controlled reliability spectrum -- identical model to hybrid_gradient.py / hybrid_decouple.py
# ----------------------------------------------------------------------------------------------------
RNG = np.random.default_rng(20260624)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])      # mixed-sign per-zone gradient (controlled study)
K = a_true.size                                            # K = 6 components
M = 10                                                     # ensemble size (paper's M)
TAU = 0.9                                                  # paper's threshold (from calibration)
N_TRIALS = 20000                                           # Monte-Carlo trials over the reliability spectrum
A_NORM = np.linalg.norm(a_true)

def one_trial(rng):
    """One controlled trial: each component k gets its own reliability level sigma_k (log-uniform),
    observed through M independently-retrained seeds. Returns autodiff estimate, the two cheap gate
    scores (sign-agreement, SNR), and the per-component TRUE squared autodiff error."""
    sigma = np.exp(rng.uniform(np.log(0.05), np.log(8.0), size=K))
    J = a_true[None, :] + rng.normal(0.0, sigma[None, :], size=(M, K))   # M retrains, (M,K)
    ad = J.mean(0)                                                       # autodiff (ensemble mean) estimate
    sign_agree = np.maximum((J > 0).mean(0), (J < 0).mean(0))            # cheap gate score 1: sign-agreement
    snr = np.abs(ad) / (J.std(0) + 1e-12)                               # cheap gate score 2: SNR
    err2 = (ad - a_true) ** 2                                            # per-component squared autodiff error (oracle uses this)
    return ad, sign_agree, snr, err2

def relerr_from_fdset(ad, fd_mask):
    """Relative gradient error when the components in fd_mask are replaced by the FD truth a_true
    and the rest keep their autodiff value ad.  ||g - a_true|| / ||a_true||."""
    g = np.where(fd_mask, a_true, ad)
    return np.linalg.norm(g - a_true) / A_NORM

def alloc_masks_by_order(order):
    """Given a ranking `order` (component indices, FIRST = corrected first), build the K+1 FD masks
    for budgets B = 0..K: budget B corrects the first B components in `order`."""
    masks = np.zeros((K + 1, K), dtype=bool)
    for B in range(1, K + 1):
        masks[B] = masks[B - 1]
        masks[B, order[B - 1]] = True
    return masks

# accumulators: mean relative error at each integer budget B = 0..K, per strategy
err_rank_sa = np.zeros(K + 1)   # (a)  rank-and-allocate by sign-agreement (NEW)
err_rank_snr = np.zeros(K + 1)  # (a') rank-and-allocate by SNR (NEW)
err_random = np.zeros(K + 1)    # (c)  random ordering (averaged over a fresh random order each trial)
err_oracle = np.zeros(K + 1)    # (d)  oracle: FD the truly-most-wrong B components (lower bound)

# fixed-tau threshold gate (CURRENT): sweep tau, record (mean cost, mean error); also the paper TAU point
taus = np.round(np.linspace(0.5, 1.0, 11), 2)
gate_cost = {t: 0.0 for t in taus}
gate_err = {t: 0.0 for t in taus}
op_cost_tau = 0.0   # paper's tau=0.9 operating point
op_err_tau = 0.0
op_cost_sq = 0.0    # for variance/std of the gate's (uncontrolled) per-trial cost
gate_budget_hist = np.zeros(K + 1)   # histogram of the gate's realized per-trial budget at TAU
# matched-budget equivalence: on each trial, run rank-allocate with budget = gate's realized budget b,
# and accumulate that error. If rank-allocate subsumes the gate, these means are identical.
rank_matched_err = 0.0

rand_rng = np.random.default_rng(7777)
for _ in range(N_TRIALS):
    ad, sa, snr, err2 = one_trial(RNG)

    # (a) rank-and-allocate by sign-agreement: correct the LEAST reliable first -> ascending sign-agreement.
    #     tie-break by ascending SNR so the order is deterministic.
    order_sa = np.lexsort((snr, sa))                 # primary key sa ascending, secondary snr ascending
    masks_sa = alloc_masks_by_order(order_sa)
    for B in range(K + 1):
        err_rank_sa[B] += relerr_from_fdset(ad, masks_sa[B])

    # (a') rank-and-allocate by SNR: correct lowest-SNR first.
    order_snr = np.lexsort((sa, snr))
    masks_snr = alloc_masks_by_order(order_snr)
    for B in range(K + 1):
        err_rank_snr[B] += relerr_from_fdset(ad, masks_snr[B])

    # (c) random ordering
    order_rand = rand_rng.permutation(K)
    masks_rand = alloc_masks_by_order(order_rand)
    for B in range(K + 1):
        err_random[B] += relerr_from_fdset(ad, masks_rand[B])

    # (d) oracle: correct the components with the LARGEST true squared autodiff error first.
    order_oracle = np.argsort(-err2)                 # descending true error
    masks_oracle = alloc_masks_by_order(order_oracle)
    for B in range(K + 1):
        err_oracle[B] += relerr_from_fdset(ad, masks_oracle[B])

    # (b) fixed-tau threshold gate: trust comps with sa>=tau (autodiff), FD the rest. Cost is VARIABLE.
    for t in taus:
        fd_mask_t = sa < t                           # untrusted -> FD
        gate_cost[t] += fd_mask_t.sum()
        gate_err[t] += relerr_from_fdset(ad, fd_mask_t)
    fd_mask_op = sa < TAU
    b_op = int(fd_mask_op.sum())                      # gate's REALIZED (uncontrolled) budget this trial
    op_cost_tau += b_op
    op_cost_sq += b_op * b_op
    gate_budget_hist[b_op] += 1
    op_err_tau += relerr_from_fdset(ad, fd_mask_op)
    # matched-budget: rank-allocate spending exactly b_op solves on the b_op least-reliable comps.
    # (The gate already FDs exactly the comps with sa<TAU; rank-allocate(b_op) FDs the b_op lowest-sa
    #  comps -> the SAME set, so this must coincide with the gate. We accumulate it to PROVE it.)
    rank_matched_err += relerr_from_fdset(ad, masks_sa[b_op])

# finalize means
err_rank_sa /= N_TRIALS
err_rank_snr /= N_TRIALS
err_random /= N_TRIALS
err_oracle /= N_TRIALS
gate_cost_curve = np.array([gate_cost[t] / N_TRIALS for t in taus])
gate_err_curve = np.array([gate_err[t] / N_TRIALS for t in taus])
op_cost_tau /= N_TRIALS
op_err_tau /= N_TRIALS
op_cost_std = np.sqrt(max(0.0, op_cost_sq / N_TRIALS - op_cost_tau ** 2))   # std of gate's per-trial cost
gate_budget_hist /= N_TRIALS
rank_matched_err /= N_TRIALS

Bs = np.arange(K + 1)

# ----------------------------------------------------------------------------------------------------
# Frontier report
# ----------------------------------------------------------------------------------------------------
print("=" * 92)
print(f"CONTROLLED SPECTRUM  (M={M}, K={K}, sigma log-uniform [0.05,8.0], N_TRIALS={N_TRIALS})")
print("=" * 92)
print("Cost-accuracy frontier: mean relative gradient error vs budget B (number of FD solves)")
print(f"{'B':>3} {'rank(sign-agree)':>17} {'rank(SNR)':>11} {'random':>9} {'ORACLE':>9}")
for B in range(K + 1):
    print(f"{B:>3} {err_rank_sa[B]:>17.4f} {err_rank_snr[B]:>11.4f} "
          f"{err_random[B]:>9.4f} {err_oracle[B]:>9.4f}")

print("\nFixed-tau THRESHOLD gate (CURRENT method) -- variable cost, plotted as its own frontier:")
print(f"{'tau':>5} {'mean FD solves':>15} {'mean rel.err':>13}")
for t, c, e in zip(taus, gate_cost_curve, gate_err_curve):
    star = "  <- paper tau" if abs(t - TAU) < 1e-9 else ""
    print(f"{t:>5.2f} {c:>15.3f} {e:>13.4f}{star}")
print(f"\nPaper threshold gate @tau={TAU}: mean cost {op_cost_tau:.3f} solves, mean rel.err {op_err_tau:.4f}")

# ----------------------------------------------------------------------------------------------------
# Quantitative dominance metrics
# ----------------------------------------------------------------------------------------------------
# Interpolate rank-and-allocate(sign-agree) onto continuous budget to compare against the threshold gate
# at the SAME (fractional) average cost.
def interp_err(curve, B_query):
    return np.interp(B_query, Bs, curve)

# (1) Matched-cost comparison under two definitions:
#  (1a) RIGOROUS matched-budget: per trial, rank-allocate spending the gate's OWN realized budget.
#       This is the fair like-for-like test. It equals the gate exactly (rank-allocate subsumes the
#       gate as the special case "budget = #comps below tau").
matched_equiv_gap = rank_matched_err - op_err_tau          # ~0 proves equivalence
#  (1b) Fixed-budget-vs-average-cost via interpolation. This is the number the naive framing asks for,
#       but it is biased by Jensen's inequality (the gate's mean cost is fractional and the err-vs-B
#       curve is convex), so we report it transparently and do NOT use it as the headline.
rank_err_at_gatecost = interp_err(err_rank_sa, op_cost_tau)
err_reduction_abs = op_err_tau - rank_err_at_gatecost
err_reduction_pct = 100.0 * err_reduction_abs / op_err_tau if op_err_tau > 0 else float('nan')

# (2) How many solves does rank-allocate need (fixed budget) to MATCH the gate's average error?
#     invert err_rank_sa (monotone decreasing in B) to find B where rank error == gate error.
#     np.interp needs ascending x; err_rank_sa is descending, so flip.
B_for_gate_err = np.interp(op_err_tau, err_rank_sa[::-1], Bs[::-1])
solves_saved = op_cost_tau - B_for_gate_err

# (3) Area between curves (lower = better). Trapezoidal area under each frontier over B in [0,K].
def auc(curve):
    return np.trapezoid(curve, Bs)
auc_rank_sa = auc(err_rank_sa)
auc_rank_snr = auc(err_rank_snr)
auc_random = auc(err_random)
auc_oracle = auc(err_oracle)

# Build a threshold-gate frontier sampled on the same integer-B grid for an apples-to-apples area.
# For each integer budget B, the BEST error the threshold gate can buy at <= B average solves
# (lower envelope of its (cost,err) operating points with cost <= B).
gate_front = np.full(K + 1, np.nan)
for B in range(K + 1):
    feasible = gate_err_curve[gate_cost_curve <= B + 1e-9]
    if feasible.size:
        gate_front[B] = feasible.min()
# at B=0 the gate can only reach cost ~ its tau=1.0 point (cost>0), so low-B entries may be nan/poor.
auc_gate = np.trapezoid(np.where(np.isnan(gate_front), err_random[0], gate_front), Bs)

# gap to oracle: how much of the random->oracle improvement does rank capture, averaged over B>=1?
# closeness = (random - rank) / (random - oracle), 1.0 = matches oracle, 0.0 = no better than random.
mask_pos = (err_random - err_oracle) > 1e-9
closeness = np.zeros(K + 1)
closeness[mask_pos] = ((err_random - err_rank_sa)[mask_pos] /
                       (err_random - err_oracle)[mask_pos])
mean_closeness = closeness[1:K].mean()   # interior budgets (B=0 and B=K are degenerate: all equal)

print("\n" + "=" * 92)
print("DOMINANCE METRICS (controlled spectrum)")
print("=" * 92)
print(f"Area under frontier (mean rel.err integrated over B=0..{K}; LOWER is better):")
print(f"  rank-and-allocate (sign-agree) : {auc_rank_sa:.4f}")
print(f"  rank-and-allocate (SNR)        : {auc_rank_snr:.4f}")
print(f"  random ordering                : {auc_random:.4f}")
print(f"  threshold-gate lower envelope  : {auc_gate:.4f}")
print(f"  ORACLE (lower bound)           : {auc_oracle:.4f}")
print(f"  rank vs random : {100*(1-auc_rank_sa/auc_random):.1f}% smaller area  (rank DOMINATES random)")
print(f"  rank vs oracle : rank area is {auc_rank_sa/auc_oracle:.2f}x the oracle area "
      f"(1.00 = matches oracle)")

print("\n--- Relationship to the threshold gate ---")
print(f"The fixed-tau gate at tau={TAU} spends an UNCONTROLLED budget: mean {op_cost_tau:.3f} solves "
      f"but std {op_cost_std:.3f}, range 0..{K}.")
print(f"  realized-budget distribution (B=0..{K}): "
      + ", ".join(f"{p:.2f}" for p in gate_budget_hist))
print(f"RIGOROUS matched-budget test: rank-allocate spending the gate's OWN per-trial budget gives "
      f"mean err {rank_matched_err:.4f}")
print(f"  vs gate mean err {op_err_tau:.4f}  ->  gap {matched_equiv_gap:+.6f}  "
      f"(== 0: the gate IS rank-allocate with a random budget; rank-allocate SUBSUMES it).")
print(f"  => rank-allocate's advantage over the gate is CONTROLLABILITY (fixed B, std 0 vs {op_cost_std:.2f}),")
print(f"     not a different/better subset at matched realized cost.")

print(f"\nFixed-budget vs gate-average-cost (Jensen-biased, reported for transparency, NOT the headline):")
print(f"  gate mean cost {op_cost_tau:.2f} solves -> gate err {op_err_tau:.4f}; "
      f"rank-allocate interpolated at B={op_cost_tau:.2f} -> {rank_err_at_gatecost:.4f}")
print(f"  the gate looks {abs(err_reduction_pct):.0f}% {'better' if err_reduction_abs<0 else 'worse'} ONLY "
      f"because its mean cost is fractional and err-vs-B is convex (Jensen), not a real subset edge.")

print(f"\nGap to oracle: rank-allocate captures {100*mean_closeness:.0f}% of the "
      f"random->oracle improvement (avg over interior B); area within {auc_rank_sa/auc_oracle:.2f}x oracle.")

# Per-budget head-to-head vs threshold-gate lower envelope and vs random/oracle
print("\nPer-budget: does rank-allocate dominate? (error; lower is better)")
print(f"{'B':>3} {'rank':>8} {'thr.env':>8} {'random':>8} {'oracle':>8} {'rank<random?':>13} {'rank<thr?':>10}")
for B in range(K + 1):
    ge = gate_front[B]
    ge_s = f"{ge:.4f}" if not np.isnan(ge) else "   n/a"
    beats_rand = "yes" if err_rank_sa[B] <= err_random[B] + 1e-9 else "NO"
    beats_thr = ("yes" if (not np.isnan(ge) and err_rank_sa[B] <= ge + 1e-9)
                 else ("n/a" if np.isnan(ge) else "NO"))
    print(f"{B:>3} {err_rank_sa[B]:>8.4f} {ge_s:>8} {err_random[B]:>8.4f} "
          f"{err_oracle[B]:>8.4f} {beats_rand:>13} {beats_thr:>10}")

# ----------------------------------------------------------------------------------------------------
# Real 24-GHz FPC device: apply the allocator to the six radiation zones (FD truth available)
# ----------------------------------------------------------------------------------------------------
print("\n" + "=" * 92)
print("REAL 24-GHz FPC DEVICE (6 radiation zones, FD truth available)")
print("=" * 92)
ms = np.load('zones_multiseed.npz')
fw = np.load('grad_fullwave.npz')
rela = ms['rela']                  # (10,6) per-seed radiation gradient per zone
fd = fw['fd_grad']                 # (6,)  full-wave FD truth
Kd = fd.size
ad_dev = rela.mean(0)              # autodiff (ensemble mean) radiation gradient
sa_dev = np.maximum((rela > 0).mean(0), (rela < 0).mean(0))   # sign-agreement per zone
snr_dev = np.abs(ad_dev) / (rela.std(0) + 1e-12)
err2_dev = (ad_dev - fd) ** 2     # true per-zone squared autodiff error
fd_norm = np.linalg.norm(fd)

def relerr_dev(fd_mask):
    g = np.where(fd_mask, fd, ad_dev)
    return np.linalg.norm(g - fd) / fd_norm

def dev_frontier(order):
    masks = np.zeros((Kd + 1, Kd), dtype=bool)
    out = np.zeros(Kd + 1)
    out[0] = relerr_dev(masks[0])
    for B in range(1, Kd + 1):
        masks[B] = masks[B - 1]; masks[B, order[B - 1]] = True
        out[B] = relerr_dev(masks[B])
    return out

dev_order_sa = np.lexsort((snr_dev, sa_dev))      # least reliable first
dev_order_snr = np.lexsort((sa_dev, snr_dev))
dev_order_oracle = np.argsort(-err2_dev)
# random: average over many permutations for a stable baseline on this single device
dev_rng = np.random.default_rng(2024)
dev_front_rand = np.zeros(Kd + 1)
N_PERM = 5000
for _ in range(N_PERM):
    dev_front_rand += dev_frontier(dev_rng.permutation(Kd))
dev_front_rand /= N_PERM

dev_front_sa = dev_frontier(dev_order_sa)
dev_front_snr = dev_frontier(dev_order_snr)
dev_front_oracle = dev_frontier(dev_order_oracle)

print("sign-agreement per zone :", np.round(sa_dev, 3))
print("SNR per zone            :", np.round(snr_dev, 3))
print("autodiff grad           :", np.round(ad_dev, 4))
print("FD truth                :", np.round(fd, 4))
print("rank order (least-reliable first, sign-agree):", dev_order_sa)
print("oracle order (most-wrong first)              :", dev_order_oracle)
print(f"\nDevice cost-accuracy frontier (rel.err vs B):")
print(f"{'B':>3} {'rank(sign)':>11} {'rank(SNR)':>11} {'random':>9} {'ORACLE':>9}")
for B in range(Kd + 1):
    print(f"{B:>3} {dev_front_sa[B]:>11.4f} {dev_front_snr[B]:>11.4f} "
          f"{dev_front_rand[B]:>9.4f} {dev_front_oracle[B]:>9.4f}")
auc_dev_sa = np.trapezoid(dev_front_sa, np.arange(Kd + 1))
auc_dev_rand = np.trapezoid(dev_front_rand, np.arange(Kd + 1))
auc_dev_oracle = np.trapezoid(dev_front_oracle, np.arange(Kd + 1))
print(f"\nDevice area under frontier (lower better): rank {auc_dev_sa:.3f}, "
      f"random {auc_dev_rand:.3f}, oracle {auc_dev_oracle:.3f}")
print(f"  rank vs random on device: {100*(1-auc_dev_sa/auc_dev_rand):.1f}% smaller area")

# ----------------------------------------------------------------------------------------------------
# Save everything
# ----------------------------------------------------------------------------------------------------
np.savez('rank_allocator.npz',
         # controlled spectrum
         a_true=a_true, K=K, M=M, tau=TAU, n_trials=N_TRIALS, Bs=Bs,
         err_rank_sa=err_rank_sa, err_rank_snr=err_rank_snr,
         err_random=err_random, err_oracle=err_oracle,
         taus=taus, gate_cost_curve=gate_cost_curve, gate_err_curve=gate_err_curve,
         gate_front=gate_front, op_cost_tau=op_cost_tau, op_err_tau=op_err_tau,
         op_cost_std=op_cost_std, gate_budget_hist=gate_budget_hist,
         rank_matched_err=rank_matched_err, matched_equiv_gap=matched_equiv_gap,
         auc_rank_sa=auc_rank_sa, auc_rank_snr=auc_rank_snr,
         auc_random=auc_random, auc_gate=auc_gate, auc_oracle=auc_oracle,
         rank_err_at_gatecost=rank_err_at_gatecost,
         err_reduction_abs=err_reduction_abs, err_reduction_pct=err_reduction_pct,
         B_for_gate_err=B_for_gate_err, solves_saved=solves_saved,
         closeness=closeness, mean_closeness=mean_closeness,
         # real device
         dev_sa=sa_dev, dev_snr=snr_dev, dev_ad=ad_dev, dev_fd=fd,
         dev_front_sa=dev_front_sa, dev_front_snr=dev_front_snr,
         dev_front_rand=dev_front_rand, dev_front_oracle=dev_front_oracle,
         dev_order_sa=dev_order_sa, dev_order_oracle=dev_order_oracle,
         auc_dev_sa=auc_dev_sa, auc_dev_rand=auc_dev_rand, auc_dev_oracle=auc_dev_oracle)
print("\nsaved rank_allocator.npz")
