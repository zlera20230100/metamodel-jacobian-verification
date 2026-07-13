# -*- coding: utf-8 -*-
# ============================================================================
# Shared-bias diagnostic for the component verification gate.
#
# The reliability-gated hybrid gradient certifies a component k
# (uses the cheap autodiff ensemble mean instead of a solver call) when the
# across-seed sign agreement sa_k = max(frac(+), frac(-)) >= tau. This
# fails under a shared seed-correlated bias: a common bias term that is
# identical across all M independently-retrained seeds inflates sign-agreement
# even when the agreed sign is wrong. The gate can then certify a sign-flipped
# component. This failure appears near rho ~ 0.5-0.7, where rho is the fraction
# of noise variance shared across seeds; see shared_bias_sweep.py and
# hybrid_robust.py.
#
# From the M-by-K seed gradients J, estimate the
# shared-variance fraction rho-hat by a one-way random-effects / intraclass-
# correlation (ICC) decomposition that treats the K components as targets and
# the M seeds as raters. The shared-bias screen keeps the sign-agreement test and
# also checks rho-hat. If rho-hat exceeds a calibrated threshold, the
# ensemble is flagged as correlated / untrustworthy and the gate ABSTAINS from
# certifying and defers to the solver.
#
# Estimator scope. rho-hat conflates two seed-shared structures: (i) the true
# signal a_true (identical across seeds) and (ii) the shared BIAS we want to
# detect. Hence raw rho-hat carries a positive offset even at rho=0. We report
# raw rho-hat and an offset-corrected version. The gate is a global regime
# detector that abstains on the whole correlated ensemble because
# a single trial-level scalar cannot point at WHICH component is the flipped one
# (verified: within a fixed rho, rho-hat has ~chance AUC for the per-component
# wrong/right label). Detection identifies the regime but does not localize the
# affected component.
#
# Reuses the exact shared-bias data model of shared_bias_sweep.py / hybrid_robust.py:
#   J[m,k] = a_true[k] + common[k] + idio[m,k],
#   common[k] ~ N(0, rho*sig_k^2)  (shared by every seed),
#   idio[m,k] ~ N(0, (1-rho)*sig_k^2),  sig_k log-uniform in [0.05, 8].
# Total per-component noise variance is sig_k^2 for every rho (std-matched), so
# Only the sharing changes; the marginal difficulty stays fixed.
#
# Output: shared_bias_screen.npz and a console summary.
# Run: python shared_bias_screen.py (CPU; OMP_NUM_THREADS=6)
# ============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '6')
import numpy as np

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

RNG = np.random.default_rng(20260624)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])
K = a_true.size
M = 10
TAU = 0.9                                    # sign-agreement trust threshold (same as the paper)
N_TRIALS = 8000
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SIG_LO, SIG_HI = 0.05, 8.0                   # log-uniform per-component noise scale
CAL_PCTL = 95.0                              # rho_gate := this percentile of rho-hat under rho=0


# ---------------------------------------------------------------------------
# data model (identical to shared_bias_sweep.py)
# ---------------------------------------------------------------------------
def draw_trial(rho, rng):
    sig = np.exp(rng.uniform(np.log(SIG_LO), np.log(SIG_HI), size=K))
    common = rng.normal(0.0, np.sqrt(rho) * sig, size=K)               # shared across seeds
    idio = rng.normal(0.0, np.sqrt(1.0 - rho) * sig[None, :], size=(M, K))
    J = a_true[None, :] + common[None, :] + idio
    return J, sig


# ---------------------------------------------------------------------------
# rho-hat: ICC(1,1) one-way random-effects estimate of the seed-shared variance
# fraction.  Targets = the K components, raters = the M seeds.
#   MSB = between-component mean square ; MSW = within-component (across-seed) MS
#   rho-hat = (MSB - MSW) / (MSB + (M-1) MSW)              [clipped to [0,1]]
# Intuition: a large shared bias makes each component's M seeds cluster tightly
# (small within / across-seed spread) -> MSW small -> rho-hat large. A purely
# idiosyncratic ensemble spreads the seeds out -> MSW large -> rho-hat small.
# ---------------------------------------------------------------------------
def rho_hat_icc(J):
    grand = J.mean()
    cm = J.mean(axis=0)                                  # per-component seed mean (K,)
    MSB = M * np.sum((cm - grand) ** 2) / (K - 1)
    MSW = np.sum((J - cm[None, :]) ** 2) / (K * (M - 1))
    denom = MSB + (M - 1) * MSW
    if denom <= 0:
        return 0.0
    return float(np.clip((MSB - MSW) / denom, 0.0, 1.0))


def rel_err(g):
    return np.linalg.norm(g - a_true) / np.linalg.norm(a_true)


# ---------------------------------------------------------------------------
# Step 0: calibrate both thresholds on the rho=0 reference.
# We only ever see J, so thresholds must be set from an i.i.d. reference cohort.
#
#  (a) GLOBAL regime trigger:  rho_gate = CAL_PCTL-th percentile of rho-hat at
#      rho=0; flag the ensemble "correlated" when rho-hat >= rho_gate. By
#      construction this costs ~(100-CAL_PCTL)% false alarms in the clean case.
#
#  (b) PER-COMPONENT spread ceiling:  s_gate = CAL_PCTL-th percentile of the
#      observed across-seed std sd_k AMONG rho=0 CERTIFIED components. Among
#      certified comps the across-seed spread sd_k ~ sqrt(1-rho)*sig_k is a
#      proxy for the true noise scale sig_k; a shared-bias sign-flip needs a
#      large common[k] ~ N(0, rho sig_k^2), i.e. it concentrates on LARGE sig_k
#      -> large observed sd_k. So a certified component with unusually large
#      sd_k is exactly the one a shared bias is most likely to have flipped.
#      (Probe AUC of sd_k for the per-component wrong label among certified
#      comps is ~0.9 at rho=0.5-0.8; the global rho-hat is ~chance per-comp.)
# ---------------------------------------------------------------------------
cal_rng = np.random.default_rng(13)
rhohat0 = []
sd_cert0 = []                                            # sd_k of certified comps at rho=0
for _ in range(20000):
    J0, _s0 = draw_trial(0.0, cal_rng)
    rhohat0.append(rho_hat_icc(J0))
    sa0 = np.maximum((J0 > 0).mean(0), (J0 < 0).mean(0))
    sd0 = J0.std(0)
    sd_cert0.extend(sd0[sa0 >= TAU].tolist())
rhohat0 = np.array(rhohat0)
sd_cert0 = np.array(sd_cert0)
RHO_GATE = float(np.percentile(rhohat0, CAL_PCTL))
S_GATE = float(np.percentile(sd_cert0, CAL_PCTL))
RHO0_MEAN = float(rhohat0.mean())                        # the signal-structure offset at rho=0
print("=" * 78)
print("CALIBRATION (rho=0 reference cohort, n=20000)")
print(f"  rho-hat | rho=0 : mean={RHO0_MEAN:.3f}  sd={rhohat0.std():.3f}  "
      f"median={np.median(rhohat0):.3f}")
print(f"  (a) global regime trigger  rho_gate = P{CAL_PCTL:.0f}(rho-hat|rho=0)      = {RHO_GATE:.3f}")
print(f"  (b) per-comp spread ceiling s_gate  = P{CAL_PCTL:.0f}(sd_k|certified,rho=0) = {S_GATE:.3f}")
print(f"      => each costs ~{100 - CAL_PCTL:.0f}% clean-case false alarm by construction")
print("=" * 78)


# ===========================================================================
# MAIN SWEEP over rho.  THREE gates are evaluated on identical trials:
#   PLAIN   : certify k iff sa_k >= TAU.
#   GLOBAL  : global regime trigger; if rho-hat >= RHO_GATE
#             the ensemble is declared correlated and the gate certifies NOTHING
#             (defers all K to the solver); else == PLAIN. This is the literal
#             "abstain when rho-hat is high" design.
#   PERCOMP : per-component spread ceiling; certify k iff
#             sa_k >= TAU AND sd_k <= S_GATE; i.e. drop certified components whose
#             observed across-seed spread (a proxy for the noise scale on which a
#             shared-bias flip rides) is anomalously large.
# Metrics per gate: false-TRUST RATE P(sign-wrong | certified) [headline safety],
# absolute false-trusts per component, solver cost (deferrals/trial, out of K),
# and assembled-gradient rel-error. Oracle = defer exactly the truly-unreliable.
# ===========================================================================
GATE_NAMES = ['plain', 'global', 'percomp']
rows = []
rhohat_mean_by_rho, rhohat_sd_by_rho = [], []
print(f"\n{'rho':>4} | {'rho^':>6} {'off':>6} | "
      f"{'FT.plain':>8} {'FT.glob':>8} {'FT.perC':>8} | "
      f"{'cst.pl':>6} {'cst.gl':>6} {'cst.pC':>6} | "
      f"{'er.pl':>6} {'er.gl':>6} {'er.pC':>6} {'er.AD':>6} | {'%flag':>6}")
for rho in RHOS:
    rng = np.random.default_rng(1000 + int(round(rho * 1000)))
    rh_list = []
    n_comp = 0
    n_flagged = 0
    n_trials_eff = 0
    # per-gate accumulators
    cert = {g: 0 for g in GATE_NAMES}               # # certified
    ftc = {g: 0 for g in GATE_NAMES}                # # false-trust (cert & wrong)
    cost = {g: 0 for g in GATE_NAMES}               # solver deferrals
    errs = {g: 0.0 for g in GATE_NAMES}             # summed rel-error
    err_ad_sum = 0.0
    err_oracle_sum = 0.0
    cost_oracle_sum = 0
    for _ in range(N_TRIALS):
        J, sig = draw_trial(rho, rng)
        ad = J.mean(0)
        sd = J.std(0)
        sa = np.maximum((J > 0).mean(0), (J < 0).mean(0))
        rh = rho_hat_icc(J)
        rh_list.append(rh)
        flagged = rh >= RHO_GATE
        n_flagged += int(flagged)
        n_trials_eff += 1
        n_comp += K

        wrong = np.sign(ad) != np.sign(a_true)
        truly_rel = (np.abs(a_true) / sig) >= 1.0

        cert_masks = {
            'plain':   sa >= TAU,
            'global':  (np.zeros(K, dtype=bool) if flagged else (sa >= TAU)),
            'percomp': (sa >= TAU) & (sd <= S_GATE),
        }
        for g in GATE_NAMES:
            cm = cert_masks[g]
            cert[g] += int(cm.sum())
            ftc[g] += int((cm & wrong).sum())
            cost[g] += int((~cm).sum())
            errs[g] += rel_err(np.where(cm, ad, a_true))
        err_ad_sum += rel_err(ad)
        err_oracle_sum += rel_err(np.where(truly_rel, ad, a_true))
        cost_oracle_sum += int((~truly_rel).sum())

    rh_arr = np.array(rh_list)
    rhohat_mean_by_rho.append(rh_arr.mean())
    rhohat_sd_by_rho.append(rh_arr.std())

    ft_rate = {g: ftc[g] / max(1, cert[g]) for g in GATE_NAMES}      # P(wrong | certified)
    ft_abs = {g: ftc[g] / n_comp for g in GATE_NAMES}               # per-component exposure
    cost_avg = {g: cost[g] / n_trials_eff for g in GATE_NAMES}      # deferrals/trial out of K
    err_avg = {g: errs[g] / n_trials_eff for g in GATE_NAMES}
    err_ad = err_ad_sum / n_trials_eff
    err_oracle = err_oracle_sum / n_trials_eff
    cost_oracle = cost_oracle_sum / n_trials_eff
    pct_flagged = 100.0 * n_flagged / n_trials_eff

    rows.append((rho, rh_arr.mean(), rh_arr.mean() - RHO0_MEAN,
                 ft_rate['plain'], ft_rate['global'], ft_rate['percomp'],
                 ft_abs['plain'], ft_abs['global'], ft_abs['percomp'],
                 cost_avg['plain'], cost_avg['global'], cost_avg['percomp'],
                 err_avg['plain'], err_avg['global'], err_avg['percomp'],
                 err_ad, err_oracle, cost_oracle, pct_flagged))
    print(f"{rho:>4.1f} | {rh_arr.mean():>6.3f} {rh_arr.mean()-RHO0_MEAN:>6.3f} | "
          f"{ft_rate['plain']:>8.4f} {ft_rate['global']:>8.4f} {ft_rate['percomp']:>8.4f} | "
          f"{cost_avg['plain']:>6.2f} {cost_avg['global']:>6.2f} {cost_avg['percomp']:>6.2f} | "
          f"{err_avg['plain']:>6.3f} {err_avg['global']:>6.3f} {err_avg['percomp']:>6.3f} "
          f"{err_ad:>6.3f} | {pct_flagged:>5.0f}%")

rows = np.array(rows)
true_rho = rows[:, 0]
rhohat_mean = rows[:, 1]

# ---------------------------------------------------------------------------
# (1) does rho-hat TRACK true rho?
# ---------------------------------------------------------------------------
pear = float(np.corrcoef(true_rho, rhohat_mean)[0, 1])
if _HAVE_SCIPY:
    spear = float(_sps.spearmanr(true_rho, rhohat_mean).statistic)
else:
    # rank corr fallback
    ar = np.argsort(np.argsort(true_rho)); br = np.argsort(np.argsort(rhohat_mean))
    spear = float(np.corrcoef(ar, br)[0, 1])
slope, intercept = np.polyfit(true_rho, rhohat_mean, 1)
# offset-corrected tracking quality
rhohat_corr = rhohat_mean - RHO0_MEAN
mae_corr = float(np.mean(np.abs(rhohat_corr - true_rho)))

print("\n" + "-" * 78)
print("(1) rho-hat tracking:  Pearson r = %.4f   Spearman = %.4f" % (pear, spear))
print("    linear fit  rho-hat ~ %.3f * rho + %.3f   (ideal slope 1, intercept = offset)" %
      (slope, intercept))
print("    offset-corrected rho-hat MAE vs true rho = %.3f" % mae_corr)

# ---------------------------------------------------------------------------
# headline: false-trust at the exploit band and the worst rho
# ---------------------------------------------------------------------------
def at(rho):
    i = int(np.argmin(np.abs(true_rho - rho)))
    return rows[i]

print("-" * 78)
print("(2) FALSE-TRUST RATE  P(certified & sign-WRONG | certified)  [plain | global | per-comp]:")
print("    (cost = solver deferrals/trial out of K=%d ; err = assembled-grad rel-error)" % K)
for r in (0.0, 0.5, 0.6, 0.7, 0.8):
    row = at(r)
    print("    rho=%.1f : FT %.4f | %.4f | %.4f   cost %.2f | %.2f | %.2f   err %.3f | %.3f | %.3f"
          % (r, row[3], row[4], row[5], row[9], row[10], row[11], row[12], row[13], row[14]))

# Risk summary at the largest rho, comparing both variants with the plain gate.
worst = rows[-1]
red_glob = (worst[3] - worst[4]) / max(1e-12, worst[3]) * 100
red_perc = (worst[3] - worst[5]) / max(1e-12, worst[3]) * 100
print("-" * 78)
print("(3) safety restoration at rho=%.1f (worst case):" % worst[0])
print("    false-trust rate   plain=%.4f | global=%.4f (%+.0f%%) | per-comp=%.4f (%+.0f%%)"
      % (worst[3], worst[4], -red_glob, worst[5], -red_perc))
print("    solver cost/K      plain=%.2f | global=%.2f | per-comp=%.2f | oracle=%.2f"
      % (worst[9], worst[10], worst[11], worst[17]))
print("    grad rel-error     plain=%.3f | global=%.3f | per-comp=%.3f | all-AD=%.3f | oracle=%.3f"
      % (worst[12], worst[13], worst[14], worst[15], worst[16]))

# ===========================================================================
# (4) THRESHOLD / OPERATING-POINT SWEEP.  The single P95 threshold above is
# strict (rho_gate=0.874) because rho-hat at rho=0 already averages ~0.36 from
# pure signal structure; it therefore only fires on extreme correlation and
# leaves the unflagged trials exposed. To show the FULL safety<->cost trade-off
# Re-evaluate the shared-bias screen over a grid of abstention
# thresholds, each calibrated as a percentile of the rho=0 reference cohort
# (so each has a known clean-case false-alarm rate). We report, per rho,
# false-trust rate and solver cost at every operating point.
# ===========================================================================
# We sweep the PER-COMPONENT spread ceiling s_gate over a grid of percentiles of
# the rho=0 certified-spread distribution. Lower percentile = stricter ceiling =
# more certified-but-noisy components rejected = lower false-trust at higher
# solver cost. (The GLOBAL trigger's frontier is already shown in the main table
# and is essentially flat in the false-trust RATE -- see the report.)
PCTLS = [70.0, 80.0, 90.0, 95.0, 99.0]
GATES = [float(np.percentile(sd_cert0, p)) for p in PCTLS]
print("\n" + "-" * 78)
print("(4) operating-point sweep -- PER-COMPONENT gate, false-trust@cost vs spread ceiling:")
hdr = "  rho |" + "".join(f"  P{int(p)}(ft/cost)".rjust(16) for p in PCTLS) + "   plain.ft"
print(hdr)
grid_ft = np.zeros((len(RHOS), len(PCTLS)))
grid_cost = np.zeros((len(RHOS), len(PCTLS)))
grid_flag = np.zeros((len(RHOS), len(PCTLS)))     # here: % of certified comps rejected by ceiling
for ri, rho in enumerate(RHOS):
    rng = np.random.default_rng(7000 + int(round(rho * 1000)))
    cert_count = np.zeros(len(PCTLS)); ft_count = np.zeros(len(PCTLS))
    cost_sum = np.zeros(len(PCTLS)); plaincert = 0; rejected = np.zeros(len(PCTLS))
    nt = 0
    for _ in range(N_TRIALS):
        J, sig = draw_trial(rho, rng)
        ad = J.mean(0); sd = J.std(0)
        sa = np.maximum((J > 0).mean(0), (J < 0).mean(0))
        cert_plain = sa >= TAU
        wrong = np.sign(ad) != np.sign(a_true)
        plaincert += cert_plain.sum()
        nt += 1
        for gi, gate in enumerate(GATES):
            cert_self = cert_plain & (sd <= gate)
            cert_count[gi] += cert_self.sum()
            ft_count[gi] += (cert_self & wrong).sum()
            cost_sum[gi] += (~cert_self).sum()
            rejected[gi] += (cert_plain & (sd > gate)).sum()
    for gi in range(len(PCTLS)):
        grid_ft[ri, gi] = ft_count[gi] / max(1, cert_count[gi])
        grid_cost[ri, gi] = cost_sum[gi] / nt
        grid_flag[ri, gi] = 100.0 * rejected[gi] / max(1, plaincert)
    line = f"  {rho:>3.1f} |"
    for gi in range(len(PCTLS)):
        line += f"  {grid_ft[ri, gi]:.3f}/{grid_cost[ri, gi]:.1f}".rjust(16)
    line += f"   {rows[ri, 3]:.3f}"
    print(line)
print("    (stricter ceiling [lower percentile] = lower false-trust, higher solver cost.)")

np.savez('shared_bias_screen.npz',
         rho=true_rho,
         rhohat_mean=rhohat_mean,
         rhohat_sd=np.array(rhohat_sd_by_rho),
         rhohat_offcorr=rhohat_corr,
         rho0_mean=RHO0_MEAN, rho_gate=RHO_GATE, s_gate=S_GATE, cal_pctl=CAL_PCTL,
         ft_plain=rows[:, 3], ft_global=rows[:, 4], ft_percomp=rows[:, 5],
         ftabs_plain=rows[:, 6], ftabs_global=rows[:, 7], ftabs_percomp=rows[:, 8],
         cost_plain=rows[:, 9], cost_global=rows[:, 10], cost_percomp=rows[:, 11],
         err_plain=rows[:, 12], err_global=rows[:, 13], err_percomp=rows[:, 14],
         err_ad=rows[:, 15], err_oracle=rows[:, 16], cost_oracle=rows[:, 17],
         pct_flagged=rows[:, 18],
         pearson=pear, spearman=spear, fit_slope=slope, fit_intercept=intercept,
         mae_offcorr=mae_corr,
         op_pctls=np.array(PCTLS), op_gates=np.array(GATES),
         grid_ft=grid_ft, grid_cost=grid_cost, grid_flag=grid_flag,
         tau=TAU, K=K, M=M, n_trials=N_TRIALS,
         rhohat0_sample=rhohat0[:2000])
print("\nsaved shared_bias_screen.npz")
