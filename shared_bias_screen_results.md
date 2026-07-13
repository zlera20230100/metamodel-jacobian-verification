# Shared-bias screen: results

This controlled experiment tests the component verification gate when retrained
models share a correlated bias. It uses the same synthetic model as
`shared_bias_sweep.py` and `hybrid_robust.py`.

- Code: `shared_bias_screen.py`
- Data: `shared_bias_screen.npz`
- Run: `python shared_bias_screen.py`
- Computation: CPU with `OMP_NUM_THREADS=6`

## Setup

For component `k` and ensemble member `m`,

```text
J[m,k] = a_true[k] + common[k] + idio[m,k]
```

The common term has variance `rho * sigma_k^2` and is shared across all
members. The member-specific term has variance `(1-rho) * sigma_k^2`. Total
noise variance therefore stays fixed as `rho` changes. The experiment uses six
components, ten retrained members, a sign-agreement threshold of 0.9, and 8,000
Monte Carlo trials for each value of `rho`.

The plain gate accepts a component when its ensemble sign agreement reaches
0.9. Two screens are compared with that rule:

1. A global screen rejects the complete ensemble when an intraclass
   correlation estimate exceeds its calibration threshold.
2. A component screen rejects accepted components whose across-member spread
   exceeds its calibration threshold.

Both thresholds are the 95th percentile of a separate `rho=0` calibration
cohort.

## Main results

The intraclass-correlation estimate tracks `rho` monotonically (Pearson
`r=0.998`; Spearman `r=1.000`) but has an intercept near 0.35. It is therefore
a relative regime indicator rather than an unbiased estimate of the shared
bias fraction.

| rho | plain risk | global-screen risk | component-screen risk | plain checks | global checks | component checks |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0007 | 0.0007 | 0.0000 | 2.19 | 2.46 | 2.39 |
| 0.5 | 0.0438 | 0.0496 | 0.0130 | 1.47 | 2.22 | 2.06 |
| 0.6 | 0.0591 | 0.0675 | 0.0224 | 1.30 | 2.31 | 1.93 |
| 0.7 | 0.0702 | 0.0825 | 0.0339 | 1.11 | 2.65 | 1.75 |
| 0.8 | 0.0879 | 0.1060 | 0.0521 | 0.89 | 3.38 | 1.48 |

The global screen detects correlated trials but does not reduce the conditional
risk among accepted components. It also uses more reference checks. A single
trial-level statistic cannot identify which component changed sign.

The component screen reduces accepted-set sign risk by about 70% at `rho=0.5`,
62% at `rho=0.6`, 52% at `rho=0.7`, and 41% at `rho=0.8`. At `rho=0.8`, its
assembled-gradient relative error is 0.593, compared with 1.310 for the plain
gate. The mean number of checked components rises from 0.89 to 1.48 out of six.

## Interpretation

Shared bias can increase sign agreement while increasing error. The global
intraclass-correlation statistic detects this regime but cannot localize the
affected component. Across-member spread provides a useful component-level
screen in this controlled construction. It reduces risk but does not prove
alignment with an independent reference gradient.

## Reproducibility

The random seeds, calibration thresholds, full `rho` grid, and output arrays
are stored in `shared_bias_screen.npz`. Repeated runs with the stated software
environment reproduce the saved arrays.
