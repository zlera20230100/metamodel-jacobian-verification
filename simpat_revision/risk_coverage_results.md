# Risk--coverage and reference-evaluation audit

This report was computed by `risk_coverage_simpat.py` from the frozen arrays
listed in the Data audit below. No training and no reference-simulator call was performed.

## Definitions and uncertainty

- A component is **accepted** when its mean-aligned member-sign agreement is at least the stated threshold.
- **Coverage** is the fraction of components accepted.
- **False-trust risk** is `P(sign-wrong | accepted)`. It is not the false-positive rate;
  `P(accepted | sign-wrong) = 1 - specificity` is reported separately in the CSV.
- Sensitivity is the fraction of sign-correct components accepted. Specificity is the fraction of
  sign-wrong components rejected.
- Primary 95% intervals are percentile intervals from 10,000 bootstrap resamples of whole
  query points (seed 20260711). Resampling query clusters preserves within-query dependence among
  gradient components. Pooled Wilson intervals are included in the CSV only as a small-count
  diagnostic and are not the primary uncertainty statement.
- The intervals quantify finite-query sampling variation conditional on the frozen trained
  ensembles and benchmark-generating setup. They do not establish cross-domain calibration or a
  universal threshold.

## Operating point at threshold 0.9

Counts are `TP/FP/FN/TN`, with "positive" meaning that the protocol accepts the component.

| benchmark | TP/FP/FN/TN | coverage (cluster 95% CI) | false-trust risk (cluster 95% CI) | sensitivity (cluster 95% CI) | specificity (cluster 95% CI) |
|---|---|---|---|---|---|
| TMM easy (near-saturated) | 831/13/21/15 | 95.9% [94.4%, 97.3%] | 1.5% [0.8%, 2.4%] | 97.5% [96.4%, 98.5%] | 53.6% [36.4%, 71.4%] |
| TMM stressed | 705/109/162/124 | 74.0% [71.5%, 76.4%] | 13.4% [10.7%, 16.4%] | 81.3% [78.9%, 83.6%] | 53.2% [47.4%, 59.4%] |
| Heat/Poisson stressed | 1216/62/213/109 | 79.9% [78.5%, 81.2%] | 4.9% [4.0%, 5.8%] | 85.1% [83.7%, 86.5%] | 63.7% [57.3%, 69.9%] |

The stressed benchmarks show the deployment trade-off clearly: threshold 0.9 leaves non-zero
residual error among accepted components. The score is therefore a selective-risk screen, not a
certificate that an accepted gradient component is correct.

## Split-sample calibration demonstration

As an illustrative rule fixed before inspecting the held-out split, 50%
of query clusters are used for calibration and the remainder for testing. Among candidate thresholds
`0.5, ..., 1.0`, the rule selects the smallest threshold (largest coverage) whose one-sided
query-cluster bootstrap risk upper bound is no greater than the prespecified target
`10%`. The upper screen is Bonferroni-adjusted across the six candidate
thresholds. The held-out test clusters are not used in threshold selection.

| benchmark | cal/test queries | selected threshold | calibration-only screen | untouched test result (cluster 95% CI) |
|---|---|---|---|---|
| TMM easy (near-saturated) | 40/40 | 0.5 | 99.3% coverage; risk 2.5%; adjusted upper 4.3% | coverage 100.0% [100.0%, 100.0%]; risk 3.6% [2.3%, 5.2%] |
| TMM stressed | 50/50 | none | no candidate passed | not evaluated |
| Heat/Poisson stressed | 50/50 | 0.7 | 92.4% coverage; risk 7.3%; adjusted upper 9.1% | coverage 91.2% [89.4%, 93.0%]; risk 7.3% [5.7%, 8.8%] |

This is a **split-sample calibration heuristic**, not an independent guarantee: the benchmark
queries share the same frozen ensemble and data-generating regime, bootstrap bounds are not
distribution-free, and a single 50/50 split is data-limited. If no candidate passes (as can occur
in the stressed TMM case), the procedure abstains rather than inspecting the test split and relaxing
the target. External deployment would require a fresh, representative calibration set.

## Central-finite-difference cost accounting at threshold 0.9

This table answers a specific counterfactual: *if every rejected component is corrected by a
central finite difference*, how many reference evaluations are required on average per query?

| benchmark | K | accepted components | FD component checks | reference evaluations | all-FD evaluations | evaluation saving vs all-FD |
|---|---|---|---|---|---|---|
| TMM easy (near-saturated) | 11 | 10.55 | 0.45 | 0.90 | 22 | 95.9% |
| TMM stressed | 11 | 8.14 | 2.86 | 5.72 | 22 | 74.0% |
| Heat/Poisson stressed | 16 | 12.78 | 3.22 | 6.44 | 32 | 79.9% |

The conversion is exact for the archived benchmark implementations:

`1 central-FD component check = 2 reference-objective evaluations`, hence
`B component checks = 2B evaluations` and a complete K-component central-FD gradient costs `2K`
evaluations. Gate scoring itself uses no *additional* reference evaluation. Fractional values in
the table are averages across query points; each individual query uses an integer number of checks.
This accounting should not be described as "B simulator solves" without the factor of two.

## Data audit

| source | queries x components | sign-correct | sign-wrong | SHA-256 |
|---|---|---|---|---|
| extbench_tmm.npz | 80 x 11 = 880 | 852 | 28 | 61ded7675b3692490b3b4dd191180396641f9deb4a72ca3159f865060b5e082f |
| extbench_tmm_hard.npz | 100 x 11 = 1100 | 867 | 233 | 6cfd600d4fb9f5a2002fe16db48facd589599b8a951355fe49d5a33ded7e6517 |
| extbench_poisson.npz | 100 x 16 = 1600 | 1429 | 171 | 62c7c35830ce745754cc0cbff8a37294f605925874c2d8160d4341f1666485b8 |

The Poisson archive also reports an easy-regime summary (`easy_K=8`,
`easy_n_sign_wrong=0`, `easy_auc=nan`). It does not
store the easy-regime component-level `all_sa`/`all_correct` arrays, so risk--coverage and threshold
intervals for that auxiliary regime cannot be reconstructed without rerunning the experiment. It
is therefore neither plotted nor filled in.

## Recommended manuscript wording

> At threshold 0.9, the gate retained [coverage] of gradient components, while the observed
> sign-error rate among retained components was [false-trust risk]. Query-cluster bootstrap
> intervals quantify uncertainty conditional on each frozen benchmark. If every rejected component
> is centrally finite-difference corrected, the mean reference-evaluation cost is twice the mean
> number of rejected components; this distinction is maintained throughout.

Replace the bracketed quantities by benchmark-specific values from the operating-point table. Do
not combine the three benchmarks into one pooled claim, because their regimes, component counts,
and base error rates differ.

## Reproducibility

- Python: 3.13.9
- NumPy: 2.3.5
- SciPy: 1.16.3
- pandas: 2.3.3
- Matplotlib: 3.10.6
- Bootstrap replicates: 10,000
- Random seed: 20260711
- Split-calibration risk target: 0.100
- Calibration fraction: 0.500
