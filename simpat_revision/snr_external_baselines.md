# External SNR and magnitude baseline audit

## Audit contract

All methods below use the same per-member gradients, ensemble-mean sign-correctness labels, query
clusters, bootstrap resamples, calibration/test split, and finite-difference correction budget.
The rebuilt member-gradient artifacts were released only after exact reproduction of the archived
`all_sa` and `all_correct` arrays and agreement with the frozen AUC and zero-budget cosine.

| benchmark | G shape (Q x M x K) | frozen sign AUC | zero-budget cosine | artifact | SHA-256 |
|---|---|---|---|---|---|
| TMM easy (near-saturated) | 80 x 10 x 11 | 0.905621 | 0.995746 | external_members_tmm_easy.npz | 470bbff02348e7e8c5df0b5086a8d12c106e69ec0ded87522f4ff392476c7b01 |
| TMM stressed | 100 x 10 x 11 | 0.722777 | 0.777042 | external_members_tmm_stressed.npz | 70c9a0b68fb94e768eedcb131b9351437b9c2862788ab76424dbaba9afa50d7f |
| Heat/Poisson stressed | 100 x 10 x 16 | 0.761204 | 0.950912 | external_members_heat_poisson_stressed.npz | 1335f6276921ad0988188af1c872e333c4213c4dc670a71b8627225688f1ae92 |

## Discrimination

The outcome is whether the **ensemble-mean gradient sign** matches the numerical reference sign.
Intervals and AUC differences use the same 10,000 query-cluster bootstrap resamples. SNR is
`abs(mean)/std` with population standard deviation (`ddof=0`), matching the archived synthetic
baseline. CV is not duplicated because its inverse has exactly the same ordering as SNR. The raw
spread diagnostic is oriented as `-std`, so larger values mean more trusted. The archived modal
score is `max(fraction positive, fraction negative)`. The action-aligned replacement is the
fraction of member signs equal to `sign(ensemble mean)`; a zero ensemble mean receives score zero.

| benchmark | score | AUC (cluster 95% CI) | AUC difference versus mean-aligned sign (paired 95% CI) |
|---|---|---|---|
| TMM easy (near-saturated) | Mean-aligned sign | 0.906 [0.841, 0.963] | 0.000 [0.000, 0.000] |
| TMM easy (near-saturated) | Modal sign agreement | 0.906 [0.842, 0.963] | 0.000 [-0.001, 0.001] |
| TMM easy (near-saturated) | SNR $|\bar g|/s$ | 0.970 [0.955, 0.983] | 0.064 [0.015, 0.119] |
| TMM easy (near-saturated) | Magnitude $|\bar g|$ | 0.953 [0.926, 0.976] | 0.048 [-0.005, 0.108] |
| TMM easy (near-saturated) | Low raw spread $-s$ | 0.220 [0.143, 0.293] | -0.685 [-0.788, -0.585] |
| TMM stressed | Mean-aligned sign | 0.724 [0.692, 0.756] | 0.000 [0.000, 0.000] |
| TMM stressed | Modal sign agreement | 0.723 [0.690, 0.755] | -0.001 [-0.003, 0.000] |
| TMM stressed | SNR $|\bar g|/s$ | 0.781 [0.751, 0.811] | 0.057 [0.041, 0.074] |
| TMM stressed | Magnitude $|\bar g|$ | 0.738 [0.706, 0.769] | 0.014 [-0.008, 0.036] |
| TMM stressed | Low raw spread $-s$ | 0.560 [0.527, 0.591] | -0.164 [-0.209, -0.120] |
| Heat/Poisson stressed | Mean-aligned sign | 0.762 [0.729, 0.793] | 0.000 [0.000, 0.000] |
| Heat/Poisson stressed | Modal sign agreement | 0.761 [0.728, 0.793] | -0.001 [-0.003, 0.001] |
| Heat/Poisson stressed | SNR $|\bar g|/s$ | 0.825 [0.801, 0.847] | 0.062 [0.049, 0.078] |
| Heat/Poisson stressed | Magnitude $|\bar g|$ | 0.842 [0.819, 0.864] | 0.080 [0.064, 0.098] |
| Heat/Poisson stressed | Low raw spread $-s$ | 0.521 [0.486, 0.556] | -0.242 [-0.277, -0.204] |

**Result:** SNR exceeded mean-aligned sign agreement in 3/3 external benchmarks, and raw magnitude exceeded it in 3/3. Mean-aligned sign agreement was not the highest-AUC primary score in any external benchmark.
The paired 95% interval for SNR minus mean-aligned sign excludes zero in all three benchmarks.
For magnitude, it excludes zero only in heat/Poisson; the easy- and stressed-TMM magnitude
differences are point-estimate advantages with intervals spanning zero.

## Threshold and matched-coverage operating points

Both modal and mean-aligned sign scores use the prespecified threshold 0.9; their operating points
are identical in these data. Comparator thresholds are descriptive only: they are chosen without
labels to match the number accepted by the mean-aligned sign threshold as closely as possible.
They are not calibration guarantees. False-trust risk is
`P(sign-wrong | accepted)`; specificity is `P(rejected | sign-wrong)` and is not interchangeable
with that operational risk.

| benchmark | score | threshold | coverage | false-trust risk (cluster 95% CI) | specificity | mean FD checks / reference evaluations |
|---|---|---|---|---|---|---|
| TMM easy (near-saturated) | Mean-aligned sign | 0.9 | 95.9% | 0.015 [0.008, 0.024] | 53.6% | 0.45 / 0.90 |
| TMM easy (near-saturated) | Modal sign agreement | 0.9 | 95.9% | 0.015 [0.008, 0.024] | 53.6% | 0.45 / 0.90 |
| TMM easy (near-saturated) | SNR $|\bar g|/s$ | 1.0876 | 95.9% | 0.013 [0.006, 0.020] | 60.7% | 0.45 / 0.90 |
| TMM easy (near-saturated) | Magnitude $|\bar g|$ | 7.9456e-05 | 95.9% | 0.018 [0.010, 0.027] | 46.4% | 0.45 / 0.90 |
| TMM stressed | Mean-aligned sign | 0.9 | 74.0% | 0.134 [0.107, 0.164] | 53.2% | 2.86 / 5.72 |
| TMM stressed | Modal sign agreement | 0.9 | 74.0% | 0.134 [0.107, 0.164] | 53.2% | 2.86 / 5.72 |
| TMM stressed | SNR $|\bar g|/s$ | 0.98397 | 74.0% | 0.130 [0.106, 0.158] | 54.5% | 2.86 / 5.72 |
| TMM stressed | Magnitude $|\bar g|$ | 0.0011415 | 74.0% | 0.152 [0.124, 0.184] | 46.8% | 2.86 / 5.72 |
| Heat/Poisson stressed | Mean-aligned sign | 0.9 | 79.9% | 0.049 [0.040, 0.058] | 63.7% | 3.22 / 6.44 |
| Heat/Poisson stressed | Modal sign agreement | 0.9 | 79.9% | 0.049 [0.040, 0.058] | 63.7% | 3.22 / 6.44 |
| Heat/Poisson stressed | SNR $|\bar g|/s$ | 1.0819 | 79.9% | 0.049 [0.039, 0.058] | 63.7% | 3.22 / 6.44 |
| Heat/Poisson stressed | Magnitude $|\bar g|$ | 0.013969 | 79.9% | 0.046 [0.037, 0.056] | 65.5% | 3.22 / 6.44 |

## Split-sample calibration

For each benchmark, the same query-cluster split is used for both sign definitions, SNR, and magnitude. Calibration
chooses the largest-coverage candidate whose one-sided cluster-bootstrap risk upper bound is no
greater than the prespecified 10.0% target. Candidate intended coverages are fixed
at 10%, ..., 100%, and the upper screen is Bonferroni-adjusted across all ten candidates. Test
queries are untouched until after selection. This is a benchmark-conditional heuristic, not a
distribution-free or cross-domain guarantee.

| benchmark | score | selected threshold | calibration-only screen | held-out test |
|---|---|---|---|---|
| TMM easy (near-saturated) | Mean-aligned sign | 0.4 | coverage 100.0%; risk 3.6%; upper 5.7% | coverage 100.0%; risk 0.027 [0.014, 0.043] |
| TMM easy (near-saturated) | Modal sign agreement | 0.5 | coverage 100.0%; risk 3.6%; upper 5.7% | coverage 100.0%; risk 0.027 [0.014, 0.043] |
| TMM easy (near-saturated) | SNR $|\bar g|/s$ | 0.061539 | coverage 100.0%; risk 3.6%; upper 5.7% | coverage 99.5%; risk 0.023 [0.011, 0.037] |
| TMM easy (near-saturated) | Magnitude $|\bar g|$ | 3.8465e-06 | coverage 100.0%; risk 3.6%; upper 5.7% | coverage 100.0%; risk 0.027 [0.014, 0.043] |
| TMM stressed | Mean-aligned sign | none | no candidate passed | not evaluated |
| TMM stressed | Modal sign agreement | none | no candidate passed | not evaluated |
| TMM stressed | SNR $|\bar g|/s$ | 2.4352 | coverage 40.0%; risk 4.5%; upper 8.2% | coverage 41.6%; risk 0.039 [0.017, 0.066] |
| TMM stressed | Magnitude $|\bar g|$ | 0.0039605 | coverage 30.2%; risk 2.4%; upper 5.9% | coverage 26.4%; risk 0.021 [0.000, 0.047] |
| Heat/Poisson stressed | Mean-aligned sign | 0.8 | coverage 86.8%; risk 6.8%; upper 8.8% | coverage 87.0%; risk 0.056 [0.042, 0.070] |
| Heat/Poisson stressed | Modal sign agreement | 0.8 | coverage 86.8%; risk 6.8%; upper 8.8% | coverage 87.0%; risk 0.056 [0.042, 0.070] |
| Heat/Poisson stressed | SNR $|\bar g|/s$ | 1.0716 | coverage 80.0%; risk 5.5%; upper 7.4% | coverage 80.1%; risk 0.044 [0.031, 0.056] |
| Heat/Poisson stressed | Magnitude $|\bar g|$ | 0.007155 | coverage 90.0%; risk 7.8%; upper 9.8% | coverage 89.5%; risk 0.059 [0.045, 0.072] |

## Fixed-budget allocator

For each integer budget `B=0,...,K`, the allocator replaces the `B` lowest-scored components by
their reference finite differences. Thus `B` is a component-check budget and costs `2B` reference
objective evaluations for central differences. Exact sign-score ties are randomized 512
times per query without consulting the truth; reported curves average over these tie breaks. Lower
area under the assembled-gradient relative-error curve is better.

| benchmark | rank | score | normalized error-frontier area (cluster 95% CI) |
|---|---|---|---|
| TMM easy (near-saturated) | 1 | Modal sign agreement | 0.212 [0.181, 0.245] |
| TMM easy (near-saturated) | 2 | Mean-aligned sign | 0.212 [0.181, 0.245] |
| TMM easy (near-saturated) | 3 | SNR $|\bar g|/s$ | 0.229 [0.196, 0.264] |
| TMM easy (near-saturated) | 4 | Magnitude $|\bar g|$ | 0.272 [0.231, 0.316] |
| TMM stressed | 1 | Modal sign agreement | 0.667 [0.580, 0.760] |
| TMM stressed | 2 | Mean-aligned sign | 0.667 [0.580, 0.759] |
| TMM stressed | 3 | SNR $|\bar g|/s$ | 0.715 [0.614, 0.825] |
| TMM stressed | 4 | Magnitude $|\bar g|$ | 0.747 [0.645, 0.859] |
| Heat/Poisson stressed | 1 | Mean-aligned sign | 0.244 [0.235, 0.253] |
| Heat/Poisson stressed | 2 | Modal sign agreement | 0.244 [0.235, 0.253] |
| Heat/Poisson stressed | 3 | SNR $|\bar g|/s$ | 0.249 [0.238, 0.260] |
| Heat/Poisson stressed | 4 | Magnitude $|\bar g|$ | 0.253 [0.242, 0.264] |

Prespecified 25% and 50% budget checkpoints (rounded to an integer component count) are:

| benchmark | budget | score | relative error (cluster 95% CI) | mean cosine |
|---|---|---|---|---|
| TMM easy (near-saturated) | 3 checks / 6 evals | Mean-aligned sign | 0.281 [0.240, 0.326] | 0.980 |
| TMM easy (near-saturated) | 3 checks / 6 evals | Modal sign agreement | 0.281 [0.240, 0.325] | 0.979 |
| TMM easy (near-saturated) | 3 checks / 6 evals | SNR $|\bar g|/s$ | 0.313 [0.266, 0.363] | 0.988 |
| TMM easy (near-saturated) | 3 checks / 6 evals | Magnitude $|\bar g|$ | 0.316 [0.269, 0.367] | 0.992 |
| TMM easy (near-saturated) | 6 checks / 12 evals | Mean-aligned sign | 0.218 [0.186, 0.252] | 0.979 |
| TMM easy (near-saturated) | 6 checks / 12 evals | Modal sign agreement | 0.218 [0.186, 0.252] | 0.979 |
| TMM easy (near-saturated) | 6 checks / 12 evals | SNR $|\bar g|/s$ | 0.249 [0.213, 0.287] | 0.983 |
| TMM easy (near-saturated) | 6 checks / 12 evals | Magnitude $|\bar g|$ | 0.299 [0.253, 0.347] | 0.981 |
| TMM stressed | 3 checks / 6 evals | Mean-aligned sign | 0.864 [0.754, 0.984] | 0.837 |
| TMM stressed | 3 checks / 6 evals | Modal sign agreement | 0.865 [0.754, 0.984] | 0.837 |
| TMM stressed | 3 checks / 6 evals | SNR $|\bar g|/s$ | 0.874 [0.759, 0.999] | 0.842 |
| TMM stressed | 3 checks / 6 evals | Magnitude $|\bar g|$ | 0.896 [0.781, 1.020] | 0.824 |
| TMM stressed | 6 checks / 12 evals | Mean-aligned sign | 0.702 [0.608, 0.804] | 0.887 |
| TMM stressed | 6 checks / 12 evals | Modal sign agreement | 0.702 [0.608, 0.804] | 0.887 |
| TMM stressed | 6 checks / 12 evals | SNR $|\bar g|/s$ | 0.758 [0.646, 0.882] | 0.895 |
| TMM stressed | 6 checks / 12 evals | Magnitude $|\bar g|$ | 0.806 [0.695, 0.929] | 0.875 |
| Heat/Poisson stressed | 4 checks / 8 evals | Mean-aligned sign | 0.321 [0.310, 0.334] | 0.958 |
| Heat/Poisson stressed | 4 checks / 8 evals | Modal sign agreement | 0.321 [0.310, 0.333] | 0.958 |
| Heat/Poisson stressed | 4 checks / 8 evals | SNR $|\bar g|/s$ | 0.323 [0.311, 0.335] | 0.958 |
| Heat/Poisson stressed | 4 checks / 8 evals | Magnitude $|\bar g|$ | 0.323 [0.311, 0.335] | 0.958 |
| Heat/Poisson stressed | 8 checks / 16 evals | Mean-aligned sign | 0.268 [0.258, 0.279] | 0.970 |
| Heat/Poisson stressed | 8 checks / 16 evals | Modal sign agreement | 0.268 [0.258, 0.279] | 0.970 |
| Heat/Poisson stressed | 8 checks / 16 evals | SNR $|\bar g|/s$ | 0.279 [0.266, 0.292] | 0.970 |
| Heat/Poisson stressed | 8 checks / 16 evals | Magnitude $|\bar g|$ | 0.282 [0.269, 0.296] | 0.969 |

Minimum budgets at three direction-cosine targets are:

| benchmark | cosine target | score | component checks | reference evaluations |
|---|---|---|---|---|
| TMM easy (near-saturated) | 0.90 | Mean-aligned sign | 0 | 0 |
| TMM easy (near-saturated) | 0.90 | Modal sign agreement | 0 | 0 |
| TMM easy (near-saturated) | 0.90 | SNR $|\bar g|/s$ | 0 | 0 |
| TMM easy (near-saturated) | 0.90 | Magnitude $|\bar g|$ | 0 | 0 |
| TMM easy (near-saturated) | 0.95 | Mean-aligned sign | 0 | 0 |
| TMM easy (near-saturated) | 0.95 | Modal sign agreement | 0 | 0 |
| TMM easy (near-saturated) | 0.95 | SNR $|\bar g|/s$ | 0 | 0 |
| TMM easy (near-saturated) | 0.95 | Magnitude $|\bar g|$ | 0 | 0 |
| TMM easy (near-saturated) | 0.99 | Mean-aligned sign | 0 | 0 |
| TMM easy (near-saturated) | 0.99 | Modal sign agreement | 0 | 0 |
| TMM easy (near-saturated) | 0.99 | SNR $|\bar g|/s$ | 0 | 0 |
| TMM easy (near-saturated) | 0.99 | Magnitude $|\bar g|$ | 0 | 0 |
| TMM stressed | 0.90 | Mean-aligned sign | 7 | 14 |
| TMM stressed | 0.90 | Modal sign agreement | 7 | 14 |
| TMM stressed | 0.90 | SNR $|\bar g|/s$ | 7 | 14 |
| TMM stressed | 0.90 | Magnitude $|\bar g|$ | 8 | 16 |
| TMM stressed | 0.95 | Mean-aligned sign | 10 | 20 |
| TMM stressed | 0.95 | Modal sign agreement | 10 | 20 |
| TMM stressed | 0.95 | SNR $|\bar g|/s$ | 11 | 22 |
| TMM stressed | 0.95 | Magnitude $|\bar g|$ | 11 | 22 |
| TMM stressed | 0.99 | Mean-aligned sign | 11 | 22 |
| TMM stressed | 0.99 | Modal sign agreement | 11 | 22 |
| TMM stressed | 0.99 | SNR $|\bar g|/s$ | 11 | 22 |
| TMM stressed | 0.99 | Magnitude $|\bar g|$ | 11 | 22 |
| Heat/Poisson stressed | 0.90 | Mean-aligned sign | 0 | 0 |
| Heat/Poisson stressed | 0.90 | Modal sign agreement | 0 | 0 |
| Heat/Poisson stressed | 0.90 | SNR $|\bar g|/s$ | 0 | 0 |
| Heat/Poisson stressed | 0.90 | Magnitude $|\bar g|$ | 0 | 0 |
| Heat/Poisson stressed | 0.95 | Mean-aligned sign | 0 | 0 |
| Heat/Poisson stressed | 0.95 | Modal sign agreement | 0 | 0 |
| Heat/Poisson stressed | 0.95 | SNR $|\bar g|/s$ | 0 | 0 |
| Heat/Poisson stressed | 0.95 | Magnitude $|\bar g|$ | 0 | 0 |
| Heat/Poisson stressed | 0.99 | Mean-aligned sign | 14 | 28 |
| Heat/Poisson stressed | 0.99 | Modal sign agreement | 14 | 28 |
| Heat/Poisson stressed | 0.99 | SNR $|\bar g|/s$ | 14 | 28 |
| Heat/Poisson stressed | 0.99 | Magnitude $|\bar g|$ | 14 | 28 |

## Modal-sign versus ensemble-mean-sign audit

A 5--5 vote is recorded as an undefined modal tie, not forced to either sign. Non-tie mismatch is
`modal_sign != sign(ensemble_mean)`. The last three columns evaluate the sign threshold 0.9.

| benchmark | modal ties | non-tie modal/mean mismatch | mean-direction false accepts | false accepts with mismatch | false accepts rescued by modal | mean-zero | member zeros | mean-aligned score differs | tau=0.9 set differs |
|---|---|---|---|---|---|---|---|---|---|
| TMM easy (near-saturated) | 5 (0.6%) | 3 (0.3%) | 13 | 0 | 0 | 0 | 0 | 3 | 0 |
| TMM stressed | 43 (3.9%) | 13 (1.2%) | 109 | 0 | 0 | 0 | 0 | 13 | 0 |
| Heat/Poisson stressed | 41 (2.6%) | 15 (0.9%) | 62 | 0 | 0 | 0 | 0 | 15 | 0 |

Modal/mean disagreements occur only at low agreement in these artifacts and contribute zero false
acceptances at threshold 0.9. There are no ensemble-mean-zero or exact member-zero gradients. The
mean-aligned definition removes the action mismatch below threshold, while leaving all reported
threshold-0.9 risks and coverages unchanged.

## Interpretation and use constraints

1. Sign agreement is not empirically superior to SNR or magnitude for classifying sign correctness;
   SNR has the highest AUC in the two TMM regimes and magnitude in heat/Poisson.
2. The method is a **score-agnostic selective gradient-verification protocol**. The AUC and
   fixed-budget objectives need not select the same score: sign-based ordering can reduce assembled-
   gradient error more effectively even when SNR has higher sign-correctness AUC. Select a score
   against the intended deployment loss using calibration data.
3. The operational algorithm uses the mean-aligned sign fraction so that the score refers to the
   direction actually returned by the ensemble mean. Retain modal agreement only as an auxiliary
   comparison; threshold-0.9 headline values remain unchanged here.
4. Positive-rescaling invariance is a robustness property; it is not evidence of superior
   discrimination.
5. The main manuscript reports the head-to-head AUC, risk--coverage, split-calibration, and
   fixed-budget results together.
6. Score selection remains inside the calibration protocol. A deployment domain may select SNR,
   magnitude, sign agreement, or abstention; test data must not be used to select the score or its
   threshold.

## Reproducibility

- Python: 3.13.9
- NumPy: 2.3.5
- SciPy: 1.16.3
- pandas: 2.3.3
- Matplotlib: 3.10.6
- Query-cluster bootstrap replicates: 10,000
- Random tie-break repetitions: 512
- Random seed: 20260711
