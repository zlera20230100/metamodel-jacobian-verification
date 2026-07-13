# Heat/Poisson benchmark results

This benchmark tests the component score on one-dimensional steady heat
conduction. The reference solver is a finite-volume boundary-value solve with
harmonic face conductivities. Reference gradients use central finite
differences through the solver.

## Benchmark regimes

The well-conditioned regime is nearly saturated: all 480 gradient components
have the correct sign, so ROC AUC is undefined and selective checking has no
role. The script retains this case to record the easy boundary.

The main regime uses high-contrast conductivities, a sign-changing source, 16
design segments, 70 training samples, and query points extending 25% beyond the
training range. These settings produce enough sign errors to evaluate ranking
and allocation.

## Main-regime results

The main evaluation contains 100 queries and 16 components per query, for 1,600
component labels.

| Metric | Value |
|---|---:|
| Sign-correct / sign-wrong components | 1,429 / 171 |
| Sign-agreement AUC | 0.761 |
| Bootstrap 95% CI | [0.720, 0.802] |
| Mean agreement, sign-correct components | 0.944 |
| Mean agreement, sign-wrong components | 0.769 |
| Mean components accepted at threshold 0.9 | 12.8 / 16 |

The uncorrected surrogate gradient has mean cosine 0.951 to the reference
gradient. At a target cosine of 0.99, score ordering requires 12 component
checks and random ordering requires 13. Both methods require no checks at
targets of 0.90 and 0.95.

## Interpretation

Sign agreement ranks component correctness above chance in this diffusion
problem, but its AUC is lower than in the stressed TMM benchmark. The allocation
benefit is also small. Here, most sign errors occur on low-magnitude components
far from the source, so correcting them has little effect on the assembled
gradient direction.

The result supports transfer of the ranking signal, not a domain-independent
claim about effect size or simulator savings. In the well-conditioned regime,
surrogate gradient signs are already correct and the gate is unnecessary.

## Reproducibility

Run `python extbench_poisson.py` to regenerate the archived arrays. The script
records the grid, training sample count, query count, design dimension, random
seeds, and finite-difference step in its output archive.
