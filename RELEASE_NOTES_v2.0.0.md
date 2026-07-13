# v2.0.0 - SIMPAT revision supplement

This release aligns the public reproducibility archive with the manuscript
*Reference-Calibrated Verification of Metamodel Design Jacobians under Finite
Simulation Budgets*.

## What changed

- Added the `simpat_revision/` package with 140 curated files.
- Added ten-member external-gradient artifacts for near-saturated TMM,
  stressed TMM, and stressed heat/Poisson regimes.
- Added head-to-head mean-aligned sign, SNR, and magnitude comparisons with
  query-cluster bootstrap confidence intervals, matched-coverage operating
  points, split calibration, and finite-budget assembled-gradient error.
- Added numerical checks for all 3,580 external reference-gradient components
  using step-size variation and independent analytic, complex-step, or grid
  derivatives.
- Added 200 repeated calibration/test splits, ten independent full training
  repeats per regime, 1,000 random tie resolutions, ensemble-size sensitivity,
  and amortized reference-cost accounting.
- Added a matched-cost 20-step TMM optimization audit. Each deployable rule
  uses exactly 240 reference calls per start; no score ordering dominates
  random allocation across all tested trajectory outcomes.
- Added the six-zone openEMS step ladder and retained its unresolved antenna
  derivatives as a scope boundary rather than quantitative reliability labels.
- Unified the publication figures with a colour-blind-safe blue hierarchy and
  deterministic vector artwork.

## Verification

- 3,580/3,580 controlled reference labels pass the declared numerical screen.
- The reference verification, risk--coverage, stability, cost, and multistep
  scripts run from the curated supplement.
- The manuscript source separately compiles to a validated 33-page review PDF.
- The release archive contains 140 files and has SHA-256 digest:

  `3e9bf1b041485b98cb9ef90fa9e34ad58f461bc0e00eea9af2395cf4b923765c`

## Scope boundary

The project-specific device PINN training framework is not bundled. Device
arrays are frozen-output redraw inputs. Antenna derivatives are numerically
unresolved and excluded from AUC, calibration, risk, cost-saving, and
assembled-gradient-error claims. No experimental device validation is claimed.

All versions are citable through the concept DOI:
<https://doi.org/10.5281/zenodo.21005573>.
