# v2.0.0 — SIMPAT revision supplement

This major release aligns the public reproducibility archive with the revised
manuscript, *Reference-Calibrated Selective Verification of Surrogate Design
Jacobians under Finite Simulation Budgets*.

## What changed

- Added the authoritative `simpat_revision/` package with 100 curated files.
- Added ten-member external-gradient artifacts for near-saturated TMM,
  stressed TMM, and stressed heat/Poisson regimes.
- Added head-to-head mean-aligned sign, SNR, and magnitude comparisons with
  query-cluster bootstrap confidence intervals, matched-coverage operating
  points, split calibration, and finite-budget assembled-gradient error.
- Added component-level numerical verification for all 3,580 external
  reference labels using h/2, h, and 2h checks plus analytic TMM or
  complex-step/grid heat derivatives.
- Added risk--coverage and exact central-FD reference-evaluation accounting.
- Added the six-zone openEMS step ladder, mesh inventory, logs, and
  repeatability audit. All antenna aperture derivatives remain unresolved and
  are excluded from quantitative reliability claims.
- Replaced the graphical abstract with deterministic vector artwork and added
  synchronized PDF, SVG, PNG, and source files.
- Updated README, citation metadata, and Zenodo metadata to the revised title
  and evidence boundary.

## Reproduction checks

- 3,580/3,580 external labels pass the declared numerical-resolution screen.
- The reference-verification, risk--coverage, and graphical-abstract scripts
  run from a fresh extraction of the curated supplement.
- All 21 Python scripts in the curated package pass `py_compile`.
- The manuscript LaTeX source separately compiles to the validated 29-page
  review PDF; manuscript sources are not part of this software release.

## Attached package

The GitHub release includes a release-curated archive of the committed
`simpat_revision/` tree as
`SIMPAT_reproducibility_supplement_v2.0.0.zip`.

SHA-256:

`44325ca61dab7f749043371e7548791b817922233e135d95b68c05b40e2a665f`

The expanded contents are committed under `simpat_revision/`. The automatic
Zenodo source archive also retains the v1 root files for provenance; the
authoritative v2 evidence is the `simpat_revision/` directory and the attached
ZIP, not the legacy root scripts.

## Scope boundary

The project-specific device PINN training framework is not bundled. Device
arrays are frozen-output redraw inputs. The antenna evidence is simulation-
based; no experimental device validation is claimed.

All versions are citable through the concept DOI:
<https://doi.org/10.5281/zenodo.21005573>.
