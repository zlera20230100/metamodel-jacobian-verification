# Reference-Calibrated Verification of Metamodel Design Jacobians under Finite Simulation Budgets

This repository contains the code and data used in the manuscript submitted to
*Simulation Modelling Practice and Theory*.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21005573.svg)](https://doi.org/10.5281/zenodo.21005573)
[![release](https://img.shields.io/github/v/release/zlera20230100/metamodel-jacobian-verification)](https://github.com/zlera20230100/metamodel-jacobian-verification/releases)

Xuan Qin, Xuan Shi, Nimako Samuel Boateng, Bokai Huang, Shengjun Wu,
Kai You, and Long Zhang (corresponding author: 20230100@huat.edu.cn)

## Repository contents

The files for version 2.0.2 are in [`simpat_revision/`](simpat_revision/).
This directory contains the archived arrays,
reference-gradient verification, risk--coverage and finite-budget analyses,
repeated calibration and training audits, matched-cost multistep optimization,
openEMS step-ladder records, deterministic figure sources, and an evidence
manifest used by the revised manuscript.

Files at the repository root reproduce the original v1.0.0 release and remain
available for provenance. They should not be used to reconstruct the revised
SIMPAT claims. The immutable v1 tree is also available at tag
[`v1.0.0`](https://github.com/zlera20230100/metamodel-jacobian-verification/tree/v1.0.0).

## Scope of version 2.0.2

- Three independently implemented reference regimes provide 3,580 external
  design-gradient components: near-saturated TMM, stressed TMM, and stressed
  heat/Poisson.
- Every external reference sign passes the declared step-size and independent-
  derivative screen. The heat benchmark also passes the 160/320/640-cell sign
  audit.
- Mean-aligned sign agreement, ensemble signal-to-noise ratio (SNR), and
  magnitude are compared against the same labels and budgets. SNR has the
  highest ROC AUC in both TMM regimes; magnitude is highest for heat/Poisson;
  sign ordering has the smallest fixed-budget assembled-gradient error-
  frontier area in all three.
- At the worked mean-aligned-sign threshold 0.9, accepted coverage/sign-error
  risk is 95.9%/1.5%, 74.0%/13.4%, and 79.9%/4.9%. Correcting rejected
  components costs 0.90/22, 5.72/22, and 6.44/32 reference evaluations per
  query relative to complete central finite differences.
- Repeated-split calibration, independent training repeats, random tie
  resolution, ensemble-size sensitivity, and amortized reference-cost analyses
  expose how the operating point changes under finite data and finite budgets.
- In the matched-cost 20-step TMM audit, random, sign, SNR, and magnitude
  allocation each use exactly 240 reference calls per start. No deployable
  score ordering dominates random allocation across all tested trajectory
  outcomes.
- In the antenna step ladder, all six openEMS aperture derivatives reverse
  sign at the largest tested step. These derivatives are unresolved and are
  excluded from AUC, calibration, risk, cost-saving, and hybrid-gradient-error
  claims.

## Quick reproduction

Use Python 3.11 or newer:

```bash
cd simpat_revision
python -m pip install -r requirements.txt

# Reproduce the archived results and score comparison
python analyze_external_baselines.py --analysis-dir . --package-dir .

# Independent reference-gradient checks (3,580 components)
python reference_gradient_verification.py --data-dir . --out-dir .

# Worked risk--coverage and central-FD cost accounting
python risk_coverage_simpat.py --data-dir . --out-dir .

# Stability and matched-cost trajectory audits
python repeated_split_calibration.py
python tie_sensitivity.py
python subensemble_sensitivity.py
python training_repeat_audit.py
python break_even_reference_cost.py
python multistep_tmm_budget_optimization.py

# Graphical abstract
Rscript graphical_abstract_simpat.R
```

The external-score analysis uses 10,000 query-cluster bootstrap replicates.
Optional reconstruction of the small external surrogate ensembles is
documented in [`simpat_revision/README.md`](simpat_revision/README.md).

## What can be reproduced

1. **Archived-result reproduction** requires no retraining or
   reference-simulator call.
2. **External-member reconstruction** retrains only the small public benchmark
   ensembles and checks them against frozen accepted artifacts.
3. **Reference-label verification** reconstructs the TMM and heat queries and
   compares central differences with analytic or complex-step/grid
   derivatives.
4. **Full-wave provenance** includes the completed openEMS step ladder and
   logs, but rerunning it requires the full project tree and local openEMS/
   CSXCAD bindings.

The project-specific device PINN training framework is not bundled. Device
figure arrays are frozen-output redraw inputs, and no experimental antenna
validation is claimed.

## Citation and versioning

Use the concept DOI for the latest version and all-version citation:

> https://doi.org/10.5281/zenodo.21005573

| Release | Role | DOI |
|---|---|---|
| v2.0.2 | Current SIMPAT package | https://doi.org/10.5281/zenodo.21352345 |
| v2.0.1 | Previous SIMPAT package | https://doi.org/10.5281/zenodo.21331830 |
| v2.0.0 | Previous SIMPAT revision package | https://doi.org/10.5281/zenodo.21331538 |
| v1.0.0 | Legacy reproducibility snapshot | https://doi.org/10.5281/zenodo.21005574 |

Release-specific metadata are recorded in [`CITATION.cff`](CITATION.cff) and
[`.zenodo.json`](.zenodo.json). Zenodo assigns a separate DOI to every
published version while retaining the concept DOI above.

## License

MIT; see [`LICENSE`](LICENSE).
