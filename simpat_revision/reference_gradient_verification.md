# Reference-gradient numerical verification

## Submission-level conclusion

- **TMM near-saturated:** all 880 components retain their sign at
  `h/2,h,2h = 0.25,0.5,1.0 nm` and agree in sign with the analytic
  characteristic-matrix derivative. No component triggers the declared
  numerical-resolution screen.
- **TMM stressed:** all 1,100 components pass the same step ladder and analytic
  derivative comparison.
- **Heat/Poisson stressed:** all 1,600 components retain their sign at
  `h_k/2,h_k,2h_k`, where `h_k=1e-3 max(|g_k|,1e-2)`, and agree with a
  complex-step derivative through the discrete finite-volume/Thomas solve.
  The signs also agree on the 160-, 320-, and 640-cell grids.
- The resolution screen therefore retains all 3,580 external labels. The
  operational mean-aligned-sign AUC, coverage, and risk values are unchanged.
- **openEMS antenna:** all six zones are positive at `h=0.025` and `h=0.05`
  but negative at `h=0.10`. The step ladder fails the numerical-resolution
  screen and no mesh-refinement result is available. The antenna gradients are
  unresolved and are excluded from AUC, risk, cost-saving, and hybrid-error
  claims.

## Frozen evidence and reference definitions

| benchmark | member-gradient artifact | nominal derivative | independent check |
|---|---|---|---|
| TMM near-saturated | `external_members_tmm_easy.npz` | central FD, `h=0.5 nm`; 80 x 11 components | analytic derivative of the full characteristic-matrix product and reflectance |
| TMM stressed | `external_members_tmm_stressed.npz` | central FD, `h=0.5 nm`; 100 x 11 components | same analytic derivative |
| Heat/Poisson stressed | `external_members_heat_poisson_stressed.npz` | relative central FD, `rel=1e-3`; 160-cell FV grid; 100 x 16 components | discrete complex step plus 160/320/640-grid audit |
| openEMS aperture proxy | `grad_fullwave.npz` plus step-ladder CSV/logs | central FD of `ln Q_ap`, nominal `h=0.05`; six zones | `h/2,h,2h` ladder fails; no mesh-refinement evidence |

The external member-gradient artifacts include the exact archived reference
gradients and the ten member gradients used to compute the operational score.
The verification script independently reconstructs the reference derivatives
and asserts equality of the archived and reconstructed reference signs before
reporting metrics.

## Declared numerical-resolution screen

For each query, let `s_q` be the largest absolute independent reference-gradient
component. A central difference is treated as a signed label only if:

1. the `h/2,h,2h` and independent-derivative signs agree;
2. `|g_ref| > max(1e-12, 1e-6 s_q, 10 |g_(h/2)-g_h|/3)`; and
3. for heat/Poisson, the maximum also includes `2|g_640-g_320|`, with matching
   signs on the 160-, 320-, and 640-cell grids.

This is a conservative, deterministic numerical screen, not a statistical
coverage guarantee. The absolute floor protects against floating-point zeros,
the query-relative term prevents numerically negligible components from being
hard-labelled, and the remaining terms inflate observed step/grid changes.

## Aggregate results

| benchmark | retained | h/2 vs h sign | 2h vs h sign | nominal vs independent sign | p95 nominal relative error |
|---|---:|---:|---:|---:|---:|
| TMM near-saturated | 880/880 | 100% | 100% | 100% | 0.02396% |
| TMM stressed | 1100/1100 | 100% | 100% | 100% | 0.03042% |
| Heat/Poisson stressed | 1600/1600 | 100% | 100% | 100% | 0.0000966% |

For heat, 160-to-320 and 320-to-640 sign retention is 100%. The corresponding
95th-percentile relative grid changes are 0.96396% and 0.24323%. Four very
small derivatives change sign only on the deliberately coarse 80-cell grid;
they are negative and stable on 160, 320, and 640 cells, so the declared screen
does not discard them.

## Operational-label metric audit

| benchmark | mean-aligned-sign AUC | tau=0.9 coverage | tau=0.9 false-trust risk | changed labels? |
|---|---:|---:|---:|---:|
| TMM near-saturated | 0.905600 | 0.959091 | 0.015403 | no |
| TMM stressed | 0.724089 | 0.740000 | 0.133907 | no |
| Heat/Poisson stressed | 0.762243 | 0.798750 | 0.048513 | no |

These are deterministic recalculations from the per-member gradients after the
resolution mask. The mask retains every component, so the operational AUC,
coverage, and risk values are unchanged.

## Antenna provenance and claim boundary

The released antenna quantity is

`d ln(Q_ap)/d g_k = [ln Q_ap(g_k+h)-ln Q_ap(g_k-h)]/(2h)`,

where `Q_ap=mean |E_z|^2` on a frequency-domain aperture plane at 24 GHz. The
nominal six-zone run used `h=0.05` and a 189 x 189 x 78 mesh (2,786,238 FDTD
cells). The new `h=0.025` and `h=0.10` runs use the same mesh recipe. Zone 0,
for example, changes from `+0.120` to `+0.361` and approximately `-0.46` across
the three steps; all other zones have the same positive/positive/negative
pattern. An independent repeat of the `h=0.10` run preserves all six negative
signs, with at most 2.01% relative difference.

This repeatability does not constitute mesh convergence. The defensible result
is that the openEMS derivative is unresolved at the tested discretisation. It
cannot supply a reference sign, physical validation, a feed-proxy derivative,
or a cross-code gradient claim. Only the antenna's response-level corroboration
is retained in the manuscript.

## Reproducibility files

- `reference_gradient_verification.py`: reconstructs the TMM/heat queries,
  performs step, analytic/complex-step, and grid checks, and reapplies the mask.
- `reference_gradient_verification_components.csv`: all 3,580 component-level
  derivatives, tolerances, signs, and relative changes.
- `reference_gradient_verification_summary.csv`: aggregate convergence metrics.
- `reference_gradient_label_metrics.csv`: mean-aligned-sign AUC, coverage, and
  risk before/after the resolution screen.
- `openems_step_ladder.py`, `openems_step_ladder_raw.csv`, and
  `openems_step_ladder_h10.csv`: checkpointed openEMS step-ladder runs;
  `grad_fullwave.npz` supplies the released nominal baseline.
- `summarize_openems_step_ladder.py`, `openems_step_ladder_summary.csv`, and
  `openems_step_ladder_h10_repeatability.csv`: six-zone summary and repeatability.
- `parse_openems_mesh_logs.py` and `openems_mesh_inventory.csv`: mesh dimensions
  printed by every new run.

Reproduction commands from the supplement directory:

```powershell
python -B reference_gradient_verification.py --data-dir . --out-dir .
python -B summarize_openems_step_ladder.py
python -B parse_openems_mesh_logs.py
```

Rerunning `openems_step_ladder.py` additionally requires a local openEMS/CSXCAD
installation and is computationally expensive; the submitted CSV and logs are
the frozen records of the completed runs. The TMM and heat checks are
solver-light and complete in seconds.
