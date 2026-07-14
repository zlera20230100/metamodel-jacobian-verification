#!/usr/bin/env python3
"""Nominal reference-evaluation break-even audit for selective Jacobian verification.

The calculation is deliberately expressed in reference-objective evaluations.
It does not convert unlike simulators or training runs into wall-clock speedups.
The default calibration and correction terms count one central-difference pair per
component and are lower bounds: numerical-screen and fitting costs must be supplied
for a deployment-level calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--analysis-dir", type=Path, default=here)
    parser.add_argument("--package-dir", type=Path, default=here)
    parser.add_argument(
        "--additional-screen-evaluations-per-query",
        type=float,
        default=0.0,
        help="Measured screening or sentinel calls beyond nominal two-call corrections.",
    )
    parser.add_argument(
        "--additional-calibration-evaluations",
        type=float,
        default=0.0,
        help="Measured calibration-screen calls beyond one FD pair per component and query.",
    )
    parser.add_argument(
        "--fit-time-reference-equivalent",
        type=float,
        default=0.0,
        help="Measured metamodel fitting cost expressed in reference-objective equivalents.",
    )
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    package_dir = args.package_dir.resolve()

    cost = pd.read_csv(analysis_dir / "fd_cost_accounting.csv")
    n_train = {
        "TMM easy (near-saturated)": 600,
        "TMM stressed": 250,
        "Heat/Poisson stressed": 70,
    }
    n_cal = {
        "TMM easy (near-saturated)": 40,
        "TMM stressed": 50,
        "Heat/Poisson stressed": 50,
    }

    rows: list[dict] = []
    for _, item in cost.iterrows():
        dataset = item["dataset"]
        k = int(item["K"])
        nominal_correction = float(item["reference_evaluations_mean"])
        complete = float(item["all_fd_reference_evaluations"])
        actual_check = nominal_correction + args.additional_screen_evaluations_per_query
        saving = complete - actual_check
        nominal_calibration = 2 * k * n_cal[dataset]
        actual_calibration = nominal_calibration + args.additional_calibration_evaluations
        training_data_evals = n_train[dataset]
        if saving <= 0:
            calibration_be = gross_be = float("inf")
        else:
            calibration_be = actual_calibration / saving
            gross_be = (
                actual_calibration
                + training_data_evals
                + args.fit_time_reference_equivalent
            ) / saving
        rows.append(
            {
                "dataset": dataset,
                "operating_rule": "fixed mean-aligned sign threshold 0.9",
                "K": k,
                "N_train_simulator_labels": training_data_evals,
                "N_cal_queries": n_cal[dataset],
                "nominal_calibration_reference_evaluations_lower_bound": nominal_calibration,
                "additional_calibration_screen_evaluations": args.additional_calibration_evaluations,
                "actual_calibration_reference_evaluations": actual_calibration,
                "nominal_selective_correction_evaluations_per_query": nominal_correction,
                "additional_screen_evaluations_per_query": args.additional_screen_evaluations_per_query,
                "actual_selective_check_evaluations_per_query": actual_check,
                "complete_FD_reference_evaluations_per_query": complete,
                "net_reference_evaluations_saved_per_query": saving,
                "calibration_only_break_even_queries": int(np.ceil(calibration_be)) if np.isfinite(calibration_be) else np.inf,
                "gross_data_and_fit_break_even_queries": int(np.ceil(gross_be)) if np.isfinite(gross_be) else np.inf,
                "fit_time_reference_equivalent": args.fit_time_reference_equivalent,
                "interpretation": "lower bound when additional screen and fit inputs are zero",
            }
        )

    frame = pd.DataFrame.from_records(rows)
    output = analysis_dir / "break_even_reference_evals.csv"
    frame.to_csv(output, index=False, float_format="%.12g")
    output_target = package_dir / output.name
    if output.resolve() != output_target.resolve():
        shutil.copy2(output, output_target)
    script_target = package_dir / Path(__file__).name
    if Path(__file__).resolve() != script_target.resolve():
        shutil.copy2(Path(__file__).resolve(), script_target)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
