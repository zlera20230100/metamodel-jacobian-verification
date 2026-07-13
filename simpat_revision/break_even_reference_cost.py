#!/usr/bin/env python3
"""Reference-evaluation break-even audit for selective Jacobian verification.

The calculation is deliberately expressed in reference-objective evaluations.
It does not convert unlike simulators or training runs into wall-clock speedups.
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
    parser.add_argument("--package-dir", type=Path, default=here.parent / "reproducibility_update")
    parser.add_argument(
        "--sentinel-evaluations-per-query",
        type=float,
        default=0.0,
        help="Additional reference-objective evaluations scheduled per deployment query.",
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
        correction = float(item["reference_evaluations_mean"])
        complete = float(item["all_fd_reference_evaluations"])
        saving = complete - correction - args.sentinel_evaluations_per_query
        calibration_evals = 2 * k * n_cal[dataset]
        training_data_evals = n_train[dataset]
        if saving <= 0:
            calibration_be = gross_be = float("inf")
        else:
            calibration_be = calibration_evals / saving
            gross_be = (calibration_evals + training_data_evals) / saving
        rows.append(
            {
                "dataset": dataset,
                "operating_rule": "fixed mean-aligned sign threshold 0.9",
                "K": k,
                "N_train_simulator_labels": training_data_evals,
                "N_cal_queries": n_cal[dataset],
                "calibration_reference_evaluations": calibration_evals,
                "selective_reference_evaluations_per_query": correction,
                "complete_FD_reference_evaluations_per_query": complete,
                "sentinel_reference_evaluations_per_query": args.sentinel_evaluations_per_query,
                "net_reference_evaluations_saved_per_query": saving,
                "calibration_only_break_even_queries": int(np.ceil(calibration_be)) if np.isfinite(calibration_be) else np.inf,
                "gross_data_break_even_queries_excluding_fit_time": int(np.ceil(gross_be)) if np.isfinite(gross_be) else np.inf,
                "fit_time_reference_equivalent": "user-supplied for deployment; unavailable in frozen audit",
            }
        )

    frame = pd.DataFrame.from_records(rows)
    output = analysis_dir / "break_even_reference_evals.csv"
    frame.to_csv(output, index=False, float_format="%.12g")
    shutil.copy2(output, package_dir / output.name)
    script_target = package_dir / Path(__file__).name
    if Path(__file__).resolve() != script_target.resolve():
        shutil.copy2(Path(__file__).resolve(), script_target)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
