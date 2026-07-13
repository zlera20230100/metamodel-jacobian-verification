#!/usr/bin/env python3
"""Repeated split-sensitivity audit for external score calibration.

The analysis uses only the frozen per-member and reference-gradient arrays.  It
does not retrain a metamodel or call a reference simulator.  For each benchmark,
the script makes 200 fixed-seed 50/50 query-cluster splits.  Thresholds are
chosen on the calibration half and evaluated once on the held-out half.

Two multiplicity conventions are reported side by side:

* ``per_score_10``: the score is prespecified and Bonferroni adjustment covers
  its ten candidate intended coverages;
* ``across_scores_40``: a sensitivity analysis that uses alpha/40 to cover four
  candidate scores times ten intended coverages.

Repeated splits measure sensitivity to the particular split.  They are not
independent replications and are not a distribution-free risk guarantee.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


METHODS = ("mean_aligned_sign", "sign_agreement", "snr", "magnitude")
METHOD_LABELS = {
    "mean_aligned_sign": "Mean-aligned sign",
    "sign_agreement": "Modal sign agreement",
    "snr": "SNR |mean|/sd",
    "magnitude": "Magnitude |mean|",
}
CORRECTIONS = {
    "per_score_10": 10,
    "across_scores_40": 40,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    if here.name.lower() == "reproducibility_update":
        package_dir = here
        analysis_dir = here.parent / "analysis"
    else:
        analysis_dir = here
        package_dir = here.parent / "reproducibility_update"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=analysis_dir)
    parser.add_argument("--package-dir", type=Path, default=package_dir)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--risk-target", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    return parser.parse_args()


def load_baseline_module(script_dir: Path):
    path = script_dir / "analyze_external_baselines.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("simpat_external_baselines", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def operating_metrics(trusted: np.ndarray, correct: np.ndarray) -> tuple[float, float, int, int]:
    accepted = int(trusted.sum())
    total = int(trusted.size)
    false_accepted = int((trusted & ~correct).sum())
    coverage = accepted / total if total else float("nan")
    risk = false_accepted / accepted if accepted else float("nan")
    return float(coverage), float(risk), accepted, false_accepted


def calibration_candidates(
    scores: np.ndarray,
    correct: np.ndarray,
    bootstrap_indices: np.ndarray,
    risk_target: float,
    family_size: int,
) -> list[dict]:
    intended_coverages = np.arange(0.1, 1.01, 0.1)
    adjusted_quantile = 1.0 - 0.05 / family_size
    flattened = scores.reshape(-1)
    descending = np.sort(flattened)[::-1]
    candidates: list[dict] = []
    seen: set[bytes] = set()

    for intended in intended_coverages:
        target = max(1, int(math.ceil(float(intended) * flattened.size)))
        threshold = float(descending[min(target - 1, descending.size - 1)])
        key = np.float64(threshold).tobytes()
        if key in seen:
            continue
        seen.add(key)

        trusted = scores >= threshold
        fp_by_query = (trusted & ~correct).sum(axis=1)
        accepted_by_query = trusted.sum(axis=1)
        fp_draw = fp_by_query[bootstrap_indices].sum(axis=1)
        accepted_draw = accepted_by_query[bootstrap_indices].sum(axis=1)
        risk_draw = np.divide(
            fp_draw,
            accepted_draw,
            out=np.full(len(bootstrap_indices), np.nan, dtype=float),
            where=accepted_draw > 0,
        )
        finite = risk_draw[np.isfinite(risk_draw)]
        risk_upper = float(np.quantile(finite, adjusted_quantile)) if finite.size else float("nan")
        coverage, risk, accepted, false_accepted = operating_metrics(trusted, correct)
        candidates.append(
            {
                "intended_coverage": float(intended),
                "threshold": threshold,
                "calibration_coverage": coverage,
                "calibration_risk": risk,
                "calibration_risk_upper": risk_upper,
                "calibration_accepted": accepted,
                "calibration_false_accepted": false_accepted,
                "passes": bool(np.isfinite(risk_upper) and risk_upper <= risk_target),
            }
        )
    return candidates


def run_one_split(
    dataset: dict,
    dataset_index: int,
    split_index: int,
    args: argparse.Namespace,
) -> list[dict]:
    nq = int(dataset["nq"])
    seed_sequence = np.random.SeedSequence([args.seed, dataset_index, split_index])
    split_seed, bootstrap_seed = seed_sequence.spawn(2)
    split_rng = np.random.default_rng(split_seed)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    permutation = split_rng.permutation(nq)
    n_cal = int(round(args.calibration_fraction * nq))
    n_cal = max(2, min(nq - 2, n_cal))
    calibration_indices = permutation[:n_cal]
    test_indices = permutation[n_cal:]
    bootstrap_indices = bootstrap_rng.integers(
        0, n_cal, size=(args.bootstrap, n_cal), endpoint=False
    )

    records: list[dict] = []
    for method in METHODS:
        calibration_scores = dataset["scores"][method][calibration_indices]
        calibration_correct = dataset["correct"][calibration_indices]
        test_scores = dataset["scores"][method][test_indices]
        test_correct = dataset["correct"][test_indices]

        for correction, family_size in CORRECTIONS.items():
            candidates = calibration_candidates(
                calibration_scores,
                calibration_correct,
                bootstrap_indices,
                args.risk_target,
                family_size,
            )
            feasible = [candidate for candidate in candidates if candidate["passes"]]
            selected = (
                max(feasible, key=lambda candidate: candidate["calibration_coverage"])
                if feasible
                else None
            )
            base = {
                "dataset": dataset["spec"].name,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "multiplicity": correction,
                "family_size": family_size,
                "split_index": split_index,
                "base_seed": args.seed,
                "calibration_queries": n_cal,
                "test_queries": nq - n_cal,
                "calibration_query_indices": ";".join(map(str, calibration_indices.tolist())),
                "test_query_indices": ";".join(map(str, test_indices.tolist())),
                "risk_target": args.risk_target,
                "bootstrap_replicates": args.bootstrap,
            }
            if selected is None:
                records.append(
                    {
                        **base,
                        "status": "abstain",
                        "selected_threshold": float("nan"),
                        "calibration_coverage": float("nan"),
                        "calibration_risk": float("nan"),
                        "calibration_risk_upper": float("nan"),
                        "test_coverage": float("nan"),
                        "test_risk": float("nan"),
                        "test_accepted": 0,
                        "test_false_accepted": 0,
                        "test_risk_at_or_below_target": False,
                    }
                )
                continue

            threshold = selected["threshold"]
            test_trusted = test_scores >= threshold
            test_coverage, test_risk, test_accepted, test_false_accepted = operating_metrics(
                test_trusted, test_correct
            )
            records.append(
                {
                    **base,
                    "status": "selected",
                    "selected_threshold": threshold,
                    "calibration_coverage": selected["calibration_coverage"],
                    "calibration_risk": selected["calibration_risk"],
                    "calibration_risk_upper": selected["calibration_risk_upper"],
                    "test_coverage": test_coverage,
                    "test_risk": test_risk,
                    "test_accepted": test_accepted,
                    "test_false_accepted": test_false_accepted,
                    "test_risk_at_or_below_target": bool(
                        np.isfinite(test_risk) and test_risk <= args.risk_target
                    ),
                }
            )
    return records


def quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else float("nan")


def summarize(runs: pd.DataFrame, risk_target: float) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["dataset", "method", "method_label", "multiplicity", "family_size"]
    for key, frame in runs.groupby(keys, sort=False):
        selected = frame[frame["status"] == "selected"]
        n_splits = len(frame)
        n_selected = len(selected)
        test_pass = int((selected["test_risk"] <= risk_target).sum()) if n_selected else 0
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_splits": n_splits,
                "n_selected": n_selected,
                "selection_rate": n_selected / n_splits,
                "abstention_rate": 1.0 - n_selected / n_splits,
                "test_risk_pass_rate_conditional": test_pass / n_selected if n_selected else float("nan"),
                "test_risk_exceed_rate_conditional": 1.0 - test_pass / n_selected if n_selected else float("nan"),
                "test_risk_pass_rate_all_splits": test_pass / n_splits,
                "coverage_q10": quantile(selected["test_coverage"], 0.10),
                "coverage_median": quantile(selected["test_coverage"], 0.50),
                "coverage_q90": quantile(selected["test_coverage"], 0.90),
                "risk_q10": quantile(selected["test_risk"], 0.10),
                "risk_median": quantile(selected["test_risk"], 0.50),
                "risk_q90": quantile(selected["test_risk"], 0.90),
                "threshold_q10": quantile(selected["selected_threshold"], 0.10),
                "threshold_median": quantile(selected["selected_threshold"], 0.50),
                "threshold_q90": quantile(selected["selected_threshold"], 0.90),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_csv(frame: pd.DataFrame, filename: str, destinations: list[Path]) -> None:
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / filename, index=False, float_format="%.10g")


def main() -> None:
    args = parse_args()
    if args.splits < 10:
        raise ValueError("Use at least 10 repeated splits")
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1000 cluster-bootstrap replicates")
    if not 0.1 <= args.calibration_fraction <= 0.9:
        raise ValueError("--calibration-fraction must be in [0.1, 0.9]")
    if not 0.0 < args.risk_target < 1.0:
        raise ValueError("--risk-target must be in (0, 1)")

    here = Path(__file__).resolve().parent
    baseline = load_baseline_module(here)
    package_dir = args.package_dir.resolve()
    datasets = [baseline.load_dataset(package_dir, spec) for spec in baseline.DATASETS]

    records: list[dict] = []
    for dataset_index, dataset in enumerate(datasets):
        for split_index in range(args.splits):
            records.extend(run_one_split(dataset, dataset_index, split_index, args))
    runs = pd.DataFrame.from_records(records)
    summary = summarize(runs, args.risk_target)

    destinations = [args.analysis_dir.resolve(), package_dir]
    write_csv(runs, "repeated_split_calibration_runs.csv", destinations)
    write_csv(summary, "repeated_split_calibration_summary.csv", destinations)

    print(f"Wrote {len(runs):,} run rows and {len(summary):,} summary rows")
    for _, row in summary.iterrows():
        print(
            f"{row['dataset']} | {row['method']} | {row['multiplicity']}: "
            f"selected={row['selection_rate']:.3f}, "
            f"coverage median={row['coverage_median']:.3f}, "
            f"risk median={row['risk_median']:.3f}, "
            f"test pass|selected={row['test_risk_pass_rate_conditional']:.3f}"
        )


if __name__ == "__main__":
    main()
