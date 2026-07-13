#!/usr/bin/env python3
"""Audit exact score ties and their effect on the correction frontier.

The script uses the frozen external per-member gradients only.  It reports tie
prevalence for all four calibration scores.  For the mean-aligned sign score it
then compares 1,000 truth-blind random tie resolutions with deterministic
secondary rules.  Lower primary score is always checked first; the secondary
rule is consulted only for exact primary-score ties.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


METHODS = ("mean_aligned_sign", "sign_agreement", "snr", "magnitude")


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
    parser.add_argument("--random-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_baseline_module(script_dir: Path):
    path = script_dir / "analyze_external_baselines.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("simpat_external_baselines_ties", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tie_prevalence(dataset: dict, method: str) -> dict:
    scores = dataset["scores"][method]
    nq, k = scores.shape
    any_tie = []
    unique_counts = []
    pair_tie_fractions = []
    max_group_sizes = []
    tied_components = []
    for row in scores:
        _, counts = np.unique(row, return_counts=True)
        unique_counts.append(len(counts))
        any_tie.append(bool(np.any(counts > 1)))
        pair_tie_fractions.append(
            float(np.sum(counts * (counts - 1) / 2) / (k * (k - 1) / 2))
            if k > 1
            else 0.0
        )
        max_group_sizes.append(int(counts.max()))
        tied_components.append(int(counts[counts > 1].sum()))
    return {
        "dataset": dataset["spec"].name,
        "method": method,
        "NQ": nq,
        "K": k,
        "queries_with_any_tie": int(np.sum(any_tie)),
        "query_any_tie_fraction": float(np.mean(any_tie)),
        "average_unique_scores_per_query": float(np.mean(unique_counts)),
        "average_pair_tie_fraction": float(np.mean(pair_tie_fractions)),
        "average_largest_tie_group": float(np.mean(max_group_sizes)),
        "average_components_in_nontrivial_ties": float(np.mean(tied_components)),
    }


def order_for_rule(dataset: dict, query: int, rule: str, rng: np.random.Generator | None) -> np.ndarray:
    primary = dataset["scores"]["mean_aligned_sign"][query]
    k = dataset["k"]
    if rule == "random":
        if rng is None:
            raise ValueError("random rule requires an RNG")
        secondary = rng.random(k)
    elif rule == "low_snr_first":
        secondary = dataset["scores"]["snr"][query]
    elif rule == "high_snr_first":
        secondary = -dataset["scores"]["snr"][query]
    elif rule == "low_magnitude_first":
        secondary = dataset["scores"]["magnitude"][query]
    elif rule == "fixed_index":
        secondary = np.arange(k, dtype=float)
    else:
        raise KeyError(rule)
    return np.lexsort((secondary, primary))


def per_query_frontier_area(
    dataset: dict, rule: str, rng: np.random.Generator | None = None
) -> np.ndarray:
    mean_gradient = dataset["mean"]
    reference = dataset["reference"]
    nq, k = mean_gradient.shape
    normalized_budget = np.arange(k + 1, dtype=float) / k
    areas = np.zeros(nq, dtype=float)
    for query in range(nq):
        order = order_for_rule(dataset, query, rule, rng)
        hybrid = mean_gradient[query].copy()
        errors = np.zeros(k + 1, dtype=float)
        denominator = np.linalg.norm(reference[query]) + 1e-12
        errors[0] = np.linalg.norm(hybrid - reference[query]) / denominator
        for budget, component in enumerate(order, start=1):
            hybrid[component] = reference[query, component]
            errors[budget] = np.linalg.norm(hybrid - reference[query]) / denominator
        areas[query] = np.trapezoid(errors, normalized_budget)
    return areas


def frontier_records(dataset: dict, repetitions: int, seed: int, dataset_index: int) -> list[dict]:
    records: list[dict] = []
    random_areas = np.zeros(repetitions, dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([seed, dataset_index]))
    for repetition in range(repetitions):
        random_areas[repetition] = per_query_frontier_area(dataset, "random", rng).mean()
    records.append(
        {
            "dataset": dataset["spec"].name,
            "primary_score": "mean_aligned_sign",
            "tie_rule": "random",
            "random_repetitions": repetitions,
            "frontier_area": float(random_areas.mean()),
            "frontier_area_q025": float(np.quantile(random_areas, 0.025)),
            "frontier_area_median": float(np.median(random_areas)),
            "frontier_area_q975": float(np.quantile(random_areas, 0.975)),
            "frontier_area_sd_across_reorderings": float(random_areas.std(ddof=1)),
        }
    )
    for rule in ("low_snr_first", "high_snr_first", "low_magnitude_first", "fixed_index"):
        areas = per_query_frontier_area(dataset, rule)
        records.append(
            {
                "dataset": dataset["spec"].name,
                "primary_score": "mean_aligned_sign",
                "tie_rule": rule,
                "random_repetitions": 0,
                "frontier_area": float(areas.mean()),
                "frontier_area_q025": float("nan"),
                "frontier_area_median": float("nan"),
                "frontier_area_q975": float("nan"),
                "frontier_area_sd_across_reorderings": float("nan"),
            }
        )
    return records


def write_csv(frame: pd.DataFrame, filename: str, destinations: list[Path]) -> None:
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / filename, index=False, float_format="%.10g")


def main() -> None:
    args = parse_args()
    if args.random_repetitions < 100:
        raise ValueError("Use at least 100 random tie reorderings")
    here = Path(__file__).resolve().parent
    baseline = load_baseline_module(here)
    package_dir = args.package_dir.resolve()
    datasets = [baseline.load_dataset(package_dir, spec) for spec in baseline.DATASETS]

    prevalence = pd.DataFrame.from_records(
        [tie_prevalence(dataset, method) for dataset in datasets for method in METHODS]
    )
    frontier = pd.DataFrame.from_records(
        record
        for dataset_index, dataset in enumerate(datasets)
        for record in frontier_records(
            dataset, args.random_repetitions, args.seed, dataset_index
        )
    )

    destinations = [args.analysis_dir.resolve(), package_dir]
    write_csv(prevalence, "tie_prevalence.csv", destinations)
    write_csv(frontier, "tie_rule_frontier_area.csv", destinations)

    print(f"Wrote {len(prevalence)} tie-prevalence rows and {len(frontier)} frontier rows")
    for _, row in frontier.iterrows():
        if row["tie_rule"] == "random":
            print(
                f"{row['dataset']} | random: {row['frontier_area']:.6f} "
                f"[{row['frontier_area_q025']:.6f}, {row['frontier_area_q975']:.6f}]"
            )
        else:
            print(f"{row['dataset']} | {row['tie_rule']}: {row['frontier_area']:.6f}")


if __name__ == "__main__":
    main()
