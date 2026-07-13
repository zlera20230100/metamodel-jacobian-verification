#!/usr/bin/env python3
"""Head-to-head external benchmark audit for sign, SNR, and magnitude scores.

Inputs are the validated per-member gradient artifacts produced by
``rebuild_external_member_gradients.py``.  All methods use the same query points,
member gradients, ensemble-mean sign-correctness label, query-cluster resamples,
calibration/test split, and finite-difference correction budgets.

The script writes detailed CSV/Markdown results to ``analysis`` and mirrors them
into the submission ``reproducibility_update`` directory.  Figures and frozen
artifacts live in ``reproducibility_update``.  It never edits the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    short: str
    filename: str


DATASETS = (
    DatasetSpec(
        "tmm_easy",
        "TMM easy (near-saturated)",
        "TMM easy",
        "external_members_tmm_easy.npz",
    ),
    DatasetSpec(
        "tmm_stressed",
        "TMM stressed",
        "TMM stressed",
        "external_members_tmm_stressed.npz",
    ),
    DatasetSpec(
        "heat_poisson_stressed",
        "Heat/Poisson stressed",
        "Heat/Poisson",
        "external_members_heat_poisson_stressed.npz",
    ),
)

PRIMARY_METHODS = ("mean_aligned_sign", "snr", "magnitude")
CALIBRATION_METHODS = ("mean_aligned_sign", "sign_agreement", "snr", "magnitude")
ALLOCATOR_METHODS = CALIBRATION_METHODS
ALL_METHODS = (*CALIBRATION_METHODS, "inverse_raw_spread")
METHOD_LABELS = {
    "mean_aligned_sign": "Mean-aligned sign",
    "sign_agreement": "Modal sign agreement",
    "snr": r"SNR $|\bar g|/s$",
    "magnitude": r"Magnitude $|\bar g|$",
    "inverse_raw_spread": r"Low raw spread $-s$",
}
METHOD_COLORS = {
    "mean_aligned_sign": "#3775BA",
    "sign_agreement": "#6B9AC4",
    "snr": "#A9C5DF",
    "magnitude": "#DCE8F1",
    "inverse_raw_spread": "#767676",
}
METHOD_MARKERS = {
    "mean_aligned_sign": "o",
    "sign_agreement": "x",
    "snr": "s",
    "magnitude": "^",
    "inverse_raw_spread": "D",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "reproducibility_update":
        package_dir = script_dir
        analysis_dir = script_dir.parent / "analysis"
    else:
        analysis_dir = script_dir
        package_dir = script_dir.parent / "reproducibility_update"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=analysis_dir)
    parser.add_argument("--package-dir", type=Path, default=package_dir)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--tie-repetitions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--risk-target", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def load_dataset(package_dir: Path, spec: DatasetSpec) -> dict:
    path = package_dir / spec.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run rebuild_external_member_gradients.py first"
        )
    with np.load(path, allow_pickle=False) as z:
        members = np.asarray(z["member_gradients"], dtype=float)
        reference = np.asarray(z["reference_gradients"], dtype=float)
        stored_mean = np.asarray(z["mean_gradients"], dtype=float)
        stored_agreement = np.asarray(z["sign_agreement"], dtype=float)
        stored_correct = np.asarray(z["mean_sign_correct"], dtype=np.int8)
        headline_auc = float(np.asarray(z["headline_auc"]).item())
        headline_cosine = float(np.asarray(z["headline_cosine"]).item())
        frozen_source = str(np.asarray(z["frozen_source_file"]).item())
        frozen_hash = str(np.asarray(z["frozen_source_sha256"]).item())
        nq = int(np.asarray(z["NQ"]).item())
        k = int(np.asarray(z["K"]).item())
        m = int(np.asarray(z["M"]).item())

    if members.shape != (nq, m, k) or reference.shape != (nq, k):
        raise ValueError(f"{path.name}: inconsistent gradient shapes")
    mean_gradient = members.mean(axis=1)
    spread = members.std(axis=1, ddof=0)
    agreement = np.maximum((members > 0).mean(axis=1), (members < 0).mean(axis=1))
    mean_sign = np.sign(mean_gradient)
    mean_aligned = (np.sign(members) == mean_sign[:, None, :]).mean(axis=1)
    # A zero ensemble mean has no actionable direction, so it must not be accepted.
    mean_aligned[mean_sign == 0] = 0.0
    correct = (np.sign(mean_gradient) == np.sign(reference)).astype(np.int8)
    if not np.array_equal(mean_gradient, stored_mean):
        raise AssertionError(f"{path.name}: stored ensemble mean is inconsistent")
    if not np.array_equal(agreement, stored_agreement):
        raise AssertionError(f"{path.name}: stored sign agreement is inconsistent")
    if not np.array_equal(correct, stored_correct):
        raise AssertionError(f"{path.name}: stored correctness labels are inconsistent")

    scores = {
        "mean_aligned_sign": mean_aligned,
        "sign_agreement": agreement,
        "snr": np.abs(mean_gradient) / (spread + 1e-12),
        "magnitude": np.abs(mean_gradient),
        "inverse_raw_spread": -spread,
    }
    return {
        "spec": spec,
        "path": path,
        "sha256": sha256(path),
        "members": members,
        "mean": mean_gradient,
        "reference": reference,
        "correct": correct.astype(bool),
        "scores": scores,
        "spread": spread,
        "nq": nq,
        "k": k,
        "m": m,
        "headline_auc": headline_auc,
        "headline_cosine": headline_cosine,
        "frozen_source": frozen_source,
        "frozen_hash": frozen_hash,
    }


def cluster_auc_analysis(
    dataset: dict,
    bootstrap_indices: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    y = dataset["correct"]
    draws = {method: np.full(len(bootstrap_indices), np.nan) for method in ALL_METHODS}
    for draw_index, query_indices in enumerate(bootstrap_indices):
        labels = y[query_indices].reshape(-1)
        if labels.all() or (~labels).all():
            continue
        for method in ALL_METHODS:
            scores = dataset["scores"][method][query_indices].reshape(-1)
            draws[method][draw_index] = auc_binary(labels, scores)

    point_sign = auc_binary(y, dataset["scores"]["sign_agreement"])
    point_mean_aligned = auc_binary(y, dataset["scores"]["mean_aligned_sign"])
    records = []
    for method in ALL_METHODS:
        point = auc_binary(y, dataset["scores"][method])
        lo, hi = percentile_interval(draws[method])
        delta = draws[method] - draws["sign_agreement"]
        delta_lo, delta_hi = percentile_interval(delta)
        delta_mean = draws[method] - draws["mean_aligned_sign"]
        delta_mean_lo, delta_mean_hi = percentile_interval(delta_mean)
        records.append(
            {
                "dataset": dataset["spec"].name,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "auc": point,
                "auc_cluster_low": lo,
                "auc_cluster_high": hi,
                "delta_auc_vs_sign": point - point_sign,
                "delta_cluster_low": delta_lo,
                "delta_cluster_high": delta_hi,
                "delta_auc_vs_mean_aligned": point - point_mean_aligned,
                "delta_mean_aligned_cluster_low": delta_mean_lo,
                "delta_mean_aligned_cluster_high": delta_mean_hi,
                "bootstrap_valid": int(np.isfinite(draws[method]).sum()),
                "NQ": dataset["nq"],
                "K": dataset["k"],
                "n_components": dataset["nq"] * dataset["k"],
                "n_sign_correct": int(y.sum()),
                "n_sign_wrong": int((~y).sum()),
            }
        )
    return records, draws


def operating_metrics(trusted: np.ndarray, correct: np.ndarray) -> dict:
    tp = int((trusted & correct).sum())
    fp = int((trusted & ~correct).sum())
    fn = int((~trusted & correct).sum())
    tn = int((~trusted & ~correct).sum())
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "coverage": safe_ratio(tp + fp, correct.size),
        "false_trust_risk": safe_ratio(fp, tp + fp),
        "sensitivity": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
    }


def bootstrap_operating(
    trusted: np.ndarray,
    correct: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, tuple[float, float]]:
    metrics = {name: [] for name in ("coverage", "false_trust_risk", "sensitivity", "specificity")}
    for query_indices in bootstrap_indices:
        result = operating_metrics(trusted[query_indices], correct[query_indices])
        for name in metrics:
            metrics[name].append(result[name])
    return {name: percentile_interval(np.asarray(values)) for name, values in metrics.items()}


def threshold_closest_to_count(scores: np.ndarray, target_count: int) -> float:
    flat = np.asarray(scores, dtype=float).reshape(-1)
    unique = np.unique(flat)
    counts = np.array([(flat >= threshold).sum() for threshold in unique])
    distance = np.abs(counts - target_count)
    candidates = np.flatnonzero(distance == distance.min())
    # Prefer the more conservative (smaller accepted set) solution on an exact tie.
    chosen = candidates[np.argmin(counts[candidates])]
    return float(unique[chosen])


def matched_coverage_operating(
    dataset: dict,
    bootstrap_indices: np.ndarray,
) -> list[dict]:
    correct = dataset["correct"]
    sign_trusted = dataset["scores"]["mean_aligned_sign"] >= 0.9 - 1e-12
    target_count = int(sign_trusted.sum())
    records = []
    for method in ALL_METHODS:
        if method in ("mean_aligned_sign", "sign_agreement"):
            threshold = 0.9
            threshold_rule = "fixed sign threshold 0.9"
        else:
            threshold = threshold_closest_to_count(dataset["scores"][method], target_count)
            threshold_rule = "descriptive full-sample match to sign coverage"
        trusted = dataset["scores"][method] >= threshold
        point = operating_metrics(trusted, correct)
        intervals = bootstrap_operating(trusted, correct, bootstrap_indices)
        record = {
            "dataset": dataset["spec"].name,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "threshold": threshold,
            "threshold_rule": threshold_rule,
            "target_sign_accepted_count": target_count,
            "accepted_count": int(trusted.sum()),
            **point,
        }
        record["fd_component_checks_mean"] = dataset["k"] * (1.0 - point["coverage"])
        record["reference_evaluations_mean"] = 2.0 * record["fd_component_checks_mean"]
        record["all_fd_reference_evaluations"] = 2 * dataset["k"]
        record["reference_evaluation_savings_fraction"] = point["coverage"]
        for metric, (lo, hi) in intervals.items():
            record[f"{metric}_cluster_low"] = lo
            record[f"{metric}_cluster_high"] = hi
        records.append(record)
    return records


def split_calibration(
    dataset: dict,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
    risk_target: float,
) -> list[dict]:
    """Common split and common predeclared coverage grid for all primary scores."""

    candidate_coverages = np.arange(0.1, 1.01, 0.1)
    alpha = 0.05
    adjusted_quantile = 1.0 - alpha / len(candidate_coverages)
    cal_boot = rng.integers(
        0,
        len(calibration_indices),
        size=(bootstrap, len(calibration_indices)),
        endpoint=False,
    )
    test_boot = rng.integers(
        0,
        len(test_indices),
        size=(bootstrap, len(test_indices)),
        endpoint=False,
    )
    records = []
    for method in CALIBRATION_METHODS:
        cal_scores = dataset["scores"][method][calibration_indices]
        cal_correct = dataset["correct"][calibration_indices]
        test_scores = dataset["scores"][method][test_indices]
        test_correct = dataset["correct"][test_indices]
        candidates = []
        seen_thresholds = set()
        flat_scores = cal_scores.reshape(-1)
        descending = np.sort(flat_scores)[::-1]
        for intended_coverage in candidate_coverages:
            accepted_target = max(1, int(math.ceil(intended_coverage * flat_scores.size)))
            threshold = float(descending[min(accepted_target - 1, descending.size - 1)])
            threshold_key = np.float64(threshold).tobytes()
            if threshold_key in seen_thresholds:
                continue
            seen_thresholds.add(threshold_key)
            trusted = cal_scores >= threshold
            fp_q = (trusted & ~cal_correct).sum(axis=1)
            trusted_q = trusted.sum(axis=1)
            fp_draw = fp_q[cal_boot].sum(axis=1)
            trusted_draw = trusted_q[cal_boot].sum(axis=1)
            risk_draw = np.divide(
                fp_draw,
                trusted_draw,
                out=np.full(bootstrap, np.nan, dtype=float),
                where=trusted_draw > 0,
            )
            finite = risk_draw[np.isfinite(risk_draw)]
            risk_upper = (
                float(np.quantile(finite, adjusted_quantile))
                if finite.size
                else float("nan")
            )
            point = operating_metrics(trusted, cal_correct)
            candidates.append(
                {
                    "threshold": threshold,
                    "intended_coverage": intended_coverage,
                    "calibration_coverage": point["coverage"],
                    "calibration_risk": point["false_trust_risk"],
                    "calibration_risk_upper": risk_upper,
                    "passes": bool(np.isfinite(risk_upper) and risk_upper <= risk_target),
                }
            )

        feasible = [candidate for candidate in candidates if candidate["passes"]]
        selected = max(feasible, key=lambda item: item["calibration_coverage"]) if feasible else None
        base = {
            "dataset": dataset["spec"].name,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "risk_target": risk_target,
            "candidate_coverage_grid": "0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9;1.0",
            "familywise_alpha": alpha,
            "calibration_queries": len(calibration_indices),
            "test_queries": len(test_indices),
            "calibration_query_indices": ";".join(map(str, calibration_indices.tolist())),
            "test_query_indices": ";".join(map(str, test_indices.tolist())),
        }
        if selected is None:
            records.append(
                {
                    **base,
                    "status": "no_candidate_met_calibration_risk_bound",
                    "selected_threshold": float("nan"),
                    "calibration_coverage": float("nan"),
                    "calibration_risk": float("nan"),
                    "calibration_risk_upper": float("nan"),
                    "test_coverage": float("nan"),
                    "test_coverage_cluster_low": float("nan"),
                    "test_coverage_cluster_high": float("nan"),
                    "test_false_trust_risk": float("nan"),
                    "test_false_trust_risk_cluster_low": float("nan"),
                    "test_false_trust_risk_cluster_high": float("nan"),
                }
            )
            continue

        threshold = selected["threshold"]
        test_trusted = test_scores >= threshold
        test_point = operating_metrics(test_trusted, test_correct)
        test_intervals = bootstrap_operating(test_trusted, test_correct, test_boot)
        records.append(
            {
                **base,
                "status": "selected_on_calibration_evaluated_on_held_out_test",
                "selected_threshold": threshold,
                "calibration_coverage": selected["calibration_coverage"],
                "calibration_risk": selected["calibration_risk"],
                "calibration_risk_upper": selected["calibration_risk_upper"],
                "test_coverage": test_point["coverage"],
                "test_coverage_cluster_low": test_intervals["coverage"][0],
                "test_coverage_cluster_high": test_intervals["coverage"][1],
                "test_false_trust_risk": test_point["false_trust_risk"],
                "test_false_trust_risk_cluster_low": test_intervals["false_trust_risk"][0],
                "test_false_trust_risk_cluster_high": test_intervals["false_trust_risk"][1],
            }
        )
    return records


def expected_allocator_curves(
    dataset: dict,
    method: str,
    tie_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Per-query expected fixed-budget curves, randomizing only exact score ties."""

    scores = dataset["scores"][method]
    mean_gradient = dataset["mean"]
    reference = dataset["reference"]
    correct = dataset["correct"]
    nq, k = scores.shape
    relerr = np.zeros((nq, k + 1))
    cosine = np.zeros((nq, k + 1))
    false_risk = np.full((nq, k + 1), np.nan)

    for query in range(nq):
        repetitions = tie_repetitions if np.unique(scores[query]).size < k else 1
        relerr_sum = np.zeros(k + 1)
        cosine_sum = np.zeros(k + 1)
        risk_sum = np.zeros(k)
        for _ in range(repetitions):
            random_tie_key = rng.random(k)
            order = np.lexsort((random_tie_key, scores[query]))
            hybrid = mean_gradient[query].copy()
            wrong = ~correct[query]
            for budget in range(k + 1):
                if budget > 0:
                    component = order[budget - 1]
                    hybrid[component] = reference[query, component]
                relerr_sum[budget] += np.linalg.norm(hybrid - reference[query]) / (
                    np.linalg.norm(reference[query]) + 1e-12
                )
                cosine_sum[budget] += float(
                    hybrid @ reference[query]
                    / (
                        np.linalg.norm(hybrid) * np.linalg.norm(reference[query])
                        + 1e-12
                    )
                )
                if budget < k:
                    uncorrected = np.ones(k, dtype=bool)
                    uncorrected[order[:budget]] = False
                    risk_sum[budget] += wrong[uncorrected].mean()
        relerr[query] = relerr_sum / repetitions
        cosine[query] = cosine_sum / repetitions
        false_risk[query, :k] = risk_sum / repetitions
    return {"relerr": relerr, "cosine": cosine, "false_risk": false_risk}


def allocator_records(
    dataset: dict,
    method: str,
    curves: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
) -> tuple[list[dict], dict]:
    k = dataset["k"]
    records = []
    for budget in range(k + 1):
        record = {
            "dataset": dataset["spec"].name,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "B_component_checks": budget,
            "reference_evaluations": 2 * budget,
            "budget_fraction": budget / k,
            "accepted_coverage": (k - budget) / k,
        }
        for metric in ("relerr", "cosine", "false_risk"):
            values = curves[metric][:, budget]
            point = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            if np.isfinite(values).any():
                bootstrap_values = np.nanmean(values[bootstrap_indices], axis=1)
            else:
                bootstrap_values = np.full(len(bootstrap_indices), np.nan)
            lo, hi = percentile_interval(bootstrap_values)
            record[f"mean_{metric}"] = point
            record[f"{metric}_cluster_low"] = lo
            record[f"{metric}_cluster_high"] = hi
        records.append(record)

    normalized_budget = np.arange(k + 1) / k
    per_query_area = np.trapezoid(curves["relerr"], normalized_budget, axis=1)
    area_boot = per_query_area[bootstrap_indices].mean(axis=1)
    area_lo, area_hi = percentile_interval(area_boot)
    summary = {
        "dataset": dataset["spec"].name,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "allocator_relerr_area": float(per_query_area.mean()),
        "allocator_relerr_area_cluster_low": area_lo,
        "allocator_relerr_area_cluster_high": area_hi,
        "tie_repetitions": tie_repetitions_global,
    }
    return records, summary


def allocator_key_results(allocator_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prespecified fixed-budget rows and minimum budgets for cosine targets."""

    key_rows = []
    target_rows = []
    for spec in DATASETS:
        dataset_frame = allocator_frame[allocator_frame["dataset"] == spec.name]
        k = int(dataset_frame["B_component_checks"].max())
        key_budgets = sorted({1, int(round(0.25 * k)), int(round(0.50 * k)), int(round(0.75 * k))})
        key_budgets = [max(0, min(k, budget)) for budget in key_budgets]
        for method in ALLOCATOR_METHODS:
            method_frame = dataset_frame[dataset_frame["method"] == method].sort_values(
                "B_component_checks"
            )
            for budget in key_budgets:
                row = method_frame[method_frame["B_component_checks"] == budget].iloc[0]
                key_rows.append(
                    {
                        "dataset": spec.name,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "B_component_checks": budget,
                        "reference_evaluations": 2 * budget,
                        "budget_fraction": budget / k,
                        "mean_relerr": row["mean_relerr"],
                        "relerr_cluster_low": row["relerr_cluster_low"],
                        "relerr_cluster_high": row["relerr_cluster_high"],
                        "mean_cosine": row["mean_cosine"],
                        "mean_false_trust_risk": row["mean_false_risk"],
                    }
                )
            for cosine_target in (0.90, 0.95, 0.99):
                qualifying = method_frame[method_frame["mean_cosine"] >= cosine_target]
                if qualifying.empty:
                    budget = k
                    achieved = float(method_frame.iloc[-1]["mean_cosine"])
                else:
                    selected = qualifying.iloc[0]
                    budget = int(selected["B_component_checks"])
                    achieved = float(selected["mean_cosine"])
                target_rows.append(
                    {
                        "dataset": spec.name,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "cosine_target": cosine_target,
                        "minimum_B_component_checks": budget,
                        "minimum_reference_evaluations": 2 * budget,
                        "achieved_mean_cosine": achieved,
                        "K": k,
                        "all_fd_reference_evaluations": 2 * k,
                    }
                )
    return pd.DataFrame.from_records(key_rows), pd.DataFrame.from_records(target_rows)


def modal_mean_record(dataset: dict, tau: float = 0.9) -> dict:
    members = dataset["members"]
    mean_sign = np.sign(dataset["mean"])
    reference_sign = np.sign(dataset["reference"])
    positive = (members > 0).sum(axis=1)
    negative = (members < 0).sum(axis=1)
    modal_sign = np.sign(positive - negative)
    tie = modal_sign == 0
    mismatch = (~tie) & (modal_sign != mean_sign)
    mean_aligned = dataset["scores"]["mean_aligned_sign"]
    modal_agreement = dataset["scores"]["sign_agreement"]
    mean_correct = mean_sign == reference_sign
    modal_correct = modal_sign == reference_sign
    accepted_mean_aligned = mean_aligned >= tau - 1e-12
    accepted_modal = modal_agreement >= tau - 1e-12
    accepted = accepted_mean_aligned
    false_accept_mean = accepted & ~mean_correct
    false_accept_modal = accepted & ~modal_correct
    return {
        "dataset": dataset["spec"].name,
        "NQ": dataset["nq"],
        "K": dataset["k"],
        "n_components": dataset["nq"] * dataset["k"],
        "modal_ties": int(tie.sum()),
        "modal_tie_fraction": float(tie.mean()),
        "non_tie_modal_mean_mismatch": int(mismatch.sum()),
        "non_tie_modal_mean_mismatch_fraction": float(mismatch.mean()),
        "ensemble_mean_zero_components": int((mean_sign == 0).sum()),
        "exact_zero_member_gradients": int((members == 0).sum()),
        "mean_aligned_score_differs_from_modal": int(
            (mean_aligned != modal_agreement).sum()
        ),
        "tau": tau,
        "accepted_components": int(accepted.sum()),
        "tau_acceptance_set_difference": int(
            (accepted_mean_aligned != accepted_modal).sum()
        ),
        "mean_direction_false_acceptances": int(false_accept_mean.sum()),
        "modal_direction_false_acceptances": int(false_accept_modal.sum()),
        "false_acceptances_with_modal_mean_mismatch": int((false_accept_mean & mismatch).sum()),
        "mean_false_acceptances_rescued_by_modal": int(
            (false_accept_mean & modal_correct).sum()
        ),
        "mean_correct_accepted_but_modal_wrong": int(
            (accepted & mean_correct & ~modal_correct).sum()
        ),
    }


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.4,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_auc(auc_frame: pd.DataFrame, package_dir: Path) -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(7.25, 3.05))
    x = np.arange(len(DATASETS), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(ALL_METHODS))
    for offset, method in zip(offsets, ALL_METHODS):
        rows = []
        for spec in DATASETS:
            rows.append(
                auc_frame[
                    (auc_frame["dataset"] == spec.name) & (auc_frame["method"] == method)
                ].iloc[0]
            )
        y = np.array([row["auc"] for row in rows], dtype=float)
        lo = np.array([row["auc_cluster_low"] for row in rows], dtype=float)
        hi = np.array([row["auc_cluster_high"] for row in rows], dtype=float)
        ax.errorbar(
            x + offset,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markerfacecolor="white" if method == "inverse_raw_spread" else METHOD_COLORS[method],
            markeredgecolor=METHOD_COLORS[method],
            markersize=5.6,
            linewidth=1.25,
            capsize=2.3,
            label=METHOD_LABELS[method],
        )
    ax.axhline(0.5, color="#4D4D4D", ls="--", lw=0.8)
    ax.text(2.48, 0.507, "chance", ha="right", va="bottom", fontsize=7.5, color="#4D4D4D")
    ax.set_xticks(x, [spec.short for spec in DATASETS])
    ax.set_ylabel("AUC for ensemble-mean sign correctness")
    ax.set_ylim(0.15, 1.02)
    ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(
        "External-score discrimination with query-cluster 95% intervals",
        x=0.09,
        y=0.985,
        ha="left",
        fontsize=10.0,
    )
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.57, 0.94), fontsize=7.4)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.78)
    fig.savefig(package_dir / "fig_external_baseline_auc.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        package_dir / "fig_external_baseline_auc.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def plot_matched_coverage(operating_frame: pd.DataFrame, package_dir: Path) -> None:
    """Compare risk at sign-tau coverage and show the shared reference cost."""

    configure_plotting()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.25, 3.05), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(DATASETS), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(PRIMARY_METHODS))
    for offset, method in zip(offsets, PRIMARY_METHODS):
        rows = [
            operating_frame[
                (operating_frame["dataset"] == spec.name)
                & (operating_frame["method"] == method)
            ].iloc[0]
            for spec in DATASETS
        ]
        risk = np.array([row["false_trust_risk"] for row in rows], dtype=float)
        lo = np.array([row["false_trust_risk_cluster_low"] for row in rows], dtype=float)
        hi = np.array([row["false_trust_risk_cluster_high"] for row in rows], dtype=float)
        ax0.errorbar(
            x + offset,
            risk,
            yerr=np.vstack([risk - lo, hi - risk]),
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=METHOD_COLORS[method],
            markersize=5.6,
            linewidth=1.2,
            capsize=2.2,
            label=METHOD_LABELS[method],
        )
    ax0.set_xticks(x, [spec.short for spec in DATASETS])
    ax0.set_ylabel("Sign-wrong among accepted\n(false-trust risk)")
    ax0.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.set_ylim(-0.008, 0.19)
    ax0.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.8)
    ax0.set_title("(a) Same accepted coverage as sign threshold 0.9", loc="left")
    ax0.legend(loc="upper left", fontsize=7.6)

    sign_rows = [
        operating_frame[
            (operating_frame["dataset"] == spec.name)
            & (operating_frame["method"] == "mean_aligned_sign")
        ].iloc[0]
        for spec in DATASETS
    ]
    selective = np.array([row["reference_evaluations_mean"] for row in sign_rows], dtype=float)
    complete = np.array([row["all_fd_reference_evaluations"] for row in sign_rows], dtype=float)
    width = 0.35
    ax1.bar(
        x - width / 2,
        complete,
        width,
        color="#CFCECE",
        edgecolor="#4D4D4D",
        linewidth=0.8,
        hatch="///",
        label="Complete central FD",
    )
    bars = ax1.bar(
        x + width / 2,
        selective,
        width,
        color="#3775BA",
        edgecolor="#2A5B8A",
        linewidth=0.8,
        label="Correct rejected",
    )
    for bar, row in zip(bars, sign_rows):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{row['fd_component_checks_mean']:.2f} checks\n{fmt_pct(row['coverage'])} accepted",
            ha="center",
            va="bottom",
            fontsize=6.9,
            color="#2A5B8A",
        )
    ax1.set_xticks(x, [spec.short for spec in DATASETS])
    ax1.set_ylabel("Reference-objective evaluations per query")
    ax1.set_ylim(0, complete.max() * 1.20)
    ax1.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.8)
    ax1.set_title("(b) Shared cost at matched coverage", loc="left")
    ax1.legend(loc="upper left", fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.90)
    fig.savefig(
        package_dir / "fig_external_baseline_matched_coverage.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        package_dir / "fig_external_baseline_matched_coverage.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def plot_frontiers(allocator_frame: pd.DataFrame, package_dir: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.85), sharex="col")
    for column, spec in enumerate(DATASETS):
        top = axes[0, column]
        bottom = axes[1, column]
        for method in PRIMARY_METHODS:
            frame = allocator_frame[
                (allocator_frame["dataset"] == spec.name)
                & (allocator_frame["method"] == method)
            ].sort_values("B_component_checks")
            risk_frame = frame[frame["accepted_coverage"] > 0]
            x_risk = risk_frame["accepted_coverage"].to_numpy(float)
            risk = risk_frame["mean_false_risk"].to_numpy(float)
            risk_lo = risk_frame["false_risk_cluster_low"].to_numpy(float)
            risk_hi = risk_frame["false_risk_cluster_high"].to_numpy(float)
            order = np.argsort(x_risk)
            top.plot(
                x_risk[order],
                risk[order],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                ms=3.0,
                lw=1.35,
                label=METHOD_LABELS[method],
            )
            top.fill_between(
                x_risk[order],
                risk_lo[order],
                risk_hi[order],
                color=METHOD_COLORS[method],
                alpha=0.10,
                linewidth=0,
            )

            x_budget = frame["budget_fraction"].to_numpy(float)
            error = frame["mean_relerr"].to_numpy(float)
            error_lo = frame["relerr_cluster_low"].to_numpy(float)
            error_hi = frame["relerr_cluster_high"].to_numpy(float)
            bottom.plot(
                x_budget,
                error,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                ms=3.0,
                lw=1.35,
            )
            bottom.fill_between(
                x_budget,
                error_lo,
                error_hi,
                color=METHOD_COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
        top.set_title(spec.short)
        top.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        top.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.75)
        bottom.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.75)
        bottom.set_xlabel("Reference-gradient budget $B/K$")
        bottom.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0, 0].set_ylabel("Sign-wrong among accepted")
    axes[1, 0].set_ylabel(r"Assembled-gradient error $\|g-g^*\|/\|g^*\|$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.text(0.085, 0.955, "(a) Fixed-budget risk--coverage", fontsize=9.4)
    fig.text(0.085, 0.485, "(b) Fixed-budget correction error", fontsize=9.4)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.11, top=0.90, hspace=0.42, wspace=0.31)
    fig.savefig(
        package_dir / "fig_external_baseline_frontiers.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        package_dir / "fig_external_baseline_frontiers.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def fmt_pct(value: float, digits: int = 1) -> str:
    return "NA" if not np.isfinite(value) else f"{100 * value:.{digits}f}%"


def fmt_ci(point: float, lo: float, hi: float, digits: int = 3) -> str:
    if not (np.isfinite(point) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def markdown_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    analysis_dir: Path,
    package_dir: Path,
    datasets: list[dict],
    auc_frame: pd.DataFrame,
    operating_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    allocator_summary: pd.DataFrame,
    allocator_key_frame: pd.DataFrame,
    allocator_target_frame: pd.DataFrame,
    modal_frame: pd.DataFrame,
    bootstrap: int,
    tie_repetitions: int,
    seed: int,
    risk_target: float,
) -> None:
    manifest_rows = [
        (
            dataset["spec"].name,
            f"{dataset['nq']} x {dataset['m']} x {dataset['k']}",
            f"{dataset['headline_auc']:.6f}",
            f"{dataset['headline_cosine']:.6f}",
            dataset["path"].name,
            dataset["sha256"],
        )
        for dataset in datasets
    ]

    auc_rows = []
    for spec in DATASETS:
        for method in ALL_METHODS:
            row = auc_frame[
                (auc_frame["dataset"] == spec.name) & (auc_frame["method"] == method)
            ].iloc[0]
            auc_rows.append(
                (
                    spec.name,
                    METHOD_LABELS[method],
                    fmt_ci(row["auc"], row["auc_cluster_low"], row["auc_cluster_high"]),
                    fmt_ci(
                        row["delta_auc_vs_mean_aligned"],
                        row["delta_mean_aligned_cluster_low"],
                        row["delta_mean_aligned_cluster_high"],
                    ),
                )
            )

    operating_rows = []
    for spec in DATASETS:
        for method in CALIBRATION_METHODS:
            row = operating_frame[
                (operating_frame["dataset"] == spec.name)
                & (operating_frame["method"] == method)
            ].iloc[0]
            operating_rows.append(
                (
                    spec.name,
                    METHOD_LABELS[method],
                    f"{row['threshold']:.5g}",
                    fmt_pct(row["coverage"]),
                    fmt_ci(
                        row["false_trust_risk"],
                        row["false_trust_risk_cluster_low"],
                        row["false_trust_risk_cluster_high"],
                    ),
                    fmt_pct(row["specificity"]),
                    f"{row['fd_component_checks_mean']:.2f} / {row['reference_evaluations_mean']:.2f}",
                )
            )

    split_rows = []
    for spec in DATASETS:
        for method in CALIBRATION_METHODS:
            row = split_frame[
                (split_frame["dataset"] == spec.name) & (split_frame["method"] == method)
            ].iloc[0]
            if np.isfinite(row["selected_threshold"]):
                selected = f"{row['selected_threshold']:.5g}"
                calibration = (
                    f"coverage {fmt_pct(row['calibration_coverage'])}; "
                    f"risk {fmt_pct(row['calibration_risk'])}; "
                    f"upper {fmt_pct(row['calibration_risk_upper'])}"
                )
                test = (
                    f"coverage {fmt_pct(row['test_coverage'])}; "
                    f"risk {fmt_ci(row['test_false_trust_risk'], row['test_false_trust_risk_cluster_low'], row['test_false_trust_risk_cluster_high'])}"
                )
            else:
                selected, calibration, test = "none", "no candidate passed", "not evaluated"
            split_rows.append((spec.name, METHOD_LABELS[method], selected, calibration, test))

    allocator_rows = []
    for spec in DATASETS:
        subset = allocator_summary[allocator_summary["dataset"] == spec.name].sort_values(
            "allocator_relerr_area"
        )
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            allocator_rows.append(
                (
                    spec.name,
                    rank,
                    METHOD_LABELS[row["method"]],
                    fmt_ci(
                        row["allocator_relerr_area"],
                        row["allocator_relerr_area_cluster_low"],
                        row["allocator_relerr_area_cluster_high"],
                    ),
                )
            )

    allocator_key_rows = []
    for spec in DATASETS:
        subset = allocator_key_frame[allocator_key_frame["dataset"] == spec.name]
        available = sorted(subset["budget_fraction"].unique())
        chosen_fractions = []
        for target_fraction in (0.25, 0.50):
            chosen_fractions.append(min(available, key=lambda value: abs(value - target_fraction)))
        for budget_fraction in sorted(set(chosen_fractions)):
            for method in ALLOCATOR_METHODS:
                row = subset[
                    (subset["method"] == method)
                    & np.isclose(subset["budget_fraction"], budget_fraction)
                ].iloc[0]
                allocator_key_rows.append(
                    (
                        spec.name,
                        f"{int(row['B_component_checks'])} checks / {int(row['reference_evaluations'])} evals",
                        METHOD_LABELS[method],
                        fmt_ci(
                            row["mean_relerr"],
                            row["relerr_cluster_low"],
                            row["relerr_cluster_high"],
                        ),
                        f"{row['mean_cosine']:.3f}",
                    )
                )

    allocator_target_rows = []
    for spec in DATASETS:
        for target in (0.90, 0.95, 0.99):
            for method in ALLOCATOR_METHODS:
                row = allocator_target_frame[
                    (allocator_target_frame["dataset"] == spec.name)
                    & (allocator_target_frame["method"] == method)
                    & np.isclose(allocator_target_frame["cosine_target"], target)
                ].iloc[0]
                allocator_target_rows.append(
                    (
                        spec.name,
                        f"{target:.2f}",
                        METHOD_LABELS[method],
                        int(row["minimum_B_component_checks"]),
                        int(row["minimum_reference_evaluations"]),
                    )
                )

    modal_rows = []
    for _, row in modal_frame.iterrows():
        modal_rows.append(
            (
                row["dataset"],
                f"{int(row['modal_ties'])} ({fmt_pct(row['modal_tie_fraction'])})",
                f"{int(row['non_tie_modal_mean_mismatch'])} ({fmt_pct(row['non_tie_modal_mean_mismatch_fraction'])})",
                int(row["mean_direction_false_acceptances"]),
                int(row["false_acceptances_with_modal_mean_mismatch"]),
                int(row["mean_false_acceptances_rescued_by_modal"]),
                int(row["ensemble_mean_zero_components"]),
                int(row["exact_zero_member_gradients"]),
                int(row["mean_aligned_score_differs_from_modal"]),
                int(row["tau_acceptance_set_difference"]),
            )
        )

    winners = (
        auc_frame[auc_frame["method"].isin(PRIMARY_METHODS)]
        .sort_values(["dataset", "auc"], ascending=[True, False])
        .groupby("dataset", sort=False)
        .first()
    )
    mean_aligned_never_best = all(
        row["method"] != "mean_aligned_sign" for _, row in winners.iterrows()
    )
    snr_beats_sign_count = int(
        (
            auc_frame[auc_frame["method"] == "snr"].set_index("dataset")["auc"]
            > auc_frame[auc_frame["method"] == "mean_aligned_sign"].set_index("dataset")["auc"]
        ).sum()
    )
    magnitude_beats_sign_count = int(
        (
            auc_frame[auc_frame["method"] == "magnitude"].set_index("dataset")["auc"]
            > auc_frame[auc_frame["method"] == "mean_aligned_sign"].set_index("dataset")["auc"]
        ).sum()
    )

    conclusion = (
        f"SNR exceeded mean-aligned sign agreement in {snr_beats_sign_count}/3 external benchmarks, and raw "
        f"magnitude exceeded it in {magnitude_beats_sign_count}/3. "
    )
    if mean_aligned_never_best:
        conclusion += "Mean-aligned sign agreement was not the highest-AUC primary score in any external benchmark."
    else:
        conclusion += "Mean-aligned sign agreement remained best in at least one external benchmark."

    report = f"""# External SNR and magnitude baseline audit

## Audit contract

All methods below use the same per-member gradients, ensemble-mean sign-correctness labels, query
clusters, bootstrap resamples, calibration/test split, and finite-difference correction budget.
The rebuilt member-gradient artifacts were released only after exact reproduction of the archived
`all_sa` and `all_correct` arrays and agreement with the frozen AUC and zero-budget cosine.

{markdown_table(
    ["benchmark", "G shape (Q x M x K)", "frozen sign AUC", "zero-budget cosine", "artifact", "SHA-256"],
    manifest_rows,
)}

## Discrimination

The outcome is whether the **ensemble-mean gradient sign** matches the numerical reference sign.
Intervals and AUC differences use the same {bootstrap:,} query-cluster bootstrap resamples. SNR is
`abs(mean)/std` with population standard deviation (`ddof=0`), matching the archived synthetic
baseline. CV is not duplicated because its inverse has exactly the same ordering as SNR. The raw
spread diagnostic is oriented as `-std`, so larger values mean more trusted. The archived modal
score is `max(fraction positive, fraction negative)`. The action-aligned replacement is the
fraction of member signs equal to `sign(ensemble mean)`; a zero ensemble mean receives score zero.

{markdown_table(
    ["benchmark", "score", "AUC (cluster 95% CI)", "AUC difference versus mean-aligned sign (paired 95% CI)"],
    auc_rows,
)}

**Result:** {conclusion}
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

{markdown_table(
    ["benchmark", "score", "threshold", "coverage", "false-trust risk (cluster 95% CI)", "specificity",
     "mean FD checks / reference evaluations"],
    operating_rows,
)}

## Split-sample calibration

For each benchmark, the same query-cluster split is used for both sign definitions, SNR, and magnitude. Calibration
chooses the largest-coverage candidate whose one-sided cluster-bootstrap risk upper bound is no
greater than the prespecified {fmt_pct(risk_target)} target. Candidate intended coverages are fixed
at 10%, ..., 100%, and the upper screen is Bonferroni-adjusted across all ten candidates. Test
queries are untouched until after selection. This is a benchmark-conditional heuristic, not a
distribution-free or cross-domain guarantee.

{markdown_table(
    ["benchmark", "score", "selected threshold", "calibration-only screen", "held-out test"],
    split_rows,
)}

## Fixed-budget allocator

For each integer budget `B=0,...,K`, the allocator replaces the `B` lowest-scored components by
their reference finite differences. Thus `B` is a component-check budget and costs `2B` reference
objective evaluations for central differences. Exact sign-score ties are randomized {tie_repetitions}
times per query without consulting the truth; reported curves average over these tie breaks. Lower
area under the assembled-gradient relative-error curve is better.

{markdown_table(
    ["benchmark", "rank", "score", "normalized error-frontier area (cluster 95% CI)"],
    allocator_rows,
)}

Prespecified 25% and 50% budget checkpoints (rounded to an integer component count) are:

{markdown_table(
    ["benchmark", "budget", "score", "relative error (cluster 95% CI)", "mean cosine"],
    allocator_key_rows,
)}

Minimum budgets at three direction-cosine targets are:

{markdown_table(
    ["benchmark", "cosine target", "score", "component checks", "reference evaluations"],
    allocator_target_rows,
)}

## Modal-sign versus ensemble-mean-sign audit

A 5--5 vote is recorded as an undefined modal tie, not forced to either sign. Non-tie mismatch is
`modal_sign != sign(ensemble_mean)`. The last three columns evaluate the sign threshold 0.9.

{markdown_table(
    ["benchmark", "modal ties", "non-tie modal/mean mismatch", "mean-direction false accepts",
     "false accepts with mismatch", "false accepts rescued by modal", "mean-zero", "member zeros",
     "mean-aligned score differs", "tau=0.9 set differs"],
    modal_rows,
)}

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
   direction actually returned by the ensemble mean. Retain modal agreement only as a historical
   comparison; threshold-0.9 headline values remain unchanged here.
4. Positive-rescaling invariance is a robustness property; it is not evidence of superior
   discrimination.
5. The main manuscript reports the head-to-head AUC, risk--coverage, split-calibration, and
   fixed-budget results together.
6. Score selection remains inside the calibration protocol. A new domain may legitimately choose SNR,
   magnitude, sign agreement, or abstention; test data must not be used to select the score or its
   threshold.

## Reproducibility

- Python: {platform.python_version()}
- NumPy: {np.__version__}
- SciPy: {scipy.__version__}
- pandas: {pd.__version__}
- Matplotlib: {matplotlib.__version__}
- Query-cluster bootstrap replicates: {bootstrap:,}
- Random tie-break repetitions: {tie_repetitions}
- Random seed: {seed}
"""
    analysis_path = analysis_dir / "snr_external_baselines.md"
    analysis_path.write_text(report, encoding="utf-8")
    package_path = package_dir / analysis_path.name
    if analysis_path.resolve() != package_path.resolve():
        shutil.copy2(analysis_path, package_path)


def save_frame(frame: pd.DataFrame, name: str, analysis_dir: Path, package_dir: Path) -> None:
    analysis_path = analysis_dir / name
    frame.to_csv(analysis_path, index=False, float_format="%.12g")
    package_path = package_dir / name
    if analysis_path.resolve() != package_path.resolve():
        shutil.copy2(analysis_path, package_path)


tie_repetitions_global = 0


def main() -> None:
    global tie_repetitions_global
    args = parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1000 query-cluster bootstrap replicates")
    if args.tie_repetitions < 32:
        raise ValueError("Use at least 32 random tie-break repetitions")
    if not 0 < args.risk_target < 1:
        raise ValueError("--risk-target must be in (0, 1)")
    if not 0.1 <= args.calibration_fraction <= 0.9:
        raise ValueError("--calibration-fraction must be in [0.1, 0.9]")
    tie_repetitions_global = args.tie_repetitions

    analysis_dir = args.analysis_dir.resolve()
    package_dir = args.package_dir.resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(package_dir, spec) for spec in DATASETS]
    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = iter(seed_sequence.spawn(8 * len(datasets)))

    auc_records = []
    operating_records = []
    split_records = []
    allocator_rows = []
    allocator_summaries = []
    modal_records = []

    for dataset in datasets:
        rng_auc = np.random.default_rng(next(child_seeds))
        bootstrap_indices = rng_auc.integers(
            0,
            dataset["nq"],
            size=(args.bootstrap, dataset["nq"]),
            endpoint=False,
        )
        records, _ = cluster_auc_analysis(dataset, bootstrap_indices)
        auc_records.extend(records)
        operating_records.extend(matched_coverage_operating(dataset, bootstrap_indices))

        rng_split = np.random.default_rng(next(child_seeds))
        permutation = rng_split.permutation(dataset["nq"])
        n_calibration = int(round(args.calibration_fraction * dataset["nq"]))
        n_calibration = max(2, min(dataset["nq"] - 2, n_calibration))
        calibration_indices = permutation[:n_calibration]
        test_indices = permutation[n_calibration:]
        split_records.extend(
            split_calibration(
                dataset,
                calibration_indices,
                test_indices,
                args.bootstrap,
                rng_split,
                args.risk_target,
            )
        )

        rng_allocator_boot = np.random.default_rng(next(child_seeds))
        allocator_bootstrap = rng_allocator_boot.integers(
            0,
            dataset["nq"],
            size=(args.bootstrap, dataset["nq"]),
            endpoint=False,
        )
        for method in ALLOCATOR_METHODS:
            rng_ties = np.random.default_rng(next(child_seeds))
            curves = expected_allocator_curves(
                dataset, method, args.tie_repetitions, rng_ties
            )
            rows, summary = allocator_records(
                dataset, method, curves, allocator_bootstrap
            )
            allocator_rows.extend(rows)
            allocator_summaries.append(summary)
        modal_records.append(modal_mean_record(dataset))

    auc_frame = pd.DataFrame.from_records(auc_records)
    operating_frame = pd.DataFrame.from_records(operating_records)
    split_frame = pd.DataFrame.from_records(split_records)
    allocator_frame = pd.DataFrame.from_records(allocator_rows)
    allocator_summary = pd.DataFrame.from_records(allocator_summaries)
    allocator_key_frame, allocator_target_frame = allocator_key_results(allocator_frame)
    modal_frame = pd.DataFrame.from_records(modal_records)
    manifest_frame = pd.DataFrame.from_records(
        [
            {
                "dataset": dataset["spec"].name,
                "member_artifact": dataset["path"].name,
                "member_artifact_sha256": dataset["sha256"],
                "frozen_source": dataset["frozen_source"],
                "frozen_source_sha256": dataset["frozen_hash"],
                "NQ": dataset["nq"],
                "M": dataset["m"],
                "K": dataset["k"],
                "headline_sign_auc": dataset["headline_auc"],
                "headline_zero_budget_cosine": dataset["headline_cosine"],
            }
            for dataset in datasets
        ]
    )

    save_frame(auc_frame, "external_baseline_auc.csv", analysis_dir, package_dir)
    save_frame(
        operating_frame,
        "external_baseline_matched_coverage.csv",
        analysis_dir,
        package_dir,
    )
    save_frame(
        split_frame,
        "external_baseline_split_calibration.csv",
        analysis_dir,
        package_dir,
    )
    save_frame(
        allocator_frame,
        "external_baseline_allocator.csv",
        analysis_dir,
        package_dir,
    )
    save_frame(
        allocator_summary,
        "external_baseline_allocator_summary.csv",
        analysis_dir,
        package_dir,
    )
    save_frame(
        allocator_key_frame,
        "external_baseline_allocator_key_budgets.csv",
        analysis_dir,
        package_dir,
    )
    save_frame(
        allocator_target_frame,
        "external_baseline_allocator_targets.csv",
        analysis_dir,
        package_dir,
    )
    save_frame(
        modal_frame,
        "external_modal_mean_disagreement.csv",
        analysis_dir,
        package_dir,
    )
    save_frame(
        manifest_frame,
        "external_member_gradient_manifest.csv",
        analysis_dir,
        package_dir,
    )

    plot_auc(auc_frame, package_dir)
    plot_matched_coverage(operating_frame, package_dir)
    plot_frontiers(allocator_frame, package_dir)
    write_report(
        analysis_dir,
        package_dir,
        datasets,
        auc_frame,
        operating_frame,
        split_frame,
        allocator_summary,
        allocator_key_frame,
        allocator_target_frame,
        modal_frame,
        args.bootstrap,
        args.tie_repetitions,
        args.seed,
        args.risk_target,
    )

    print("AUC summary:")
    print(
        auc_frame[auc_frame["method"].isin(CALIBRATION_METHODS)][
            ["dataset", "method", "auc", "auc_cluster_low", "auc_cluster_high"]
        ].to_string(index=False)
    )
    print(f"Wrote analysis CSV/Markdown to {analysis_dir}")
    print(f"Mirrored package artifacts and figures to {package_dir}")


if __name__ == "__main__":
    main()
