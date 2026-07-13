#!/usr/bin/env python3
"""Risk--coverage and finite-difference cost audit for the SIMPAT rework.

This script reads only the supplied frozen ``.npz`` benchmark artifacts and,
when available, the validated per-member gradient artifacts. It does not train
a surrogate or call a reference simulator. Components belonging
to the same query point are treated as a cluster when constructing percentile
bootstrap confidence intervals.

Primary definitions
-------------------
``accepted``
    mean-aligned sign agreement >= threshold.
``coverage``
    P(accepted), the fraction of gradient components left to the surrogate.
``false-trust risk``
    P(sign-wrong | accepted), the selective prediction error among accepted
    components.  This differs from P(accepted | sign-wrong) = 1 - specificity.
``FD component check``
    one central finite-difference derivative component.  In the archived
    scripts it requires two reference-objective evaluations, at x+h and x-h.

Outputs are written next to this script unless ``--out-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    short_name: str
    filename: str
    color: str
    marker: str


DATASETS = (
    DatasetSpec(
        name="TMM easy (near-saturated)",
        short_name="TMM easy\n(near-saturated)",
        filename="extbench_tmm.npz",
        color="#3775BA",
        marker="o",
    ),
    DatasetSpec(
        name="TMM stressed",
        short_name="TMM\nstressed",
        filename="extbench_tmm_hard.npz",
        color="#C76B3C",
        marker="s",
    ),
    DatasetSpec(
        name="Heat/Poisson stressed",
        short_name="Heat/Poisson\nstressed",
        filename="extbench_poisson.npz",
        color="#A9C5DF",
        marker="^",
    ),
)

METRICS = ("coverage", "false_trust_risk", "sensitivity", "specificity")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_data = here if (here / "extbench_tmm.npz").exists() else here.parent.parent / "repo_paper2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--out-dir", type=Path, default=here)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="Number of query-cluster bootstrap replicates (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--risk-target",
        type=float,
        default=0.10,
        help="Illustrative false-trust-risk target for split calibration (default: 0.10).",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.50,
        help="Fraction of query clusters reserved for threshold calibration (default: 0.50).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(z: np.lib.npyio.NpzFile, key: str, cast):
    if key not in z.files:
        raise KeyError(f"{key!r} is missing from {z.fid.name}")
    return cast(np.asarray(z[key]).item())


def load_dataset(data_dir: Path, spec: DatasetSpec) -> dict:
    path = data_dir / spec.filename
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        nq = scalar(z, "NQ", int)
        k = scalar(z, "K", int)
        stored_tau = scalar(z, "tau", float)
        archived_score = np.asarray(z["all_sa"], dtype=float)
        archived_correct = np.asarray(z["all_correct"], dtype=int)
        stored_trust_mean = scalar(z, "n_trust_mean", float)
        extra = {}
        if spec.filename == "extbench_poisson.npz":
            for key in ("easy_auc", "easy_frac_wrong", "easy_n_sign_wrong", "easy_K"):
                extra[key] = np.asarray(z[key]).item() if key in z.files else None

    member_names = {
        "extbench_tmm.npz": "external_members_tmm_easy.npz",
        "extbench_tmm_hard.npz": "external_members_tmm_stressed.npz",
        "extbench_poisson.npz": "external_members_heat_poisson_stressed.npz",
    }
    member_path = data_dir / member_names[spec.filename]
    if member_path.is_file():
        with np.load(member_path, allow_pickle=False) as z:
            gradients = np.asarray(z["member_gradients"], dtype=float)  # (Q,M,K)
            reference = np.asarray(z["reference_gradients"], dtype=float)
        mean_gradient = gradients.mean(axis=1)
        mean_sign = np.sign(mean_gradient)
        score = (np.sign(gradients) == mean_sign[:, None, :]).mean(axis=1)
        score[mean_sign == 0] = 0.0
        correct = (mean_sign == np.sign(reference)).astype(int)
        if not np.array_equal(correct.reshape(-1), archived_correct.reshape(-1)):
            raise ValueError(f"{member_path.name}: rebuilt correctness labels differ from archive")
    else:
        score = archived_score.reshape(nq, k)
        correct = archived_correct.reshape(nq, k)

    expected = nq * k
    if score.size != expected or correct.size != expected:
        raise ValueError(
            f"{path.name}: expected NQ*K={expected} entries, got "
            f"{score.size} scores and {correct.size} labels"
        )
    if not np.isfinite(score).all() or not ((0.0 <= score) & (score <= 1.0)).all():
        raise ValueError(f"{path.name}: sign-agreement values must be finite and in [0, 1]")
    if not np.isin(correct, [0, 1]).all():
        raise ValueError(f"{path.name}: all_correct must contain only 0/1 labels")

    score = score.reshape(nq, k)
    correct = correct.reshape(nq, k).astype(bool)
    recomputed = float((score >= stored_tau - 1e-12).sum(axis=1).mean())
    if not math.isclose(recomputed, stored_trust_mean, rel_tol=0, abs_tol=1e-10):
        raise ValueError(
            f"{path.name}: stored n_trust_mean={stored_trust_mean} does not match "
            f"recomputed value {recomputed}"
        )

    return {
        "spec": spec,
        "path": path,
        "sha256": sha256(path),
        "nq": nq,
        "k": k,
        "stored_tau": stored_tau,
        "score": score,
        "correct": correct,
        "extra": extra,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a pooled binomial proportion.

    These intervals are retained as a transparent small-count diagnostic.  The
    query-cluster bootstrap is primary because gradient components within one
    query are not assumed independent.
    """

    if total <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(numerator).shape, np.nan, dtype=float)
    return np.divide(numerator, denominator, out=out, where=np.asarray(denominator) > 0)


def cluster_bootstrap_metrics(
    trusted: np.ndarray,
    correct: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling whole query points with replacement."""

    trusted_q = trusted.sum(axis=1)
    tp_q = (trusted & correct).sum(axis=1)
    fp_q = (trusted & ~correct).sum(axis=1)
    tn_q = (~trusted & ~correct).sum(axis=1)
    correct_q = correct.sum(axis=1)
    wrong_q = (~correct).sum(axis=1)

    def boot_sum(per_query: np.ndarray) -> np.ndarray:
        return per_query[bootstrap_indices].sum(axis=1)

    n_components = trusted.shape[0] * trusted.shape[1]
    trusted_b = boot_sum(trusted_q)
    tp_b = boot_sum(tp_q)
    fp_b = boot_sum(fp_q)
    tn_b = boot_sum(tn_q)
    correct_b = boot_sum(correct_q)
    wrong_b = boot_sum(wrong_q)

    draws = {
        "coverage": trusted_b / n_components,
        "false_trust_risk": divide_array(fp_b, trusted_b),
        "sensitivity": divide_array(tp_b, correct_b),
        "specificity": divide_array(tn_b, wrong_b),
    }
    return {name: percentile_ci(values) for name, values in draws.items()}


def operating_record(
    dataset: dict,
    threshold: float,
    bootstrap_indices: np.ndarray,
) -> dict:
    score = dataset["score"]
    correct = dataset["correct"]
    trusted = score >= threshold - 1e-12

    tp = int((trusted & correct).sum())
    fp = int((trusted & ~correct).sum())
    fn = int((~trusted & correct).sum())
    tn = int((~trusted & ~correct).sum())
    n = int(correct.size)
    n_trusted = tp + fp
    n_correct = tp + fn
    n_wrong = tn + fp

    points = {
        "coverage": safe_ratio(n_trusted, n),
        "false_trust_risk": safe_ratio(fp, n_trusted),
        "sensitivity": safe_ratio(tp, n_correct),
        "specificity": safe_ratio(tn, n_wrong),
    }
    cluster_cis = cluster_bootstrap_metrics(trusted, correct, bootstrap_indices)
    wilson_counts = {
        "coverage": (n_trusted, n),
        "false_trust_risk": (fp, n_trusted),
        "sensitivity": (tp, n_correct),
        "specificity": (tn, n_wrong),
    }

    record = {
        "dataset": dataset["spec"].name,
        "source_file": dataset["path"].name,
        "threshold": threshold,
        "NQ": dataset["nq"],
        "K": dataset["k"],
        "n_components": n,
        "n_sign_correct": n_correct,
        "n_sign_wrong": n_wrong,
        "TP_trusted_correct": tp,
        "FP_trusted_wrong": fp,
        "FN_rejected_correct": fn,
        "TN_rejected_wrong": tn,
        "false_trust_rate": safe_ratio(fp, n_wrong),
    }
    for metric in METRICS:
        record[metric] = points[metric]
        record[f"{metric}_cluster_low"] = cluster_cis[metric][0]
        record[f"{metric}_cluster_high"] = cluster_cis[metric][1]
        wilson = wilson_interval(*wilson_counts[metric])
        record[f"{metric}_wilson_low"] = wilson[0]
        record[f"{metric}_wilson_high"] = wilson[1]

    k = dataset["k"]
    coverage = points["coverage"]
    record["trusted_components_mean"] = k * coverage
    record["fd_component_checks_mean"] = k * (1.0 - coverage)
    record["reference_evaluations_mean"] = 2.0 * record["fd_component_checks_mean"]
    record["all_fd_reference_evaluations"] = 2 * k
    record["reference_evaluation_savings_fraction"] = coverage
    return record


def split_sample_calibration(
    dataset: dict,
    thresholds: np.ndarray,
    bootstrap: int,
    seed: np.random.SeedSequence,
    risk_target: float,
    calibration_fraction: float,
    alpha: float = 0.05,
) -> tuple[list[dict], dict]:
    """Illustrative threshold selection with disjoint calibration/test queries.

    Candidate thresholds are screened on calibration clusters only.  To account
    for searching over several candidates, the selection screen uses a
    Bonferroni-adjusted one-sided query-cluster bootstrap upper quantile.  The
    chosen threshold is then evaluated once on untouched test clusters.

    This is deliberately labelled a heuristic: a finite-sample bootstrap screen
    on one benchmark split is neither distribution-free nor an external-domain
    guarantee.
    """

    rng = np.random.default_rng(seed)
    nq = dataset["nq"]
    n_cal = int(round(nq * calibration_fraction))
    n_cal = max(2, min(nq - 2, n_cal))
    permutation = rng.permutation(nq)
    cal_idx = permutation[:n_cal]
    test_idx = permutation[n_cal:]
    score_cal = dataset["score"][cal_idx]
    correct_cal = dataset["correct"][cal_idx]
    score_test = dataset["score"][test_idx]
    correct_test = dataset["correct"][test_idx]

    boot_cal = rng.integers(0, n_cal, size=(bootstrap, n_cal), endpoint=False)
    simultaneous_quantile = 1.0 - alpha / len(thresholds)
    candidates: list[dict] = []
    for threshold in thresholds:
        trusted = score_cal >= threshold - 1e-12
        fp_q = (trusted & ~correct_cal).sum(axis=1)
        trusted_q = trusted.sum(axis=1)
        fp_boot = fp_q[boot_cal].sum(axis=1)
        trusted_boot = trusted_q[boot_cal].sum(axis=1)
        risk_boot = divide_array(fp_boot, trusted_boot)
        finite = risk_boot[np.isfinite(risk_boot)]
        upper = float(np.quantile(finite, simultaneous_quantile)) if finite.size else float("nan")
        risk = safe_ratio(int(fp_q.sum()), int(trusted_q.sum()))
        candidates.append(
            {
                "dataset": dataset["spec"].name,
                "threshold": float(threshold),
                "calibration_queries": n_cal,
                "test_queries": int(nq - n_cal),
                "risk_target": risk_target,
                "calibration_coverage": float(trusted.mean()),
                "calibration_false_trust_risk": risk,
                "calibration_risk_upper_bonferroni_cluster": upper,
                "passes_calibration_screen": bool(np.isfinite(upper) and upper <= risk_target),
                "selected": False,
            }
        )

    feasible = [record for record in candidates if record["passes_calibration_screen"]]
    selected = min(feasible, key=lambda record: record["threshold"]) if feasible else None
    if selected is not None:
        selected["selected"] = True
        threshold = float(selected["threshold"])
        trusted_test = score_test >= threshold - 1e-12
        boot_test = rng.integers(
            0, len(test_idx), size=(bootstrap, len(test_idx)), endpoint=False
        )
        test_cis = cluster_bootstrap_metrics(trusted_test, correct_test, boot_test)
        tp = int((trusted_test & correct_test).sum())
        fp = int((trusted_test & ~correct_test).sum())
        fn = int((~trusted_test & correct_test).sum())
        tn = int((~trusted_test & ~correct_test).sum())
        test_coverage = float(trusted_test.mean())
        test_risk = safe_ratio(fp, tp + fp)
        status = "selected_on_calibration_evaluated_on_held_out_test"
    else:
        threshold = float("nan")
        test_cis = {metric: (float("nan"), float("nan")) for metric in METRICS}
        tp = fp = fn = tn = 0
        test_coverage = test_risk = float("nan")
        status = "no_candidate_met_calibration_risk_bound"

    summary = {
        "dataset": dataset["spec"].name,
        "risk_target": risk_target,
        "candidate_thresholds": ";".join(f"{x:.1f}" for x in thresholds),
        "familywise_alpha": alpha,
        "calibration_fraction": calibration_fraction,
        "calibration_queries": n_cal,
        "test_queries": int(nq - n_cal),
        "calibration_query_indices": ";".join(str(int(x)) for x in cal_idx),
        "test_query_indices": ";".join(str(int(x)) for x in test_idx),
        "selected_threshold": threshold,
        "status": status,
        "test_TP_trusted_correct": tp,
        "test_FP_trusted_wrong": fp,
        "test_FN_rejected_correct": fn,
        "test_TN_rejected_wrong": tn,
        "test_coverage": test_coverage,
        "test_coverage_cluster_low": test_cis["coverage"][0],
        "test_coverage_cluster_high": test_cis["coverage"][1],
        "test_false_trust_risk": test_risk,
        "test_false_trust_risk_cluster_low": test_cis["false_trust_risk"][0],
        "test_false_trust_risk_cluster_high": test_cis["false_trust_risk"][1],
    }
    if selected is not None:
        summary.update(
            {
                "calibration_coverage_at_selection": selected["calibration_coverage"],
                "calibration_risk_at_selection": selected["calibration_false_trust_risk"],
                "calibration_risk_upper_at_selection": selected[
                    "calibration_risk_upper_bonferroni_cluster"
                ],
            }
        )
    else:
        summary.update(
            {
                "calibration_coverage_at_selection": float("nan"),
                "calibration_risk_at_selection": float("nan"),
                "calibration_risk_upper_at_selection": float("nan"),
            }
        )
    return candidates, summary


def fmt_pct(value: float, digits: int = 1) -> str:
    return "NA" if not np.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def fmt_interval(point: float, lo: float, hi: float, digits: int = 1) -> str:
    if not (np.isfinite(point) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    return f"{fmt_pct(point, digits)} [{fmt_pct(lo, digits)}, {fmt_pct(hi, digits)}]"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    headers = [str(x) for x in headers]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def write_results(
    out_dir: Path,
    data_dir: Path,
    datasets: list[dict],
    metrics: pd.DataFrame,
    split_summary: pd.DataFrame,
    bootstrap: int,
    seed: int,
    risk_target: float,
    calibration_fraction: float,
) -> None:
    op = metrics[np.isclose(metrics["threshold"], 0.9)].copy()

    operating_rows = []
    for _, row in op.iterrows():
        operating_rows.append(
            (
                row["dataset"],
                f"{int(row['TP_trusted_correct'])}/{int(row['FP_trusted_wrong'])}/"
                f"{int(row['FN_rejected_correct'])}/{int(row['TN_rejected_wrong'])}",
                fmt_interval(
                    row["coverage"], row["coverage_cluster_low"], row["coverage_cluster_high"]
                ),
                fmt_interval(
                    row["false_trust_risk"],
                    row["false_trust_risk_cluster_low"],
                    row["false_trust_risk_cluster_high"],
                ),
                fmt_interval(
                    row["sensitivity"],
                    row["sensitivity_cluster_low"],
                    row["sensitivity_cluster_high"],
                ),
                fmt_interval(
                    row["specificity"],
                    row["specificity_cluster_low"],
                    row["specificity_cluster_high"],
                ),
            )
        )

    cost_rows = []
    for _, row in op.iterrows():
        cost_rows.append(
            (
                row["dataset"],
                int(row["K"]),
                f"{row['trusted_components_mean']:.2f}",
                f"{row['fd_component_checks_mean']:.2f}",
                f"{row['reference_evaluations_mean']:.2f}",
                int(row["all_fd_reference_evaluations"]),
                fmt_pct(row["reference_evaluation_savings_fraction"]),
            )
        )

    source_rows = []
    for dataset in datasets:
        correct = dataset["correct"]
        source_rows.append(
            (
                dataset["path"].name,
                f"{dataset['nq']} x {dataset['k']} = {correct.size}",
                int(correct.sum()),
                int((~correct).sum()),
                dataset["sha256"],
            )
        )

    split_rows = []
    for _, row in split_summary.iterrows():
        if np.isfinite(row["selected_threshold"]):
            selected_tau = f"{row['selected_threshold']:.1f}"
            cal_result = (
                f"{fmt_pct(row['calibration_coverage_at_selection'])} coverage; "
                f"risk {fmt_pct(row['calibration_risk_at_selection'])}; "
                f"adjusted upper {fmt_pct(row['calibration_risk_upper_at_selection'])}"
            )
            test_result = (
                f"coverage {fmt_interval(row['test_coverage'], row['test_coverage_cluster_low'], row['test_coverage_cluster_high'])}; "
                f"risk {fmt_interval(row['test_false_trust_risk'], row['test_false_trust_risk_cluster_low'], row['test_false_trust_risk_cluster_high'])}"
            )
        else:
            selected_tau = "none"
            cal_result = "no candidate passed"
            test_result = "not evaluated"
        split_rows.append(
            (
                row["dataset"],
                f"{int(row['calibration_queries'])}/{int(row['test_queries'])}",
                selected_tau,
                cal_result,
                test_result,
            )
        )

    poisson = next(d for d in datasets if d["path"].name == "extbench_poisson.npz")
    easy = poisson["extra"]

    text = f"""# Risk--coverage and reference-evaluation audit

This report was generated by `risk_coverage_simpat.py` from the frozen arrays
listed in the Data audit below. No training and no reference-simulator call was performed.

## Definitions and uncertainty

- A component is **accepted** when its mean-aligned member-sign agreement is at least the stated threshold.
- **Coverage** is the fraction of components accepted.
- **False-trust risk** is `P(sign-wrong | accepted)`. It is not the false-positive rate;
  `P(accepted | sign-wrong) = 1 - specificity` is reported separately in the CSV.
- Sensitivity is the fraction of sign-correct components accepted. Specificity is the fraction of
  sign-wrong components rejected.
- Primary 95% intervals are percentile intervals from {bootstrap:,} bootstrap resamples of whole
  query points (seed {seed}). Resampling query clusters preserves within-query dependence among
  gradient components. Pooled Wilson intervals are included in the CSV only as a small-count
  diagnostic and are not the primary uncertainty statement.
- The intervals quantify finite-query sampling variation conditional on the frozen trained
  ensembles and benchmark-generating setup. They do not establish cross-domain calibration or a
  universal threshold.

## Operating point at threshold 0.9

Counts are `TP/FP/FN/TN`, with "positive" meaning that the protocol accepts the component.

{markdown_table(
    ["benchmark", "TP/FP/FN/TN", "coverage (cluster 95% CI)",
     "false-trust risk (cluster 95% CI)", "sensitivity (cluster 95% CI)",
     "specificity (cluster 95% CI)"],
    operating_rows,
)}

The stressed benchmarks show the deployment trade-off clearly: threshold 0.9 leaves non-zero
residual error among accepted components. The score is therefore a selective-risk screen, not a
certificate that an accepted gradient component is correct.

## Split-sample calibration demonstration

As an illustrative rule fixed before inspecting the held-out split, {100 * calibration_fraction:.0f}%
of query clusters are used for calibration and the remainder for testing. Among candidate thresholds
`0.5, ..., 1.0`, the rule selects the smallest threshold (largest coverage) whose one-sided
query-cluster bootstrap risk upper bound is no greater than the prespecified target
`{100 * risk_target:.0f}%`. The upper screen is Bonferroni-adjusted across the six candidate
thresholds. The held-out test clusters are not used in threshold selection.

{markdown_table(
    ["benchmark", "cal/test queries", "selected threshold", "calibration-only screen",
     "untouched test result (cluster 95% CI)"],
    split_rows,
)}

This is a **split-sample calibration heuristic**, not an independent guarantee: the benchmark
queries share the same frozen ensemble and data-generating regime, bootstrap bounds are not
distribution-free, and a single 50/50 split is data-limited. If no candidate passes (as can occur
in the stressed TMM case), the procedure abstains rather than inspecting the test split and relaxing
the target. External deployment would require a fresh, representative calibration set.

## Central-finite-difference cost accounting at threshold 0.9

This table answers a specific counterfactual: *if every rejected component is corrected by a
central finite difference*, how many reference evaluations are required on average per query?

{markdown_table(
    ["benchmark", "K", "accepted components", "FD component checks", "reference evaluations",
     "all-FD evaluations", "evaluation saving vs all-FD"],
    cost_rows,
)}

The conversion is exact for the archived benchmark implementations:

`1 central-FD component check = 2 reference-objective evaluations`, hence
`B component checks = 2B evaluations` and a complete K-component central-FD gradient costs `2K`
evaluations. Gate scoring itself uses no *additional* reference evaluation. Fractional values in
the table are averages across query points; each individual query uses an integer number of checks.
This accounting should not be described as "B simulator solves" without the factor of two.

## Data audit

{markdown_table(
    ["source", "queries x components", "sign-correct", "sign-wrong", "SHA-256"], source_rows
)}

The Poisson archive also reports an easy-regime summary (`easy_K={easy.get('easy_K')}`,
`easy_n_sign_wrong={easy.get('easy_n_sign_wrong')}`, `easy_auc={easy.get('easy_auc')}`). It does not
store the easy-regime component-level `all_sa`/`all_correct` arrays, so risk--coverage and threshold
intervals for that auxiliary regime cannot be reconstructed without rerunning the experiment. It
is therefore not plotted or silently imputed.

## Recommended manuscript wording

> At threshold 0.9, the gate retained [coverage] of gradient components, while the observed
> sign-error rate among retained components was [false-trust risk]. Query-cluster bootstrap
> intervals quantify uncertainty conditional on each frozen benchmark. If every rejected component
> is centrally finite-difference corrected, the mean reference-evaluation cost is twice the mean
> number of rejected components; this distinction is maintained throughout.

Replace the bracketed quantities by benchmark-specific values from the operating-point table. Do
not combine the three benchmarks into one pooled claim, because their regimes, component counts,
and base error rates differ.

## Reproducibility

- Python: {platform.python_version()}
- NumPy: {np.__version__}
- SciPy: {scipy.__version__}
- pandas: {pd.__version__}
- Matplotlib: {matplotlib.__version__}
- Bootstrap replicates: {bootstrap:,}
- Random seed: {seed}
- Split-calibration risk target: {risk_target:.3f}
- Calibration fraction: {calibration_fraction:.3f}
"""
    (out_dir / "risk_coverage_results.md").write_text(text, encoding="utf-8")

    cost_columns = [
        "dataset",
        "threshold",
        "NQ",
        "K",
        "coverage",
        "coverage_cluster_low",
        "coverage_cluster_high",
        "trusted_components_mean",
        "fd_component_checks_mean",
        "reference_evaluations_mean",
        "all_fd_reference_evaluations",
        "reference_evaluation_savings_fraction",
        "false_trust_risk",
    ]
    op[cost_columns].to_csv(out_dir / "fd_cost_accounting.csv", index=False, float_format="%.10g")


def plot_figure(out_dir: Path, datasets: list[dict], metrics: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.25, 3.15), gridspec_kw={"wspace": 0.36})

    for dataset in datasets:
        spec = dataset["spec"]
        frame = metrics[metrics["dataset"] == spec.name].sort_values("threshold")
        x = frame["coverage"].to_numpy()
        y = frame["false_trust_risk"].to_numpy()
        xlo = frame["coverage_cluster_low"].to_numpy()
        xhi = frame["coverage_cluster_high"].to_numpy()
        ylo = frame["false_trust_risk_cluster_low"].to_numpy()
        yhi = frame["false_trust_risk_cluster_high"].to_numpy()
        ax0.plot(x, y, color=spec.color, lw=1.6, marker=spec.marker, ms=4.5, label=spec.name)
        ax0.errorbar(
            x,
            y,
            xerr=np.vstack([x - xlo, xhi - x]),
            yerr=np.vstack([y - ylo, yhi - y]),
            fmt="none",
            ecolor=spec.color,
            elinewidth=0.7,
            capsize=1.7,
            alpha=0.62,
            zorder=0,
        )
        row = frame[np.isclose(frame["threshold"], 0.9)].iloc[0]
        ax0.scatter(
            [row["coverage"]],
            [row["false_trust_risk"]],
            s=62,
            marker=spec.marker,
            facecolor="white",
            edgecolor=spec.color,
            linewidth=1.5,
            zorder=4,
        )

    ax0.set_title("(a) Selective risk versus accepted coverage", loc="left", pad=8)
    ax0.set_xlabel("Accepted gradient components (coverage)")
    ax0.set_ylabel("Sign-wrong among accepted\n(false-trust risk)")
    ax0.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.set_xlim(0.54, 1.015)
    ax0.set_ylim(-0.008, 0.245)
    ax0.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.75)
    ax0.legend(loc="upper left", fontsize=7.7, handlelength=2.2)
    ax0.text(
        0.03,
        0.04,
        "Open markers: $\\tau=0.9$ (one per benchmark)\n"
        "Error bars: query-cluster bootstrap 95% CI",
        transform=ax0.transAxes,
        fontsize=7.2,
        color="#4D4D4D",
    )

    op = metrics[np.isclose(metrics["threshold"], 0.9)].copy()
    xloc = np.arange(len(datasets), dtype=float)
    width = 0.34
    full = []
    selective = []
    selective_lo = []
    selective_hi = []
    labels = []
    residual_risk = []
    for dataset in datasets:
        spec = dataset["spec"]
        row = op[op["dataset"] == spec.name].iloc[0]
        k = float(row["K"])
        value = float(row["reference_evaluations_mean"])
        lo = 2.0 * k * (1.0 - float(row["coverage_cluster_high"]))
        hi = 2.0 * k * (1.0 - float(row["coverage_cluster_low"]))
        full.append(2.0 * k)
        selective.append(value)
        selective_lo.append(value - lo)
        selective_hi.append(hi - value)
        labels.append(spec.short_name)
        residual_risk.append(float(row["false_trust_risk"]))

    ax1.bar(
        xloc - width / 2,
        full,
        width,
        color="#CFCECE",
        edgecolor="#4D4D4D",
        linewidth=0.8,
        hatch="///",
        label="Complete central FD",
    )
    bars = ax1.bar(
        xloc + width / 2,
        selective,
        width,
        color="#3775BA",
        edgecolor="#2A5B8A",
        linewidth=0.8,
        hatch="...",
        label=r"Correct rejected ($\tau=0.9$)",
        yerr=np.vstack([selective_lo, selective_hi]),
        capsize=2.0,
        error_kw={"elinewidth": 0.8, "ecolor": "#2A5B8A"},
    )
    for bar, risk in zip(bars, residual_risk):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.25,
            f"risk\n{100 * risk:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#2A5B8A",
        )
    ax1.set_title("(b) Reference-evaluation accounting", loc="left", pad=8)
    ax1.set_ylabel("Reference-objective evaluations per query")
    ax1.set_xticks(xloc, labels)
    ax1.set_ylim(0, max(full) * 1.18)
    ax1.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.75)
    ax1.legend(loc="upper left", fontsize=7.5)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.90)
    fig.savefig(out_dir / "fig_risk_coverage.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / "fig_risk_coverage.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1000 cluster-bootstrap replicates")
    if not (0.0 < args.risk_target < 1.0):
        raise ValueError("--risk-target must be strictly between 0 and 1")
    if not (0.1 <= args.calibration_fraction <= 0.9):
        raise ValueError("--calibration-fraction must be between 0.1 and 0.9")
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [load_dataset(data_dir, spec) for spec in DATASETS]
    thresholds = np.round(np.arange(0.5, 1.01, 0.1), 1)
    seed_sequence = np.random.SeedSequence(args.seed)
    metric_seeds = seed_sequence.spawn(len(datasets))
    split_seeds = seed_sequence.spawn(len(datasets))

    records = []
    for dataset, child_seed in zip(datasets, metric_seeds):
        rng = np.random.default_rng(child_seed)
        bootstrap_indices = rng.integers(
            0, dataset["nq"], size=(args.bootstrap, dataset["nq"]), endpoint=False
        )
        for threshold in thresholds:
            records.append(operating_record(dataset, float(threshold), bootstrap_indices))

    metrics = pd.DataFrame.from_records(records)
    metrics.to_csv(out_dir / "risk_coverage_metrics.csv", index=False, float_format="%.10g")

    candidate_records: list[dict] = []
    selection_records: list[dict] = []
    for dataset, child_seed in zip(datasets, split_seeds):
        candidates, selection = split_sample_calibration(
            dataset=dataset,
            thresholds=thresholds,
            bootstrap=args.bootstrap,
            seed=child_seed,
            risk_target=args.risk_target,
            calibration_fraction=args.calibration_fraction,
        )
        candidate_records.extend(candidates)
        selection_records.append(selection)
    split_candidates = pd.DataFrame.from_records(candidate_records)
    split_summary = pd.DataFrame.from_records(selection_records)
    split_candidates.to_csv(
        out_dir / "split_calibration_candidates.csv", index=False, float_format="%.10g"
    )
    split_summary.to_csv(
        out_dir / "split_calibration_selection.csv", index=False, float_format="%.10g"
    )

    write_results(
        out_dir,
        data_dir,
        datasets,
        metrics,
        split_summary,
        args.bootstrap,
        args.seed,
        args.risk_target,
        args.calibration_fraction,
    )
    plot_figure(out_dir, datasets, metrics)

    op = metrics[np.isclose(metrics["threshold"], 0.9)]
    print(f"Wrote outputs to: {out_dir}")
    print("Threshold 0.9 audit:")
    for _, row in op.iterrows():
        print(
            f"  {row['dataset']}: coverage={row['coverage']:.4f}, "
            f"false-trust risk={row['false_trust_risk']:.4f}, "
            f"sensitivity={row['sensitivity']:.4f}, specificity={row['specificity']:.4f}, "
            f"FD checks={row['fd_component_checks_mean']:.2f}, "
            f"reference evaluations={row['reference_evaluations_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
