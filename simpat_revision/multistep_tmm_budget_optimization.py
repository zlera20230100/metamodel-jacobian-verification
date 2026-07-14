#!/usr/bin/env python3
"""Matched-cost multi-step optimization on the archived stressed-TMM benchmark.

This experiment closes the loop from component reliability to repeated design
updates without running an expensive full-wave solver.  It reconstructs the
exact archived stressed-TMM ensemble, uses the archived calibration-only SNR
and magnitude thresholds, and starts from held-out query points.  Every method
The primary comparison gives random, sign-order, SNR-order, and magnitude-order
selection exactly B component checks per update.  All-autodiff and complete FD
are retained as zero- and full-cost endpoints.  Threshold-calibrated variable-
budget variants are retained as secondary context.

Reference-call accounting follows the manuscript: one centrally differenced
component costs two objective evaluations.  Reference objective values along
the trajectory are computed only for post-hoc assessment and are reported in a
separate column; they are not fed to any optimizer or line search.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (
    "all_autodiff",
    "complete_fd",
    "matched_random",
    "matched_sign_order",
    "matched_snr_order",
    "matched_magnitude_order",
    "calibrated_snr",
    "calibrated_magnitude",
    "calibrated_sign_abstain",
)

METHOD_LABELS = {
    "all_autodiff": "All autodiff",
    "complete_fd": "Complete central FD",
    "matched_random": "Random order (B = 6)",
    "matched_sign_order": "Sign order (B = 6)",
    "matched_snr_order": "SNR order (B = 6)",
    "matched_magnitude_order": "Magnitude order (B = 6)",
    "calibrated_snr": "Calibrated SNR (variable)",
    "calibrated_magnitude": "Calibrated magnitude (variable)",
    "calibrated_sign_abstain": "Sign: abstain (complete FD)",
}

PLOT_METHODS = (
    "all_autodiff",
    "complete_fd",
    "matched_random",
    "matched_sign_order",
    "matched_snr_order",
    "matched_magnitude_order",
)

MATCHED_METHODS = (
    "matched_random",
    "matched_sign_order",
    "matched_snr_order",
    "matched_magnitude_order",
)

COLORS = {
    "all_autodiff": "#5C5C5C",
    "complete_fd": "#173F5F",
    "matched_random": "#8B9298",
    "matched_sign_order": "#3775BA",
    "matched_snr_order": "#6B9AC4",
    "matched_magnitude_order": "#A9C5DF",
    "calibrated_snr": "#3775BA",
    "calibrated_magnitude": "#6B9AC4",
}

LINESTYLES = {
    "all_autodiff": "--",
    "complete_fd": "-",
    "matched_random": ":",
    "matched_sign_order": "-.",
    "matched_snr_order": (0, (5, 2)),
    "matched_magnitude_order": (0, (2, 1)),
    "calibrated_snr": "-.",
    "calibrated_magnitude": (0, (5, 2)),
}

MARKERS = {
    "all_autodiff": "X",
    "complete_fd": "D",
    "matched_random": "o",
    "matched_sign_order": "s",
    "matched_snr_order": "^",
    "matched_magnitude_order": "v",
    "calibrated_snr": "^",
    "calibrated_magnitude": "v",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here)
    parser.add_argument("--out-dir", type=Path, default=here)
    parser.add_argument("--mirror-dir", type=Path, default=here)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dimensionless-step", type=float, default=0.02)
    parser.add_argument("--bound-fraction", type=float, default=0.30)
    parser.add_argument("--random-checks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tmm_objective(reference_module, thickness: np.ndarray, wavelength: float) -> float:
    reflection, _ = reference_module.tmm_r_and_grad(thickness, wavelength)
    return float(abs(reflection) ** 2)


def central_fd_components(
    reference_module,
    thickness: np.ndarray,
    wavelength: float,
    mask: np.ndarray,
    step_nm: float = 0.5,
) -> tuple[np.ndarray, int, float, int]:
    """Return selected central-FD components and an analytic audit."""

    gradient = np.full(thickness.size, np.nan, dtype=float)
    selected = np.flatnonzero(mask)
    for component in selected:
        plus = thickness.copy()
        minus = thickness.copy()
        plus[component] += step_nm
        minus[component] -= step_nm
        gradient[component] = (
            tmm_objective(reference_module, plus, wavelength)
            - tmm_objective(reference_module, minus, wavelength)
        ) / (2.0 * step_nm)
    if not selected.size:
        return gradient, 0, float("nan"), 0

    _, analytic = reference_module.tmm_r_and_grad(thickness, wavelength)
    denominator = np.linalg.norm(analytic[selected]) + 1e-15
    relative_l2 = float(np.linalg.norm(gradient[selected] - analytic[selected]) / denominator)
    nonzero = (np.abs(gradient[selected]) > 1e-12) | (np.abs(analytic[selected]) > 1e-12)
    sign_mismatches = int(
        np.sum(np.sign(gradient[selected][nonzero]) != np.sign(analytic[selected][nonzero]))
    )
    return gradient, 2 * selected.size, relative_l2, sign_mismatches


def parse_calibration(path: Path) -> tuple[dict[str, float], list[int], bool]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    stressed = [row for row in rows if row["dataset"] == "TMM stressed"]
    thresholds: dict[str, float] = {}
    test_indices: list[int] | None = None
    sign_abstains = False
    for row in stressed:
        method = row["method"]
        if test_indices is None and row.get("test_query_indices"):
            test_indices = [int(value) for value in row["test_query_indices"].split(";")]
        if method in ("snr", "magnitude"):
            if row["status"] != "selected_on_calibration_evaluated_on_held_out_test":
                raise RuntimeError(f"{method} has no archived calibrated threshold")
            thresholds[method] = float(row["selected_threshold"])
        if method == "mean_aligned_sign":
            sign_abstains = row["status"] == "no_candidate_met_calibration_risk_bound"
    if set(thresholds) != {"snr", "magnitude"}:
        raise RuntimeError("Missing SNR or magnitude threshold")
    if test_indices is None:
        raise RuntimeError("Missing held-out query indices")
    return thresholds, test_indices, sign_abstains


def reconstruct_archived_ensemble(data_dir: Path, rebuild_module, reference_module):
    import torch

    torch.set_num_threads(3)
    archive_path = data_dir / "external_members_tmm_stressed.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        queries = np.asarray(archive["query_points"], dtype=float)
        reference_gradients = np.asarray(archive["reference_gradients"], dtype=float)
        archived_members = np.asarray(archive["member_gradients"], dtype=float)
        wavelength = float(np.asarray(archive["lam_eval"]).item())
        training_size = int(np.asarray(archive["NS"]).item())

    nominal = np.asarray(reference_module.TMM_NOMINAL, dtype=float)
    rng = np.random.default_rng(7)
    training_x = nominal[None, :] * (
        1.0 + 0.30 * (2.0 * rng.random((training_size, nominal.size)) - 1.0)
    )
    training_y = np.array(
        [tmm_objective(reference_module, row, wavelength) for row in training_x]
    )
    nets, xm, xs, _, ysd = rebuild_module.train_ensemble(training_x, training_y, members=10)

    rebuilt = np.empty_like(archived_members)
    for query_index, query in enumerate(queries):
        rebuilt[query_index] = np.array(
            [rebuild_module.member_gradient(net, query, xm, xs, ysd) for net in nets]
        )
    max_abs = float(np.max(np.abs(rebuilt - archived_members)))
    if not np.allclose(rebuilt, archived_members, rtol=2e-7, atol=2e-9):
        raise AssertionError(f"Rebuilt ensemble differs from archive; max abs={max_abs:.3e}")
    return {
        "queries": queries,
        "reference_gradients": reference_gradients,
        "archived_members": archived_members,
        "wavelength": wavelength,
        "nominal": nominal,
        "nets": nets,
        "xm": xm,
        "xs": xs,
        "ysd": ysd,
        "max_abs_rebuild_difference": max_abs,
    }


def member_gradients_at(state: dict, rebuild_module, thickness: np.ndarray) -> np.ndarray:
    return np.array(
        [
            rebuild_module.member_gradient(
                net, thickness, state["xm"], state["xs"], state["ysd"]
            )
            for net in state["nets"]
        ]
    )


def selected_mask(
    method: str,
    member_gradients: np.ndarray,
    thresholds: dict[str, float],
    random_checks: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = member_gradients.mean(axis=0)
    spread = member_gradients.std(axis=0, ddof=0)
    snr = np.abs(mean) / (spread + 1e-12)
    magnitude = np.abs(mean)
    mean_sign = np.sign(mean)
    mean_aligned_sign = (np.sign(member_gradients) == mean_sign[None, :]).mean(axis=0)
    mean_aligned_sign[mean_sign == 0] = 0.0
    k = mean.size
    if method == "all_autodiff":
        mask = np.zeros(k, dtype=bool)
    elif method in ("complete_fd", "calibrated_sign_abstain"):
        mask = np.ones(k, dtype=bool)
    elif method == "matched_random":
        mask = np.zeros(k, dtype=bool)
        mask[rng.choice(k, size=random_checks, replace=False)] = True
    elif method in (
        "matched_sign_order",
        "matched_snr_order",
        "matched_magnitude_order",
    ):
        score = {
            "matched_sign_order": mean_aligned_sign,
            "matched_snr_order": snr,
            "matched_magnitude_order": magnitude,
        }[method]
        # Check the B least-trusted components.  A seeded random secondary key
        # prevents the highly discrete sign score from inheriting index bias.
        order = np.lexsort((rng.random(k), score))
        mask = np.zeros(k, dtype=bool)
        mask[order[:random_checks]] = True
    elif method == "calibrated_snr":
        mask = snr < thresholds["snr"]
    elif method == "calibrated_magnitude":
        mask = magnitude < thresholds["magnitude"]
    else:
        raise KeyError(method)
    return mask, snr, magnitude


def run_method(
    method: str,
    start_index: int,
    initial: np.ndarray,
    state: dict,
    rebuild_module,
    reference_module,
    thresholds: dict[str, float],
    args: argparse.Namespace,
) -> list[dict]:
    thickness = initial.copy()
    nominal = state["nominal"]
    lower = nominal * (1.0 - args.bound_fraction)
    upper = nominal * (1.0 + args.bound_fraction)
    random_rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, start_index, METHODS.index(method)])
    )
    objective = tmm_objective(reference_module, thickness, state["wavelength"])
    best = objective
    verification_calls = 0
    assessment_calls = 1
    records = [
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "start_index": start_index,
            "iteration": 0,
            "objective": objective,
            "best_so_far_objective": best,
            "objective_over_initial": 1.0,
            "best_over_initial": 1.0,
            "components_checked_this_update": 0,
            "verification_reference_calls_this_update": 0,
            "cumulative_verification_reference_calls": 0,
            "cumulative_assessment_reference_calls": assessment_calls,
            "cumulative_total_reference_calls_if_assessment_counted": assessment_calls,
            "fd_analytic_relative_l2": float("nan"),
            "fd_analytic_sign_mismatches": 0,
        }
    ]
    initial_objective = objective

    for iteration in range(1, args.iterations + 1):
        members = member_gradients_at(state, rebuild_module, thickness)
        mean_gradient = members.mean(axis=0)
        mask, _, _ = selected_mask(
            method, members, thresholds, args.random_checks, random_rng
        )
        reference_values, calls, rel_l2, sign_mismatches = central_fd_components(
            reference_module, thickness, state["wavelength"], mask
        )
        hybrid = mean_gradient.copy()
        hybrid[mask] = reference_values[mask]
        verification_calls += calls

        dimensionless_gradient = hybrid * nominal
        norm = np.linalg.norm(dimensionless_gradient)
        if norm > 1e-14:
            direction = dimensionless_gradient / norm
            dimensionless = thickness / nominal - 1.0
            dimensionless -= args.dimensionless_step * direction
            thickness = nominal * (1.0 + dimensionless)
            thickness = np.clip(thickness, lower, upper)

        objective = tmm_objective(reference_module, thickness, state["wavelength"])
        assessment_calls += 1
        best = min(best, objective)
        records.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "start_index": start_index,
                "iteration": iteration,
                "objective": objective,
                "best_so_far_objective": best,
                "objective_over_initial": objective / initial_objective,
                "best_over_initial": best / initial_objective,
                "components_checked_this_update": int(mask.sum()),
                "verification_reference_calls_this_update": calls,
                "cumulative_verification_reference_calls": verification_calls,
                "cumulative_assessment_reference_calls": assessment_calls,
                "cumulative_total_reference_calls_if_assessment_counted": verification_calls
                + assessment_calls,
                "fd_analytic_relative_l2": rel_l2,
                "fd_analytic_sign_mismatches": sign_mismatches,
            }
        )
    return records


def summarize(trajectories: pd.DataFrame, budget_cap: int) -> pd.DataFrame:
    starts = trajectories[trajectories["iteration"] == 0][
        ["method", "start_index", "objective"]
    ].rename(columns={"objective": "initial_objective"})
    final = trajectories.groupby(["method", "start_index"], as_index=False).tail(1).copy()
    final = final.merge(starts, on=["method", "start_index"], how="left")
    final["relative_best_improvement"] = (
        final["initial_objective"] - final["best_so_far_objective"]
    ) / final["initial_objective"]
    final["failure_less_than_1pct_improvement"] = final["relative_best_improvement"] < 0.01
    final["failure_final_worse_than_start"] = final["objective"] > final["initial_objective"]

    rows: list[dict] = []
    for method in METHODS:
        frame = final[final["method"] == method]
        fd_audit = trajectories[
            (trajectories["method"] == method)
            & np.isfinite(trajectories["fd_analytic_relative_l2"])
        ]
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_starts": len(frame),
                "iterations": int(frame["iteration"].max()),
                "common_verification_budget_cap": budget_cap,
                "median_initial_objective": frame["initial_objective"].median(),
                "median_final_objective": frame["objective"].median(),
                "median_final_over_initial": frame["objective_over_initial"].median(),
                "final_objective_q25": frame["objective"].quantile(0.25),
                "final_objective_q75": frame["objective"].quantile(0.75),
                "median_best_objective": frame["best_so_far_objective"].median(),
                "best_objective_q25": frame["best_so_far_objective"].quantile(0.25),
                "best_objective_q75": frame["best_so_far_objective"].quantile(0.75),
                "median_best_over_initial": frame["best_over_initial"].median(),
                "median_relative_best_improvement": frame[
                    "relative_best_improvement"
                ].median(),
                "failure_less_than_1pct_improvement_rate": frame[
                    "failure_less_than_1pct_improvement"
                ].mean(),
                "failure_final_worse_than_start_rate": frame[
                    "failure_final_worse_than_start"
                ].mean(),
                "mean_verification_reference_calls": frame[
                    "cumulative_verification_reference_calls"
                ].mean(),
                "median_verification_reference_calls": frame[
                    "cumulative_verification_reference_calls"
                ].median(),
                "min_verification_reference_calls": frame[
                    "cumulative_verification_reference_calls"
                ].min(),
                "max_verification_reference_calls": frame[
                    "cumulative_verification_reference_calls"
                ].max(),
                "assessment_reference_calls_per_start": frame[
                    "cumulative_assessment_reference_calls"
                ].median(),
                "median_total_calls_if_assessment_counted": frame[
                    "cumulative_total_reference_calls_if_assessment_counted"
                ].median(),
                "max_fd_analytic_relative_l2": fd_audit[
                    "fd_analytic_relative_l2"
                ].max()
                if not fd_audit.empty
                else float("nan"),
                "total_fd_analytic_sign_mismatches": int(
                    fd_audit["fd_analytic_sign_mismatches"].sum()
                )
                if not fd_audit.empty
                else 0,
            }
        )
    return pd.DataFrame.from_records(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def plot_results(trajectories: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    configure_plotting()
    matched_rows = summary[summary["method"].isin(MATCHED_METHODS)]
    matched_calls = int(matched_rows["median_verification_reference_calls"].iloc[0])
    iterations = int(matched_rows["iterations"].iloc[0])
    matched_checks = matched_calls // (2 * iterations)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.35))
    ax0, ax1, ax2, ax3 = axes.ravel()

    def draw_trajectories(
        ax: plt.Axes,
        methods: tuple[str, ...],
        title: str,
        labels: dict[str, str],
        legend_columns: int,
    ) -> None:
        for method in methods:
            frame = trajectories[trajectories["method"] == method]
            grouped = frame.groupby("iteration")["best_over_initial"]
            x = np.array(sorted(frame["iteration"].unique()), dtype=int)
            median = grouped.median().reindex(x).to_numpy()
            q25 = grouped.quantile(0.25).reindex(x).to_numpy()
            q75 = grouped.quantile(0.75).reindex(x).to_numpy()
            ax.plot(
                x,
                median,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                lw=1.8,
                marker=MARKERS[method],
                markevery=(3, 4),
                markersize=3.2,
                markeredgecolor="#2E3B47",
                markeredgewidth=0.4,
                label=labels[method],
            )
            ax.fill_between(
                x,
                q25,
                q75,
                color=COLORS[method],
                alpha=0.07,
                linewidth=0,
            )
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel("Projected-gradient update")
        ax.set_xlim(0, trajectories["iteration"].max())
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis="y", color="#E3E3E3", lw=0.6, alpha=0.75)
        ax.legend(
            fontsize=5.8,
            loc="upper right",
            ncol=legend_columns,
            handlelength=2.7,
            columnspacing=0.9,
        )

    draw_trajectories(
        ax0,
        ("all_autodiff", "complete_fd"),
        "(a) Zero- and full-cost endpoints",
        {
            "all_autodiff": "All autodiff (0 calls)",
            "complete_fd": "Complete FD (440 calls)",
        },
        1,
    )
    ax0.set_ylabel("Best reference objective / initial")

    draw_trajectories(
        ax1,
        MATCHED_METHODS,
        f"(b) Matched trajectories: B = {matched_checks} ({matched_calls} calls)",
        {
            "matched_random": "Random",
            "matched_sign_order": "Sign",
            "matched_snr_order": "SNR",
            "matched_magnitude_order": "Magnitude",
        },
        2,
    )

    x_positions = np.arange(len(MATCHED_METHODS), dtype=float)
    for x, method in zip(x_positions, MATCHED_METHODS):
        row = summary[summary["method"] == method].iloc[0]
        best = row["median_best_over_initial"]
        final = row["median_final_over_initial"]
        ax2.plot([x, x], [best, final], color=COLORS[method], lw=1.35, alpha=0.95)
        ax2.scatter(x, best, s=42, marker="o", color=COLORS[method], edgecolor="#2E3B47", linewidth=0.65, zorder=3)
        ax2.scatter(x, final, s=44, marker="^", color=COLORS[method], edgecolor="#2E3B47", linewidth=0.65, zorder=3)
    ax2.set_title(f"(c) Matched endpoints: {matched_calls} calls", loc="left", pad=6)
    ax2.set_ylabel("Median objective / initial")
    ax2.set_xticks(x_positions, ["Random", "Sign", "SNR", "Magnitude"], rotation=20, ha="right")
    ax2.grid(axis="y", color="#E3E3E3", lw=0.6, alpha=0.75)
    ax2.scatter(
        [], [], s=30, marker="o", facecolors="none", edgecolors="#333333",
        linewidth=0.75, label="Best-so-far",
    )
    ax2.scatter(
        [], [], s=32, marker="^", facecolors="none", edgecolors="#333333",
        linewidth=0.75, label="Final",
    )
    ax2.legend(fontsize=6.2, loc="upper left", handletextpad=0.55)

    context_methods = (
        "all_autodiff",
        *MATCHED_METHODS,
        "calibrated_snr",
        "calibrated_magnitude",
        "complete_fd",
    )
    context_points: dict[str, tuple[float, float]] = {}
    for method in context_methods:
        row = summary[summary["method"] == method].iloc[0]
        point = (
            float(row["median_verification_reference_calls"]),
            float(row["median_best_over_initial"]),
        )
        context_points[method] = point
        ax3.scatter(
            *point,
            s=38,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolor="#2E3B47",
            linewidth=0.65,
            zorder=3,
        )

    direct_labels = {
        "all_autodiff": (16, 30, "All autodiff", "left"),
        "matched_snr_order": (-16, 30, "SNR, B = 6", "right"),
        "matched_sign_order": (-34, 0, "Sign, B = 6", "right"),
        "matched_random": (-34, 0, "Random, B = 6", "right"),
        "matched_magnitude_order": (-34, 0, "Magnitude, B = 6", "right"),
        "calibrated_snr": (-16, 30, "Calibrated SNR", "right"),
        "calibrated_magnitude": (-16, -30, "Calibrated\nmagnitude", "right"),
        "complete_fd": (-34, 0, "Complete FD", "right"),
    }
    for method, (dx, dy, label, align) in direct_labels.items():
        ax3.annotate(
            label,
            xy=context_points[method],
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.0,
            color="#333333",
            ha=align,
            va="center",
            linespacing=0.92,
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS[method],
                "lw": 0.65,
                "shrinkA": 2,
                "shrinkB": 3,
            },
        )
    ax3.set_title("(d) Verification-cost context", loc="left", pad=6)
    ax3.set_xlabel("Verification reference calls")
    ax3.set_ylabel("Median best objective / initial")
    ax3.set_xlim(-25, 465)
    ax3.set_ylim(0.048, 0.122)
    ax3.grid(axis="both", color="#E3E3E3", lw=0.6, alpha=0.75)
    ax3.text(
        0.02,
        0.97,
        "Monitoring is assessment-only",
        transform=ax3.transAxes,
        fontsize=5.8,
        color="#4D4D4D",
        va="top",
    )

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.96, hspace=0.42, wspace=0.36)
    fig.savefig(out_dir / "fig_multistep_tmm_budget.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / "fig_multistep_tmm_budget.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        out_dir / "fig_multistep_tmm_budget.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        out_dir / "fig_multistep_tmm_budget.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def mirror(paths: list[Path], mirror_dir: Path) -> None:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = mirror_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    if not 5 <= args.starts <= 10:
        raise ValueError("--starts must be between 5 and 10")
    if not 15 <= args.iterations <= 30:
        raise ValueError("--iterations must be between 15 and 30")
    if not 1 <= args.random_checks <= 10:
        raise ValueError("--random-checks must be in [1, 10]")
    if not 0.0 < args.dimensionless_step <= 0.05:
        raise ValueError("--dimensionless-step must be in (0, 0.05]")

    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    mirror_dir = args.mirror_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rebuild = load_module(data_dir / "rebuild_external_member_gradients.py", "simpat_rebuild")
    reference = load_module(data_dir / "reference_gradient_verification.py", "simpat_reference")
    thresholds, heldout_indices, sign_abstains = parse_calibration(
        data_dir / "external_baseline_split_calibration.csv"
    )
    if not sign_abstains:
        raise AssertionError("Archived stressed-TMM sign calibration should abstain")

    state = reconstruct_archived_ensemble(data_dir, rebuild, reference)
    start_indices = heldout_indices[: args.starts]
    budget_cap = 2 * state["nominal"].size * args.iterations
    records: list[dict] = []
    for start_index in start_indices:
        initial = state["queries"][start_index]
        for method in METHODS:
            records.extend(
                run_method(
                    method,
                    start_index,
                    initial,
                    state,
                    rebuild,
                    reference,
                    thresholds,
                    args,
                )
            )
    trajectories = pd.DataFrame.from_records(records)
    summary = summarize(trajectories, budget_cap)
    matched_summary = summary[summary["method"].isin(MATCHED_METHODS)].copy()

    trajectories_path = out_dir / "multistep_tmm_budget_trajectories.csv"
    summary_path = out_dir / "multistep_tmm_budget_summary.csv"
    matched_summary_path = out_dir / "multistep_tmm_matched_cost_summary.csv"
    trajectories.to_csv(trajectories_path, index=False, float_format="%.10g")
    summary.to_csv(summary_path, index=False, float_format="%.10g")
    matched_summary.to_csv(matched_summary_path, index=False, float_format="%.10g")
    plot_results(trajectories, summary, out_dir)

    protocol = {
        "benchmark": "TMM stressed",
        "objective": "minimize reflectance at archived lam_eval",
        "lam_eval_nm": state["wavelength"],
        "heldout_start_indices": start_indices,
        "starts": args.starts,
        "iterations": args.iterations,
        "dimensionless_step": args.dimensionless_step,
        "bounds_relative_to_nominal": [-args.bound_fraction, args.bound_fraction],
        "matched_cost_methods": list(MATCHED_METHODS),
        "matched_component_checks_per_update": args.random_checks,
        "matched_verification_reference_calls_per_start": 2
        * args.random_checks
        * args.iterations,
        "matched_selection_direction": "verify the B lowest reliability scores; sign-score ties use seeded random secondary ordering",
        "common_verification_reference_call_cap": budget_cap,
        "central_fd_step_nm": 0.5,
        "snr_threshold": thresholds["snr"],
        "magnitude_threshold": thresholds["magnitude"],
        "sign_calibration": "abstain; equivalent to complete central FD",
        "assessment_objective_calls": "one at start and after each update; reported separately and not supplied to optimizers",
        "ensemble_rebuild_max_abs_gradient_difference": state[
            "max_abs_rebuild_difference"
        ],
        "random_seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    protocol_path = out_dir / "multistep_tmm_budget_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    script_path = Path(__file__).resolve()
    outputs = [
        script_path,
        trajectories_path,
        summary_path,
        matched_summary_path,
        protocol_path,
        out_dir / "fig_multistep_tmm_budget.svg",
        out_dir / "fig_multistep_tmm_budget.pdf",
        out_dir / "fig_multistep_tmm_budget.png",
        out_dir / "fig_multistep_tmm_budget.tiff",
    ]
    mirror(outputs, mirror_dir)

    print(json.dumps(protocol, indent=2))
    print(summary.to_string(index=False))
    matched_calls = 2 * args.random_checks * args.iterations
    print(f"\nStrict matched-cost comparison (all methods use exactly {matched_calls} calls):")
    print(matched_summary.to_string(index=False))
    print(f"Mirrored outputs to {mirror_dir}")


if __name__ == "__main__":
    main()
