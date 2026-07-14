#!/usr/bin/env python3
"""Audit sensitivity to training-data and ensemble-member random seeds.

The archived query points and reference gradients are held fixed.  Each outer
repeat regenerates only the surrogate training sample and trains a fresh block
of ten MLP members.  The script reports sign-correctness AUC, risk and coverage
at tau=0.9, and the zero-reference-budget cosine of the ensemble-mean gradient.

This script covers only the three external TMM and heat/Poisson benchmarks.  It
does not import, invoke, or retrain the antenna/openEMS workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


REGIMES = (
    "tmm_easy",
    "tmm_stressed",
    "heat_poisson_stressed",
)

ARCHIVES = {
    "tmm_easy": "external_members_tmm_easy.npz",
    "tmm_stressed": "external_members_tmm_stressed.npz",
    "heat_poisson_stressed": "external_members_heat_poisson_stressed.npz",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--data-seed-base", type=int, default=17001)
    parser.add_argument("--member-seed-base", type=int, default=29000)
    parser.add_argument("--archive-dir", type=Path, default=here)
    parser.add_argument("--out-dir", type=Path, default=here)
    parser.add_argument("--mirror-dir", type=Path, default=here)
    parser.add_argument("--regime", choices=("all", *REGIMES), default="all")
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
        / (n_pos * n_neg)
    )


def train_ensemble(
    X: np.ndarray,
    Y: np.ndarray,
    member_seed_start: int,
    members: int,
) -> tuple[list, np.ndarray, np.ndarray, float]:
    import torch
    import torch.nn as nn

    k = X.shape[1]
    xm, xs = X.mean(axis=0), X.std(axis=0)
    ym, ysd = float(Y.mean()), float(Y.std())
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32)
    Yt = torch.tensor((Y - ym) / ysd, dtype=torch.float32).view(-1, 1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(k, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x)

    nets = []
    for offset in range(members):
        seed = member_seed_start + offset
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        net = MLP()
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
        loss_function = nn.MSELoss()
        indices = torch.randperm(Xt.size(0), generator=generator)
        train_indices = indices[: int(0.9 * Xt.size(0))]
        for _ in range(400):
            permutation = train_indices[
                torch.randperm(train_indices.numel(), generator=generator)
            ]
            for batch in permutation.split(256):
                optimizer.zero_grad()
                loss_function(net(Xt[batch]), Yt[batch]).backward()
                optimizer.step()
        net.eval()
        nets.append(net)
    return nets, xm, xs, ysd


def member_gradients(
    nets: list,
    queries: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ysd: float,
) -> np.ndarray:
    import torch

    gradients = np.empty((queries.shape[0], len(nets), queries.shape[1]), dtype=float)
    for query_index, query in enumerate(queries):
        for member_index, net in enumerate(nets):
            xn = torch.tensor((query - xm) / xs, dtype=torch.float32).view(1, -1)
            xn.requires_grad_(True)
            derivative = torch.autograd.grad(net(xn), xn)[0]
            gradients[query_index, member_index] = (
                derivative.detach().numpy().ravel() * (ysd / xs)
            )
    return gradients


def tmm_problem(stressed: bool):
    lam0 = 1550.0
    n_high, n_low, n_substrate, n_incident = 2.35, 1.45, 1.52, 1.0
    indices = np.array([n_high, n_low] * 5 + [n_high])
    nominal = lam0 / (4.0 * indices)
    nominal[len(indices) // 2] *= 2.0

    def reflectance(thickness, wavelength):
        matrix = np.eye(2, dtype=complex)
        for refractive_index, depth in zip(indices, thickness):
            delta = 2.0 * np.pi * refractive_index * depth / wavelength
            cosine, sine = np.cos(delta), np.sin(delta)
            matrix = matrix @ np.array(
                [
                    [cosine, 1j * sine / refractive_index],
                    [1j * refractive_index * sine, cosine],
                ],
                dtype=complex,
            )
        B = matrix[0, 0] + matrix[0, 1] * n_substrate
        C = matrix[1, 0] + matrix[1, 1] * n_substrate
        coefficient = (n_incident * B - C) / (n_incident * B + C)
        return float(np.abs(coefficient) ** 2)

    if stressed:
        scan = np.linspace(lam0 - 80, lam0 + 80, 1601)
        ns = 250
    else:
        scan = np.linspace(lam0 - 60, lam0 + 60, 481)
        ns = 600
    scan_values = np.array([reflectance(nominal, wavelength) for wavelength in scan])
    upper = scan > lam0
    wavelength = float(scan[upper][np.argmin(np.abs(scan_values[upper] - 0.4))])

    def training_sample(seed: int):
        rng = np.random.default_rng(seed)
        X = nominal[None, :] * (
            1.0 + 0.30 * (2.0 * rng.random((ns, nominal.size)) - 1.0)
        )
        Y = np.array([reflectance(row, wavelength) for row in X])
        return X, Y

    return training_sample


def heat_poisson_problem():
    ngrid = 160
    left_temperature, right_temperature = 1.0, 0.0
    centres = (np.arange(ngrid) + 0.5) / ngrid
    hgrid = 1.0 / ngrid
    k, q_sharp, ns, span = 16, 3.0, 70, 0.8
    segment = np.minimum((centres * k).astype(int), k - 1)
    source = 40.0 * np.exp(
        -((centres - 0.30) / (0.06 + 0.14 / q_sharp)) ** 2
    )
    source -= 25.0 * np.exp(-((centres - 0.70) / 0.08) ** 2)
    true_rng = np.random.default_rng(3)
    true_conductivity = np.exp(true_rng.normal(0.0, 0.8, k))

    def solve(conductivity):
        cell = np.asarray(conductivity, dtype=float)[segment]
        faces = 2.0 * cell[:-1] * cell[1:] / (cell[:-1] + cell[1:])
        inv_h2 = 1.0 / (hgrid * hgrid)
        lower = np.zeros(ngrid)
        diagonal = np.zeros(ngrid)
        upper = np.zeros(ngrid)
        rhs = source.copy()
        lower[1:] = -faces * inv_h2
        diagonal[1:] += faces * inv_h2
        upper[:-1] = -faces * inv_h2
        diagonal[:-1] += faces * inv_h2
        diagonal[0] += 2.0 * cell[0] * inv_h2
        rhs[0] += 2.0 * cell[0] * inv_h2 * left_temperature
        diagonal[-1] += 2.0 * cell[-1] * inv_h2
        rhs[-1] += 2.0 * cell[-1] * inv_h2 * right_temperature
        cprime, dprime = upper.copy(), rhs.copy()
        cprime[0] /= diagonal[0]
        dprime[0] /= diagonal[0]
        for index in range(1, ngrid):
            denominator = diagonal[index] - lower[index] * cprime[index - 1]
            cprime[index] = upper[index] / denominator
            dprime[index] = (
                rhs[index] - lower[index] * dprime[index - 1]
            ) / denominator
        temperature = np.empty(ngrid)
        temperature[-1] = dprime[-1]
        for index in range(ngrid - 2, -1, -1):
            temperature[index] = dprime[index] - cprime[index] * temperature[index + 1]
        return temperature

    target = solve(true_conductivity)

    def objective(conductivity):
        return float(np.mean((solve(conductivity) - target) ** 2))

    def training_sample(seed: int):
        rng = np.random.default_rng(seed)
        X = 1.0 + span * (2.0 * rng.random((ns, k)) - 1.0)
        X = np.clip(X, 0.05, None)
        Y = np.array([objective(row) for row in X])
        return X, Y

    return training_sample


def load_fixed_evidence(archive_dir: Path, regime: str):
    archive_path = archive_dir / ARCHIVES[regime]
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        queries = np.asarray(archive["query_points"], dtype=float).copy()
        reference = np.asarray(archive["reference_gradients"], dtype=float).copy()
    return archive_path, queries, reference


def calculate_metrics(
    gradients: np.ndarray,
    reference: np.ndarray,
    tau: float,
) -> dict[str, float]:
    mean_gradient = gradients.mean(axis=1)
    agreement = np.maximum(
        (gradients > 0).mean(axis=1),
        (gradients < 0).mean(axis=1),
    )
    correct = np.sign(mean_gradient) == np.sign(reference)
    accepted = agreement >= tau
    coverage = float(accepted.mean())
    risk = float((~correct[accepted]).mean()) if accepted.any() else float("nan")
    cosine = np.sum(mean_gradient * reference, axis=1) / (
        np.linalg.norm(mean_gradient, axis=1)
        * np.linalg.norm(reference, axis=1)
        + 1e-12
    )
    return {
        "auc": roc_auc_binary(correct, agreement),
        "coverage_tau_0_9": coverage,
        "risk_tau_0_9": risk,
        "zero_budget_cosine": float(cosine.mean()),
        "fraction_sign_correct": float(correct.mean()),
        "accepted_components": int(accepted.sum()),
        "total_components": int(accepted.size),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_rows: list[dict]) -> list[dict]:
    summary_rows = []
    metrics = (
        "auc",
        "coverage_tau_0_9",
        "risk_tau_0_9",
        "zero_budget_cosine",
        "fraction_sign_correct",
        "elapsed_seconds",
    )
    for regime in REGIMES:
        selected = [row for row in run_rows if row["regime"] == regime]
        if not selected:
            continue
        for metric in metrics:
            values = np.array([float(row[metric]) for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "n_outer_repeats": len(selected),
                    "n_finite": finite.size,
                    "mean": float(finite.mean()) if finite.size else float("nan"),
                    "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                    "min": float(finite.min()) if finite.size else float("nan"),
                    "median": float(np.median(finite)) if finite.size else float("nan"),
                    "max": float(finite.max()) if finite.size else float("nan"),
                }
            )
    return summary_rows


def mirror(paths: list[Path], mirror_dir: Path) -> None:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = mirror_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.members < 2:
        raise ValueError("--repeats must be >=1 and --members must be >=2")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    os.environ.setdefault("MKL_NUM_THREADS", "3")
    import torch

    torch.set_num_threads(3)
    selected_regimes = REGIMES if args.regime == "all" else (args.regime,)
    generators = {
        "tmm_easy": tmm_problem(stressed=False),
        "tmm_stressed": tmm_problem(stressed=True),
        "heat_poisson_stressed": heat_poisson_problem(),
    }
    evidence = {}
    for regime in selected_regimes:
        archive_path, queries, reference = load_fixed_evidence(
            args.archive_dir.resolve(), regime
        )
        evidence[regime] = (archive_path, queries, reference)

    started = time.perf_counter()
    run_rows = []
    for repeat in range(args.repeats):
        data_seed = args.data_seed_base + repeat
        member_start = args.member_seed_base + repeat * args.members
        for regime in selected_regimes:
            archive_path, queries, reference = evidence[regime]
            run_started = time.perf_counter()
            X, Y = generators[regime](data_seed)
            nets, xm, xs, ysd = train_ensemble(
                X, Y, member_start, args.members
            )
            gradients = member_gradients(nets, queries, xm, xs, ysd)
            metrics = calculate_metrics(gradients, reference, args.tau)
            elapsed = time.perf_counter() - run_started
            row = {
                "regime": regime,
                "outer_repeat": repeat,
                "training_data_seed": data_seed,
                "member_seed_start": member_start,
                "member_seed_end": member_start + args.members - 1,
                "members": args.members,
                "training_samples": X.shape[0],
                "queries": queries.shape[0],
                "components": queries.shape[1],
                "tau": args.tau,
                "archive_file": archive_path.name,
                "query_points_sha256": array_sha256(queries),
                "reference_gradients_sha256": array_sha256(reference),
                **metrics,
                "elapsed_seconds": elapsed,
            }
            run_rows.append(row)
            print(
                f"repeat={repeat} regime={regime} seed={data_seed} "
                f"members={member_start}-{member_start + args.members - 1} "
                f"AUC={metrics['auc']:.4f} coverage={metrics['coverage_tau_0_9']:.4f} "
                f"risk={metrics['risk_tau_0_9']:.4f} "
                f"cos0={metrics['zero_budget_cosine']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            del nets, gradients

    total_elapsed = time.perf_counter() - started
    out_dir = args.out_dir.resolve()
    runs_path = out_dir / "training_repeat_audit_runs.csv"
    summary_path = out_dir / "training_repeat_audit_summary.csv"
    log_path = out_dir / "training_repeat_audit_runlog.txt"
    run_fields = list(run_rows[0].keys())
    write_csv(runs_path, run_rows, run_fields)
    summary_rows = summarize(run_rows)
    summary_fields = list(summary_rows[0].keys())
    write_csv(summary_path, summary_rows, summary_fields)

    command = " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    log_text = (
        "Training-repeat audit\n"
        f"command: {command}\n"
        f"python: {sys.version.replace(os.linesep, ' ')}\n"
        f"torch: {torch.__version__}\n"
        f"regimes: {','.join(selected_regimes)}\n"
        f"outer_repeats: {args.repeats}\n"
        f"members_per_repeat: {args.members}\n"
        f"tau: {args.tau}\n"
        f"total_elapsed_seconds: {total_elapsed:.6f}\n"
        "fixed_evidence: archived query_points and reference_gradients; hashes are in the run CSV\n"
        "excluded: antenna training and openEMS\n"
    )
    log_path.write_text(log_text, encoding="utf-8")
    script_path = Path(__file__).resolve()
    mirror([script_path, runs_path, summary_path, log_path], args.mirror_dir.resolve())
    print(f"wrote {runs_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {log_path}", flush=True)
    print(f"mirrored outputs to {args.mirror_dir.resolve()}", flush=True)
    print(f"total elapsed {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
