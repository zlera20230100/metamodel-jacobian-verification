#!/usr/bin/env python3
"""Rebuild and freeze per-member gradients for the three external benchmarks.

The original frozen artifacts retain sign agreement and sign-correctness labels,
but not the member gradients required to audit SNR and magnitude baselines.  This
script replays the archived benchmark definitions and fixed training seeds without
modifying ``repo_paper2``.  A rebuilt artifact is written only after exact agreement
with the original ``all_sa`` and ``all_correct`` arrays, plus numerical agreement
with the archived AUC and zero-correction cosine.

Run without ``--worker`` to launch each benchmark in a fresh Python process.  This
preserves the benchmark-specific PyTorch thread settings.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


DATASET_FILES = {
    "tmm_easy": ("extbench_tmm.npz", "external_members_tmm_easy.npz"),
    "tmm_stressed": ("extbench_tmm_hard.npz", "external_members_tmm_stressed.npz"),
    "heat_poisson_stressed": (
        "extbench_poisson.npz",
        "external_members_heat_poisson_stressed.npz",
    ),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_repro = here.parent / "reproducibility_update"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASET_FILES),
        default="all",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--frozen-dir", type=Path, default=default_repro)
    parser.add_argument("--out-dir", type=Path, default=default_repro)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with average ranks for ties, avoiding a new runtime dependency."""

    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train_ensemble(X: np.ndarray, Y: np.ndarray, members: int = 10):
    import torch
    import torch.nn as nn

    k = X.shape[1]
    xm, xs = X.mean(0), X.std(0)
    ym, ysd = Y.mean(), Y.std()
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
    for seed in range(members):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        net = MLP()
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
        loss_function = nn.MSELoss()
        n = Xt.size(0)
        indices = torch.randperm(n, generator=generator)
        train_indices = indices[: int(0.9 * n)]
        for _ in range(400):
            permutation = train_indices[
                torch.randperm(train_indices.numel(), generator=generator)
            ]
            for batch in permutation.split(256):
                optimizer.zero_grad()
                loss_function(net(Xt[batch]), Yt[batch]).backward()
                optimizer.step()
        nets.append(net)
    return nets, xm, xs, float(ym), float(ysd)


def member_gradient(net, xphys, xm, xs, ysd):
    import torch

    xn = torch.tensor(((xphys - xm) / xs), dtype=torch.float32).view(1, -1)
    xn.requires_grad_(True)
    net(xn).backward()
    return xn.grad.detach().numpy().ravel() * (ysd / xs)


def run_tmm(stressed: bool) -> dict:
    import torch

    if stressed:
        torch.set_num_threads(3)

    lam0 = 1550.0
    nH, nL, nsub, n0 = 2.35, 1.45, 1.52, 1.0
    idx = np.array([nH, nL] * 5 + [nH])
    nominal = lam0 / (4.0 * idx)
    nominal[len(idx) // 2] *= 2.0
    k = idx.size

    def tmm_R(thick, lam):
        matrix = np.eye(2, dtype=complex)
        for nj, dj in zip(idx, thick):
            delta = 2.0 * np.pi * nj * dj / lam
            c, s = np.cos(delta), np.sin(delta)
            matrix = matrix @ np.array(
                [[c, 1j * s / nj], [1j * nj * s, c]], dtype=complex
            )
        B = matrix[0, 0] + matrix[0, 1] * nsub
        C = matrix[1, 0] + matrix[1, 1] * nsub
        reflection = (n0 * B - C) / (n0 * B + C)
        return float(np.abs(reflection) ** 2)

    if stressed:
        scan = np.linspace(lam0 - 80, lam0 + 80, 1601)
        ns, nq, roam = 250, 100, 0.10
    else:
        scan = np.linspace(lam0 - 60, lam0 + 60, 481)
        ns, nq, roam = 600, 80, 0.12
    reflectance = np.array([tmm_R(nominal, lam) for lam in scan])
    upper = scan > lam0
    lam_eval = float(scan[upper][np.argmin(np.abs(reflectance[upper] - 0.4))])

    def reference_gradient(thick, h=0.5):
        gradient = np.zeros(k)
        for component in range(k):
            plus = thick.copy()
            minus = thick.copy()
            plus[component] += h
            minus[component] -= h
            gradient[component] = (
                tmm_R(plus, lam_eval) - tmm_R(minus, lam_eval)
            ) / (2.0 * h)
        return gradient

    rng = np.random.default_rng(7)
    X = nominal[None, :] * (1.0 + 0.30 * (2.0 * rng.random((ns, k)) - 1.0))
    Y = np.array([tmm_R(row, lam_eval) for row in X])
    nets, xm, xs, _, ysd = train_ensemble(X, Y)
    queries = nominal[None, :] * (1.0 + roam * (2.0 * rng.random((nq, k)) - 1.0))

    member_gradients = np.empty((nq, len(nets), k), dtype=float)
    reference_gradients = np.empty((nq, k), dtype=float)
    for query_index, query in enumerate(queries):
        member_gradients[query_index] = np.array(
            [member_gradient(net, query, xm, xs, ysd) for net in nets]
        )
        reference_gradients[query_index] = reference_gradient(query)

    return {
        "query_points": queries,
        "member_gradients": member_gradients,
        "reference_gradients": reference_gradients,
        "NS": ns,
        "NQ": nq,
        "M": len(nets),
        "K": k,
        "lam_eval": lam_eval,
        "dataset": "TMM stressed" if stressed else "TMM easy (near-saturated)",
    }


def run_heat_poisson() -> dict:
    import torch

    torch.set_num_threads(6)
    ngrid = 160
    TL, TR = 1.0, 0.0
    xc = (np.arange(ngrid) + 0.5) / ngrid
    hgrid = 1.0 / ngrid
    k, q_sharp, ns, roam, span, nq = 16, 3.0, 70, 0.25, 0.8, 100

    segment = np.minimum((xc * k).astype(int), k - 1)
    source = 40.0 * np.exp(-((xc - 0.30) / (0.06 + 0.14 / q_sharp)) ** 2)
    source -= 25.0 * np.exp(-((xc - 0.70) / 0.08) ** 2)
    rng_true = np.random.default_rng(3)
    g_true = np.exp(rng_true.normal(0.0, 0.8, k))

    def conduction_solve(gseg):
        gcell = np.asarray(gseg, dtype=float)[segment]
        faces = 2.0 * gcell[:-1] * gcell[1:] / (gcell[:-1] + gcell[1:])
        inv_h2 = 1.0 / (hgrid * hgrid)
        n = source.size
        lower = np.zeros(n)
        diagonal = np.zeros(n)
        upper = np.zeros(n)
        rhs = source.copy()
        lower[1:] = -faces * inv_h2
        diagonal[1:] += faces * inv_h2
        upper[:-1] = -faces * inv_h2
        diagonal[:-1] += faces * inv_h2
        diagonal[0] += 2.0 * gcell[0] * inv_h2
        rhs[0] += 2.0 * gcell[0] * inv_h2 * TL
        diagonal[-1] += 2.0 * gcell[-1] * inv_h2
        rhs[-1] += 2.0 * gcell[-1] * inv_h2 * TR
        cprime = upper.copy()
        dprime = rhs.copy()
        cprime[0] /= diagonal[0]
        dprime[0] /= diagonal[0]
        for index in range(1, n):
            denominator = diagonal[index] - lower[index] * cprime[index - 1]
            cprime[index] = upper[index] / denominator
            dprime[index] = (
                rhs[index] - lower[index] * dprime[index - 1]
            ) / denominator
        temperature = np.empty(n)
        temperature[-1] = dprime[-1]
        for index in range(n - 2, -1, -1):
            temperature[index] = dprime[index] - cprime[index] * temperature[index + 1]
        return temperature

    target = conduction_solve(g_true)

    def objective(g):
        temperature = conduction_solve(g)
        return float(np.mean((temperature - target) ** 2))

    def reference_gradient(g, relative_step=1e-3):
        gradient = np.zeros(k)
        for component in range(k):
            step = relative_step * max(abs(g[component]), 1e-2)
            plus = g.copy()
            minus = g.copy()
            plus[component] += step
            minus[component] -= step
            gradient[component] = (objective(plus) - objective(minus)) / (2.0 * step)
        return gradient

    nominal = np.full(k, 1.0)
    rng = np.random.default_rng(7)
    X = nominal[None, :] * (1.0 + span * (2.0 * rng.random((ns, k)) - 1.0))
    X = np.clip(X, 0.05, None)
    Y = np.array([objective(row) for row in X])
    nets, xm, xs, _, ysd = train_ensemble(X, Y)
    rng_queries = np.random.default_rng(11)
    queries = nominal[None, :] * (
        1.0 + roam * (2.0 * rng_queries.random((nq, k)) - 1.0)
    )
    queries = np.clip(queries, 0.05, None)

    member_gradients = np.empty((nq, len(nets), k), dtype=float)
    reference_gradients = np.empty((nq, k), dtype=float)
    for query_index, query in enumerate(queries):
        member_gradients[query_index] = np.array(
            [member_gradient(net, query, xm, xs, ysd) for net in nets]
        )
        reference_gradients[query_index] = reference_gradient(query)

    return {
        "query_points": queries,
        "member_gradients": member_gradients,
        "reference_gradients": reference_gradients,
        "NS": ns,
        "NQ": nq,
        "M": len(nets),
        "K": k,
        "ngrid": ngrid,
        "q_sharp": q_sharp,
        "roam": roam,
        "span": span,
        "dataset": "Heat/Poisson stressed",
    }


def validate_and_save(result: dict, dataset_key: str, frozen_dir: Path, out_dir: Path) -> Path:
    frozen_name, output_name = DATASET_FILES[dataset_key]
    frozen_path = frozen_dir / frozen_name
    if not frozen_path.is_file():
        raise FileNotFoundError(frozen_path)

    members = result["member_gradients"]
    mean_gradient = members.mean(axis=1)
    reference = result["reference_gradients"]
    sign_agreement = np.maximum((members > 0).mean(axis=1), (members < 0).mean(axis=1))
    mean_sign_correct = (np.sign(mean_gradient) == np.sign(reference)).astype(np.int8)
    cosine = np.sum(mean_gradient * reference, axis=1) / (
        np.linalg.norm(mean_gradient, axis=1) * np.linalg.norm(reference, axis=1) + 1e-12
    )
    auc = roc_auc_binary(mean_sign_correct, sign_agreement)

    with np.load(frozen_path, allow_pickle=False) as frozen:
        archived_sa = np.asarray(frozen["all_sa"], dtype=float).reshape(sign_agreement.shape)
        archived_correct = np.asarray(frozen["all_correct"], dtype=np.int8).reshape(
            mean_sign_correct.shape
        )
        archived_auc = float(np.asarray(frozen["auc"]).item())
        archived_cosine = float(np.asarray(frozen["cos_ad"]).item())

    if not np.array_equal(sign_agreement, archived_sa):
        differing = int(np.count_nonzero(sign_agreement != archived_sa))
        max_difference = float(np.max(np.abs(sign_agreement - archived_sa)))
        raise AssertionError(
            f"{dataset_key}: rebuilt sign agreement differs from frozen array at "
            f"{differing} components (max difference {max_difference})"
        )
    if not np.array_equal(mean_sign_correct, archived_correct):
        differing = int(np.count_nonzero(mean_sign_correct != archived_correct))
        raise AssertionError(
            f"{dataset_key}: rebuilt sign-correct labels differ at {differing} components"
        )
    if not np.isclose(auc, archived_auc, rtol=0.0, atol=1e-14):
        raise AssertionError(f"{dataset_key}: rebuilt AUC {auc} != frozen {archived_auc}")
    if not np.isclose(cosine.mean(), archived_cosine, rtol=0.0, atol=1e-7):
        raise AssertionError(
            f"{dataset_key}: rebuilt zero-budget cosine {cosine.mean()} != frozen {archived_cosine}"
        )

    positive_votes = (members > 0).sum(axis=1)
    negative_votes = (members < 0).sum(axis=1)
    modal_sign = np.sign(positive_votes - negative_votes).astype(np.int8)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / output_name
    payload = dict(result)
    payload.update(
        {
            "mean_gradients": mean_gradient,
            "sign_agreement": sign_agreement,
            "mean_sign_correct": mean_sign_correct,
            "modal_sign": modal_sign,
            "positive_votes": positive_votes,
            "negative_votes": negative_votes,
            "headline_auc": auc,
            "headline_cosine": float(cosine.mean()),
            "frozen_source_file": np.array(frozen_name),
            "frozen_source_sha256": np.array(file_sha256(frozen_path)),
            "rebuild_python": np.array(sys.version.split()[0]),
        }
    )
    np.savez_compressed(output_path, **payload)
    print(
        f"validated {dataset_key}: sign arrays exact, AUC={auc:.12f}, "
        f"cosine={cosine.mean():.12f}; wrote {output_path}",
        flush=True,
    )
    return output_path


def worker(dataset: str, frozen_dir: Path, out_dir: Path) -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if dataset == "tmm_stressed":
        os.environ.setdefault("OMP_NUM_THREADS", "3")
        os.environ.setdefault("MKL_NUM_THREADS", "3")
        result = run_tmm(stressed=True)
    elif dataset == "tmm_easy":
        result = run_tmm(stressed=False)
    elif dataset == "heat_poisson_stressed":
        os.environ.setdefault("OMP_NUM_THREADS", "6")
        os.environ.setdefault("MKL_NUM_THREADS", "6")
        result = run_heat_poisson()
    else:
        raise ValueError(dataset)
    validate_and_save(result, dataset, frozen_dir, out_dir)


def orchestrate(args: argparse.Namespace) -> None:
    datasets = list(DATASET_FILES) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset",
            dataset,
            "--worker",
            "--frozen-dir",
            str(args.frozen_dir.resolve()),
            "--out-dir",
            str(args.out_dir.resolve()),
        ]
        print(f"launching isolated worker: {dataset}", flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.dataset == "all":
            raise ValueError("--worker requires one dataset")
        worker(args.dataset, args.frozen_dir.resolve(), args.out_dir.resolve())
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
