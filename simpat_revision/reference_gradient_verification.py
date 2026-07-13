"""Numerical verification of the frozen external reference-gradient labels.

This script reconstructs the exact query designs used by the released TMM and
1-D heat/Poisson benchmark generators.  It does *not* retrain the metamodels.
Instead it verifies the central-difference derivative used as the label against
an independent derivative of the same reference model:

* TMM: analytic differentiation of the characteristic-matrix product.
* Heat/Poisson: complex-step differentiation through the discrete tridiagonal
  finite-volume solve, plus an 80/160/320/640-cell grid study.

The archived per-member gradients are used to reconstruct the operational
mean-aligned-sign score and ensemble-mean correctness label. Components are
filtered only if the reference derivative is declared unresolved, so the script
shows whether numerical resolution changes the reported AUC/risk values.

Outputs
-------
reference_gradient_verification_components.csv
reference_gradient_verification_summary.csv
reference_gradient_label_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Resolve portable input/output paths for analysis and packaged use."""
    local_member_file = HERE / "external_members_tmm_easy.npz"
    default_data = HERE if local_member_file.is_file() else HERE.parent / "reproducibility_update"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    return parser.parse_args()


def safe_rel(a: float, b: float, floor: float) -> float:
    """Symmetric relative difference with a query-scale denominator floor."""
    return abs(a - b) / max(abs(a), abs(b), floor)


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC from average ranks; avoids a scikit-learn dependency."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=float)
    i = 0
    while i < score.size:
        j = i + 1
        while j < score.size and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sign_with_tol(v: float, tol: float) -> int:
    return 0 if abs(v) <= tol else (1 if v > 0 else -1)


# ---------------------------------------------------------------------------
# Thin-film transfer-matrix model and exact derivative
# ---------------------------------------------------------------------------


N_H, N_L, N_SUB, N_0 = 2.35, 1.45, 1.52, 1.0
LAM0 = 1550.0
IDX = np.array([N_H, N_L] * 5 + [N_H], dtype=float)
QW = LAM0 / (4.0 * IDX)
TMM_NOMINAL = QW.copy()
TMM_NOMINAL[5] *= 2.0


def tmm_r_and_grad(thick: np.ndarray, lam: float) -> tuple[complex, np.ndarray]:
    """Amplitude reflection coefficient and analytic dR/dd for all layers."""
    mats: list[np.ndarray] = []
    dmats: list[np.ndarray] = []
    for n_j, d_j in zip(IDX, thick):
        delta = 2.0 * np.pi * n_j * d_j / lam
        c, s = np.cos(delta), np.sin(delta)
        mats.append(np.array([[c, 1j * s / n_j],
                              [1j * n_j * s, c]], dtype=complex))
        dd = 2.0 * np.pi * n_j / lam
        dmats.append(dd * np.array([[-s, 1j * c / n_j],
                                    [1j * n_j * c, -s]], dtype=complex))

    prefix = [np.eye(2, dtype=complex)]
    for mat in mats:
        prefix.append(prefix[-1] @ mat)
    suffix = [np.eye(2, dtype=complex) for _ in range(len(mats) + 1)]
    for j in range(len(mats) - 1, -1, -1):
        suffix[j] = mats[j] @ suffix[j + 1]

    m = prefix[-1]
    b = m[0, 0] + m[0, 1] * N_SUB
    cval = m[1, 0] + m[1, 1] * N_SUB
    num = N_0 * b - cval
    den = N_0 * b + cval
    r = num / den
    grad = np.empty(len(mats), dtype=float)
    for j in range(len(mats)):
        dm = prefix[j] @ dmats[j] @ suffix[j + 1]
        db = dm[0, 0] + dm[0, 1] * N_SUB
        dc = dm[1, 0] + dm[1, 1] * N_SUB
        dnum = N_0 * db - dc
        dden = N_0 * db + dc
        dr = (dnum * den - num * dden) / (den * den)
        grad[j] = 2.0 * np.real(np.conj(r) * dr)
    return r, grad


def tmm_r(thick: np.ndarray, lam: float) -> float:
    r, _ = tmm_r_and_grad(thick, lam)
    return float(abs(r) ** 2)


def tmm_fd(thick: np.ndarray, lam: float, h: float) -> np.ndarray:
    out = np.empty(thick.size, dtype=float)
    for k in range(thick.size):
        xp = thick.copy(); xp[k] += h
        xm = thick.copy(); xm[k] -= h
        out[k] = (tmm_r(xp, lam) - tmm_r(xm, lam)) / (2.0 * h)
    return out


def tmm_working_point(tag: str) -> float:
    if tag == "TMM_easy":
        scan = np.linspace(LAM0 - 60.0, LAM0 + 60.0, 481)
    else:
        scan = np.linspace(LAM0 - 80.0, LAM0 + 80.0, 1601)
    rs = np.array([tmm_r(TMM_NOMINAL, x) for x in scan])
    up = scan > LAM0
    return float(scan[up][np.argmin(np.abs(rs[up] - 0.4))])


def reconstruct_tmm_queries(tag: str) -> tuple[np.ndarray, float]:
    if tag == "TMM_easy":
        ns, nq, roam = 600, 80, 0.12
    else:
        ns, nq, roam = 250, 100, 0.10
    rng = np.random.default_rng(7)
    # Consume the released training-design draw before the query draw.
    rng.random((ns, IDX.size))
    qp = TMM_NOMINAL[None, :] * (
        1.0 + roam * (2.0 * rng.random((nq, IDX.size)) - 1.0)
    )
    return qp, tmm_working_point(tag)


# ---------------------------------------------------------------------------
# 1-D heat/Poisson finite-volume reference and complex-step derivative
# ---------------------------------------------------------------------------


def heat_problem(ngrid: int, kseg: int = 16, q_sharp: float = 3.0,
                 gtrue_seed: int = 3):
    xc = (np.arange(ngrid) + 0.5) / ngrid
    hgrid = 1.0 / ngrid
    seg = np.minimum((xc * kseg).astype(int), kseg - 1)
    q = (40.0 * np.exp(-((xc - 0.30) / (0.06 + 0.14 / q_sharp)) ** 2)
         - 25.0 * np.exp(-((xc - 0.70) / 0.08) ** 2))
    rng = np.random.default_rng(gtrue_seed)
    gtrue = np.exp(rng.normal(0.0, 0.8, kseg))

    def solve(gseg: np.ndarray) -> np.ndarray:
        gseg = np.asarray(gseg)
        dtype = np.result_type(gseg.dtype, np.float64)
        gcell = gseg[seg].astype(dtype, copy=False)
        gf = 2.0 * gcell[:-1] * gcell[1:] / (gcell[:-1] + gcell[1:])
        inv_h2 = 1.0 / (hgrid * hgrid)
        n = q.size
        lower = np.zeros(n, dtype=dtype)
        diag = np.zeros(n, dtype=dtype)
        upper = np.zeros(n, dtype=dtype)
        b = q.astype(dtype, copy=True)
        lower[1:] = -gf * inv_h2; diag[1:] += gf * inv_h2
        upper[:-1] = -gf * inv_h2; diag[:-1] += gf * inv_h2
        diag[0] += 2.0 * gcell[0] * inv_h2
        b[0] += 2.0 * gcell[0] * inv_h2 * 1.0
        diag[-1] += 2.0 * gcell[-1] * inv_h2
        # Right Dirichlet boundary is zero, hence no b contribution.
        cp = upper.copy(); dp = b.copy()
        cp[0] /= diag[0]; dp[0] /= diag[0]
        for i in range(1, n):
            den = diag[i] - lower[i] * cp[i - 1]
            cp[i] = upper[i] / den
            dp[i] = (b[i] - lower[i] * dp[i - 1]) / den
        t = np.empty(n, dtype=dtype)
        t[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            t[i] = dp[i] - cp[i] * t[i + 1]
        return t

    target = solve(gtrue)

    def objective(gseg: np.ndarray):
        dt = solve(gseg) - target
        # Deliberately no conjugate: this is the holomorphic extension required
        # by complex-step differentiation of the real squared-error objective.
        return np.mean(dt * dt)

    return objective


def heat_fd(objective, g: np.ndarray, rel: float) -> np.ndarray:
    out = np.empty(g.size, dtype=float)
    for k in range(g.size):
        h = rel * max(abs(g[k]), 1e-2)
        xp = g.copy(); xp[k] += h
        xm = g.copy(); xm[k] -= h
        out[k] = float((objective(xp) - objective(xm)) / (2.0 * h))
    return out


def heat_complex_step(objective, g: np.ndarray, h: float = 1e-24) -> np.ndarray:
    out = np.empty(g.size, dtype=float)
    for k in range(g.size):
        z = np.asarray(g, dtype=complex).copy()
        z[k] += 1j * h
        out[k] = float(np.imag(objective(z)) / h)
    return out


def reconstruct_heat_queries() -> np.ndarray:
    rng = np.random.default_rng(11)
    qp = 1.0 + 0.25 * (2.0 * rng.random((100, 16)) - 1.0)
    return np.clip(qp, 0.05, None)


def make_component_row(benchmark: str, q: int, k: int, xk: float,
                       h2: float, h: float, h_2: float,
                       g_h2: float, g_h: float, g_2h: float, g_ind: float,
                       query_gradient_scale: float,
                       grid80: float = math.nan, grid160: float = math.nan,
                       grid320: float = math.nan,
                       grid640: float = math.nan) -> dict:
    # Second-order central-FD truncation estimate from the nested h/2,h pair.
    trunc = abs(g_h2 - g_h) / 3.0
    # Conservative unresolved rule: a small scale floor plus 10x the local
    # truncation estimate.  For heat, grid-sign inconsistency is handled below.
    tol = max(1e-12, 1e-6 * query_gradient_scale, 10.0 * trunc)
    grid_consistent = True
    if np.isfinite(grid80):
        # Treat twice the last refinement change as a conservative grid-error
        # contribution to the near-zero threshold.
        tol = max(tol, 2.0 * abs(grid640 - grid320))
        s80 = sign_with_tol(grid80, tol)
        s160 = sign_with_tol(grid160, tol)
        s320 = sign_with_tol(grid320, tol)
        s640 = sign_with_tol(grid640, tol)
        # N=160 is the released oracle.  Require sign agreement with two finer
        # grids; N=80 is retained as a deliberately coarse diagnostic, but a
        # coarse-only flip does not invalidate a converged 160/320/640 label.
        grid_consistent = (s160 != 0 and s160 == s320 == s640)
    sind = sign_with_tol(g_ind, tol)
    unresolved = (sind == 0) or (not grid_consistent)
    if unresolved:
        sind = 0
    return {
        "benchmark": benchmark,
        "query": q,
        "component": k,
        "x_k": xk,
        "step_h_over_2": h2,
        "step_h": h,
        "step_2h": h_2,
        "g_h_over_2": g_h2,
        "g_h": g_h,
        "g_2h": g_2h,
        "g_independent": g_ind,
        "g_grid80_independent": grid80,
        "g_grid160_independent": grid160,
        "g_grid320_independent": grid320,
        "g_grid640_independent": grid640,
        "zero_tolerance": tol,
        "unresolved": int(unresolved),
        "sign_h_over_2": sign_with_tol(g_h2, tol),
        "sign_h": sign_with_tol(g_h, tol),
        "sign_2h": sign_with_tol(g_2h, tol),
        "sign_independent": sind,
        "retain_h2_vs_h": int(sign_with_tol(g_h2, tol) == sign_with_tol(g_h, tol) != 0),
        "retain_2h_vs_h": int(sign_with_tol(g_2h, tol) == sign_with_tol(g_h, tol) != 0),
        "retain_h_vs_independent": int(sign_with_tol(g_h, tol) == sind != 0),
        "grid_sign_consistent": int(grid_consistent),
        "relative_change_h2_h": safe_rel(g_h2, g_h, tol),
        "relative_change_2h_h": safe_rel(g_2h, g_h, tol),
        "relative_error_h_independent": safe_rel(g_h, g_ind, tol),
    }


def tmm_rows(tag: str) -> list[dict]:
    qp, lam = reconstruct_tmm_queries(tag)
    rows: list[dict] = []
    h = 0.5  # nm, exactly as in the released generator
    for q, x in enumerate(qp):
        _, exact = tmm_r_and_grad(x, lam)
        gh2 = tmm_fd(x, lam, h / 2.0)
        gh = tmm_fd(x, lam, h)
        g2h = tmm_fd(x, lam, 2.0 * h)
        for k in range(x.size):
            rows.append(make_component_row(
                tag, q, k, float(x[k]), h / 2.0, h, 2.0 * h,
                float(gh2[k]), float(gh[k]), float(g2h[k]), float(exact[k]),
                float(np.max(np.abs(exact)))
            ))
    return rows


def heat_rows() -> list[dict]:
    qp = reconstruct_heat_queries()
    objectives = {n: heat_problem(n) for n in (80, 160, 320, 640)}
    rows: list[dict] = []
    rel = 1e-3  # exactly as in the released generator
    for q, x in enumerate(qp):
        gh2 = heat_fd(objectives[160], x, rel / 2.0)
        gh = heat_fd(objectives[160], x, rel)
        g2h = heat_fd(objectives[160], x, 2.0 * rel)
        cs = {n: heat_complex_step(objectives[n], x) for n in (80, 160, 320, 640)}
        for k in range(x.size):
            hk = rel * max(abs(x[k]), 1e-2)
            rows.append(make_component_row(
                "heat_Poisson", q, k, float(x[k]), hk / 2.0, hk, 2.0 * hk,
                float(gh2[k]), float(gh[k]), float(g2h[k]), float(cs[160][k]),
                float(np.max(np.abs(cs[160]))),
                float(cs[80][k]), float(cs[160][k]), float(cs[320][k]),
                float(cs[640][k])
            ))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if a.size else float("nan")


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for benchmark in ("TMM_easy", "TMM_stressed", "heat_Poisson"):
        rr = [r for r in rows if r["benchmark"] == benchmark]
        resolved = [r for r in rr if not r["unresolved"]]
        def arr(key, selected=resolved):
            return np.asarray([float(r[key]) for r in selected], dtype=float)
        independent = "analytic characteristic-matrix derivative" if benchmark.startswith("TMM") \
            else "complex-step derivative through discrete FV solve"
        if benchmark == "heat_Poisson":
            grid80 = arr("g_grid80_independent", rr)
            grid160 = arr("g_grid160_independent", rr)
            grid320 = arr("g_grid320_independent", rr)
            grid640 = arr("g_grid640_independent", rr)
            grid_tol = arr("zero_tolerance", rr)
            def grid_sign(v, tol):
                return np.where(np.abs(v) <= tol, 0, np.where(v > 0, 1, -1))
            s80 = grid_sign(grid80, grid_tol)
            s160 = grid_sign(grid160, grid_tol)
            s320 = grid_sign(grid320, grid_tol)
            s640 = grid_sign(grid640, grid_tol)
            rel80_160 = np.abs(grid80 - grid160) / np.maximum.reduce(
                [np.abs(grid80), np.abs(grid160), grid_tol])
            rel160_320 = np.abs(grid160 - grid320) / np.maximum.reduce(
                [np.abs(grid160), np.abs(grid320), grid_tol])
            rel320_640 = np.abs(grid320 - grid640) / np.maximum.reduce(
                [np.abs(grid320), np.abs(grid640), grid_tol])
            grid_stats = {
                "grid_sign_retention_80_160": float(np.mean((s80 == s160) & (s160 != 0))),
                "grid_sign_retention_160_320": float(np.mean((s160 == s320) & (s320 != 0))),
                "grid_sign_retention_320_640": float(np.mean((s320 == s640) & (s640 != 0))),
                "p95_relative_grid_change_160_320": percentile(rel160_320, 95),
                "max_relative_grid_change_160_320": percentile(rel160_320, 100),
                "p95_relative_grid_change_320_640": percentile(rel320_640, 95),
                "max_relative_grid_change_320_640": percentile(rel320_640, 100),
            }
        else:
            grid_stats = {
                "grid_sign_retention_80_160": float("nan"),
                "grid_sign_retention_160_320": float("nan"),
                "grid_sign_retention_320_640": float("nan"),
                "p95_relative_grid_change_160_320": float("nan"),
                "max_relative_grid_change_160_320": float("nan"),
                "p95_relative_grid_change_320_640": float("nan"),
                "max_relative_grid_change_320_640": float("nan"),
            }
        summary = {
            "benchmark": benchmark,
            "queries": len({int(r["query"]) for r in rr}),
            "components_per_query": max(int(r["component"]) for r in rr) + 1,
            "total_components": len(rr),
            "independent_reference": independent,
            "baseline_step": "0.5 nm absolute" if benchmark.startswith("TMM") else "1e-3 relative",
            "zero_rule": "max(1e-12,1e-6*query_max_gradient,10*|g_h/2-g_h|/3); heat adds 2*|g_640-g_320| and requires 160/320/640 sign consistency",
            "unresolved_components": sum(int(r["unresolved"]) for r in rr),
            "resolved_components": len(resolved),
            "sign_retention_h2_vs_h": arr("retain_h2_vs_h").mean() if resolved else float("nan"),
            "sign_retention_2h_vs_h": arr("retain_2h_vs_h").mean() if resolved else float("nan"),
            "sign_retention_h_vs_independent": arr("retain_h_vs_independent").mean() if resolved else float("nan"),
            "grid_sign_consistency": arr("grid_sign_consistent", rr).mean(),
            "median_relative_change_h2_h": percentile(arr("relative_change_h2_h"), 50),
            "p95_relative_change_h2_h": percentile(arr("relative_change_h2_h"), 95),
            "max_relative_change_h2_h": percentile(arr("relative_change_h2_h"), 100),
            "median_relative_error_h_independent": percentile(arr("relative_error_h_independent"), 50),
            "p95_relative_error_h_independent": percentile(arr("relative_error_h_independent"), 95),
            "max_relative_error_h_independent": percentile(arr("relative_error_h_independent"), 100),
        }
        summary.update(grid_stats)
        out.append(summary)
    return out


def label_metrics(rows: list[dict], data_dir: Path) -> list[dict]:
    mapping = {
        "TMM_easy": data_dir / "external_members_tmm_easy.npz",
        "TMM_stressed": data_dir / "external_members_tmm_stressed.npz",
        "heat_Poisson": data_dir / "external_members_heat_poisson_stressed.npz",
    }
    out = []
    for benchmark, path in mapping.items():
        rr = [r for r in rows if r["benchmark"] == benchmark]
        resolved = np.asarray([not bool(r["unresolved"]) for r in rr])
        ref_sign_stable = np.asarray([
            (int(r["sign_h"]) == int(r["sign_independent"]) != 0) for r in rr
        ])
        data = np.load(path)
        members = np.asarray(data["member_gradients"], dtype=float)
        reference = np.asarray(data["reference_gradients"], dtype=float)
        mean_gradient = members.mean(axis=1)
        mean_sign = np.sign(mean_gradient)
        score = (np.sign(members) == mean_sign[:, None, :]).mean(axis=1)
        score[mean_sign == 0] = 0.0
        correct = (mean_sign == np.sign(reference)).astype(int)
        reconstructed_sign = np.asarray([int(r["sign_h"]) for r in rr]).reshape(reference.shape)
        if not np.array_equal(np.sign(reference).astype(int), reconstructed_sign):
            raise RuntimeError(f"{benchmark}: archived and reconstructed nominal reference signs differ")
        score = score.reshape(-1)
        correct = correct.reshape(-1)
        if score.size != len(rr):
            raise RuntimeError(f"{benchmark}: frozen array size {score.size} != reconstructed {len(rr)}")
        keep = resolved & ref_sign_stable
        y = correct[keep]
        s = score[keep]
        accepted = s >= 0.9
        risk = float((1 - y[accepted]).mean()) if accepted.any() else float("nan")
        out.append({
            "benchmark": benchmark,
            "frozen_components": int(score.size),
            "retained_after_reference_verification": int(keep.sum()),
            "removed_as_unresolved_or_sign_unstable": int((~keep).sum()),
            "operational_score": "mean_aligned_sign",
            "auc_before_resolution_screen": auc_rank(correct, score),
            "auc_after_resolution_screen": auc_rank(y, s),
            "verified_tau_0p9_coverage": float(accepted.mean()),
            "verified_tau_0p9_false_trust_risk": risk,
            "labels_unchanged": int(keep.all()),
        })
    return out


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = tmm_rows("TMM_easy") + tmm_rows("TMM_stressed") + heat_rows()
    summaries = summarize(rows)
    metrics = label_metrics(rows, data_dir)
    write_csv(out_dir / "reference_gradient_verification_components.csv", rows)
    write_csv(out_dir / "reference_gradient_verification_summary.csv", summaries)
    write_csv(out_dir / "reference_gradient_label_metrics.csv", metrics)
    for row in summaries:
        print(row)
    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
