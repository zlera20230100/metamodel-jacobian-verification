"""Summarize archived/new openEMS finite-difference step checks.

The summary deliberately separates step-size evidence from mesh evidence.  A
successful h/2,h,2h sign check does not set ``mesh_verified`` to true.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parents[1]
RAW = HERE / "openems_step_ladder_raw.csv"
FROZEN = (
    HERE / "grad_fullwave.npz"
    if (HERE / "grad_fullwave.npz").is_file()
    else PAPER_ROOT / "repo_paper2" / "grad_fullwave.npz"
)
OUT = HERE / "openems_step_ladder_summary.csv"
REPEAT_OUT = HERE / "openems_step_ladder_h10_repeatability.csv"


def rel(a: float, b: float, floor: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def main() -> None:
    raw = []
    h10_repeat = []
    for path in (RAW, HERE / "openems_step_ladder_h10.csv"):
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows_here = list(csv.DictReader(f))
                raw.extend(rows_here)
                if path.name == "openems_step_ladder_h10.csv":
                    h10_repeat = rows_here
    lookup = {(round(float(r["step"]), 12), int(r["zone"])): r for r in raw}
    frozen = np.load(FROZEN)
    zones = np.asarray(frozen["zones"], dtype=int)
    q0 = float(frozen["Q_base"])
    qplus = np.asarray(frozen["Q_plus"], dtype=float)
    qminus = np.asarray(frozen["Q_minus"], dtype=float)

    rows = []
    for i, zone in enumerate(zones):
        def raw_at(step):
            r = lookup.get((round(step, 12), int(zone)))
            return r
        def grad(step):
            r = raw_at(step)
            return float(r["fd_grad"]) if r is not None else float("nan")
        def one_sided(step):
            r = raw_at(step)
            if r is None:
                return float("nan"), float("nan")
            qp, qm = float(r["q_plus"]), float(r["q_minus"])
            return ((np.log(qp) - np.log(q0)) / step,
                    (np.log(q0) - np.log(qm)) / step)
        g025, g05, g10 = grad(0.025), grad(0.05), grad(0.10)
        plus025, minus025 = one_sided(0.025)
        plus05, minus05 = one_sided(0.05)
        plus10, minus10 = one_sided(0.10)
        available = bool(np.isfinite(g025) and np.isfinite(g05) and np.isfinite(g10))
        if available:
            trunc = abs(g025 - g05) / 3.0
            scale = max(abs(g025), abs(g05), abs(g10))
            zero_tol = max(1e-3, 1e-2 * scale, 10.0 * trunc,
                           0.5 * abs(plus025 - minus025))
            s025 = 0 if abs(g025) <= zero_tol else int(np.sign(g025))
            s05 = 0 if abs(g05) <= zero_tol else int(np.sign(g05))
            s10 = 0 if abs(g10) <= zero_tol else int(np.sign(g10))
            retain025 = int(s025 == s05 != 0)
            retain10 = int(s10 == s05 != 0)
            r025 = rel(g025, g05, 1e-12)
            r10 = rel(g10, g05, 1e-12)
        else:
            zero_tol = float("nan"); s025 = s05 = s10 = 0
            retain025 = retain10 = 0; r025 = r10 = float("nan")
        rows.append({
            "zone": int(zone),
            "g_h_over_2_h0p025": g025,
            "g_h_h0p05": g05,
            "g_2h_h0p10": g10,
            "zero_tolerance_step_only": zero_tol,
            "raw_sign_h_over_2": int(np.sign(g025)) if np.isfinite(g025) else 0,
            "raw_sign_h": int(np.sign(g05)) if np.isfinite(g05) else 0,
            "raw_sign_2h": int(np.sign(g10)) if np.isfinite(g10) else 0,
            "raw_sign_retention_h2_vs_h": int(
                available and np.sign(g025) == np.sign(g05) != 0),
            "raw_sign_retention_2h_vs_h": int(
                available and np.sign(g10) == np.sign(g05) != 0),
            "sign_h_over_2": s025,
            "sign_h": s05,
            "sign_2h": s10,
            "sign_retention_h2_vs_h": retain025,
            "sign_retention_2h_vs_h": retain10,
            "relative_change_h2_h": r025,
            "relative_change_2h_h": r10,
            "one_sided_plus_slope_h0p025": plus025,
            "one_sided_minus_slope_h0p025": minus025,
            "one_sided_sign_consistent_h0p025": int(
                available and np.sign(plus025) == np.sign(minus025) != 0),
            "one_sided_plus_slope_h0p05": float(plus05),
            "one_sided_minus_slope_h0p05": float(minus05),
            "one_sided_sign_consistent_h0p05": int(np.sign(plus05) == np.sign(minus05) != 0),
            "one_sided_plus_slope_h0p10": plus10,
            "one_sided_minus_slope_h0p10": minus10,
            "one_sided_sign_consistent_h0p10": int(
                available and np.sign(plus10) == np.sign(minus10) != 0),
            "nominal_mesh_cells": "189x189x78",
            "nominal_mesh_total": 2786238,
            "mesh_verified": 0,
            "closure_status": (
                "step-sign stable under declared tolerance; mesh refinement absent"
                if available and retain025 and retain10 else
                "step ladder fails numerical-resolution screen; mesh refinement absent"
                if available else
                "step ladder incomplete; mesh refinement absent"
            ),
        })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # h=0.10 was executed twice.  The dedicated h10 file is the reported run
    # (and overrides the later checkpoint file in ``lookup`` above); the later
    # run is retained as a sign-repeatability check rather than silently
    # discarded.  This table is evidence of repeat execution, not a mesh study.
    raw_lookup = {}
    if RAW.exists():
        with RAW.open(newline="", encoding="utf-8") as f:
            raw_lookup = {
                (round(float(r["step"]), 12), int(r["zone"])): r
                for r in csv.DictReader(f)
            }
    repeat_rows = []
    for r in h10_repeat:
        if round(float(r["step"]), 12) != 0.10:
            continue
        zone = int(r["zone"])
        later = raw_lookup.get((0.10, zone))
        if later is None:
            continue
        g_reported = float(r["fd_grad"])
        g_repeat = float(later["fd_grad"])
        repeat_rows.append({
            "zone": zone,
            "reported_h0p10_gradient": g_reported,
            "repeat_h0p10_gradient": g_repeat,
            "absolute_difference": abs(g_reported - g_repeat),
            "relative_difference": abs(g_reported - g_repeat)
            / max(abs(g_reported), abs(g_repeat), 1e-12),
            "sign_match": int(np.sign(g_reported) == np.sign(g_repeat) != 0),
            "reported_source": "openems_step_ladder_h10.csv",
            "repeat_source": "openems_step_ladder_raw.csv",
            "mesh_refinement_available": 0,
        })
    if repeat_rows:
        with REPEAT_OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(repeat_rows[0]))
            w.writeheader(); w.writerows(repeat_rows)
    print(f"wrote {OUT}")
    if repeat_rows:
        print(f"wrote {REPEAT_OUT}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
