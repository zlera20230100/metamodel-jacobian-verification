"""Evaluate the two fixed-step openEMS antenna variants reported in the manuscript.

Both variants were evaluated only at +/-5%.  This script checks whether their
two one-sided slopes relative to the archived baseline even have the same sign.
It does not convert them into verified gradients.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "openems_variant_one_sided_diagnostic.csv"


def row(case, zone, h, q0, qp, qm, central, mesh, source):
    plus = float(np.log(qp / q0) / h)
    minus = float(np.log(q0 / qm) / h)
    return {
        "case": case,
        "zone": zone,
        "step": h,
        "q_base": q0,
        "q_plus": qp,
        "q_minus": qm,
        "central_gradient": central,
        "one_sided_plus_slope": plus,
        "one_sided_minus_slope": minus,
        "one_sided_sign_consistent": int(np.sign(plus) == np.sign(minus) != 0),
        "mesh_cells": mesh,
        "step_ladder_available": 0,
        "mesh_refinement_available": 0,
        "status": "fixed-step observation; not reference-gradient evidence",
        "source": source,
    }


def main() -> None:
    rows = []
    p = ROOT / "_A2_grad_responsive.npz"
    d = np.load(p)
    h = float(d["g_pert"]); q0 = float(d["Q_base"])
    for i, z in enumerate(np.asarray(d["zones"], dtype=int)):
        rows.append(row("lower_Q_AIRGAP_2p5mm", int(z), h, q0,
                        float(d["Q_plus"][i]), float(d["Q_minus"][i]),
                        float(d["fd_grad"][i]), "189x189x71", p.name))

    # The tuned large-aperture NPZ omitted Q+/-; recover the printed values from
    # the archived source log. They have four-to-five significant digits,
    # sufficient for the one-sided sign diagnostic but not precision claims.
    log = (ROOT / "_gradR2t.log").read_text(encoding="utf-8", errors="replace")
    qvals = {}
    for tag, val in re.findall(r"\[(base|z[0-2][pm])\].*?Q_ap=([0-9.eE+-]+)", log):
        qvals[tag] = float(val)
    central = np.load(ROOT / "_A2_grad_responsive2_tuned.npz")["fd_grad"]
    for z in range(3):
        rows.append(row("large_near_resonant_5x5", z, 0.05, qvals["base"],
                        qvals[f"z{z}p"], qvals[f"z{z}m"], float(central[z]),
                        "205x203x78", "_gradR2t.log + _A2_grad_responsive2_tuned.npz"))

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
