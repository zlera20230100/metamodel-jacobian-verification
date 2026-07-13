"""Run the missing openEMS finite-difference step ladder without editing source data.

The released antenna audit contains all six zones only at h=0.05.  This script
imports the original full-wave model from ``论文/_A2_grad_fullwave.py`` and runs
the same objective at h=0.025 and h=0.10.  Results are checkpointed after every
zone pair so an interrupted multi-hour run does not lose completed simulations.

This is intentionally a step-size study only.  It does not claim mesh
refinement; the accompanying audit records that limitation explicitly.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parents[1]
ORIGINAL = PAPER_ROOT / "_A2_grad_fullwave.py"
FROZEN = (
    HERE / "grad_fullwave.npz"
    if (HERE / "grad_fullwave.npz").is_file()
    else PAPER_ROOT / "repo_paper2" / "grad_fullwave.npz"
)
RAW_CSV = Path(os.environ.get("OPENEMS_RAW_CSV", HERE / "openems_step_ladder_raw.csv"))


def load_original_module():
    spec = importlib.util.spec_from_file_location("a2_grad_fullwave_source", ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ORIGINAL}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rows() -> list[dict]:
    if not RAW_CSV.exists():
        return []
    with RAW_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(rows: list[dict]) -> None:
    fields = ["step", "zone", "q_plus", "q_minus", "fd_grad",
              "s11_plus", "s11_minus", "source"]
    with RAW_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="0.025,0.10")
    ap.add_argument("--zones", default="0,1,2,3,4,5")
    args = ap.parse_args()
    steps = [float(x) for x in args.steps.split(",")]
    zones = [int(x) for x in args.zones.split(",")]

    rows = load_rows()
    completed = {(round(float(r["step"]), 12), int(r["zone"])) for r in rows}
    mod = load_original_module()

    # Seed the table with the released h=0.05 data, which are not rerun here.
    frozen = np.load(FROZEN)
    for i, zone in enumerate(np.asarray(frozen["zones"], dtype=int)):
        key = (0.05, int(zone))
        if key not in completed:
            rows.append({
                "step": 0.05,
                "zone": int(zone),
                "q_plus": float(frozen["Q_plus"][i]),
                "q_minus": float(frozen["Q_minus"][i]),
                "fd_grad": float(frozen["fd_grad"][i]),
                "s11_plus": "",
                "s11_minus": "",
                "source": "frozen repo_paper2/grad_fullwave.npz",
            })
            completed.add(key)
    save_rows(rows)

    for step in steps:
        for zone in zones:
            key = (round(step, 12), zone)
            if key in completed:
                print(f"skip completed h={step:g}, zone={zone}", flush=True)
                continue
            gp = np.ones(mod.K); gp[zone] = 1.0 + step
            gm = np.ones(mod.K); gm[zone] = 1.0 - step
            code = str(step).replace("0.", "").replace(".", "p")
            print(f"\n=== h={step:g}, zone {zone}, plus ===", flush=True)
            rp = mod.run_design(gp, f"h{code}_z{zone}p")
            print(f"\n=== h={step:g}, zone {zone}, minus ===", flush=True)
            rm = mod.run_design(gm, f"h{code}_z{zone}m")
            grad = (np.log(rp["Q_ap"]) - np.log(rm["Q_ap"])) / (2.0 * step)
            rows.append({
                "step": step,
                "zone": zone,
                "q_plus": rp["Q_ap"],
                "q_minus": rm["Q_ap"],
                "fd_grad": grad,
                "s11_plus": rp["s11_24"],
                "s11_minus": rm["s11_24"],
                "source": "new openEMS run; same model and mesh recipe",
            })
            completed.add(key)
            rows.sort(key=lambda r: (float(r["step"]), int(r["zone"])))
            save_rows(rows)
            print(f"checkpoint: h={step:g}, zone={zone}, grad={grad:+.6g}", flush=True)


if __name__ == "__main__":
    main()
