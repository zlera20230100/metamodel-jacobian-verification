"""Extract actual mesh dimensions printed by the openEMS step-ladder runs."""

from __future__ import annotations

import csv
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "openems_mesh_inventory.csv"
PATTERN = re.compile(
    r"\[(h(?P<code>025|1)_z(?P<zone>[0-5])(?P<direction>[pm]))\] "
    r"mesh (?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)=(?P<millions>[0-9.]+)M"
)


def main() -> None:
    rows = []
    for path in (HERE / "openems_step_ladder.log", HERE / "openems_step_ladder_h10.log"):
        if not path.exists():
            continue
        raw = path.read_bytes()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
        text = raw.decode(encoding, errors="replace")
        for m in PATTERN.finditer(text):
            nx, ny, nz = (int(m.group(x)) for x in ("nx", "ny", "nz"))
            rows.append({
                "log": path.name,
                "step": 0.025 if m.group("code") == "025" else 0.10,
                "zone": int(m.group("zone")),
                "direction": "plus" if m.group("direction") == "p" else "minus",
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "total_cells": nx * ny * nz,
            })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        fields = ["log", "step", "zone", "direction", "nx", "ny", "nz", "total_cells"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
