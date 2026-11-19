"""Generate the platform's example reference files.

Produces, under webapp/examples/:

* example_line_measurements.csv — measurements for three named line
  segments ("Feeder A", "Feeder B", "Feeder C"), generated with the
  VERIFIED physics engine (src/simulate.py: prefault_state /
  solve_fault / sample_fault_row / sample_nofault_row). Healthy rows
  and AG / BC / BCG faults at several positions per segment, with
  ground truth (Loc, G, C, B, A) and a `segment` column that maps
  rows to the segments in example_network.json.

* example_network.json — a matching network the Studio can import.

Usage:
    cd webapp && ../.venv/bin/python examples/make_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from simulate import (
    SIGNATURES,
    prefault_state,
    sample_fault_row,
    sample_nofault_row,
    solve_fault,
)

OUT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# SEGMENT DEFINITIONS (name -> length km); labels match the network
# ------------------------------------------------------------------

SEGMENTS = [
    ("Feeder A", 60.0),
    ("Feeder B", 90.0),
    ("Feeder C", 40.0),
]

LOAD_LEVEL = 0.75          # of thermal rating (mid-range, in-distribution)
SWEEP_LENGTH = 8           # locations per fault sweep (matches the
                           # training event structure the LSTM expects)

FAULT_SIGNATURES = ["AG", "BCG"]

R_F = 25.0                    # ohm fault resistance
THETA_INC = np.deg2rad(45.0)  # inception angle


def main() -> None:
    state = prefault_state(LOAD_LEVEL)

    records: list[dict] = []

    # healthy traffic per segment (steady-state samples)
    for name, _length in SEGMENTS:
        for j in range(8):
            row = sample_nofault_row(state, j)
            row["Loc"] = 0
            row["G"] = row["C"] = row["B"] = row["A"] = 0
            row["segment"] = name
            records.append(row)

    # Fault scenarios per segment: each signature becomes ONE sweep of
    # locations ascending along the segment — the same event structure
    # the LSTM was trained on (Loc strictly ascending, then resets).
    # Positions are scaled to each segment's own length so the physics
    # stays inside the trained operating envelope.
    for name, length in SEGMENTS:
        positions = np.linspace(
            0.15 * length, length, SWEEP_LENGTH
        )
        for sig in FAULT_SIGNATURES:
            _, _, encoding = SIGNATURES[sig]
            for pos_km in positions:
                fault = solve_fault(sig, R_F, float(pos_km), LOAD_LEVEL, state)
                row = sample_fault_row(state, fault, R_F, THETA_INC, k=0)
                row["Loc"] = round(float(pos_km), 3)
                row["G"], row["C"], row["B"], row["A"] = encoding
                row["segment"] = name
                records.append(row)

    columns = [
        "segment", "Loc", "G", "C", "B", "A",
        "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
    ]
    df = pd.DataFrame(records, columns=columns)
    csv_path = OUT_DIR / "example_line_measurements.csv"
    df.to_csv(csv_path, index=False)

    print(f"Rows: {len(df)}  ({df['segment'].value_counts().to_dict()})")
    print(f"Fault rows: {int((df[[*'GCBA']].sum(axis=1) > 0).sum())}")
    print(f"Wrote {csv_path}")

    print("Example network JSON lives next to it: example_network.json")


if __name__ == "__main__":
    main()
