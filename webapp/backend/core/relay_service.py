"""Relay decision service.

Two modes, kept strictly apart:

* ``exact`` — the dissertation baseline: trip decisions from the six
  loop-impedance measurements exported by src/simulate.py
  (relay_measurements_sim.csv). Bit-identical to predictions_relay.csv.

* ``emulated`` — for user-uploaded CSVs, which carry raw phase
  currents/voltages rather than solved loop impedances. Each row is
  passed through the same verified ``simulate.loop_impedances`` /
  ``relay.detect_relay`` logic, treating the row's values as the
  steady-state phasor measurement at the relay bus. Because a single
  instantaneous sample is not a full-cycle phasor estimate, this mode
  is documented as an APPROXIMATION of the benchmark relay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend import config
from backend.core import loaders


def relay_trip_from_measurements(measurements: pd.DataFrame) -> np.ndarray:
    """Exact mode: trip from exported loop-impedance columns."""

    loaders.ensure_src_path()
    from relay import detect_relay

    return detect_relay(measurements[config.RELAY_LOOP_COLUMNS]).to_numpy()


def relay_trip_from_features(df: pd.DataFrame) -> np.ndarray:
    """Emulated mode: trip decisions from raw phase measurements.

    ``simulate.loop_impedances`` consumes (v_abc, i_abc) phasor arrays
    shaped (n_rows, 3); it is the same verified code the physics
    engine uses, applied row-wise to the uploaded samples.
    """

    loaders.ensure_src_path()
    from relay import detect_relay
    from simulate import loop_impedances

    v_abc = df[["Va", "Vb", "Vc"]].to_numpy(dtype=float)
    i_abc = df[["Ia", "Ib", "Ic"]].to_numpy(dtype=float)

    rows = [
        loop_impedances(v_abc[i], i_abc[i])
        for i in range(len(df))
    ]
    loops = pd.DataFrame(
        rows,
        columns=config.RELAY_LOOP_COLUMNS,
        index=df.index,
    )

    return detect_relay(loops).to_numpy()
