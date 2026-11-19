"""
Scenario identification and group-aware partitioning (R9).

Unit of analysis (Section 3.8.2): the simulation run. In
classdetectfault_sim.csv each fault run is recorded as one
consecutive sweep over the eight line locations
(Loc = 15, 30, ..., 120 km); a new run starts at every Loc == 15
marker. Verified structure (2026-11-12, regenerated data):

  - 5 fault-signature blocks (C+B, B+A, G+A, C+B+A, G+C+B+A),
    3,000 fault rows in total, arranged as 375 complete 8-row
    sweeps (no truncated sweeps — the full factorial sweeps are
    laid out contiguously);
  - the 2,365 no-fault rows form ONE continuous steady-state
    recording (adjacent-row lag-1 autocorrelation = 0.951 on the
    voltages, matching the 1 kHz / 50 Hz sampling grid), which is
    segmented into 296 non-overlapping observation windows of 8
    rows (the last window has 5 rows) — the same window size as
    one fault sweep — so that both classes share the same unit of
    analysis.

Partitioning (deterministic and shared by every training script,
so all models are evaluated on the same held-out rows):

  - Fault scenarios: stratified group selection — within each
    fault signature a seeded permutation draws a test_size
    fraction of runs for test (Section 3.12.1), so no run spans
    train/test and class proportions are preserved.
  - No-fault windows: contiguous temporal blocking — the final
    test_size fraction of the recording goes to the test
    partition — because the recording is one continuous signal
    and random scattering of its windows would place adjacent,
    highly correlated samples on opposite sides of the split.
    Residual autocorrelation across the train/test boundary of a
    continuous recording is acknowledged as a limitation.

Scenario ids are stable: they depend only on the row order of
classdetectfault_sim.csv, not on any model or split.
"""

from __future__ import annotations

import numpy as np

TEST_SIZE = 0.20
RANDOM_STATE = 42

NOFAULT_WINDOW = 8

NOFAULT_SIGNATURE = "0000"


def fault_signature(df) -> np.ndarray:

    """String signature 'GCBA' of each row (e.g. '1001' = G+A)."""

    return (
        df["G"].astype(int).astype(str)
        + df["C"].astype(int).astype(str)
        + df["B"].astype(int).astype(str)
        + df["A"].astype(int).astype(str)
    ).to_numpy()


def assign_scenario_ids(df) -> np.ndarray:

    """
    Per-row scenario id.

    A new scenario starts when:

      - a fault row begins a new location sweep (Loc == 15), or
        follows a no-fault row (block change / safety), or
      - a no-fault row starts a new observation window (every
        NOFAULT_WINDOW consecutive no-fault rows).
    """

    sig = fault_signature(df)

    loc = df["Loc"].to_numpy()

    is_fault = sig != NOFAULT_SIGNATURE

    n = len(df)

    ids = np.empty(n, dtype=int)

    current = -1

    window_count = 0

    prev_fault = False

    for i in range(n):

        if is_fault[i]:

            if not prev_fault or loc[i] == 15:

                current += 1

            window_count = 0

            prev_fault = True

        else:

            if prev_fault or window_count == NOFAULT_WINDOW:

                current += 1

                window_count = 1

            else:

                window_count += 1

            prev_fault = False

        ids[i] = current

    return ids


def scenario_summary(df) -> dict:

    """Counts of scenarios, rows, and signatures per scenario."""

    ids = assign_scenario_ids(df)

    sig = fault_signature(df)

    fault_mask = sig != NOFAULT_SIGNATURE

    fault_ids = ids[fault_mask]

    nofault_ids = ids[~fault_mask]

    n_fault_scenarios = len(np.unique(fault_ids))

    n_nofault_windows = len(np.unique(nofault_ids))

    sizes = []

    for s in np.unique(ids):

        sizes.append(int((ids == s).sum()))

    return {
        "n_rows": int(n := len(df)),
        "n_scenarios": int(len(np.unique(ids))),
        "n_fault_scenarios": int(n_fault_scenarios),
        "n_nofault_windows": int(n_nofault_windows),
        "n_fault_rows": int(fault_mask.sum()),
        "n_nofault_rows": int((~fault_mask).sum()),
        "scenario_sizes_min": int(min(sizes)),
        "scenario_sizes_max": int(max(sizes)),
    }


def partition_scenario_ids(
    df,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, dict]:

    """
    Partition scenario ids into train and test.

    Returns (train_ids, test_ids, info). Fault scenarios are
    split with stratified group selection; no-fault windows are
    blocked contiguously (final fraction of the recording -> test).
    """

    ids = assign_scenario_ids(df)

    sig = fault_signature(df)

    fault_mask = sig != NOFAULT_SIGNATURE

    fault_row_positions = np.where(fault_mask)[0]

    fault_scenario_ids = np.unique(ids[fault_row_positions])

    nofault_scenario_ids = np.unique(ids[~fault_mask])

    # ---- fault scenarios: stratified group selection -----------
    # Stratified at scenario level (Section 3.12.1): within each
    # fault signature, a test_size fraction of runs is drawn with
    # a seeded permutation, so class proportions are preserved
    # while no run spans the split.

    rng = np.random.default_rng(random_state)

    train_ids: set[int] = set()

    test_ids: set[int] = set()

    for signature in np.unique(sig[fault_mask]):

        signature_rows = fault_row_positions[
            sig[fault_row_positions] == signature
        ]

        signature_scenarios = np.unique(ids[signature_rows])

        n_test = int(
            round(test_size * len(signature_scenarios))
        )

        chosen = rng.permutation(signature_scenarios)[:n_test]

        test_ids |= set(int(s) for s in chosen)

    train_ids |= (
        set(int(s) for s in fault_scenario_ids) - test_ids
    )

    # ---- no-fault windows: contiguous temporal blocking --------

    n_test_windows = int(
        round(test_size * len(nofault_scenario_ids))
    )

    nofault_test_ids = set(
        nofault_scenario_ids[-n_test_windows:]
    )

    nofault_train_ids = set(
        nofault_scenario_ids[:-n_test_windows]
    )

    train_ids |= nofault_train_ids

    test_ids |= nofault_test_ids

    # ---- composition info ---------------------------------------

    train_mask = np.isin(ids, list(train_ids))

    test_mask = np.isin(ids, list(test_ids))

    spanning = set(train_ids) & set(test_ids)

    def _sig_counts(mask):

        values, counts = np.unique(sig[mask], return_counts=True)

        return {
            v: int(c) for v, c in zip(values, counts)
        }

    info = {
        "n_scenarios": int(len(np.unique(ids))),
        "n_fault_scenarios": int(len(fault_scenario_ids)),
        "n_nofault_windows": int(len(nofault_scenario_ids)),
        "train_scenarios": len(train_ids),
        "test_scenarios": len(test_ids),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_fault_rows": int((train_mask & fault_mask).sum()),
        "test_fault_rows": int((test_mask & fault_mask).sum()),
        "train_nofault_rows": int(
            (train_mask & ~fault_mask).sum()
        ),
        "test_nofault_rows": int(
            (test_mask & ~fault_mask).sum()
        ),
        "train_signature_counts": _sig_counts(train_mask),
        "test_signature_counts": _sig_counts(test_mask),
        "spanning_scenarios": len(spanning),
        "nofault_test_windows": n_test_windows,
        "nofault_blocking": "contiguous tail of the recording",
    }

    return (
        np.where(train_mask)[0],
        np.where(test_mask)[0],
        info,
    )


def group_partition(
    df,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, dict]:

    """
    Row-position partition for the training scripts: returns
    (train_positions, test_positions, info) suitable for
    X.iloc[train_pos] / y.iloc[train_pos] indexing.
    """

    train_pos, test_pos, info = partition_scenario_ids(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    return train_pos, test_pos, info


def log_split_info(logger, info: dict) -> None:

    """Print the split composition block required by R9."""

    logger.print("")
    logger.print(
        "SCENARIO-LEVEL SPLIT (Section 3.8.2, R9):"
    )
    logger.print(
        f"  scenarios total          : "
        f"{info['n_scenarios']} "
        f"({info['n_fault_scenarios']} fault runs, "
        f"{info['n_nofault_windows']} no-fault windows)"
    )
    logger.print(
        f"  train scenarios          : "
        f"{info['train_scenarios']} "
        f"({info['train_rows']} rows)"
    )
    logger.print(
        f"  test scenarios           : "
        f"{info['test_scenarios']} "
        f"({info['test_rows']} rows)"
    )
    logger.print(
        f"  train fault/no-fault rows: "
        f"{info['train_fault_rows']} / "
        f"{info['train_nofault_rows']}"
    )
    logger.print(
        f"  test fault/no-fault rows : "
        f"{info['test_fault_rows']} / "
        f"{info['test_nofault_rows']}"
    )
    logger.print(
        f"  scenarios spanning both  : "
        f"{info['spanning_scenarios']} "
        f"(0 by construction)"
    )
    logger.print(
        "  no-fault windows are blocked contiguously "
        "(tail of the recording -> test); fault runs are "
        "partitioned with stratified group selection."
    )
