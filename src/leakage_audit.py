"""
R9 leakage audit: quantify scenario spanning under the historical
row-level train/test splits, and document the new shared
scenario-level split.

Unit of analysis (Section 3.8.2): the simulation run. Scenario ids
are assigned by src/scenarios.py. For each historical predictions
file, the held-out rows are the row_ids present in the file; a
scenario SPANS the split when some of its rows are held out and
others are not — those runs contributed (near-)duplicate
measurements to both sides of the split.

Outputs (outputs/leakage_audit/):

    report_<timestamp>.txt
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from scenarios import (
    assign_scenario_ids,
    partition_scenario_ids,
    scenario_summary,
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUTS_DIR / "leakage_audit"

DATA_FILE = DATA_DIR / "classdetectfault_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = OUTPUT_DIR / f"report_{timestamp}.txt"


# ============================================================
# REPORT LOGGER
# ============================================================

class ReportLogger:

    def __init__(self, path: Path):

        self.path = path
        self.file = path.open("w", encoding="utf-8")

    def print(self, message=""):

        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):

        self.file.close()


def main():

    logger = ReportLogger(REPORT_FILE)

    try:

        logger.print("=" * 70)
        logger.print("R9 LEAKAGE AUDIT — SCENARIO-LEVEL SPLIT")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")
        logger.print(f"Dataset: {DATA_FILE}")
        logger.print("")

        df = pd.read_csv(DATA_FILE)

        ids = assign_scenario_ids(df)

        summary = scenario_summary(df)

        # ----------------------------------------------------
        # SCENARIO STRUCTURE
        # ----------------------------------------------------

        logger.print("-" * 70)
        logger.print("SCENARIO STRUCTURE (Section 3.8.2)")
        logger.print("-" * 70)

        for key, value in summary.items():
            logger.print(f"  {key}: {value}")

        logger.print("")
        logger.print(
            "  Fault runs are the 8-row location sweeps "
            "(Loc = 15..120 km); the no-fault rows form one "
            "continuous steady-state recording, segmented into "
            "8-row observation windows."
        )
        logger.print("")

        # ----------------------------------------------------
        # HISTORICAL SPLIT LEAKAGE
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print(
            "HISTORICAL ROW-LEVEL SPLITS — SCENARIO SPANNING"
        )
        logger.print("=" * 70)

        files = sorted(OUTPUTS_DIR.glob("*/predictions_*.csv"))

        rows = []

        for path in files:

            model = path.parent.name

            pred = pd.read_csv(path)

            test_row_ids = pred["row_id"].to_numpy()

            in_test = np.zeros(len(df), dtype=bool)

            in_test[test_row_ids] = True

            spanning = 0
            test_scenarios = 0

            for s in np.unique(ids):

                scenario_rows = ids == s

                n_test = int(in_test[scenario_rows].sum())

                if n_test == 0:
                    continue

                test_scenarios += 1

                n_total = int(scenario_rows.sum())

                if n_test < n_total:
                    spanning += 1

            rows.append({
                "model": model,
                "test_rows": len(test_row_ids),
                "scenarios_touched": test_scenarios,
                "scenarios_total": int(len(np.unique(ids))),
                "scenarios_spanning": spanning,
            })

            logger.print(
                f"  {model:>6}: test rows {len(test_row_ids):>5}, "
                f"scenarios touched {test_scenarios:>4} / "
                f"{len(np.unique(ids))}, spanning "
                f"{spanning:>4} "
                f"({spanning / max(test_scenarios, 1):.1%} of "
                f"touched scenarios leak rows into training)"
            )

        logger.print("")
        logger.print(
            "  A spanning scenario has rows in BOTH partitions: "
            "the model trains on 7 locations of a run and is "
            "tested on the 8th (and, for the continuous no-fault "
            "recording, on adjacent time samples of the same "
            "signal). This is the leakage mechanism challenged in "
            "R9."
        )
        logger.print("")

        # ----------------------------------------------------
        # NEW SHARED SCENARIO-LEVEL SPLIT
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print("NEW SHARED SCENARIO-LEVEL SPLIT")
        logger.print("=" * 70)

        train_pos, test_pos, info = partition_scenario_ids(df)

        for key, value in info.items():
            logger.print(f"  {key}: {value}")

        logger.print("")
        logger.print(
            "  Every training script uses this same deterministic "
            "split, so all models are evaluated on the same "
            f"{info['test_rows']} held-out rows and no scenario "
            "spans the boundary."
        )

        logger.print("")

    finally:

        logger.close()


if __name__ == "__main__":

    main()
