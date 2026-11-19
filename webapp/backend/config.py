"""Central path & constant configuration for the MOFDAEL web platform.

Single source of truth for every directory the backend touches. The
project's dissertation artifacts (src/, data/, outputs/) are treated as
read-only inputs; everything the platform generates lives under
webapp/storage/.
"""

from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------
# PROJECT LAYOUT
# ------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent

WEBAPP_DIR = BACKEND_DIR.parent

PROJECT_ROOT = WEBAPP_DIR.parent

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ------------------------------------------------------------------
# PLATFORM STORAGE (all generated artifacts live here)
# ------------------------------------------------------------------

STORAGE_DIR = WEBAPP_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
RUNS_DIR = STORAGE_DIR / "runs"
SELFCHECK_DIR = STORAGE_DIR / "selfcheck"
DB_FILE = STORAGE_DIR / "platform.db"

for _dir in (STORAGE_DIR, DATASETS_DIR, RUNS_DIR, SELFCHECK_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# MODEL / DATA CONSTANTS (mirror the training scripts)
# ------------------------------------------------------------------

FEATURE_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
TARGET_COLUMNS = ["G", "C", "B", "A"]
LOCATION_COLUMN = "Loc"

LINE_LENGTH_KM = 120.0

# Loop impedance columns exported by src/simulate.py and consumed by
# the exact distance-relay decision.
RELAY_LOOP_COLUMNS = ["z_ga", "z_gb", "z_gc", "z_ab", "z_bc", "z_ca"]

# ------------------------------------------------------------------
# BENCHMARK DATASETS (used by the self-check / registry grounding)
# ------------------------------------------------------------------

BENCHMARK_DATA_FILE = DATA_DIR / "classdetectfault_sim.csv"
RELAY_MEASUREMENTS_FILE = OUTPUTS_DIR / "simulate" / "relay_measurements_sim.csv"

# ------------------------------------------------------------------
# TRAINING-SPLIT CONSTANTS (needed to reconstruct ANN/RNN scalers)
# ------------------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42


def ensure_src_on_path() -> None:
    """Make the verified training/physics modules importable.

    The platform reuses the training scripts' architecture classes,
    physics engine, and relay logic instead of re-deriving them
    (traceability rule, AGENTS.md section 5).
    """

    src = str(SRC_DIR)
    if src not in __import__("sys").path:
        __import__("sys").path.insert(0, src)
