"""
Conventional distance-relay detection logic for H1 baseline.

Implements the protection baseline described in corrected Chapter Three
§3.9 (Conventional Protection Baseline) and §3.9.4 (detection logic).

A distance relay measures apparent impedance from phase voltages and
currents, applies zero-sequence compensation for ground faults, and
trips when the compensated impedance falls within the Zone 1 reach
setting.

MEASUREMENT CHAIN: the relay decides on the six standard distance-
relay loop impedances (three residual-compensated ground loops and
three phase-phase loops) exported by src/simulate.py in
outputs/simulate/relay_measurements_sim.csv, aligned row-by-row with
the dataset. These are steady-state phasor quantities of the fault
state in effect at each exported sample — the measurement a numerical
relay forms once its full-cycle data window lies inside the post-
fault steady state. Raw instantaneous samples are deliberately NOT
used: |V(t)|/|I(t)| dips below the Zone 1 setting near every voltage
zero-crossing, which would make the decision an artifact of the
sampling phase (see docs/simulation_caveats.md item 4).

Ground loop (residual compensation, current-side):

    Z_g,p = |V_p| / |I_p + k0 * 3I0|,
    k0    = (Z0L - Z1L) / (3 * Z1L)          (complex, per km)

which recovers Z1 * d exactly for a bolted single-line-to-ground
fault fed from one end. (The earlier voltage-side compensation
|V + k0*3I0| / |I| over-measures by z_s/z1 ~ 1.8x with
z_s = (Z0 + 2 Z1)/3 — corrected 2026-11-12.)

All parameters are taken from corrected Table 3.2 (Chapter Three
§3.3.1):

    Z1 = 0.080 + j0.390 Ω/km   positive-sequence impedance
    Z0 = 0.228 + j1.352 Ω/km   zero-sequence impedance
    Line length = 120 km
    Zone 1 reach = 80% of |Z1| * line length = 38.22 Ω

Nominal system: 161 kV line-line (phase voltage = 161/sqrt(3) kV).

Outputs (written to outputs/relay/ following the train-script standard):

    predictions_relay.csv   per-scenario predictions
    report_<timestamp>.txt  accuracy metrics, parameters, discordance table
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION (matches train-script standard)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "relay"

DATA_FILE = DATA_DIR / "classdetectfault_sim.csv"

# Per-row loop-impedance measurements exported by src/simulate.py
MEASUREMENTS_FILE = (
    PROJECT_ROOT / "outputs" / "simulate" / "relay_measurements_sim.csv"
)

LOOP_COLUMNS = ["z_ga", "z_gb", "z_gc", "z_ab", "z_bc", "z_ca"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TIMESTAMP & OUTPUT FILES
# ============================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = OUTPUT_DIR / f"report_{timestamp}.txt"

PREDICTIONS_FILE = OUTPUT_DIR / "predictions_relay.csv"

# ============================================================
# LINE PARAMETERS (corrected Table 3.2)
# ============================================================

R1_PER_KM = 0.080   # positive-sequence resistance, Ω/km
X1_PER_KM = 0.390   # positive-sequence reactance, Ω/km
R0_PER_KM = 0.228   # zero-sequence resistance, Ω/km
X0_PER_KM = 1.352   # zero-sequence reactance, Ω/km

Z1_PER_KM = complex(R1_PER_KM, X1_PER_KM)
Z0_PER_KM = complex(R0_PER_KM, X0_PER_KM)

LINE_LENGTH_KM = 120.0
ZONE1_REACH_PCT = 0.80  # Zone 1 reach as % of line impedance

# Nominal phase voltage (V): 161 kV line-line / sqrt(3)
V_NOM_PHASE_V = 161_000.0 / np.sqrt(3.0)

TARGET_COLUMNS = ["G", "C", "B", "A"]


# ============================================================
# REPORT LOGGER (matches train-script ReportLogger)
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


# ============================================================
# DETECTION LOGIC
# ============================================================


def zone1_reach_ohms() -> float:
    """Zone 1 reach in ohms (80% of |Z1| positive-sequence line impedance)."""
    z1_line = abs(Z1_PER_KM) * LINE_LENGTH_KM
    return ZONE1_REACH_PCT * z1_line


def compensation_factor() -> complex:
    """Zero-sequence compensation factor k0 = (Z0 - Z1) / (3 * Z1)."""
    return (Z0_PER_KM - Z1_PER_KM) / (3.0 * Z1_PER_KM)


def detect_relay(meas: pd.DataFrame) -> pd.Series:
    """
    Distance-relay trip decision per row from the exported loop
    impedances: trip when the minimum loop impedance across the six
    loops (3 ground + 3 phase-phase) falls inside the Zone 1 reach.

    Returns a boolean Series indexed like ``meas``.
    """

    z_min = meas[LOOP_COLUMNS].min(axis=1)

    reach = zone1_reach_ohms()

    return z_min < reach


# ============================================================
# PREDICTION FILE GENERATION
# ============================================================


def generate_predictions_csv(
    data_path: str | Path,
    measurements_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Generate predictions_relay.csv from the raw dataset plus the
    exported loop-impedance measurements.
    """
    data_path = Path(data_path)
    measurements_path = Path(measurements_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df.reset_index().rename(columns={"index": "row_id"})

    meas = pd.read_csv(measurements_path)

    trip = detect_relay(meas)

    # Ground truth: fault occurred if any target is active
    fault_true = (df[TARGET_COLUMNS].sum(axis=1) > 0).astype(int)

    result = pd.DataFrame(
        {
            "row_id": df["row_id"],
            "overall_true": fault_true,
            "overall_pred": trip.astype(int),
        }
    )

    # Per-target columns: relay reports fault presence, not type
    for t in TARGET_COLUMNS:
        result[f"{t}_true"] = df[t]
        result[f"{t}_pred"] = trip.astype(int)

    result.to_csv(output_path, index=False)
    return result


# ============================================================
# MAIN
# ============================================================


def main():
    logger = ReportLogger(REPORT_FILE)

    try:

        logger.print("=" * 70)
        logger.print("CONVENTIONAL DISTANCE RELAY — DETECTION BASELINE (H1)")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")
        logger.print(f"Dataset: {DATA_FILE}")
        logger.print("")

        # ----------------------------------------------------
        # LINE PARAMETERS
        # ----------------------------------------------------

        logger.print("LINE PARAMETERS (corrected Table 3.2, §3.3.1)")
        logger.print("-" * 70)
        logger.print(
            f"Positive-sequence impedance Z1: {R1_PER_KM:.3f} + "
            f"j{X1_PER_KM:.3f} Ω/km (|Z1| = {abs(Z1_PER_KM):.4f} Ω/km)"
        )
        logger.print(
            f"Zero-sequence impedance  Z0 : {R0_PER_KM:.3f} + "
            f"j{X0_PER_KM:.3f} Ω/km"
        )
        logger.print(f"Line length                 : {LINE_LENGTH_KM:.0f} km")
        logger.print(f"Zone 1 reach setting        : {ZONE1_REACH_PCT * 100:.0f}%")
        logger.print(
            f"Zone 1 reach (ohms)         : {zone1_reach_ohms():.2f} Ω"
        )
        k0 = compensation_factor()
        logger.print(
            f"Zero-sequence compensation k0: {k0.real:.4f} + j{k0.imag:.4f}"
        )
        logger.print(
            "Measurement                 : |V_phase| / |I_phase + k0*3I0|"
        )
        logger.print(
            "Measurement                 : 6 loop impedances "
            "(3 ground + 3 phase-phase), steady-state phasors"
        )
        logger.print(f"Measurements file           : {MEASUREMENTS_FILE}")
        logger.print(
            f"Nominal phase voltage       : {V_NOM_PHASE_V:.1f} V"
        )
        logger.print("")

        # ----------------------------------------------------
        # LOAD DATA & RUN DETECTION
        # ----------------------------------------------------

        df = pd.read_csv(DATA_FILE)
        df = df.reset_index().rename(columns={"index": "row_id"})

        meas = pd.read_csv(MEASUREMENTS_FILE)

        if len(meas) != len(df):
            raise RuntimeError(
                "relay measurements file is not row-aligned with the "
                f"dataset ({len(meas)} measurements vs {len(df)} rows); "
                "re-run src/simulate.py"
            )

        logger.print(f"Dataset shape: {df.shape}")
        logger.print("")

        trip = detect_relay(meas)
        fault_true = (df[TARGET_COLUMNS].sum(axis=1) > 0).astype(int)

        # ----------------------------------------------------
        # ACCURACY METRICS
        # ----------------------------------------------------

        accuracy = (trip == fault_true).astype(int).mean()
        n_total = len(df)
        n_fault = int(fault_true.sum())
        n_no_fault = n_total - n_fault
        n_trip = int(trip.sum())

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(fault_true, trip)
        tn, fp, fn, tp = cm.ravel()

        no_fault_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fault_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # ----------------------------------------------------
        # PER-LINE ACCURACY (G, C, B, A targets)
        # ----------------------------------------------------

        logger.print("PER-LINE ACCURACY")
        logger.print("-" * 70)
        per_line_records = []
        for t in TARGET_COLUMNS:
            t_true = df[t].to_numpy()
            t_pred = trip.astype(int)
            t_acc = (t_true == t_pred).astype(int).mean()
            t_cm = confusion_matrix(t_true, t_pred)
            t_tn, t_fp, t_fn, t_tp = t_cm.ravel()
            t_no_fault = int((t_true == 0).sum())
            t_fault = int((t_true == 1).sum())
            t_tn_rate = t_tn / (t_tn + t_fp) if (t_tn + t_fp) > 0 else 0.0
            t_tp_rate = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0

            logger.print(
                f"  {t}: accuracy={t_acc:.6f}  "
                f"no-fault(TN)={t_tn_rate:.6f} ({t_tn} of {t_no_fault})  "
                f"fault(TP)={t_tp_rate:.6f} ({t_tp} of {t_fault})"
            )

            per_line_records.append({
                "target": t,
                "accuracy": t_acc,
                "tn": t_tn, "fp": t_fp, "fn": t_fn, "tp": t_tp,
                "tn_rate": t_tn_rate,
                "tp_rate": t_tp_rate,
                "n_no_fault": t_no_fault,
                "n_fault": t_fault,
            })

        logger.print("")

        logger.print("DETECTION RESULTS (overall)")
        logger.print("-" * 70)
        logger.print(f"Total scenarios        : {n_total}")
        logger.print(f"Fault scenarios        : {n_fault}")
        logger.print(f"No-fault scenarios     : {n_no_fault}")
        logger.print(f"Relay trips            : {n_trip}")
        logger.print(f"Overall accuracy       : {accuracy:.6f}")
        logger.print(f"No-fault accuracy (TN) : {no_fault_acc:.6f} ({tn} of {tn + fp})")
        logger.print(f"Fault accuracy (TP)    : {fault_acc:.6f} ({tp} of {tp + fn})")
        logger.print("")
        logger.print("Confusion matrix (rows=true, cols=pred):")
        logger.print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        logger.print("")

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        result = pd.DataFrame(
            {
                "row_id": df["row_id"],
                "overall_true": fault_true,
                "overall_pred": trip.astype(int),
            }
        )

        for t in TARGET_COLUMNS:
            result[f"{t}_true"] = df[t]
            result[f"{t}_pred"] = trip.astype(int)

        result.to_csv(PREDICTIONS_FILE, index=False)

        logger.print("OUTPUT FILES")
        logger.print("-" * 70)
        logger.print(f"Predictions: {PREDICTIONS_FILE}")
        logger.print(f"Report:      {REPORT_FILE}")
        logger.print("")
        logger.print(f"Predictions saved ({len(result)} rows) to {PREDICTIONS_FILE}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
