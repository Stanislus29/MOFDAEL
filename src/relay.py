"""
Conventional distance-relay detection logic for H1 baseline.

Implements the protection baseline described in corrected Chapter Three
§3.9 (Conventional Protection Baseline) and §3.9.4 (detection logic).

A distance relay measures apparent impedance from phase voltages and
currents, applies zero-sequence compensation for ground faults, and
trips when the compensated impedance falls within the Zone 1 reach
setting.

All parameters are taken from corrected Table 3.2 (Chapter Three
§3.3.1):

    X1 = 0.390 Ω/km   positive-sequence reactance
    X0 = 1.352 Ω/km   zero-sequence reactance
    Line length = 120 km
    Zone 1 reach = 80% of line impedance

Nominal system: 400 kV line-line (phase voltage = 400/sqrt(3) kV).

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

DATA_FILE = DATA_DIR / "classdetectfault_data.csv"

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

X1_PER_KM = 0.390  # positive-sequence reactance, Ω/km
X0_PER_KM = 1.352  # zero-sequence reactance, Ω/km
LINE_LENGTH_KM = 120.0
ZONE1_REACH_PCT = 0.80  # Zone 1 reach as % of line impedance

# Nominal phase voltage (V): 400 kV line-line / sqrt(3)
V_NOM_PHASE_V = 400_000.0 / np.sqrt(3.0)

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
    """Zone 1 reach in ohms (80% of positive-sequence line impedance)."""
    z1_line = X1_PER_KM * LINE_LENGTH_KM
    return ZONE1_REACH_PCT * z1_line


def compensation_factor() -> float:
    """Zero-sequence compensation factor k0 = (Z0 - Z1) / (3 * Z1)."""
    return (X0_PER_KM - X1_PER_KM) / (3.0 * X1_PER_KM)


def detect_relay(df: pd.DataFrame) -> pd.Series:
    """
    Distance-relay trip decision per row.

    Returns a boolean Series indexed like ``df``: True = relay trips
    (fault detected within Zone 1), False = no trip.
    """
    # Scale per-unit voltages to actual volts
    va_v = df["Va"] * V_NOM_PHASE_V
    vb_v = df["Vb"] * V_NOM_PHASE_V
    vc_v = df["Vc"] * V_NOM_PHASE_V

    # Zero-sequence compensation
    k0 = compensation_factor()
    sum_i = df["Ia"] + df["Ib"] + df["Ic"]

    # Compensated apparent impedance per phase (ohms)
    z_a = np.abs(va_v + k0 * sum_i) / (np.abs(df["Ia"]) + 1e-6)
    z_b = np.abs(vb_v + k0 * sum_i) / (np.abs(df["Ib"]) + 1e-6)
    z_c = np.abs(vc_v + k0 * sum_i) / (np.abs(df["Ic"]) + 1e-6)

    # Minimum impedance across phases = closest fault to relay
    z_min = np.minimum(z_a, np.minimum(z_b, z_c))

    # Zone 1 trip decision
    reach = zone1_reach_ohms()
    return z_min < reach


# ============================================================
# PREDICTION FILE GENERATION
# ============================================================


def generate_predictions_csv(
    data_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Generate predictions_relay.csv from the raw dataset.

    The relay makes a binary fault-detection decision per scenario.
    Per-target columns mirror the overall decision (a conventional
    relay detects fault presence; it does not classify fault type the
    way the learned models do — this is documented in §3.9).
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df.reset_index().rename(columns={"index": "row_id"})

    trip = detect_relay(df)

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
        logger.print(f"Positive-sequence reactance X1 : {X1_PER_KM:.3f} Ω/km")
        logger.print(f"Zero-sequence reactance     X0 : {X0_PER_KM:.3f} Ω/km")
        logger.print(f"Line length                 : {LINE_LENGTH_KM:.0f} km")
        logger.print(f"Zone 1 reach setting        : {ZONE1_REACH_PCT * 100:.0f}%")
        logger.print(
            f"Zone 1 reach (ohms)         : {zone1_reach_ohms():.2f} Ω"
        )
        logger.print(
            f"Zero-sequence compensation k0: {compensation_factor():.4f}"
        )
        logger.print(
            f"Nominal phase voltage       : {V_NOM_PHASE_V:.1f} V"
        )
        logger.print("")

        # ----------------------------------------------------
        # LOAD DATA & RUN DETECTION
        # ----------------------------------------------------

        df = pd.read_csv(DATA_FILE)
        df = df.reset_index().rename(columns={"index": "row_id"})

        logger.print(f"Dataset shape: {df.shape}")
        logger.print("")

        trip = detect_relay(df)
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
