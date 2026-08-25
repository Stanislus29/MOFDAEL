from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT / MODEL
# ============================================================

# ------------------------------------------------------------
# Input CSV
# ------------------------------------------------------------

INPUT_FILE = DATA_DIR / "classdetectfault_data.csv"


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
# Picks the most recently trained model in MODEL_DIR so this
# never silently runs against a stale file. Set MODEL_FILE
# directly if you need a specific past run instead.

def find_latest_model():

    candidates = sorted(MODEL_DIR.glob("svm_model_*.joblib"))

    if not candidates:

        print(f"ERROR: No trained models found in:\n{MODEL_DIR}")
        sys.exit(1)

    return candidates[-1]


MODEL_FILE = find_latest_model()


# ============================================================
# DATA SELECTION
# ============================================================
#
# Available modes:
#
#   "full"
#       Process the entire CSV.
#
#   "rows"
#       Process specific CSV rows.
#
#   "range"
#       Process all locations between two values.
#
#   "random"
#       Randomly select N rows.
#
# ============================================================

SELECTION_MODE = "full"


# ------------------------------------------------------------
# ROW SELECTION
# ------------------------------------------------------------
#
# Used when:
#
#     SELECTION_MODE = "rows"
#
# IMPORTANT:
# These are ZERO-BASED dataframe indices.
#
# Example:
#
# [0]       -> first row
# [5]       -> sixth row
# [0, 5, 10]
#
# ------------------------------------------------------------

SELECTED_ROWS = [
    0,
    5,
    10,
]


# ------------------------------------------------------------
# LOCATION RANGE
# ------------------------------------------------------------
#
# Used when:
#
#     SELECTION_MODE = "range"
#
# Example:
#
# (20, 25)
#
# selects locations:
#
# 20 <= Loc <= 25
#
# ------------------------------------------------------------

LOCATION_RANGE = (
    0,
    25,
)


# ------------------------------------------------------------
# RANDOM SELECTION
# ------------------------------------------------------------
#
# Used when:
#
#     SELECTION_MODE = "random"
#
# ------------------------------------------------------------

RANDOM_SAMPLE_SIZE = 20

RANDOM_STATE = 42


# ============================================================
# TERMINAL OUTPUT
# ============================================================

# Number of rows displayed in terminal.

TERMINAL_PREVIEW_ROWS = 20


# ============================================================
# MODEL TARGETS
# ============================================================

TARGET_COLUMNS = [
    "G",
    "C",
    "B",
    "A",
]


# ============================================================
# FEATURES
# ============================================================

# IMPORTANT:
# Loc is deliberately NOT used as an ML feature.

FEATURE_COLUMNS = [
    "Ia",
    "Ib",
    "Ic",
    "Va",
    "Vb",
    "Vc",
]


# ============================================================
# TIMESTAMP / OUTPUT FILES
# ============================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

TXT_OUTPUT_FILE = (
    OUTPUT_DIR
    / f"inference_{timestamp}.txt"
)

CSV_OUTPUT_FILE = (
    OUTPUT_DIR
    / f"inference_{timestamp}.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_models():

    if not MODEL_FILE.exists():

        print(
            f"ERROR: Model not found:\n"
            f"{MODEL_FILE}"
        )

        sys.exit(1)

    print(
        f"Loading model:\n"
        f"  {MODEL_FILE}"
    )

    models = joblib.load(
        MODEL_FILE
    )

    missing_models = [
        target
        for target in TARGET_COLUMNS
        if target not in models
    ]

    if missing_models:

        print(
            "\nERROR: Saved model is missing:"
        )

        for target in missing_models:

            print(
                f"  {target}"
            )

        sys.exit(1)

    return models


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        print(
            f"ERROR: Input CSV not found:\n"
            f"{INPUT_FILE}"
        )

        sys.exit(1)

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = (
        ["Loc"]
        + FEATURE_COLUMNS
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print(
            "\nERROR: Missing required columns:"
        )

        for column in missing:

            print(
                f"  {column}"
            )

        sys.exit(1)

    return df


# ============================================================
# SELECT DATA
# ============================================================

def select_data(df):

    print("")
    print("=" * 50)

    print(
        "DATA SELECTION"
    )

    print("=" * 50)

    print(
        f"Mode: {SELECTION_MODE}"
    )

    # --------------------------------------------------------
    # FULL DATASET
    # --------------------------------------------------------

    if SELECTION_MODE == "full":

        selected = df.copy()

        print(
            f"Selected: all {len(selected)} rows"
        )

    # --------------------------------------------------------
    # SPECIFIC ROWS
    # --------------------------------------------------------

    elif SELECTION_MODE == "rows":

        invalid_rows = [
            row
            for row in SELECTED_ROWS
            if row < 0 or row >= len(df)
        ]

        if invalid_rows:

            print(
                "\nERROR: Invalid row indices:"
            )

            print(
                invalid_rows
            )

            sys.exit(1)

        selected = df.iloc[
            SELECTED_ROWS
        ].copy()

        print(
            f"Selected rows: "
            f"{SELECTED_ROWS}"
        )

    # --------------------------------------------------------
    # LOCATION RANGE
    # --------------------------------------------------------

    elif SELECTION_MODE == "range":

        minimum, maximum = (
            LOCATION_RANGE
        )

        if minimum > maximum:

            minimum, maximum = (
                maximum,
                minimum
            )

        selected = df[
            (df["Loc"] >= minimum)
            &
            (df["Loc"] <= maximum)
        ].copy()

        print(
            f"Location range: "
            f"{minimum} - {maximum}"
        )

        print(
            f"Selected: "
            f"{len(selected)} rows"
        )

    # --------------------------------------------------------
    # RANDOM SAMPLE
    # --------------------------------------------------------

    elif SELECTION_MODE == "random":

        if RANDOM_SAMPLE_SIZE <= 0:

            print(
                "ERROR: "
                "RANDOM_SAMPLE_SIZE must "
                "be greater than zero."
            )

            sys.exit(1)

        if RANDOM_SAMPLE_SIZE > len(df):

            print(
                f"WARNING: Requested "
                f"{RANDOM_SAMPLE_SIZE} rows, "
                f"but dataset only contains "
                f"{len(df)} rows."
            )

            print(
                "Using the entire dataset."
            )

            selected = df.copy()

        else:

            selected = df.sample(
                n=RANDOM_SAMPLE_SIZE,
                random_state=RANDOM_STATE
            ).copy()

        print(
            f"Random sample: "
            f"{len(selected)} rows"
        )

    # --------------------------------------------------------
    # INVALID MODE
    # --------------------------------------------------------

    else:

        print(
            f"\nERROR: Unknown "
            f"SELECTION_MODE: "
            f"{SELECTION_MODE}"
        )

        print(
            "\nValid modes:"
        )

        print(
            "  full"
        )

        print(
            "  rows"
        )

        print(
            "  range"
        )

        print(
            "  random"
        )

        sys.exit(1)

    if len(selected) == 0:

        print(
            "\nERROR: Selection produced "
            "zero rows."
        )

        sys.exit(1)

    return selected


# ============================================================
# DETERMINE FAULT TYPE
# ============================================================

def determine_fault_type(
    predictions
):

    active_faults = [
        target
        for target in TARGET_COLUMNS
        if predictions[target] == 1
    ]

    if not active_faults:

        return "NONE"

    return "-".join(
        active_faults
    )


# ============================================================
# DETERMINE CONFIDENCE
# ============================================================

def get_confidences(
    predictions,
    probabilities
):

    confidences = {}

    for target in TARGET_COLUMNS:

        prediction = predictions[
            target
        ]

        probability = probabilities[
            target
        ]

        if prediction == 1:

            confidence = probability[1]

        else:

            confidence = probability[0]

        confidences[target] = confidence

    return confidences


# ============================================================
# FORMAT FAULT NAME
# ============================================================

def format_fault_name(
    fault_type
):

    if fault_type == "NONE":

        return "None"

    replacements = {
        "G": "Ground",
        "A": "Phase A",
        "B": "Phase B",
        "C": "Phase C",
    }

    parts = fault_type.split("-")

    return " - ".join(
        replacements.get(
            part,
            part
        )
        for part in parts
    )


# ============================================================
# BUILD HUMAN-READABLE REPORT
# ============================================================

def build_text_report(
    results,
    report_lines
):

    report_lines.append(
        "=" * 50
    )

    report_lines.append(
        "FAULT DETECTION RESULTS"
    )

    report_lines.append(
        "=" * 50
    )

    report_lines.append("")

    for index, row in results.iterrows():

        report_lines.append(
            f"Row {index + 1}"
        )

        report_lines.append(
            f"Location: {row['Loc']}"
        )

        report_lines.append("")

        if row["fault_type"] == "NONE":

            report_lines.append(
                "Status: NO FAULT"
            )

            report_lines.append("")

            report_lines.append(
                "Confidence:"
            )

            # For no-fault samples, show the
            # confidence that each line is normal.

            for target in TARGET_COLUMNS:

                confidence = row[
                    f"{target}_confidence"
                ]

                line_name = {
                    "G": "Ground",
                    "A": "Phase A",
                    "B": "Phase B",
                    "C": "Phase C",
                }[target]

                report_lines.append(
                    f"    {line_name}: "
                    f"{confidence:.1f}% normal"
                )

        else:

            report_lines.append(
                "Status: FAULT DETECTED"
            )

            report_lines.append(
                f"Fault type: "
                f"{format_fault_name(row['fault_type'])}"
            )

            report_lines.append("")

            report_lines.append(
                "Confidence:"
            )

            # Only display the lines which
            # were actually classified as faulty.

            for target in TARGET_COLUMNS:

                if row[
                    f"{target}_fault"
                ] == 1:

                    confidence = row[
                        f"{target}_confidence"
                    ]

                    line_name = {
                        "G": "Ground",
                        "A": "Phase A",
                        "B": "Phase B",
                        "C": "Phase C",
                    }[target]

                    report_lines.append(
                        f"    {line_name}: "
                        f"{confidence:.1f}%"
                    )

        report_lines.append("")

        report_lines.append(
            "-" * 50
        )

        report_lines.append("")

    return report_lines


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 50)

    print(
        "TRANSMISSION LINE FAULT DETECTION"
    )

    print(
        "RBF SUPPORT VECTOR MACHINE"
    )

    print("=" * 50)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    models = load_models()

    df = load_data()

    print(
        f"\nInput rows: {len(df)}"
    )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    selected_df = select_data(
        df
    )

    print("\nSELECTED DATA:")
    print(
        selected_df[
            ["Loc"] + FEATURE_COLUMNS
        ].head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = selected_df[
        FEATURE_COLUMNS
    ].copy()

    if X.isnull().any().any():

        print(
            "\nERROR: Input contains "
            "missing feature values."
        )

        print(
            X.isnull().sum()
        )

        sys.exit(1)

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print("")
    print(
        "Running inference..."
    )

    predictions = {}
    probabilities = {}

    start = time.perf_counter()

    for target in TARGET_COLUMNS:

        model = models[target]

        predictions[target] = (
            model.predict(X)
        )

        probabilities[target] = (
            model.predict_proba(X)
        )

    end = time.perf_counter()

    total_latency = (
        end - start
    )

    latency_per_sample = (
        total_latency
        / len(X)
    )

    # --------------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame()

    results["Loc"] = (
        selected_df["Loc"]
        .values
    )

    # Binary predictions

    for target in TARGET_COLUMNS:

        results[
            f"{target}_fault"
        ] = predictions[target]

    # --------------------------------------------------------
    # FAULT TYPE + CONFIDENCE
    # --------------------------------------------------------

    fault_types = []

    overall_confidences = []

    individual_confidences = {
        target: []
        for target in TARGET_COLUMNS
    }

    for row_index in range(
        len(selected_df)
    ):

        row_predictions = {
            target:
            predictions[target][row_index]
            for target in TARGET_COLUMNS
        }

        row_probabilities = {
            target:
            probabilities[target][row_index]
            for target in TARGET_COLUMNS
        }

        fault_type = (
            determine_fault_type(
                row_predictions
            )
        )

        confidences = (
            get_confidences(
                row_predictions,
                row_probabilities
            )
        )

        fault_types.append(
            fault_type
        )

        for target in TARGET_COLUMNS:

            individual_confidences[
                target
            ].append(
                confidences[target]
            )

        # Overall confidence is the lowest
        # confidence among the four decisions.

        overall_confidences.append(
            min(
                confidences.values()
            )
        )

    results[
        "fault_type"
    ] = fault_types

    for target in TARGET_COLUMNS:

        results[
            f"{target}_confidence"
        ] = (
            np.array(
                individual_confidences[target]
            )
            * 100
        )

    results[
        "overall_confidence"
    ] = (
        np.array(
            overall_confidences
        )
        * 100
    )

    # ========================================================
    # CSV OUTPUT
    # ========================================================
    #
    # Keep the CSV compact and easy to analyse.
    #
    # Loc | Fault | G | C | B | A | Confidence
    #
    # ========================================================

    csv_results = pd.DataFrame()

    csv_results["Loc"] = (
        results["Loc"]
    )

    csv_results["Fault"] = (
        results["fault_type"]
        .replace(
            "NONE",
            "None"
        )
    )

    for target in TARGET_COLUMNS:

        csv_results[target] = (
            results[
                f"{target}_fault"
            ]
            .astype(int)
        )

    csv_results["Confidence"] = (
        results[
            "overall_confidence"
        ]
        .map(
            lambda x:
            f"{x:.1f}%"
        )
    )

    csv_results.to_csv(
        CSV_OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # TERMINAL TABLE
    # ========================================================

    print("")
    print("=" * 50)

    print(
        f"RESULTS "
        f"(first {min(TERMINAL_PREVIEW_ROWS, len(csv_results))} rows)"
    )

    print("=" * 50)

    preview = csv_results.head(
        TERMINAL_PREVIEW_ROWS
    )

    print(
        preview.to_string(
            index=False
        )
    )

    # ========================================================
    # HUMAN-READABLE TXT REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "TRANSMISSION LINE FAULT DETECTION"
    )

    report_lines.append(
        "RBF SUPPORT VECTOR MACHINE"
    )

    report_lines.append("")

    report_lines.append(
        f"Generated: {timestamp}"
    )

    report_lines.append(
        f"Input: {INPUT_FILE}"
    )

    report_lines.append(
        f"Model: {MODEL_FILE}"
    )

    report_lines.append(
        f"Selection mode: {SELECTION_MODE}"
    )

    report_lines.append(
        f"Samples analysed: {len(selected_df)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Inference latency: "
        f"{total_latency * 1000:.4f} ms"
    )

    report_lines.append(
        f"Latency per sample: "
        f"{latency_per_sample * 1000:.4f} ms"
    )

    report_lines.append("")

    build_text_report(
        results,
        report_lines
    )

    # --------------------------------------------------------
    # SAVE TXT
    # --------------------------------------------------------

    with TXT_OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

    # ========================================================
    # FINAL TERMINAL MESSAGE
    # ========================================================

    print("")
    print("=" * 50)

    print(
        "INFERENCE COMPLETE"
    )

    print("=" * 50)

    print(
        f"\nTXT report:"
    )

    print(
        f"  {TXT_OUTPUT_FILE}"
    )

    print(
        f"\nCSV results:"
    )

    print(
        f"  {CSV_OUTPUT_FILE}"
    )

    print(
        f"\nProcessed:"
        f" {len(selected_df)} rows"
    )

    print(
        f"Total latency:"
        f" {total_latency * 1000:.4f} ms"
    )

    print(
        f"Latency/sample:"
        f" {latency_per_sample * 1000:.4f} ms"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()