from __future__ import annotations

from pathlib import Path
from datetime import datetime
from itertools import combinations

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESULTS_DIR = OUTPUTS_DIR / "combination_accuracy"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = RESULTS_DIR / f"report_{timestamp}.txt"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMNS = ["G", "C", "B", "A"]


# ============================================================
# REPORT LOGGER
# ============================================================

class ReportLogger:

    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")

    def print(self, message=""):
        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# ENUMERATE ALL nCr LINE COMBINATIONS (r = 1 TO 4)
# ============================================================

def all_combinations():

    combos = []

    for r in range(1, len(TARGET_COLUMNS) + 1):

        for combo in combinations(TARGET_COLUMNS, r):

            combos.append(combo)

    return combos


def combo_label(combo):

    return "+".join(combo)


def matches_combo(df, combo, suffix):

    mask = pd.Series(True, index=df.index)

    for column in TARGET_COLUMNS:

        expected = 1 if column in combo else 0

        mask &= df[f"{column}_{suffix}"] == expected

    return mask


# ============================================================
# COMPUTE ACCURACY FOR ONE MODEL'S PREDICTIONS
# ============================================================

def compute_combination_accuracy(df, model_name, logger):

    exact_match = pd.Series(True, index=df.index)

    for column in TARGET_COLUMNS:

        exact_match &= df[f"{column}_true"] == df[f"{column}_pred"]

    records = []

    logger.print(f"\n{'=' * 70}")
    logger.print(f"MODEL: {model_name}")
    logger.print("=" * 70)

    for degree in range(1, len(TARGET_COLUMNS) + 1):

        degree_label = {
            1: "single-line",
            2: "double-line",
            3: "triple-line",
            4: "quadruple-line",
        }[degree]

        degree_mask = pd.Series(False, index=df.index)

        for combo in combinations(TARGET_COLUMNS, degree):

            degree_mask |= matches_combo(df, combo, "true")

        n_degree = int(degree_mask.sum())

        degree_accuracy = (
            exact_match[degree_mask].mean() if n_degree > 0 else float("nan")
        )

        logger.print(
            f"\n{degree_label} faults (n={n_degree}): "
            f"{'accuracy = ' + format(degree_accuracy, '.6f') if n_degree > 0 else 'no test examples'}"
        )

        records.append({
            "model": model_name,
            "degree": degree,
            "degree_label": degree_label,
            "combination": "ANY",
            "n": n_degree,
            "accuracy": degree_accuracy,
        })

        for combo in combinations(TARGET_COLUMNS, degree):

            combo_mask = matches_combo(df, combo, "true")

            n_combo = int(combo_mask.sum())

            combo_accuracy = (
                exact_match[combo_mask].mean() if n_combo > 0 else float("nan")
            )

            label = combo_label(combo)

            logger.print(
                f"  {label:12s} n={n_combo:6d}  "
                f"{'accuracy = ' + format(combo_accuracy, '.6f') if n_combo > 0 else 'no test examples'}"
            )

            records.append({
                "model": model_name,
                "degree": degree,
                "degree_label": degree_label,
                "combination": label,
                "n": n_combo,
                "accuracy": combo_accuracy,
            })

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    logger = ReportLogger(REPORT_FILE)

    try:

        logger.print("=" * 70)
        logger.print("PER-COMBINATION ACCURACY (single- through quadruple-line)")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")

        files = sorted(OUTPUTS_DIR.glob("*/predictions_*.csv"))

        logger.print(f"\nPrediction files found: {len(files)}")

        all_records = []

        for path in files:

            model_name = path.stem.replace("predictions_", "")

            df = pd.read_csv(path)

            all_records += compute_combination_accuracy(
                df, model_name, logger
            )

        results_df = pd.DataFrame(all_records)

        results_file = RESULTS_DIR / f"results_{timestamp}.csv"

        results_df.to_csv(results_file, index=False)

        logger.print("\n" + "=" * 70)
        logger.print("OUTPUT FILES")
        logger.print("=" * 70)
        logger.print(f"Report:  {REPORT_FILE}")
        logger.print(f"Results: {results_file}")

    finally:

        logger.close()


if __name__ == "__main__":
    main()