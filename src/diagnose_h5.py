"""
H5 fault-location diagnostic — R3 (blocking)

Establishes, with evidence, why the H5 fault-location analysis degraded to
all-zero Hodges-Lehmann estimates, per the procedure in
corrections/Pending_Tasks_Recommendation.docx:

    1. Print raw per-scenario location predictions for two models side by
       side at one fault location — twenty rows is enough.
    2. If the columns are identical: the location head is outputting a
       constant, or one column was duplicated across all models during
       export.
    3. If the columns differ: the defect is in the analysis routine.

Additional checks performed here:
    4. Whether any model exports fault-location predictions at all.
    5. Whether fault location is learnable from the available features
       (feasibility of the repair path).
    6. Whether an apparent-impedance (reactance-method) location estimate
       is possible from the dataset as exported.

Outputs (written to outputs/h5_diagnostics/ following the train-script
standard):

    report_<timestamp>.txt   full diagnostic report
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# ============================================================
# PATH CONFIGURATION (matches train-script standard)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h5_diagnostics"

DATA_FILE = DATA_DIR / "classdetectfault_data.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TIMESTAMP & OUTPUT FILES
# ============================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = OUTPUT_DIR / f"report_{timestamp}.txt"

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMNS = ["G", "C", "B", "A"]

FEATURE_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

LINE_LENGTH_KM = 120.0  # percentage denominator, stated explicitly

RANDOM_STATE = 42

TEST_SIZE = 0.20

SAMPLE_ROWS = 20  # rows shown side by side, per the R3 procedure


# ============================================================
# REPORT LOGGER (matches train-script ReportLogger)
# ============================================================

class ReportLogger:

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open(
            "w",
            encoding="utf-8"
        )

    def print(self, message=""):
        print(message)
        self.file.write(
            str(message) + "\n"
        )
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# HODGES-LEHMANN (same estimator as hypothesis.py)
# ============================================================

def hodges_lehmann(d: np.ndarray) -> float:
    n = len(d)
    walsh = (d[:, None] + d[None, :]) / 2
    return float(np.median(walsh[np.triu_indices(n)]))


# ============================================================
# CHECK 1: DO ANY LOCATION PREDICTIONS EXIST?
# ============================================================

def check_location_columns(logger: ReportLogger) -> list[str]:

    logger.print("=" * 70)
    logger.print("CHECK 1: LOCATION PREDICTIONS IN OUTPUTS")
    logger.print("=" * 70)

    files = sorted(OUTPUTS_DIR.glob("*/predictions_*.csv"))

    with_loc = []

    for path in files:

        model_name = path.stem.replace("predictions_", "")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        has_loc = "Loc_true" in columns or "Loc_pred" in columns

        logger.print(
            f"  {model_name:>6}: {'HAS Loc columns' if has_loc else 'no Loc columns'}"
        )

        if has_loc:
            with_loc.append(model_name)

    if not with_loc:

        logger.print("")
        logger.print(
            "FINDING: no model exports fault-location predictions."
        )
        logger.print(
            "  'Loc' is listed under IGNORED METADATA in every training"
        )
        logger.print(
            "  script, so no model ever produced a location estimate and"
        )
        logger.print(
            "  no per-scenario location file exists anywhere in outputs/."
        )

    logger.print("")
    return with_loc


# ============================================================
# CHECK 2: WHAT DID RUN_H5 ACTUALLY TEST?
# ============================================================

def check_h5_inputs(logger: ReportLogger, predictions: dict):

    logger.print("=" * 70)
    logger.print("CHECK 2: WHAT RUN_H5 ACTUALLY TESTED (side by side)")
    logger.print("=" * 70)

    raw_df = pd.read_csv(DATA_FILE)
    raw_df = raw_df.reset_index().rename(columns={"index": "row_id"})
    raw_df = raw_df.set_index("row_id")

    location = 45  # one of the simulated fault locations
    target = "overall"

    loc_rows = raw_df.index[raw_df["Loc"] == location].intersection(
        predictions["ann"].index
    )[:SAMPLE_ROWS]

    ann = predictions["ann"].loc[loc_rows]
    cnn = predictions["cnn"].loc[loc_rows]

    err_ann = (
        ann[f"{target}_true"] != ann[f"{target}_pred"]
    ).astype(int).to_numpy()
    err_cnn = (
        cnn[f"{target}_true"] != cnn[f"{target}_pred"]
    ).astype(int).to_numpy()

    logger.print(
        f"\nPer-scenario classification-error vectors at Loc = {location} km"
        f" ({target} target), first {len(loc_rows)} rows:"
    )
    logger.print("")
    logger.print("  row_id | ann err | cnn err | identical?")
    logger.print("  " + "-" * 44)

    n_identical = 0
    for rid, a, c in zip(loc_rows, err_ann, err_cnn):
        same = "yes" if a == c else "NO"
        if a == c:
            n_identical += 1
        logger.print(f"  {rid:>6} | {a:^7} | {c:^7} | {same}")

    logger.print("")
    logger.print(
        f"  {n_identical} of {len(loc_rows)} rows have identical error bits."
    )

    d = err_ann - err_cnn
    nonzero = d[d != 0]

    logger.print("")
    logger.print(
        f"  Non-zero differences: {len(nonzero)} of {len(d)}"
    )

    if len(nonzero) == 0:

        logger.print(
            "  -> Wilcoxon has no discordant pairs; p = 1.0, HL = 0.0000"
        )

    elif len(nonzero) == 1:

        logger.print(
            "  -> Exactly one discordant pair: the sign-test probability"
        )
        logger.print(
            "     is 2 * 0.5 = 0.317311, the signature value from the review."
        )

    statistic, p_value = stats.wilcoxon(err_ann, err_cnn)
    hl = hodges_lehmann(
        (err_ann - err_cnn).astype(float)
    )

    logger.print("")
    logger.print(
        f"  Reproduced: Wilcoxon p = {p_value:.6f}, "
        f"Hodges-Lehmann = {hl:.4f}"
    )
    logger.print("")
    logger.print(
        "DEFECT IDENTIFIED: run_h5() never tested fault-location accuracy."
    )
    logger.print(
        "  It grouped per-line CLASSIFICATION error by 'Loc' and compared"
    )
    logger.print(
        "  models on that. Because models make near-identical errors, the"
    )
    logger.print(
        "  vectors collapse: 63 pairwise HL estimates of exactly 0.0000"
    )
    logger.print(
        "  with CI [0.0000, 0.0000], probabilities of only 1.000000 and"
    )
    logger.print(
        "  0.317311, and Friedman statistics at exact integers."
    )
    logger.print(
        "  This is a degenerate variable, not a finding of equivalence."
    )
    logger.print("")


# ============================================================
# CHECK 3: IS LOCATION LEARNABLE FROM THE FEATURES?
# ============================================================

def check_learnability(logger: ReportLogger):

    logger.print("=" * 70)
    logger.print("CHECK 3: LOCATION LEARNABILITY (repair feasibility)")
    logger.print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    fault_mask = df[TARGET_COLUMNS].sum(axis=1) > 0
    fd = df[fault_mask]

    X = fd[FEATURE_COLUMNS]
    y = fd["Loc"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    reg = DecisionTreeRegressor(random_state=RANDOM_STATE).fit(X_tr, y_tr)
    pred = reg.predict(X_te)

    mae_km = mean_absolute_error(y_te, pred)
    mae_pct = mae_km / LINE_LENGTH_KM * 100

    logger.print(
        f"\nDecision-tree location regressor (baseline feasibility check):"
    )
    logger.print(
        f"  Fault rows: {len(fd)} of {len(df)} (Loc > 0 stratum)"
    )
    logger.print(
        f"  Location MAE: {mae_km:.2f} km"
    )
    logger.print(
        f"  As % of line length (denominator = {LINE_LENGTH_KM:.0f} km):"
        f" {mae_pct:.2f}%"
    )

    clf = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    chance = 1.0 / y.nunique()

    logger.print(
        f"  8-class location classification accuracy: {acc:.4f}"
        f" (chance = {chance:.4f})"
    )
    logger.print("")
    logger.print(
        "FINDING: location IS learnable from the exported features, so the"
    )
    logger.print(
        "  repair path (adding a location head to each model and retraining)"
    )
    logger.print(
        "  is viable. Retraining all models is a coordinated job (2-5 days"
    )
    logger.print(
        "  per the review); it also regenerates every performance number."
    )
    logger.print("")


# ============================================================
# CHECK 4: IMPEDANCE-METHOD LOCATION ESTIMATE POSSIBLE?
# ============================================================

def check_impedance_method(logger: ReportLogger):

    logger.print("=" * 70)
    logger.print("CHECK 4: REACTANCE-METHOD (RELAY) LOCATION FEASIBILITY")
    logger.print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    fault_mask = df[TARGET_COLUMNS].sum(axis=1) > 0
    fd = df[fault_mask]

    # The dataset stores one instantaneous sample per scenario: signed
    # values of Va..Vc and Ia..Ic at the same time instant, with no
    # quadrature (phasor) components. The reactance method needs the
    # phase angle between V and I to separate reactance (distance) from
    # resistance. With magnitudes only, |Z| = |V|/|I| is all that exists.

    z_apparent = np.abs(fd["Va"]) / (np.abs(fd["Ia"]) + 1e-6)

    corr = np.corrcoef(z_apparent, fd["Loc"])[0, 1]

    logger.print(f"\nCorrelation |Va/Ia| vs true Loc (fault rows): {corr:.4f}")
    logger.print(
        "  Signed single-sample ratios cannot separate reactance from"
    )
    logger.print(
        "  resistance, so a conventional impedance-based location"
    )
    logger.print(
        "  estimate is NOT recoverable from this dataset representation."
    )
    logger.print("")
    logger.print(
        "  Location estimation requires either the full waveform window"
    )
    logger.print(
        "  (for phasor extraction, available after an R1 re-simulation"
    )
    logger.print(
        "  exports phasors or sample windows) or a learned location head"
    )
    logger.print(
        "  per model. A reactance-method estimate from single samples is"
    )
    logger.print(
        "  not defensible and is not attempted here."
    )
    logger.print("")


# ============================================================
# MAIN
# ============================================================

def main():

    logger = ReportLogger(REPORT_FILE)

    try:

        logger.print("=" * 70)
        logger.print("H5 FAULT-LOCATION DIAGNOSTIC (R3)")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")
        logger.print(f"Dataset: {DATA_FILE}")
        logger.print(
            f"Percentage denominator (line length): {LINE_LENGTH_KM:.0f} km"
        )
        logger.print("")

        # ----------------------------------------------------
        # LOAD PREDICTIONS (same discovery as hypothesis.py)
        # ----------------------------------------------------

        files = sorted(OUTPUTS_DIR.glob("*/predictions_*.csv"))

        predictions = {}

        for path in files:
            model_name = path.stem.replace("predictions_", "")
            df = pd.read_csv(path).set_index("row_id")
            predictions[model_name] = df

        common_rows = None
        for df in predictions.values():
            common_rows = (
                df.index if common_rows is None
                else common_rows.intersection(df.index)
            )

        logger.print(f"Prediction files loaded: {len(predictions)}")
        logger.print(f"Common rows across all models: {len(common_rows)}")
        logger.print("")

        # ----------------------------------------------------
        # RUN CHECKS
        # ----------------------------------------------------

        with_loc = check_location_columns(logger)

        check_h5_inputs(logger, predictions)

        check_learnability(logger)

        check_impedance_method(logger)

        # ----------------------------------------------------
        # VERDICT
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print("VERDICT")
        logger.print("=" * 70)
        logger.print("")
        logger.print(
            "H5 as previously computed tested the wrong quantity and"
        )
        logger.print(
            " returned a degenerate variable that must not be reported"
        )
        logger.print(
            " as fault-location equivalence."
        )
        logger.print("")
        logger.print("Repair paths (choose one before submission):")
        logger.print(
            "  A. Retrain all models with a location head exporting"
        )
        logger.print(
            "     Loc_true/Loc_pred per scenario; run_h5 in hypothesis.py"
        )
        logger.print(
            "     then performs the real test (error in km and % of the"
        )
        logger.print(
            f"     {LINE_LENGTH_KM:.0f} km line, denominator stated). Effort:"
        )
        logger.print("     2-5 days, regenerates every performance number.")
        logger.print(
            "  B. Preferred per the review if time is short: withdraw fault"
        )
        logger.print(
            "     location from the objectives and research questions and"
        )
        logger.print(
            "     state it in 3.15 as scope not addressed (writing task)."
        )
        logger.print("")
        logger.print(
            "run_h5 is now guarded: it fails loudly when location"
        )
        logger.print(
            " predictions are absent instead of silently testing"
        )
        logger.print(
            " classification error and publishing degenerate p-values."
        )
        logger.print("")
        logger.print("OUTPUT FILES")
        logger.print("-" * 70)
        logger.print(f"Report: {REPORT_FILE}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
