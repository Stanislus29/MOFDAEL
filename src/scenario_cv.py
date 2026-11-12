"""
R9 reconciliation: stratified group K-fold cross-validation
(K = 7) as specified in Section 3.12.1, with grouping enforced at
scenario level (Section 3.8.2).

Purpose
-------

Chapter Three claims a stratified K = 7 scenario-grouped
cross-validation for hyperparameter selection. The training
pipeline in fact uses a single stratified group-aware holdout
split with hyperparameters fixed a priori. This script provides
the missing evidence and the reconciliation:

  1. Fold composition — scenarios per fold, rows per fold, and
     fault-signature proportions per fold, with verification that
     no scenario spans more than one fold.
  2. Descriptive per-fold overall detection accuracy for each
     model, trained with the SAME fixed hyperparameters as the
     canonical scripts (imported directly from src/<model>.py so
     every number is traceable to the canonical configuration).
  3. Mean ± SD across folds per model — the across-fold
     dispersion measure required by Section 3.8.7.

The CV is descriptive, not a hyperparameter search; the chapter
text must state this (see TODO.md R9).

The LSTM is not included in the per-fold model sweep: it consumes
scenario-level sequences through its own padding/weighting
pipeline, and its scenario-level compliance is already
established by the shared scenario partition (its events map onto
scenarios one-to-one). Its holdout evaluation is unchanged.

IMPLEMENTATION NOTE: each model's CV runs in a SEPARATE python
subprocess (`--model <name>`). The canonical training scripts
import torch (ann, rnn) and tensorflow (cnn, pinn); loading both
runtimes into one process corrupts the native heap (SIGABRT in
free()). Lazy per-model imports plus subprocess isolation keep
each interpreter single-runtime. Fold assignment is deterministic
(StratifiedGroupKFold, fixed seed), so every subprocess reproduces
identical folds.

Outputs (outputs/scenario_cv/):

    report_<timestamp>.txt
    cv_results_<timestamp>.csv
    partial_<model>.csv      (intermediate, per subprocess)
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from scenarios import assign_scenario_ids

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_cv"

DATA_FILE = DATA_DIR / "classdetectfault_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMNS = ["G", "C", "B", "A"]

FEATURE_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

N_SPLITS = 7

RANDOM_STATE = 42

# Models included in the descriptive per-fold sweep. The LSTM is
# excluded (see module docstring).
CV_MODELS = ["knn", "dtr", "svm", "ann", "rnn", "cnn", "pinn"]


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


class SilentLogger:

    """Shim passed to the canonical train_fault_model functions."""

    def print(self, message=""):
        pass


def load_data():

    df = pd.read_csv(DATA_FILE)

    scenario_ids = assign_scenario_ids(df)

    sig = (
        df["G"].astype(int).astype(str)
        + df["C"].astype(int).astype(str)
        + df["B"].astype(int).astype(str)
        + df["A"].astype(int).astype(str)
    ).to_numpy()

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMNS]

    return df, scenario_ids, sig, X, y


def make_folds(scenario_ids, sig):

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    X_stub = np.zeros((len(sig), 1))

    return list(
        sgkf.split(X_stub, sig, groups=scenario_ids)
    )


def train_and_predict(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    target: str,
    silent: SilentLogger,
) -> np.ndarray:

    """
    Train one binary fault classifier on a CV fold using the
    canonical builder from the model's own script (imported
    lazily so each subprocess loads a single ML runtime), and
    return binary predictions for the fold's test rows.
    """

    if model_name == "knn":

        import knn as knn_module

        model = knn_module.create_knn()

        model.fit(X_train, y_train)

        return model.predict(X_test).astype(int)

    if model_name == "dtr":

        import dtr as dtr_module

        model = dtr_module.create_dt()

        model.fit(X_train, y_train)

        return model.predict(X_test).astype(int)

    if model_name == "svm":

        import svm as svm_module

        model = svm_module.create_svm()

        model.fit(X_train, y_train)

        return model.predict(X_test).astype(int)

    if model_name == "ann":

        import ann as ann_module

        model = ann_module.train_fault_model(
            X_train, y_train, target, silent
        )[0]

        return model.predict(X_test).astype(int)

    if model_name == "rnn":

        import rnn as rnn_module

        model = rnn_module.train_fault_model(
            X_train, y_train, target, silent
        )[0]

        return model.predict(X_test).astype(int)

    if model_name == "cnn":

        import cnn as cnn_module

        scaler = StandardScaler().fit(X_train)

        model = cnn_module.train_fault_model(
            scaler.transform(X_train),
            y_train,
            target,
            silent,
        )[0]

        predict_fn = cnn_module.make_predict_fn(model, scaler)

        return predict_fn(X_test)[0]

    if model_name == "pinn":

        import pinn as pinn_module

        # Canonical pinn.py trains on float32-scaled inputs; the
        # physics residual multiplies the current columns by a
        # float32 constant, so the batch dtype must match.

        scaler = StandardScaler().fit(X_train)

        X_train_scaled = scaler.transform(X_train).astype(
            np.float32
        )

        current_mean = scaler.mean_[
            pinn_module.CURRENT_INDICES
        ].astype(np.float32)

        current_std = scaler.scale_[
            pinn_module.CURRENT_INDICES
        ].astype(np.float32)

        model = pinn_module.train_fault_model(
            X_train_scaled,
            y_train,
            target,
            current_mean,
            current_std,
            silent,
        )[0]

        predict_fn = pinn_module.make_predict_fn(model, scaler)

        return predict_fn(X_test)[0]

    raise ValueError(f"unknown model: {model_name}")


def run_single_model(model_name: str) -> None:

    """
    Subprocess entry point: descriptive CV for one model.
    Writes partial_<model>.csv.
    """

    df, scenario_ids, sig, X, y = load_data()

    folds = make_folds(scenario_ids, sig)

    silent = SilentLogger()

    partial = OUTPUT_DIR / f"partial_{model_name}.csv"

    # Resume: folds are deterministic, so any folds already
    # recorded in the partial CSV (from an interrupted earlier
    # run) are kept and only the remaining folds are trained.

    records = []

    done_folds = set()

    if partial.exists():

        try:

            existing = pd.read_csv(partial)

            if {"detection_accuracy", "exact_match_accuracy"} <= set(
                existing.columns
            ):

                records = existing.to_dict("records")

                done_folds = set(existing["fold"].astype(int))

        except pd.errors.EmptyDataError:

            pass

    for k, (train_pos, test_pos) in enumerate(folds):

        if k in done_folds:

            continue

        X_tr, X_te = X.iloc[train_pos], X.iloc[test_pos]

        y_tr, y_te = y.iloc[train_pos], y.iloc[test_pos]

        correct = np.ones(len(test_pos), dtype=bool)

        det_pred = np.zeros(len(test_pos), dtype=bool)

        for target in TARGET_COLUMNS:

            pred = train_and_predict(
                model_name,
                X_tr,
                y_tr[target],
                X_te,
                target,
                silent,
            )

            correct &= pred == y_te[target].to_numpy()

            det_pred |= pred.astype(bool)

        det_true = y_te.any(axis=1).to_numpy()

        records.append({
            "model": model_name,
            "fold": k,
            "test_rows": len(test_pos),
            "detection_accuracy": float(
                (det_pred == det_true).mean()
            ),
            "exact_match_accuracy": float(correct.mean()),
        })

        pd.DataFrame(records).to_csv(
            OUTPUT_DIR / f"partial_{model_name}.csv", index=False
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    if args.model is not None:

        run_single_model(args.model)

        return

    # --------------------------------------------------------
    # DRIVER: fold composition, subprocesses, merge
    # --------------------------------------------------------

    report_path = OUTPUT_DIR / f"report_{timestamp}.txt"

    results_path = OUTPUT_DIR / f"cv_results_{timestamp}.csv"

    logger = ReportLogger(report_path)

    try:

        logger.print("=" * 70)
        logger.print(
            "R9: STRATIFIED GROUP K-FOLD CV (K = 7) — "
            "SECTION 3.12.1 RECONCILIATION"
        )
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")
        logger.print(f"Dataset: {DATA_FILE}")
        logger.print(
            f"Models: {', '.join(CV_MODELS)} "
            "(LSTM excluded — see report notes; each model's CV "
            "runs in an isolated subprocess)"
        )
        logger.print("")

        df, scenario_ids, sig, X, y = load_data()

        folds = make_folds(scenario_ids, sig)

        # ----------------------------------------------------
        # FOLD COMPOSITION
        # ----------------------------------------------------

        logger.print("-" * 70)
        logger.print("FOLD COMPOSITION (grouped at scenario level)")
        logger.print("-" * 70)
        logger.print(
            f"  {'fold':>4} | {'scenarios':>9} | {'rows':>6} | "
            "signature row proportions"
        )

        composition_ok = True

        for k, (_, test_pos) in enumerate(folds):

            fold_scenarios = np.unique(scenario_ids[test_pos])

            fold_sig = sig[test_pos]

            values, counts = np.unique(
                fold_sig, return_counts=True
            )

            props = ", ".join(
                f"{v}:{c / len(fold_sig):.1%}"
                for v, c in zip(values, counts)
            )

            logger.print(
                f"  {k:>4} | {len(fold_scenarios):>9} | "
                f"{len(test_pos):>6} | {props}"
            )

            for s in fold_scenarios:

                scenario_rows = np.where(scenario_ids == s)[0]

                if not np.all(np.isin(scenario_rows, test_pos)):

                    composition_ok = False

        logger.print("")
        logger.print(
            f"  scenarios spanning folds: "
            f"{'0 — verified' if composition_ok else 'VIOLATION'}"
        )
        logger.print(
            "  (grouping enforced at scenario level, "
            "Section 3.8.2)"
        )
        logger.print("")

        # ----------------------------------------------------
        # PER-MODEL SUBPROCESSES
        # ----------------------------------------------------

        logger.print("-" * 70)
        logger.print(
            "DESCRIPTIVE PER-FOLD OVERALL DETECTION ACCURACY "
            "(fixed hyperparameters)"
        )
        logger.print("-" * 70)

        for model_name in CV_MODELS:

            partial = OUTPUT_DIR / f"partial_{model_name}.csv"

            # Reuse a valid completed partial (7 folds, both
            # metric columns) so a re-run after a failure only
            # retrains the missing models.

            reusable = False

            if partial.exists():

                try:

                    existing = pd.read_csv(partial)

                    reusable = (
                        len(existing) == N_SPLITS
                        and "detection_accuracy" in existing.columns
                        and "exact_match_accuracy" in existing.columns
                    )

                except pd.errors.EmptyDataError:

                    reusable = False

            if not reusable:

                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--model",
                        model_name,
                    ],
                    cwd=str(PROJECT_ROOT),
                )

                if result.returncode != 0 or not partial.exists():

                    logger.print(
                        f"  {model_name:>6}: SUBPROCESS FAILED "
                        f"(rc = {result.returncode})"
                    )

                    continue

            part = pd.read_csv(partial)

            det = part["detection_accuracy"].to_numpy()

            exact = part["exact_match_accuracy"].to_numpy()

            logger.print(
                f"  {model_name:>6}: detection "
                + " ".join(f"{a:.4f}" for a in det)
                + f"  |  mean {det.mean():.4f} "
                f"± {det.std(ddof=1):.4f}"
            )

            logger.print(
                f"  {'':>6}  exact-match "
                + " ".join(f"{a:.4f}" for a in exact)
                + f"  |  mean {exact.mean():.4f} "
                f"± {exact.std(ddof=1):.4f}"
            )

        logger.print("")
        logger.print(
            "  Reconciliation: the K = 7 scenario-grouped CV of "
            "Section 3.12.1 is reported here as DESCRIPTIVE "
            "evidence with the final fixed hyperparameters; it "
            "was not used for hyperparameter selection. The "
            "inferential holdout split is the shared scenario-"
            "level partition of src/scenarios.py. The chapter "
            "text must state both facts (see TODO.md R9)."
        )
        logger.print("")

        # ----------------------------------------------------
        # MERGE PARTIALS
        # ----------------------------------------------------

        partials = [
            OUTPUT_DIR / f"partial_{m}.csv" for m in CV_MODELS
        ]

        partials = [p for p in partials if p.exists()]

        if partials:

            merged = pd.concat(
                [pd.read_csv(p) for p in partials],
                ignore_index=True,
            )

            merged.to_csv(results_path, index=False)

            for p in partials:
                p.unlink()

        logger.print("=" * 70)
        logger.print("OUTPUT FILES")
        logger.print("=" * 70)
        logger.print(f"Report : {report_path}")
        logger.print(f"Results: {results_path}")

    finally:

        logger.close()


if __name__ == "__main__":

    main()
