from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cnn"

DATA_FILE = DATA_DIR / "classdetectfault_data.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR = OUTPUT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# TIMESTAMP
# ============================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

REPORT_FILE = (
    OUTPUT_DIR
    / f"report_{timestamp}.txt"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "metrics.csv"
)

LOSS_FILE = (
    OUTPUT_DIR
    / "loss_history.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "predictions_cnn.csv"
)

ROBUSTNESS_FILE = (
    OUTPUT_DIR
    / "robustness_cnn.csv"
)

MANIFEST_FILE = MODEL_DIR / f"cnn_manifest_{timestamp}.joblib"

# ============================================================
# DATA CONFIGURATION
# ============================================================

# IMPORTANT:
# Loc is intentionally NOT here.

FEATURE_COLUMNS = [
    "Ia",
    "Ib",
    "Ic",
    "Va",
    "Vb",
    "Vc",
]

TARGET_COLUMNS = [
    "G",
    "C",
    "B",
    "A",
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

CNN_FILTERS = [32, 64]

CNN_KERNEL_SIZE = 2

CNN_DENSE_UNITS = 64

CNN_DROPOUT = 0.3

CNN_LEARNING_RATE = 0.001

CNN_EPOCHS = 100

CNN_BATCH_SIZE = 32

CNN_VALIDATION_SPLIT = 0.1

CNN_EARLY_STOPPING_PATIENCE = 10

ROBUSTNESS_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2]

ROBUSTNESS_TRIALS = 10

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# REPORT LOGGER
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
# BUILD CNN
# ============================================================

def create_cnn(input_length):

    model = models.Sequential()

    model.add(
        layers.Input(shape=(input_length, 1))
    )

    for filters in CNN_FILTERS:

        model.add(
            layers.Conv1D(
                filters=filters,
                kernel_size=CNN_KERNEL_SIZE,
                activation="relu",
            )
        )

    model.add(
        layers.Flatten()
    )

    model.add(
        layers.Dense(
            CNN_DENSE_UNITS,
            activation="relu"
        )
    )

    model.add(
        layers.Dropout(CNN_DROPOUT)
    )

    model.add(
        layers.Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=CNN_LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# PREDICTION HELPER
# ============================================================

def make_predict_fn(model, scaler):

    def predict_fn(X_raw):

        X_scaled = scaler.transform(X_raw)

        X_reshaped = X_scaled.reshape(
            -1, X_scaled.shape[1], 1
        )

        probabilities = model.predict(
            X_reshaped,
            verbose=0
        ).flatten()

        return (probabilities > 0.5).astype(int), probabilities

    return predict_fn


# ============================================================
# TRAIN ONE FAULT CLASSIFIER
# ============================================================

def train_fault_model(
    X_train_scaled,
    y_train,
    target,
    logger
):

    logger.print("")
    logger.print("=" * 70)

    logger.print(
        f"TRAINING CNN: {target}"
    )

    logger.print("=" * 70)

    logger.print(
        f"Filters        : {CNN_FILTERS}"
    )

    logger.print(
        f"Kernel size    : {CNN_KERNEL_SIZE}"
    )

    logger.print(
        f"Learning rate  : {CNN_LEARNING_RATE}"
    )

    logger.print(
        f"Max epochs     : {CNN_EPOCHS}"
    )

    logger.print(
        f"Batch size     : {CNN_BATCH_SIZE}"
    )

    model = create_cnn(
        X_train_scaled.shape[1]
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )

    class_weight_dict = {
        0: class_weights[0],
        1: class_weights[1],
    }

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CNN_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    X_reshaped = X_train_scaled.reshape(
        -1, X_train_scaled.shape[1], 1
    )

    start = time.perf_counter()

    history = model.fit(
        X_reshaped,
        y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_split=CNN_VALIDATION_SPLIT,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=0,
    )

    end = time.perf_counter()

    training_time = end - start

    best_epoch = int(
        np.argmin(history.history["val_loss"])
    )

    logger.print(
        f"\nTraining time: "
        f"{training_time:.6f} s"
    )

    logger.print(
        f"Epochs run   : "
        f"{len(history.history['loss'])}"
    )

    logger.print(
        f"Best epoch   : "
        f"{best_epoch + 1}"
    )

    logger.print(
        f"Best train loss: "
        f"{history.history['loss'][best_epoch]:.6f}"
    )

    logger.print(
        f"Best val loss  : "
        f"{history.history['val_loss'][best_epoch]:.6f}"
    )

    return model, training_time, history, best_epoch


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    predict_fn,
    X_test,
    y_test,
    target,
    logger
):

    predictions, probabilities = predict_fn(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    test_loss = float(
        log_loss(
            y_test,
            np.column_stack([1 - probabilities, probabilities]),
            labels=[0, 1]
        )
    )

    logger.print("")
    logger.print(
        f"--- {target} RESULTS ---"
    )

    logger.print(
        f"Accuracy : {accuracy:.6f}"
    )

    logger.print(
        f"Precision: {precision:.6f}"
    )

    logger.print(
        f"Recall   : {recall:.6f}"
    )

    logger.print(
        f"F1 Score : {f1:.6f}"
    )

    logger.print(
        f"Test log loss: {test_loss:.8f}"
    )

    logger.print(
        "\nConfusion Matrix:"
    )

    logger.print(
        str(cm)
    )

    logger.print(
        "\nClassification Report:"
    )

    logger.print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_loss": test_loss,
        "predictions": predictions,
    }


# ============================================================
# LATENCY
# ============================================================

def measure_latency(predict_fn, X_test):

    # Warm-up
    predict_fn(X_test.iloc[:1])

    start = time.perf_counter()

    predict_fn(X_test)

    end = time.perf_counter()

    total = end - start

    per_sample = (
        total / len(X_test)
    )

    return total, per_sample


# ============================================================
# ROBUSTNESS
# ============================================================

def test_robustness(predict_fn, X_test, y_test, noise_levels, trials):

    X_values = X_test.to_numpy(dtype=float)
    feature_std = np.std(X_values, axis=0)

    records = []

    for noise_level in noise_levels:

        accuracies = []
        f1s = []

        for trial in range(trials):

            rng = np.random.default_rng(RANDOM_STATE + trial)

            noise = rng.normal(
                0,
                noise_level * feature_std,
                size=X_values.shape
            )

            X_noisy = pd.DataFrame(
                X_values + noise,
                columns=X_test.columns
            )

            predictions, _ = predict_fn(X_noisy)

            accuracies.append(accuracy_score(y_test, predictions))
            f1s.append(f1_score(y_test, predictions, labels=[0, 1], zero_division=0))

        records.append({
            "noise_level": noise_level,
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
        })

    return records


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics, path):

    metrics_df = pd.DataFrame(
        metrics
    )

    if path.exists():

        metrics_df.to_csv(
            path,
            mode="a",
            header=False,
            index=False
        )

    else:

        metrics_df.to_csv(
            path,
            index=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    logger = ReportLogger(
        REPORT_FILE
    )

    try:

        logger.print(
            "=" * 70
        )

        logger.print(
            "TRANSMISSION LINE FAULT DETECTION"
        )

        logger.print(
            "CONVOLUTIONAL NEURAL NETWORK"
        )

        logger.print(
            "=" * 70
        )

        logger.print(
            f"Timestamp: {timestamp}"
        )

        logger.print(
            f"Dataset: {DATA_FILE}"
        )

        logger.print("")

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        if not DATA_FILE.exists():

            logger.print(
                f"ERROR: Dataset not found:"
            )

            logger.print(
                str(DATA_FILE)
            )

            sys.exit(1)

        df = pd.read_csv(
            DATA_FILE
        )

        logger.print(
            f"Dataset shape: {df.shape}"
        )

        # ----------------------------------------------------
        # CHECK COLUMNS
        # ----------------------------------------------------

        required = (
            ["Loc"]
            + FEATURE_COLUMNS
            + TARGET_COLUMNS
        )

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            logger.print(
                "\nMissing columns:"
            )

            for column in missing:

                logger.print(
                    f"  {column}"
                )

            sys.exit(1)

        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        initial_rows = len(df)

        df = df.dropna(
            subset=(
                FEATURE_COLUMNS
                + TARGET_COLUMNS
            )
        )

        removed = (
            initial_rows - len(df)
        )

        logger.print(
            f"Rows removed: {removed}"
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        # Loc is NOT used.

        X = df[
            FEATURE_COLUMNS
        ]

        y = df[
            TARGET_COLUMNS
        ]

        # ----------------------------------------------------
        # DISPLAY FEATURE INFORMATION
        # ----------------------------------------------------

        logger.print("")
        logger.print(
            "FEATURES:"
        )

        for feature in FEATURE_COLUMNS:

            logger.print(
                f"  {feature}"
            )

        logger.print("")
        logger.print(
            "IGNORED METADATA:"
        )

        logger.print(
            "  Loc"
        )

        logger.print("")
        logger.print(
            "TARGETS:"
        )

        for target in TARGET_COLUMNS:

            logger.print(
                f"  {target}"
            )

        # ----------------------------------------------------
        # CLASS DISTRIBUTION
        # ----------------------------------------------------

        logger.print("")
        logger.print(
            "=" * 70
        )

        logger.print(
            "CLASS DISTRIBUTION"
        )

        logger.print(
            "=" * 70
        )

        for target in TARGET_COLUMNS:

            logger.print(
                f"\n{target}:"
            )

            counts = (
                y[target]
                .value_counts()
                .sort_index()
            )

            for value, count in counts.items():

                percentage = (
                    count / len(y)
                ) * 100

                logger.print(
                    f"  {value}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        X_train, X_test, \
        y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        test_row_id = pd.Series(X_test.index, name="row_id").reset_index(drop=True)

        logger.print("")
        logger.print(
            f"Training samples: "
            f"{len(X_train)}"
        )

        logger.print(
            f"Testing samples: "
            f"{len(X_test)}"
        )

        # ----------------------------------------------------
        # SCALE FEATURES
        # ----------------------------------------------------
        # Fit once, shared across all four targets, since the
        # feature columns are the same regardless of which
        # target is being trained.

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ----------------------------------------------------
        # TRAIN MODELS
        # ----------------------------------------------------

        models_dict = {}

        model_paths = {}

        metrics = []

        loss_records = []

        robustness_records = []

        predictions_wide = pd.DataFrame({
            "row_id": test_row_id
        })

        for target in TARGET_COLUMNS:

            model, training_time, history, best_epoch = (
                train_fault_model(
                    X_train_scaled,
                    y_train[target].to_numpy(),
                    target,
                    logger
                )
            )

            models_dict[target] = model

            predict_fn = make_predict_fn(model, scaler)

            for epoch in range(len(history.history["loss"])):

                loss_records.append({
                    "timestamp": timestamp,
                    "target": target,
                    "epoch": epoch + 1,
                    "train_loss": history.history["loss"][epoch],
                    "val_loss": history.history["val_loss"][epoch],
                    "train_accuracy": history.history["accuracy"][epoch],
                    "val_accuracy": history.history["val_accuracy"][epoch],
                })

            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            results = evaluate_model(
                predict_fn,
                X_test,
                y_test[target],
                target,
                logger
            )

            predictions_wide[f"{target}_true"] = y_test[target].reset_index(drop=True)
            predictions_wide[f"{target}_pred"] = results["predictions"]

            # ------------------------------------------------
            # LATENCY
            # ------------------------------------------------

            total_latency, \
            per_sample_latency = (
                measure_latency(
                    predict_fn,
                    X_test
                )
            )

            logger.print(
                f"Total inference latency: "
                f"{total_latency * 1000:.6f} ms"
            )

            logger.print(
                f"Latency/sample: "
                f"{per_sample_latency * 1000:.6f} ms"
            )

            # ------------------------------------------------
            # ROBUSTNESS
            # ------------------------------------------------

            robustness_sweep = test_robustness(
                predict_fn,
                X_test,
                y_test[target],
                ROBUSTNESS_NOISE_LEVELS,
                ROBUSTNESS_TRIALS
            )

            logger.print("\nRobustness sweep:")

            for record in robustness_sweep:

                logger.print(
                    f"  noise {record['noise_level'] * 100:.1f}%: "
                    f"accuracy {record['accuracy_mean']:.4f} "
                    f"(+/- {record['accuracy_std']:.4f}), "
                    f"f1 {record['f1_mean']:.4f} "
                    f"(+/- {record['f1_std']:.4f})"
                )

                robustness_records.append({
                    "timestamp": timestamp,
                    "target": target,
                    **record,
                })

            reference = robustness_sweep[0]

            retention = (
                reference["accuracy_mean"] / results["accuracy"]
                if results["accuracy"] > 0
                else 0
            )

            # ------------------------------------------------
            # SAVE THIS TARGET'S MODEL
            # ------------------------------------------------

            model_path = (
                MODEL_DIR
                / f"cnn_model_{timestamp}_{target}.keras"
            )

            model.save(model_path)

            model_paths[target] = str(model_path)

            metrics.append({

                "timestamp": timestamp,

                "target": target,

                "filters": str(CNN_FILTERS),

                "kernel_size": CNN_KERNEL_SIZE,

                "learning_rate": CNN_LEARNING_RATE,

                "epochs_run": len(history.history["loss"]),

                "best_epoch": best_epoch + 1,

                "accuracy": results[
                    "accuracy"
                ],

                "precision": results[
                    "precision"
                ],

                "recall": results[
                    "recall"
                ],

                "f1_score": results[
                    "f1"
                ],

                "training_time_seconds":
                    training_time,

                "test_log_loss":
                    results["test_loss"],

                "latency_ms_total":
                    total_latency * 1000,

                "latency_ms_per_sample":
                    per_sample_latency * 1000,

                "robustness_accuracy":
                    reference["accuracy_mean"],

                "robustness_f1":
                    reference["f1_mean"],

                "robustness_retention":
                    retention,
            })

        # ----------------------------------------------------
        # OVERALL FAULT OCCURRED (any line flagged)
        # ----------------------------------------------------

        true_cols = [f"{t}_true" for t in TARGET_COLUMNS]
        pred_cols = [f"{t}_pred" for t in TARGET_COLUMNS]

        predictions_wide["overall_true"] = (
            predictions_wide[true_cols].astype(int).max(axis=1)
        )
        predictions_wide["overall_pred"] = (
            predictions_wide[pred_cols].astype(int).max(axis=1)
        )

        overall_accuracy = accuracy_score(
            predictions_wide["overall_true"],
            predictions_wide["overall_pred"]
        )

        logger.print("")
        logger.print(
            f"Overall fault-occurred accuracy: {overall_accuracy:.6f}"
        )

        # ----------------------------------------------------
        # SAVE ROW-LEVEL PREDICTIONS AND ROBUSTNESS SWEEP
        # ----------------------------------------------------

        predictions_wide.to_csv(
            PREDICTIONS_FILE,
            index=False
        )

        pd.DataFrame(robustness_records).to_csv(
            ROBUSTNESS_FILE,
            index=False
        )

        # ============================================================
        # SAVE MODEL MANIFEST AND SCALER
        # ============================================================
        # Keras models are saved individually per target (.keras
        # files above); this manifest ties target names to those
        # paths plus the shared scaler, so inference can rebuild
        # everything from one file.

        joblib.dump(
            {
                "model_paths": model_paths,
                "scaler": scaler,
            },
            MANIFEST_FILE
        )

        logger.print("")
        logger.print(
            f"Manifest saved to:"
        )

        logger.print(
            f"  {MANIFEST_FILE}"
        )

        # ----------------------------------------------------
        # SAVE LOSS DATA
        # ----------------------------------------------------

        loss_df = pd.DataFrame(
            loss_records
        )

        if LOSS_FILE.exists():

            loss_df.to_csv(
                LOSS_FILE,
                mode="a",
                header=False,
                index=False
            )

        else:

            loss_df.to_csv(
                LOSS_FILE,
                index=False
            )

        # ----------------------------------------------------
        # SAVE METRICS
        # ----------------------------------------------------

        save_metrics(
            metrics,
            METRICS_FILE
        )

        # ----------------------------------------------------
        # OVERALL
        # ----------------------------------------------------

        metrics_df = pd.DataFrame(
            metrics
        )

        logger.print("")
        logger.print(
            "=" * 70
        )

        logger.print(
            "OVERALL RESULTS"
        )

        logger.print(
            "=" * 70
        )

        logger.print(
            f"Mean Accuracy : "
            f"{metrics_df['accuracy'].mean():.6f}"
        )

        logger.print(
            f"Mean Precision: "
            f"{metrics_df['precision'].mean():.6f}"
        )

        logger.print(
            f"Mean F1       : "
            f"{metrics_df['f1_score'].mean():.6f}"
        )

        logger.print(
            f"Mean Latency  : "
            f"{metrics_df['latency_ms_per_sample'].mean():.6f} ms"
        )

        logger.print(
            f"Mean Robustness: "
            f"{metrics_df['robustness_accuracy'].mean():.6f}"
        )

        # ----------------------------------------------------
        # OUTPUT FILES
        # ----------------------------------------------------

        logger.print("")
        logger.print(
            "=" * 70
        )

        logger.print(
            "OUTPUT FILES"
        )

        logger.print(
            "=" * 70
        )

        logger.print(
            f"Report:"
        )

        logger.print(
            f"  {REPORT_FILE}"
        )

        logger.print(
            f"\nMetrics:"
        )

        logger.print(
            f"  {METRICS_FILE}"
        )

        logger.print(
            f"\nLoss history (per epoch):"
        )

        logger.print(
            f"  {LOSS_FILE}"
        )

        logger.print(
            f"\nPredictions:"
        )

        logger.print(
            f"  {PREDICTIONS_FILE}"
        )

        logger.print(
            f"\nRobustness sweep:"
        )

        logger.print(
            f"  {ROBUSTNESS_FILE}"
        )

        logger.print("")
        logger.print(
            "TRAINING COMPLETE"
        )

    finally:

        logger.close()


if __name__ == "__main__":
    main()