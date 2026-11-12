from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from scenarios import group_partition, log_split_info
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
    mean_absolute_error,
    mean_squared_error,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pinn"

DATA_FILE = DATA_DIR / "classdetectfault_sim.csv"

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
    / "predictions_pinn.csv"
)

LATENCY_FILE = (
    OUTPUT_DIR
    / "latency_per_row_pinn.csv"
)

ROBUSTNESS_FILE = (
    OUTPUT_DIR
    / "robustness_pinn.csv"
)

MANIFEST_FILE = MODEL_DIR / f"pinn_manifest_{timestamp}.joblib"

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

# Indices of Ia, Ib, Ic within FEATURE_COLUMNS, used to build the
# physics residual (Kirchhoff current law: Ia + Ib + Ic == 0 for a
# balanced/no-fault system, and departs from zero under fault).
CURRENT_INDICES = [0, 1, 2]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

PINN_HIDDEN_UNITS = [64, 64, 32]

PINN_DROPOUT = 0.3

PINN_LEARNING_RATE = 0.001

PINN_EPOCHS = 100

PINN_BATCH_SIZE = 32

PINN_VALIDATION_SPLIT = 0.1

PINN_EARLY_STOPPING_PATIENCE = 10

# Weight of the physics residual term relative to the data loss.
PINN_PHYSICS_WEIGHT = 0.1

# Fault-location percentage-error denominator. Percentage errors
# are reported as a fraction of the full protected line length,
# stated explicitly here and in every output. Regression targets
# are scaled to [0, 1] by this constant before training.
LINE_LENGTH_KM = 120.0

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
# BUILD PINN
# ============================================================
# A plain feedforward classifier plus a physics residual computed
# from the raw (unscaled) currents inside the model itself, so the
# residual is in physically meaningful units regardless of how the
# classification branch is scaled.

class FaultPINN(models.Model):

    def __init__(self, input_dim, current_mean, current_std):

        super().__init__()

        self.current_mean = tf.constant(
            current_mean,
            dtype=tf.float32
        )

        self.current_std = tf.constant(
            current_std,
            dtype=tf.float32
        )

        self.hidden_layers = []

        for units in PINN_HIDDEN_UNITS:

            self.hidden_layers.append(
                layers.Dense(units, activation="tanh")
            )

        self.dropout = layers.Dropout(PINN_DROPOUT)

        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):

        x = inputs

        for layer in self.hidden_layers:

            x = layer(x)
            x = self.dropout(x, training=training)

        return self.output_layer(x)

    def physics_residual(self, X_raw):

        currents = tf.gather(
            X_raw,
            CURRENT_INDICES,
            axis=1
        )

        currents_unscaled = (
            currents * self.current_std
            + self.current_mean
        )

        current_sum = tf.reduce_sum(
            currents_unscaled,
            axis=1
        )

        return current_sum


def create_pinn(input_dim, current_mean, current_std):

    model = FaultPINN(
        input_dim,
        current_mean,
        current_std
    )

    dummy_input = tf.zeros((1, input_dim))
    model(dummy_input, training=False)

    return model


# ============================================================
# PREDICTION HELPER
# ============================================================

def make_predict_fn(model, scaler):

    def predict_fn(X_raw):

        X_scaled = scaler.transform(X_raw)

        probabilities = model(
            X_scaled,
            training=False
        ).numpy().flatten()

        return (probabilities > 0.5).astype(int), probabilities

    return predict_fn


# ============================================================
# FAULT LOCATION HEAD (PHYSICS-INFORMED REGRESSION)
# ============================================================

class LocationPINN(FaultPINN):

    """Regression variant of FaultPINN for fault location. Same
    trunk and physics residual; linear output trained with MSE on
    the scaled target. The physics residual (sum of unscaled
    currents vanishes only in the unfaulted state) is retained
    with the same PINN_PHYSICS_WEIGHT so the head stays a member
    of the physics-informed family."""

    def __init__(self, input_dim, current_mean, current_std):

        super().__init__(
            input_dim,
            current_mean,
            current_std
        )

        # Replace the classification head with a linear one.
        self.output_layer = layers.Dense(1, activation=None)


def create_pinn_location(input_dim, current_mean, current_std):

    model = LocationPINN(
        input_dim,
        current_mean,
        current_std
    )

    dummy_input = tf.zeros((1, input_dim))
    model(dummy_input, training=False)

    return model


def make_location_predict_fn(model, scaler):

    def predict_fn(X_raw):

        X_scaled = scaler.transform(X_raw)

        output = model(
            X_scaled,
            training=False
        ).numpy().flatten()

        # Undo the scaling: network output is a fraction of the
        # line length; report in km.
        return output * LINE_LENGTH_KM

    return predict_fn


# ============================================================
# TRAIN ONE FAULT CLASSIFIER
# ============================================================

def train_fault_model(
    X_train_scaled,
    y_train,
    target,
    current_mean,
    current_std,
    logger
):

    logger.print("")
    logger.print("=" * 70)

    logger.print(
        f"TRAINING PINN: {target}"
    )

    logger.print("=" * 70)

    logger.print(
        f"Hidden units   : {PINN_HIDDEN_UNITS}"
    )

    logger.print(
        f"Physics weight : {PINN_PHYSICS_WEIGHT}"
    )

    logger.print(
        f"Learning rate  : {PINN_LEARNING_RATE}"
    )

    logger.print(
        f"Max epochs     : {PINN_EPOCHS}"
    )

    logger.print(
        f"Batch size     : {PINN_BATCH_SIZE}"
    )

    model = create_pinn(
        X_train_scaled.shape[1],
        current_mean,
        current_std
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

    n_samples = X_train_scaled.shape[0]

    n_val = int(n_samples * PINN_VALIDATION_SPLIT)

    X_fit, X_val = X_train_scaled[:-n_val], X_train_scaled[-n_val:]
    y_fit, y_val = y_train[:-n_val], y_train[-n_val:]

    sample_weight_fit = np.array(
        [class_weight_dict[label] for label in y_fit]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=PINN_LEARNING_RATE
    )

    bce = tf.keras.losses.BinaryCrossentropy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (X_fit, y_fit, sample_weight_fit)
    )

    dataset = dataset.shuffle(
        buffer_size=n_samples,
        seed=RANDOM_STATE
    ).batch(PINN_BATCH_SIZE)

    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0

    start = time.perf_counter()

    for epoch in range(PINN_EPOCHS):

        epoch_data_loss = []
        epoch_physics_loss = []
        epoch_acc = []

        for X_batch, y_batch, w_batch in dataset:

            with tf.GradientTape() as tape:

                y_pred = model(X_batch, training=True)
                y_pred = tf.reshape(y_pred, [-1])

                data_loss = bce(
                    y_batch,
                    y_pred,
                    sample_weight=w_batch
                )

                residual = model.physics_residual(X_batch)
                physics_loss = tf.reduce_mean(tf.square(residual))

                total_loss = (
                    data_loss
                    + PINN_PHYSICS_WEIGHT * physics_loss
                )

            gradients = tape.gradient(
                total_loss,
                model.trainable_variables
            )

            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

            epoch_data_loss.append(float(data_loss))
            epoch_physics_loss.append(float(physics_loss))

            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.cast(y_pred > 0.5, tf.int32),
                        tf.cast(y_batch, tf.int32)
                    ),
                    tf.float32
                )
            )

            epoch_acc.append(float(acc))

        val_pred = model(X_val, training=False)
        val_pred = tf.reshape(val_pred, [-1])

        val_loss = float(bce(y_val, val_pred))

        val_acc = float(
            tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.cast(val_pred > 0.5, tf.int32),
                        tf.cast(y_val, tf.int32)
                    ),
                    tf.float32
                )
            )
        )

        train_loss = float(np.mean(epoch_data_loss))
        train_acc = float(np.mean(epoch_acc))

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_weights = [w.numpy() for w in model.trainable_variables]
            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PINN_EARLY_STOPPING_PATIENCE:
                break

    if best_weights is not None:

        for var, weight in zip(model.trainable_variables, best_weights):
            var.assign(weight)

    end = time.perf_counter()

    training_time = end - start

    best_epoch = int(
        np.argmin(history["val_loss"])
    )

    logger.print(
        f"\nTraining time: "
        f"{training_time:.6f} s"
    )

    logger.print(
        f"Epochs run   : "
        f"{len(history['loss'])}"
    )

    logger.print(
        f"Best epoch   : "
        f"{best_epoch + 1}"
    )

    logger.print(
        f"Best train loss: "
        f"{history['loss'][best_epoch]:.6f}"
    )

    logger.print(
        f"Best val loss  : "
        f"{history['val_loss'][best_epoch]:.6f}"
    )

    return model, training_time, history, best_epoch


def train_location_model(
    X_loc_scaled,
    y_loc_scaled,
    current_mean,
    current_std,
    logger
):
    """Train the fault-location regression head (fault rows
    only; physics-informed trunk, linear output, MSE on the
    scaled target)."""

    logger.print("")
    logger.print("=" * 70)

    logger.print(
        "TRAINING PINN: Loc"
    )

    logger.print("=" * 70)

    logger.print(
        f"Hidden units   : {PINN_HIDDEN_UNITS}"
    )

    logger.print(
        "Output         : linear (location, km)"
    )

    logger.print(
        "Loss           : mse + physics residual "
        "(target scaled by line length)"
    )

    logger.print(
        f"Physics weight : {PINN_PHYSICS_WEIGHT}"
    )

    logger.print(
        f"Learning rate  : {PINN_LEARNING_RATE}"
    )

    logger.print(
        f"Max epochs     : {PINN_EPOCHS}"
    )

    logger.print(
        f"Batch size     : {PINN_BATCH_SIZE}"
    )

    model = create_pinn_location(
        X_loc_scaled.shape[1],
        current_mean,
        current_std
    )

    # Shuffle training rows before the tail validation split;
    # the index order groups locations, so a plain tail split
    # would make the validation strata non-representative.
    rng = np.random.default_rng(RANDOM_STATE)

    order = rng.permutation(len(X_loc_scaled))

    X_loc_scaled = X_loc_scaled[order]
    y_loc_scaled = y_loc_scaled[order]

    n_val = int(
        X_loc_scaled.shape[0] * PINN_VALIDATION_SPLIT
    )

    X_fit, X_val = (
        X_loc_scaled[:-n_val],
        X_loc_scaled[-n_val:],
    )
    y_fit, y_val = (
        y_loc_scaled[:-n_val],
        y_loc_scaled[-n_val:],
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=PINN_LEARNING_RATE
    )

    mse = tf.keras.losses.MeanSquaredError()

    dataset = tf.data.Dataset.from_tensor_slices(
        (X_fit, y_fit)
    )

    dataset = dataset.shuffle(
        buffer_size=len(X_fit),
        seed=RANDOM_STATE
    ).batch(PINN_BATCH_SIZE)

    history = {
        "loss": [],
        "val_loss": [],
    }

    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0

    start = time.perf_counter()

    for epoch in range(PINN_EPOCHS):

        epoch_data_loss = []
        epoch_physics_loss = []

        for X_batch, y_batch in dataset:

            with tf.GradientTape() as tape:

                y_pred = model(X_batch, training=True)
                y_pred = tf.reshape(y_pred, [-1])

                data_loss = mse(y_batch, y_pred)

                residual = model.physics_residual(X_batch)
                physics_loss = tf.reduce_mean(tf.square(residual))

                total_loss = (
                    data_loss
                    + PINN_PHYSICS_WEIGHT * physics_loss
                )

            gradients = tape.gradient(
                total_loss,
                model.trainable_variables
            )

            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

            epoch_data_loss.append(float(data_loss))
            epoch_physics_loss.append(float(physics_loss))

        val_pred = model(X_val, training=False)
        val_pred = tf.reshape(val_pred, [-1])

        val_loss = float(mse(y_val, val_pred))

        train_loss = float(np.mean(epoch_data_loss))

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_weights = [w.numpy() for w in model.trainable_variables]
            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PINN_EARLY_STOPPING_PATIENCE:
                break

    if best_weights is not None:

        for var, weight in zip(model.trainable_variables, best_weights):
            var.assign(weight)

    end = time.perf_counter()

    training_time = end - start

    best_epoch = int(
        np.argmin(history["val_loss"])
    )

    logger.print(
        f"\nTraining time: {training_time:.6f} s"
    )

    logger.print(
        f"Epochs run   : {len(history['loss'])}"
    )

    logger.print(
        f"Best epoch   : {best_epoch + 1}"
    )

    logger.print(
        f"Best train mse (scaled): "
        f"{history['loss'][best_epoch]:.8f}"
    )

    logger.print(
        f"Best val mse (scaled)  : "
        f"{history['val_loss'][best_epoch]:.8f}"
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

    latencies = []

    for i in range(len(X_test)):

        row = X_test.iloc[[i]]

        start = time.perf_counter()

        predict_fn(row)

        end = time.perf_counter()

        latencies.append(end - start)

    total = sum(latencies)

    per_sample = (
        total / len(latencies)
    )

    return total, per_sample, latencies


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

        record = {"noise_level": noise_level}

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

            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, labels=[0, 1], zero_division=0)

            accuracies.append(acc)
            f1s.append(f1)

            record[f"accuracy_trial_{trial}"] = acc
            record[f"f1_trial_{trial}"] = f1

        record["accuracy_mean"] = np.mean(accuracies)
        record["accuracy_std"] = np.std(accuracies)
        record["f1_mean"] = np.mean(f1s)
        record["f1_std"] = np.std(f1s)

        records.append(record)

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
            "PHYSICS-INFORMED NEURAL NETWORK"
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

        # Loc is the fault-location regression target (fault rows
        # only) for the location head; it is never a classifier
        # feature.

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
            "ADDITIONAL TARGET:"
        )

        logger.print(
            "  Loc (fault location, km; regression head,"
            " fault rows only)"
        )

        logger.print("")
        logger.print(
            "TARGETS:"
        )

        for target in TARGET_COLUMNS:

            logger.print(
                f"  {target}"
            )

        logger.print("")
        logger.print(
            "PHYSICS CONSTRAINT:"
        )

        logger.print(
            "  Kirchhoff current law: Ia + Ib + Ic -> 0"
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

        train_pos, test_pos, split_info = group_partition(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]

        test_row_id = pd.Series(X_test.index, name="row_id").reset_index(drop=True)

        log_split_info(logger, split_info)

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

        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        # Mean/std of the current columns only, needed inside the
        # model to unscale currents before computing the physics
        # residual in physical units.

        current_mean = scaler.mean_[CURRENT_INDICES].astype(np.float32)
        current_std = scaler.scale_[CURRENT_INDICES].astype(np.float32)

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

        latency_wide = pd.DataFrame({
            "row_id": test_row_id
        })

        for target in TARGET_COLUMNS:

            model, training_time, history, best_epoch = (
                train_fault_model(
                    X_train_scaled,
                    y_train[target].to_numpy(),
                    target,
                    current_mean,
                    current_std,
                    logger
                )
            )

            models_dict[target] = model

            predict_fn = make_predict_fn(model, scaler)

            for epoch in range(len(history["loss"])):

                loss_records.append({
                    "timestamp": timestamp,
                    "target": target,
                    "epoch": epoch + 1,
                    "train_loss": history["loss"][epoch],
                    "val_loss": history["val_loss"][epoch],
                    "train_accuracy": history["accuracy"][epoch],
                    "val_accuracy": history["val_accuracy"][epoch],
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
            per_sample_latency, \
            row_latencies = (
                measure_latency(
                    predict_fn,
                    X_test
                )
            )

            latency_wide[f"{target}_latency"] = row_latencies

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
                / f"pinn_model_{timestamp}_{target}"
            )

            model.save_weights(
                str(model_path) + ".weights.h5"
            )

            model_paths[target] = str(model_path) + ".weights.h5"

            metrics.append({

                "timestamp": timestamp,

                "target": target,

                "hidden_units": str(PINN_HIDDEN_UNITS),

                "physics_weight": PINN_PHYSICS_WEIGHT,

                "learning_rate": PINN_LEARNING_RATE,

                "epochs_run": len(history["loss"]),

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
        # FAULT LOCATION (PHYSICS-INFORMED REGRESSION HEAD)
        # ----------------------------------------------------
        # Fifth task of this script: the regression variant of the
        # same physics-informed trunk on fault rows only (Loc > 0).
        # The physics residual term is kept with the same
        # PINN_PHYSICS_WEIGHT so the head remains a member of the
        # physics-informed family. No-fault rows have no fault
        # location to estimate, so their Loc_pred is left empty.
        # Errors feed H5 in src/hypothesis.py, reported in km and
        # as a percentage of the protected line length
        # (LINE_LENGTH_KM, stated explicitly).

        fault_mask = (
            df[TARGET_COLUMNS].sum(axis=1) > 0
        ) & (df["Loc"] > 0)

        fault_idx_all = df.index[fault_mask]

        loc_test_idx = X_test.index.intersection(
            fault_idx_all
        )
        loc_train_idx = fault_idx_all.difference(
            X_test.index
        )

        location_scaler = StandardScaler()
        X_loc_train_scaled = location_scaler.fit_transform(
            X.loc[loc_train_idx]
        ).astype(np.float32)

        loc_current_mean = (
            location_scaler.mean_[CURRENT_INDICES]
            .astype(np.float32)
        )
        loc_current_std = (
            location_scaler.scale_[CURRENT_INDICES]
            .astype(np.float32)
        )

        location_model, location_training_time, \
            location_history, location_best_epoch = \
            train_location_model(
                X_loc_train_scaled,
                (
                    df.loc[loc_train_idx, "Loc"]
                    .to_numpy(dtype=np.float64)
                    / LINE_LENGTH_KM
                ).astype(np.float32),
                loc_current_mean,
                loc_current_std,
                logger
            )

        models_dict["Loc"] = location_model

        location_predict_fn = make_location_predict_fn(
            location_model,
            location_scaler
        )

        loc_pred = location_predict_fn(
            X.loc[loc_test_idx]
        )
        loc_true = df.loc[loc_test_idx, "Loc"].to_numpy(dtype=float)
        loc_error = np.abs(loc_pred - loc_true)

        mae_km = float(mean_absolute_error(loc_true, loc_pred))
        rmse_km = float(
            np.sqrt(mean_squared_error(loc_true, loc_pred))
        )
        mae_pct = mae_km / LINE_LENGTH_KM * 100

        logger.print(
            f"Test fault rows: {len(loc_test_idx)}"
        )
        logger.print(
            f"MAE : {mae_km:.4f} km "
            f"({mae_pct:.4f}% of {LINE_LENGTH_KM:.0f} km line)"
        )
        logger.print(
            f"RMSE: {rmse_km:.4f} km"
        )

        per_location_mae = (
            pd.DataFrame({
                "Loc": loc_true,
                "error": loc_error,
            })
            .groupby("Loc")["error"]
            .mean()
        )

        logger.print("\nMAE by true location (km):")

        for location_value, error_mean in per_location_mae.items():
            logger.print(
                f"  {int(location_value):>3} km: {error_mean:.4f}"
            )

        model_path = (
            MODEL_DIR
            / f"pinn_model_{timestamp}_Loc"
        )
        location_model.save_weights(
            str(model_path) + ".weights.h5"
        )
        model_paths["Loc"] = str(model_path) + ".weights.h5"

        metrics.append({

            "timestamp": timestamp,

            "target": "Loc",

            "hidden_units": str(PINN_HIDDEN_UNITS),

            "physics_weight": PINN_PHYSICS_WEIGHT,

            "learning_rate": PINN_LEARNING_RATE,

            "epochs_run": len(location_history["loss"]),

            "best_epoch": location_best_epoch + 1,

            "mae_km": mae_km,

            "mae_pct_line": mae_pct,

            "rmse_km": rmse_km,

            "training_time_seconds": location_training_time,
        })

        # Export-aligned columns. Rows follow X_test.index order,
        # matching the per-line prediction columns; values are
        # empty (NaN) on no-fault test rows.
        test_fault_mask = fault_mask.loc[X_test.index]

        loc_true_test = df.loc[X_test.index, "Loc"].where(
            test_fault_mask
        )

        loc_pred_test = pd.Series(
            np.nan,
            index=X_test.index
        )
        loc_pred_test.loc[loc_test_idx] = loc_pred

        predictions_wide["Loc_true"] = loc_true_test.to_numpy()
        predictions_wide["Loc_pred"] = loc_pred_test.to_numpy()

        # ----------------------------------------------------
        # SAVE ROW-LEVEL PREDICTIONS AND ROBUSTNESS SWEEP
        # ----------------------------------------------------

        predictions_wide.to_csv(
            PREDICTIONS_FILE,
            index=False
        )

        latency_wide.to_csv(
            LATENCY_FILE,
            index=False
        )

        pd.DataFrame(robustness_records).to_csv(
            ROBUSTNESS_FILE,
            index=False
        )

        # ============================================================
        # SAVE MODEL MANIFEST AND SCALER
        # ============================================================
        # Weights are saved individually per target (.weights.h5
        # files above); this manifest ties target names to those
        # paths plus the shared scaler and architecture config, so
        # inference can rebuild everything from one file.

        joblib.dump(
            {
                "model_paths": model_paths,
                "scaler": scaler,
                "hidden_units": PINN_HIDDEN_UNITS,
                "current_indices": CURRENT_INDICES,
                "current_mean": current_mean,
                "current_std": current_std,
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
            f"\nPer-row latency:"
        )

        logger.print(
            f"  {LATENCY_FILE}"
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