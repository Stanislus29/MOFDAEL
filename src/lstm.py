from __future__ import annotations

import os

# Bitwise-reproducible training runs (the traceability rule in
# AGENTS.md requires every number to be regenerable). TF_DETE-
# RMINISTIC_OPS forces deterministic op kernels (GPU); single-
# threaded execution removes oneDNN threaded fp reductions,
# which are the CPU source of run-to-run wobble. These must be
# set BEFORE tensorflow is imported.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from scenarios import (
    assign_scenario_ids,
    partition_scenario_ids,
    log_split_info,
)
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

# Explicit determinism (API form of the env vars above): fixes
# op kernels that still use nondeterministic reductions on CPU.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as exc:  # pragma: no cover - environmental
    print(f"WARNING: could not enable TF op determinism: {exc}")
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lstm"

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
# TIMESTAMP & OUTPUT FILES
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
    / "predictions_lstm.csv"
)

LATENCY_FILE = (
    OUTPUT_DIR
    / "latency_per_row_lstm.csv"
)

ROBUSTNESS_FILE = (
    OUTPUT_DIR
    / "robustness_lstm.csv"
)

MANIFEST_FILE = MODEL_DIR / f"lstm_manifest_{timestamp}.joblib"

# ============================================================
# DATA CONFIGURATION
# ============================================================

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

LSTM_UNITS = [64, 32]

LEARNING_RATE = 0.001

EPOCHS = 100

BATCH_SIZE = 16

VALIDATION_SPLIT = 0.1

EARLY_STOPPING_PATIENCE = 10

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
# GROUP ROWS INTO EVENTS
# ============================================================

def build_events(df):

    loc = df["Loc"].to_numpy()

    new_event = np.ones(len(loc), dtype=bool)
    new_event[1:] = loc[1:] <= loc[:-1]

    event_id = np.cumsum(new_event) - 1

    events = []

    for _, group in df.groupby(event_id):

        group = group.sort_values("Loc")

        events.append(group)

    return events


# ============================================================
# BUILD PADDED, WEIGHTED BATCH FOR ONE TARGET
# ============================================================

def build_batch(events, indices, scaler, target, class_weight_map):

    X_list = [
        scaler.transform(events[i][FEATURE_COLUMNS].to_numpy())
        for i in indices
    ]

    y_list = [
        events[i][target].to_numpy()
        for i in indices
    ]

    lengths = [len(events[i]) for i in indices]

    max_len = max(lengths)

    X_padded = pad_sequences(
        X_list,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    y_padded = pad_sequences(
        y_list,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    sample_weight = np.zeros((len(indices), max_len), dtype="float32")

    for row, length in enumerate(lengths):

        labels = y_padded[row, :length].astype(int)

        sample_weight[row, :length] = [
            class_weight_map[label] for label in labels
        ]

    return X_padded, y_padded[..., np.newaxis], sample_weight, lengths


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def create_lstm(num_features):

    inputs = layers.Input(shape=(None, num_features))

    x = layers.Masking(mask_value=0.0)(inputs)

    for units in LSTM_UNITS:

        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True)
        )(x)

    outputs = layers.TimeDistributed(
        layers.Dense(1, activation="sigmoid")
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# TRAIN ONE FAULT CLASSIFIER
# ============================================================

def train_fault_model(
    events,
    train_idx,
    scaler,
    target,
    logger
):

    logger.print("")
    logger.print("=" * 70)

    logger.print(
        f"TRAINING LSTM: {target}"
    )

    logger.print("=" * 70)

    logger.print(
        f"Units          : {LSTM_UNITS} (bidirectional)"
    )

    logger.print(
        f"Learning rate  : {LEARNING_RATE}"
    )

    logger.print(
        f"Max epochs     : {EPOCHS}"
    )

    logger.print(
        f"Batch size     : {BATCH_SIZE} events"
    )

    train_labels = np.concatenate([
        events[i][target].to_numpy() for i in train_idx
    ])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels
    )

    class_weight_map = {
        0: class_weights[0],
        1: class_weights[1],
    }

    X_train, y_train, sample_weight, _ = build_batch(
        events, train_idx, scaler, target, class_weight_map
    )

    model = create_lstm(len(FEATURE_COLUMNS))

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
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
        f"Events trained on: {len(train_idx)}"
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
# PREDICT AND UNPACK BACK TO ORIGINAL ROWS
# ============================================================

def predict_events(model, events, indices, scaler):

    X_list = [
        scaler.transform(events[i][FEATURE_COLUMNS].to_numpy())
        for i in indices
    ]

    lengths = [len(events[i]) for i in indices]

    max_len = max(lengths)

    X_padded = pad_sequences(
        X_list,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    probabilities = model.predict(
        X_padded,
        verbose=0
    )

    row_ids = []
    flat_probs = []

    for row, (i, length) in enumerate(zip(indices, lengths)):

        row_ids.extend(events[i].index.to_numpy())
        flat_probs.extend(probabilities[row, :length, 0])

    flat_probs = np.array(flat_probs)

    return (
        np.array(row_ids),
        flat_probs,
        (flat_probs > 0.5).astype(int),
    )


# ============================================================
# FAULT LOCATION HEAD (TIME-DISTRIBUTED REGRESSION)
# ============================================================

def build_batch_loc(events, indices, scaler):

    X_list = [
        scaler.transform(events[i][FEATURE_COLUMNS].to_numpy())
        for i in indices
    ]

    y_list = [
        events[i]["Loc"].to_numpy(dtype=np.float64)
        / LINE_LENGTH_KM
        for i in indices
    ]

    lengths = [len(events[i]) for i in indices]

    max_len = max(lengths)

    X_padded = pad_sequences(
        X_list,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    y_padded = pad_sequences(
        y_list,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    # Every real (non-padded) timestep carries weight 1; padded
    # slots are masked out.
    sample_weight = np.zeros(
        (len(indices), max_len),
        dtype="float32"
    )

    for row, length in enumerate(lengths):

        sample_weight[row, :length] = 1.0

    return (
        X_padded,
        y_padded[..., np.newaxis],
        sample_weight,
        lengths,
    )


def create_lstm_location(num_features):

    # Location head: the same bidirectional sequence trunk with a
    # linear time-distributed output, MSE loss on the scaled
    # target. No class weighting for regression.

    inputs = layers.Input(shape=(None, num_features))

    x = layers.Masking(mask_value=0.0)(inputs)

    for units in LSTM_UNITS:

        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True)
        )(x)

    outputs = layers.TimeDistributed(
        layers.Dense(1, activation=None)
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_location_model(
    events,
    train_idx,
    scaler,
    logger
):

    logger.print("")
    logger.print("=" * 70)

    logger.print(
        "TRAINING LSTM: Loc"
    )

    logger.print("=" * 70)

    logger.print(
        f"Units          : {LSTM_UNITS} (bidirectional)"
    )

    logger.print(
        "Output         : time-distributed linear (km)"
    )

    logger.print(
        "Loss           : mse (target scaled by line length)"
    )

    logger.print(
        f"Learning rate  : {LEARNING_RATE}"
    )

    logger.print(
        f"Max epochs     : {EPOCHS}"
    )

    logger.print(
        f"Batch size     : {BATCH_SIZE} events"
    )

    X_train, y_train, sample_weight, _ = build_batch_loc(
        events, train_idx, scaler
    )

    model = create_lstm_location(len(FEATURE_COLUMNS))

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=0,
    )

    end = time.perf_counter()

    training_time = end - start

    best_epoch = int(
        np.argmin(history.history["val_loss"])
    )

    logger.print(
        f"\nTraining time: {training_time:.6f} s"
    )

    logger.print(
        f"Events trained on: {len(train_idx)}"
    )

    logger.print(
        f"Epochs run   : {len(history.history['loss'])}"
    )

    logger.print(
        f"Best epoch   : {best_epoch + 1}"
    )

    logger.print(
        f"Best train mse (scaled): "
        f"{history.history['loss'][best_epoch]:.8f}"
    )

    logger.print(
        f"Best val mse (scaled)  : "
        f"{history.history['val_loss'][best_epoch]:.8f}"
    )

    return model, training_time, history, best_epoch


def predict_events_loc(model, events, indices, scaler):

    X_list = [
        scaler.transform(events[i][FEATURE_COLUMNS].to_numpy())
        for i in indices
    ]

    lengths = [len(events[i]) for i in indices]

    max_len = max(lengths)

    X_padded = pad_sequences(
        X_list,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    output = model.predict(
        X_padded,
        verbose=0
    )

    row_ids = []
    flat_out = []

    for row, (i, length) in enumerate(zip(indices, lengths)):

        row_ids.extend(events[i].index.to_numpy())
        flat_out.extend(output[row, :length, 0])

    # Undo the scaling: network output is a fraction of the
    # line length; report in km.
    flat_out = np.array(flat_out) * LINE_LENGTH_KM

    return (
        np.array(row_ids),
        flat_out,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    predictions,
    probabilities,
    y_true,
    target,
    logger
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    test_loss = float(
        log_loss(
            y_true,
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
            y_true,
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
    }


# ============================================================
# SIMULATED STREAMING LATENCY
# ============================================================

def measure_streaming_latency(model, events, indices, scaler):

    row_ids = []
    row_latencies = []

    for i in indices:

        event_df = events[i]

        event_ids = event_df.index.to_numpy()

        scaled_features = scaler.transform(
            event_df[FEATURE_COLUMNS].to_numpy()
        )

        # Warmup for this event shape
        _ = model(
            scaled_features[:1][np.newaxis, ...],
            training=False
        )

        # Sequentially pass prefix sub-sequences (1..k) to simulate streaming arrival
        for k in range(1, len(event_df) + 1):

            prefix_input = scaled_features[:k][np.newaxis, ...]

            start = time.perf_counter()

            _ = model(prefix_input, training=False)

            end = time.perf_counter()

            row_ids.append(event_ids[k - 1])

            row_latencies.append((end - start) * 1000)  # ms

    total_latency_sec = sum(row_latencies) / 1000.0

    per_sample_ms = np.mean(row_latencies)

    return (
        np.array(row_ids),
        np.array(row_latencies),
        total_latency_sec,
        per_sample_ms,
    )


# ============================================================
# ROBUSTNESS
# ============================================================

def test_robustness(model, events, indices, scaler, target, noise_levels, trials):

    lengths = [len(events[i]) for i in indices]

    max_len = max(lengths)

    raw = [
        events[i][FEATURE_COLUMNS].to_numpy()
        for i in indices
    ]

    raw_padded = pad_sequences(
        raw,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    feature_std = np.std(
        raw_padded.reshape(-1, raw_padded.shape[-1]),
        axis=0
    )

    y_true = np.concatenate([
        events[i][target].to_numpy() for i in indices
    ])

    records = []

    for noise_level in noise_levels:

        accuracies = []
        f1s = []

        for trial in range(trials):

            rng = np.random.default_rng(RANDOM_STATE + trial)

            noise = rng.normal(
                0,
                noise_level * feature_std,
                size=raw_padded.shape
            )

            noisy_scaled = scaler.transform(
                (raw_padded + noise).reshape(-1, raw_padded.shape[-1])
            ).reshape(raw_padded.shape)

            probabilities = model.predict(noisy_scaled, verbose=0)

            flat_preds = []

            for row, length in enumerate(lengths):

                flat_preds.extend(
                    (probabilities[row, :length, 0] > 0.5).astype(int)
                )

            accuracies.append(accuracy_score(y_true, flat_preds))

            f1s.append(
                f1_score(y_true, flat_preds, labels=[0, 1], zero_division=0)
            )

        record = {
            "noise_level": noise_level,
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
        }

        for trial in range(trials):
            record[f"accuracy_trial_{trial}"] = accuracies[trial]

            record[f"f1_trial_{trial}"] = f1s[trial]

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
            "LSTM OVER LOCATION SEQUENCES"
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
        # GROUP INTO EVENTS
        # ----------------------------------------------------

        events = build_events(df)

        lengths = [len(event) for event in events]

        logger.print("")

        logger.print(
            f"Events detected: {len(events)}"
        )

        logger.print(
            f"Locations per event: "
            f"min {min(lengths)}, "
            f"mean {np.mean(lengths):.1f}, "
            f"max {max(lengths)}"
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
                df[target]
                .value_counts()
                .sort_index()
            )

            for value, count in counts.items():

                percentage = (
                    count / len(df)
                ) * 100

                logger.print(
                    f"  {value}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

        # ----------------------------------------------------
        # SCENARIO-LEVEL TRAIN / TEST SPLIT (Section 3.8.2, R9)
        # ----------------------------------------------------
        # Events map onto scenarios: each fault event is one
        # 8-row location sweep of a simulation run; no-fault
        # singleton events belong to the 8-row observation
        # windows of the continuous no-fault recording. Events
        # are assigned to train/test by the shared scenario
        # partition used by every training script.

        scenario_ids = assign_scenario_ids(df)

        train_pos, test_pos, split_info = (
            partition_scenario_ids(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
        )

        train_scenario_ids = set(
            scenario_ids[train_pos].tolist()
        )

        test_scenario_ids = set(
            scenario_ids[test_pos].tolist()
        )

        event_scenario = np.array([
            scenario_ids[df.index.get_loc(group.index[0])]
            for group in events
        ])

        train_idx = np.where(
            np.isin(event_scenario, list(train_scenario_ids))
        )[0]

        test_idx = np.where(
            np.isin(event_scenario, list(test_scenario_ids))
        )[0]

        logger.print("")

        logger.print(
            f"Training events: {len(train_idx)}"
        )

        logger.print(
            f"Testing events : {len(test_idx)}"
        )

        log_split_info(logger, split_info)

        # ----------------------------------------------------
        # SCALE FEATURES
        # ----------------------------------------------------

        train_rows = np.vstack([
            events[i][FEATURE_COLUMNS].to_numpy() for i in train_idx
        ])

        scaler = StandardScaler().fit(train_rows)

        # ----------------------------------------------------
        # TRAIN MODELS & EVALUATE
        # ----------------------------------------------------

        models_dict = {}

        model_paths = {}

        metrics = []

        loss_records = []

        robustness_records = []

        predictions_wide = None

        latency_wide = None

        for target in TARGET_COLUMNS:

            model, training_time, history, best_epoch = (
                train_fault_model(
                    events,
                    train_idx,
                    scaler,
                    target,
                    logger
                )
            )

            models_dict[target] = model

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
            # EVALUATE PREDICTIONS
            # ------------------------------------------------

            row_ids, probabilities, predictions = predict_events(
                model, events, test_idx, scaler
            )

            y_true = df.loc[row_ids, target].to_numpy()

            results = evaluate_model(
                predictions,
                probabilities,
                y_true,
                target,
                logger
            )

            target_df = pd.DataFrame({
                "row_id": row_ids,
                f"{target}_true": y_true,
                f"{target}_pred": predictions,
            })

            if predictions_wide is None:

                predictions_wide = target_df

            else:

                predictions_wide = predictions_wide.merge(
                    target_df, on="row_id"
                )

            # ------------------------------------------------
            # STREAMING LATENCY PER ROW
            # ------------------------------------------------

            lat_ids, row_latencies, total_latency, per_sample_latency = (
                measure_streaming_latency(
                    model, events, test_idx, scaler
                )
            )

            logger.print(
                f"Total streaming latency: "
                f"{total_latency * 1000:.6f} ms"
            )

            logger.print(
                f"Latency/sample (streaming): "
                f"{per_sample_latency:.6f} ms"
            )

            target_lat_df = pd.DataFrame({
                "row_id": lat_ids,
                f"{target}_latency_ms": row_latencies,
            })

            if latency_wide is None:

                latency_wide = target_lat_df

            else:

                latency_wide = latency_wide.merge(
                    target_lat_df, on="row_id"
                )

            # ------------------------------------------------
            # ROBUSTNESS
            # ------------------------------------------------

            robustness_sweep = test_robustness(
                model,
                events,
                test_idx,
                scaler,
                target,
                ROBUSTNESS_NOISE_LEVELS,
                ROBUSTNESS_TRIALS,
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
                / f"lstm_model_{timestamp}_{target}.keras"
            )

            model.save(model_path)

            model_paths[target] = str(model_path)

            metrics.append({

                "timestamp": timestamp,

                "target": target,

                "lstm_units": str(LSTM_UNITS),

                "learning_rate": LEARNING_RATE,

                "num_train_events": len(train_idx),

                "num_test_events": len(test_idx),

                "epochs_run": len(history.history["loss"]),

                "best_epoch": best_epoch + 1,

                "accuracy": results["accuracy"],

                "precision": results["precision"],

                "recall": results["recall"],

                "f1_score": results["f1"],

                "training_time_seconds": training_time,

                "test_log_loss": results["test_loss"],

                "latency_ms_total": total_latency * 1000,

                "latency_ms_per_sample": per_sample_latency,

                "robustness_accuracy": reference["accuracy_mean"],

                "robustness_f1": reference["f1_mean"],

                "robustness_retention": retention,
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
        # FAULT LOCATION (TIME-DISTRIBUTED REGRESSION HEAD)
        # ----------------------------------------------------
        # Fifth task of this script: the regression variant of the
        # same sequence architecture. In this dataset an event is a
        # sweep over the fault locations (Loc = 15..120 km along
        # the time axis), so the natural location head is time-
        # distributed: one location value per timestep, MSE loss
        # on the scaled target. Predictions are unpacked back to
        # rows the same way as the per-line flags, so the output
        # rows align with the per-line predictions above.
        #
        # CAVEAT (repeated in the report): because every event
        # sweeps Loc in ascending order, the timestep POSITION
        # alone pins the distance; a position prior can drive the
        # location error towards zero without using the
        # instantaneous samples. The reported LSTM location error
        # therefore reflects this dataset representation and is
        # NOT directly comparable to the per-row models (knn, dtr,
        # svm, ann, rnn). See the LOCATION ACCURACY caveat below
        # and TODO.md R3.
        #
        # Errors feed H5 in src/hypothesis.py, reported in km and
        # as a percentage of the protected line length
        # (LINE_LENGTH_KM, stated explicitly).

        # Fault events only: single no-fault events (Loc = 0) have
        # no fault location to estimate.

        location_train_idx = [
            i for i in train_idx
            if events[i]["Loc"].max() > 0
        ]
        location_test_idx = [
            i for i in test_idx
            if events[i]["Loc"].max() > 0
        ]

        location_model, location_training_time, \
            location_history, location_best_epoch = \
            train_location_model(
                events,
                location_train_idx,
                scaler,
                logger
            )

        models_dict["Loc"] = location_model

        loc_row_ids, loc_pred = predict_events_loc(
            location_model, events, location_test_idx, scaler
        )

        loc_true = df.loc[loc_row_ids, "Loc"].to_numpy(
            dtype=float
        )
        loc_error = np.abs(loc_pred - loc_true)

        mae_km = float(mean_absolute_error(loc_true, loc_pred))
        rmse_km = float(
            np.sqrt(mean_squared_error(loc_true, loc_pred))
        )
        mae_pct = mae_km / LINE_LENGTH_KM * 100

        logger.print(
            f"Test fault rows: {len(loc_row_ids)}"
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

        logger.print("")
        logger.print(
            "LOCATION ACCURACY CAVEAT:"
        )
        logger.print(
            "  Every training event sweeps Loc in ascending order"
        )
        logger.print(
            "  (timestep k = distance k), so the timestep position"
        )
        logger.print(
            "  alone pins the location. The LSTM head exploits this"
        )
        logger.print(
            "  position prior; its error does not measure location"
        )
        logger.print(
            "  estimation from a single instantaneous sample and is"
        )
        logger.print(
            "  NOT directly comparable to the per-row models."
        )
        logger.print(
            "  Interpret with this limitation."
        )

        model_path = (
            MODEL_DIR
            / f"lstm_model_{timestamp}_Loc.keras"
        )

        location_model.save(model_path)

        model_paths["Loc"] = str(model_path)

        metrics.append({

            "timestamp": timestamp,

            "target": "Loc",

            "lstm_units": str(LSTM_UNITS),

            "learning_rate": LEARNING_RATE,

            "num_train_events": len(location_train_idx),

            "mae_km": mae_km,

            "mae_pct_line": mae_pct,

            "rmse_km": rmse_km,

            "training_time_seconds": location_training_time,

            "epochs_run": len(location_history.history["loss"]),

            "best_epoch": location_best_epoch + 1,
        })

        loc_df = pd.DataFrame({
            "row_id": loc_row_ids,
            "Loc_true": loc_true,
            "Loc_pred": loc_pred,
        })

        predictions_wide = predictions_wide.merge(
            loc_df, on="row_id", how="left"
        )

        # Aggregate total latency across targets per row
        lat_cols = [f"{t}_latency_ms" for t in TARGET_COLUMNS]

        latency_wide["total_latency_ms"] = latency_wide[lat_cols].sum(axis=1)

        # ----------------------------------------------------
        # EXPORT OUTPUT CSVs
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
        # OVERALL SUMMARY
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
            f"\nLatency per row:"
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