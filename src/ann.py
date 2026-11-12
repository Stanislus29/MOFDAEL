from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scenarios import group_partition, log_split_info
from sklearn.preprocessing import StandardScaler
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

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ann"

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
    / "predictions_ann.csv"
)

LATENCY_FILE = (
    OUTPUT_DIR
    / "latency_per_row_ann.csv"
)

ROBUSTNESS_FILE = (
    OUTPUT_DIR
    / "robustness_ann.csv"
)

MODEL_FILE = MODEL_DIR / f"ann_model_{timestamp}.pt"

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
# HYPERPARAMETERS & CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

ANN_HIDDEN_LAYERS = [64, 32]
ANN_DROPOUT = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 30

# Fault-location percentage-error denominator. Percentage errors
# are reported as a fraction of the full protected line length,
# stated explicitly here and in every output. Regression targets
# are scaled to [0, 1] by this constant before training.
LINE_LENGTH_KM = 120.0

ROBUSTNESS_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2]
ROBUSTNESS_TRIALS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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
# ANN MODULE & WRAPPER
# ============================================================

class ANNModel(nn.Module):
    """Feedforward Multi-Layer Perceptron for binary classification."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_layers: list[int] = [64, 32],
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        in_dim = input_size

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANNClassifierWrapper:
    """Scikit-learn compatible wrapper for the PyTorch ANN model."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_layers: list[int] = [64, 32],
        dropout: float = 0.1,
        lr: float = 0.001,
        epochs: int = 30,
        batch_size: int = 64,
        device: torch.device = DEVICE
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.scaler = StandardScaler()
        self.model = ANNModel(
            input_size=input_size,
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(self.device)

    def _prepare_tensor(self, X: np.ndarray | pd.DataFrame) -> torch.Tensor:
        if isinstance(X, np.ndarray) and hasattr(self.scaler, "feature_names_in_"):
            X = pd.DataFrame(X, columns=self.scaler.feature_names_in_)

        X_scaled = self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> list[float]:
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        loss_history = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            total_epoch_loss = epoch_loss / len(dataset)
            loss_history.append(total_epoch_loss)

        return loss_history

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = self._prepare_tensor(X).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs_pos = torch.sigmoid(logits).cpu().numpy().flatten()

        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class ANNRegressorWrapper(ANNClassifierWrapper):
    """Regression variant of the classifier wrapper, used as the
    fault-location head. Same architecture and preprocessing; the
    linear network output is used directly and trained with MSE
    on the scaled target (km / LINE_LENGTH_KM)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> list[float]:
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(
            y.to_numpy(dtype=np.float64) / LINE_LENGTH_KM,
            dtype=torch.float32,
        ).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        loss_history = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            total_epoch_loss = epoch_loss / len(dataset)
            loss_history.append(total_epoch_loss)

        return loss_history

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = self._prepare_tensor(X).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor).cpu().numpy().flatten()

        # Undo the scaling: network output is a fraction of the
        # line length; report in km.
        return output * LINE_LENGTH_KM


# ============================================================
# LOG LOSS
# ============================================================

def calculate_log_loss(model: ANNClassifierWrapper, X, y):
    probabilities = model.predict_proba(X)
    return float(log_loss(y, probabilities, labels=[0, 1]))


# ============================================================
# TRAIN ONE FAULT CLASSIFIER
# ============================================================

def train_fault_model(X_train, y_train, target, logger):

    logger.print("")
    logger.print("=" * 70)
    logger.print(f"TRAINING ANN: {target}")
    logger.print("=" * 70)

    logger.print(f"Architecture: {ANN_HIDDEN_LAYERS}")
    logger.print(f"Epochs      : {EPOCHS}")
    logger.print(f"Batch Size  : {BATCH_SIZE}")
    logger.print(f"Device      : {DEVICE}")

    model = ANNClassifierWrapper(
        input_size=X_train.shape[1],
        hidden_layers=ANN_HIDDEN_LAYERS,
        dropout=ANN_DROPOUT,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    start = time.perf_counter()
    epoch_losses = model.fit(X_train, y_train)
    end = time.perf_counter()

    training_time = end - start
    final_loss = calculate_log_loss(model, X_train, y_train)

    logger.print(f"\nTraining time: {training_time:.6f} s")
    logger.print(f"Final training log loss: {final_loss:.8f}")

    return model, training_time, final_loss, epoch_losses


def train_location_model(X_train, y_train, logger):
    """Train the fault-location regression head (fault rows only)."""

    logger.print("")
    logger.print("=" * 70)
    logger.print("TRAINING ANN: Loc")
    logger.print("=" * 70)

    logger.print(f"Architecture: {ANN_HIDDEN_LAYERS}")
    logger.print("Output      : linear (location, km)")
    logger.print("Loss        : MSELoss (target scaled by line length)")
    logger.print(f"Epochs      : {EPOCHS}")
    logger.print(f"Batch Size  : {BATCH_SIZE}")
    logger.print(f"Device      : {DEVICE}")

    model = ANNRegressorWrapper(
        input_size=X_train.shape[1],
        hidden_layers=ANN_HIDDEN_LAYERS,
        dropout=ANN_DROPOUT,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    start = time.perf_counter()
    epoch_losses = model.fit(X_train, y_train)
    end = time.perf_counter()

    training_time = end - start
    final_loss = epoch_losses[-1]

    logger.print(f"\nTraining time: {training_time:.6f} s")
    logger.print(
        f"Final training MSE (scaled target): {final_loss:.8f}"
    )

    return model, training_time, final_loss, epoch_losses


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, target, logger):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, labels=[0, 1], zero_division=0)
    recall = recall_score(y_test, predictions, labels=[0, 1], zero_division=0)
    f1 = f1_score(y_test, predictions, labels=[0, 1], zero_division=0)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    logger.print("")
    logger.print(f"--- {target} RESULTS ---")
    logger.print(f"Accuracy : {accuracy:.6f}")
    logger.print(f"Precision: {precision:.6f}")
    logger.print(f"Recall   : {recall:.6f}")
    logger.print(f"F1 Score : {f1:.6f}")

    logger.print("\nConfusion Matrix:")
    logger.print(str(cm))

    logger.print("\nClassification Report:")
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
        "predictions": predictions,
    }


# ============================================================
# LATENCY
# ============================================================

def measure_latency(model, X_test):

    # Warm-up
    model.predict(X_test.iloc[:1])

    latencies = []

    for i in range(len(X_test)):
        row = X_test.iloc[[i]]

        start = time.perf_counter()
        model.predict(row)
        end = time.perf_counter()

        latencies.append(end - start)

    total = sum(latencies)
    per_sample = total / len(latencies)

    return total, per_sample, latencies


# ============================================================
# ROBUSTNESS
# ============================================================

def test_robustness(model, X_test, y_test, noise_levels, trials):

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

            predictions = model.predict(X_noisy)

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

    metrics_df = pd.DataFrame(metrics)

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

    logger = ReportLogger(REPORT_FILE)

    try:
        logger.print("=" * 70)
        logger.print("TRANSMISSION LINE FAULT DETECTION")
        logger.print("ARTIFICIAL NEURAL NETWORK (ANN)")
        logger.print("=" * 70)

        logger.print(f"Timestamp: {timestamp}")
        logger.print(f"Dataset: {DATA_FILE}")
        logger.print("")

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        if not DATA_FILE.exists():
            logger.print("ERROR: Dataset not found:")
            logger.print(str(DATA_FILE))
            sys.exit(1)

        df = pd.read_csv(DATA_FILE)
        logger.print(f"Dataset shape: {df.shape}")

        # ----------------------------------------------------
        # CHECK COLUMNS
        # ----------------------------------------------------

        required = ["Loc"] + FEATURE_COLUMNS + TARGET_COLUMNS
        missing = [col for col in required if col not in df.columns]

        if missing:
            logger.print("\nMissing columns:")
            for column in missing:
                logger.print(f"  {column}")
            sys.exit(1)

        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        initial_rows = len(df)
        df = df.dropna(subset=(FEATURE_COLUMNS + TARGET_COLUMNS))
        removed = initial_rows - len(df)

        logger.print(f"Rows removed: {removed}")

        # ----------------------------------------------------
        # FEATURES & TARGETS
        # ----------------------------------------------------

        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMNS]

        logger.print("\nFEATURES:")
        for feature in FEATURE_COLUMNS:
            logger.print(f"  {feature}")

        logger.print(
            "\nADDITIONAL TARGET:\n"
            "  Loc (fault location, km; regression head,"
            " fault rows only)"
        )

        logger.print("\nTARGETS:")
        for target in TARGET_COLUMNS:
            logger.print(f"  {target}")

        # ----------------------------------------------------
        # CLASS DISTRIBUTION
        # ----------------------------------------------------

        logger.print("")
        logger.print("=" * 70)
        logger.print("CLASS DISTRIBUTION")
        logger.print("=" * 70)

        for target in TARGET_COLUMNS:
            logger.print(f"\n{target}:")
            counts = y[target].value_counts().sort_index()

            for value, count in counts.items():
                percentage = (count / len(y)) * 100
                logger.print(f"  {value}: {count} ({percentage:.2f}%)")

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
        logger.print(f"Training samples: {len(X_train)}")
        logger.print(f"Testing samples: {len(X_test)}")

        # ----------------------------------------------------
        # TRAIN MODELS
        # ----------------------------------------------------

        models = {}
        metrics = []
        loss_records = []
        robustness_records = []

        predictions_wide = pd.DataFrame({"row_id": test_row_id})
        latency_wide = pd.DataFrame({"row_id": test_row_id})

        for target in TARGET_COLUMNS:

            model, training_time, final_loss, epoch_losses = train_fault_model(
                X_train,
                y_train[target],
                target,
                logger
            )

            models[target] = model

            for epoch_idx, ep_loss in enumerate(epoch_losses):
                loss_records.append({
                    "timestamp": timestamp,
                    "target": target,
                    "epoch": epoch_idx + 1,
                    "log_loss": ep_loss,
                })

            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            results = evaluate_model(
                model,
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

            total_latency, per_sample_latency, row_latencies = measure_latency(
                model,
                X_test
            )

            latency_wide[f"{target}_latency"] = row_latencies

            logger.print(f"Total inference latency: {total_latency * 1000:.6f} ms")
            logger.print(f"Latency/sample: {per_sample_latency * 1000:.6f} ms")

            # ------------------------------------------------
            # ROBUSTNESS
            # ------------------------------------------------

            robustness_sweep = test_robustness(
                model,
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

            metrics.append({
                "timestamp": timestamp,
                "target": target,
                "architecture": str(ANN_HIDDEN_LAYERS),
                "epochs": EPOCHS,
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1_score": results["f1"],
                "training_time_seconds": training_time,
                "log_loss": final_loss,
                "latency_ms_total": total_latency * 1000,
                "latency_ms_per_sample": per_sample_latency * 1000,
                "robustness_accuracy": reference["accuracy_mean"],
                "robustness_f1": reference["f1_mean"],
                "robustness_retention": retention,
            })

        # ----------------------------------------------------
        # OVERALL FAULT OCCURRED
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
        # FAULT LOCATION (REGRESSION HEAD, FAULT ROWS ONLY)
        # ----------------------------------------------------
        # Fifth task of this script: a regressor of the same
        # architecture trained on fault rows only (Loc > 0).
        # No-fault rows have no fault location to estimate, so
        # their Loc_pred is left empty. The head is trained on the
        # fault rows OUTSIDE the whole test split and evaluated on
        # the fault rows OF the test split, so its output rows
        # align row-for-row with the per-line predictions above.
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

        location_model, location_training_time, \
            location_final_loss, location_epoch_losses = \
            train_location_model(
                X.loc[loc_train_idx],
                df.loc[loc_train_idx, "Loc"],
                logger
            )

        models["Loc"] = location_model

        loc_pred = location_model.predict(
            X.loc[loc_test_idx]
        )
        loc_true = df.loc[loc_test_idx, "Loc"].to_numpy()
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

        metrics.append({
            "timestamp": timestamp,
            "target": "Loc",
            "architecture": str(ANN_HIDDEN_LAYERS),
            "epochs": EPOCHS,
            "mae_km": mae_km,
            "mae_pct_line": mae_pct,
            "rmse_km": rmse_km,
            "training_time_seconds": location_training_time,
            "log_loss": location_final_loss,
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

        predictions_wide.to_csv(PREDICTIONS_FILE, index=False)
        latency_wide.to_csv(LATENCY_FILE, index=False)
        pd.DataFrame(robustness_records).to_csv(ROBUSTNESS_FILE, index=False)

        # ============================================================
        # SAVE TRAINED MODELS
        # ============================================================

        torch.save(
            {target: m.model.state_dict() for target, m in models.items()},
            MODEL_FILE
        )

        logger.print("")
        logger.print(f"Models saved to:\n  {MODEL_FILE}")

        # ----------------------------------------------------
        # SAVE LOSS & METRICS DATA
        # ----------------------------------------------------

        loss_df = pd.DataFrame(loss_records)
        if LOSS_FILE.exists():
            loss_df.to_csv(LOSS_FILE, mode="a", header=False, index=False)
        else:
            loss_df.to_csv(LOSS_FILE, index=False)

        save_metrics(metrics, METRICS_FILE)

        # ----------------------------------------------------
        # OVERALL RESULTS
        # ----------------------------------------------------

        metrics_df = pd.DataFrame(metrics)

        logger.print("")
        logger.print("=" * 70)
        logger.print("OVERALL RESULTS")
        logger.print("=" * 70)

        logger.print(f"Mean Accuracy : {metrics_df['accuracy'].mean():.6f}")
        logger.print(f"Mean Precision: {metrics_df['precision'].mean():.6f}")
        logger.print(f"Mean F1       : {metrics_df['f1_score'].mean():.6f}")
        logger.print(f"Mean Latency  : {metrics_df['latency_ms_per_sample'].mean():.6f} ms")
        logger.print(f"Mean Robustness: {metrics_df['robustness_accuracy'].mean():.6f}")

        # ----------------------------------------------------
        # OUTPUT FILES
        # ----------------------------------------------------

        logger.print("")
        logger.print("=" * 70)
        logger.print("OUTPUT FILES")
        logger.print("=" * 70)
        logger.print(f"Report          : {REPORT_FILE}")
        logger.print(f"Metrics         : {METRICS_FILE}")
        logger.print(f"Loss            : {LOSS_FILE}")
        logger.print(f"Predictions     : {PREDICTIONS_FILE}")
        logger.print(f"Per-row latency : {LATENCY_FILE}")
        logger.print(f"Robustness sweep: {ROBUSTNESS_FILE}")

        logger.print("\nTRAINING COMPLETE")

    finally:
        logger.close()


if __name__ == "__main__":
    main()