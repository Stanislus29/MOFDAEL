"""Feature preparation and reconstruction of un-persisted scalers.

Concerns kept here (and nowhere else):

1. Column validation for uploaded CSVs.
2. Reconstructing the ANN/RNN classifiers' and location heads'
   StandardScaler statistics from the deterministic, seeded training
   split (``scenarios.group_partition``), because the torch
   checkpoints persist weights only.
3. Grouping flat CSV rows into fault events for the LSTM's
   sequence input (the same rule the training script uses).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend import config


# ------------------------------------------------------------------
# COLUMN VALIDATION
# ------------------------------------------------------------------


@dataclass
class DatasetReport:
    """Validation summary for one uploaded CSV."""

    n_rows: int
    n_features: int
    missing_required: list[str]
    has_targets: bool
    has_location: bool
    fault_rows: int
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "missing_required": self.missing_required,
            "has_targets": self.has_targets,
            "has_location": self.has_location,
            "fault_rows": self.fault_rows,
            "warnings": self.warnings,
            "ok": self.ok,
        }


def validate_dataframe(df: pd.DataFrame) -> DatasetReport:
    """Check an uploaded CSV against the platform's feature contract."""

    missing = [
        c for c in config.FEATURE_COLUMNS if c not in df.columns
    ]
    has_targets = all(t in df.columns for t in config.TARGET_COLUMNS)
    has_location = config.LOCATION_COLUMN in df.columns

    fault_rows = 0
    if has_targets:
        fault_rows = int((df[config.TARGET_COLUMNS].sum(axis=1) > 0).sum())

    warnings: list[str] = []
    if missing:
        warnings.append(
            "Missing required feature columns: " + ", ".join(missing)
        )
    if not has_targets:
        warnings.append(
            "No ground-truth target columns (G/C/B/A) found — accuracy "
            "cannot be computed, only raw predictions."
        )
    if has_targets and not has_location:
        warnings.append(
            "No Loc column — fault-location heads will be skipped and "
            "LSTM rows are treated as single-sample events."
        )
    for col in config.FEATURE_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"Column {col} is not numeric and was coerced.")

    return DatasetReport(
        n_rows=len(df),
        n_features=len(config.FEATURE_COLUMNS),
        missing_required=missing,
        has_targets=has_targets,
        has_location=has_location,
        fault_rows=fault_rows,
        warnings=warnings,
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input frame (numeric, float)."""

    return df[config.FEATURE_COLUMNS].astype(float)


# ------------------------------------------------------------------
# SCALER RECONSTRUCTION (torch families: weights-only checkpoints)
# ------------------------------------------------------------------


def _training_split_positions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the seeded, scenario-grouped train/test split used by
    every training script (TEST_SIZE=0.20, RANDOM_STATE=42)."""

    from scenarios import group_partition

    train_pos, test_pos, _ = group_partition(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    return train_pos, test_pos


def _location_train_indices(
    df: pd.DataFrame, test_idx: pd.Index
) -> pd.Index:
    """Rows the location heads were trained on: fault rows of the train
    split (any target active AND Loc > 0), mirroring all four scripts
    that train a location head."""

    fault_mask = (
        df[config.TARGET_COLUMNS].sum(axis=1) > 0
    ) & (df[config.LOCATION_COLUMN] > 0)
    loc_train_idx = df.index[fault_mask].difference(test_idx)
    return loc_train_idx


def reconstruct_class_scaler(df: pd.DataFrame):
    """StandardScaler as fitted on the classifier training split."""

    from sklearn.preprocessing import StandardScaler

    train_pos, _ = _training_split_positions(df)
    return StandardScaler().fit(df.loc[train_pos, config.FEATURE_COLUMNS])


def reconstruct_location_scaler(df: pd.DataFrame):
    """StandardScaler as fitted on the location heads' training rows."""

    from sklearn.preprocessing import StandardScaler

    _, test_pos = None, None
    train_pos, test_pos = _training_split_positions(df)
    test_idx = df.index[test_pos]
    loc_idx = _location_train_indices(df, test_idx)
    return StandardScaler().fit(df.loc[loc_idx, config.FEATURE_COLUMNS])


# ------------------------------------------------------------------
# EVENT GROUPING (LSTM sequence input)
# ------------------------------------------------------------------


def build_events(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split rows into fault events with the LSTM training script's
    rule: a new event starts whenever Loc resets (loc[i] <= loc[i-1]).

    Without a Loc column every row is its own single-sample event.
    """

    if config.LOCATION_COLUMN not in df.columns:
        return [df.loc[[idx]] for idx in df.index]

    loc = df[config.LOCATION_COLUMN].to_numpy()

    new_event = np.ones(len(loc), dtype=bool)
    new_event[1:] = loc[1:] <= loc[:-1]

    event_id = np.cumsum(new_event) - 1

    return [group for _, group in df.groupby(event_id)]
