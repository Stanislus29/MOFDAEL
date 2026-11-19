"""Per-model-family inference adapters behind one uniform interface.

Every adapter converts a raw feature DataFrame into a PredictionBatch.
The engine never sees framework-specific code — adding a model family
means adding one adapter class here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend import config
from backend.core import loaders, preprocessing


# ------------------------------------------------------------------
# RESULT CONTAINER
# ------------------------------------------------------------------


@dataclass
class PredictionBatch:
    """Uniform inference output for one model over one dataset."""

    model_id: str
    row_ids: np.ndarray
    overall_pred: np.ndarray
    overall_prob: np.ndarray | None = None
    per_target_pred: dict[str, np.ndarray] = field(default_factory=dict)
    per_target_prob: dict[str, np.ndarray] | None = None
    loc_pred: np.ndarray | None = None
    latency_sampled_ms: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# ADAPTER INTERFACE
# ------------------------------------------------------------------


class ModelAdapter(ABC):
    """Common interface: load once, predict any prepared frame."""

    family: str

    def __init__(self, spec, loaded: loaders.LoadedModel):
        self.spec = spec
        self.loaded = loaded

    @abstractmethod
    def predict(self, features: pd.DataFrame, df: pd.DataFrame) -> PredictionBatch:
        """``features`` = validated float feature frame; ``df`` = the
        full uploaded frame (for Loc-driven event grouping)."""

    # -- shared latency probe --------------------------------------

    def _latency_probe(
        self,
        predict_one,
        features: pd.DataFrame,
        max_rows: int = 100,
    ) -> dict:
        """Per-row latency measured on a bounded sample (same spirit as
        the training scripts' measure_latency, capped for uploads)."""

        n = min(max_rows, len(features))
        if n == 0:
            return {"sampled_rows": 0, "mean_ms_per_row": None}

        sub = features.iloc[:n]
        start = time.perf_counter()
        for i in range(n):
            predict_one(sub.iloc[i : i + 1])
        elapsed = time.perf_counter() - start
        return {
            "sampled_rows": n,
            "mean_ms_per_row": elapsed / n * 1000.0,
        }


# ------------------------------------------------------------------
# SKLEARN (svm / knn / dtr)
# ------------------------------------------------------------------


class SklearnAdapter(ModelAdapter):
    family = "sklearn"

    def predict(self, features, df):
        preds, probs = {}, {}
        for t, est in self.loaded.estimators.items():
            preds[t] = est.predict(features).astype(int)
            if hasattr(est, "predict_proba"):
                probs[t] = est.predict_proba(features)[:, 1].astype(float)
            else:
                probs[t] = None

        available_probs = [p for p in probs.values() if p is not None]
        overall_pred = (
            np.max(np.vstack([preds[t] for t in preds]), axis=0)
            if preds
            else np.zeros(len(features), dtype=int)
        )
        overall_prob = (
            np.max(np.vstack(available_probs), axis=0)
            if available_probs
            else None
        )

        probe = self._latency_probe(
            lambda x: self.loaded.estimators["G"].predict(x),
            features,
        )

        return PredictionBatch(
            model_id=self.spec.model_id,
            row_ids=features.index.to_numpy(),
            overall_pred=overall_pred,
            overall_prob=overall_prob,
            per_target_pred=preds,
            per_target_prob=probs if all(p is not None for p in probs.values()) else None,
            loc_pred=None,
            latency_sampled_ms=probe,
        )


# ------------------------------------------------------------------
# TORCH FLAT (ann / rnn)
# ------------------------------------------------------------------


class TorchFlatAdapter(ModelAdapter):
    family = "torch_flat"

    def _forward(self, model, features: pd.DataFrame) -> np.ndarray:
        import torch

        X = features.to_numpy(dtype=np.float32)
        reshape = self.loaded.extra.get("reshape")
        if reshape is not None:
            X = reshape(X)
        with torch.no_grad():
            logits = model(torch.tensor(X)).numpy().flatten()
        return logits

    def predict(self, features, df):
        class_scaler = self.loaded.class_scaler
        Xs = pd.DataFrame(
            class_scaler.transform(features),
            index=features.index,
            columns=features.columns,
        )

        preds, probs = {}, {}
        for t in self.loaded.targets:
            if t == "Loc":
                continue
            logits = self._forward(self.loaded.estimators[t], Xs)
            p = 1.0 / (1.0 + np.exp(-logits))
            probs[t] = p.astype(float)
            preds[t] = (p >= 0.5).astype(int)

        overall_pred = np.max(np.vstack(list(preds.values())), axis=0)
        overall_prob = np.max(np.vstack(list(probs.values())), axis=0)

        loc_pred = None
        if "Loc" in self.loaded.estimators and self.loaded.location_scaler is not None:
            Xl = pd.DataFrame(
                self.loaded.location_scaler.transform(features),
                index=features.index,
                columns=features.columns,
            )
            loc_pred = self._forward(self.loaded.estimators["Loc"], Xl) * config.LINE_LENGTH_KM

        probe = self._latency_probe(
            lambda x: self._forward(self.loaded.estimators["G"], x),
            features,
        )

        return PredictionBatch(
            model_id=self.spec.model_id,
            row_ids=features.index.to_numpy(),
            overall_pred=overall_pred,
            overall_prob=overall_prob,
            per_target_pred=preds,
            per_target_prob=probs,
            loc_pred=loc_pred,
            latency_sampled_ms=probe,
        )


# ------------------------------------------------------------------
# KERAS FLAT (cnn)
# ------------------------------------------------------------------


class KerasFlatAdapter(ModelAdapter):
    family = "keras_flat"

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, X.shape[1], 1)

    def predict(self, features, df):
        Xs = self.loaded.class_scaler.transform(features)
        Xr = self._reshape(Xs)

        preds, probs = {}, {}
        for t in self.loaded.targets:
            if t == "Loc":
                continue
            p = self.loaded.estimators[t].predict(Xr, verbose=0).flatten()
            probs[t] = p.astype(float)
            preds[t] = (p > 0.5).astype(int)

        overall_pred = np.max(np.vstack(list(preds.values())), axis=0)
        overall_prob = np.max(np.vstack(list(probs.values())), axis=0)

        loc_pred = None
        if "Loc" in self.loaded.targets and self.loaded.location_scaler is not None:
            Xl = self._reshape(self.loaded.location_scaler.transform(features))
            loc_pred = (
                self.loaded.estimators["Loc"].predict(Xl, verbose=0).flatten()
                * config.LINE_LENGTH_KM
            )

        probe = self._latency_probe(
            lambda x: self.loaded.estimators["G"].predict(
                self._reshape(self.loaded.class_scaler.transform(x)), verbose=0
            ),
            features,
        )

        return PredictionBatch(
            model_id=self.spec.model_id,
            row_ids=features.index.to_numpy(),
            overall_pred=overall_pred,
            overall_prob=overall_prob,
            per_target_pred=preds,
            per_target_prob=probs,
            loc_pred=loc_pred,
            latency_sampled_ms=probe,
        )


# ------------------------------------------------------------------
# KERAS EVENT (lstm)
# ------------------------------------------------------------------


class KerasEventAdapter(ModelAdapter):
    family = "keras_event"

    def predict(self, features, df):
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        events = preprocessing.build_events(df)
        event_indices = list(range(len(events)))

        X_list = [
            self.loaded.class_scaler.transform(
                events[i][config.FEATURE_COLUMNS].to_numpy()
            )
            for i in event_indices
        ]
        lengths = [len(events[i]) for i in event_indices]
        max_len = max(lengths) if lengths else 1
        X_padded = pad_sequences(
            X_list, maxlen=max_len, dtype="float32", padding="post", value=0.0
        )

        preds, probs = {}, {}
        for t in self.loaded.targets:
            if t == "Loc":
                continue
            out = self.loaded.estimators[t].predict(X_padded, verbose=0)
            p_flat, pred_flat, rows_flat = [], [], []
            for row, (i, L) in enumerate(zip(event_indices, lengths)):
                for pos, r in enumerate(events[i].index[:L]):
                    rows_flat.append(r)
                    p_flat.append(out[row, pos, 0])
            probs[t] = pd.Series(np.array(p_flat), index=rows_flat).reindex(
                features.index
            ).to_numpy()
            preds[t] = (probs[t] > 0.5).astype(int)

        overall_pred = np.nanmax(np.vstack([p for p in preds.values()]), axis=0).astype(int)
        with np.errstate(all="ignore"):
            overall_prob = np.nanmax(np.vstack([p for p in probs.values()]), axis=0)

        # Location head: evaluated per row over fault events.
        loc_pred = None
        if "Loc" in self.loaded.targets:
            fault_events = [
                i for i in event_indices if events[i][config.LOCATION_COLUMN].max() > 0
            ] if config.LOCATION_COLUMN in df.columns else []
            if fault_events:
                Xf = [
                    self.loaded.class_scaler.transform(
                        events[i][config.FEATURE_COLUMNS].to_numpy()
                    )
                    for i in fault_events
                ]
                lengths_f = [len(events[i]) for i in fault_events]
                Xf_padded = pad_sequences(
                    Xf, maxlen=max(lengths_f), dtype="float32",
                    padding="post", value=0.0,
                )
                out = self.loaded.estimators["Loc"].predict(Xf_padded, verbose=0)
                loc_rows, loc_vals = [], []
                for row, (i, L) in enumerate(zip(fault_events, lengths_f)):
                    for pos, r in enumerate(events[i].index[:L]):
                        loc_rows.append(r)
                        loc_vals.append(out[row, pos, 0] * config.LINE_LENGTH_KM)
                loc_pred = (
                    pd.Series(loc_vals, index=loc_rows)
                    .reindex(features.index)
                    .to_numpy()
                )
            else:
                loc_pred = np.full(len(features), np.nan)

        probe = self._latency_probe(
            lambda x: self.loaded.estimators["G"].predict(
                pad_sequences(
                    [self.loaded.class_scaler.transform(x.to_numpy())],
                    maxlen=max_len, dtype="float32", padding="post", value=0.0,
                ),
                verbose=0,
            ),
            features,
        )

        return PredictionBatch(
            model_id=self.spec.model_id,
            row_ids=features.index.to_numpy(),
            overall_pred=overall_pred,
            overall_prob=overall_prob,
            per_target_pred=preds,
            per_target_prob=probs,
            loc_pred=loc_pred,
            latency_sampled_ms=probe,
            notes=[
                "Rows grouped into fault events by Loc resets "
                "(single-sample events when Loc is absent)."
            ],
        )


# ------------------------------------------------------------------
# KERAS PINN
# ------------------------------------------------------------------


class KerasPinnAdapter(ModelAdapter):
    family = "keras_pinn"

    def predict(self, features, df):
        Xs = self.loaded.class_scaler.transform(features).astype(np.float32)

        preds, probs = {}, {}
        for t in self.loaded.targets:
            if t == "Loc":
                continue
            p = self.loaded.estimators[t].predict(Xs, verbose=0).flatten()
            probs[t] = p.astype(float)
            preds[t] = (p > 0.5).astype(int)

        overall_pred = np.max(np.vstack(list(preds.values())), axis=0)
        overall_prob = np.max(np.vstack(list(probs.values())), axis=0)

        loc_pred = None
        if "Loc" in self.loaded.targets and self.loaded.location_scaler is not None:
            Xl = self.loaded.location_scaler.transform(features).astype(np.float32)
            loc_pred = (
                self.loaded.estimators["Loc"].predict(Xl, verbose=0).flatten()
                * config.LINE_LENGTH_KM
            )

        probe = self._latency_probe(
            lambda x: self.loaded.estimators["G"].predict(
                self.loaded.class_scaler.transform(x).astype(np.float32),
                verbose=0,
            ),
            features,
        )

        return PredictionBatch(
            model_id=self.spec.model_id,
            row_ids=features.index.to_numpy(),
            overall_pred=overall_pred,
            overall_prob=overall_prob,
            per_target_pred=preds,
            per_target_prob=probs,
            loc_pred=loc_pred,
            latency_sampled_ms=probe,
        )


# ------------------------------------------------------------------
# RELAY (physics)
# ------------------------------------------------------------------


class RelayAdapter(ModelAdapter):
    family = "relay"

    def predict(self, features, df):
        from backend.core import relay_service

        if all(c in df.columns for c in config.RELAY_LOOP_COLUMNS):
            trip = relay_service.relay_trip_from_measurements(df)
            mode = "exact"
        else:
            trip = relay_service.relay_trip_from_features(df)
            mode = "emulated"

        start = time.perf_counter()
        relay_service.relay_trip_from_features(features)
        elapsed_ms = (time.perf_counter() - start) / max(len(features), 1) * 1000

        trip_int = trip.astype(int)

        return PredictionBatch(
            model_id="relay",
            row_ids=features.index.to_numpy(),
            overall_pred=trip_int,
            overall_prob=None,
            per_target_pred={},
            per_target_prob=None,
            loc_pred=None,
            latency_sampled_ms={
                "sampled_rows": len(features),
                "mean_ms_per_row": elapsed_ms,
            },
            notes=[f"Relay mode: {mode}"],
        )


# ------------------------------------------------------------------
# FACTORY
# ------------------------------------------------------------------


def get_adapter(spec, benchmark_df: pd.DataFrame) -> ModelAdapter:
    loaded = loaders.load_model(spec, benchmark_df)

    family = spec.family
    if family == "sklearn":
        return SklearnAdapter(spec, loaded)
    if family in ("torch_mlp", "torch_rnn"):
        return TorchFlatAdapter(spec, loaded)
    if family == "keras_flat":
        return KerasFlatAdapter(spec, loaded)
    if family == "keras_event":
        return KerasEventAdapter(spec, loaded)
    if family == "keras_pinn":
        return KerasPinnAdapter(spec, loaded)
    if family == "relay":
        return RelayAdapter(spec, loaded)
    raise ValueError(f"No adapter for family: {family}")
