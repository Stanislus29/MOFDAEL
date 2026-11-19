"""Heavy-artifact loading, isolated from discovery and inference.

Importing TensorFlow / torch happens only inside these functions, so
lightweight operations (registry discovery, dataset validation) never
pay the framework load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend import config
from backend.core import preprocessing
from backend.core.registry import ModelSpec


class LoadedModel:
    """Container for everything an adapter needs at predict time."""

    def __init__(self):
        self.class_scaler = None      # classifier input scaler (or None)
        self.location_scaler = None   # location-head scaler (or None)
        self.targets: list[str] = []
        self.estimators: dict = {}    # target -> estimator/module
        self.extra: dict = {}


def ensure_src_path() -> None:
    config.ensure_src_on_path()


# ------------------------------------------------------------------
# SKLEARN (svm / knn / dtr)
# ------------------------------------------------------------------


def load_sklearn(spec: ModelSpec) -> LoadedModel:
    import joblib

    loaded = LoadedModel()
    payload = joblib.load(spec.artifacts["G"])
    loaded.targets = [t for t in ("G", "C", "B", "A") if t in payload]
    loaded.estimators = {t: payload[t] for t in loaded.targets}
    loaded.extra["kind"] = spec.model_id
    return loaded


# ------------------------------------------------------------------
# TORCH FLAT (ann / rnn)
# ------------------------------------------------------------------


def load_torch_flat(spec: ModelSpec, benchmark_df: pd.DataFrame) -> LoadedModel:
    import torch

    ensure_src_path()

    if spec.family == "torch_mlp":
        from ann import ANNModel as _TorchArch
        arch_kwargs = dict(hidden_layers=[64, 32], dropout=0.1)
        reshape = None
    else:
        from rnn import RNNModel as _TorchArch
        arch_kwargs = dict(hidden_size=64, num_layers=2, dropout=0.1)
        reshape = lambda x: x[:, np.newaxis, :]  # noqa: E731

    loaded = LoadedModel()
    loaded.targets = [t for t in ("G", "C", "B", "A") if t in spec.artifacts]
    if spec.has_loc:
        loaded.targets = loaded.targets + ["Loc"]

    # Weight-only checkpoints: rebuild the exact training-split scalers.
    loaded.class_scaler = preprocessing.reconstruct_class_scaler(benchmark_df)
    if spec.has_loc:
        loaded.location_scaler = preprocessing.reconstruct_location_scaler(
            benchmark_df
        )

    import torch as _torch

    state = _torch.load(
        spec.artifacts[loaded.targets[0]],
        map_location="cpu",
        weights_only=False,
    )

    def _build(target: str):
        input_size = 6
        if spec.family == "torch_mlp":
            model = _TorchArch(input_size=input_size, **arch_kwargs)
        else:
            model = _TorchArch(input_size=input_size, **arch_kwargs)
        model.load_state_dict(state[target])
        model.eval()
        return model

    loaded.estimators = {t: _build(t) for t in loaded.targets}
    loaded.extra["reshape"] = reshape
    loaded.extra["framework"] = "torch"
    return loaded


# ------------------------------------------------------------------
# KERAS FLAT (cnn)
# ------------------------------------------------------------------


def load_keras_flat(spec: ModelSpec) -> LoadedModel:
    import joblib
    import tensorflow as tf

    ensure_src_path()

    loaded = LoadedModel()
    manifest = joblib.load(spec.manifest_path)
    loaded.class_scaler = manifest["scaler"]
    loaded.targets = [t for t in ("G", "C", "B", "A", "Loc") if t in manifest["model_paths"]]

    for t in loaded.targets:
        loaded.estimators[t] = tf.keras.models.load_model(
            manifest["model_paths"][t]
        )

    if spec.has_loc and "Loc" in loaded.targets:
        # The location head was fitted with its own scaler on the
        # fault rows of the training split (see src/cnn.py); the
        # manifest only stores the classifier's scaler.
        benchmark_df = pd.read_csv(config.BENCHMARK_DATA_FILE)
        loaded.location_scaler = preprocessing.reconstruct_location_scaler(
            benchmark_df
        )

    loaded.extra["framework"] = "keras"
    return loaded


# ------------------------------------------------------------------
# KERAS EVENT (lstm)
# ------------------------------------------------------------------


def load_keras_event(spec: ModelSpec) -> LoadedModel:
    import joblib
    import tensorflow as tf

    ensure_src_path()

    loaded = LoadedModel()
    manifest = joblib.load(spec.manifest_path)
    loaded.class_scaler = manifest["scaler"]
    loaded.targets = [t for t in ("G", "C", "B", "A", "Loc") if t in manifest["model_paths"]]

    for t in loaded.targets:
        loaded.estimators[t] = tf.keras.models.load_model(
            manifest["model_paths"][t]
        )

    loaded.extra["framework"] = "keras"
    return loaded


# ------------------------------------------------------------------
# KERAS PINN
# ------------------------------------------------------------------


def load_keras_pinn(spec: ModelSpec) -> LoadedModel:
    import joblib
    import numpy as np
    import tensorflow as tf

    ensure_src_path()
    from pinn import create_pinn, create_pinn_location, CURRENT_INDICES

    loaded = LoadedModel()
    manifest = joblib.load(spec.manifest_path)
    loaded.class_scaler = manifest["scaler"]
    loaded.targets = [t for t in ("G", "C", "B", "A", "Loc") if t in manifest["model_paths"]]

    current_mean = np.asarray(manifest["current_mean"], dtype=np.float32)
    current_std = np.asarray(manifest["current_std"], dtype=np.float32)

    for t in loaded.targets:
        if t == "Loc":
            continue
        model = create_pinn(6, current_mean, current_std)
        model.load_weights(manifest["model_paths"][t])
        loaded.estimators[t] = model

    if spec.has_loc and "Loc" in loaded.targets:
        # The PINN location head carries its own scaler (and its own
        # current statistics) — see src/pinn.py train_location_model.
        benchmark_df = pd.read_csv(config.BENCHMARK_DATA_FILE)
        loaded.location_scaler = preprocessing.reconstruct_location_scaler(
            benchmark_df
        )
        loc_scaler = loaded.location_scaler
        loc_model = create_pinn_location(
            6,
            loc_scaler.mean_[CURRENT_INDICES].astype(np.float32),
            loc_scaler.scale_[CURRENT_INDICES].astype(np.float32),
        )
        loc_model.load_weights(manifest["model_paths"]["Loc"])
        loaded.estimators["Loc"] = loc_model

    loaded.extra["framework"] = "keras"
    return loaded


# ------------------------------------------------------------------
# RELAY (physics)
# ------------------------------------------------------------------


def load_relay(spec: ModelSpec) -> LoadedModel:
    ensure_src_path()
    from relay import detect_relay, zone1_reach_ohms  # noqa: F401

    loaded = LoadedModel()
    loaded.targets = ["trip"]
    loaded.estimators["trip"] = detect_relay
    loaded.extra["zone1_reach_ohms"] = zone1_reach_ohms()
    loaded.extra["framework"] = "physics"
    return loaded


def load_model(spec: ModelSpec, benchmark_df: pd.DataFrame) -> LoadedModel:
    """Dispatch on family."""

    if spec.family == "sklearn":
        return load_sklearn(spec)
    if spec.family in ("torch_mlp", "torch_rnn"):
        return load_torch_flat(spec, benchmark_df)
    if spec.family == "keras_flat":
        return load_keras_flat(spec)
    if spec.family == "keras_event":
        return load_keras_event(spec)
    if spec.family == "keras_pinn":
        return load_keras_pinn(spec)
    if spec.family == "relay":
        return load_relay(spec)
    raise ValueError(f"No loader for family: {spec.family}")
