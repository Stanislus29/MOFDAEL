"""Artifact discovery — resolves the latest trained artifacts per model.

This module is intentionally import-light (no TensorFlow / torch): it
only inspects the outputs/ directory tree and reads manifests lazily.
Loading happens in core.loaders, driven by the specs produced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from backend import config


# ------------------------------------------------------------------
# MODEL CATALOG
# ------------------------------------------------------------------

# Display order and metadata for the nine benchmark methods.
MODEL_CATALOG: dict[str, dict] = {
    "relay": {
        "name": "Distance Relay",
        "family": "relay",
        "kind": "physics",
        "description": "Numerical distance relay: six loop impedances "
        "vs a Zone 1 reach of 80% of |Z1| x 120 km.",
        "has_loc": False,
        "has_confidence": False,
    },
    "ann": {
        "name": "Artificial Neural Network",
        "family": "torch_mlp",
        "kind": "machine_learning",
        "description": "Feedforward MLP (64-32), one binary head per "
        "phase/ground plus a location regressor.",
        "has_loc": True,
        "has_confidence": True,
    },
    "cnn": {
        "name": "Convolutional Neural Network",
        "family": "keras_flat",
        "kind": "machine_learning",
        "description": "1-D CNN over the six scaled channels, one head "
        "per target plus a location regressor.",
        "has_loc": True,
        "has_confidence": True,
    },
    "lstm": {
        "name": "Long Short-Term Memory",
        "family": "keras_event",
        "kind": "machine_learning",
        "description": "Bidirectional LSTM over fault-event sequences "
        "(time-distributed per-row verdicts).",
        "has_loc": True,
        "has_confidence": True,
    },
    "pinn": {
        "name": "Physics-Informed Neural Network",
        "family": "keras_pinn",
        "kind": "machine_learning",
        "description": "MLP with a Kirchhoff physics residual "
        "(Ia + Ib + Ic = 0) inside the model.",
        "has_loc": True,
        "has_confidence": True,
    },
    "rnn": {
        "name": "Recurrent Neural Network",
        "family": "torch_rnn",
        "kind": "machine_learning",
        "description": "Two-layer RNN over the six features, one "
        "binary head per target plus a location regressor.",
        "has_loc": True,
        "has_confidence": True,
    },
    "svm": {
        "name": "Support Vector Machine",
        "family": "sklearn",
        "kind": "machine_learning",
        "description": "RBF SVC (probability calibrated) behind a "
        "StandardScaler, one pipeline per target.",
        "has_loc": False,
        "has_confidence": True,
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "family": "sklearn",
        "kind": "machine_learning",
        "description": "Distance-weighted KNN behind a StandardScaler, "
        "one pipeline per target.",
        "has_loc": False,
        "has_confidence": True,
    },
    "dtr": {
        "name": "Decision Tree",
        "family": "sklearn",
        "kind": "machine_learning",
        "description": "Depth-10 decision tree on raw features, one "
        "classifier per target.",
        "has_loc": False,
        "has_confidence": True,
    },
}

MODEL_IDS = list(MODEL_CATALOG)


# ------------------------------------------------------------------
# SPECS
# ------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Everything needed to locate and load one trained model."""

    model_id: str
    family: str
    name: str
    kind: str
    description: str
    has_loc: bool
    has_confidence: bool
    artifact_dir: config.Path
    targets: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)  # target -> path
    scaler_path: str | None = None       # manifest-held scaler (keras families)
    manifest_path: str | None = None
    extra: dict = field(default_factory=dict)
    trained_at: datetime | None = None

    def to_public_dict(self) -> dict:
        """JSON-safe view for the API layer."""

        return {
            "model_id": self.model_id,
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "description": self.description,
            "has_loc": self.has_loc,
            "has_confidence": self.has_confidence,
            "targets": list(self.targets),
            "trained_at": self.trained_at.isoformat()
            if self.trained_at
            else None,
            "available": bool(self.artifacts) or self.family == "relay",
        }


# ------------------------------------------------------------------
# DISCOVERY
# ------------------------------------------------------------------


def _latest(directory: config.Path, pattern_glob: str) -> config.Path | None:
    candidates = sorted(directory.glob(pattern_glob))
    return candidates[-1] if candidates else None


def _manifest_targets(manifest_path: config.Path) -> list[str]:
    import joblib

    manifest = joblib.load(manifest_path)
    paths = manifest.get("model_paths", {})
    return [t for t in ("G", "C", "B", "A", "Loc") if t in paths]


def _resolve_manifest_model(model_id: str) -> ModelSpec | None:
    """cnn / lstm / pinn: per-target artifacts tied together by a joblib
    manifest that also carries the fitted scaler."""

    meta = MODEL_CATALOG[model_id]
    model_dir = config.OUTPUTS_DIR / model_id / "models"
    manifest_path = _latest(model_dir, f"{model_id}_manifest_*.joblib")
    if manifest_path is None:
        return None

    import joblib

    manifest = joblib.load(manifest_path)
    paths = manifest.get("model_paths", {})

    spec = ModelSpec(
        model_id=model_id,
        family=meta["family"],
        name=meta["name"],
        kind=meta["kind"],
        description=meta["description"],
        has_loc=meta["has_loc"] and "Loc" in paths,
        has_confidence=meta["has_confidence"],
        artifact_dir=model_dir,
        targets=_manifest_targets(manifest_path),
        artifacts={t: str(p) for t, p in paths.items()},
        scaler_path=str(manifest_path),
        manifest_path=str(manifest_path),
        extra={"scaler": manifest.get("scaler")},
    )

    ts = manifest_path.stem.rsplit("_", 2)[-2:]
    try:
        spec.trained_at = datetime.strptime("_".join(ts), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        pass
    return spec


def _resolve_single_file_model(model_id: str) -> ModelSpec | None:
    """svm / knn / dtr: one joblib file mapping target -> estimator."""

    meta = MODEL_CATALOG[model_id]
    model_dir = config.OUTPUTS_DIR / model_id / "models"
    model_path = _latest(model_dir, f"{model_id}_model_*.joblib")
    if model_path is None:
        return None

    import joblib

    payload = joblib.load(model_path)
    targets = [t for t in ("G", "C", "B", "A") if t in payload]

    spec = ModelSpec(
        model_id=model_id,
        family=meta["family"],
        name=meta["name"],
        kind=meta["kind"],
        description=meta["description"],
        has_loc=False,
        has_confidence=meta["has_confidence"],
        artifact_dir=model_dir,
        targets=targets,
        artifacts={t: str(model_path) for t in targets},
    )

    ts = model_path.stem.rsplit("_", 2)[-2:]
    try:
        spec.trained_at = datetime.strptime("_".join(ts), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        pass
    return spec


def _resolve_torch_model(model_id: str) -> ModelSpec | None:
    """ann / rnn: a torch checkpoint {target: state_dict}; scalers are
    reconstructed from the deterministic training split in
    core.preprocessing."""

    meta = MODEL_CATALOG[model_id]
    model_dir = config.OUTPUTS_DIR / model_id / "models"
    model_path = _latest(model_dir, f"{model_id}_model_*.pt")
    if model_path is None:
        return None

    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    targets = [t for t in ("G", "C", "B", "A", "Loc") if t in state]

    spec = ModelSpec(
        model_id=model_id,
        family=meta["family"],
        name=meta["name"],
        kind=meta["kind"],
        description=meta["description"],
        has_loc=meta["has_loc"] and "Loc" in state,
        has_confidence=meta["has_confidence"],
        artifact_dir=model_dir,
        targets=targets,
        artifacts={t: str(model_path) for t in targets},
    )

    ts = model_path.stem.rsplit("_", 2)[-2:]
    try:
        spec.trained_at = datetime.strptime("_".join(ts), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        pass
    return spec


def resolve_model(model_id: str) -> ModelSpec | None:
    """Resolve the latest trained artifacts for one model id."""

    if model_id not in MODEL_CATALOG:
        raise KeyError(f"Unknown model id: {model_id}")

    if model_id == "relay":
        meta = MODEL_CATALOG["relay"]
        measurements_ok = config.RELAY_MEASUREMENTS_FILE.exists()
        ts = None
        if measurements_ok:
            ts = datetime.fromtimestamp(
                config.RELAY_MEASUREMENTS_FILE.stat().st_mtime
            )
        return ModelSpec(
            model_id="relay",
            family=meta["family"],
            name=meta["name"],
            kind=meta["kind"],
            description=meta["description"],
            has_loc=False,
            has_confidence=False,
            artifact_dir=config.OUTPUTS_DIR / "relay",
            targets=["trip"],
            artifacts={"trip": str(config.RELAY_MEASUREMENTS_FILE)},
            trained_at=ts,
        )

    family = MODEL_CATALOG[model_id]["family"]
    if family in ("keras_flat", "keras_event", "keras_pinn"):
        return _resolve_manifest_model(model_id)
    if family == "sklearn":
        return _resolve_single_file_model(model_id)
    if family in ("torch_mlp", "torch_rnn"):
        return _resolve_torch_model(model_id)
    return None


def list_models() -> list[ModelSpec]:
    """Resolve every catalog entry; unavailable models are returned
    with available=False so the UI can grey them out."""

    specs = []
    for model_id in MODEL_IDS:
        try:
            spec = resolve_model(model_id)
        except Exception:
            spec = None
        if spec is None:
            meta = MODEL_CATALOG[model_id]
            spec = ModelSpec(
                model_id=model_id,
                family=meta["family"],
                name=meta["name"],
                kind=meta["kind"],
                description=meta["description"],
                has_loc=meta["has_loc"],
                has_confidence=meta["has_confidence"],
                artifact_dir=config.OUTPUTS_DIR / model_id,
            )
        specs.append(spec)
    return specs
