"""Inference engine — orchestrates adapters and produces run artifacts.

Responsibilities (only these):
* validate + prepare an uploaded dataset,
* run each requested model's adapter,
* assemble per-model prediction/latency CSVs and a run summary,
* compute cross-model consensus per row.

Framework and artifact details live in core.adapters / core.loaders;
persistence lives in backend.storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend import config
from backend.core import adapters, preprocessing, zones as zones_mod
from backend.core.registry import resolve_model
from backend.core.types import ModelResult, RunResult


# ------------------------------------------------------------------
# RUN ID
# ------------------------------------------------------------------


def new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def run_dir(run_id: str) -> Path:
    d = config.RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------
# CORE EXECUTION
# ------------------------------------------------------------------


def _benchmark_frame() -> pd.DataFrame:
    return pd.read_csv(config.BENCHMARK_DATA_FILE)


def run_models(
    df: pd.DataFrame,
    model_ids: list[str],
    line_length_km: float = config.LINE_LENGTH_KM,
) -> RunResult:
    """Run the selected models over one validated dataset frame."""

    benchmark_df = _benchmark_frame()
    features = preprocessing.prepare_features(df)
    report = preprocessing.validate_dataframe(df)
    has_targets = report.has_targets

    fault_true = (
        df[config.TARGET_COLUMNS].sum(axis=1) > 0
    ).astype(int) if has_targets else None

    results: dict[str, ModelResult] = {}
    created = datetime.now()

    for model_id in model_ids:
        spec = resolve_model(model_id)
        if spec is None or (not spec.to_public_dict()["available"]):
            results[model_id] = ModelResult(
                model_id=model_id,
                predictions=pd.DataFrame(),
                latency={},
                notes=["Model artifacts not found — skipped."],
            )
            continue

        adapter = adapters.get_adapter(spec, benchmark_df)
        batch = adapter.predict(features, df)

        out = pd.DataFrame({"row_id": features.index.to_numpy()})
        out["overall_pred"] = batch.overall_pred.astype(int)
        if fault_true is not None:
            out["overall_true"] = fault_true.to_numpy()
        for t, p in batch.per_target_pred.items():
            out[f"{t}_pred"] = p.astype(int)
            if has_targets and t in df.columns:
                out[f"{t}_true"] = df[t].to_numpy()
            elif has_targets:
                out[f"{t}_true"] = 0
        if batch.overall_prob is not None:
            out["overall_confidence"] = np.round(batch.overall_prob, 6)
        for t, p in (batch.per_target_prob or {}).items():
            out[f"{t}_confidence"] = np.round(p, 6)
        if batch.loc_pred is not None and report.has_location:
            out["Loc_true"] = df[config.LOCATION_COLUMN].to_numpy()
            out["Loc_pred"] = batch.loc_pred
            err = np.abs(
                pd.Series(batch.loc_pred) - df[config.LOCATION_COLUMN].to_numpy()
            )
            out["Loc_error_km"] = err.round(6)
            out["Loc_error_pct"] = (err / line_length_km * 100).round(6)

        results[model_id] = ModelResult(
            model_id=model_id,
            predictions=out,
            latency=batch.latency_sampled_ms,
            notes=batch.notes,
        )

    consensus = build_consensus(results, fault_true, features.index)
    summary = build_summary(
        results, consensus, line_length_km, created, len(df)
    )

    result = RunResult(
        run_id=new_run_id(),
        dataset_id="inline",
        model_results=results,
        consensus=consensus,
        summary=summary,
        created_at=created.isoformat(),
    )

    # Spatial fault zones for the Line Monitor (consensus + location
    # column or a model location head required).
    try:
        if consensus is not None:
            result.zones = zones_mod.build_segment_zones(df, result)
    except Exception:
        result.zones = None

    return result


# ------------------------------------------------------------------
# CONSENSUS & SUMMARY
# ------------------------------------------------------------------


def build_consensus(
    results: dict[str, ModelResult],
    fault_true: pd.Series | None,
    index: pd.Index,
) -> pd.DataFrame | None:
    """Row-level agreement across models: how many methods flag a
    fault, majority verdict, and per-phase majority."""

    det_cols = {}
    for mid, res in results.items():
        if res.predictions.empty:
            continue
        det_cols[mid] = res.predictions["overall_pred"].to_numpy()

    if not det_cols:
        return None

    matrix = np.vstack(list(det_cols.values()))
    consensus = pd.DataFrame({"row_id": index.to_numpy()})
    consensus["n_models"] = len(det_cols)
    consensus["n_flag_fault"] = matrix.sum(axis=0).astype(int)
    consensus["majority_fault"] = (
        consensus["n_flag_fault"] > len(det_cols) / 2
    ).astype(int)
    consensus["unanimous_fault"] = (
        consensus["n_flag_fault"] == len(det_cols)
    ).astype(int)
    consensus["flag_fraction"] = (
        consensus["n_flag_fault"] / len(det_cols)
    ).round(6)

    if fault_true is not None:
        consensus["overall_true"] = fault_true.to_numpy()

    # Per-target majority across models that provide the head.
    for t in config.TARGET_COLUMNS:
        votes = []
        for mid, res in results.items():
            col = f"{t}_pred"
            if not res.predictions.empty and col in res.predictions.columns:
                votes.append(res.predictions[col].to_numpy())
        if votes:
            consensus[f"{t}_majority"] = (
                np.mean(np.vstack(votes), axis=0) >= 0.5
            ).astype(int)

    return consensus


def build_summary(
    results: dict[str, ModelResult],
    consensus: pd.DataFrame | None,
    line_length_km: float,
    created: datetime,
    n_rows: int,
) -> dict:
    """Headline numbers per model: accuracy (when truth exists),
    mean confidence, location MAE, sampled latency."""

    def _py(value):
        """Coerce numpy scalars to plain Python for JSON/API safety."""

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    per_model = {}
    for mid, res in results.items():
        entry: dict = {
            "n_rows": int(len(res.predictions)),
            "latency": {k: _py(v) for k, v in res.latency.items()},
            "notes": res.notes,
        }
        p = res.predictions
        if not p.empty:
            if "overall_true" in p.columns:
                entry["accuracy"] = float(
                    (p["overall_pred"] == p["overall_true"]).mean()
                )
            if "overall_confidence" in p.columns:
                entry["mean_confidence"] = float(
                    p["overall_confidence"].mean()
                )
            if "Loc_error_km" in p.columns:
                fault_rows = p["Loc_error_km"].notna()
                if fault_rows.any():
                    entry["loc_mae_km"] = float(
                        p.loc[fault_rows, "Loc_error_km"].mean()
                    )
                    entry["loc_mae_pct"] = (
                        entry["loc_mae_km"] / line_length_km * 100
                    )
        per_model[mid] = entry

    consensus_flags = (
        int(consensus["n_flag_fault"].sum()) if consensus is not None else None
    )

    return {
        "created_at": created.isoformat(),
        "n_rows": n_rows,
        "line_length_km": line_length_km,
        "models": per_model,
        "n_flag_fault_consensus": consensus_flags,
    }


# ------------------------------------------------------------------
# PERSIST RUN ARTIFACTS (naming per AGENTS.md)
# ------------------------------------------------------------------


def write_run_artifacts(result: RunResult) -> Path:
    """Write predictions_<model>.csv, latency block, consensus.csv and
    summary.json into the run directory."""

    out_dir = run_dir(result.run_id)

    for mid, res in result.model_results.items():
        if not res.predictions.empty:
            res.predictions.to_csv(
                out_dir / f"predictions_{mid}.csv", index=False
            )
        if res.latency:
            pd.DataFrame([res.latency]).to_csv(
                out_dir / f"latency_summary_{mid}.csv", index=False
            )

    if result.consensus is not None:
        result.consensus.to_csv(out_dir / "consensus.csv", index=False)

    if result.zones is not None:
        zones_mod.write_zones_file(out_dir, result.zones)

    import json

    (out_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, default=str)
    )

    return out_dir
