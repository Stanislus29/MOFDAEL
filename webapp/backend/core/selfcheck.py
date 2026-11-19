"""Benchmark reproduction audit (the platform's ground-truth check).

Runs every resolved model through the platform's inference stack on
the benchmark test rows and compares against the dissertation's
exported predictions (outputs/<model>/predictions_<model>.csv).

A model "passes" only when the platform reproduces the benchmark —
detection agreement and location agreement are reported per target,
and a timestamped report is written under
webapp/storage/selfcheck/report_<timestamp>.txt using the AGENTS.md
ReportLogger pattern.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend import config
from backend.core import adapters, preprocessing
from backend.core.registry import MODEL_IDS, resolve_model

# Tolerances
AGREEMENT_PASS = 0.999          # fraction of matching detection rows
LOC_TOL_KM = 1e-3               # location-head reproduction tolerance


class ReportLogger:
    """stdout + file logger (AGENTS.md section 2)."""

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("w", encoding="utf-8")

    def print(self, message=""):
        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def _benchmark_predictions(model_id: str) -> pd.DataFrame:
    path = config.OUTPUTS_DIR / model_id / f"predictions_{model_id}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def check_model(model_id: str, logger: ReportLogger) -> dict:
    """Compare platform inference against one model's benchmark CSV."""

    spec = resolve_model(model_id)
    if spec is None:
        logger.print(f"[{model_id}] SKIP — artifacts not found")
        return {"model": model_id, "status": "SKIP"}

    bench = _benchmark_predictions(model_id)
    df = pd.read_csv(config.BENCHMARK_DATA_FILE)

    if model_id == "relay":
        # The measurements file carries the exported loop-impedance
        # columns, so the audit runs the exact benchmark decision.
        from backend.core import relay_service

        meas = pd.read_csv(config.RELAY_MEASUREMENTS_FILE)
        trip_arr = relay_service.relay_trip_from_measurements(meas)
        agree = float((trip_arr == bench["overall_pred"].to_numpy()).mean())
        loc_max = None
    else:
        features = preprocessing.prepare_features(df)
        adapter = adapters.get_adapter(spec, df)
        batch = adapter.predict(features, df)

        by_row = pd.DataFrame({"row_id": batch.row_ids})
        for t, p in batch.per_target_pred.items():
            by_row[f"{t}_pred"] = p
        merged = bench.merge(by_row, on="row_id", suffixes=("_bench", "_plat"))

        agreements = {}
        for t in ("G", "C", "B", "A"):
            bcol, pcol = f"{t}_pred_bench", f"{t}_pred_plat"
            if bcol in merged.columns:
                agreements[t] = float(
                    (merged[bcol] == merged[pcol]).mean()
                )
        agree = float(np.mean(list(agreements.values()))) if agreements else None

        loc_max = None
        if batch.loc_pred is not None and "Loc_pred" in bench.columns:
            plat_loc = pd.DataFrame(
                {"row_id": batch.row_ids, "loc_plat": batch.loc_pred}
            )
            m2 = bench.merge(plat_loc, on="row_id")
            if m2["loc_plat"].notna().any():
                loc_max = float(
                    np.nanmax(
                        np.abs(m2["loc_plat"] - m2["Loc_pred"])
                    )
                )

    passed = agree is not None and agree >= AGREEMENT_PASS and (
        loc_max is None or loc_max <= LOC_TOL_KM
    )

    logger.print(f"[{model_id}]")
    if agree is not None:
        logger.print(f"  detection agreement : {agree:.6f}")
    if loc_max is not None:
        logger.print(f"  loc max |diff| (km) : {loc_max:.8f}")
    logger.print(f"  status              : {'PASS' if passed else 'FAIL'}")
    logger.print("")

    return {
        "model": model_id,
        "status": "PASS" if passed else "FAIL",
        "agreement": agree,
        "loc_max_diff_km": loc_max,
    }


def run_selfcheck() -> dict:
    """Audit every catalog model; write and return the report."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = config.SELFCHECK_DIR / f"report_{timestamp}.txt"
    logger = ReportLogger(report_path)

    try:
        logger.print("=" * 70)
        logger.print("MOFDAEL PLATFORM — BENCHMARK REPRODUCTION AUDIT")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")
        logger.print(f"Benchmark dataset: {config.BENCHMARK_DATA_FILE.name}")
        logger.print("")

        results = []
        for model_id in MODEL_IDS:
            try:
                results.append(check_model(model_id, logger))
            except Exception as exc:  # keep the audit running
                logger.print(f"[{model_id}] ERROR — {exc}")
                results.append(
                    {"model": model_id, "status": "ERROR", "error": str(exc)}
                )

        n_pass = sum(r["status"] == "PASS" for r in results)
        logger.print("=" * 70)
        logger.print(f"SUMMARY: {n_pass}/{len(results)} models reproduce the benchmark")
        logger.print(f"Report  : {report_path}")
        logger.print("=" * 70)

        return {
            "timestamp": timestamp,
            "report_path": str(report_path),
            "results": results,
            "n_pass": n_pass,
            "n_total": len(results),
        }
    finally:
        logger.close()
