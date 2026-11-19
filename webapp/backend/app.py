"""FastAPI application factory.

Routes are thin: parse/validate input, call core, shape the response.
All business logic lives in backend.core.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend import config, storage
from backend.core import engine as engine_mod
from backend.core import preprocessing
from backend.core.registry import MODEL_IDS, list_models, resolve_model

# Reuse the verified training/physics modules (ann, rnn, pinn, lstm,
# simulate, relay) — traceability rule.
config.ensure_src_on_path()

app = FastAPI(
    title="MOFDAEL Line Monitoring API",
    version="0.1.0",
    description="Inference platform for the eight machine-learning "
    "architectures and the distance-relay baseline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# HEALTH & CATALOG
# ------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def api_list_models():
    return {"models": [s.to_public_dict() for s in list_models()]}


@app.get("/api/models/{model_id}")
def api_model_detail(model_id: str):
    if model_id not in MODEL_IDS:
        raise HTTPException(404, f"Unknown model: {model_id}")
    spec = resolve_model(model_id)
    return spec.to_public_dict() if spec else {"model_id": model_id, "available": False}


# ------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------


@app.post("/api/datasets")
async def api_upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV of line measurements (Ia…Ic, Va…Vc required;
    G/C/B/A and Loc optional). Returns a validation report."""

    raw = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    report = preprocessing.validate_dataframe(df)
    if not report.ok:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Dataset rejected — missing required feature columns.",
                "report": report.to_dict(),
            },
        )

    dataset_id = f"ds_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}"
    stored_path = config.DATASETS_DIR / f"{dataset_id}.csv"
    df.to_csv(stored_path, index=False)

    import json

    storage.insert_dataset(
        dataset_id=dataset_id,
        filename=file.filename or dataset_id + ".csv",
        stored_path=stored_path,
        n_rows=report.n_rows,
        has_targets=report.has_targets,
        has_location=report.has_location,
        fault_rows=report.fault_rows,
        report_json=json.dumps(report.to_dict()),
    )

    return {"dataset_id": dataset_id, "report": report.to_dict()}


@app.get("/api/datasets")
def api_list_datasets():
    return {"datasets": storage.list_datasets()}


@app.get("/api/datasets/{dataset_id}")
def api_dataset_detail(dataset_id: str):
    rec = storage.get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(404, "Dataset not found")
    rec["report"] = json.loads(rec["report_json"])
    del rec["report_json"]
    return rec


@app.get("/api/datasets/{dataset_id}/preview")
def api_dataset_preview(dataset_id: str, rows: int = 20):
    rec = storage.get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(404, "Dataset not found")
    df = pd.read_csv(Path(rec["stored_path"]))
    return {"columns": list(df.columns), "rows": df.head(rows).to_dict("records")}


# ------------------------------------------------------------------
# INFERENCE RUNS
# ------------------------------------------------------------------


@app.post("/api/runs")
def api_start_run(body: dict):
    """Run selected models over an uploaded dataset (synchronous).

    Body: {"dataset_id": "...", "model_ids": ["ann", ...],
           "line_length_km": 120.0 (optional)}
    """

    dataset_id = body.get("dataset_id")
    model_ids = body.get("model_ids") or []

    if not dataset_id:
        raise HTTPException(400, "dataset_id is required")
    if not model_ids:
        raise HTTPException(400, "model_ids must contain at least one model")
    unknown = [m for m in model_ids if m not in MODEL_IDS]
    if unknown:
        raise HTTPException(400, f"Unknown model ids: {unknown}")

    rec = storage.get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(404, "Dataset not found")

    df = pd.read_csv(Path(rec["stored_path"]))
    line_length_km = float(body.get("line_length_km", config.LINE_LENGTH_KM))

    run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}"
    storage.insert_run(run_id, dataset_id, model_ids)

    result = engine_mod.run_models(df, model_ids, line_length_km)
    result.run_id = run_id
    result.dataset_id = dataset_id

    out_dir = engine_mod.write_run_artifacts(result)
    storage.finish_run(
        run_id,
        summary_json=json.dumps(result.summary, default=str),
        output_dir=str(out_dir),
    )

    return {
        "run_id": run_id,
        "summary": result.summary,
        "output_dir": str(out_dir),
    }


@app.get("/api/runs")
def api_list_runs():
    return {"runs": storage.list_runs()}


@app.get("/api/runs/{run_id}")
def api_run_detail(run_id: str):
    rec = storage.get_run(run_id)
    if rec is None:
        raise HTTPException(404, "Run not found")
    if rec["summary_json"]:
        rec["summary"] = json.loads(rec["summary_json"])
        del rec["summary_json"]
    return rec


@app.get("/api/runs/{run_id}/predictions/{model_id}")
def api_run_predictions(run_id: str, model_id: str):
    rec = storage.get_run(run_id)
    if rec is None:
        raise HTTPException(404, "Run not found")
    path = Path(rec["output_dir"]) / f"predictions_{model_id}.csv"
    if not path.exists():
        raise HTTPException(404, f"No predictions for {model_id} in this run")
    return FileResponse(path, media_type="text/csv")


@app.get("/api/runs/{run_id}/zones")
def api_run_zones(run_id: str):
    """Spatial fault zones for the Line Monitor."""

    rec = storage.get_run(run_id)
    if rec is None:
        raise HTTPException(404, "Run not found")
    path = Path(rec["output_dir"]) / "zones.json"
    if not path.exists():
        return {"segments": [], "note": "No location data for this run."}
    return json.loads(path.read_text())


@app.get("/api/datasets/{dataset_id}/file")
def api_dataset_file(dataset_id: str):
    rec = storage.get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(404, "Dataset not found")
    return FileResponse(Path(rec["stored_path"]), media_type="text/csv")


# ------------------------------------------------------------------
# NETWORKS
# ------------------------------------------------------------------


@app.get("/api/networks")
def api_list_networks():
    return {"networks": storage.list_networks()}


@app.post("/api/networks")
def api_create_network(body: dict):
    name = (body.get("name") or "").strip()
    doc = body.get("doc")
    if not name:
        raise HTTPException(400, "name is required")
    if not isinstance(doc, dict):
        raise HTTPException(400, "doc must be a network document object")

    network_id = body.get("network_id") or (
        f"net_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}"
    )
    storage.upsert_network(network_id, name, json.dumps(doc))
    return {"network_id": network_id, "name": name}


@app.get("/api/networks/{network_id}")
def api_get_network(network_id: str):
    rec = storage.get_network(network_id)
    if rec is None:
        raise HTTPException(404, "Network not found")
    rec["doc"] = json.loads(rec["doc_json"])
    del rec["doc_json"]
    return rec


@app.put("/api/networks/{network_id}")
def api_update_network(network_id: str, body: dict):
    rec = storage.get_network(network_id)
    if rec is None:
        raise HTTPException(404, "Network not found")
    name = (body.get("name") or rec["name"]).strip()
    doc = body.get("doc")
    if not isinstance(doc, dict):
        raise HTTPException(400, "doc must be a network document object")
    storage.update_network(network_id, name, json.dumps(doc))
    return {"network_id": network_id, "name": name}


@app.delete("/api/networks/{network_id}")
def api_delete_network(network_id: str):
    if not storage.delete_network(network_id):
        raise HTTPException(404, "Network not found")
    return {"deleted": network_id}


# ------------------------------------------------------------------
# SELF-CHECK
# ------------------------------------------------------------------


@app.post("/api/selfcheck")
def api_selfcheck():
    from backend.core import selfcheck

    return selfcheck.run_selfcheck()
