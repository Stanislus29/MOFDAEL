# MOFDAEL Web Platform — Backend (Phase 1)

Interactive inference platform for the dissertation's eight machine-learning
architectures and the numerical distance-relay baseline. This phase delivers a
fully working **FastAPI backend**: artifact discovery, unified inference,
dataset upload, run orchestration, and a benchmark reproduction audit.

## Architecture & separation of concerns

```
backend/
├── config.py               # paths & constants (single source of truth)
├── app.py                  # FastAPI factory + thin routes
├── storage.py              # SQLite persistence (datasets, runs)
├── core/
│   ├── registry.py         # artifact DISCOVERY only (no TF/torch imports)
│   ├── loaders.py          # heavy artifact LOADING (framework imports live here)
│   ├── preprocessing.py    # column validation, scaler/event reconstruction
│   ├── adapters.py         # per-family adapters behind ONE uniform interface
│   ├── engine.py           # orchestration: run models, consensus, artifacts
│   ├── relay_service.py    # exact vs emulated relay decisions
│   └── selfcheck.py        # benchmark reproduction audit (ReportLogger style)
└── storage/                # generated artifacts (datasets/, runs/, selfcheck/)
```

Dependency direction: `app → storage/core → core internals → config`.
Routes never touch model artifacts directly; the engine never sees
framework-specific code. Adding a model family = one loader + one adapter.

## Reuse contract (traceability rule)

The platform re-uses the verified training modules rather than re-deriving
anything (`config.ensure_src_on_path()` makes `src/` importable):

- `ann.ANNModel`, `rnn.RNNModel` — torch architectures for weight-only
  checkpoints
- `pinn.create_pinn` / `pinn.create_pinn_location` — physics-informed models
- `simulate.loop_impedances`, `relay.detect_relay`, `relay.zone1_reach_ohms` —
  relay physics
- `scenarios.group_partition` — deterministic seeded split used to rebuild the
  ANN/RNN scalers (their checkpoints persist weights only)

Scaler facts encoded here (verified against the benchmark):

| Model | Classifier scaler | Location scaler |
|---|---|---|
| svm / knn | inside saved sklearn Pipeline | n/a (no Loc head) |
| dtr | none (raw features) | n/a |
| cnn / pinn | manifest | **reconstructed** (fault rows of train split) |
| ann / rnn | **reconstructed** (`group_partition`) | **reconstructed** (fault rows) |
| lstm | manifest | manifest (same scaler) |

## Self-check (ground truth)

`POST /api/selfcheck` (or `python -m backend.core.selfcheck`) runs every model
through the platform stack on the benchmark test rows and compares against
`outputs/<model>/predictions_<model>.csv`. Latest result: **9/9 PASS**
(detection agreement 1.000000; location heads match to < 2e-5 km). Reports are
written to `storage/selfcheck/report_<timestamp>.txt`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/models` | catalog of 9 methods + availability |
| POST | `/api/datasets` | upload CSV (`Ia…Ic,Va…Vc` required; `G/C/B/A`, `Loc` optional) |
| GET | `/api/datasets`, `/api/datasets/{id}`, `/api/datasets/{id}/preview` | list/detail/preview |
| POST | `/api/runs` | `{dataset_id, model_ids, line_length_km?}` → synchronous run |
| GET | `/api/runs`, `/api/runs/{id}` | history / summary |
| GET | `/api/runs/{id}/predictions/{model}` | per-model predictions CSV |
| GET/POST | `/api/networks` | list / upsert saved networks |
| GET/PUT/DELETE | `/api/networks/{id}` | network document CRUD |
| POST | `/api/selfcheck` | benchmark reproduction audit |

Run outputs follow the AGENTS.md naming standard inside
`storage/runs/<run_id>/`: `predictions_<model>.csv`, `consensus.csv`,
`summary.json`, `latency_summary_<model>.csv`. **Nothing is ever written to
`outputs/`**, so `src/hypothesis.py`'s prediction glob stays intact.

## Per-row output columns

`row_id`, `overall_pred` (+`overall_true` when targets present),
`{G,C,B,A}_pred/true`, `overall_confidence`, `{G,C,B,A}_confidence` where the
model produces probabilities, and `Loc_true/Loc_pred/Loc_error_km/
Loc_error_pct` for models with a location head.

## Known limits (documented, by design)

- Relay on uploaded CSVs runs in **emulated** mode (row values treated as
  steady-state phasors). Exact mode is only possible with exported loop
  impedances; the self-check uses exact mode. Emulated results are flagged in
  the response notes.
- LSTM groups rows into events via `Loc` resets; without `Loc`, each row is a
  single-sample event (noted in run output).
- Confidence is absent for the relay by design (deterministic impedance trip).
- Per-row latency is measured on a bounded sample (≤ 100 rows) to keep runs
  responsive; batch prediction produces the actual results.

## Run

```bash
cd webapp
./run_platform.sh        # backend (:8790) + frontend (:5173) in one command
```

Then open http://localhost:5173. (Backend-only: `./run_backend.sh`;
frontend-only: `./run_frontend.sh` — the proxy errors with ECONNREFUSED if
the backend is not running.)

## Network Studio (Phase 3)

SVG canvas editor with pan (drag background), zoom (wheel), 10 px snap:

- **Tools** — Select/move, Draw line (click component → component)
- **Components** — source, substation, busbar, transformer, relay, CT/PT,
  load, junction
- **Line segments** — length (km) and the verified per-km impedances
  (Z₁ = 0.080 + j0.390, Z₀ = 0.228 + j1.352 Ω/km defaults)
- **Persistence** — Save/New + JSON export/import; networks stored in the
  platform SQLite DB via `/api/networks`
- **Keyboard** — Delete removes selection, Esc cancels placement

Browser smoke test (Playwright, needs the backend + frontend running):

```bash
cd webapp/frontend && node tests/studio_smoke.cjs
```

## Frontend (`frontend/`)

Vite + React. The theme lives in `src/styles/theme.css` as CSS custom
properties (off-white `--bg`, dark navy `--ink`, navy `--navy`,
deep blue `--deep-blue`); views are thin compositions of the shared
primitives in `src/components/`. `src/api/client.js` is the only
place that talks to the backend.

| View | Status |
|---|---|
| Network Studio | **working** — SVG editor, palette, segments, persistence |
| Line Monitor | shell + empty state (Phase 4: schematic + heat zones) |
| Model Lab | **working** — dataset upload, model selection, run, summary table |
| Performance | shell + empty state (Phase 5: metrics charts) |

## Production build

```bash
cd webapp/frontend
npx vite build          # outputs dist/
```

Then: `../.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8790`

