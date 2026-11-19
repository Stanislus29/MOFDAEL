"""SQLite persistence for uploaded datasets and inference runs.

Kept deliberately small: two tables (datasets, runs) using the
stdlib sqlite3 driver — no ORM, no migration machinery.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from backend import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id  TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    n_rows      INTEGER NOT NULL,
    has_targets INTEGER NOT NULL,
    has_location INTEGER NOT NULL,
    fault_rows  INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    dataset_id  TEXT,
    model_ids   TEXT NOT NULL,
    status      TEXT NOT NULL,
    summary_json TEXT,
    output_dir  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS networks (
    network_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    doc_json    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------


def insert_dataset(
    dataset_id: str,
    filename: str,
    stored_path: Path,
    n_rows: int,
    has_targets: bool,
    has_location: bool,
    fault_rows: int,
    report_json: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                filename,
                str(stored_path),
                n_rows,
                int(has_targets),
                int(has_location),
                fault_rows,
                report_json,
                now_iso(),
            ),
        )


def list_datasets() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_dataset(dataset_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------------
# RUNS
# ------------------------------------------------------------------


def insert_run(run_id: str, dataset_id: str | None, model_ids: list[str]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, dataset_id, ",".join(model_ids), "running", None, None, now_iso()),
        )


def finish_run(run_id: str, summary_json: str, output_dir: str, status: str = "done") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, summary_json = ?, output_dir = ? "
            "WHERE run_id = ?",
            (status, summary_json, output_dir, run_id),
        )


def get_run(run_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# NETWORKS
# ------------------------------------------------------------------


def insert_network(network_id: str, name: str, doc_json: str) -> None:
    ts = now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO networks VALUES (?, ?, ?, ?, ?)",
            (network_id, name, doc_json, ts, ts),
        )


def update_network(network_id: str, name: str, doc_json: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE networks SET name = ?, doc_json = ?, updated_at = ? "
            "WHERE network_id = ?",
            (name, doc_json, now_iso(), network_id),
        )


def upsert_network(network_id: str, name: str, doc_json: str) -> None:
    if get_network(network_id) is None:
        insert_network(network_id, name, doc_json)
    else:
        update_network(network_id, name, doc_json)


def list_networks() -> list[dict]:
    """List without the (potentially large) document body."""

    with _connect() as conn:
        rows = conn.execute(
            "SELECT network_id, name, created_at, updated_at "
            "FROM networks ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_network(network_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM networks WHERE network_id = ?", (network_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_network(network_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM networks WHERE network_id = ?", (network_id,)
        )
    return cur.rowcount > 0
