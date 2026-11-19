"""Shared result containers for the inference engine and zones.

Kept free of heavy imports so both engine and zones can use them
without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ModelResult:
    model_id: str
    predictions: pd.DataFrame
    latency: dict
    notes: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    dataset_id: str
    model_results: dict[str, ModelResult]
    consensus: pd.DataFrame | None
    summary: dict
    created_at: str
    zones: list | None = None
