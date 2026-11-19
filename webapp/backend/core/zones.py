"""Fault-zone extraction: turns run predictions into spatial zones.

A "zone" is one fault event (the platform's event structure: rows
grouped by Loc resets) whose rows the model panel flagged as a fault.
Zones are what the Line Monitor renders as heat bands; they are
computed once per run and stored as zones.json.

Positioning rule (explicit, documented):
* Ground-truth ``Loc`` when the dataset carries it (the honest
  reference the models are judged against),
* otherwise the mean predicted location (Loc_pred) of the event's
  rows for models that have a location head,
* otherwise no position: the segment still reports detection status
  but cannot show a spatial band.

An event whose ground-truth targets are all zero but which was still
flagged is reported with ``suspected_false_positive = True`` so the
monitor can mark it as such instead of pretending it is a fault.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backend import config
from backend.core import preprocessing
from backend.core.types import RunResult

SEGMENT_COLUMN_CANDIDATES = ["segment", "line", "line_id", "segment_id"]


def find_segment_column(df: pd.DataFrame) -> str | None:
    for col in SEGMENT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


# ------------------------------------------------------------------
# ZONE CONTAINERS
# ------------------------------------------------------------------


@dataclass
class Zone:
    loc_mean_km: float | None
    loc_min_km: float | None
    loc_max_km: float | None
    n_rows: int
    flag_fraction: float          # mean consensus fraction across rows
    phases: dict                  # {"G":0/1, "C":..., "B":..., "A":...}
    position_source: str          # 'loc_truth' | 'loc_pred' | 'none'
    suspected_false_positive: bool = False


@dataclass
class SegmentZones:
    segment: str                  # the raw value from the dataset column
    n_rows: int
    n_flagged: int
    flagged_fraction: float
    zones: list[Zone] = field(default_factory=list)
    per_model: dict = field(default_factory=dict)
    has_location: bool = False


# ------------------------------------------------------------------
# CORE EXTRACTION
# ------------------------------------------------------------------


def build_segment_zones(
    df: pd.DataFrame,
    result: RunResult,
) -> list[SegmentZones]:
    """Extract per-segment fault zones from a completed run."""

    consensus = result.consensus
    if consensus is None:
        return []

    seg_col = find_segment_column(df)
    if seg_col is not None:
        segment_values = [str(v) for v in df[seg_col].unique()]
        row_segments = df[seg_col].astype(str)
    else:
        segment_values = ["(whole dataset)"]
        row_segments = pd.Series(
            ["(whole dataset)"] * len(df), index=df.index
        )

    has_loc_truth = config.LOCATION_COLUMN in df.columns
    has_targets = all(t in df.columns for t in config.TARGET_COLUMNS)

    # Event structure: same grouping the LSTM family uses (new event
    # whenever Loc resets). Without a Loc column every row is its own
    # event.
    row_event = pd.Series(-1, index=df.index, dtype=int)
    for eid, ev in enumerate(preprocessing.build_events(df)):
        row_event.loc[ev.index] = eid

    has_loc_pred = any(
        "Loc_pred" in res.predictions.columns
        for res in result.model_results.values()
        if not res.predictions.empty
    )

    out: list[SegmentZones] = []

    for seg in segment_values:
        mask = (row_segments == seg).to_numpy()
        seg_row_ids = df.index[mask]
        seg_cons = consensus[consensus["row_id"].isin(seg_row_ids)]
        flagged = seg_cons[seg_cons["n_flag_fault"] > 0]

        sz = SegmentZones(
            segment=seg,
            n_rows=int(mask.sum()),
            n_flagged=int(len(flagged)),
            flagged_fraction=round(len(flagged) / max(len(seg_cons), 1), 6),
            has_location=bool(has_loc_truth or has_loc_pred),
        )

        # per-model verdicts for this segment
        for model_id, res in result.model_results.items():
            p = res.predictions
            if p.empty:
                continue
            m = p[p["row_id"].isin(seg_row_ids)]
            entry = {
                "flags": int(m["overall_pred"].sum()),
                "n_rows": int(len(m)),
                "flag_fraction": round(float(m["overall_pred"].mean()), 6),
            }
            if "overall_confidence" in m.columns:
                entry["mean_confidence"] = round(
                    float(m["overall_confidence"].mean()), 6
                )
            sz.per_model[model_id] = entry

        # one zone per flagged fault event
        if len(flagged):
            event_ids = row_event.loc[flagged["row_id"]]
            sz.zones = [
                _event_zone(
                    df, ev_rows, has_loc_truth, has_targets, result
                )
                for _, ev_rows in flagged.groupby(event_ids.to_numpy())
            ]
            sz.zones.sort(
                key=lambda z: (
                    z.loc_mean_km is None,
                    z.loc_mean_km if z.loc_mean_km is not None else 0.0,
                )
            )

        out.append(sz)

    return out


def _event_zone(
    df: pd.DataFrame,
    ev_rows: pd.DataFrame,
    has_loc_truth: bool,
    has_targets: bool,
    result: RunResult,
) -> Zone:
    """Build one Zone from the flagged consensus rows of one event."""

    row_ids = ev_rows["row_id"].to_numpy()
    phases = {
        t: int(round(float(ev_rows[f"{t}_majority"].mean())))
        for t in config.TARGET_COLUMNS
        if f"{t}_majority" in ev_rows.columns
    }

    loc_truth = (
        df.loc[row_ids, config.LOCATION_COLUMN].to_numpy(dtype=float)
        if has_loc_truth
        else None
    )

    # predicted locations: mean across models that provide one
    loc_pred_frames = []
    for res in result.model_results.values():
        p = res.predictions
        if not p.empty and "Loc_pred" in p.columns:
            loc_pred_frames.append(p.set_index("row_id")["Loc_pred"])
    if loc_pred_frames:
        loc_pred = (
            pd.concat(loc_pred_frames, axis=1)
            .groupby(level=0)
            .mean()
            .reindex(row_ids)
            .mean(axis=1, skipna=True)
            .to_numpy()
        )
    else:
        loc_pred = None

    positions, source = _zone_positions(loc_truth, loc_pred)
    valid = ~np.isnan(positions)

    # ground truth says healthy everywhere, yet the panel flagged it
    false_positive = False
    if has_targets:
        targets = df.loc[row_ids, config.TARGET_COLUMNS].to_numpy()
        false_positive = bool((targets.sum(axis=1) == 0).all())

    return Zone(
        loc_mean_km=float(np.nanmean(positions)) if valid.any() else None,
        loc_min_km=float(np.nanmin(positions)) if valid.any() else None,
        loc_max_km=float(np.nanmax(positions)) if valid.any() else None,
        n_rows=int(len(ev_rows)),
        flag_fraction=round(float(ev_rows["flag_fraction"].mean()), 6),
        phases=phases,
        position_source=source,
        suspected_false_positive=false_positive,
    )


def _zone_positions(
    loc_truth: np.ndarray | None,
    loc_pred: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Choose the zone position source: truth first, predictions second."""

    if (
        loc_truth is not None
        and np.isfinite(loc_truth).any()
        and not np.allclose(np.nan_to_num(loc_truth, nan=-1.0), -1.0)
    ):
        return np.asarray(loc_truth, dtype=float), "loc_truth"
    if loc_pred is not None:
        return np.asarray(loc_pred, dtype=float), "loc_pred"
    n = len(loc_truth) if loc_truth is not None else (
        len(loc_pred) if loc_pred is not None else 0
    )
    return np.full(n, np.nan), "none"


# ------------------------------------------------------------------
# SERIALIZATION
# ------------------------------------------------------------------


def zones_to_dict(segments: list[SegmentZones]) -> dict:
    return {
        "version": 1,
        "segments": [asdict(s) for s in segments],
    }


def write_zones_file(run_dir: Path, segments: list[SegmentZones]) -> Path:
    path = run_dir / "zones.json"
    path.write_text(json.dumps(zones_to_dict(segments), indent=2, default=str))
    return path
