"""
Pre-test data screens (Section 3.8.4) and dispersion helpers
(Section 3.8.7) — reusable across generators and analyses.

Screens (applied before any reported number):

1. Between-model variance. For each outcome, the proportion of
   scenarios on which all models agree is computed. Agreement
   above 99% flags the outcome as potentially degenerate and
   suspends the analysis pending inspection of raw predictions.

2. Discordance census. For every paired comparison the 2x2
   contingency table is computed and retained. An odds ratio of
   exactly zero or infinity indicates an empty discordant cell
   and triggers inspection.

3. Marginal reconciliation. Accuracy computed from each
   prediction vector is checked against the corresponding entry
   in the descriptive performance table. A discrepancy exceeding
   0.5 percentage points suspends the analysis.

Reporting standard (Section 3.8.7): point estimates are not
reported without an accompanying measure of dispersion. The
bootstrap helpers here provide that dispersion for holdout
estimates (single split, no folds); robustness-trial dispersion
comes from the robustness sweep files.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# SCREEN 1: BETWEEN-MODEL VARIANCE
# ============================================================

AGREEMENT_THRESHOLD = 0.99


def between_model_agreement(
    predictions: dict[str, np.ndarray],
) -> dict:

    """
    Proportion of scenarios on which all models agree.

    predictions: mapping model name -> 0/1 array over the same
    scenarios (aligned and same length).
    """

    names = sorted(predictions.keys())

    if len(names) < 2:

        return {
            "agreement": np.nan,
            "flagged": False,
            "n_scenarios": 0,
            "n_models": len(names),
        }

    matrix = np.column_stack(
        [predictions[name] for name in names]
    )

    all_agree = (
        (matrix.min(axis=1) == matrix.max(axis=1))
    )

    agreement = float(all_agree.mean())

    return {
        "agreement": agreement,
        "flagged": agreement > AGREEMENT_THRESHOLD,
        "n_scenarios": matrix.shape[0],
        "n_models": len(names),
    }


# ============================================================
# SCREEN 2: DISCORDANCE CENSUS
# ============================================================

def discordance_census(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict:

    """
    2x2 contingency table between two prediction vectors
    (a rows, b columns), retained per the screen. An odds ratio
    of exactly zero or infinity (empty discordant cell) flags
    inspection.
    """

    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    n11 = int(((pred_a == 1) & (pred_b == 1)).sum())
    n10 = int(((pred_a == 1) & (pred_b == 0)).sum())
    n01 = int(((pred_a == 0) & (pred_b == 1)).sum())
    n00 = int(((pred_a == 0) & (pred_b == 0)).sum())

    discordant = n10 + n01

    if n10 == 0 and n01 == 0:

        odds_ratio = np.nan
        empty_cell = True

    elif n01 == 0 or n10 == 0:

        odds_ratio = np.inf if n01 == 0 else 0.0
        empty_cell = True

    else:

        odds_ratio = (n11 * n00) / (n10 * n01)
        empty_cell = False

    return {
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "discordant": discordant,
        "odds_ratio": odds_ratio,
        "empty_cell": empty_cell,
    }


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    """Confusion counts of a prediction vector against truth."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": tp + fp + fn + tn,
    }


# ============================================================
# SCREEN 3: MARGINAL RECONCILIATION
# ============================================================

RECONCILIATION_TOLERANCE = 0.005  # 0.5 percentage points


def marginal_reconciliation(
    computed: float,
    recorded: float,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> dict:

    """
    Accuracy computed from the prediction vector vs the entry in
    the descriptive performance table. Discrepancy above the
    tolerance (0.5 pp) suspends the analysis.
    """

    if computed is None or recorded is None:

        return {
            "computed": computed,
            "recorded": recorded,
            "discrepancy": np.nan,
            "passed": False,
            "reason": "missing value",
        }

    computed = float(computed)
    recorded = float(recorded)

    if np.isnan(computed) or np.isnan(recorded):

        return {
            "computed": computed,
            "recorded": recorded,
            "discrepancy": np.nan,
            "passed": False,
            "reason": "missing value",
        }

    discrepancy = abs(float(computed) - float(recorded))

    return {
        "computed": float(computed),
        "recorded": float(recorded),
        "discrepancy": discrepancy,
        "passed": discrepancy <= tolerance,
        "reason": (
            None if discrepancy <= tolerance
            else f"discrepancy {discrepancy * 100:.2f} pp "
            f"exceeds {tolerance * 100:.1f} pp"
        ),
    }


def f1_bound_check(
    precision: float,
    recall: float,
    f1: float,
) -> dict:

    """
    Macro-averaged bound: F1 <= (precision + recall) / 2
    (harmonic mean cannot exceed the arithmetic mean). A violation
    indicates an arithmetic or transcription error in the source
    table.
    """

    bound = (precision + recall) / 2

    # Small numerical slack for rounding in recorded tables.
    eps = 1e-9

    passed = f1 <= bound + eps

    return {
        "bound": bound,
        "f1": f1,
        "passed": bool(passed),
    }


# ============================================================
# METRICS (zero-division safe)
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    counts = confusion_counts(y_true, y_pred)

    n = counts["total"]

    accuracy = (
        (counts["tp"] + counts["tn"]) / n if n else np.nan
    )

    precision = (
        counts["tp"] / (counts["tp"] + counts["fp"])
        if (counts["tp"] + counts["fp"]) else np.nan
    )

    recall = (
        counts["tp"] / (counts["tp"] + counts["fn"])
        if (counts["tp"] + counts["fn"]) else np.nan
    )

    if (
        np.isnan(precision)
        or np.isnan(recall)
        or (precision + recall) == 0
    ):

        f1 = np.nan

    else:

        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
    }


# ============================================================
# BOOTSTRAP DISPERSION (Section 3.8.7)
# ============================================================

def bootstrap_metric_sd(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict:

    """
    Bootstrap standard deviation of a metric over resampled test
    rows. Provides the dispersion measure required by Section
    3.8.7 for holdout estimates (single split, no folds).
    Returns the point estimate, the bootstrap SD, and the
    percentile interval.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    n = len(y_true)

    point = compute_metrics(y_true, y_pred)[metric]

    if n == 0:

        return {
            "point": np.nan,
            "sd": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    values = np.empty(iterations, dtype=float)

    for b in range(iterations):

        idx = rng.integers(0, n, n)

        values[b] = compute_metrics(
            y_true[idx], y_pred[idx]
        )[metric]

    values = values[~np.isnan(values)]

    if len(values) == 0:

        return {
            "point": point,
            "sd": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    return {
        "point": point,
        "sd": float(values.std(ddof=1)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }
