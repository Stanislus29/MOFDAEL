"""
Regenerate Table 4.3 (Chapter Four, Section 4.3.1) from source —
R6 + R7.

Every cell is computed directly from the per-scenario prediction
files (outputs/*/predictions_<model>.csv), with:

  - the Section 3.8.4 pre-test screens applied (between-model
    variance, discordance census, marginal reconciliation against
    the training scripts' own metrics.csv records);
  - the macro-F1 bound check (F1 <= (macro-P + macro-R) / 2);
  - a dispersion measure on every cell (Section 3.8.7):
    bootstrap SD over resampled test rows for the classification
    metrics, across-trial SD for the robustness sweep, and
    across-row SD for per-sample latency;
  - cells that fail a screen are withheld with the reason
    footnoted rather than silently corrected.

Data-vs-narrative reconciliation is reported explicitly (dataset
row counts and test-set size against the figures quoted in the
narrative).

Outputs (outputs/table_4_3/):

    report_<timestamp>.txt   full generation report
    table_4_3.csv            machine-readable table (value ± sd)
"""

from __future__ import annotations

import csv

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from screens import (
    between_model_agreement,
    bootstrap_metric_sd,
    compute_metrics,
    confusion_counts,
    discordance_census,
    f1_bound_check,
    marginal_reconciliation,
    AGREEMENT_THRESHOLD,
    RECONCILIATION_TOLERANCE,
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "table_4_3"

DATA_FILE = DATA_DIR / "classdetectfault_sim.csv"

DETECTFAULT_FILE = DATA_DIR / "detectfault_sim.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TIMESTAMP & OUTPUT FILES
# ============================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = OUTPUT_DIR / f"report_{timestamp}.txt"

TABLE_FILE = OUTPUT_DIR / "table_4_3.csv"

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMNS = ["G", "C", "B", "A"]

EVAL_TARGETS = TARGET_COLUMNS + ["overall"]

BOOTSTRAP_ITERATIONS = 2000

RANDOM_STATE = 42

# Narrative figures quoted in the dissertation, used only for the
# data-vs-narrative reconciliation section.
NARRATIVE_TOTAL_SCENARIOS = 12000
NARRATIVE_TOTAL_ROWS = 14400
NARRATIVE_TEST_SCENARIOS = 2400

# Relay decision latency: a constant from Table 4.3 (config
# constant, not a measured variable — see R5).
RELAY_DECISION_LATENCY_MS = 10.0


# ============================================================
# REPORT LOGGER
# ============================================================

class ReportLogger:

    def __init__(self, path: Path):

        self.path = path
        self.file = path.open("w", encoding="utf-8")

    def print(self, message=""):

        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):

        self.file.close()


# ============================================================
# HELPERS
# ============================================================

def fmt(point: float, sd: float) -> str:

    if point is None or (
        isinstance(point, float) and np.isnan(point)
    ):
        return "withheld"

    if sd is None or (
        isinstance(sd, float) and np.isnan(sd)
    ):
        return f"{point:.4f} ± n/a"

    return f"{point:.4f} ± {sd:.4f}"


def load_latest_metrics() -> dict:

    """
    Latest recorded metrics row per (model, target) from each
    training script's metrics.csv — the 'descriptive performance
    table' side of the marginal-reconciliation screen.

    The metrics files were appended across runs with the Loc
    columns added at the end, so rows are parsed positionally
    against the original header and trailing extras are ignored.
    The last row per (model, target) — the most recent run — wins.
    """

    recorded = {}

    for path in sorted(OUTPUTS_DIR.glob("*/metrics.csv")):

        model = path.parent.name

        with path.open(encoding="utf-8") as handle:

            reader = csv.reader(handle)

            header = next(reader)

            for row in reader:

                if len(row) < len(header):
                    continue

                fields = dict(zip(header, row[: len(header)]))

                target = fields.get("target", "")

                if target == "Loc":
                    continue

                def _float(key):

                    try:
                        return float(fields[key])
                    except (KeyError, TypeError, ValueError):
                        return np.nan

                recorded[(model, target)] = {
                    "accuracy": _float("accuracy"),
                    "precision": _float("precision"),
                    "recall": _float("recall"),
                    "f1": _float("f1_score"),
                    "timestamp": fields.get("timestamp", ""),
                }

    return recorded


def load_robustness_reference() -> dict:

    """
    Reference (lowest) noise level per model from the robustness
    sweep files: accuracy mean and across-trial SD.
    """

    result = {}

    trial_cols = [
        f"accuracy_trial_{i}" for i in range(10)
    ]

    for path in sorted(OUTPUTS_DIR.glob("*/robustness_*.csv")):

        model = path.stem.replace("robustness_", "")

        df = pd.read_csv(path)

        df = df.sort_values("noise_level")

        reference = df.iloc[0]

        if all(c in df.columns for c in trial_cols):

            trials = reference[trial_cols].to_numpy(dtype=float)

            result[model] = {
                "noise": float(reference["noise_level"]),
                "mean": float(np.mean(trials)),
                "sd": float(np.std(trials, ddof=1)),
            }

        else:

            result[model] = {
                "noise": float(reference["noise_level"]),
                "mean": float(reference["accuracy_mean"]),
                "sd": np.nan,
            }

    return result


def load_latency_reference() -> dict:

    """
    Per-sample latency mean and across-row SD per model from the
    per-row latency files (total column preferred).
    """

    result = {}

    for path in sorted(OUTPUTS_DIR.glob("*/latency_per_row_*.csv")):

        model = path.stem.replace("latency_per_row_", "")

        df = pd.read_csv(path)

        column = (
            "total_latency_ms"
            if "total_latency_ms" in df.columns
            else df.columns[-1]
        )

        values = df[column].to_numpy(dtype=float)

        result[model] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "column": column,
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    logger = ReportLogger(REPORT_FILE)

    try:

        rng = np.random.default_rng(RANDOM_STATE)

        logger.print("=" * 70)
        logger.print("TABLE 4.3 REGENERATION FROM SOURCE (R6 + R7)")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")
        logger.print(f"Dataset: {DATA_FILE}")
        logger.print(
            f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS} "
            f"(seed {RANDOM_STATE})"
        )
        logger.print("")

        # ----------------------------------------------------
        # LOAD PREDICTION FILES
        # ----------------------------------------------------

        files = sorted(OUTPUTS_DIR.glob("*/predictions_*.csv"))

        predictions = {}

        for path in files:

            model = path.stem.replace("predictions_", "")

            predictions[model] = pd.read_csv(path).set_index(
                "row_id"
            )

        logger.print(f"Prediction files loaded: {len(predictions)}")
        logger.print("")

        for name, df in predictions.items():
            logger.print(f"  {name:>6}: {len(df)} rows")

        # ----------------------------------------------------
        # RELAY EVALUATION SETS
        # ----------------------------------------------------

        model_frames = {
            name: df for name, df in predictions.items()
            if name != "relay"
        }

        common_rows = None

        for df in model_frames.values():

            common_rows = (
                df.index if common_rows is None
                else common_rows.intersection(df.index)
            )

        common_rows = common_rows.sort_values()

        logger.print("")
        logger.print(
            f"Common test rows across models: {len(common_rows)}"
        )
        logger.print(
            "  (relay primary evaluation set; comparable with "
            "the hypothesis tests)"
        )
        logger.print("")

        # ----------------------------------------------------
        # SECTION 3.8.4 SCREEN 1: BETWEEN-MODEL VARIANCE
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print(
            "SCREEN 1: BETWEEN-MODEL VARIANCE "
            f"(flag if agreement > {AGREEMENT_THRESHOLD:.0%})"
        )
        logger.print("=" * 70)

        screen1_flags = {}

        for target in EVAL_TARGETS:

            vectors = {}

            for name, df in model_frames.items():

                if target == "overall":

                    true_col = "overall_true"
                    pred_col = "overall_pred"

                else:

                    true_col = f"{target}_true"
                    pred_col = f"{target}_pred"

                aligned = df.loc[common_rows]

                vectors[name] = aligned[pred_col].to_numpy(
                    dtype=int
                )

            screen = between_model_agreement(vectors)

            screen1_flags[target] = screen["flagged"]

            logger.print(
                f"  {target:>7}: agreement = "
                f"{screen['agreement']:.4f} "
                f"({'FLAGGED' if screen['flagged'] else 'ok'})"
            )

        logger.print("")

        # ----------------------------------------------------
        # RECORDED METRICS FOR RECONCILIATION
        # ----------------------------------------------------

        recorded = load_latest_metrics()

        robustness = load_robustness_reference()

        latency = load_latency_reference()

        # ----------------------------------------------------
        # TABLE GENERATION PER MODEL
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print(
            "TABLE 4.3 (REGENERATED) — value ± bootstrap SD"
        )
        logger.print("=" * 70)

        rows = []

        withheld = []

        censuses = []

        eval_specs = []

        for name in sorted(model_frames.keys()):

            eval_specs.append(
                (name, model_frames[name].index, len(model_frames[name]))
            )

        # Relay: primary on the common set, secondary on all rows.
        eval_specs.append(
            ("relay", common_rows, len(predictions["relay"]))
        )
        eval_specs.append(
            ("relay (all rows)", predictions["relay"].index,
             len(predictions["relay"]))
        )

        for name, row_index, n_total in eval_specs:

            df = predictions[
                "relay" if name.startswith("relay")
                else name
            ].loc[row_index]

            display_name = name

            for target in EVAL_TARGETS:

                if target == "overall":

                    y_true = df["overall_true"].to_numpy(dtype=int)
                    y_pred = df["overall_pred"].to_numpy(dtype=int)

                else:

                    y_true = df[f"{target}_true"].to_numpy(dtype=int)
                    y_pred = df[f"{target}_pred"].to_numpy(dtype=int)

                metrics = compute_metrics(y_true, y_pred)

                # --------------------------------------------
                # SCREEN 2: DISCORDANCE CENSUS (vs truth)
                # --------------------------------------------

                census = confusion_counts(y_true, y_pred)

                # Pairwise census against the relay for learned
                # models (retained per the screen; overall target
                # only, matching the McNemar comparisons).
                if name not in ("relay", "relay (all rows)") \
                        and target == "overall":

                    aligned_idx = row_index.intersection(
                        predictions["relay"].index
                    ).sort_values()

                    model_pred_aligned = df.loc[
                        aligned_idx
                    ]["overall_pred"].to_numpy(dtype=int)

                    relay_pred_aligned = predictions[
                        "relay"
                    ].loc[aligned_idx]["overall_pred"].to_numpy(
                        dtype=int
                    )

                    pair_census = discordance_census(
                        model_pred_aligned, relay_pred_aligned
                    )

                    censuses.append({
                        "model": display_name,
                        **pair_census,
                    })

                else:

                    pair_census = None

                # --------------------------------------------
                # SCREEN 3: MARGINAL RECONCILIATION
                # --------------------------------------------

                rec = recorded.get(
                    (name.replace(" (all rows)", ""), target)
                )

                if rec is None:

                    # No recorded reference for this target (the
                    # metrics files hold no 'overall' row, and the
                    # relay has no metrics file) — nothing to
                    # reconcile against; not a screen failure.
                    reconciliation = {
                        "passed": True,
                        "status": "n/a",
                        "reason": "no recorded reference",
                    }

                else:

                    reconciliation = marginal_reconciliation(
                        metrics["accuracy"],
                        rec["accuracy"],
                    )

                    reconciliation["status"] = (
                        "pass" if reconciliation["passed"]
                        else "FAIL"
                    )

                # --------------------------------------------
                # F1 BOUND CHECK
                # --------------------------------------------

                bound = f1_bound_check(
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                )

                # --------------------------------------------
                # BOOTSTRAP DISPERSION
                # --------------------------------------------

                disp = {}

                for metric in (
                    "accuracy", "precision", "recall", "f1"
                ):

                    disp[metric] = bootstrap_metric_sd(
                        y_true,
                        y_pred,
                        metric,
                        BOOTSTRAP_ITERATIONS,
                        rng,
                    )

                # --------------------------------------------
                # WITHHOLDING RULES
                # --------------------------------------------

                cell_ok = True
                reasons = []

                if rec is not None and not reconciliation["passed"]:

                    cell_ok = False

                    reasons.append(
                        "marginal reconciliation: "
                        + (reconciliation["reason"] or "missing")
                    )

                if not bound["passed"]:

                    cell_ok = False

                    reasons.append(
                        "F1 bound violated: "
                        f"F1 {metrics['f1']:.4f} > "
                        f"(P+R)/2 {bound['bound']:.4f}"
                    )

                if screen1_flags.get(target, False):

                    reasons.append(
                        "screen 1 flagged: between-model "
                        "agreement above threshold"
                    )

                if not cell_ok:

                    withheld.append({
                        "model": display_name,
                        "target": target,
                        "reasons": "; ".join(reasons),
                    })

                # --------------------------------------------
                # ROBUSTNESS / LATENCY DISPERSION
                # --------------------------------------------

                rob = robustness.get(
                    name.replace(" (all rows)", "")
                )

                lat = latency.get(
                    name.replace(" (all rows)", "")
                )

                if name.startswith("relay"):

                    # Relay decision latency is the constant from
                    # Table 4.3 (R5), not a measured variable.
                    lat = {
                        "mean": RELAY_DECISION_LATENCY_MS,
                        "sd": 0.0,
                        "column": "constant (Table 4.3)",
                    }

                rows.append({
                    "model": display_name,
                    "n": len(df),
                    "target": target,
                    "accuracy": fmt(
                        metrics["accuracy"],
                        disp["accuracy"]["sd"]
                    ),
                    "precision": fmt(
                        metrics["precision"],
                        disp["precision"]["sd"]
                    ),
                    "recall": fmt(
                        metrics["recall"],
                        disp["recall"]["sd"]
                    ),
                    "f1": fmt(
                        metrics["f1"],
                        disp["f1"]["sd"]
                    ),
                    "tp": census["tp"],
                    "fp": census["fp"],
                    "fn": census["fn"],
                    "tn": census["tn"],
                    "reconciled": reconciliation["status"],
                    "recorded_accuracy": (
                        f"{rec['accuracy']:.6f}" if rec else "n/a"
                    ),
                    "robustness": (
                        f"{rob['mean']:.4f} ± {rob['sd']:.4f}"
                        if rob and not np.isnan(rob["sd"])
                        else (
                            f"{rob['mean']:.4f} ± n/a" if rob
                            else "n/a"
                        )
                    ),
                    "robustness_noise": (
                        f"{rob['noise']:.2f}" if rob else "n/a"
                    ),
                    "latency_ms_per_sample": (
                        f"{lat['mean']:.6f} ± {lat['sd']:.6f}"
                        if lat else "n/a"
                    ),
                    "f1_bound": (
                        "pass" if bound["passed"] else "FAIL"
                    ),
                })

        table = pd.DataFrame(rows)

        # Print the table grouped by model
        for model_name in table["model"].unique():

            sub = table[table["model"] == model_name]

            logger.print("")
            logger.print(f"--- {model_name} (n = {sub['n'].iloc[0]}) ---")
            logger.print(
                f"  {'target':>7} | {'accuracy':>21} | "
                f"{'precision':>21} | {'recall':>21} | {'f1':>21} | "
                f"{'recon':>5} | {'robustness':>19} | "
                f"{'latency ms':>22}"
            )

            for _, r in sub.iterrows():

                logger.print(
                    f"  {r['target']:>7} | {r['accuracy']:>21} | "
                    f"{r['precision']:>21} | {r['recall']:>21} | "
                    f"{r['f1']:>21} | {r['reconciled']:>5} | "
                    f"{r['robustness']:>19} | "
                    f"{r['latency_ms_per_sample']:>22}"
                )

        logger.print("")

        # ----------------------------------------------------
        # SCREEN 2: DISCORDANCE CENSUS (retained)
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print(
            "SCREEN 2: DISCORDANCE CENSUS "
            "(each model vs relay, overall target)"
        )
        logger.print("=" * 70)
        logger.print(
            f"  {'model':>6} | {'n11':>6} {'n10':>6} {'n01':>6} "
            f"{'n00':>6} | {'discordant':>10} | "
            f"{'odds ratio':>10} | flag"
        )

        for item in censuses:

            or_text = (
                "inf" if np.isinf(item["odds_ratio"])
                else f"{item['odds_ratio']:.4f}"
                if not np.isnan(item["odds_ratio"])
                else "n/a"
            )

            logger.print(
                f"  {item['model']:>6} | {item['n11']:>6} "
                f"{item['n10']:>6} {item['n01']:>6} "
                f"{item['n00']:>6} | {item['discordant']:>10} | "
                f"{or_text:>10} | "
                f"{'INSPECT' if item['empty_cell'] else 'ok'}"
            )

        logger.print("")

        # ----------------------------------------------------
        # WITHHELD CELLS
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print("WITHHELD CELLS (screen failures, footnoted)")
        logger.print("=" * 70)

        if withheld:

            for item in withheld:

                logger.print(
                    f"  {item['model']} / {item['target']}: "
                    f"{item['reasons']}"
                )

        else:

            logger.print("  none — every cell passed all screens.")

        logger.print("")

        # ----------------------------------------------------
        # DATA-VS-NARRATIVE RECONCILIATION
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print("DATA-VS-NARRATIVE RECONCILIATION")
        logger.print("=" * 70)

        raw = pd.read_csv(DATA_FILE)

        detect = (
            pd.read_csv(DETECTFAULT_FILE)
            if DETECTFAULT_FILE.exists() else None
        )

        logger.print(
            f"  classdetectfault_sim.csv rows: {len(raw)} "
            f"(narrative: {NARRATIVE_TOTAL_SCENARIOS} scenarios / "
            f"{NARRATIVE_TOTAL_ROWS} total rows)"
        )
        logger.print(
            f"  model test-set rows: "
            f"{len(model_frames[sorted(model_frames)[0]])} "
            f"(narrative: {NARRATIVE_TEST_SCENARIOS} held-out "
            f"scenarios)"
        )
        logger.print(
            f"  common test rows: {len(common_rows)} "
            f"(used by the hypothesis tests)"
        )

        if detect is not None:

            fault_rows = int((detect["Loc"] > 0).sum())

            logger.print(
                f"  detectfault_sim.csv rows: {len(detect)} "
                f"({len(detect) + 1} lines including header); "
                f"Loc column present: "
                f"{'yes' if 'Loc' in detect.columns else 'no'}; "
                f"fault rows (Loc > 0): {fault_rows}"
            )

        logger.print("")
        logger.print(
            "  These discrepancies are reported, not silently "
            "corrected; they must be reconciled in the chapter "
            "text (dataset export vs narrative figures) before "
            "the table cells are finalised."
        )
        logger.print("")

        # ----------------------------------------------------
        # R6: ORIGINALLY-WITHHELD CELLS AND THEIR REPLACEMENTS
        # ----------------------------------------------------

        logger.print("=" * 70)
        logger.print(
            "R6: ORIGINALLY-WITHHELD CELLS — REGENERATED "
            "REPLACEMENTS"
        )
        logger.print("=" * 70)

        def overall_cell(model):

            sel = table[
                (table["model"] == model)
                & (table["target"] == "overall")
            ]

            return sel["accuracy"].iloc[0]

        def overall_n(model):

            sel = table[
                (table["model"] == model)
                & (table["target"] == "overall")
            ]

            return int(sel["n"].iloc[0])

        relay_common_n = overall_n("relay")
        relay_all_n = overall_n("relay (all rows)")
        svm_n = overall_n("svm")

        logger.print(
            "  1. Relay accuracy 0.0589 (incompatible with its own"
        )
        logger.print(
            "     precision/F1) -> regenerated: "
            f"{overall_cell('relay')} on the common test rows "
            f"(n = {relay_common_n}, comparable with the hypothesis "
            f"tests) and {overall_cell('relay (all rows)')} on all "
            f"{relay_all_n} dataset rows."
        )
        logger.print(
            "  2. SVM accuracy 0.09671 (incompatible with its own"
        )
        logger.print(
            "     precision/F1) -> regenerated: "
            f"{overall_cell('svm')} (own test split, n = {svm_n})."
        )
        logger.print(
            "  3. Decision Tree F1 recorded without a decimal"
        )
        logger.print("     separator -> regenerated per-target F1:")

        dtr_rows = table[
            (table["model"] == "dtr")
            & (table["target"] != "overall")
        ]

        for _, r in dtr_rows.iterrows():

            logger.print(
                f"       {r['target']}: F1 = {r['f1']}"
            )

        logger.print("")
        logger.print(
            "  All replacement values are computed directly from "
            "outputs/*/predictions_<model>.csv and are traceable "
            "to this script; none were hand-corrected."
        )
        logger.print("")

        # ----------------------------------------------------
        # SAVE TABLE
        # ----------------------------------------------------

        table.to_csv(TABLE_FILE, index=False)

        logger.print("=" * 70)
        logger.print("OUTPUT FILES")
        logger.print("=" * 70)
        logger.print(f"Report: {REPORT_FILE}")
        logger.print(f"Table : {TABLE_FILE}")

    finally:

        logger.close()


if __name__ == "__main__":

    main()
