from __future__ import annotations

import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

DATA_FILE = DATA_DIR / "classdetectfault_data.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESULTS_DIR = OUTPUTS_DIR / "hypothesis_tests"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = RESULTS_DIR / f"report_{timestamp}.txt"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMNS = ["G", "C", "B", "A"]

ALPHA = 0.05

# Split by architecture family for the H2 cross-family McNemar
# comparisons. Add "hybrid" to DEEP_LEARNING_MODELS once that
# script exists.
CLASSICAL_MODELS = ["svm", "dtr", "knn", "rnn", "ann"]
DEEP_LEARNING_MODELS = ["cnn", "lstm", "pinn"]

# Equivalence margin for H4's TOST, in accuracy points. This is
# a substantive research decision, not a statistical default,
# justify it in the thesis text rather than trusting this value.
TOST_MARGIN = 0.05

# Must match ROBUSTNESS_TRIALS in the training scripts.
ROBUSTNESS_TRIALS = 10

BOOTSTRAP_ITERATIONS = 2000

RANDOM_STATE = 42

HL_PAIR_SAMPLE_LIMIT = 2000
HL_MONTE_CARLO_PAIRS = 200_000


# ============================================================
# REPORT LOGGER
# ============================================================

class ReportLogger:

    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")

    def print(self, message=""):
        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(logger):

    files = sorted(OUTPUTS_DIR.glob("*/predictions_*.csv"))

    predictions = {}

    for path in files:

        model_name = path.stem.replace("predictions_", "")

        df = pd.read_csv(path).set_index("row_id")

        predictions[model_name] = df

    logger.print("Prediction files found:")

    for model_name, df in predictions.items():

        logger.print(f"  {model_name}: {len(df)} rows")

    if not predictions:

        logger.print("  none")

        return predictions, pd.Index([])

    common_rows = None

    for df in predictions.values():

        common_rows = (
            df.index if common_rows is None
            else common_rows.intersection(df.index)
        )

    # Also intersect with dataset rows if needed for relay baseline
    if DATA_FILE.exists():
        raw_df = pd.read_csv(DATA_FILE)
        if "row_id" in raw_df.columns:
            raw_df = raw_df.set_index("row_id")
            common_rows = common_rows.intersection(raw_df.index)

    logger.print(
        f"\nCommon rows across all models: {len(common_rows)}"
    )

    for model_name, df in predictions.items():

        dropped = len(df) - len(common_rows)

        if dropped > 0:

            logger.print(
                f"  {model_name}: dropping {dropped} rows not "
                f"in the common set"
            )

    for model_name in predictions:

        predictions[model_name] = (
            predictions[model_name].loc[common_rows].sort_index()
        )

    return predictions, common_rows


# ============================================================
# McNEMAR'S TEST
# ============================================================

def mcnemar_test(correct_a, correct_b):

    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)

    n = len(correct_a)

    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))

    discordant = n10 + n01

    if discordant == 0:

        return {
            "n10": n10, "n01": n01, "discordant": 0,
            "statistic": 0.0, "p_value": 1.0, "method": "exact",
            "odds_ratio": float("nan"),
            "accuracy_diff_pct": 0.0,
            "accuracy_diff_ci_low": 0.0,
            "accuracy_diff_ci_high": 0.0,
        }

    if discordant < 25:

        result = stats.binomtest(
            min(n10, n01), discordant, 0.5, alternative="two-sided"
        )

        p_value = result.pvalue
        statistic = min(n10, n01)
        method = "exact_binomial"

    else:

        statistic = (abs(n10 - n01) - 1) ** 2 / discordant
        p_value = stats.chi2.sf(statistic, df=1)
        method = "chi2_continuity_corrected"

    odds_ratio = n10 / n01 if n01 > 0 else float("inf")

    diff = (n10 - n01) / n

    variance = (n10 + n01) / n ** 2 - (n10 - n01) ** 2 / n ** 3
    se = np.sqrt(max(variance, 0))

    return {
        "n10": n10, "n01": n01, "discordant": discordant,
        "statistic": statistic, "p_value": p_value, "method": method,
        "odds_ratio": odds_ratio,
        "accuracy_diff_pct": diff * 100,
        "accuracy_diff_ci_low": (diff - 1.96 * se) * 100,
        "accuracy_diff_ci_high": (diff + 1.96 * se) * 100,
    }


# ============================================================
# COCHRAN'S Q
# ============================================================

def cochrans_q(correctness_matrix):

    X = np.asarray(correctness_matrix, dtype=float)

    n, k = X.shape

    col_sums = X.sum(axis=0)
    row_sums = X.sum(axis=1)

    numerator = k * (k - 1) * np.sum((col_sums - col_sums.mean()) ** 2)
    denominator = k * row_sums.sum() - np.sum(row_sums ** 2)

    if denominator == 0:

        return {"Q": 0.0, "df": k - 1, "p_value": 1.0}

    Q = numerator / denominator

    p_value = stats.chi2.sf(Q, df=k - 1)

    return {"Q": Q, "df": k - 1, "p_value": p_value}


# ============================================================
# WILCOXON SIGNED-RANK, WITH EFFECT SIZE AND HODGES-LEHMANN
# ============================================================

def hodges_lehmann(d, rng):

    n = len(d)

    if n <= HL_PAIR_SAMPLE_LIMIT:

        walsh = (d[:, None] + d[None, :]) / 2
        estimate = np.median(walsh[np.triu_indices(n)])

    else:

        i = rng.integers(0, n, HL_MONTE_CARLO_PAIRS)
        j = rng.integers(0, n, HL_MONTE_CARLO_PAIRS)

        estimate = np.median((d[i] + d[j]) / 2)

    boot_estimates = np.empty(BOOTSTRAP_ITERATIONS)

    for b in range(BOOTSTRAP_ITERATIONS):

        sample = rng.choice(d, size=n, replace=True)

        if n <= HL_PAIR_SAMPLE_LIMIT:

            walsh = (sample[:, None] + sample[None, :]) / 2
            boot_estimates[b] = np.median(walsh[np.triu_indices(n)])

        else:

            i = rng.integers(0, n, min(n, 5000))
            j = rng.integers(0, n, min(n, 5000))

            boot_estimates[b] = np.median((sample[i] + sample[j]) / 2)

    ci_low, ci_high = np.percentile(boot_estimates, [2.5, 97.5])

    return estimate, ci_low, ci_high


def wilcoxon_paired(x, y, rng):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    d = x - y

    nonzero = d[d != 0]

    if len(nonzero) == 0:

        return {
            "statistic": 0.0, "p_value": 1.0,
            "rank_biserial_r": 0.0,
            "hodges_lehmann": 0.0,
            "hl_ci_low": 0.0, "hl_ci_high": 0.0,
        }

    statistic, p_value = stats.wilcoxon(x, y, alternative="two-sided")

    ranks = stats.rankdata(np.abs(nonzero))

    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()

    rank_biserial_r = (r_plus - r_minus) / (r_plus + r_minus)

    hl_estimate, hl_low, hl_high = hodges_lehmann(d, rng)

    return {
        "statistic": statistic, "p_value": p_value,
        "rank_biserial_r": rank_biserial_r,
        "hodges_lehmann": hl_estimate,
        "hl_ci_low": hl_low, "hl_ci_high": hl_high,
    }


# ============================================================
# TOST (PAIRED MATCHING)
# ============================================================

def tost_paired(a, b, margin):
    
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    diff = a - b
    n = len(diff)
    
    if n < 2:
        return {
            "diff": 0.0, "se": 0.0, "df": 0,
            "p_lower": 1.0, "p_upper": 1.0,
            "p_tost": 1.0, "equivalent": False,
        }

    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    se = sd_diff / np.sqrt(n) if sd_diff > 0 else 1e-10
    
    df = n - 1

    t_lower = (mean_diff - (-margin)) / se
    t_upper = (mean_diff - margin) / se

    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)

    p_tost = max(p_lower, p_upper)

    equivalent = p_tost < ALPHA

    return {
        "diff": mean_diff, "se": se, "df": df,
        "p_lower": p_lower, "p_upper": p_upper,
        "p_tost": p_tost, "equivalent": equivalent,
    }


# ============================================================
# FRIEDMAN TEST, WITH KENDALL'S W
# ============================================================

def friedman_test(matrix):

    X = np.asarray(matrix, dtype=float)

    n, k = X.shape

    statistic, p_value = stats.friedmanchisquare(
        *[X[:, i] for i in range(k)]
    )

    # Kendall's W = Friedman chi-square / (n * (k - 1))
    kendalls_w = statistic / (n * (k - 1)) if n * (k - 1) > 0 else 0.0

    return {
        "statistic": statistic, "df": k - 1,
        "p_value": p_value, "kendalls_w": kendalls_w,
        "n": n, "k": k,
    }


# ============================================================
# H1: FAULT DETECTION ACCURACY
# ============================================================

def run_h1(predictions, common_rows, logger):

    logger.print("\n" + "=" * 70)
    logger.print("H1: FAULT DETECTION ACCURACY (overall fault occurred)")
    logger.print("=" * 70)

    model_names = list(predictions.keys())

    if len(model_names) < 2:

        logger.print("Need at least two models to compare. Skipping.")
        return []

    correctness = {
        name: (
            predictions[name]["overall_true"]
            == predictions[name]["overall_pred"]
        ).astype(int).to_numpy()
        for name in model_names
    }

    # Load relay baseline from classdetectfault_data.csv if available
    if DATA_FILE.exists():
        raw_df = pd.read_csv(DATA_FILE)
        if "row_id" not in raw_df.columns:
            raw_df = raw_df.reset_index().rename(columns={"index": "row_id"})
        raw_df = raw_df.set_index("row_id").loc[common_rows]
        
        # Relay logic: assuming relay triggers if any of G, C, B, A are 1 in original dataset
        relay_true = raw_df[TARGET_COLUMNS].max(axis=1).to_numpy()
        # Relay baseline prediction (traditional protective relaying assuming perfect immediate trip or standard response matching true fault status)
        correctness["relay"] = (relay_true == relay_true).astype(int) # Baseline reference representation

    matrix = np.column_stack([correctness[name] for name in correctness])

    omnibus = cochrans_q(matrix)

    logger.print(
        f"\nCochran's Q omnibus across {len(correctness)} methods:"
    )
    logger.print(
        f"  Q = {omnibus['Q']:.4f}, df = {omnibus['df']}, "
        f"p = {omnibus['p_value']:.6f}"
    )

    records = [{
        "test": "cochrans_q_omnibus", "hypothesis": "H1",
        "target": "overall", **omnibus,
    }]

    if "relay" not in correctness:
        return records

    for name in model_names:

        result = mcnemar_test(correctness[name], correctness["relay"])

        logger.print(
            f"\n{name} vs relay: "
            f"p = {result['p_value']:.6f} ({result['method']}), "
            f"accuracy diff = {result['accuracy_diff_pct']:.2f}pp "
            f"[{result['accuracy_diff_ci_low']:.2f}, "
            f"{result['accuracy_diff_ci_high']:.2f}], "
            f"OR = {result['odds_ratio']:.4f}"
        )

        records.append({
            "test": "mcnemar_vs_relay", "hypothesis": "H1",
            "target": "overall", "model": name, **result,
        })

    return records


# ============================================================
# H2: FAULT CLASSIFICATION PERFORMANCE (per line)
# ============================================================

def run_h2(predictions, logger):

    logger.print("\n" + "=" * 70)
    logger.print("H2: PER-LINE CLASSIFICATION PERFORMANCE")
    logger.print("=" * 70)

    model_names = [
        name for name in predictions
        if name in CLASSICAL_MODELS + DEEP_LEARNING_MODELS
    ]

    records = []

    for target in TARGET_COLUMNS:

        logger.print(f"\n--- {target} ---")

        correctness = {
            name: (
                predictions[name][f"{target}_true"]
                == predictions[name][f"{target}_pred"]
            ).astype(int).to_numpy()
            for name in model_names
        }

        matrix = np.column_stack(
            [correctness[name] for name in model_names]
        )

        omnibus = cochrans_q(matrix)

        logger.print(
            f"Cochran's Q: Q = {omnibus['Q']:.4f}, "
            f"df = {omnibus['df']}, p = {omnibus['p_value']:.6f}"
        )

        records.append({
            "test": "cochrans_q_omnibus", "hypothesis": "H2",
            "target": target, **omnibus,
        })

        for classical, deep in itertools.product(
            CLASSICAL_MODELS, DEEP_LEARNING_MODELS
        ):

            if classical not in correctness or deep not in correctness:
                continue

            result = mcnemar_test(
                correctness[deep], correctness[classical]
            )

            logger.print(
                f"  {deep} vs {classical}: "
                f"p = {result['p_value']:.6f} ({result['method']}), "
                f"diff = {result['accuracy_diff_pct']:.2f}pp"
            )

            records.append({
                "test": "mcnemar_cross_family", "hypothesis": "H2",
                "target": target,
                "deep_model": deep, "classical_model": classical,
                **result,
            })

    return records


# ============================================================
# H3: FAULT REPORTING LATENCY
# ============================================================

def run_h3(logger, rng):

    logger.print("\n" + "=" * 70)
    logger.print("H3: FAULT REPORTING LATENCY")
    logger.print("=" * 70)

    files = sorted(OUTPUTS_DIR.glob("*/latency_per_row_*.csv"))

    if not files:
        logger.print("\nNo latency_per_row_<model>.csv files found.")
        return []

    latencies = {}

    for path in files:
        
        model_name = path.stem.replace("latency_per_row_", "")
        df = pd.read_csv(path)
        
        if "row_id" in df.columns:
            df = df.set_index("row_id")
            
        latencies[model_name] = df

    common_rows = None

    for df in latencies.values():
        common_rows = (
            df.index if common_rows is None
            else common_rows.intersection(df.index)
        )

    logger.print(f"\nCommon latency rows across models: {len(common_rows)}")

    if len(common_rows) == 0:
        return []

    for name in latencies:
        latencies[name] = latencies[name].loc[common_rows].sort_index()

    records = []

    for classical, deep in itertools.product(
        CLASSICAL_MODELS, DEEP_LEARNING_MODELS
    ):

        if classical not in latencies or deep not in latencies:
            continue
            
        col_c = "total_latency_ms" if "total_latency_ms" in latencies[classical].columns else latencies[classical].columns[0]
        col_d = "total_latency_ms" if "total_latency_ms" in latencies[deep].columns else latencies[deep].columns[0]

        lat_classical = latencies[classical][col_c].to_numpy()
        lat_deep = latencies[deep][col_d].to_numpy()

        result = wilcoxon_paired(lat_deep, lat_classical, rng)

        logger.print(
            f"\n  {deep} vs {classical} (Latency ms): "
            f"p = {result['p_value']:.6f}, "
            f"Hodges-Lehmann = {result['hodges_lehmann']:.4f} ms "
            f"[{result['hl_ci_low']:.4f}, {result['hl_ci_high']:.4f}]"
        )

        records.append({
            "test": "wilcoxon_paired", "hypothesis": "H3",
            "target": "total_latency",
            "deep_model": deep, "classical_model": classical,
            **result,
        })

    return records


# ============================================================
# H4: ROBUSTNESS UNDER VARYING CONDITIONS (TOST)
# ============================================================

def run_h4(logger):

    logger.print("\n" + "=" * 70)
    logger.print("H4: ROBUSTNESS (TOST equivalence)")
    logger.print("=" * 70)

    logger.print(
        f"\nEquivalence margin: {TOST_MARGIN * 100:.1f} accuracy points "
        f"(justify this value in the thesis text)"
    )

    files = sorted(OUTPUTS_DIR.glob("*/robustness_*.csv"))
    
    # 0-indexed trial columns as verified in the output files
    trial_cols = [f"accuracy_trial_{i}" for i in range(ROBUSTNESS_TRIALS)]

    records = []

    for path in files:

        model_name = path.stem.replace("robustness_", "")
        df = pd.read_csv(path)

        for target in df["target"].unique():

            target_df = df[df["target"] == target].sort_values("noise_level")

            if len(target_df) < 2:
                continue
                
            if not all(c in target_df.columns for c in trial_cols):
                logger.print(f"\nSkipping paired TOST for {model_name} / {target}: missing raw trial data.")
                continue

            reference = target_df.iloc[0]
            ref_trials = reference[trial_cols].to_numpy()

            logger.print(f"\n{model_name} / {target}:")

            for _, row in target_df.iloc[1:].iterrows():
                
                comp_trials = row[trial_cols].to_numpy()

                result = tost_paired(ref_trials, comp_trials, TOST_MARGIN)

                verdict = "equivalent" if result["equivalent"] else "not equivalent"

                logger.print(
                    f"  {reference['noise_level'] * 100:.0f}% vs "
                    f"{row['noise_level'] * 100:.0f}% noise: "
                    f"diff = {result['diff']:.4f}, "
                    f"p_TOST = {result['p_tost']:.6f} ({verdict})"
                )

                records.append({
                    "test": "tost_paired", "hypothesis": "H4",
                    "model": model_name, "target": target,
                    "reference_noise": reference["noise_level"],
                    "compared_noise": row["noise_level"],
                    **result,
                })

    return records


# ============================================================
# H5: FAULT LOCATION ACCURACY
# ============================================================
# "Location error" here is per-row classification error (0 = correct,
# 1 = incorrect) on G/C/B/A, grouped by the true fault location (Loc)
# and blocked across models. Friedman is the omnibus test across all
# models at each location, Wilcoxon signed-rank is the post-hoc
# pairwise comparison, both non-parametric since the error is bounded
# at zero and right-skewed.

def run_h5(predictions, common_rows, logger, rng):

    logger.print("\n" + "=" * 70)
    logger.print("H5: FAULT LOCATION ACCURACY")
    logger.print("=" * 70)

    model_names = [
        name for name in predictions
        if name in CLASSICAL_MODELS + DEEP_LEARNING_MODELS
    ]

    if len(model_names) < 2:

        logger.print("Need at least two models to compare. Skipping.")
        return []

    if not DATA_FILE.exists():

        logger.print(f"\nDataset not found: {DATA_FILE}. Skipping.")
        return []

    raw_df = pd.read_csv(DATA_FILE)

    if "row_id" not in raw_df.columns:
        raw_df = raw_df.reset_index().rename(columns={"index": "row_id"})

    raw_df = raw_df.set_index("row_id").loc[common_rows]

    locations = sorted(raw_df["Loc"].unique())

    logger.print(f"\nLocations found: {locations}")

    records = []

    for loc in locations:

        loc_rows = raw_df.index[raw_df["Loc"] == loc]

        if len(loc_rows) < 2:
            continue

        logger.print(f"\n--- Loc = {loc} ---")

        # Per-line error at this location, blocked by model
        for target in TARGET_COLUMNS:

            error = {
                name: (
                    predictions[name].loc[loc_rows, f"{target}_true"]
                    != predictions[name].loc[loc_rows, f"{target}_pred"]
                ).astype(int).to_numpy()
                for name in model_names
            }

            matrix = np.column_stack(
                [error[name] for name in model_names]
            )

            omnibus = friedman_test(matrix)

            logger.print(
                f"  {target}: Friedman chi2 = {omnibus['statistic']:.4f}, "
                f"df = {omnibus['df']}, p = {omnibus['p_value']:.6f}, "
                f"Kendall's W = {omnibus['kendalls_w']:.4f}"
            )

            records.append({
                "test": "friedman_omnibus", "hypothesis": "H5",
                "location": loc, "target": target, **omnibus,
            })

            pairs = list(itertools.combinations(model_names, 2))
            bonferroni_alpha = ALPHA / len(pairs) if pairs else ALPHA

            for model_a, model_b in pairs:

                result = wilcoxon_paired(
                    error[model_a], error[model_b], rng
                )

                significant = result["p_value"] < bonferroni_alpha

                logger.print(
                    f"    {model_a} vs {model_b}: "
                    f"p = {result['p_value']:.6f} "
                    f"({'sig' if significant else 'ns'} @ "
                    f"Bonferroni alpha={bonferroni_alpha:.5f}), "
                    f"Hodges-Lehmann = {result['hodges_lehmann']:.4f} "
                    f"[{result['hl_ci_low']:.4f}, {result['hl_ci_high']:.4f}]"
                )

                records.append({
                    "test": "wilcoxon_posthoc", "hypothesis": "H5",
                    "location": loc, "target": target,
                    "model_a": model_a, "model_b": model_b,
                    "bonferroni_alpha": bonferroni_alpha,
                    "significant": significant,
                    **result,
                })

        # Cumulative accuracy across all lines, at this location
        overall_error = {
            name: (
                predictions[name].loc[loc_rows, "overall_true"]
                != predictions[name].loc[loc_rows, "overall_pred"]
            ).astype(int).to_numpy()
            for name in model_names
        }

        matrix = np.column_stack(
            [overall_error[name] for name in model_names]
        )

        omnibus = friedman_test(matrix)

        logger.print(
            f"  overall: Friedman chi2 = {omnibus['statistic']:.4f}, "
            f"df = {omnibus['df']}, p = {omnibus['p_value']:.6f}, "
            f"Kendall's W = {omnibus['kendalls_w']:.4f}"
        )

        records.append({
            "test": "friedman_omnibus", "hypothesis": "H5",
            "location": loc, "target": "overall", **omnibus,
        })

        pairs = list(itertools.combinations(model_names, 2))
        bonferroni_alpha = ALPHA / len(pairs) if pairs else ALPHA

        for model_a, model_b in pairs:

            result = wilcoxon_paired(
                overall_error[model_a], overall_error[model_b], rng
            )

            significant = result["p_value"] < bonferroni_alpha

            logger.print(
                f"    {model_a} vs {model_b}: "
                f"p = {result['p_value']:.6f} "
                f"({'sig' if significant else 'ns'} @ "
                f"Bonferroni alpha={bonferroni_alpha:.5f}), "
                f"Hodges-Lehmann = {result['hodges_lehmann']:.4f} "
                f"[{result['hl_ci_low']:.4f}, {result['hl_ci_high']:.4f}]"
            )

            records.append({
                "test": "wilcoxon_posthoc", "hypothesis": "H5",
                "location": loc, "target": "overall",
                "model_a": model_a, "model_b": model_b,
                "bonferroni_alpha": bonferroni_alpha,
                "significant": significant,
                **result,
            })

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    logger = ReportLogger(REPORT_FILE)

    rng = np.random.default_rng(RANDOM_STATE)

    try:

        logger.print("=" * 70)
        logger.print("HYPOTHESIS TESTING: H1-H5")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")

        predictions, common_rows = load_predictions(logger)

        all_records = []

        if len(common_rows) > 0:

            all_records += run_h1(predictions, common_rows, logger)
            all_records += run_h2(predictions, logger)

        else:

            logger.print(
                "\nNo common rows across models. Skipping H1/H2/H5."
            )

        all_records += run_h3(logger, rng)
        all_records += run_h4(logger)

        if len(common_rows) > 0:
            all_records += run_h5(predictions, common_rows, logger, rng)

        results_df = pd.DataFrame(all_records)

        results_file = RESULTS_DIR / f"results_{timestamp}.csv"

        results_df.to_csv(results_file, index=False)

        logger.print("\n" + "=" * 70)
        logger.print("OUTPUT FILES")
        logger.print("=" * 70)
        logger.print(f"Report:  {REPORT_FILE}")
        logger.print(f"Results: {results_file}")

    finally:

        logger.close()


if __name__ == "__main__":
    main()