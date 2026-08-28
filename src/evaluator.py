"""
Statistical Hypothesis Evaluation & Logging Framework
Transmission Line Fault Detection Pipeline

Automatically exports evaluation results to:
evaluation/eval_{date}_{time}/h{N}_{date}_{time}.txt (.csv)
"""

import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q


# ==============================================================================
# DIRECTORY & FILE MANAGEMENT HELPER
# ==============================================================================

def prepare_eval_dir(base_dir: str = "evaluation") -> tuple[Path, str]:
    """Creates a timestamped subfolder inside the evaluation directory."""
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path(base_dir) / f"eval_{now_str}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir, now_str


# ==============================================================================
# H1: FAULT DETECTION ACCURACY
# ==============================================================================

def compute_mcnemar_exact_or_cc(b: int, c: int, n_total: int, alpha: float = 0.05):
    """Computes McNemar test statistic, p-value, odds ratio, and 95% CI."""
    b_c = b + c
    if b_c == 0:
        return {
            "method": "N/A (No discordant pairs)",
            "statistic": 0.0,
            "p_value": 1.0,
            "odds_ratio": 1.0,
            "acc_diff_pct": 0.0,
            "ci_lower_pct": 0.0,
            "ci_upper_pct": 0.0,
        }

    if b_c < 25:
        method = "Exact Binomial"
        res = stats.binomtest(b, n=b_c, p=0.5, alternative="two-sided")
        stat = float(b)
        p_val = res.pvalue
    else:
        method = "Continuity-Corrected Chi-Square"
        stat = float((abs(b - c) - 1.0) ** 2 / b_c)
        p_val = stats.chi2.sf(stat, df=1)

    odds_ratio = float(b / c) if c > 0 else np.inf
    diff = (b - c) / n_total
    se_diff = np.sqrt((b + c - (b - c) ** 2 / n_total) / (n_total ** 2))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    return {
        "method": method,
        "statistic": stat,
        "p_value": p_val,
        "odds_ratio": odds_ratio,
        "acc_diff_pct": diff * 100.0,
        "ci_lower_pct": (diff - z_crit * se_diff) * 100.0,
        "ci_upper_pct": (diff + z_crit * se_diff) * 100.0,
    }


def evaluate_h1(
    predictions_df: pd.DataFrame,
    model_names: list[str],
    baseline_col: str,
    output_dir: Path,
    timestamp: str
):
    """H1 evaluation with CSV and formatted TXT report logging."""
    correct_df = pd.DataFrame(index=predictions_df.index)
    all_methods = [baseline_col] + model_names

    for m in all_methods:
        correct_df[m] = (predictions_df[f"{m}_pred"] == predictions_df["overall_true"]).astype(int)

    q_res = cochrans_q(correct_df[all_methods])
    n_total = len(correct_df)

    results = []
    txt_lines = [
        "=" * 80,
        "HYPOTHESIS 1 (H1): FAULT DETECTION ACCURACY",
        "=" * 80,
        f"\nOmnibus Cochran's Q Test (Across {len(all_methods)} methods):",
        f"  Q-statistic        : {q_res.statistic:.4f}",
        f"  p-value            : {q_res.pvalue:.4e}",
        f"  Degrees of Freedom : {q_res.df}",
        "\nPairwise McNemar Tests vs. Baseline Relay ('" + baseline_col + "'):",
        "-" * 80
    ]

    for m in model_names:
        b = int(((correct_df[m] == 1) & (correct_df[baseline_col] == 0)).sum())
        c = int(((correct_df[m] == 0) & (correct_df[baseline_col] == 1)).sum())

        mcn = compute_mcnemar_exact_or_cc(b, c, n_total)
        mcn["model"] = m
        mcn["discordant_b"] = b
        mcn["discordant_c"] = c
        results.append(mcn)

        txt_lines.extend([
            f"Model: {m}",
            f"  Discordant Pairs (b={b}, c={c}) | Test Used: {mcn['method']}",
            f"  Stat: {mcn['statistic']:.4f} | p-val: {mcn['p_value']:.4e} | Odds Ratio: {mcn['odds_ratio']:.4f}",
            f"  Acc Diff: {mcn['acc_diff_pct']:.2f}% (95% CI: [{mcn['ci_lower_pct']:.2f}%, {mcn['ci_upper_pct']:.2f}%])\n"
        ])

    # Save formatted report (.txt)
    txt_path = output_dir / f"h1_{timestamp}.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    # Save structured table (.csv)
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / f"h1_{timestamp}.csv", index=False)

    print("\n".join(txt_lines))
    return res_df


# ==============================================================================
# H2: FAULT CLASSIFICATION PERFORMANCE
# ==============================================================================

def evaluate_h2(
    predictions_df: pd.DataFrame,
    dl_models: list[str],
    ml_models: list[str],
    output_dir: Path,
    timestamp: str,
    fold_f1_df: pd.DataFrame = None
):
    """H2 evaluation with CSV and formatted TXT report logging."""
    learned_models = dl_models + ml_models
    correct_df = pd.DataFrame(index=predictions_df.index)

    for m in learned_models:
        correct = np.ones(len(predictions_df), dtype=bool)
        for target in ["G", "C", "B", "A"]:
            correct &= (predictions_df[f"{m}_{target}_pred"] == predictions_df[f"{target}_true"])
        correct_df[m] = correct.astype(int)

    q_res = cochrans_q(correct_df[learned_models])
    n_total = len(correct_df)
    cross_family_results = []

    txt_lines = [
        "=" * 80,
        "HYPOTHESIS 2 (H2): FAULT CLASSIFICATION PERFORMANCE",
        "=" * 80,
        f"\nOmnibus Cochran's Q Test (Across {len(learned_models)} learned models):",
        f"  Q-statistic : {q_res.statistic:.4f}",
        f"  p-value     : {q_res.pvalue:.4e}",
        f"\nPairwise Cross-Family McNemar Comparisons ({len(dl_models)} DL x {len(ml_models)} ML = 12 pairs):",
        "-" * 80
    ]

    for dl in dl_models:
        for ml in ml_models:
            b = int(((correct_df[dl] == 1) & (correct_df[ml] == 0)).sum())
            c = int(((correct_df[dl] == 0) & (correct_df[ml] == 1)).sum())
            
            mcn = compute_mcnemar_exact_or_cc(b, c, n_total)
            pair_name = f"{dl} vs {ml}"
            mcn["pair"] = pair_name
            cross_family_results.append(mcn)

            txt_lines.append(f"{pair_name:<20} | Stat: {mcn['statistic']:8.4f} | p-val: {mcn['p_value']:.4e} | OR: {mcn['odds_ratio']:.3f}")

    if fold_f1_df is not None:
        txt_lines.extend(["\nSupplementary Analysis: Fold-level Macro F1 Score Comparison", "-" * 60])
        f1_data = [fold_f1_df[m].values for m in learned_models]
        friedman_stat, friedman_p = stats.friedmanchisquare(*f1_data)
        txt_lines.append(f"  Friedman Test Statistic: {friedman_stat:.4f} | p-value: {friedman_p:.4e}")

        if friedman_p < 0.05:
            txt_lines.append("\n  Post-hoc Wilcoxon Signed-Rank Tests (Holm-Bonferroni Corrected):")
            pw_tests = []
            for i in range(len(learned_models)):
                for j in range(i + 1, len(learned_models)):
                    m1, m2 = learned_models[i], learned_models[j]
                    res = stats.wilcoxon(fold_f1_df[m1], fold_f1_df[m2], zero_method="pratt")
                    pw_tests.append({"pair": f"{m1} vs {m2}", "p_raw": res.pvalue})
            
            pw_df = pd.DataFrame(pw_tests).sort_values("p_raw")
            m_comp = len(pw_df)
            pw_df["p_adj"] = [min(1.0, p * (m_comp - k)) for k, p in enumerate(pw_df["p_raw"])]
            
            for _, row in pw_df.iterrows():
                txt_lines.append(f"    {row['pair']:<20} | Raw p: {row['p_raw']:.4e} | Adj p: {row['p_adj']:.4e}")

    # Save formatted report (.txt)
    txt_path = output_dir / f"h2_{timestamp}.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    # Save structured table (.csv)
    res_df = pd.DataFrame(cross_family_results)
    res_df.to_csv(output_dir / f"h2_{timestamp}.csv", index=False)

    print("\n".join(txt_lines))
    return res_df


# ==============================================================================
# H3: FAULT REPORTING LATENCY
# ==============================================================================

def compute_hodges_lehmann_and_rank_biserial(latencies_a: np.ndarray, latencies_b: np.ndarray):
    """Computes Wilcoxon signed-rank, rank-biserial correlation, and Hodges-Lehmann CI."""
    diffs = latencies_a - latencies_b
    diffs = diffs[diffs != 0]
    n = len(diffs)

    if n == 0:
        return {"p_value": 1.0, "rank_biserial": 0.0, "hl_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    w_stat, p_value = stats.wilcoxon(latencies_a, latencies_b, zero_method="pratt", alternative="two-sided")
    ranks = stats.rankdata(np.abs(diffs))
    pos_mask = diffs > 0
    w_pos = np.sum(ranks[pos_mask])
    w_neg = np.sum(ranks[~pos_mask])
    rank_biserial = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0.0

    walsh_averages = np.sort([(diffs[i] + diffs[j]) / 2.0 for i in range(n) for j in range(i, n)])
    hl_estimate = float(np.median(walsh_averages))

    k = int(stats.wilcox.ppf(0.025, N=n)) if n <= 50 else int(n * (n + 1) / 4 - 1.96 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24))
    k = max(0, min(k, len(walsh_averages) - 1))

    return {
        "statistic": w_stat,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "hl_estimate": hl_estimate,
        "ci_lower": float(walsh_averages[k]),
        "ci_upper": float(walsh_averages[-k - 1]),
    }


def evaluate_h3(
    latency_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    output_dir: Path,
    timestamp: str
):
    """H3 evaluation with CSV and formatted TXT report logging."""
    lat_a = latency_df[model_a].values
    lat_b = latency_df[model_b].values

    res = compute_hodges_lehmann_and_rank_biserial(lat_a, lat_b)

    txt_lines = [
        "=" * 80,
        f"HYPOTHESIS 3 (H3): FAULT REPORTING LATENCY ({model_a} vs. {model_b})",
        "=" * 80,
        f"Wilcoxon Signed-Rank Test Statistic   : {res['statistic']:.4f}",
        f"Two-sided p-value                     : {res['p_value']:.4e}",
        f"Rank-Biserial Correlation (Effect)    : {res['rank_biserial']:.4f}",
        f"Hodges-Lehmann Median Difference      : {res['hl_estimate']:.6f} ms",
        f"95% Non-parametric Confidence Interval: [{res['ci_lower']:.6f} ms, {res['ci_upper']:.6f} ms]"
    ]

    # Save formatted report (.txt)
    txt_path = output_dir / f"h3_{timestamp}.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    # Save structured summary (.csv)
    res_df = pd.DataFrame([{**{"comparison": f"{model_a}_vs_{model_b}"}, **res}])
    res_df.to_csv(output_dir / f"h3_{timestamp}.csv", index=False)

    print("\n".join(txt_lines))
    return res_df


# ==============================================================================
# H4: ROBUSTNESS UNDER VARYING OPERATING CONDITIONS (TOST EQUIVALENCE)
# ==============================================================================

def evaluate_h4_tost(baseline_accs: np.ndarray, noisy_accs: np.ndarray, equivalence_margin: float = 0.02, alpha: float = 0.05):
    """Two One-Sided Tests (TOST) for statistical equivalence."""
    diffs = noisy_accs - baseline_accs
    mean_diff = np.mean(diffs)
    n = len(diffs)
    se = np.std(diffs, ddof=1) / np.sqrt(n)

    t_upper = (mean_diff - equivalence_margin) / se
    p_upper = stats.t.cdf(t_upper, df=n - 1)

    t_lower = (mean_diff - (-equivalence_margin)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)

    p_tost = max(p_upper, p_lower)
    return {
        "mean_diff": mean_diff,
        "se": se,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "p_tost": p_tost,
        "equivalent": p_tost < alpha,
    }


def evaluate_h4(
    robustness_df: pd.DataFrame,
    models: list[str],
    output_dir: Path,
    timestamp: str,
    margin: float = 0.02
):
    """H4 evaluation with CSV and formatted TXT report logging."""
    results = []
    noise_levels = robustness_df["noise_level"].unique()

    txt_lines = [
        "=" * 80,
        f"HYPOTHESIS 4 (H4): ROBUSTNESS UNDER VARYING CONDITIONS (TOST Margin: ±{margin * 100:.1f}%)",
        "=" * 80
    ]

    for m in models:
        txt_lines.extend([f"\nModel: {m}", "-" * 60])
        m_df = robustness_df[robustness_df["model"] == m]
        baseline_acc = m_df[m_df["noise_level"] == 0.0]["accuracy"].values

        for noise in noise_levels:
            if noise == 0.0:
                continue

            noisy_acc = m_df[m_df["noise_level"] == noise]["accuracy"].values
            if len(baseline_acc) == 0 or len(noisy_acc) == 0:
                continue

            tost_res = evaluate_h4_tost(baseline_acc, noisy_acc, equivalence_margin=margin)
            tost_res["model"] = m
            tost_res["noise_level"] = noise
            results.append(tost_res)

            status = "EQUIVALENT" if tost_res["equivalent"] else "NOT EQUIVALENT"
            txt_lines.append(
                f"  Noise Level: {noise * 100:4.1f}% | Mean Diff: {tost_res['mean_diff'] * 100:6.2f}% | "
                f"TOST p-val: {tost_res['p_tost']:.4e} | Conclusion: {status}"
            )

    # Save formatted report (.txt)
    txt_path = output_dir / f"h4_{timestamp}.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    # Save structured table (.csv)
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / f"h4_{timestamp}.csv", index=False)

    print("\n".join(txt_lines))
    return res_df


# ==============================================================================
# PIPELINE ENTRY POINT & RUNNER
# ==============================================================================

def run_hypothesis_evaluations(predictions_df, latency_df, robustness_df, fold_f1_df=None):
    """Executes all statistical hypothesis evaluations and logs output files."""
    eval_dir, timestamp = prepare_eval_dir(base_dir="evaluation")
    print(f"Evaluation session initialized. Artifacts directory: {eval_dir.resolve()}\n")

    # H1
    evaluate_h1(
        predictions_df=predictions_df,
        model_names=["CNN", "LSTM", "SVM", "KNN"],
        baseline_col="Relay",
        output_dir=eval_dir,
        timestamp=timestamp
    )

    # H2
    evaluate_h2(
        predictions_df=predictions_df,
        dl_models=["CNN", "LSTM"],
        ml_models=["SVM", "KNN"],
        fold_f1_df=fold_f1_df,
        output_dir=eval_dir,
        timestamp=timestamp
    )

    # H3
    evaluate_h3(
        latency_df=latency_df,
        model_a="CNN",
        model_b="Relay",
        output_dir=eval_dir,
        timestamp=timestamp
    )

    # H4
    evaluate_h4(
        robustness_df=robustness_df,
        models=["CNN", "LSTM"],
        margin=0.02,
        output_dir=eval_dir,
        timestamp=timestamp
    )
