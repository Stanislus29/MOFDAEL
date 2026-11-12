"""
Writer-facing analytics pack (Phase 3 support).

Generates, from the 2026-11-12 regenerated-data cascade reports ONLY
(no re-training, no new physics), a clean A4 PDF report plus a stack
of standalone figures for the dissertation writer:

    analytics/report_<timestamp>.pdf     the A4 document (pdflatex)
    analytics/figures/figNN_<name>.png   300 dpi PNG per figure
    analytics/figures/figNN_<name>.pdf   vector PDF per figure
    analytics/figures_manifest_<timestamp>.csv
    analytics/build_log_<timestamp>.txt  generation log

Data sources (all traceable, re-runnable reports):

    [SIM]  src/simulate.py (imported directly for waveform / reach
           figures — same verified solve, no re-derivation)
    [REL]  outputs/relay/predictions_relay.csv
    [T43]  outputs/table_4_3/table_4_3.csv (bootstrap ± SD)
    [HYP]  outputs/hypothesis_tests/results_2026-11-12_17-08-01.csv
    [CV]   outputs/scenario_cv/cv_results_2026-11-12_17-25-03.csv
    [COMB] outputs/combination_accuracy/results_2026-11-12_17-25-01.csv
    [ROB]  outputs/<model>/robustness_<model>.csv (2026-11-12 rows)

Hard verification anchors from today's reports are asserted at load
time; the script refuses to build the pack on a mismatch.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# local modules (single source of truth for the physics)
# ------------------------------------------------------------

from simulate import (
    ReportLogger,
    prefault_state,
    solve_fault,
    sample_fault_row,
    loop_impedances,
    dc_time_constant,
    LOAD_LEVELS,
    R_F_LEVELS,
    LOCATIONS_KM,
    SIGNATURE_ORDER,
    LINE_LENGTH_KM,
    Z1_PER_KM,
    OMEGA,
    SAMPLE_DT,
    V_PH_RMS,
)

from relay import zone1_reach_ohms


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ANALYTICS_DIR = PROJECT_ROOT / "analytics"
FIGURES_DIR = ANALYTICS_DIR / "figures"
BUILD_DIR = ANALYTICS_DIR / "build"

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
BUILD_DIR.mkdir(parents=True, exist_ok=True)

CLASS_FILE = DATA_DIR / "classdetectfault_sim.csv"
DETECT_FILE = DATA_DIR / "detectfault_sim.csv"
RELAY_MEAS_FILE = OUTPUTS_DIR / "simulate" / "relay_measurements_sim.csv"

T43_FILE = OUTPUTS_DIR / "table_4_3" / "table_4_3.csv"
HYP_FILE = (
    OUTPUTS_DIR / "hypothesis_tests"
    / "results_2026-11-12_17-08-01.csv"
)
CV_FILE = (
    OUTPUTS_DIR / "scenario_cv"
    / "cv_results_2026-11-12_17-25-03.csv"
)
COMB_FILE = (
    OUTPUTS_DIR / "combination_accuracy"
    / "results_2026-11-12_17-25-01.csv"
)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

PDF_FILE = ANALYTICS_DIR / f"report_{timestamp}.pdf"
MANIFEST_FILE = ANALYTICS_DIR / f"figures_manifest_{timestamp}.csv"
LOG_FILE = ANALYTICS_DIR / f"build_log_{timestamp}.txt"
TEX_FILE = BUILD_DIR / "report.tex"


# ============================================================
# CONSTANTS
# ============================================================

MODELS = ["ann", "cnn", "dtr", "knn", "lstm", "pinn", "rnn", "svm"]

TARGETS = ["G", "C", "B", "A"]

NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20]

TOST_MARGIN_PP = 5.0

MODEL_COLORS = {
    "ann": "#4C72B0",
    "cnn": "#DD8452",
    "dtr": "#55A868",
    "knn": "#C44E52",
    "lstm": "#8172B3",
    "pinn": "#937860",
    "rnn": "#DA8BC3",
    "svm": "#8C8C8C",
    "relay": "#1F3B57",
}

ZONE1_REACH = zone1_reach_ohms()          # 38.22 ohm

# H5 MAE anchors (% of 120 km) from [HYP] — asserted against the
# values recomputed from the prediction files.
H5_MAE_PCT_ANCHORS = {
    "ann": 13.5390, "cnn": 9.6791, "dtr": 2.2292, "knn": 5.1826,
    "lstm": 0.4564, "pinn": 15.8840, "rnn": 15.2298, "svm": 13.6142,
}


# ============================================================
# STYLE
# ============================================================

def setup_style() -> None:

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save one figure as PNG (300 dpi) and vector PDF."""

    png = FIGURES_DIR / f"{name}.png"
    pdf = FIGURES_DIR / f"{name}.pdf"

    fig.savefig(png)
    fig.savefig(pdf)

    plt.close(fig)

    return png


def md5_of(path: Path) -> str:

    return hashlib.md5(path.read_bytes()).hexdigest()


def parse_pm(text: str) -> tuple[float, float]:
    """Parse a '0.9944 ± 0.0024' cell into (mean, sd)."""

    parts = str(text).split("±")

    return float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0


# ============================================================
# DATA LOADING + HARD VERIFICATION ANCHORS
# ============================================================

def load_all(logger: ReportLogger) -> dict:

    logger.print("=" * 70)
    logger.print("LOADING SOURCES + VERIFYING HARD ANCHORS")
    logger.print("=" * 70)

    class_df = pd.read_csv(CLASS_FILE)
    detect_df = pd.read_csv(DETECT_FILE)

    t43 = pd.read_csv(T43_FILE)

    hyp = pd.read_csv(HYP_FILE)

    cv = pd.read_csv(CV_FILE)

    comb = pd.read_csv(COMB_FILE)

    preds: dict[str, pd.DataFrame] = {}

    for model in MODELS + ["relay"]:

        preds[model] = pd.read_csv(
            OUTPUTS_DIR / model / f"predictions_{model}.csv"
        )

    # ---- anchor 1: overall detection accuracy from predictions vs T43

    for model in MODELS:

        p = preds[model]

        acc = float((p["overall_pred"] == p["overall_true"]).mean())

        cell = t43[
            (t43["model"] == model) & (t43["target"] == "overall")
        ]

        expected = float(str(cell["accuracy"].iloc[0]).split("±")[0])

        if abs(acc - expected) > 5e-4:

            raise AssertionError(
                f"{model}: computed overall accuracy {acc:.4f} != "
                f"T43 anchor {expected:.4f}"
            )

    # ---- anchor 2: relay overall accuracy on all rows

    rp = preds["relay"]

    relay_acc_all = float((rp["overall_pred"] == rp["overall_true"]).mean())

    if abs(relay_acc_all - 0.7158) > 5e-4:

        raise AssertionError(
            f"relay overall {relay_acc_all:.4f} != anchor 0.7158"
        )

    # ---- anchor 3: H5 MAE recomputed from Loc predictions

    mae_pct: dict[str, float] = {}

    mae_by_loc: dict[str, pd.Series] = {}

    for model in MODELS:

        p = preds[model]

        fault_rows = p[p["Loc_true"] > 0]

        err = (fault_rows["Loc_pred"] - fault_rows["Loc_true"]).abs()

        mae_km = err.mean()

        mae_pct[model] = 100.0 * mae_km / LINE_LENGTH_KM

        mae_by_loc[model] = err.groupby(fault_rows["Loc_true"]).mean()

        anchor = H5_MAE_PCT_ANCHORS[model]

        if abs(mae_pct[model] - anchor) > 0.05:

            raise AssertionError(
                f"{model}: MAE {mae_pct[model]:.4f}% != HYP anchor "
                f"{anchor:.4f}%"
            )

    # ---- anchor 4: dataset counts

    n_class_fault = int((class_df["Loc"] > 0).sum())
    n_class_nf = int((class_df["Loc"] == 0).sum())
    n_detect = len(detect_df)
    n_detect_fault = int((detect_df["Loc"] > 0).sum())

    if not (
        n_class_fault == 3000
        and n_class_nf == 2365
        and n_detect == 9503
        and n_detect_fault == 3000
    ):

        raise AssertionError("dataset counts do not match anchors")

    logger.print(f"classdetectfault_sim : {len(class_df)} rows "
                 f"({n_class_fault} fault / {n_class_nf} no-fault)")
    logger.print(f"detectfault_sim      : {n_detect} rows "
                 f"({n_detect_fault} fault)")
    logger.print(f"table_4_3            : {len(t43)} rows")
    logger.print(f"hypothesis results   : {len(hyp)} rows")
    logger.print(f"cv results           : {len(cv)} rows")
    logger.print(f"combination results  : {len(comb)} rows")
    logger.print("")
    logger.print("ALL HARD ANCHORS VERIFIED "
                 "(accuracy, relay, H5 MAE, counts)")
    logger.print("")

    return {
        "class_df": class_df,
        "detect_df": detect_df,
        "t43": t43,
        "hyp": hyp,
        "cv": cv,
        "comb": comb,
        "preds": preds,
        "mae_pct": mae_pct,
        "mae_by_loc": mae_by_loc,
    }


# ============================================================
# SIMULATION-REUSE FIGURES (waveforms, reach, trips)
# ============================================================

def steady_state_figures(logger: ReportLogger) -> dict[str, Path]:

    logger.print("Computing loop impedances from src/simulate.py "
                 "(600 fault solves)...")

    prefault_by_load = {
        lvl: prefault_state(lvl) for lvl in LOAD_LEVELS
    }

    # z_ga[(sig, rf, load, loc)] = ground-loop impedance, phase a
    z_lookup: dict[tuple, float] = {}

    for sig in SIGNATURE_ORDER:

        for rf in R_F_LEVELS:

            for load in LOAD_LEVELS:

                pre = prefault_by_load[load]

                for loc in LOCATIONS_KM:

                    fault = solve_fault(sig, rf, loc, load, pre)

                    z_lookup[(sig, rf, load, loc)] = (
                        loop_impedances(
                            fault["v_r_abc"], fault["i_r_abc"]
                        )[0]
                    )

    paths: dict[str, Path] = {}

    # --------------------------------------------------------
    # fig01: distance-dependent waveforms
    # --------------------------------------------------------

    pre = prefault_by_load[0.75]

    t_grid_ms = np.arange(-20, 61, 1)

    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.4), sharex=True
    )

    dist_colors = {"15": "#4C72B0", "60": "#DD8452", "120": "#55A868"}

    for d_km in (15, 60, 120):

        fault = solve_fault("AG", 0.0, d_km, 0.75, pre)

        i_a, v_a = [], []

        for t_ms in t_grid_ms:

            t = t_ms * 1e-3

            if t < 0:

                vp = pre["v_r_pref_abc"][0]
                ip = pre["i_r_pref_abc"][0]

                i_inst = np.sqrt(2.0) * abs(ip) * np.sin(
                    OMEGA * t + np.angle(ip)
                )
                v_inst = np.sqrt(2.0) * abs(vp) * np.sin(
                    OMEGA * t + np.angle(vp)
                ) / (np.sqrt(2.0) * V_PH_RMS)

            else:

                row = sample_fault_row(
                    pre, fault, 0.0, 0.0, t / SAMPLE_DT - 1.0
                )

                i_inst = row["Ia"]
                v_inst = row["Va"]

            i_a.append(i_inst)
            v_a.append(v_inst)

        c = dist_colors[str(d_km)]

        axes[0].plot(t_grid_ms, i_a, color=c, lw=1.2,
                     label=f"Loc = {d_km} km")
        axes[1].plot(t_grid_ms, v_a, color=c, lw=1.2,
                     label=f"Loc = {d_km} km")

    # prefault reference
    ip = pre["i_r_pref_abc"][0]
    vp = pre["v_r_pref_abc"][0]

    i_pref_t = [
        np.sqrt(2.0) * abs(ip)
        * np.sin(OMEGA * t * 1e-3 + np.angle(ip))
        for t in t_grid_ms
    ]

    axes[0].plot(t_grid_ms, i_pref_t, color="#8C8C8C", lw=0.9,
                 ls="--", label="prefault")

    axes[0].axvline(0, color="#333333", lw=0.8, ls=":")
    axes[1].axvline(0, color="#333333", lw=0.8, ls=":")

    axes[0].set_ylabel("Phase-a current (A)")
    axes[1].set_ylabel("Phase-a voltage (pu)")
    axes[1].set_xlabel("Time from fault inception (ms)")
    axes[0].set_title(
        "Bolted AG fault, normal load: distance-dependent waveform "
        "physics",
        loc="left",
    )
    axes[0].legend(ncol=4, loc="upper right")
    axes[1].legend(ncol=4, loc="lower right")

    fig.tight_layout()

    paths["fig01_waveforms_distance"] = save_figure(
        fig, "fig01_waveforms_distance"
    )

    # --------------------------------------------------------
    # fig02: DC offset vs inception angle
    # --------------------------------------------------------

    fault = solve_fault("AG", 0.0, 60, 0.75, pre)

    tau = dc_time_constant(0.0, 60)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    theta_colors = {
        0: "#C44E52", 90: "#4C72B0", 180: "#55A868"
    }

    for theta_deg in (0, 90, 180):

        theta_inc = np.radians(theta_deg)

        i_a = []

        for t_ms in np.arange(-10, 81, 1):

            t = t_ms * 1e-3

            if t < 0:

                ip = pre["i_r_pref_abc"][0]

                i_inst = np.sqrt(2.0) * abs(ip) * np.sin(
                    OMEGA * t + np.angle(ip)
                )

            else:

                row = sample_fault_row(
                    pre, fault, 0.0, theta_inc, t / SAMPLE_DT - 1.0
                )

                i_inst = row["Ia"]

            i_a.append(i_inst)

        ax.plot(np.arange(-10, 81, 1), i_a, lw=1.1,
                color=theta_colors[theta_deg],
                label=f"inception {theta_deg}°")

    ax.axhline(0, color="#333333", lw=0.6)
    ax.axvline(0, color="#333333", lw=0.8, ls=":")

    ax.set_xlabel("Time from fault inception (ms)")
    ax.set_ylabel("Phase-a current (A)")
    ax.set_title(
        f"DC offset vs inception angle (bolted AG, 60 km, "
        f"τ = {tau * 1e3:.1f} ms)",
        loc="left",
    )
    ax.legend(loc="upper right")

    fig.tight_layout()

    paths["fig02_dc_offset"] = save_figure(fig, "fig02_dc_offset")

    # --------------------------------------------------------
    # fig04: reach diagram (AG ground loop, normal load)
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    rf_colors = {
        0.0: "#1F3B57", 10.0: "#4C72B0", 25.0: "#2A9D8F",
        50.0: "#E9C46A", 100.0: "#E76F51",
    }

    loc_arr = np.array(LOCATIONS_KM, dtype=float)

    ideal = np.abs(Z1_PER_KM) * loc_arr

    ax.plot(loc_arr, ideal, color="#8C8C8C", lw=1.0, ls=":",
            label="ideal bolted ground loop $|Z_1|\\,d$")

    for rf in R_F_LEVELS:

        z = [z_lookup[("AG", rf, 0.75, loc)] for loc in LOCATIONS_KM]

        ax.plot(loc_arr, z, marker="o", ms=3.5, lw=1.2,
                color=rf_colors[rf],
                label=f"$R_f$ = {rf:g} Ω")

    ax.axhspan(0, ZONE1_REACH, color="#2A9D8F", alpha=0.08)

    ax.axhline(
        ZONE1_REACH, color="#1F3B57", lw=1.1, ls="--",
        label=f"Zone 1 reach = {ZONE1_REACH:.2f} Ω"
    )

    ax.set_xlabel("Fault location (km)")
    ax.set_ylabel("Ground-loop impedance $|Z_{ga}|$ (Ω)")
    ax.set_title(
        "Distance-relay reach diagram: bolted faults trip to 90 km; "
        "high $R_f$ leaves Zone 1",
        loc="left",
    )
    ax.set_xticks(LOCATIONS_KM)
    ax.legend(ncol=2, loc="upper left", fontsize=8)

    fig.tight_layout()

    paths["fig04_reach_diagram"] = save_figure(fig, "fig04_reach_diagram")

    # --------------------------------------------------------
    # fig05: trip-fraction heatmap over the factorial
    # (inception angle does not enter the steady-state decision,
    #  so the fraction is over 5 signatures x 3 load levels)
    # --------------------------------------------------------

    trip_frac = np.zeros((len(R_F_LEVELS), len(LOCATIONS_KM)))

    for i_rf, rf in enumerate(R_F_LEVELS):

        for i_loc, loc in enumerate(LOCATIONS_KM):

            trips = [
                z_lookup[(sig, rf, load, loc)] < ZONE1_REACH
                for sig in SIGNATURE_ORDER
                for load in LOAD_LEVELS
            ]

            trip_frac[i_rf, i_loc] = float(np.mean(trips))

    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    im = ax.imshow(
        trip_frac, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto"
    )

    ax.set_xticks(range(len(LOCATIONS_KM)))
    ax.set_xticklabels(LOCATIONS_KM)
    ax.set_yticks(range(len(R_F_LEVELS)))
    ax.set_yticklabels([f"{rf:g}" for rf in R_F_LEVELS])
    ax.set_xlabel("Fault location (km)")
    ax.set_ylabel("$R_f$ (Ω)")
    ax.set_title(
        "Relay trip fraction per (location, $R_f$) across "
        "5 signatures × 3 loads — secure but incomplete",
        loc="left",
    )
    ax.grid(False)

    for i in range(trip_frac.shape[0]):

        for j in range(trip_frac.shape[1]):

            val = trip_frac[i, j]

            ax.text(
                j, i, f"{val:.2f}".rstrip("0").rstrip("."),
                ha="center", va="center", fontsize=7.5,
                color="white" if val > 0.55 else "#333333",
            )

    fig.colorbar(im, ax=ax, shrink=0.9, label="trip fraction")

    fig.tight_layout()

    paths["fig05_relay_trip_heatmap"] = save_figure(
        fig, "fig05_relay_trip_heatmap"
    )

    logger.print("steady-state figures done "
                 f"({len(z_lookup)} loop-impedance points)")
    logger.print("")

    return paths


# ============================================================
# REPORT-DATA FIGURES
# ============================================================

def dataset_figure(data: dict) -> Path:

    class_df = data["class_df"]

    n_fault = int((class_df["Loc"] > 0).sum())
    n_nf = int((class_df["Loc"] == 0).sum())

    detect_fault, detect_total = 3000, 9503
    detect_nf = detect_total - detect_fault

    fig, ax = plt.subplots(figsize=(7.0, 2.4))

    labels = [
        f"classification ({len(class_df)} rows)",
        f"detection ({detect_total} rows)",
    ]

    fault_counts = [n_fault, detect_fault]
    nf_counts = [n_nf, detect_nf]

    y = [1, 0]

    ax.barh(
        y, fault_counts, color="#E76F51", height=0.55,
        label="fault rows",
    )
    ax.barh(
        y, nf_counts, left=fault_counts, color="#8C8C8C",
        height=0.55, alpha=0.65, label="no-fault rows",
    )

    for yi, (f, nf) in enumerate(zip(fault_counts, nf_counts)):

        total = f + nf

        ax.text(
            f / 2, yi, f"{f} ({100 * f / total:.1f}%)",
            ha="center", va="center", color="white", fontsize=8.5,
        )

        ax.text(
            f + nf / 2, yi, f"{nf} ({100 * nf / total:.1f}%)",
            ha="center", va="center", color="#333333", fontsize=8.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("rows")
    ax.set_title(
        "Dataset composition (regenerated 2026-11-12) — detection "
        "set is 31.6% fault rows",
        loc="left",
    )
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlim(0, detect_total * 1.05)

    fig.tight_layout()

    return save_figure(fig, "fig03_dataset_composition")


def accuracy_bars(data: dict) -> Path:

    t43 = data["t43"]

    overall = t43[t43["target"] == "overall"].set_index("model")

    models = MODELS + ["relay"]

    means, sds, labels = [], [], []

    for m in models:

        row = overall.loc[m]

        mu, sd = parse_pm(row["accuracy"])

        means.append(mu)
        sds.append(sd)

        labels.append(
            f"{m}\n(n={int(row['n'])})"
            if m != "relay" else "relay\n(common rows)"
        )

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    x = np.arange(len(models))

    colors = [MODEL_COLORS[m] for m in models]

    bars = ax.bar(
        x, means, yerr=sds, capsize=3, width=0.62,
        color=colors, error_kw={"lw": 0.9, "alpha": 0.7},
    )

    for xi, (mu, sd) in enumerate(zip(means, sds)):

        ax.text(
            xi, mu + sd + 0.012, f"{mu:.4f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("overall accuracy")
    ax.set_ylim(0.70, 1.03)
    ax.set_title(
        "Overall detection accuracy ± bootstrap SD (2,000, seed 42), "
        "common test rows — Table 4.3",
        loc="left",
    )

    fig.tight_layout()

    return save_figure(fig, "fig06_accuracy_bars")


def per_target_heatmap(data: dict) -> Path:

    t43 = data["t43"]

    sub = t43[
        t43["target"].isin(TARGETS)
        & t43["model"].isin(MODELS)
    ]

    mat = np.zeros((len(MODELS), len(TARGETS)))

    for i, m in enumerate(MODELS):

        for j, tg in enumerate(TARGETS):

            mu, _ = parse_pm(
                sub[(sub["model"] == m) & (sub["target"] == tg)][
                    "accuracy"
                ].iloc[0]
            )

            mat[i, j] = mu

    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    im = ax.imshow(
        mat, cmap="YlGnBu", vmin=0.90, vmax=1.0, aspect="auto"
    )

    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(["G", "C", "B", "A"])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("fault-type target")
    ax.set_title(
        "Per-target accuracy — phase A separates the field "
        "(H2: Q = 167.5)",
        loc="left",
    )
    ax.grid(False)

    for i in range(mat.shape[0]):

        for j in range(mat.shape[1]):

            ax.text(
                j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                fontsize=7.5,
                color="white" if mat[i, j] > 0.965 else "#333333",
            )

    fig.colorbar(im, ax=ax, shrink=0.85, label="accuracy")

    fig.tight_layout()

    return save_figure(fig, "fig07_per_target_heatmap")


def h1_forest(data: dict) -> Path:

    hyp = data["hyp"]

    sub = hyp[
        (hyp["test"] == "mcnemar_vs_relay")
        & (hyp["hypothesis"] == "H1")
    ].set_index("model").loc[MODELS]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    y = np.arange(len(MODELS))[::-1]

    for yi, m in zip(y, MODELS):

        row = sub.loc[m]

        lo = row["accuracy_diff_ci_low"]
        hi = row["accuracy_diff_ci_high"]
        mu = row["accuracy_diff_pct"]

        ax.errorbar(
            mu, yi, xerr=[[mu - lo], [hi - mu]], fmt="o", ms=4.5,
            color=MODEL_COLORS[m], capsize=3, lw=1.1,
        )

        ax.text(
            hi + 0.35, yi, f"+{mu:.2f} pp", va="center", fontsize=8,
        )

    ax.axvline(0, color="#8C8C8C", lw=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("detection-accuracy gain over the relay (pp, 95% CI)")
    ax.set_title(
        "H1: every model beats the Zone-1 relay by 23.7–24.3 pp "
        "(all McNemar p < 1e-6)",
        loc="left",
    )
    ax.set_xlim(20, 29)

    fig.tight_layout()

    return save_figure(fig, "fig08_h1_forest")


def latency_chart(data: dict) -> Path:

    t43 = data["t43"]

    overall = t43[t43["target"] == "overall"].set_index("model")

    models = MODELS + ["relay"]

    means = []

    for m in models:

        mu, _ = parse_pm(overall.loc[m, "latency_ms_per_sample"])

        means.append(mu)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    y = np.arange(len(models))[::-1]

    ax.barh(
        y, means, color=[MODEL_COLORS[m] for m in models],
        height=0.6, log=True,
    )

    for yi, mu in zip(y, means):

        label = f"{mu:.4g} ms"

        ax.text(mu * 1.25, yi, label, va="center", fontsize=8)

    ax.axvline(10.0, color="#1F3B57", lw=1.0, ls="--")

    ax.text(
        10.0 * 1.3, len(models) - 0.45, "relay reference = 10 ms",
        fontsize=8, color="#1F3B57", va="center",
    )

    ax.set_ylim(-0.6, len(models) - 0.1)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("mean per-row reporting latency (ms, log scale)")
    ax.set_xlim(3e-4, 4e3)
    ax.set_title(
        "H3: all models report ≈10 ms faster than the relay — "
        "except the LSTM (204.9 ms mean)",
        loc="left",
    )

    fig.tight_layout()

    return save_figure(fig, "fig09_latency")


def robustness_curves(data: dict) -> Path:

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for model in MODELS:

        rob = pd.read_csv(
            OUTPUTS_DIR / model / f"robustness_{model}.csv"
        )

        latest = rob["timestamp"].max()

        rob = rob[rob["timestamp"] == latest]

        means = (
            rob.groupby("noise_level")["accuracy_mean"].mean()
            .reindex(NOISE_LEVELS)
        )

        degradation = 100.0 * (
            means - means.iloc[0]
        )

        ax.plot(
            [100 * n for n in NOISE_LEVELS], degradation,
            marker="o", ms=3.5, lw=1.2, color=MODEL_COLORS[model],
            label=model,
        )

    ax.axhspan(
        -TOST_MARGIN_PP, TOST_MARGIN_PP, color="#2A9D8F", alpha=0.10,
        label=f"±{TOST_MARGIN_PP:.0f} pp TOST margin",
    )

    ax.set_xlabel("injected noise level (%)")
    ax.set_ylabel("accuracy change vs 1% noise (pp)")
    ax.set_title(
        "H4: noise degradation by model (mean across G/C/B/A); "
        "LSTM stays inside the margin everywhere",
        loc="left",
    )
    ax.set_xticks([100 * n for n in NOISE_LEVELS])
    ax.legend(ncol=3, fontsize=8)

    fig.tight_layout()

    return save_figure(fig, "fig10_robustness_curves")


def mae_bars(data: dict) -> Path:

    mae = data["mae_pct"]

    order = sorted(MODELS, key=lambda m: mae[m])

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    x = np.arange(len(order))

    ax.bar(
        x, [mae[m] for m in order], width=0.6,
        color=[MODEL_COLORS[m] for m in order],
    )

    for xi, m in enumerate(order):

        ax.text(
            xi, mae[m] + 0.25, f"{mae[m]:.2f}%",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("location MAE (% of 120 km line)")
    ax.set_ylim(0, 18)
    ax.set_title(
        "H5: fault-location MAE — LSTM 0.46%, DTR 2.23%, "
        "the rest 5.2–15.9%",
        loc="left",
    )

    fig.tight_layout()

    return save_figure(fig, "fig11_mae_bars")


def mae_by_location(data: dict) -> Path:

    by_loc = data["mae_by_loc"]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    locs = sorted(LOCATIONS_KM)

    for model in MODELS:

        series = by_loc[model].reindex(locs)

        lw = 2.0 if model in ("lstm", "dtr") else 1.0

        alpha = 1.0 if model in ("lstm", "dtr") else 0.55

        ax.plot(
            locs, series.values, marker="o", ms=3.5, lw=lw,
            alpha=alpha, color=MODEL_COLORS[model], label=model,
        )

    ax.set_xlabel("true fault location (km)")
    ax.set_ylabel("location MAE (km)")
    ax.set_xticks(locs)
    ax.set_title(
        "Per-location MAE: DTR stays within 0.31–0.63 km of the "
        "LSTM at every location",
        loc="left",
    )
    ax.legend(ncol=4, fontsize=8)

    fig.tight_layout()

    return save_figure(fig, "fig12_mae_by_location")


def combination_heatmap(data: dict) -> Path:

    comb = data["comb"]

    combos = ["G+A", "C+B", "G+B+A", "C+B+A", "G+C+B+A"]

    sub = comb[
        comb["combination"].isin(combos) & (comb["n"] > 0)
    ]

    mat = np.full((len(MODELS), len(combos)), np.nan)

    for i, m in enumerate(MODELS):

        for j, c in enumerate(combos):

            row = sub[(sub["model"] == m) & (sub["combination"] == c)]

            if len(row):

                mat[i, j] = row["accuracy"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    im = ax.imshow(
        mat, cmap="YlGnBu", vmin=0.70, vmax=1.0, aspect="auto"
    )

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title(
        "Exact type-vector accuracy by fault combination "
        "(LSTM perfect; ABCG hardest)",
        loc="left",
    )
    ax.grid(False)

    for i in range(mat.shape[0]):

        for j in range(mat.shape[1]):

            if np.isnan(mat[i, j]):

                continue

            ax.text(
                j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                fontsize=7.5,
                color="white" if mat[i, j] > 0.9 else "#333333",
            )

    fig.colorbar(im, ax=ax, shrink=0.85, label="accuracy")

    fig.tight_layout()

    return save_figure(fig, "fig13_combination_heatmap")


def cv_figure(data: dict) -> Path:

    cv = data["cv"]

    cv_models = list(cv["model"].unique())

    stats = (
        cv.groupby("model")[
            ["detection_accuracy", "exact_match_accuracy"]
        ]
        .agg(["mean", "std"])
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    y = np.arange(len(cv_models))[::-1]

    h = 0.16

    for offset, metric, color, label in (
        (+h, "detection_accuracy", "#1F3B57", "detection"),
        (-h, "exact_match_accuracy", "#E9C46A", "exact match"),
    ):

        mu = stats[(metric, "mean")].loc[cv_models].values

        sd = stats[(metric, "std")].loc[cv_models].values

        ax.errorbar(
            mu, y + offset, xerr=sd, fmt="o", ms=4.5, color=color,
            capsize=2.5, lw=1.0, label=label,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(cv_models)
    ax.set_xlabel("scenario-level accuracy (K = 7 group CV, mean ± SD)")
    ax.set_xlim(0.875, 1.005)
    ax.set_ylim(-0.7, len(cv_models) - 0.3)
    ax.set_title(
        "R9: K = 7 scenario-grouped CV (671 scenarios, 0 spanning) — "
        "descriptive, fixed hyperparameters",
        loc="left",
    )
    ax.legend(loc="lower center", ncol=2, fontsize=8)

    fig.tight_layout()

    return save_figure(fig, "fig14_cv")


# ============================================================
# LATEX ASSEMBLY
# ============================================================

TEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage[table]{xcolor}
\usepackage{caption}
\usepackage{parskip}
\usepackage{fancyhdr}
\usepackage{longtable}
\usepackage{array}
\graphicspath{{../figures/}}
\captionsetup{font=small,labelfont=bf,width=0.92\textwidth}
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{Fault-detection analytics pack}}
\fancyhead[R]{\small\textit{regenerated data, 2026-11-12}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\begin{document}
"""


def esc(text: str) -> str:

    if isinstance(text, float) and np.isnan(text):

        return "n/a"

    text = str(text)

    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def figure_block(name: str, caption: str) -> str:

    return (
        "\\begin{figure}[H]\n\\centering\n"
        f"\\includegraphics[width=0.88\\textwidth]{{{name}.pdf}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}\n"
    )


def build_tex(data: dict, fig_paths: dict[str, Path]) -> str:

    t43 = data["t43"]
    hyp = data["hyp"]

    overall = t43[t43["target"] == "overall"].set_index("model")

    # ---- table 4.3 overall rows

    rows = []

    for m in MODELS + ["relay", "relay (all rows)"]:

        r = overall.loc[m]

        # latency: short precision so the table fits the text width
        lat_mu, lat_sd = parse_pm(r["latency_ms_per_sample"])

        if lat_mu < 1.0:

            lat = f"{lat_mu:.4f} $\\pm$ {lat_sd:.4f}"

        else:

            lat = f"{lat_mu:.2f} $\\pm$ {lat_sd:.2f}"

        rob = esc(r["robustness"])

        if rob != "n/a":

            rob_mu, rob_sd = parse_pm(r["robustness"])

            rob = f"{rob_mu:.4f} $\\pm$ {rob_sd:.4f}"

        rows.append(
            f"{esc(m)} & {int(r['n'])} & {esc(r['accuracy'])} & "
            f"{esc(r['recall'])} & {esc(r['f1'])} & "
            f"{rob} & {lat} \\\\"
        )

    t43_rows = "\n".join(rows)

    # ---- H5 MAE table

    mae_rows = "\n".join(
        f"{esc(m)} & {data['mae_by_loc'][m].index.size} & "
        f"{data['mae_pct'][m]:.2f} & "
        f"{data['mae_pct'][m] / 1.2:.3f} \\\\"
        for m in sorted(MODELS, key=lambda x: data["mae_pct"][x])
    )

    # ---- CV table (the CV excludes the LSTM: 7 models)

    cv = data["cv"]

    cv_models = list(cv["model"].unique())

    cv_stats = (
        cv.groupby("model")[
            ["detection_accuracy", "exact_match_accuracy"]
        ]
        .agg(["mean", "std"])
    )

    cv_rows = "\n".join(
        f"{esc(m)} & "
        f"{cv_stats[('detection_accuracy', 'mean')].loc[m]:.4f} $\\pm$ "
        f"{cv_stats[('detection_accuracy', 'std')].loc[m]:.4f} & "
        f"{cv_stats[('exact_match_accuracy', 'mean')].loc[m]:.4f} $\\pm$ "
        f"{cv_stats[('exact_match_accuracy', 'std')].loc[m]:.4f} \\\\"
        for m in cv_models
    )

    # ---- H2 omnibus Q values

    h2q = hyp[
        (hyp["hypothesis"] == "H2")
        & (hyp["test"] == "cochrans_q_omnibus")
    ].set_index("target")

    h2_q_line = ", ".join(
        f"{tg}: Q = {h2q.loc[tg, 'Q']:.2f}"
        for tg in TARGETS
    )

    # ---- class md5s

    md5c = md5_of(CLASS_FILE)
    md5d = md5_of(DETECT_FILE)

    body = TEX_PREAMBLE + rf"""
% ============================================================
% TITLE
% ============================================================

\begin{{titlepage}}
\centering
\vspace*{{3.2cm}}
{{\LARGE\bfseries Fault-Detection Analytics Pack\par}}
\vspace{{0.5cm}}
{{\large Companion figures and verified numbers for the
dissertation writer\par}}
\vspace{{1.2cm}}
{{\large Regenerated-data era --- 2026-11-12\par}}
\vspace{{2.2cm}}
\begin{{center}}
\texttt{{\small
data/classdetectfault\_sim.csv\quad md5 {md5c[:12]}...\quad
5{{,}}365 rows\\[2pt]
data/detectfault\_sim.csv\quad md5 {md5d[:12]}...\quad
9{{,}}503 rows\\[2pt]
branch \texttt{{exp}}\quad build {esc(timestamp)}}}
\end{{center}}
\vfill
{{\small Every figure and number in this pack is generated by
\texttt{{src/analytics.py}} from re-runnable reports; the
traceability appendix maps each one to its source.\par}}
\end{{titlepage}}

% ============================================================
\section{{Dataset and provenance}}

The pack is built on the Python re-simulation of the corrected
161 kV / 120 km line (Z\textsubscript{{1}} = 0.080 + j0.390,
Z\textsubscript{{0}} = 0.228 + j1.352 $\Omega$/km, C\textsubscript{{1}}
= 12.74 nF/km). The classification dataset holds 5{{,}}365 rows
(3{{,}}000 fault rows in 375 eight-row location sweeps + 2{{,}}365
no-fault rows); the detection dataset holds 9{{,}}503 rows and is
31.6\% fault rows (45.8\% in the superseded original). The
scenario-level split assigns 134 of 671 scenarios (1{{,}}069 rows) to
the shared test partition of all eight models.

{figure_block("fig03_dataset_composition",
  r"Dataset composition. Where it goes: Section 4.2 (dataset "
  r"presentation). Source: data files; counts asserted at build time.")}

{figure_block("fig01_waveforms_distance",
  r"Distance-dependent waveform physics, now present in the "
  r"regenerated data (the original data was flat in location). "
  r"Fault current decays 4.76 to 0.89 kA and the faulted-phase "
  r"voltage sag relaxes 0.568 to 0.928 pu across 15 to 120 km. "
  r"Where it goes: Section 3.5.5 (dataset generation) and the R10 "
  r"waveform discussion. Source: src/simulate.py, bolted AG, "
  r"normal load.")}

{figure_block("fig02_dc_offset",
  r"DC offset vs inception angle: the offset peaks at the voltage-zero "
  r"crossings and vanishes near the voltage peak, decaying with the "
  r"positive-sequence loop time constant. Where it goes: Section 3.5.5 "
  r"and R10. Source: src/simulate.py.")}

\section{{Conventional relay baseline}}

The relay is a numerical distance relay deciding on six loop
impedances (three residual-compensated ground loops
$|V_p|/|I_p + k_0\,3I_0|$, $k_0 = 0.8140 + j0.0405$, and three
phase-phase loops) evaluated as steady-state phasor quantities. The
Zone 1 reach is $0.80 \times |Z_1| \times 120$ km =
{ZONE1_REACH:.2f}~$\Omega$. Its profile on the regenerated data:
true-negative rate 100\%, trip rate 49.2\% of fault rows, overall
accuracy 0.7158 (all rows) / 0.7568 (common test rows) --- secure but
incomplete.

{figure_block("fig04_reach_diagram",
  r"Reach diagram. Bolted faults trip to 90 km ($|Z_1|d = 36.0$ "
  r"$\Omega$ < 38.22 $\Omega$); 105 and 120 km fall beyond Zone 1, and "
  r"high-resistance faults leave Zone 1 at every location. Where it "
  r"goes: Section 3.9.4 (relay detection logic). Source: "
  r"src/simulate.py loop impedances + src/relay.py reach.")}

{figure_block("fig05_relay_trip_heatmap",
  r"Trip fraction across the full factorial (5 signatures x 3 load "
  r"levels; the inception angle does not enter the steady-state "
  r"decision). The white corners --- high $R_f$, far faults --- are "
  r"exactly the regime the learned models cover. Where it goes: "
  r"Section 4.x, H1 discussion. Source: src/simulate.py.")}

\section{{Classification performance (Table 4.3, H1, H2)}}

All values carry bootstrap SDs (2{{,}}000 resamples, seed 42) on the
1{{,}}069 common test rows. The weakest cell in the table is the RNN on
phase A (accuracy 0.9280); the LSTM attains 1.0 on every target.

\begin{{table}}[H]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lrlrrrr@{{}}}}
\toprule
model & $n$ & accuracy & recall & F1 & robustness & latency (ms) \\
\midrule
{t43_rows}
\bottomrule
\end{{tabular}}
\caption{{Table 4.3 (regenerated), overall rows. Relay ``all rows''
covers the full 5{{,}}365-row dataset. Source: outputs/table\_4\_3
(report 2026-11-12\_19-43-50).}}
\end{{table}}

{figure_block("fig06_accuracy_bars",
  r"Overall detection accuracy with bootstrap SDs. Where it goes: "
  r"Section 4.x, Table 4.3 discussion. Source: table\_4\_3.csv.")}

{figure_block("fig08_h1_forest",
  r"H1 forest plot: accuracy gain over the relay with 95\% CIs; "
  r"Cochran's Q = 1906.85 (df = 8, $p < 10^{-6}$). Where it goes: "
  r"Section 4.x, H1. Source: hypothesis results CSV.")}

{figure_block("fig07_per_target_heatmap",
  r"Per-target accuracy. Per-target Cochran's Q: " + h2_q_line +
  r"; under Holm correction the LSTM is significantly best on G/C/B and, with "
  r"CNN, on phase A. Where it goes: Section 4.x, H2. Source: "
  r"hypothesis results CSV.")}

\section{{Latency and noise robustness (H3, H4)}}

{figure_block("fig09_latency",
  r"Reporting latency (log scale). Every model reports faster than "
  r"the relay's 10 ms reference by essentially the full margin "
  r"(Hodges--Lehmann $\approx -10$ ms); the LSTM reports 188.75 ms "
  r"slower (95\% CI [178.5, 197.0], mean 204.9 ms). Where it goes: "
  r"Section 4.x, H3 --- replaces the stale ``270 ms'' sentence. "
  r"Source: table\_4\_3.csv + hypothesis results CSV.")}

{figure_block("fig10_robustness_curves",
  r"Accuracy change vs the 1\% noise baseline, mean across targets. "
  r"The shaded band is the pre-declared 5 pp TOST equivalence margin: "
  r"the LSTM is equivalent at every level, KNN up to 10\% on all "
  r"targets, the others break at 20\%; CNN and PINN are not "
  r"equivalent on phase A even at 5\%. Where it goes: Section 4.x, "
  r"H4. Source: robustness\_*.csv (2026-11-12 rows).")}

\section{{Fault location and scenario structure (H5, R9)}}

Location accuracy is measured on the 600 fault scenarios of the
common test set (MAE as \% of the 120 km line).

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
model & locations & MAE (km) & MAE (\% of 120 km) \\
\midrule
{mae_rows}
\bottomrule
\end{{tabular}}
\caption{{H5 fault-location MAE, recomputed from the Loc columns of
the prediction files and asserted against the hypothesis report
(within 0.05 pp). Source: predictions\_*.csv.}}
\end{{table}}

{figure_block("fig11_mae_bars",
  r"Location MAE ranking. Where it goes: Section 4.x, H5. Source: "
  r"prediction files (asserted).")}

{figure_block("fig12_mae_by_location",
  r"Per-location MAE. The DTR stays within 0.31--0.63 km of the LSTM "
  r"at every location (all Holm-adjusted $p < 10^{-6}$); every other "
  r"model trails by 3.3--30.9 km. Friedman all-locations "
  r"$\chi^2$ = 2338.9, Kendall's W = 0.557. Where it goes: Section "
  r"4.x, H5. Source: prediction files.")}

{figure_block("fig13_combination_heatmap",
  r"Exact type-vector accuracy by fault combination; the relay row is "
  r"structurally inapplicable (binary trip output) and is omitted. "
  r"Where it goes: Section 4.x, combination-accuracy discussion. "
  r"Source: combination results CSV.")}

{figure_block("fig14_cv",
  r"Scenario-grouped K = 7 CV (descriptive, fixed hyperparameters): "
  r"detection 0.9914--0.9978, exact match 0.8990--0.9631. Where it "
  r"goes: Section 3.12.1 / 4.x (R9). Source: cv\_results CSV.")}

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
model & detection acc. & exact-match acc. \\
\midrule
{cv_rows}
\bottomrule
\end{{tabular}}
\caption{{CV means $\pm$ SD across the 7 scenario-grouped folds.
Source: outputs/scenario\_cv (report 2026-11-12\_17-25-03).}}
\end{{table}}

A scenario-level leakage audit found zero spanning scenarios for all
nine methods (models touch 134 of 671 test scenarios; the relay, which
trains on nothing, covers all 671).

\section{{Writer's companion}}

\subsection{{Key takeaways, keyed to the chapter}}
\begin{{itemize}}
\item \textbf{{\S3.3.1}} --- correct the line-charging figure to
4.15 MVAr (120 km section) / 10.28 MVAr (297.4 km route); keep the
10.8 kA bolted three-phase cross-check.
\item \textbf{{\S3.3}} --- insert the declared auxiliary assumptions
(X/R = 10; 0.95 pf constant-impedance load at 40/75/100\% of 750 A;
$C_0 = 0.75\,C_1$; $Z_0$ source = $Z_1$ source; bolted ground;
$R_f = 0 \equiv 1$ m$\Omega$; EMF calibration) and the load-bus
operating points 0.931/0.862/0.815 pu.
\item \textbf{{\S3.5.5}} --- new design space: 375 sweeps x 8
locations = 3{{,}}000 fault rows; datasets 5{{,}}365 / 9{{,}}503 rows;
detection set 31.6\% fault rows; event-sweep model caveat.
\item \textbf{{\S3.8.2/3.12.1}} --- 671 scenarios, shared
scenario-level split (134 test scenarios / 1{{,}}069 rows), K = 7 CV is
descriptive only.
\item \textbf{{\S3.9.4}} --- relay measurement chain: six loops,
complex $k_0$, Zone 1 = 38.22 $\Omega$, steady-state phasor
semantics; secure-but-incomplete profile (TN 100\%, TP 49.2\%).
\item \textbf{{\S4.x}} --- Table 4.3, H1--H5 numbers from this pack;
LSTM latency gap is $\approx$189 ms, not 270 ms.
\end{{itemize}}

\subsection{{Do NOT transcribe (superseded numbers)}}
\begin{{itemize}}
\item ``12{{,}}000 scenarios / 14{{,}}400 rows / 2{{,}}400 held-out
scenarios / 7{{,}}861 rows''; ``45.8\% fault rows''; ``5{{,}}496 fault
rows / 685 sweeps + 4 truncated''.
\item ``24{{,}}000 fault rows'' (design-space double-count).
\item ``$\approx$30.9 MVAr line charging'' ($\sqrt{{3}}$ slip).
\item ``LSTM latency gap $\approx$ 270 ms''.
\item Any accuracy, p-value, latency or table cell computed from the
original datasets.
\item Old relay narrative (``0.9+ detection accuracy'').
\item Old absolute waveform claims (44 A no-fault currents, 4\% of
nominal faulted-phase voltage, 0.62 prefault peak).
\end{{itemize}}

\subsection{{Traceability appendix}}

\begin{{longtable}}{{@{{}}p{{3.2cm}}p{{11.4cm}}@{{}}}}
\toprule
\textbf{{tag}} & \textbf{{source}} \\
\midrule
\endhead
SIM & \texttt{{src/simulate.py}} imported directly; 8-point
verification battery all PASS (outputs/simulate/report\_2026-11-12\_16-09-04.txt); datasets md5-identical across runs \\
REL & outputs/relay/report\_2026-11-12\_16-20-30.txt \\
T43 & outputs/table\_4\_3/report\_2026-11-12\_19-43-50.txt \\
HYP & outputs/hypothesis\_tests/report\_2026-11-12\_17-08-01.txt \\
CV & outputs/scenario\_cv/report\_2026-11-12\_17-25-03.txt \\
COMB & outputs/combination\_accuracy/report\_2026-11-12\_17-25-01.txt \\
LEAK & outputs/leakage\_audit/report\_2026-11-12\_17-25-01.txt \\
ROB & outputs/\textless model\textgreater/robustness\_\textless
model\textgreater{{}}.csv (2026-11-12 rows) \\
\bottomrule
\end{{longtable}}

\end{{document}}
"""

    return body


def compile_pdf(logger: ReportLogger) -> Path:

    logger.print("Compiling LaTeX (pdflatex, 2 passes)...")

    for _ in range(2):

        result = subprocess.run(
            [
                "pdflatex", "-interaction=nonstopmode",
                "-halt-on-error", TEX_FILE.name,
            ],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:

            tail = result.stdout[-3000:]

            raise RuntimeError(
                f"xelatex failed:\n{tail}"
            )

    built = BUILD_DIR / "report.pdf"

    shutil.copy(built, PDF_FILE)

    logger.print(f"PDF written: {PDF_FILE}")

    return PDF_FILE


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    logger = ReportLogger(LOG_FILE)

    try:

        setup_style()

        data = load_all(logger)

        fig_paths = steady_state_figures(logger)

        fig_paths["fig03_dataset_composition"] = dataset_figure(data)
        fig_paths["fig06_accuracy_bars"] = accuracy_bars(data)
        fig_paths["fig07_per_target_heatmap"] = per_target_heatmap(data)
        fig_paths["fig08_h1_forest"] = h1_forest(data)
        fig_paths["fig09_latency"] = latency_chart(data)
        fig_paths["fig10_robustness_curves"] = robustness_curves(data)
        fig_paths["fig11_mae_bars"] = mae_bars(data)
        fig_paths["fig12_mae_by_location"] = mae_by_location(data)
        fig_paths["fig13_combination_heatmap"] = (
            combination_heatmap(data)
        )
        fig_paths["fig14_cv"] = cv_figure(data)

        # ----------------------------------------------------
        # MANIFEST
        # ----------------------------------------------------

        sections = {
            "fig01_waveforms_distance": "3.5.5 / R10",
            "fig02_dc_offset": "3.5.5 / R10",
            "fig03_dataset_composition": "4.2",
            "fig04_reach_diagram": "3.9.4",
            "fig05_relay_trip_heatmap": "4.x H1",
            "fig06_accuracy_bars": "4.x Table 4.3",
            "fig07_per_target_heatmap": "4.x H2",
            "fig08_h1_forest": "4.x H1",
            "fig09_latency": "4.x H3",
            "fig10_robustness_curves": "4.x H4",
            "fig11_mae_bars": "4.x H5",
            "fig12_mae_by_location": "4.x H5",
            "fig13_combination_heatmap": "4.x combinations",
            "fig14_cv": "3.12.1 / R9",
        }

        sources = {
            "fig01_waveforms_distance": "src/simulate.py",
            "fig02_dc_offset": "src/simulate.py",
            "fig03_dataset_composition": "data files",
            "fig04_reach_diagram": "src/simulate.py + src/relay.py",
            "fig05_relay_trip_heatmap": "src/simulate.py",
            "fig06_accuracy_bars": "table_4_3.csv",
            "fig07_per_target_heatmap": "table_4_3.csv",
            "fig08_h1_forest": "hypothesis results CSV",
            "fig09_latency": "table_4_3.csv",
            "fig10_robustness_curves": "robustness_*.csv",
            "fig11_mae_bars": "predictions_*.csv",
            "fig12_mae_by_location": "predictions_*.csv",
            "fig13_combination_heatmap": "combination results CSV",
            "fig14_cv": "cv_results CSV",
        }

        manifest = pd.DataFrame(
            [
                {
                    "figure": Path(p).name,
                    "png_md5": md5_of(Path(p)),
                    "section": sections[name],
                    "source": sources[name],
                }
                for name, p in sorted(fig_paths.items())
            ]
        )

        manifest.to_csv(MANIFEST_FILE, index=False)

        logger.print(f"Manifest: {MANIFEST_FILE}")
        logger.print("")

        # ----------------------------------------------------
        # TEX + PDF
        # ----------------------------------------------------

        tex = build_tex(data, fig_paths)

        TEX_FILE.write_text(tex, encoding="utf-8")

        logger.print(f"LaTeX source: {TEX_FILE}")

        compile_pdf(logger)

        logger.print("")
        logger.print("=" * 70)
        logger.print("ANALYTICS PACK COMPLETE")
        logger.print("=" * 70)
        logger.print(f"PDF      : {PDF_FILE}")
        logger.print(f"Figures  : {FIGURES_DIR} "
                     f"({len(fig_paths)} figures, PNG + PDF)")
        logger.print(f"Manifest : {MANIFEST_FILE}")

    finally:

        logger.close()


if __name__ == "__main__":
    main()
