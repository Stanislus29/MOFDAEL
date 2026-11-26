"""
Waveform time-series generation for transmission-line faults.

Generates, purely in-memory (no retraining, no data-file writes),
post-fault phase current/voltage waveforms for the AB, ABG, ABCG and
ABC fault signatures. Waveforms are regenerated on the fly from the
verified physics in src/simulate.py (solve_fault + sample_fault_row)
over a post-fault window, so no existing data file, relay baseline,
model or hypothesis result is touched.

Outputs (all under outputs/waveforms/):

    report_<YYYY-MM-DD_HH-MM-SS>.txt   timestamped report (ReportLogger)
    waveforms_<SIG>.csv                per-signature time series
    waveforms_all.csv                  combined time series (all signatures)
    figures/waveform_<SIG>.png         per-signature Ia/Ib/Ic time series
    figures/waveforms_comparison.png   phase-a comparison across signatures
"""

from __future__ import annotations

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

from simulate import (  # noqa: E402  (import after path setup is intentional)

    solve_fault,

    sample_fault_row,

    prefault_state,

    SIGNATURES,

    LOAD_LEVELS,

    R_F_BOLTED_EPS,

    SAMPLE_DT,

)


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "waveforms"

FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = OUTPUT_DIR / f"report_{timestamp}.txt"

ALL_FILE = OUTPUT_DIR / "waveforms_all.csv"

# ============================================================
# ANALYSIS CONFIGURATION
# ============================================================

# Fault signatures to generate (user-requested set).
WAVEFORM_SIGNATURES = ["AB", "ABG", "ABCG", "ABC"]

# Bolted fault, 60 km along the 120 km line, normal load (75% of
# thermal rating), inception at the voltage zero crossing (0 deg) --
# the inception angle that maximises the decaying DC offset.
R_F = R_F_BOLTED_EPS

D_KM = 60.0

LOAD_LEVEL = LOAD_LEVELS[1]

THETA_INC_DEG = 0.0

# Post-fault analysis window (1 kHz grid, from simulate.SAMPLE_DT).
N_SAMPLES = 600          # 600 ms of post-fault evolution

T_MAX_MS = N_SAMPLES * SAMPLE_DT * 1e3

PHASE_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

REPORT_WIDTH = 70


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
# WAVEFORM GENERATION (via src/simulate.py physics)
# ============================================================

def generate_waveform(sig_name: str) -> pd.DataFrame:

    """
    Simulate one post-fault time series for a signature.

    Returns a DataFrame with columns time_ms, Ia, Ib, Ic, Va, Vb, Vc,
    length N_SAMPLES. Voltages are per-unit (prefault peak = 1.0),
    currents in amperes.
    """

    state = prefault_state(LOAD_LEVEL)

    fault = solve_fault(sig_name, R_F, D_KM, LOAD_LEVEL, state)

    theta_inc = np.deg2rad(THETA_INC_DEG)

    rows = [

        sample_fault_row(state, fault, R_F, theta_inc, k)

        for k in range(N_SAMPLES)

    ]

    t_ms = (np.arange(N_SAMPLES) + 1) * SAMPLE_DT * 1e3

    df = pd.DataFrame(rows, columns=PHASE_COLUMNS)

    df.insert(0, "time_ms", t_ms)

    return df


def dc_offset_amplitude(sig_name: str) -> float:

    """Initial DC-offset amplitude A_p on phase a at t = 0 (amperes)."""

    state = prefault_state(LOAD_LEVEL)

    fault = solve_fault(sig_name, R_F, D_KM, LOAD_LEVEL, state)

    theta_inc = np.deg2rad(THETA_INC_DEG)

    i_ac_0 = np.sqrt(2.0) * abs(fault["i_r_abc"][0]) * np.sin(

        theta_inc + np.angle(fault["i_r_abc"][0])

    )

    i_pref_0 = np.sqrt(2.0) * abs(state["i_r_pref_abc"][0]) * np.sin(

        theta_inc + np.angle(state["i_r_pref_abc"][0])

    )

    return float(i_pref_0 - i_ac_0)


# ============================================================
# PLOTTING
# ============================================================

def plot_timeseries(sig_name: str, df: pd.DataFrame) -> plt.Figure:

    """
    All six phase quantities (Ia, Ib, Ic, Va, Vb, Vc) for one signature.

    Layout: 3 rows x 2 columns -- currents on the left, voltages on the
    right. Returns the figure (caller is responsible for saving / closing).
    """

    fig, axes = plt.subplots(3, 2, figsize=(9.6, 7.2), sharex=True)

    layout = [

        (("Ia", "Va"), ("Phase a current", "Phase a voltage")),

        (("Ib", "Vb"), ("Phase b current", "Phase b voltage")),

        (("Ic", "Vc"), ("Phase c current", "Phase c voltage")),

    ]

    for row, ((c_left, c_right), (l_left, l_right)) in enumerate(layout):

        ax_l = axes[row, 0]

        ax_r = axes[row, 1]

        ax_l.plot(

            df["time_ms"], df[c_left], linewidth=1.0, color="#1f77b4"

        )

        ax_l.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)

        ax_l.set_ylabel(l_left + f" ({c_left})")

        if row == 0:

            ax_l.set_title(sig_name, fontsize=10, loc="left")

        ax_l.grid(True, alpha=0.3)

        ax_r.plot(

            df["time_ms"], df[c_right], linewidth=1.0, color="#d62728"

        )

        ax_r.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)

        ax_r.set_ylabel(l_right + f" ({c_right})")

        ax_r.grid(True, alpha=0.3)

    for ax in axes.flat:

        ax.set_xlabel("time after fault inception (ms)")

    fig.suptitle(

        f"{sig_name} fault waveforms  --  bolted, {D_KM:g} km, "

        f"{100 * LOAD_LEVEL:g}% load, inception {THETA_INC_DEG:g} deg  "

        f"(DC offset A_a = {dc_offset_amplitude(sig_name):.1f} A)",

        fontsize=10,

    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_comparison(dfs: dict[str, pd.DataFrame]) -> plt.Figure:

    """
    All six phase quantities across the four signatures.

    Layout: 2 rows x 3 columns -- one panel per quantity (Ia, Ib, Ic,
    Va, Vb, Vc), each panel shows all four signatures. Returns the figure.
    """

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=True)

    quantities = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

    labels = {

        "Ia": "Phase a current (A)",

        "Ib": "Phase b current (A)",

        "Ic": "Phase c current (A)",

        "Va": "Phase a voltage (pu)",

        "Vb": "Phase b voltage (pu)",

        "Vc": "Phase c voltage (pu)",

    }

    colors = plt.cm.tab10.colors

    for ax, qty in zip(axes.flat, quantities):

        for i, sig in enumerate(WAVEFORM_SIGNATURES):

            ax.plot(

                dfs[sig]["time_ms"],

                dfs[sig][qty],

                linewidth=1.0,

                color=colors[i],

                label=sig,

            )

        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)

        ax.set_ylabel(labels[qty])

        ax.set_title(qty, fontsize=10)

        ax.grid(True, alpha=0.3)

        if ax is axes.flat[-1]:

            ax.legend(fontsize=8, loc="best")

    for ax in axes.flat:

        ax.set_xlabel("time after inception (ms)")

    fig.suptitle(

        "Phase current and voltage vs time  --  "

        f"bolted, {D_KM:g} km, {100 * LOAD_LEVEL:g}% load, "

        f"inception {THETA_INC_DEG:g} deg (max DC offset)",

        fontsize=11,

    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    logger = ReportLogger(REPORT_FILE)

    try:

        logger.print("=" * REPORT_WIDTH)

        logger.print("FAULT WAVEFORM GENERATION -- AB, ABG, ABCG, ABC")

        logger.print("=" * REPORT_WIDTH)

        logger.print(f"Timestamp: {timestamp}")

        logger.print(f"Source physics: src/simulate.py")

        logger.print("")

        # ---- parameters ------------------------------------------------

        logger.print("-" * REPORT_WIDTH)

        logger.print("CONFIGURATION")

        logger.print("-" * REPORT_WIDTH)

        logger.print(f"Signatures       : {', '.join(WAVEFORM_SIGNATURES)}")

        logger.print(f"Fault resistance : {R_F} ohm (bolted)")

        logger.print(f"Fault location   : {D_KM} km (of 120 km line)")

        logger.print(f"Load level       : {100 * LOAD_LEVEL:g}% (normal)")

        logger.print(

            f"Inception angle  : {THETA_INC_DEG} deg "

            "(voltage zero crossing -> max DC offset)"

        )

        logger.print(f"Sampling interval: {SAMPLE_DT * 1e3:.3f} ms (1 kHz)")

        logger.print(f"Window length    : {N_SAMPLES} samples = {T_MAX_MS:.0f} ms")

        logger.print("")

        # ---- generate + write ------------------------------------------

        dfs: dict[str, pd.DataFrame] = {}

        combined_frames: list[pd.DataFrame] = []

        for sig in WAVEFORM_SIGNATURES:

            df = generate_waveform(sig)

            dfs[sig] = df

            df_with_sig = df.copy()

            df_with_sig.insert(0, "signature", sig)

            combined_frames.append(df_with_sig)

            sig_file = OUTPUT_DIR / f"waveforms_{sig}.csv"

            df.to_csv(sig_file, index=False)

            logger.print("-" * REPORT_WIDTH)

            logger.print(f"WAVEFORM -- {sig} FAULT")

            logger.print("-" * REPORT_WIDTH)

            logger.print(f"Samples          : {len(df)}")

            logger.print(f"Time span        : {df['time_ms'].iloc[0]:.1f} .. "

                         f"{df['time_ms'].iloc[-1]:.1f} ms")

            logger.print(f"DC offset (phase a): {dc_offset_amplitude(sig):.4f} A")

            for phase in ("Ia", "Ib", "Ic"):

                logger.print(

                    f"  {phase}: peak {df[phase].abs().max():.1f} A, "

                    f"final {df[phase].iloc[-1]:.1f} A"

                )

            logger.print(f"Written          : {sig_file}")

            logger.print("")

        combined = pd.concat(combined_frames, ignore_index=True)

        combined.to_csv(ALL_FILE, index=False)

        logger.print("-" * REPORT_WIDTH)

        logger.print("COMBINED WAVEFORM FILE")

        logger.print("-" * REPORT_WIDTH)

        logger.print(f"Rows             : {len(combined)}")

        logger.print(f"Signatures       : {', '.join(WAVEFORM_SIGNATURES)}")

        logger.print(f"Written          : {ALL_FILE}")

        logger.print("")

        # ---- figures ---------------------------------------------------

        fig_paths: list[Path] = []

        for sig in WAVEFORM_SIGNATURES:

            fig = plot_timeseries(sig, dfs[sig])

            path = FIGURES_DIR / f"waveform_{sig}.png"

            fig.savefig(path, dpi=200)

            plt.close(fig)

            fig_paths.append(path)

        fig = plot_comparison(dfs)

        path = FIGURES_DIR / "waveforms_comparison.png"

        fig.savefig(path, dpi=200)

        plt.close(fig)

        fig_paths.append(path)

        logger.print("-" * REPORT_WIDTH)

        logger.print("OUTPUT FILES")

        logger.print("-" * REPORT_WIDTH)

        logger.print(f"Report      : {REPORT_FILE}")

        logger.print(f"Per-sig CSVs: {', '.join(str(OUTPUT_DIR / f'waveforms_{s}.csv') for s in WAVEFORM_SIGNATURES)}")

        logger.print(f"Combined    : {ALL_FILE}")

        for p in fig_paths:

            logger.print(f"Figure      : {p}")

        logger.print("")

        logger.print("Waveform generation complete.")

    finally:

        logger.close()


if __name__ == "__main__":

    main()
