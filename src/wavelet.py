"""
Wavelet analysis of transmission-line fault transients.

Generates, purely in-memory (no retraining, no data-file writes),
Continuous Wavelet Transform (CWT) scalograms and time-frequency
features for the ABG, AB, ABC and ABCG fault signatures.

Waveforms are regenerated on the fly from the verified physics in
src/simulate.py (solve_fault + sample_fault_row) over a post-fault
window, so no existing data file, relay baseline, model or hypothesis
result is touched.

Outputs (all under outputs/wavelet/):

    report_<YYYY-MM-DD_HH-MM-SS>.txt   timestamped report (ReportLogger)
    wavelet_features.csv                per-signature time-frequency features
    figures/wavelet_<SIG>_scalogram.png per-signature 3-phase scalograms
    figures/wavelet_comparison.png      phase-a scalogram comparison
    figures/wavelet_timeseries.png      phase-a time-series comparison
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np

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
    SIGNATURE_ORDER,
    LOAD_LEVELS,
    R_F_BOLTED_EPS,
    SAMPLE_DT,
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "wavelet"

FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = OUTPUT_DIR / f"report_{timestamp}.txt"

FEATURES_FILE = OUTPUT_DIR / "wavelet_features.csv"

# ============================================================
# ANALYSIS CONFIGURATION
# ============================================================

# Fault signatures to analyse (user-requested set).
WAVELET_SIGNATURES = ["ABG", "AB", "ABC", "ABCG"]

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

# CWT scales (in samples) -- log-spaced. With dt = 1 ms and Morlet
# w0 = 6, pseudo-frequency f = w0 / (2 * pi * a * dt), so this spans
# roughly 16 Hz (a = 60) to 477 Hz (a = 2): the fundamental (50 Hz)
# plus the decaying DC/transient content.
W0 = 6.0                 # Morlet centre-frequency parameter
SCALES = np.logspace(np.log10(2.0), np.log10(60.0), 36)

# Frequency bands (Hz) for band-energy features.
BAND_TRANSIENT = (0.0, 30.0)     # decaying DC / low-freq transient
BAND_FUNDAMENTAL = (45.0, 55.0)  # 50 Hz fundamental
BAND_HIGH = (60.0, 1000.0)       # higher-frequency content

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
# CONTINUOUS WAVELET TRANSFORM (Morlet)
# ============================================================

def morlet(t: np.ndarray, w0: float = W0) -> np.ndarray:

    """Complex Morlet mother wavelet."""

    return (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-0.5 * t ** 2)


def pseudo_frequency(scales: np.ndarray, dt: float,
                     w0: float = W0) -> np.ndarray:

    """Pseudo-frequency (Hz) of each CWT scale for the Morlet wavelet."""

    return w0 / (2.0 * np.pi * scales * dt)


def cwt(signal: np.ndarray, scales: np.ndarray,
        dt: float, w0: float = W0) -> np.ndarray:

    """
    Continuous wavelet transform by direct (zero-padded) convolution.

    W[a, n] = sum_j x[n + j] * conj(psi_a[j]),
    psi_a[j] = a^-1/2 * psi(j / a),  psi = Morlet.

    Returns a complex array of shape (len(scales), len(signal)).
    """

    n = len(signal)

    W = np.zeros((len(scales), n), dtype=complex)

    for i, a in enumerate(scales):

        half = int(np.ceil(6.0 * a))  # +/-6a samples; exp(-18) ~ 1.5e-8

        j = np.arange(-half, half + 1, dtype=float)

        psi_a = (a ** -0.5) * morlet(j / a, w0)

        cj = np.conj(psi_a)

        xpad = np.zeros(n + 2 * half)

        xpad[half:half + n] = signal

        col = np.zeros(n, dtype=complex)

        for jj in range(-half, half + 1):

            col += xpad[half + jj:half + jj + n] * cj[jj + half]

        W[i] = col

    return W


# ============================================================
# WAVEFORM GENERATION (via src/simulate.py physics)
# ============================================================

def generate_waveforms(sig_name: str) -> dict[str, np.ndarray]:

    """
    Simulate one post-fault time series for a signature.

    Returns dict with keys 't_ms' (1-D, ms) and 'Ia','Ib','Ic',
    'Va','Vb','Vc' (1-D, A / pu), length N_SAMPLES.
    """

    state = prefault_state(LOAD_LEVEL)

    fault = solve_fault(sig_name, R_F, D_KM, LOAD_LEVEL, state)

    theta_inc = np.deg2rad(THETA_INC_DEG)

    rows = [
        sample_fault_row(state, fault, R_F, theta_inc, k)
        for k in range(N_SAMPLES)
    ]

    t_ms = (np.arange(N_SAMPLES) + 1) * SAMPLE_DT * 1e3

    out: dict[str, np.ndarray] = {"t_ms": t_ms}

    for col in ("Ia", "Ib", "Ic", "Va", "Vb", "Vc"):

        out[col] = np.array([r[col] for r in rows], dtype=float)

    return out


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
# FEATURE EXTRACTION
# ============================================================

def band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:

    lo, hi = band

    return (freqs >= lo) & (freqs <= hi)


def extract_features(W: np.ndarray, freqs: np.ndarray,
                     t_ms: np.ndarray) -> dict[str, float]:

    """Time-frequency features from one scalogram (|W| over scales x time)."""

    mag = np.abs(W)

    energy = mag ** 2                       # scale x time

    total_energy = float(energy.sum())

    band_energies = {
        f"energy_{name.replace(' ', '_')}": float(
            energy[band_mask(freqs, band)].sum()
        )
        for name, band in (
            ("transient", BAND_TRANSIENT),
            ("fundamental", BAND_FUNDAMENTAL),
            ("high", BAND_HIGH),
        )
    }

    energy_by_scale = energy.sum(axis=1)    # energy per scale

    dom_idx = int(np.argmax(energy_by_scale))

    dominant_freq = float(freqs[dom_idx])

    centroid_freq = float(
        (freqs * energy_by_scale).sum() / (energy_by_scale.sum() + 1e-12)
    )

    energy_over_time = energy.sum(axis=0)   # energy per time sample

    time_of_max_ms = float(t_ms[int(np.argmax(energy_over_time))])

    return {
        "total_energy": total_energy,
        "dominant_freq_hz": dominant_freq,
        "centroid_freq_hz": centroid_freq,
        "time_of_max_energy_ms": time_of_max_ms,
        **band_energies,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_scalogram(sig_name: str, W_by_phase: dict[str, np.ndarray],
                   freqs: np.ndarray, t_ms: np.ndarray,
                   features: dict[str, dict[str, float]]) -> Path:

    """Three-panel scalogram (Ia, Ib, Ic) for one signature."""

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.6),
                             sharex=True, sharey=True)

    phase_labels = {"Ia": "Phase a", "Ib": "Phase b", "Ic": "Phase c"}

    vmax = max(np.abs(W_by_phase[p]).max() for p in ("Ia", "Ib", "Ic"))

    for ax, phase in zip(axes, ("Ia", "Ib", "Ic")):

        W = W_by_phase[phase]

        extent = [t_ms[0], t_ms[-1], freqs[-1], freqs[0]]

        im = ax.imshow(
            20.0 * np.log10(np.abs(W) + 1e-12),
            extent=extent, aspect="auto", cmap="magma",
            vmin=20.0 * np.log10(vmax + 1e-12) - 60.0,
            vmax=20.0 * np.log10(vmax + 1e-12),
        )

        ax.set_yscale("log")

        ax.set_ylabel("pseudo-frequency (Hz)")

        ax.set_title(
            f"{phase_labels[phase]} ({phase})  |  "
            f"dom {features[phase]['dominant_freq_hz']:.1f} Hz,  "
            f"centroid {features[phase]['centroid_freq_hz']:.1f} Hz",
            fontsize=9, loc="left",
        )

        fig.colorbar(im, ax=ax, shrink=0.85, label="|W| (dB)")

    axes[-1].set_xlabel("time after fault inception (ms)")

    fig.suptitle(
        f"{sig_name} fault  --  CWT (Morlet, w0={W0})  |  "
        f"bolted, {D_KM:g} km, {100 * LOAD_LEVEL:g}% load, "
        f"inception {THETA_INC_DEG:g} deg  |  DC offset A_a = "
        f"{dc_offset_amplitude(sig_name):.1f} A",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = FIGURES_DIR / f"wavelet_{sig_name}_scalogram.png"

    fig.savefig(path, dpi=200)

    plt.close(fig)

    return path


def plot_comparison(W_by_sig: dict[str, dict[str, np.ndarray]],
                    freqs: np.ndarray, t_ms: np.ndarray) -> Path:

    """2x2 phase-a scalogram comparison across the four signatures."""

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2),
                             sharex=True, sharey=True)

    vmax = max(
        np.abs(W_by_sig[s]["Ia"]).max() for s in WAVELET_SIGNATURES
    )

    for ax, sig in zip(axes.flat, WAVELET_SIGNATURES):

        W = W_by_sig[sig]["Ia"]

        extent = [t_ms[0], t_ms[-1], freqs[-1], freqs[0]]

        ax.imshow(
            20.0 * np.log10(np.abs(W) + 1e-12),
            extent=extent, aspect="auto", cmap="magma",
            vmin=20.0 * np.log10(vmax + 1e-12) - 60.0,
            vmax=20.0 * np.log10(vmax + 1e-12),
        )

        ax.set_yscale("log")

        ax.set_title(sig, fontsize=10)

    for ax in axes.flat:

        ax.set_xlabel("time (ms)")

        ax.set_ylabel("pseudo-frequency (Hz)")

    fig.suptitle(
        "Phase-a CWT scalogram comparison (|W|, dB)  --  "
        f"bolted, {D_KM:g} km, {100 * LOAD_LEVEL:g}% load, "
        f"inception {THETA_INC_DEG:g} deg",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = FIGURES_DIR / "wavelet_comparison.png"

    fig.savefig(path, dpi=200)

    plt.close(fig)

    return path


def plot_timeseries(W_by_sig: dict[str, dict[str, np.ndarray]],
                    waveforms: dict[str, dict[str, np.ndarray]]) -> Path:

    """Phase-a current time series for the four signatures."""

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True,
                             sharey=True)

    for ax, sig in zip(axes.flat, WAVELET_SIGNATURES):

        t = waveforms[sig]["t_ms"]

        ia = waveforms[sig]["Ia"]

        ax.plot(t, ia, linewidth=1.0)

        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)

        ax.set_title(sig, fontsize=10)

        ax.grid(True, alpha=0.3)

        ax.set_ylabel("Ia (A)")

    axes[-1][0].set_xlabel("time after inception (ms)")

    axes[-1][1].set_xlabel("time after inception (ms)")

    fig.suptitle(
        "Phase-a current vs time  --  "
        f"bolted, {D_KM:g} km, {100 * LOAD_LEVEL:g}% load, "
        f"inception {THETA_INC_DEG:g} deg (max DC offset)",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = FIGURES_DIR / "wavelet_timeseries.png"

    fig.savefig(path, dpi=200)

    plt.close(fig)

    return path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    logger = ReportLogger(REPORT_FILE)

    try:

        logger.print("=" * REPORT_WIDTH)

        logger.print("FAULT WAVELET ANALYSIS -- ABG, AB, ABC, ABCG")

        logger.print("=" * REPORT_WIDTH)

        logger.print(f"Timestamp: {timestamp}")

        logger.print(f"Source physics: src/simulate.py")

        logger.print("")

        # ---- parameters ------------------------------------------------

        logger.print("-" * REPORT_WIDTH)

        logger.print("CONFIGURATION")

        logger.print("-" * REPORT_WIDTH)

        logger.print(f"Signatures       : {', '.join(WAVELET_SIGNATURES)}")

        logger.print(f"Fault resistance : {R_F} ohm (bolted)")

        logger.print(f"Fault location   : {D_KM} km (of {120.0} km line)")

        logger.print(f"Load level       : {100 * LOAD_LEVEL:g}% (normal)")

        logger.print(
            f"Inception angle  : {THETA_INC_DEG} deg "
            "(voltage zero crossing -> max DC offset)"
        )

        logger.print(f"Sampling interval: {SAMPLE_DT * 1e3:.3f} ms (1 kHz)")

        logger.print(f"Window length    : {N_SAMPLES} samples = {T_MAX_MS:.0f} ms")

        logger.print(f"Mother wavelet   : Morlet, w0 = {W0}")

        logger.print(f"CWT scales       : {len(SCALES)}, "
                     f"{SCALES[0]:.2f}..{SCALES[-1]:.2f} samples")

        logger.print(
            f"Pseudo-frequency : "
            f"{pseudo_frequency(np.array([SCALES[-1]]), SAMPLE_DT, W0).item():.1f} Hz .. "
            f"{pseudo_frequency(np.array([SCALES[0]]), SAMPLE_DT, W0).item():.1f} Hz"
        )

        logger.print(f"Frequency bands  : transient {BAND_TRANSIENT} Hz, "
                     f"fundamental {BAND_FUNDAMENTAL} Hz, "
                     f"high {BAND_HIGH} Hz")

        logger.print("")

        # ---- generate + analyse ----------------------------------------

        freqs = pseudo_frequency(SCALES, SAMPLE_DT, W0)

        waveforms: dict[str, dict[str, np.ndarray]] = {}

        W_by_sig: dict[str, dict[str, np.ndarray]] = {}

        feature_rows: list[dict] = []

        for sig in WAVELET_SIGNATURES:

            waveforms[sig] = generate_waveforms(sig)

            W_by_phase: dict[str, np.ndarray] = {}

            sig_features: dict[str, dict[str, float]] = {}

            for phase in ("Ia", "Ib", "Ic"):

                W = cwt(waveforms[sig][phase], SCALES, SAMPLE_DT, W0)

                W_by_phase[phase] = W

                sig_features[phase] = extract_features(W, freqs,
                                                       waveforms[sig]["t_ms"])

                feature_rows.append({

                    "signature": sig,

                    "phase": phase,

                    "r_f_ohm": R_F,

                    "d_km": D_KM,

                    "load_pct": 100 * LOAD_LEVEL,

                    "inception_deg": THETA_INC_DEG,

                    "total_energy": sig_features[phase]["total_energy"],

                    "energy_transient": sig_features[phase][
                        "energy_transient"],

                    "energy_fundamental": sig_features[phase][
                        "energy_fundamental"],

                    "energy_high": sig_features[phase]["energy_high"],

                    "dominant_freq_hz": sig_features[phase][
                        "dominant_freq_hz"],

                    "centroid_freq_hz": sig_features[phase][
                        "centroid_freq_hz"],

                    "time_of_max_energy_ms": sig_features[phase][
                        "time_of_max_energy_ms"],

                })

            W_by_sig[sig] = W_by_phase

            logger.print("-" * REPORT_WIDTH)

            logger.print(f"RESULTS -- {sig} FAULT")

            logger.print("-" * REPORT_WIDTH)

            logger.print(
                f"  DC offset amplitude (phase a, t=0) : "
                f"{dc_offset_amplitude(sig):.4f} A"
            )

            for phase in ("Ia", "Ib", "Ic"):

                f = sig_features[phase]

                logger.print(

                    f"  {phase}: dominant {f['dominant_freq_hz']:.2f} Hz, "

                    f"centroid {f['centroid_freq_hz']:.2f} Hz, "

                    f"max energy at {f['time_of_max_energy_ms']:.1f} ms"

                )

            logger.print("")

        # ---- write features --------------------------------------------

        import pandas as pd  # local import to keep top import order tidy

        features_df = pd.DataFrame(feature_rows)

        features_df.to_csv(FEATURES_FILE, index=False)

        logger.print("-" * REPORT_WIDTH)

        logger.print("FEATURE SUMMARY (signature level)")

        logger.print("-" * REPORT_WIDTH)

        for sig in WAVELET_SIGNATURES:

            sub = features_df[features_df["signature"] == sig]

            logger.print(

                f"  {sig}: dominant = "

                + ", ".join(

                    f"{r['phase']} {r['dominant_freq_hz']:.1f} Hz"

                    for _, r in sub.iterrows()

                )

                + "; transient energy = "

                + ", ".join(

                    f"{r['phase']} {r['energy_transient']:.4e}"

                    for _, r in sub.iterrows()

                )

            )

        logger.print("")

        # ---- figures ---------------------------------------------------

        fig_paths: list[Path] = []

        for sig in WAVELET_SIGNATURES:

            fig_paths.append(

                plot_scalogram(sig, W_by_sig[sig], freqs,
                               waveforms[sig]["t_ms"],
                               {p: extract_features(
                                   W_by_sig[sig][p], freqs,
                                   waveforms[sig]["t_ms"])
                               for p in ("Ia", "Ib", "Ic")})

            )

        fig_paths.append(

            plot_comparison(W_by_sig, freqs,
                            waveforms[WAVELET_SIGNATURES[0]]["t_ms"])

        )

        fig_paths.append(

            plot_timeseries(W_by_sig, waveforms)

        )

        logger.print("-" * REPORT_WIDTH)

        logger.print("OUTPUT FILES")

        logger.print("-" * REPORT_WIDTH)

        logger.print(f"Report      : {REPORT_FILE}")

        logger.print(f"Features    : {FEATURES_FILE}")

        for p in fig_paths:

            logger.print(f"Figure      : {p}")

        logger.print("")

        logger.print("Wavelet analysis complete.")

    finally:

        logger.close()


if __name__ == "__main__":

    main()
