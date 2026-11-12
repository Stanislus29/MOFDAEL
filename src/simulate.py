"""
R1 dataset regeneration: Python re-simulation of the transmission-line
fault dataset under the CORRECTED line parameters (Table 3.2 of the
corrected dissertation).

Why this script exists
----------------------

The original raw datasets (data/classdetectfault_data.csv,
data/detectfault_data.csv) were generated from a Simulink model whose
Table 3.2 parameters were wrong by orders of magnitude (X1 0.003
Ω/km, X0 0.006 Ω/km, capacitance in F/km, fault level in kVA) and
whose fault physics is internally inconsistent (median fault current
is flat in Loc, faulted-phase voltage sits at ~4% of nominal
regardless of distance). Under R1, results computed from those data
cannot be corrected by fixing the table — only re-simulation can.

This script re-generates both datasets from scratch with an
analytical time-domain simulation of the corrected 161 kV line model.
Every number downstream (relay baseline, all 8 model trainings,
hypothesis tests, tables, audits, CV) is regenerable from this file
plus the standard training scripts.

Line parameters (corrected Table 3.2)
-------------------------------------

    Nominal voltage        161 kV line-to-line RMS (GRIDCo)
    Modelled section       120 km (Section 3.15 "modelled section only")
    Z1                     0.080 + j0.390  Ω/km
    Z0                     0.228 + j1.352  Ω/km
    C1                     12.74 nF/km
    Fault level            3000 MVA at the source bus
    Frequency              50 Hz
    Thermal rating         750 A continuous

Declared auxiliary modelling assumptions (Table 3.2 does not specify
them; each is a standard HV textbook value and is stated in the
report):

    Source X/R             10
    Load power factor      0.95 lagging (constant-impedance load)
    Zero-seq capacitance   C0 = 0.75 * C1
    Source zero-seq Z      equal to positive-sequence (solidly grounded)
    Negative sequence      Z2_line = Z1_line, Z2_source = Z1_source
                           (transposed line)
    Ground return          bolted (0 Ω); R_f is the per-phase
                           fault-path resistance; for the single-line-
                           to-ground signature R_f is the total
                           phase-to-ground path resistance
    Bolted faults          R_f = 0 is implemented as 1 mΩ
    Prefault operating     the source EMF is calibrated per load level
                           so the relay-bus prefault voltage is exactly
                           1.0 per-unit at 0 degrees

Fault physics
-------------

1.  Pre-fault steady state: the loaded line (source EMF behind the
    source impedance, two nominal-pi line segments split at the fault
    point, constant-impedance load at the receiving end) is solved in
    the symmetrical-component domain. Zero sequence has no load path
    (delta load termination).

2.  Fault solution: for each fault signature the phase-domain fault
    admittance matrix at the fault bus is transformed into the
    sequence domain and combined with the Thevenin impedances at the
    fault point (source side plus, for positive/negative sequence,
    the parallel backfeed through the far section and the load).
    The 3x3 sequence-current system
        (I + Yf_seq @ diag(Z_th)) I_f = Yf_seq @ V_pref_F
    is solved directly; no closed-form per-type formulas are used,
    the interconnection lives entirely in Yf_seq.

3.  Signatures implemented (the five present in the original data,
    decoded from the G/C/B/A encoding):

        AG    G=1,C=0,B=0,A=1   single line to ground (phase a)
        BC    G=0,C=1,B=1,A=0   phase-to-phase (b-c, ungrounded)
        BCG   G=0,C=1,B=1,A=1   two line to ground (b-c)
        ABG   G=1,C=0,B=1,A=1   two line to ground (a-b)
        ABCG  G=1,C=1,B=1,A=1   three phase to ground

4.  Time domain: phase quantities are synthesised as
        v_p(t) = sqrt(2)|V_p| sin(w t + theta_inc + angle(V_p))
    with the decaying DC offset on the currents fixed by continuity
    of the faulted loop current at the inception instant:
        i_p(t) = sqrt(2)|I_p| sin(w t + theta_inc + angle(I_p))
                 + A_p exp(-t/tau)
        A_p    = i_pref_p(0) - sqrt(2)|I_p| sin(theta_inc + angle(I_p))
        tau    = X_loop / (w R_loop)   (positive-seq loop)
    The inception angle theta_inc is the design factor (0, 45, 90,
    135, 180 degrees relative to the phase-a voltage zero crossing).

Sampling / dataset representation (mirrors the original data model)
-------------------------------------------------------------------

    - Waveforms are defined at a 1 kHz sampling grid (50 Hz system,
      18 degrees per sample), consistent with Section 3.5.5.
    - Fault rows: each fault scenario is exported as one 8-row
      location sweep (Loc = 15..120 km, ascending). Row k of a sweep
      is the instantaneous sample of the post-fault waveform at
      t = (k+1) ms after fault inception — consecutive samples on
      the 1 kHz grid, so a sweep spans 8 ms of post-fault evolution
      including the DC-offset decay, and the inception angle shifts
      where in the fundamental cycle the window sits.
    - No-fault rows: ONE continuous steady-state recording (1 kHz
      consecutive samples) at the normal load level, segmented
      downstream into 8-row observation windows by
      src/scenarios.py.
    - Units: voltages are per-unit instantaneous (prefault phase
      peak = 1.0); currents are instantaneous amperes.
    - Noise: none baked in. The robustness machinery applies the
      SNR sweep synthetically exactly as the pipeline does today.

Datasets written (originals are NOT touched)
--------------------------------------------

    data/classdetectfault_sim.csv
        Loc, G, C, B, A, Ia..Vc. Fault rows first (sweep-major),
        then the first 2,365 samples of the no-fault recording
        (mirrors the original 7,861-row layout).
    data/detectfault_sim.csv
        Loc, Output (S), Ia..Vc. The same fault rows (Output (S)=1)
        followed by all 6,503 no-fault samples (Output (S)=0);
        mirrors the original detection dataset's structure.

Verification cross-checks (Section 3.3.1) are computed and printed
in the report; see VERIFICATION section of the output report.
"""

from __future__ import annotations

import cmath
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "simulate"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

REPORT_FILE = OUTPUT_DIR / f"report_{timestamp}.txt"

CLASS_FILE = DATA_DIR / "classdetectfault_sim.csv"

DETECT_FILE = DATA_DIR / "detectfault_sim.csv"

# Per-row distance-relay measurements (six loop impedances, ohms),
# aligned row-by-row with classdetectfault_sim.csv. Consumed by
# src/relay.py for the Zone 1 decision.
RELAY_MEASUREMENTS_FILE = OUTPUT_DIR / "relay_measurements_sim.csv"

# ============================================================
# PARAMETERS (corrected Table 3.2 + declared assumptions)
# ============================================================

F_NOM = 50.0                      # Hz
OMEGA = 2.0 * np.pi * F_NOM       # rad/s

V_LL_RMS = 161_000.0              # V, line-to-line
V_PH_RMS = V_LL_RMS / np.sqrt(3.0)  # V, phase-to-neutral

LINE_LENGTH_KM = 120.0            # modelled section (Section 3.15)

Z1_PER_KM = 0.080 + 0.390j        # ohm/km, corrected
Z0_PER_KM = 0.228 + 1.352j        # ohm/km, corrected
Z2_PER_KM = Z1_PER_KM             # transposed-line assumption

C1_PER_KM = 12.74e-9              # F/km, corrected
C0_PER_KM = 0.75 * C1_PER_KM      # declared assumption

FAULT_LEVEL_VA = 3000e6           # 3000 MVA at the source bus
ZS_MAG = V_LL_RMS ** 2 / FAULT_LEVEL_VA   # ohm, 8.6403
SOURCE_XR = 10.0                  # declared assumption
ZS_R = ZS_MAG / np.sqrt(1.0 + SOURCE_XR ** 2)
ZS_X = SOURCE_XR * ZS_R
ZS1 = ZS_R + 1j * ZS_X            # positive-seq source impedance
ZS2 = ZS1                         # negative-seq (transposed system)
ZS0 = ZS1                         # solidly grounded, declared

THERMAL_RATING_A = 750.0          # A continuous
LOAD_PF = 0.95                    # declared assumption
LOAD_LEVELS = [0.40, 0.75, 1.00]  # of thermal rating
LOAD_LEVEL_NAMES = {0.40: "light (40%)", 0.75: "normal (75%)", 1.00: "heavy (100%)"}

R_F_LEVELS = [0.0, 10.0, 25.0, 50.0, 100.0]   # ohm per phase
R_F_BOLTED_EPS = 1e-3             # R_f = 0 implemented as 1 mOhm

INCEPTION_ANGLES_DEG = [0.0, 45.0, 90.0, 135.0, 180.0]

LOCATIONS_KM = [15, 30, 45, 60, 75, 90, 105, 120]

SAMPLE_DT = 1.0e-3                # 1 kHz sampling grid
SWEEP_ROWS = 8                    # one row per location

NOFAULT_LEVEL = 0.75              # NF stratum recorded at normal load
NOFAULT_SAMPLES_DETECT = 6503     # mirrors original detection dataset
NOFAULT_SAMPLES_CLASS = 2365      # mirrors original classification set

# Fault signatures: name -> (faulted phases (a=0,b=1,c=2), grounded,
# (G, C, B, A) encoding used by the datasets)
SIGNATURES: dict[str, tuple[tuple[int, ...], bool, tuple[int, int, int, int]]] = {
    "AG": ((0,), True, (1, 0, 0, 1)),
    "BC": ((1, 2), False, (0, 1, 1, 0)),
    "BCG": ((1, 2), True, (0, 1, 1, 1)),
    "ABG": ((0, 1), True, (1, 0, 1, 1)),
    "ABCG": ((0, 1, 2), True, (1, 1, 1, 1)),
}

SIGNATURE_ORDER = ["AG", "BC", "BCG", "ABG", "ABCG"]


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
# SYMMETRICAL-COMPONENT HELPERS
# ============================================================

_A_ROT = cmath.exp(2j * np.pi / 3.0)   # operator a = 1 angle 120


def abc_to_120(x_abc: np.ndarray) -> np.ndarray:
    """Phase phasors -> sequence phasors [I1, I2, I0]."""

    a = _A_ROT

    A = np.array([
        [1, 1, 1],
        [1, a, a * a],
        [1, a * a, a],
    ], dtype=complex) / 3.0

    return A @ np.asarray(x_abc, dtype=complex)


def seq120_to_abc(x_120: np.ndarray) -> np.ndarray:
    """Sequence phasors [I1, I2, I0] -> phase phasors [a, b, c]."""

    a = _A_ROT

    Ainv = np.array([
        [1, 1, 1],
        [1, a * a, a],
        [1, a, a * a],
    ], dtype=complex)

    return Ainv @ np.asarray(x_120, dtype=complex)


def seq_params(seq: int) -> tuple[complex, complex, complex]:
    """Per-sequence (line z per km, line shunt Y per km, source Z)."""

    if seq == 0:

        return Z0_PER_KM, 1j * OMEGA * C0_PER_KM, ZS0

    if seq == 1:

        return Z1_PER_KM, 1j * OMEGA * C1_PER_KM, ZS1

    return Z2_PER_KM, 1j * OMEGA * C1_PER_KM, ZS2


def load_impedance(load_level: float) -> complex:
    """
    Constant-impedance three-phase load (per phase, wye-equivalent)
    drawing load_level * thermal rating at LOAD_PF lagging when the
    phase voltage is nominal.
    """

    i_rms = load_level * THERMAL_RATING_A

    phi = np.arccos(LOAD_PF)   # impedance angle, inductive (+)

    return V_PH_RMS / i_rms * np.exp(1j * phi)


# ============================================================
# NETWORK SOLUTION (sequence domain, 3 nodes: R=relay, F=fault, L=load)
# ============================================================

def seq_load_z(seq: int, load_z: complex | None) -> complex | None:
    """
    The delta-connected load has no zero-sequence path: the load
    admittance exists in the positive- and negative-sequence
    networks only.
    """

    return load_z if seq in (1, 2) else None


def build_nodal(seq: int, d_km: float, load_z: complex | None) -> tuple[np.ndarray, bool]:
    """
    Nodal admittance matrix (nodes R, F, L) for one sequence.

    Merged layout when the fault sits at the receiving end
    (d = L): nodes F and L collapse; returns (Y, merged).
    """

    z_km, y_sh_km, _ = seq_params(seq)

    merged = (LINE_LENGTH_KM - d_km) < 1e-9

    load_z = seq_load_z(seq, load_z)

    y_series_rf = 1.0 / (z_km * d_km)

    if merged:

        Y = np.zeros((2, 2), dtype=complex)

        y_sh = y_sh_km * LINE_LENGTH_KM / 2.0  # each pi half at R and FL

        # branch R-FL contributes its admittance to BOTH endpoints
        Y[0, 0] += y_sh + 1.0 / (z_km * d_km)

        Y[1, 1] += y_sh + 1.0 / (z_km * d_km)

        Y[0, 1] = Y[1, 0] = -1.0 / (z_km * d_km)

        if load_z is not None:

            Y[1, 1] += 1.0 / load_z

        return Y, merged

    y_series_fl = 1.0 / (z_km * (LINE_LENGTH_KM - d_km))

    y_sh_r = y_sh_km * d_km / 2.0

    y_sh_f = y_sh_km * LINE_LENGTH_KM / 2.0

    y_sh_l = y_sh_km * (LINE_LENGTH_KM - d_km) / 2.0

    Y = np.zeros((3, 3), dtype=complex)

    # each branch adds its admittance to both endpoint self-terms
    Y[0, 0] += y_sh_r + y_series_rf

    Y[1, 1] += y_sh_f + y_series_rf + y_series_fl

    Y[2, 2] += y_sh_l + y_series_fl

    Y[0, 1] = Y[1, 0] = -y_series_rf

    Y[1, 2] = Y[2, 1] = -y_series_fl

    if load_z is not None:

        Y[2, 2] += 1.0 / load_z

    return Y, merged


def solve_network(seq: int, d_km: float, load_z: complex | None,
                  e_source: complex, fault_inj: complex | None) -> np.ndarray:
    """
    Solve the nodal system for one sequence. The source branch is
    (e_source behind Zs) at node R; fault_inj is the current drawn
    FROM the network by the fault at node F (injected as negative).
    Returns node voltages [R, F, L] (or [R, FL] when merged).
    """

    _, _, zs = seq_params(seq)

    Y, merged = build_nodal(seq, d_km, load_z)

    Y[0, 0] += 1.0 / zs

    b = np.zeros(Y.shape[0], dtype=complex)

    b[0] += e_source / zs

    if fault_inj is not None:

        f_idx = 1 if not merged else 1  # merged: fault = node 1 (FL)

        b[f_idx] += -fault_inj

    return np.linalg.solve(Y, b)


def thevenin_at_fault(seq: int, d_km: float, load_z: complex | None) -> complex:
    """
    Driving-point Thevenin impedance at the fault node with the
    independent source killed (EMF shorted, source impedance kept).
    """

    _, _, zs = seq_params(seq)

    Y, merged = build_nodal(seq, d_km, load_z)

    Y[0, 0] += 1.0 / zs

    Yinv = np.linalg.inv(Y)

    return Yinv[1, 1]


def prefault_state(load_level: float, d_km: float = LINE_LENGTH_KM / 2.0) -> dict:
    """
    Pre-fault solve for one load level at a canonical fault position.
    The source EMF (positive sequence) is calibrated so the relay-bus
    phase-a prefault voltage is exactly V_PH_RMS angle 0.

    Sequence ordering throughout this module: [zero, positive,
    negative] — the same ordering produced by abc_to_120() and
    consumed by seq120_to_abc().
    """

    load_z = load_impedance(load_level)

    d_ref = float(d_km)

    e_unit = V_PH_RMS * (1.0 + 0j)

    v_r_unit = solve_network(1, d_ref, load_z, e_unit, None)[0]

    e1 = e_unit * (V_PH_RMS / v_r_unit)

    e_by_seq = [0j, e1, 0j]  # [zero, positive, negative]

    v_r_120 = np.zeros(3, dtype=complex)

    v_f_120 = np.zeros(3, dtype=complex)

    i_r_120 = np.zeros(3, dtype=complex)

    for seq, e_s in enumerate(e_by_seq):

        v_nodes = solve_network(
            seq, d_ref, seq_load_z(seq, load_z), e_s, None
        )

        _, _, zs = seq_params(seq)

        v_r_120[seq] = v_nodes[0]

        v_f_120[seq] = v_nodes[1]

        i_r_120[seq] = (e_s - v_nodes[0]) / zs

    return {
        "load_z": load_z,
        "e_by_seq": e_by_seq,
        "v_r_pref_120": v_r_120,
        "i_r_pref_120": i_r_120,
        "v_f_pref_120": v_f_120,
        "v_r_pref_abc": seq120_to_abc(v_r_120),
        "i_r_pref_abc": seq120_to_abc(i_r_120),
    }


def fault_admittance_seq(sig_name: str, r_f: float) -> np.ndarray:
    """
    Fault admittance matrix in the sequence domain (3x3) for one
    signature at fault resistance r_f (per phase).

    Phase-domain construction: faulted phases carry 1/r_f to the
    star point; the star point is solidly grounded for grounded
    signatures (declared bolted ground return). Phase-to-phase
    (BC) couples the two phases through 1/r_f with no ground path.
    """

    phases, grounded, _ = SIGNATURES[sig_name]

    r_eff = max(r_f, R_F_BOLTED_EPS)

    Yf_phase = np.zeros((3, 3), dtype=complex)

    if grounded:

        for p in phases:

            Yf_phase[p, p] += 1.0 / r_eff

    else:

        # ungrounded phase-to-phase: current (Vb - Vc)/r_f
        p, q = phases

        Yf_phase[p, p] += 1.0 / r_eff

        Yf_phase[p, q] += -1.0 / r_eff

        Yf_phase[q, p] += -1.0 / r_eff

        Yf_phase[q, q] += 1.0 / r_eff

    # transform to the sequence domain
    a = _A_ROT

    A = np.array([
        [1, 1, 1],
        [1, a, a * a],
        [1, a * a, a],
    ], dtype=complex) / 3.0

    return A @ Yf_phase @ np.linalg.inv(A)


def solve_fault(sig_name: str, r_f: float, d_km: float,
                load_level: float, prefault: dict) -> dict:
    """
    Full fault solve for one scenario. The prefault voltage at the
    fault node is computed on the SAME tapped network (tap at d_km)
    so the Thevenin superposition is exact for that network. Returns
    relay-bus phasors (post-fault) in the phase domain plus
    supporting sequence data. Sequence ordering: [zero, pos, neg].
    """

    load_z = load_impedance(load_level)

    # prefault voltage at the fault node, per sequence, same network
    v_pref_f = np.zeros(3, dtype=complex)

    for seq in range(3):

        v_nodes = solve_network(
            seq, d_km, seq_load_z(seq, load_z),
            prefault["e_by_seq"][seq], None,
        )

        v_pref_f[seq] = v_nodes[1]

    i_f = np.zeros(3, dtype=complex)     # [I0, I1, I2] into the fault

    if sig_name != "NF":

        Yf_seq = fault_admittance_seq(sig_name, r_f)

        z_th = np.array([
            thevenin_at_fault(
                seq, d_km, seq_load_z(seq, load_z)
            )
            for seq in range(3)
        ])

        # (I + Yf_seq @ diag(Z_th)) I_f = Yf_seq @ V_pref_F
        M = np.eye(3, dtype=complex) + Yf_seq @ np.diag(z_th)

        i_f = np.linalg.solve(M, Yf_seq @ v_pref_f)

    # network solve with the fault drawing i_f at node F
    v_r_120 = np.zeros(3, dtype=complex)

    i_r_120 = np.zeros(3, dtype=complex)

    for seq in range(3):

        e_s = prefault["e_by_seq"][seq]

        v_nodes = solve_network(
            seq, d_km, seq_load_z(seq, load_z), e_s, i_f[seq]
        )

        _, _, zs = seq_params(seq)

        v_r_120[seq] = v_nodes[0]

        i_r_120[seq] = (e_s - v_nodes[0]) / zs

    return {
        "d_km": d_km,
        "v_r_abc": seq120_to_abc(v_r_120),
        "i_r_abc": seq120_to_abc(i_r_120),
        "i_f_abc": seq120_to_abc(i_f),
        "i_f_120": i_f,
        "v_r_120": v_r_120,
    }


# ============================================================
# TIME-DOMAIN SAMPLING
# ============================================================

def dc_time_constant(r_f: float, d_km: float) -> float:
    """Positive-sequence loop L/R (seconds) for the DC decay."""

    z_loop = ZS1 + Z1_PER_KM * d_km + max(r_f, R_F_BOLTED_EPS)

    x_loop = z_loop.imag

    r_loop = z_loop.real

    return x_loop / (OMEGA * r_loop)


def sample_fault_row(state: dict, fault: dict, r_f: float,
                     theta_inc: float, k: int) -> dict[str, float]:
    """
    Instantaneous sample k (0-based) of a fault sweep: t = (k+1) ms
    after inception. Voltages per-unit (prefault peak = 1.0),
    currents in amperes.
    """

    t = (k + 1) * SAMPLE_DT

    v_pref_abc = state["v_r_pref_abc"]

    i_pref_abc = state["i_r_pref_abc"]

    v_abc = fault["v_r_abc"]

    i_abc = fault["i_r_abc"]

    tau = dc_time_constant(r_f, fault["d_km"])

    out: dict[str, float] = {}

    for p, col in enumerate(("a", "b", "c")):

        # voltage: post-fault phasor, no DC offset
        v_inst = (
            np.sqrt(2.0) * abs(v_abc[p])
            * np.sin(OMEGA * t + theta_inc + np.angle(v_abc[p]))
        )

        # current: post-fault AC plus DC offset fixed by continuity
        # of the loop current at the inception instant (t = 0)
        i_ac_0 = np.sqrt(2.0) * abs(i_abc[p]) * np.sin(
            theta_inc + np.angle(i_abc[p])
        )

        i_pref_0 = np.sqrt(2.0) * abs(i_pref_abc[p]) * np.sin(
            theta_inc + np.angle(i_pref_abc[p])
        )

        a_dc = i_pref_0 - i_ac_0

        i_inst = (
            np.sqrt(2.0) * abs(i_abc[p])
            * np.sin(OMEGA * t + theta_inc + np.angle(i_abc[p]))
            + a_dc * np.exp(-t / tau)
        )

        out[f"V{col}"] = v_inst / (np.sqrt(2.0) * V_PH_RMS)

        out[f"I{col}"] = i_inst

    return out


def sample_nofault_row(state: dict, j: int) -> dict[str, float]:
    """
    Sample j (0-based) of the continuous no-fault recording:
    t = j ms on the 1 kHz grid, steady state (no DC offset).
    """

    t = j * SAMPLE_DT

    v_abc = state["v_r_pref_abc"]

    i_abc = state["i_r_pref_abc"]

    out: dict[str, float] = {}

    for p, col in enumerate(("a", "b", "c")):

        out[f"V{col}"] = (
            np.sqrt(2.0) * abs(v_abc[p])
            * np.sin(OMEGA * t + np.angle(v_abc[p]))
        ) / (np.sqrt(2.0) * V_PH_RMS)

        out[f"I{col}"] = (
            np.sqrt(2.0) * abs(i_abc[p])
            * np.sin(OMEGA * t + np.angle(i_abc[p]))
        )

    return out


# ============================================================
# DISTANCE-RELAY MEASUREMENT CHAIN
# ============================================================

# Residual compensation factor from the LINE impedances (the relay
# compensates the protected line's zero sequence, not the source's).
K0_LINE = (Z0_PER_KM - Z1_PER_KM) / (3.0 * Z1_PER_KM)


def loop_impedances(
    v_abc: np.ndarray, i_abc: np.ndarray
) -> list[float]:
    """
    Six distance-relay loop impedances (ohms) from relay-bus phasors:

        ground loops     Z_g,p = |V_p| / |I_p + k0 * 3I0|
        phase-phase loops Z_pp = |V_p - V_q| / |I_p - I_q|

    These are steady-state phasor quantities of the fault state in
    effect at the exported sample — the measurement a numerical
    distance relay forms once its full-cycle data window lies inside
    the post-fault steady state. Raw instantaneous samples are NOT
    usable for the impedance decision: |V(t)|/|I(t)| dips below the
    Zone 1 setting near every voltage zero-crossing (see
    docs/simulation_caveats.md item 4).
    """

    sum_i = i_abc.sum()

    z_ground = [
        abs(v_abc[p]) / (abs(i_abc[p] + K0_LINE * sum_i) + 1e-9)
        for p in range(3)
    ]

    z_phase_phase = [
        abs(v_abc[p] - v_abc[q]) / (abs(i_abc[p] - i_abc[q]) + 1e-9)
        for p, q in ((0, 1), (1, 2), (2, 0))
    ]

    return z_ground + z_phase_phase


MEAS_COLUMNS = [
    "row_id", "z_ga", "z_gb", "z_gc", "z_ab", "z_bc", "z_ca",
]


# ============================================================
# DATASET ASSEMBLY
# ============================================================

def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # prefault states per load level (voltage phasors in per-unit
    # terms are produced at sampling time; here in volts)
    prefault_by_load = {
        level: prefault_state(level) for level in LOAD_LEVELS
    }

    fault_records: list[dict] = []

    meas_records: list[dict] = []

    # sweep-major ordering: load -> Rf -> inception angle ->
    # signature; within a sweep Loc ascends 15..120 (the Loc == 15
    # marker starts a new scenario for src/scenarios.py)
    for load_level in LOAD_LEVELS:

        state = prefault_by_load[load_level]

        for r_f in R_F_LEVELS:

            for theta_deg in INCEPTION_ANGLES_DEG:

                theta_inc = np.deg2rad(theta_deg)

                for sig_name in SIGNATURE_ORDER:

                    _, _, encoding = SIGNATURES[sig_name]

                    for k, d_km in enumerate(LOCATIONS_KM):

                        fault = solve_fault(
                            sig_name, r_f, float(d_km),
                            load_level, state,
                        )

                        row = sample_fault_row(
                            state, fault, r_f, theta_inc, k
                        )

                        row["Loc"] = d_km

                        row["G"], row["C"], row["B"], row["A"] = encoding

                        fault_records.append(row)

                        z_loops = loop_impedances(
                            fault["v_r_abc"], fault["i_r_abc"]
                        )

                        meas_records.append({
                            "row_id": len(fault_records) - 1,
                            "z_ga": z_loops[0], "z_gb": z_loops[1],
                            "z_gc": z_loops[2], "z_ab": z_loops[3],
                            "z_bc": z_loops[4], "z_ca": z_loops[5],
                        })

    fault_df = pd.DataFrame(
        fault_records,
        columns=["Loc", "G", "C", "B", "A",
                 "Ia", "Ib", "Ic", "Va", "Vb", "Vc"],
    )

    # no-fault stratum: ONE continuous steady-state recording at the
    # normal load level (the detection file carries all samples; the
    # classification file carries the first NOFAULT_SAMPLES_CLASS)
    nf_state = prefault_by_load[NOFAULT_LEVEL]

    nf_records = [
        sample_nofault_row(nf_state, j)
        for j in range(NOFAULT_SAMPLES_DETECT)
    ]

    nf_df = pd.DataFrame(
        nf_records,
        columns=["Ia", "Ib", "Ic", "Va", "Vb", "Vc"],
    )

    class_df = fault_df.copy()

    class_nf = nf_df.iloc[:NOFAULT_SAMPLES_CLASS].copy()

    class_nf["Loc"] = 0

    class_nf["G"] = class_nf["C"] = class_nf["B"] = class_nf["A"] = 0

    class_df = pd.concat(
        [class_df, class_nf[fault_df.columns]], ignore_index=True
    )

    # relay measurements for the no-fault rows: steady state, identical
    # for every sample (balanced prefault -> 3I0 = 0 -> loop impedances
    # equal the load-side quantities)
    nf_loops = loop_impedances(
        nf_state["v_r_pref_abc"], nf_state["i_r_pref_abc"]
    )

    for j in range(NOFAULT_SAMPLES_CLASS):

        meas_records.append({
            "row_id": len(fault_records) + j,
            "z_ga": nf_loops[0], "z_gb": nf_loops[1],
            "z_gc": nf_loops[2], "z_ab": nf_loops[3],
            "z_bc": nf_loops[4], "z_ca": nf_loops[5],
        })

    meas_df = pd.DataFrame(meas_records, columns=MEAS_COLUMNS)

    detect_df = fault_df.copy()

    detect_df = detect_df.drop(columns=["G", "C", "B", "A"])

    detect_df["Output (S)"] = 1

    detect_nf = nf_df.copy()

    detect_nf["Loc"] = 0

    detect_nf["Output (S)"] = 0

    detect_df = pd.concat(
        [detect_df, detect_nf[detect_df.columns]], ignore_index=True
    )

    return class_df, detect_df, meas_df


# ============================================================
# VERIFICATION CROSS-CHECKS (Section 3.3.1)
# ============================================================

def verification_block(logger: ReportLogger) -> None:

    logger.print("-" * 70)
    logger.print("VERIFICATION CROSS-CHECKS (Section 3.3.1)")
    logger.print("-" * 70)

    # ---- 1. three-phase bolted fault current at the source bus ----
    # Reference: S / (sqrt(3) V_LL) = 3000 MVA / (sqrt(3) * 161 kV)
    i_ref = FAULT_LEVEL_VA / (np.sqrt(3.0) * V_LL_RMS)

    state = prefault_state(LOAD_LEVELS[1])

    # d -> 0: fault at the relay bus
    fault0 = solve_fault("ABCG", 0.0, 1e-6, LOAD_LEVELS[1], state)

    i_3ph = max(abs(f) for f in fault0["i_f_abc"])

    logger.print(
        f"1. Three-phase bolted fault current, d -> 0:\n"
        f"   computed  : {i_3ph / 1e3:.2f} kA (peak of phase modules)\n"
        f"   reference : {i_ref / 1e3:.2f} kA "
        f"= 3000 MVA / (sqrt(3) x 161 kV)\n"
        f"   status    : "
        f"{'PASS' if abs(i_3ph - i_ref) / i_ref < 0.05 else 'FAIL'}"
        f" (within 5%)"
    )

    # ---- 2. line charging ------------------------------------------
    q_section = V_PH_RMS ** 2 * OMEGA * C1_PER_KM * LINE_LENGTH_KM

    route_km = 297.4

    q_route = V_PH_RMS ** 2 * OMEGA * C1_PER_KM * route_km

    logger.print(
        f"2. Line charging (three-phase, V = phase voltage):\n"
        f"   120 km modelled section : {q_section / 1e6:.2f} MVAr\n"
        f"   full 297.4 km route     : {q_route / 1e6:.2f} MVAr\n"
        f"   RECONCILIATION NOTE for Section 3.3.1: the chapter's"
        f"\n   ~30.9 MVAr figure back-solves exactly as "
        f"(161 kV)^2 * w * C * L\n   with 161 kV used as the PHASE "
        f"voltage - a sqrt(3) slip. With the\n   correct phase "
        f"voltage the full-route value is "
        f"{q_route / 1e6:.2f} MVAr; the\n   regenerated section "
        f"carries {q_section / 1e6:.2f} MVAr. The chapter text"
        f"\n   must be corrected when transcribing Section 3.3.1."
    )

    # ---- 3. apparent-impedance sanity (bolted AG sweep) -------------
    # Standard ground-distance measurement: the operating current is
    # residual-compensated, Z_app = |V_phase| / |I_phase + k0 * 3I0|,
    # k0 = (Z0L - Z1L) / (3 Z1L). For a bolted SLG fault fed from one
    # end this recovers z1*d exactly (the voltage-side compensation
    # used by src/relay.py today over-measures by z_s/z1 ~ 1.8x and is
    # corrected in Phase 2).
    logger.print(
        "3. Relay apparent impedance, bolted AG sweep, normal load\n"
        "   (standard ground loop |V_phase| / |I_phase + k0*3I0|;"
        " theoretical\n   positive-seq line impedance z1*d shown"
        " for comparison):"
    )

    k0 = (Z0_PER_KM - Z1_PER_KM) / (3.0 * Z1_PER_KM)

    zone1_ohms = 0.80 * abs(Z1_PER_KM * LINE_LENGTH_KM)

    for d_km in LOCATIONS_KM:

        fault = solve_fault("AG", 0.0, float(d_km), LOAD_LEVELS[1], state)

        v_abc = fault["v_r_abc"]

        i_abc = fault["i_r_abc"]

        sum_i = i_abc.sum()

        z_apps = [
            abs(v_abc[p]) / (abs(i_abc[p] + k0 * sum_i) + 1e-9)
            for p in range(3)
        ]

        z_app = min(z_apps)

        z_theory = abs(Z1_PER_KM * d_km)

        zone = "Zone 1" if z_app <= zone1_ohms else "beyond Zone 1"

        logger.print(
            f"   d = {d_km:>3} km : Z_app = {z_app:7.2f} ohm "
            f"(z1*d = {z_theory:7.2f} ohm) -> {zone}"
        )

    logger.print(
        f"   Zone 1 reach (80% of |z1| * 120 km): {zone1_ohms:.2f} ohm"
        f" -> faults beyond d = 96 km (105, 120 km) must NOT trip"
        f" instantaneously (Zone 2 behaviour)."
    )

    # ---- 4. distance signal sanity ----------------------------------
    logger.print(
        "4. Distance signal sanity, bolted AG sweep, normal load\n"
        "   (faulted-phase rms magnitudes at the relay bus):"
    )

    for d_km in LOCATIONS_KM:

        fault = solve_fault("AG", 0.0, float(d_km), LOAD_LEVELS[1], state)

        i_f_mag = abs(fault["i_f_abc"][0])

        v_a_mag = abs(fault["v_r_abc"][0]) / V_PH_RMS

        logger.print(
            f"   d = {d_km:>3} km : |I_fault| = {i_f_mag / 1e3:6.2f} kA, "
            f"|Va| = {v_a_mag:5.3f} pu"
        )

    logger.print("")

    # ---- 5. signature boundary conditions (sequence domain) ---------
    logger.print(
        "5. Symmetrical-component boundary conditions per signature\n"
        "   (bolted faults at 60 km, normal load; sequence currents\n"
        "   into the fault, ordering [I0, I1, I2]):"
    )

    d_check = 60.0

    for sig_name in SIGNATURE_ORDER:

        fault = solve_fault(sig_name, 0.0, d_check, LOAD_LEVELS[1], state)

        i0, i1, i2 = fault["i_f_120"]

        base = abs(i1) + 1e-9

        if sig_name == "AG":

            checks = [("I0 = I1 = I2", max(abs(i0 - i1), abs(i2 - i1)) / base)]

        elif sig_name == "BC":

            checks = [("I0 = 0", abs(i0) / base),
                      ("I1 = -I2", abs(i1 + i2) / base)]

        elif sig_name in ("BCG", "ABG"):

            # healthy phase carries no fault current (BCG: phase a,
            # ABG: phase c); express via the phase-domain transform
            healthy = 0 if sig_name == "BCG" else 2

            i_abc_f = seq120_to_abc(fault["i_f_120"])

            checks = [("healthy-phase I_f = 0",
                       abs(i_abc_f[healthy]) / base)]

        else:  # ABCG

            checks = [("I0 = 0", abs(i0) / base),
                      ("I2 = 0", abs(i2) / base)]

        worst = max(v for _, v in checks)

        label = " ; ".join(name for name, _ in checks)

        logger.print(
            f"   {sig_name:5s}: {label:20s} worst rel. dev "
            f"{worst:.2e} -> {'PASS' if worst < 1e-6 else 'FAIL'}"
        )

    logger.print("")

    # ---- 6. fault-resistance and load-level behaviour ----------------
    logger.print(
        "6. Fault-resistance sweep (AG, 60 km, normal load) and\n"
        "   load-level operating points (prefault steady state):"
    )

    for r_f in R_F_LEVELS:

        fault = solve_fault("AG", r_f, d_check, LOAD_LEVELS[1], state)

        logger.print(
            f"   R_f = {r_f:5.0f} ohm : |I_f,a| = {abs(fault['i_f_abc'][0]):8.1f} A"
        )

    for level in LOAD_LEVELS:

        st = prefault_state(level)

        i_r = abs(st["i_r_pref_abc"][1])   # positive-seq relay current

        v_nodes = solve_network(
            1, LINE_LENGTH_KM / 2.0, seq_load_z(1, st["load_z"]),
            st["e_by_seq"][1], None,
        )

        v_l_pu = abs(v_nodes[2]) / V_PH_RMS

        logger.print(
            f"   load {LOAD_LEVEL_NAMES[level]:13s}: |I_R| = {i_r:6.1f} A,"
            f" |V_load bus| = {v_l_pu:.3f} pu"
        )

    logger.print("")

    # ---- 7. DC-offset law --------------------------------------------
    logger.print(
        "7. DC offset, bolted AG at 60 km, normal load\n"
        "   (phase-a offset amplitude vs inception angle; the asymmetry\n"
        "   is largest when inception lands near the phase-a voltage\n"
        "   zero crossing, smallest near the voltage peak):"
    )

    fault = solve_fault("AG", 0.0, d_check, LOAD_LEVELS[1], state)

    tau = dc_time_constant(0.0, d_check)

    i0_col = fault["i_r_abc"][0]

    i0_pref = state["i_r_pref_abc"][0]

    a_dc_by_theta: dict[float, float] = {}

    for theta_deg in INCEPTION_ANGLES_DEG:

        theta_inc = np.deg2rad(theta_deg)

        i_ac_0 = np.sqrt(2.0) * abs(i0_col) * np.sin(
            theta_inc + np.angle(i0_col)
        )

        i_pref_0 = np.sqrt(2.0) * abs(i0_pref) * np.sin(
            theta_inc + np.angle(i0_pref)
        )

        a_dc_by_theta[theta_deg] = i_pref_0 - i_ac_0

        logger.print(
            f"   theta_inc = {theta_deg:5.1f} deg : A_dc(a) = "
            f"{a_dc_by_theta[theta_deg]:9.1f} A"
        )

    # the exported samples must follow A_dc * exp(-t/tau) exactly
    theta_inc = 0.0

    a_dc = a_dc_by_theta[0.0]

    worst = 0.0

    for k in range(SWEEP_ROWS):

        t = (k + 1) * SAMPLE_DT

        i_ac_t = np.sqrt(2.0) * abs(i0_col) * np.sin(
            OMEGA * t + np.angle(i0_col)
        )

        expected = i_ac_t + a_dc * np.exp(-t / tau)

        row = sample_fault_row(state, fault, 0.0, theta_inc, k)

        got = row["Ia"]

        worst = max(worst, abs(got - expected) / (abs(expected) + 1e-9))

    logger.print(
        f"   tau (positive-seq loop L/R) = {tau * 1e3:.1f} ms\n"
        f"   exported samples follow A_dc*exp(-t/tau): worst rel. dev "
        f"{worst:.2e} -> {'PASS' if worst < 1e-9 else 'FAIL'}"
    )

    logger.print("")

    # ---- 8. power balance (prefault steady state) ---------------------
    logger.print(
        "8. Power balance, prefault steady state, normal load\n"
        "   (three-phase complex power: source EMF = load + line\n"
        "   series consumption + shunt charging):"
    )

    d_ref = LINE_LENGTH_KM / 2.0

    e1 = state["e_by_seq"][1]

    v_r, v_f, v_l = solve_network(
        1, d_ref, seq_load_z(1, state["load_z"]), e1, None
    )

    _, _, zs = seq_params(1)

    i_r = (e1 - v_r) / zs

    s_source = 3.0 * e1 * np.conj(i_r)

    # the source impedance itself consumes 3|I|^2 Zs
    s_source_z = 3.0 * abs(i_r) ** 2 * zs

    s_load = 3.0 * v_l * np.conj(v_l / state["load_z"])

    i_rf = (v_r - v_f) / (Z1_PER_KM * d_ref)

    i_fl = (v_f - v_l) / (Z1_PER_KM * (LINE_LENGTH_KM - d_ref))

    s_series = (3.0 * (v_r - v_f) * np.conj(i_rf)
                + 3.0 * (v_f - v_l) * np.conj(i_fl))

    y_sh_km = 1j * OMEGA * C1_PER_KM

    s_shunt = 3.0 * (
        abs(v_r) ** 2 * np.conj(y_sh_km * d_ref / 2.0)
        + abs(v_f) ** 2 * np.conj(y_sh_km * LINE_LENGTH_KM / 2.0)
        + abs(v_l) ** 2 * np.conj(
            y_sh_km * (LINE_LENGTH_KM - d_ref) / 2.0)
    )

    imbalance = abs(
        s_source - (s_source_z + s_load + s_series + s_shunt)
    ) / (abs(s_source) + 1e-9)

    logger.print(
        f"   source : {s_source.real / 1e6:7.2f} MW, "
        f"{s_source.imag / 1e6:+8.2f} MVAr\n"
        f"   source impedance: {s_source_z.real / 1e6:6.3f} MW, "
        f"{s_source_z.imag / 1e6:+7.2f} MVAr\n"
        f"   load   : {s_load.real / 1e6:7.2f} MW, "
        f"{s_load.imag / 1e6:+8.2f} MVAr (0.95 pf lagging)\n"
        f"   line I2R loss : {s_series.real / 1e6:6.3f} MW\n"
        f"   line series Q  : {s_series.imag / 1e6:+7.2f} MVAr\n"
        f"   shunt charging : {s_shunt.imag / 1e6:+7.2f} MVAr\n"
        f"   complex balance: relative imbalance {imbalance:.2e} -> "
        f"{'PASS' if imbalance < 1e-9 else 'FAIL'}"
    )

    logger.print("")


# ============================================================
# MAIN
# ============================================================

def main():

    logger = ReportLogger(REPORT_FILE)

    try:
        logger.print("=" * 70)
        logger.print("R1 DATASET REGENERATION (CORRECTED TABLE 3.2)")
        logger.print("=" * 70)
        logger.print(f"Timestamp: {timestamp}")
        logger.print("")

        logger.print("-" * 70)
        logger.print("PARAMETERS")
        logger.print("-" * 70)
        logger.print(f"Nominal voltage        : 161 kV line-to-line RMS "
                     f"(phase = {V_PH_RMS:.1f} V)")
        logger.print(f"Frequency              : {F_NOM:.0f} Hz")
        logger.print(f"Modelled section       : {LINE_LENGTH_KM:.0f} km")
        logger.print(f"Z1                     : {Z1_PER_KM.real:.3f} + "
                     f"j{Z1_PER_KM.imag:.3f} ohm/km")
        logger.print(f"Z0                     : {Z0_PER_KM.real:.3f} + "
                     f"j{Z0_PER_KM.imag:.3f} ohm/km")
        logger.print(f"C1                     : {C1_PER_KM * 1e9:.2f} nF/km")
        logger.print(f"C0 (assumed 0.75*C1)   : {C0_PER_KM * 1e9:.2f} nF/km")
        logger.print(f"Fault level            : {FAULT_LEVEL_VA / 1e6:.0f} MVA "
                     f"-> |Zs| = {ZS_MAG:.4f} ohm")
        logger.print(f"Source impedance       : {ZS_R:.4f} + j{ZS_X:.4f} ohm "
                     f"(X/R = {SOURCE_XR:.0f}, declared)")
        logger.print(f"Load                   : constant impedance at "
                     f"{LOAD_PF:.2f} pf lagging (declared)")
        logger.print(f"Load levels            : "
                     + ", ".join(
                         f"{LOAD_LEVEL_NAMES[k]} = {k * THERMAL_RATING_A:.0f} A"
                         for k in LOAD_LEVELS))
        logger.print(f"Fault resistance       : {R_F_LEVELS} ohm per phase "
                     f"(0 implemented as {R_F_BOLTED_EPS * 1e3:.0f} mOhm)")
        logger.print(f"Inception angles       : {INCEPTION_ANGLES_DEG} deg")
        logger.print(f"Locations              : {LOCATIONS_KM} km")
        logger.print(f"Signatures             : {', '.join(SIGNATURE_ORDER)}")
        logger.print(f"Sampling               : {1 / SAMPLE_DT:.0f} Hz grid; "
                     f"fault sweep row k sampled at t = (k+1) ms")
        logger.print(f"Units                  : V per-unit instantaneous "
                     f"(prefault peak = 1.0), I amperes")
        logger.print(f"No-fault stratum       : continuous recording at "
                     f"{LOAD_LEVEL_NAMES[NOFAULT_LEVEL]} load "
                     f"({NOFAULT_SAMPLES_DETECT} samples; class file "
                     f"carries first {NOFAULT_SAMPLES_CLASS})")
        logger.print("")

        logger.print("-" * 70)
        logger.print("VERIFICATION CROSS-CHECKS")
        logger.print("-" * 70)
        verification_block(logger)

        logger.print("-" * 70)
        logger.print("GENERATING DATASETS")
        logger.print("-" * 70)

        class_df, detect_df, meas_df = build_datasets()

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        class_df.to_csv(CLASS_FILE, index=False)

        detect_df.to_csv(DETECT_FILE, index=False)

        meas_df.to_csv(RELAY_MEASUREMENTS_FILE, index=False)

        # ---- structure report ---------------------------------------
        logger.print(
            f"classdetectfault_sim.csv : {len(class_df)} rows "
            f"({(class_df['Loc'] > 0).sum()} fault rows = "
            f"{(class_df['Loc'] > 0).sum() // SWEEP_ROWS} sweeps, "
            f"{(class_df['Loc'] == 0).sum()} no-fault rows)"
        )
        logger.print(
            f"detectfault_sim.csv      : {len(detect_df)} rows "
            f"({(detect_df['Output (S)'] == 1).sum()} fault / "
            f"{(detect_df['Output (S)'] == 0).sum()} no-fault)"
        )
        logger.print("")

        sig_counts = class_df[class_df["Loc"] > 0].groupby(
            ["G", "C", "B", "A"]
        ).size()
        logger.print("Fault rows per signature (G,C,B,A):")
        for sig, n in sig_counts.items():
            logger.print(f"  {sig}: {n}")
        logger.print("")

        logger.print("No-fault vs fault feature ranges (separation margin):")

        nf_rows = class_df[class_df["Loc"] == 0]

        fault_rows = class_df[class_df["Loc"] > 0]

        for col in ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]:

            logger.print(
                f"  {col}: no-fault [{nf_rows[col].min():9.2f}, "
                f"{nf_rows[col].max():9.2f}], "
                f"fault [{fault_rows[col].min():9.2f}, "
                f"{fault_rows[col].max():9.2f}]"
            )

        logger.print("")

        logger.print("=" * 70)
        logger.print("OUTPUT FILES")
        logger.print("=" * 70)
        logger.print(f"Report           : {REPORT_FILE}")
        logger.print(f"Classification   : {CLASS_FILE}")
        logger.print(f"Detection        : {DETECT_FILE}")
        logger.print(f"Relay measurements: {RELAY_MEASUREMENTS_FILE}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
