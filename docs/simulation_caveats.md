# Simulation-pass notes: caveats for the reviewer (R1 regeneration)

**Scope:** findings from the 2026-11-12 Python re-simulation pass
(`src/simulate.py`) that a reviewer of Chapters Three and Four must
watch for when the documentation is transcribed. Every item below is
traceable to `outputs/simulate/report_2026-11-12_16-09-04.txt` (all
verification checks PASS) and to the generated datasets
`data/classdetectfault_sim.csv`, `data/detectfault_sim.csv`.

---

## 1. Design-space arithmetic correction (spec slip, now fixed)

The earlier plan text said "5 types × 8 Loc × 5 R_f × 5 angles ×
3 loads = 3,000 sweeps = 24,000 fault rows". That double-counts Loc:
Loc is the *within-sweep* dimension (8 rows, 15→120 km), not a
scenario dimension. The full factorial over the locked levels is:

- **375 sweeps** (5 signatures × 5 R_f × 5 inception angles × 3 load
  levels) × 8 locations = **3,000 fault rows** (600 per signature);
- plus the no-fault stratum: 2,365 rows (classification file) /
  6,503 rows (detection file).

Do **not** transcribe "24,000 fault rows" anywhere.

## 2. Class balance shift in the detection dataset

The no-fault stratum was mirrored from the original size (6,503
samples) while the fault stratum follows the factorial (3,000 rows):

- detection dataset: **31.6% fault rows vs 45.8% in the original**;
- classification dataset: 3,000 fault + 2,365 no-fault = 5,365 rows
  (vs 7,861 originally).

Chapter Four must state the new counts and, if any confusion-matrix
or per-class metric is compared against the old run, the different
prior must be called out.

## 3. Section 3.3.1 line-charging figure is wrong by √3

The chapter's ~30.9 MVAr back-solves exactly as
(161 kV)²·ω·C·L with **161 kV used as the PHASE voltage** — a √3
slip. Correct values (12.74 nF/km, phase voltage 92.95 kV):

- 120 km modelled section: **4.15 MVAr**;
- full 297.4 km route: **10.28 MVAr**.

The chapter text must be corrected when transcribing §3.3.1.

## 4. Relay baseline: measurement chain and formula decisions

Two distinct problems were found and fixed in `src/relay.py`:

**(a) Apparent-impedance formula.** `relay.py` compensated on the
**voltage** side: `Z_app = |V_a + k0·ΣI| / |I_a|`. For a bolted
single-line-to-ground fault fed from one end this measures
`z_s·d + k0` with `z_s = (Z0 + 2·Z1)/3`, i.e. it **over-measures by
z_s/z1 ≈ 1.81×** (0.129 + j0.711 Ω/km vs Z1 = 0.080 + j0.390 Ω/km) —
verified numerically in this pass.

The standard ground-distance loop compensates the **operating
current**:

```
Z_g,p = |V_p| / |I_p + k0 · 3I0|,   k0 = (Z0L − Z1L)/(3·Z1L)
```

which recovers `Z1·d` exactly for a bolted SLG fault (checked: all
eight locations within 0.5 Ω of z1·d). Phase-phase loops use
`Z_pp = |V_p − V_q| / |I_p − I_q|`. The relay decision takes the
minimum over all six loops (3 ground + 3 phase-phase).

**(b) Raw-sample impedance decisions are invalid.** The first
re-run of the relay on the regenerated data tripped on **474 no-fault
rows**: evaluating `|V(t)|/|I(t)|` on raw instantaneous samples dips
below the Zone-1 setting near every voltage zero-crossing, so the
decision becomes an artifact of the sampling phase, not of protection
physics. A real numerical relay decides on full-cycle
Fourier-filtered phasors. However, the event-sweep dataset never
provides a full post-fault data window for any row (8 samples at
different fault locations), so a literal window emulation would trip
on almost nothing.

**Resolution (document this in §3.9.4):** `src/simulate.py` exports,
per dataset row, the six loop impedances of the fault state in effect
at that sample as steady-state phasor quantities
(`outputs/simulate/relay_measurements_sim.csv`, aligned by `row_id`);
`relay.py` applies the Zone-1 decision on those measurements. The
semantics: a distance relay that has seen the fault for at least one
full data window. The resulting baseline profile is physically
sensible: **TN = 100% (no false trips), TP = 49.2%** — per sweep,
bolted faults trip for Loc ≤ 90 km, R_f = 10 Ω similar, R_f = 25 Ω
≈ 4/8 locations, R_f = 50 Ω ≈ 2/8, R_f = 100 Ω ≈ 0. The chapter must
describe the relay as **secure but incomplete** (high-resistance and
far faults are outside Zone 1) rather than quoting a raw-sample
accuracy.

Also fixed: nominal voltage 400 kV → **161 kV**; Zone 1 reach
37.44 Ω → **38.22 Ω** (= 0.80 × |Z1| × 120 km using the complex
|Z1| = 0.398 Ω/km magnitude, not the reactance alone).

## 5. Units changed relative to the original datasets

- Voltages: per-unit instantaneous, **prefault phase peak = 1.0**
  (original data had a ~0.62 prefault peak and a physically
  impossible ~4%-of-nominal faulted-phase voltage at every distance).
- Currents: instantaneous **amperes** (original NF currents were
  ~44 A median — an artefact of the 12.74 F/km capacitance error).
- NF current peaks are now ≈ 657 A; fault peaks reach ≈ 4.6 kA.

Any chapter sentence quoting absolute waveform values from the old
data is superseded.

## 6. Within-sweep representation (mirrors the original, but document it)

Row k of a fault sweep is the instantaneous sample of the
post-fault waveform at **t = (k+1) ms** after inception; the fault
*location* changes from row to row (the original data model). So a
sweep mixes the Loc ramp with post-fault time evolution (including
the DC-offset decay, τ ≈ 18 ms for bolted close-in faults). This is
the representation the scenario machinery (8-row sweeps keyed on
Loc == 15) is built on; it must be described as an **event-sweep
model, not a continuous time series**, unlike the no-fault stratum
which IS one continuous 1 kHz recording (lag-1 autocorrelation
0.9510 vs 0.9511 theoretical for 1 kHz / 50 Hz).

## 7. Declared auxiliary modelling assumptions (Table 3.2 silent)

Each must appear in the chapter's parameter table or the simulation
is not reproducible:

| Assumption | Value |
|---|---|
| Source X/R | 10 |
| Load model | constant impedance, 0.95 pf lagging, 40/75/100% of 750 A |
| Zero-seq capacitance | C0 = 0.75·C1 |
| Source zero-seq impedance | = Z1 source (solidly grounded) |
| Negative sequence | Z2_line = Z1_line, Z2_source = Z1_source (transposed) |
| Ground return | bolted; R_f = per-phase fault-path resistance; for SLG R_f is the total phase-to-ground path |
| R_f = 0 | implemented as 1 mΩ |
| Prefault calibration | source EMF scaled per load level so relay-bus prefault = 1.0 pu ∠0° |
| No-fault stratum | recorded at normal (75%) load only |

## 8. Operating point: load-bus voltage sag is significant

Prefault load-bus voltages are 0.931 / 0.862 / 0.815 pu at 40 / 75 /
100% load (a consequence of the corrected 0.390 Ω/km reactance with
a constant-impedance load). The chapter should state these operating
points; "1.0 pu everywhere" would be wrong. The relay bus is held at
1.0 pu by construction (EMF calibration).

## 9. Sequence-ordering and dataset-structure conventions

- Sequence vectors are ordered **[zero, positive, negative]**
  throughout `simulate.py` (matches `abc_to_120` / `seq120_to_abc`).
- Files keep the original layout: fault rows first, sweep-major,
  Loc ascending 15→120 within each sweep, then the no-fault block
  (`Loc = 0`). `scenarios.py` compatibility verified: 375 clean
  sweeps + 296 no-fault windows = **671 scenarios** for the
  group-aware splits and K = 7 CV.
- The detection file's fault rows are identical to the
  classification file's fault rows (mirrors the original pairing).

## 10. Verification lessons (why the battery matters)

Two of the eight verification checks initially FAILED — both were
errors in the *check formulas*, not the simulation, and both are
instructive:

1. **ABG boundary condition:** the healthy phase is c, so the
   constraint is `I_c = I0 + a·I1 + a²·I2 = 0`, not `I1+I2+I0 = 0`
   (that is phase a's current — faulted in ABG). Writing the SLG-only
   relation for a LLG signature silently passes on BCG and fails on
   ABG.
2. **Power balance:** the source impedance consumes
   `3·|I|²·Zs = 0.568 MW + j5.68 MVAr`; omitting it leaves a ~4%
   complex imbalance that looks like a network bug but is an
   accounting omission.

Final battery: (1) 10.76 kA three-phase reference (11.01 kA, PASS);
(2) line charging √3 reconciliation; (3) Z_app vs z1·d (≤0.5 Ω error
at all 8 locations); (4) distance signal (|I_f| 4.76→0.89 kA, |Va|
sag 0.568→0.928 pu across 15→120 km — physically consistent where
the original data was flat); (5) sequence boundary conditions ≤1e-11
relative; (6) R_f monotonicity + load operating points; (7) DC-offset
law (peak ±2,279 A at voltage-zero inception, −69 A near voltage
peak, τ = 18.0 ms, exported samples follow A_dc·e^(−t/τ) exactly);
(8) prefault complex power balance to 6.6e-15. The regeneration is
bit-identical across runs (md5-verified).

## 11. Downstream numbers are ALL superseded

Every accuracy, latency, robustness, hypothesis-test p-value, CV
statistic, and table produced from the original data is superseded by
the Phase 2 cascade re-run on the `_sim` files. Known stale items to
watch in the chapter text:

- the old "LSTM latency gap ~270 ms" (measured ~691 ms on the current
  split; will change again);
- TODO numbers quoting model accuracies against the original data;
- the detection dataset row counts and any derived percentages.

Noise handling is unchanged: nothing is baked into the waveforms;
the robustness machinery applies the SNR sweep synthetically exactly
as before.
