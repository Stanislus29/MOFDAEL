# Phase 3 draft — replacement text for Chapters Three and Four

**Era:** regenerated-data pass of 2026-11-12 (`src/simulate.py` →
`data/classdetectfault_sim.csv`, `data/detectfault_sim.csv` → full
cascade re-run). Every number below is traceable to a re-runnable
report as required by the traceability rule (AGENTS.md §5).

**Source reports (all 2026-11-12):**

| Tag | Report |
|---|---|
| [SIM] | `outputs/simulate/report_2026-11-12_16-09-04.txt` (8-point verification battery, all PASS) |
| [REL] | `outputs/relay/report_2026-11-12_16-20-30.txt` |
| [HYP] | `outputs/hypothesis_tests/report_2026-11-12_17-08-01.txt` |
| [T43] | `outputs/table_4_3/report_2026-11-12_19-43-50.txt` |
| [COMB] | `outputs/combination_accuracy/report_2026-11-12_17-25-01.txt` |
| [LEAK] | `outputs/leakage_audit/report_2026-11-12_17-25-01.txt` |
| [CV] | `outputs/scenario_cv/report_2026-11-12_17-25-03.txt` |
| [CAV] | `docs/simulation_caveats.md` (reviewer caveats; this draft stays consistent with it) |

Text blocks marked **[REPLACE]** substitute existing passages;
**[INSERT]** blocks are new; **[DELETE-FLAG]** lists superseded
sentences that must not survive transcription.

---

# PART A — CHAPTER THREE

## A1. §3.3.1 — line parameters and derived quantities [REPLACE]

Keep the corrected Table 3.2 parameters, but present the impedances
as complex quantities and fix the line-charging figure:

> The protected line is a 161 kV (line-line), 50 Hz, 120 km
> transmission section with positive-sequence impedance
> Z₁ = 0.080 + j0.390 Ω/km and zero-sequence impedance
> Z₀ = 0.228 + j1.352 Ω/km. The line's shunt capacitance is
> C₁ = 12.74 nF/km (C₀ = 0.75·C₁), giving a total charging
> reactive power of 4.15 MVAr for the 120 km modelled section and
> 10.28 MVAr for the full 297.4 km transmission route.

**[DELETE-FLAG]** the "~30.9 MVAr" sentence. It back-solves exactly
as (161 kV)²·ω·C·L, i.e. 161 kV was used as the *phase* voltage — a
√3 slip. The correct values (per-phase voltage 92.95 kV) are 4.15 and
10.28 MVAr [SIM §2; CAV item 3]. Do not print the slip explanation in
the thesis; just correct the figure.

Also [REPLACE] the fault-level cross-check sentence:

> A bolted three-phase fault at the source bus produces a symmetrical
> fault current of approximately 10.8 kA (11.01 kA in the implemented
> simulation), consistent with the 3,000 MVA fault level.

## A2. §3.3 — auxiliary modelling assumptions [INSERT — new table]

Table 3.2 is silent on the following, without which the simulation is
not reproducible. Insert as a short table or a bulleted paragraph:

> The simulation additionally adopts the following declared
> assumptions: (i) the grid source is modelled behind a fault-level
> impedance with X/R ratio 10; (ii) the load is a constant-impedance
> load at 0.95 power factor (lagging) drawing 40%, 75%, or 100% of
> the 750 A rated load current; (iii) zero-sequence line capacitance
> is 0.75 of the positive-sequence value; (iv) the source
> zero-sequence impedance equals its positive-sequence value
> (solidly grounded source) and negative-sequence impedances equal
> the positive-sequence ones (fully transposed line); (v) the
> ground-return path is treated as bolted, and the fault resistance
> R_f is the per-phase fault-path resistance (for a
> single-line-to-ground fault it is the total phase-to-ground path
> resistance); (vi) the R_f = 0 case is implemented as 1 mΩ; and
> (vii) the source EMF magnitude is calibrated at each load level so
> that the relay-bus prefault voltage is exactly 1.0 pu ∠0°.

**Operating points to state (one sentence):** with the corrected
0.390 Ω/km reactance and the constant-impedance load, the prefault
load-bus voltage is 0.931 / 0.862 / 0.815 pu at 40% / 75% / 100%
load; the relay bus is held at 1.0 pu by the EMF calibration
[SIM §6; CAV item 8]. Writing "1.0 pu everywhere" would be wrong.

## A3. §3.5.5 (dataset generation) — design space and structure [REPLACE]

> The fault design space is a full factorial over five fault
> signatures (AG, BC, BCG, ABG, ABCG), five fault-resistance levels
> (R_f ∈ {0, 10, 25, 50, 100} Ω, with R_f = 0 implemented as 1 mΩ),
> five fault-inception angles (0°, 45°, 90°, 135°, 180°), and three
> load levels (40%, 75%, 100% of rated load), giving 375 fault
> events. Each event is recorded as a sweep across the eight line
> locations (Loc = 15, 30, …, 120 km), so the fault stratum contains
> 375 × 8 = 3,000 rows (600 per signature). Waveforms are sampled on
> a 1 kHz grid; row k of a sweep stores the instantaneous
> six-phase sample taken at t = (k+1) ms after fault inception.
>
> The no-fault stratum is a continuous steady-state recording at the
> normal (75%) load level: 2,365 samples in the classification
> dataset and 6,503 in the detection dataset. The classification
> dataset therefore contains 5,365 rows and the detection dataset
> 9,503 rows; the detection dataset is 31.6% fault rows. The fault
> rows of both files are identical.

**[DELETE-FLAG]** every count from the original-data era: "12,000
scenarios", "14,400 total rows", "24,000 fault rows" (the design-space
double-count; the true factorial is 375 sweeps — [CAV item 1]),
"7,861 rows", "45.8% fault rows", "5,496 fault rows / 685 sweeps +
4 truncated sweeps".

**One honest caveat sentence to include (event-sweep model):**

> Note that a fault sweep mixes the location ramp with the post-fault
> time evolution of the waveform (row k is sampled k+1 ms after
> inception); it is an event-sweep representation, not a continuous
> time series of a single fault — unlike the no-fault stratum, which
> is one continuous recording (lag-1 autocorrelation 0.951,
> consistent with the 1 kHz / 50 Hz sampling grid).

## A4. §3.8.2 (unit of analysis) and §3.12.1 (CV) [REPLACE]

> The unit of analysis is the simulation run: each 8-row fault sweep
> is one scenario (375 fault scenarios) and the no-fault recording is
> segmented into 296 non-overlapping 8-row windows (the last has 5
> rows), giving 671 scenarios in total. All eight models share one
> deterministic scenario-level partition (no scenario spans the
> train/test boundary), so every model is evaluated on the same
> 134 test scenarios (1,069 rows). The K = 7 stratified group
> cross-validation reported in Section 4.x is descriptive evidence
> computed with the final fixed hyperparameters; it was not used for
> hyperparameter selection.

[Source: src/scenarios.py docstring; LEAK; CV reconciliation note.]

## A5. §3.9.4 (conventional protection baseline) — measurement chain [REPLACE]

This is the most important Chapter Three correction. The relay
baseline is a numerical distance relay evaluated on the simulated
relay-bus quantities. Replace the detection-logic passage with:

> The relay measures the six standard distance-loop impedances from
> the relay-bus phasor quantities: three residual-compensated ground
> loops,
>
>     Z_g,p = |V_p| / |I_p + k0 · 3I0|,   k0 = (Z₀L − Z₁L)/(3·Z₁L),
>
> and three phase-phase loops,
>
>     Z_pp,pq = |V_p − V_q| / |I_p − I_q|.
>
> The compensation factor is complex, k0 = 0.8140 + j0.0405, and the
> Zone 1 reach is 80% of the positive-sequence line impedance
> magnitude, 0.80 × |Z₁| × 120 km = 38.22 Ω. The relay trips when the
> minimum loop impedance falls inside the Zone 1 reach.
>
> The loop impedances are evaluated as steady-state phasor
> quantities of the fault state in effect at each exported sample —
> the measurement a numerical relay forms once its full-cycle data
> window lies inside the post-fault steady state. Raw instantaneous
> samples are deliberately not used for the impedance decision:
> |V(t)|/|I(t)| evaluated on raw samples dips below the Zone 1
> setting near every voltage zero-crossing, which would make the
> trip decision an artifact of the sampling phase rather than of
> protection physics.

**[DELETE-FLAG]** the old 400 kV nominal-voltage sentence, the
voltage-side compensation formula |V + k0·3I0|/|I|, and the
37.44 Ω reach.

**Expected profile sentence (supports the Chapter Four narrative):**

> The resulting baseline is secure but incomplete: it never trips on
> a no-fault sample (true-negative rate 100%) and trips on 49.2% of
> fault rows. Bolted faults are detected up to 90 km (Z₁·d ≤ 36.0 Ω
> < 38.22 Ω), while R_f = 100 Ω faults fall outside Zone 1 at every
> location; faults at 105 and 120 km are beyond the reach even when
> bolted. [REL; CAV item 4]

## A6. §3.6/§3.7 — feature enumeration (R4) [INSERT/REPLACE]

> Each observation is a six-element instantaneous sample of the three
> phase currents (Ia, Ib, Ic, in amperes) and three phase voltages
> (Va, Vb, Vc, in per-unit of the prefault phase peak). No derived
> features (wavelets, phasors, sequence components) are computed; the
> models consume the same six raw channels, which is what makes the
> model-versus-relay comparison a like-for-like contrast of decision
> logic on identical information. The fault-location regression head
> consumes the same six channels with Loc (km) as the target.

**[DELETE-FLAG]** any sentence quoting absolute waveform values from
the original data (e.g. "~44 A no-fault currents", "~4% of nominal
faulted-phase voltage", "prefault peak ≈ 0.62"). In the regenerated
data the units are: voltages per-unit with prefault peak = 1.0;
currents in amperes with ≈657 A no-fault peaks and ≈4.6 kA fault
peaks [CAV item 5; SIM §4].

---

# PART B — CHAPTER FOUR

## B1. §4.2 — dataset presentation [REPLACE]

> The regenerated datasets comprise 5,365 rows for classification
> (3,000 fault rows in 375 eight-row sweeps + 2,365 no-fault rows)
> and 9,503 rows for fault detection (9,503 = 3,000 + 6,503), with
> the no-fault stratum recorded at the normal load level. The
> scenario-level split assigns 134 of the 671 scenarios (1,069 rows)
> to the test partition, shared by all eight models; the relay is
> evaluated both on those rows and on the full 5,365-row dataset.

## B2. Table 4.3 (R6 + R7) — full replacement [REPLACE table]

Values are accuracy / precision / recall / F1, ± bootstrap SD
(2,000 resamples, seed 42), on the 1,069 common test rows. Reproduce
the per-target table from [T43] §"TABLE 4.3 (REGENERATED)"; the
overall rows are:

| Model | Accuracy | Recall | F1 | Latency (ms) |
|---|---|---|---|---|
| ann | 0.9944 ± 0.0024 | 0.9900 ± 0.0041 | 0.9950 ± 0.0020 | 0.001073 ± 0.000232 |
| cnn | 0.9981 ± 0.0014 | 0.9967 ± 0.0023 | 0.9983 ± 0.0012 | 0.055665 ± 0.007032 |
| dtr | 0.9972 ± 0.0016 | 0.9950 ± 0.0029 | 0.9975 ± 0.0015 | 0.000699 ± 0.000077 |
| knn | 0.9963 ± 0.0019 | 0.9933 ± 0.0034 | 0.9967 ± 0.0017 | 0.014761 ± 0.002658 |
| lstm | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 204.905730 ± 103.846783 |
| pinn | 0.9972 ± 0.0016 | 0.9950 ± 0.0028 | 0.9975 ± 0.0014 | 0.003759 ± 0.000538 |
| rnn | 0.9935 ± 0.0024 | 0.9883 ± 0.0044 | 0.9941 ± 0.0023 | 0.000766 ± 0.000044 |
| svm | 0.9991 ± 0.0009 | 0.9983 ± 0.0016 | 0.9992 ± 0.0009 | 0.001067 ± 0.000165 |
| relay (common rows) | 0.7568 ± 0.0130 | 0.5667 ± 0.0201 | 0.7234 ± 0.0165 | 10.0 (reference) |
| relay (all 5,365 rows) | 0.7158 ± 0.0063 | 0.4917 ± 0.0091 | 0.6592 ± 0.0082 | 10.0 (reference) |

Text notes to carry into §4.x:

- The weakest per-target cell in the table is rnn on phase A
  (accuracy 0.9280 ± 0.0078, F1 0.9164 ± 0.0096); lstm attains 1.0
  on every target. [T43]
- All eight per-target reconstructions pass; no withheld cells
  remain after the R6 regeneration. [T43]

## B3. §4.x — H₁ (detection vs relay) [REPLACE]

> Against the conventional Zone-1 distance-relay baseline, all eight
> learned models are superior by 23.7 to 24.3 percentage points on
> the 1,069 common test rows (relay overall accuracy 0.7568): ann
> +23.76 [21.16, 26.36], cnn +24.13, dtr +24.04, knn +23.95,
> lstm +24.32, pinn +24.04, rnn +23.67, svm +24.23 pp (95% CIs in
> the report). Every pairwise McNemar test is significant
> (p < 10⁻⁶, Holm-adjusted p < 10⁻⁶), and Cochran's Q across the
> nine methods is Q = 1906.85 (df = 8, p < 10⁻⁶). The gap is a
> direct consequence of the relay's protection logic: as a Zone-1
> element it is secure (no false trips) but misses every
> high-resistance fault and every fault beyond 96 km, whereas the
> learned models saturate detection. [HYP H1; REL]

## B4. §4.x — H₂ (per-line classification) [REPLACE]

> Per-target Cochran's Q is significant for all four lines (G:
> Q = 40.04, p = 1.0×10⁻⁶; C: Q = 60.13; B: Q = 37.68, p = 3×10⁻⁶;
> A: Q = 167.49; all df = 7). Under Holm-corrected exact/continuity-
> corrected pairwise tests, the lstm is the only method significantly
> better than the field on G, C and B (gaps of 1.5–2.8 pp); on phase
> A the field separates more sharply: lstm beats all seven
> competitors (up to 7.2 pp over rnn) and cnn beats five (up to
> 6.27 pp over rnn), with pinn's advantages over svm/knn not
> surviving Holm correction (adjusted p = 0.062 / 0.085). The
> practical reading is that fault-type classification is nearly
> saturated for all models except on phase A, where the recurrent
> architectures differ measurably. [HYP H2]

## B5. §4.x — H₃ (reporting latency) [REPLACE]

> Taking the relay's 10.0 ms reference reporting latency, every model
> reports faster by essentially the full reference margin
> (Hodges–Lehmann ≈ −9.95 to −10.0 ms, all p < 10⁻⁶, Holm-adjusted)
> — dtr 0.0007 ms, rnn 0.0008 ms, ann/svm ≈ 0.001 ms, pinn 0.0038 ms,
> knn 0.0148 ms, cnn 0.0557 ms mean per-row latency — with one
> exception: the lstm reports ≈ 188.8 ms slower than the relay
> (HL = +188.75 ms, 95% CI [178.5, 197.0]; mean 204.9 ± 103.8 ms),
> and ≈ 198.8 ms slower than every other model. The lstm's latency
> therefore dominates its otherwise perfect accuracy profile and
> must be discussed as its principal deployment drawback.
> [HYP H3; T43]

**[DELETE-FLAG]** the old "~270 ms" LSTM gap sentence — the current
measured gap is ≈ 189 ms vs the relay (mean latency 204.9 ms).

## B6. §4.x — H₄ (noise robustness, TOST equivalence) [REPLACE + margin justification]

> Equivalence to the 1%-noise baseline was tested with two one-sided
> tests (TOST) at a pre-declared margin of 5.0 accuracy points. The
> margin is justified as the largest accuracy loss that would not
> change the operational conclusion about a model's deployability:
> it is roughly one-fifth of the 23.7–24.3 pp relay-versus-model
> gap, and it is far above the resolution of the pairwise
> comparisons (which detect differences as small as ≈ 1 pp at
> n = 1,069), so a model inside the margin is for practical purposes
> unaffected by the injected noise.
>
> Under this margin, the lstm is equivalent to its noise-free
> baseline at every noise level on every target (largest observed
> degradation 0.22 pp at 20% noise). The knn is equivalent up to
> 10% noise on all four targets (largest 10%-noise difference
> 3.7 pp). The remaining models are equivalent at 5% noise on most
> targets but break at 10% (differences 5.2–12.7 pp) and all break
> at 20% noise (differences 10.3–36.1 pp). Two cells are not
> equivalent even at 5% noise: cnn on phase A (6.9 pp) and pinn on
> phase A (4.7 pp, p_TOST = 0.072). The ranking that emerges is
> lstm ≫ knn > {ann, dtr, rnn, svm, cnn, pinn}, with phase A again
> the most noise-sensitive line. [HYP H4]

## B7. §4.x — H₅ (fault-location accuracy) [REPLACE]

> Fault-location accuracy is measured on the 600 fault scenarios of
> the common test set as MAE in km (percentage of the 120 km line
> length). The lstm is dominant at 0.55 km (0.46%), followed by the
> dtr at 2.68 km (2.23%); the knn attains 6.22 km (5.18%) and the
> remaining models degrade to 11.6–19.1 km (9.7–15.9%), in the order
> cnn 9.68%, ann 13.54%, svm 13.61%, rnn 15.23%, pinn 15.88%.
> Friedman's test across all locations is decisive
> (χ² = 2338.9, df = 7, p < 10⁻⁶, Kendall's W = 0.557), and the
> per-location families are significant at every location
> (W = 0.47–0.74). Post-hoc Holm tests show the dtr within
> 0.31–0.63 km of the lstm at every one of the eight locations
> (all adjusted p < 10⁻⁶ in the lstm's favour), while every other
> model trails the lstm by 3.3–30.9 km depending on location. The
> two best location estimators are therefore the sequence model
> (lstm) and the regression tree (dtr), and the physically driven
> distance signal of the regenerated data (fault current
> 4.76 → 0.89 kA, faulted-phase voltage sag 0.568 → 0.928 pu across
> 15 → 120 km) is what makes this ordering possible — the original
> data contained no distance-dependent physics at all. [HYP H5; SIM]

## B8. §4.x — per-combination accuracy [REPLACE]

> Exact fault-type-vector accuracy by combination count shows the
> lstm perfect on every multi-line combination (double, triple and
> quadruple faults, all 1.000). Among the remaining models the
> quadruple-line (ABCG) class is the hardest: knn 73.3%, dtr 75.8%,
> pinn 80.0%, ann 82.5%, svm 84.2%, cnn 85.8%, rnn 85.0%. The relay
> is included only for structural completeness: its output is a
> binary trip decision, so it cannot match a type vector (0% on
> double- and triple-line combinations; the 46.7% ABCG figure
> reflects trips, not classification). [COMB]

## B9. §4.x — scenario-level cross-validation (R9) [REPLACE]

> The K = 7 stratified group CV over all 671 scenarios (zero
> scenarios spanning folds) is reported as descriptive evidence with
> the final fixed hyperparameters. Scenario-level detection accuracy
> lies between 0.9914 ± 0.0032 (rnn) and 0.9978 ± 0.0018 (cnn);
> exact four-target match accuracy lies between 0.8990 ± 0.0135
> (rnn) and 0.9631 ± 0.0117 (cnn). The gap between detection and
> exact-match accuracy quantifies the residual type-confusion on
> hard signatures. [CV]

## B10. §4.x — leakage audit [INSERT one sentence]

> A scenario-level leakage audit confirms the integrity of the
> evaluation: no scenario appears in more than one fold or spans a
> train/test boundary for any method (models touch 134 of the 671
> test-partition scenarios; the relay, which trains on nothing, is
> evaluated on all 671), and no duplicated scenario content exists
> between partitions. [LEAK]

---

# PART C — superseded numbers: do NOT transcribe

1. "12,000 scenarios / 14,400 total rows / 2,400 held-out scenarios /
   7,861 rows" — original-data narrative. New: 5,365 / 9,503 rows;
   671 scenarios; 134 test scenarios; 1,069 test rows. [T43
   reconciliation section states these discrepancies explicitly;
   resolve them in the chapter text as above.]
2. "~30.9 MVAr line charging" — √3 slip. New: 4.15 MVAr (section) /
   10.28 MVAr (route). [SIM]
3. "24,000 fault rows" — design-space double-count. New: 3,000 fault
   rows from 375 sweeps. [CAV item 1]
4. "LSTM latency gap ≈ 270 ms" — superseded: ≈ 189 ms vs the relay
   (HL +188.75, CI [178.5, 197.0]); mean per-row latency 204.9 ms.
   [HYP H3; T43]
5. Any model accuracy, latency, robustness, p-value, CV statistic or
   table cell computed from the original datasets — all superseded
   by the 2026-11-12 cascade reports listed at the top of this file.
6. Old relay narrative ("0.9+ detection accuracy") — superseded by
   the secure-but-incomplete profile: TN 100%, TP 49.2%, overall
   0.7158 (all rows) / 0.7568 (common rows). [REL; T43]
7. Absolute waveform claims from the old data (44 A no-fault
   currents, 4%-of-nominal faulted-phase voltage, 0.62 prefault
   peak, 12.74 F/km artefacts). [CAV item 5]

## Chapter Five note

The conclusions chapter must be re-checked against the new profile:
the headline claims that survive are (i) learned models ≫ Zone-1
relay on detection (≈ 24 pp, p < 10⁻⁶), (ii) the lstm is the best
detector and by far the best locator (MAE 0.46%) but the slowest
reporter (≈ 189 ms above the relay reference), (iii) the dtr is the
fastest high-fidelity locator (2.23% MAE at 0.0007 ms), and
(iv) noise robustness beyond 10% noise degrades every model except
the lstm materially.
