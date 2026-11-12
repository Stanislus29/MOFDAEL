# To-Do: Mr Fiase Dissertation Code Base Corrections

Companion to `corrections/Dissertation_Ch3_Ch4_Corrected.docx`,
`corrections/Pending_Tasks_Recommendation.docx`, and
`corrections/Declaring-Pending_Items.docx`.

Items R1–R10 below are marked ⟦PENDING R#⟧ in the corrected dissertation
working copy. All fourteen PENDING markers must be resolved or converted
into declared limitations before submission.

**Prerequisite question (answered: yes).** A per-scenario results file
exists: every `outputs/<model>/` directory contains
`predictions_<model>.csv` (one row per held-out scenario, columns include
`row_id`, `<target>_true`, `<target>_pred` for G/C/B/A),
`latency_per_row_<model>.csv`, and `robustness_<model>.csv` (per-trial
columns `accuracy_trial_0` … `accuracy_trial_9`).  
→ R2, R3, R5, R6, R7, R9 all close from this one file (~2 weeks).
→ If the file did not exist, models would need re-running with
per-scenario logging first (+3–4 weeks). Not applicable here.

---

## BLOCKING — must fix, cannot declare as limitation

### R2 — Verify the relay prediction vector and re-run H₁ — COMPLETED
**Where:** `src/hypothesis.py` → `run_h1()`; new `src/relay.py`; new `outputs/relay/predictions_relay.csv` and `outputs/relay/report_<timestamp>.txt`

**Bug fixed:** The relay baseline compared ground truth to itself
(`relay_true == relay_true`), producing a vector of all 1s — every
scenario "correct", every McNemar discordant cell empty, odds ratios of
exactly 0.0000, and Cochran's Q p = 0.4653 against a 36–40 point gap.

**Code changes made:**
1. **New `src/relay.py`** — impedance-based distance relay using corrected
   Table 3.2 parameters (X₁ = 0.390 Ω/km, X₀ = 1.352 Ω/km, line length
   120 km, Zone 1 reach 80% = 37.44 Ω). Zero-sequence compensation
   k₀ = 0.8222. Compensated per-phase impedance `Z = \|V + k₀·ΣI\| / \|I\|`
   with nominal 400 kV phase-voltage scaling. Trip when min phase
   impedance < Zone 1 reach. Follows train-script standard:
   `PROJECT_ROOT`/`DATA_DIR`/`OUTPUT_DIR` path config, `ReportLogger`,
   timestamped `report_<timestamp>.txt` alongside `predictions_relay.csv`.
2. **New `outputs/relay/predictions_relay.csv`** — 7,861 per-scenario
   relay predictions in the same format as model predictions
   (`row_id`, `overall_true`, `overall_pred`, per-target columns).
3. **New `outputs/relay/report_<timestamp>.txt`** — line parameters
   (Table 3.2), detection thresholds, accuracy metrics (overall,
   no-fault TN, fault TP), confusion matrix (TN/FP/FN/TP), output file
   paths.
4. **Edited `src/hypothesis.py`** — removed the tautology block
   (lines 409–419); relay now loads as a regular prediction file via
   `load_predictions()`; excluded relay from pairwise self-comparison;
   added Holm step-down correction across the family of McNemar
   comparisons (new `p_value_holm` field in output records).

**Results (339 common test rows):**
| Metric | Value |
|--------|-------|
| Cochran's Q | Q = 389.04, df = 7, p < 0.001 |
| Relay accuracy | 0.8201 |
| Model accuracies | 0.9941 – 1.0000 |
| Accuracy gap (models − relay) | 17.40 – 17.99 pp |
| Odds ratio | ∞ (relay never corrects a model error) |
| Holm-adjusted p | < 0.001 for all 7 comparisons |

H₁ is now supported: all learned models significantly outperform the
relay, and results survive Holm multiplicity correction.

**⚠️ Outstanding discrepancy:** Relay accuracy is **0.82**, not **0.6000**
as reported in Table 4.3. Expected — implemented from scratch without
the original Simulink model. Investigate alongside R1 (re-simulation
with corrected parameters) and R6 (Table 4.3 regeneration). The relay
logic is now physically grounded rather than fabricated.

**Remaining sub-items:**
- Document relay characteristic and phasor filter (§3.9.2 / §3.9.6 — R2a/R2b, writing task)
- Replace Table 4.11 with confirmatory result (writing task, results now available)

---

### R3 — Diagnose the fault location predictor — REPAIR COMPLETED (Path A), writing caveat remains
**Where:** Every `src/*.py` training script; `src/hypothesis.py` → `run_h5()`

**The bug:** `Loc` is explicitly dropped in every model script
(`logger.print("\nIGNORED METADATA:\n  Loc")`), so there is no fault
location predictor anywhere in the codebase. `run_h5()` therefore tests
per-line *classification* error grouped by `Loc`, not location error —
which is why all 63 Hodges–Lehmann estimates are exactly `0.0000` with
CIs `[0.0000, 0.0000]` (models' error vectors are near-identical, not
equivalent).

**Repair executed (retrain with sibling location heads, decision taken
against the two fallback options):**

1. **Diagnosis evidence:** `src/diagnose_h5.py` →
   `outputs/h5_diagnostics/report_2026-10-27_17-48-04.txt` establishes
   that no model exported location predictions, that `run_h5()` tested
   the wrong quantity, that location IS learnable (DTR feasibility:
   MAE 6.16 km / 5.14%), and that a reactance-method relay estimate is
   impossible from single-sample data (|Va/Ia| vs Loc correlation
   0.0056).
2. **Sibling head added to all 8 training scripts:** each script now
   also trains one location regressor of its own estimator family,
   on fault rows only (Loc > 0), trained outside the whole test
   split and evaluated on the fault rows of the same split; exports
   `Loc_true` / `Loc_pred` (km, NaN on no-fault rows) appended to
   `predictions_<model>.csv`, plus a LOCATION ACCURACY section
   (MAE/RMSE in km and % of the 120 km line, MAE per Table 3.3
   location). Detection columns unchanged for the torch models;
   TF-based models (cnn, lstm, pinn) show ~1 pp run-to-run wobble —
   per decision 2026-10-28, numbers as provided by the new runs are
   reported; explanations in the write-up.
3. **Head configurations (documented in each report):**
   - knn: KNeighborsRegressor (k=5, distance, Pipeline w/ scaler)
   - dtr: DecisionTreeRegressor defaults (unbounded depth;
     the classifier's max_depth=10 degraded MAE to 29.2 km)
   - svm: SVR (family-matched kernel/C/gamma; ~31 km MAE
     insensitive to C/gamma — honest architecture limitation)
   - ann/rnn: subclassed wrappers, linear output, MSELoss on
     scaled target (km/120)
   - cnn: same Conv1D trunk, linear Dense head, MSE, early stopping
   - lstm: time-distributed linear head over the Loc-sweep
     sequence (build_batch_loc / create_lstm_location; fault
     events only; LOCATION ACCURACY CAVEAT printed in report)
   - pinn: LocationPINN (FaultPINN subclass, linear output),
     MSE + same physics residual weight, shuffled before the
     tail val split
4. **run_h5() repaired:** requires ≥2 models with Loc columns and
   fails loudly otherwise; tests absolute location error per row
   (Friedman omnibus pooled and per Table 3.3 location; Wilcoxon
   post-hoc with Holm; percentage over the 120 km line, stated);
   per-location slicing bug fixed 2026-10-28 (statistics now unique
   per location).

**New results (all fault scenarios pooled, n = 250 common rows):**
| Model | MAE (km) | MAE (% of 120 km) |
|-------|----------|-------------------|
| knn   | 4.80     | 4.00% |
| dtr   | 5.46     | 4.55% |
| lstm  | 0.48 (position-prior caveat) | 0.40% |
| ann   | 29.93    | 24.94% |
| cnn   | 29.91    | 24.92% |
| pinn  | 29.92    | 24.94% |
| rnn   | 29.87    | 24.89% |
| svm   | 30.30    | 25.25% |

Friedman chi2 = 1077.23, df = 7, p < 0.001, Kendall's W = 0.6156;
the ~30 km cluster is statistically indistinguishable; LSTM, knn,
dtr separate from it and from each other (all Holm p < 0.001 except
knn vs lstm = 0.0012).

**Interpretation notes for the write-up (MUST be stated):**
- The four plain-net heads (ann/cnn/pinn/rnn) and svr regress toward
  the mean location; on 6 instantaneous samples the distance signal
  is weak for margin-style estimators. Instance/tree methods exploit
  it directly.
- LSTM's near-zero MAE partly reflects the dataset representation:
  fault events sweep Loc 15→120 km along the timestep axis, so the
  position prior pins the distance. Not comparable to per-row models
  without further design work (caveat printed in its report).
- SRV/SVM head config was not tuned to the test set; C/gamma
  sensitivity checked (all ≈31 km).

**Bonus fix:** `svm.py` wrote outputs to
`outputs/svm/training reports/` (subdirectory), so hypothesis.py's
`outputs/*/predictions_*.csv` glob never found SVM — it had been
silently excluded from every H1–H4 comparison. Path config fixed to
`outputs/svm/`; inputs moved up; H1/H2/H4 now include SVM.

**Remaining R3 sub-item (writing):** acknowledge the TF run-noise
(~1 pp detection flag flips between re-runs of cnn/lstm/pinn) and
the LSTM position-prior limitation in the methods/limitations text.

---

## SERIOUS — examiner will require correction if unresolved

### R1 — Re-simulate under corrected line parameters (2–3 weeks incl. re-training)
**Where:** Outside this Python repo (Simulink/MATLAB model, Table 3.2 in Chapter Three §3.3.1).

**STATUS: SIMULATION + CASCADE RE-RUN COMPLETE (2026-11-12, Phases 1–2).**
Re-simulation was done in Python (`src/simulate.py`), not Simulink:
an analytical time-domain simulation of the corrected 161 kV line
(source behind fault-level impedance, nominal-pi line segments split
at the fault point, constant-impedance load, symmetrical-component
fault solve, DC-offset continuity sampling). Outputs:

- `data/classdetectfault_sim.csv` — 5,365 rows (3,000 fault rows =
  375 eight-row location sweeps + 2,365 no-fault rows);
- `data/detectfault_sim.csv` — 9,503 rows (same 3,000 fault rows +
  6,503 no-fault rows);
- `outputs/simulate/report_2026-11-12_16-09-04.txt` — parameters,
  declared auxiliary assumptions, and an 8-point verification
  battery (all PASS; the regeneration is bit-identical across runs,
  verified by md5).

**Design-space correction (spec arithmetic):** the earlier plan text
said "5 types × 8 Loc × 5 R_f × 5 angles × 3 loads = 3,000 sweeps =
24,000 fault rows". That double-counts Loc: Loc is the *within-sweep*
dimension (8 rows), not a scenario dimension. The full factorial over
the locked levels is 5 sigs × 5 R_f × 5 angles × 3 loads = **375
sweeps = 3,000 fault rows** (600 rows per signature). Downstream
scenario counts: 375 fault scenarios + 296 no-fault windows = 671
scenarios for the group-aware splits and K = 7 CV. Consequence to
declare in the chapter: the detection dataset is 31.6% fault rows vs
45.8% in the original (no-fault stratum mirrored from the original
size while the fault stratum follows the factorial).

**Verification battery (report section, all PASS):** (1) three-phase
bolted fault current at the source bus 11.01 kA vs the 10.76 kA
reference (within 5%); (2) line charging 4.15 MVAr (120 km section)
and 10.28 MVAr (full 297.4 km route) — the chapter's ~30.9 MVAr
figure back-solves as (161 kV)²·ωC·L, i.e. 161 kV used as the PHASE
voltage (a √3 slip) — §3.3.1 text must be corrected when
transcribing; (3) relay apparent impedance (standard ground loop,
|V|/|I_a + k0·3I0|) matches z1·d within 0.5 Ω at every location,
Zone 1 reach 38.22 Ω correctly excludes 105/120 km faults; (4)
distance signal |I_fault| 4.76 → 0.89 kA and |Va| sag 0.568 → 0.928
pu across 15 → 120 km; (5) symmetrical-component boundary conditions
per signature (AG: I0=I1=I2; BC: I0=0, I1=−I2; BCG/ABG: healthy-phase
fault current = 0; ABCG: I0=I2=0) at ≤1e-11 relative; (6) R_f sweep
monotone (1,707 → 710 A over 0 → 100 Ω) and load operating points
(0.931/0.862/0.815 pu load-bus voltage at 40/75/100% load); (7) DC
offset vs inception angle peaks at the voltage-zero crossings
(±2,279 A) and vanishes near the voltage peak (−69 A), τ = 18.0 ms,
exported samples follow A_dc·e^(−t/τ) exactly; (8) prefault complex
power balance closes to 6.6e-15 (source = source-Z + load + line
series + charging). Faulted-phase quantities are physically
consistent (the original data was flat in Loc).

**Phase 2 (cascade, COMPLETE 2026-11-12):** data files switched to
the `_sim` files in `relay.py`, the 8 training scripts,
`hypothesis.py`, `scenario_cv.py`, `leakage_audit.py`, `table_4_3.py`
(and `inference_svm.py`, `diagnose_h5.py`);
`src/scenarios.py` structure note updated (375 clean sweeps, lag-1
0.951). Relay fixes applied and re-run: 161 kV, Zone 1 reach
38.22 Ω, standard current-side residual compensation, and a
**measurement-chain redesign** — the relay now decides on the six
loop impedances (3 ground + 3 phase-phase) exported per row by
`src/simulate.py` (`outputs/simulate/relay_measurements_sim.csv`),
because raw-sample impedance decisions are invalid on correct
waveform physics (zero-crossing artifact; see
`docs/simulation_caveats.md` item 4).

Cascade results (all steps OK, reports 2026-11-12 16:21–17:25):

- Relay baseline (`outputs/relay/report_2026-11-12_16-20-30.txt`):
  TN 100%, TP 49.2%, overall 0.7158 on all 5,365 rows — secure but
  incomplete (high-resistance and far faults outside Zone 1).
- H₁–H₅ (`outputs/hypothesis_tests/report_2026-11-12_17-08-01.txt`):
  H₁ all 8 models beat the relay by 23.67–24.32 pp on the 1,069
  common test rows (relay 0.7568 there), McNemar p < 1e-6 all,
  Cochran's Q = 1906.85 (df 8, p ≈ 0). H₂ per-target Q significant
  (G 40.04, C 60.13, B 37.68, A 167.49); post-hoc Holm: lstm
  significantly best on G/C/B (1.5–2.8 pp), lstm and cnn best on A
  (lstm up to 7.2 pp over rnn, cnn up to 6.27 pp over rnn). H₃ all
  models report ≈10 ms faster than the relay (HL ≈ −10 ms) except
  lstm ≈ 188.75 ms slower [178.5, 197.0]. H₄ TOST at 5 pp: lstm
  equivalent at all noise levels; knn equivalent up to 10% on all
  targets; most models break at 20%; cnn/A and pinn/A not
  equivalent even at 5%. H₅ MAE (% of 120 km): lstm 0.456, dtr
  2.229, knn 5.183, cnn 9.679, ann 13.539, svm 13.614, rnn 15.230,
  pinn 15.884; Friedman all-locations χ² = 2338.9, W = 0.5569.
- Table 4.3 (`outputs/table_4_3/report_2026-11-12_17-24-42.txt`):
  regenerated with bootstrap SDs (2,000, seed 42); withheld-cell
  replacements regenerated (relay overall 0.7568 ± 0.0130 on the
  common test rows / 0.7158 ± 0.0063 on all rows; svm 0.9991 ±
  0.0009; dtr per-target F1s). All screens pass, no withheld cells.
- combination_accuracy (`..._17-25-01.txt`): lstm 100% on all
  multi-line combinations; weakest overall = dtr/rnn/knn on
  quadruple-line faults.
- leakage_audit (`..._17-25-01.txt`): clean — 0 spanning scenarios
  for all 9 methods (models touch 134/671 test scenarios).
- scenario_cv (`..._17-25-03.txt`): K = 7 scenario-grouped CV,
  descriptive, fixed hyperparameters; detection 0.9914 (rnn) to
  0.9978 (cnn); exact-match 0.8990 (rnn) to 0.9631 (cnn).

Chapter-writing implications are consolidated in
`docs/simulation_caveats.md` and the Phase 3 draft in
`corrections/`.

**Original text (pre-Phase-1):**
**The bug:** Four parameters wrong, two by orders of magnitude:
| Parameter | Was | Now | Error |
|-----------|-----|-----|-------|
| X₁ positive-sequence reactance | 0.003 Ω/km | 0.390 Ω/km | 130× |
| X₀ zero-sequence reactance | 0.006 Ω/km | 1.352 Ω/km | 225× |
| Capacitance | 12.74 F/km | 12.74 nF/km | 10⁹× |
| Fault level | 3000 kVA | 3000 MVA | 10³× |

**Code impact:** Fault currents, relay impedance trajectory, and
apparent-impedance location estimates all derive from these. Correcting
the table does *not* correct results computed from erroneous values —
only re-simulation does. R1 invalidates downstream Python work if done
late; start it in parallel with Week 1 debugging.

**Verification cross-checks already in §3.3.1:** line charging ≈ 30.9 MVAr;
three-phase fault current ≈ 10.76 kA.

**Relative ordering should survive** (all models saw identical data);
absolute accuracies, decision latencies, and the relay baseline will
shift. If time does not permit re-simulation, declare in §3.15 that
results were generated under superseded parameters and argue the
ordering-preservation case (weaker but honest).

---

### R5 — Test H₃ against the relay — CODE REPAIR COMPLETED, interpretation task remains
**Where:** `src/hypothesis.py` → `run_h3()`

**The bug:** All twelve latency comparisons are model-against-model.
The relay — the comparator named in the hypothesis — appears in none of
them; H₃ is untested.

**Code changes made (2026-10-28):**
1. **Relay reference latency:** new constant `RELAY_DECISION_LATENCY_MS
   = 10.0` in the CONFIGURATION section (the constant per-scenario
   decision cycle from Table 4.3 / the simulation logs — a fixed
   reference, not a measured random variable; documented in the report
   header). Aligned to the common test rows inside `run_h3()`.
2. **Primary family: relay vs each of the 8 models** — paired Wilcoxon
   (d = model − relay; positive = model slower), Holm step-down
   correction across the 8 comparisons (`p_value_holm` field in the
   results CSV), Hodges–Lehmann median difference with bootstrap CI,
   and rank-biserial r printed for every pair.
3. **Supplementary table retained:** the model-versus-model
   cross-family comparisons stay in the report and results CSV under
   `test = wilcoxon_paired_supplementary` (Table 4.15), explicitly
   labelled as not part of the primary H3 family; rank-biserial r now
   printed for these as well.

**Results (339 common latency rows, relay = 10.0 ms constant):**
| Pair | HL (ms) | direction | Holm-adj p | rank-biserial r |
|------|---------|-----------|------------|-----------------|
| relay vs ann | −9.999 | model faster | < 0.001 | −1.000 |
| relay vs cnn | −9.939 | model faster | < 0.001 | −1.000 |
| relay vs dtr | −9.999 | model faster | < 0.001 | −1.000 |
| relay vs knn | −9.986 | model faster | < 0.001 | −1.000 |
| relay vs lstm | **+270.00** | **model slower** | < 0.001 | +1.000 |
| relay vs pinn | −9.996 | model faster | < 0.001 | −1.000 |
| relay vs rnn | −9.999 | model faster | < 0.001 | −1.000 |
| relay vs svm | −9.999 | model faster | < 0.001 | −1.000 |

**Interpretation note (differs from the review's expectation):** the
review expected support for ANN/DT/RNN/SVM/PINN and refutation for
k-NN/CNN/LSTM. Measured per-row latencies show every model except
LSTM reporting faster than the 10 ms relay cycle (k-NN ≈ 0.014 ms,
CNN ≈ 0.06 ms — both well under); LSTM's streaming per-event latency
(~280 ms) is the only refutation. Rank-biserial r = ±1.000 reflects
that every row falls on the same side of the constant threshold, not
a degenerate test. The write-up should present the measured ordering
and explain why the expected split does not materialise (the 10 ms
relay cycle dwarfs single-row model inference on this hardware; LSTM
is the exception because its latency is measured per streamed event,
not per row).

**Optional (not done):** add fault locations at 85%/90%/95% of line
length to capture Zone 1 vs Zone 2 clearing behaviour — depends on
R1 re-simulation (new scenarios must be generated in Simulink first).

---

## EXPECTED — reporting standards not met; credibility damage if left

### R6 — Regenerate Table 4.3 from source (½ day given the results file)
**Where:** Generated tables from `outputs/*/predictions_*.csv` and
`outputs/*/metrics.csv`; Chapter Four §4.3.1.

**STATUS: CODE COMPLETE (2026-10-28).** Table regenerated by
`src/table_4_3.py` → `outputs/table_4_3/report_2026-10-28_11-08-00.txt`
+ `table_4_3.csv`. Reusable §3.8.4 screens in `src/screens.py`.

**What was done:**
1. Every cell computed directly from `predictions_*.csv`
   (50 rows: 8 models × G/C/B/A/overall on their own test splits +
   relay on common-339 primary and all-7,861 secondary).
2. §3.8.4 screens applied as reusable code (`src/screens.py`):
   between-model variance (all targets < 99% agreement, ok — max
   0.9882 on B), discordance census (each model vs relay, overall:
   all ORs finite, no empty cells), marginal reconciliation vs the
   latest `metrics.csv` row per target (all 40 model×target cells
   pass at 0.5 pp tolerance), plus F1 ≤ (P+R)/2 bound check (all
   pass). **No cells withheld.**
3. The three originally-withheld cells replaced by computed values
   (printed in the report's R6 section, none hand-corrected):
   - Relay accuracy 0.0589 → **0.8201 ± 0.0202** (common-339) /
     **0.8172 ± 0.0043** (all 7,861 rows)
   - SVM accuracy 0.09671 → **0.9981 ± 0.0011** (own split, n=1,573)
   - DTR F1 no decimal → G 0.8739, C 0.9934, B 0.9905, A 0.9966
4. Data-vs-narrative reconciliation reported in the generator output
   (not silently corrected): classdetectfault 7,861 rows vs 12,000/
   14,400 claimed; test split 1,573 rows vs 2,400 claimed; common
   test rows 339. **Correction to earlier note:** `detectfault_data.csv`
   has 11,999 data rows (12,000 lines incl. header) and **does** have
   a `Loc` column (5,496 fault rows Loc>0) — the "no Loc" claim was
   wrong. This actually supports the R3 location task on the
   detection dataset too.

**Writing task remaining:** transcribe regenerated cells + footnotes
into the dissertation Table 4.3 (relay row = common-339 with the
all-7,861 value footnoted; n column stated per row; data-vs-narrative
discrepancies reconciled in chapter text).

---

### R7 — Add dispersion to every estimate (½ day)
**Where:** Performance-table generator drawing on `outputs/*/robustness_*.csv` / `predictions_*.csv`; Chapter Four §4.3.1.

**STATUS: CODE COMPLETE (2026-10-28), together with R6.** Every cell
of the regenerated Table 4.3 now carries a dispersion measure.

**What was done:** The pipeline has a single holdout split (no folds —
fold-based dispersion is R9 scope), so dispersion is implemented as:
1. **Classification cells** (Acc/P/R/F1): bootstrap SD over test rows
   (2,000 resamples, seed 42) via `bootstrap_metric_sd()` in
   `src/screens.py` — printed as `value ± SD` in every cell.
2. **Robustness column**: mean ± SD across the 10 sweep trials at the
   reference noise level (1%) from `robustness_*.csv`.
3. **Latency column**: mean ± SD across per-row latencies from
   `latency_per_row_*.csv`; relay = constant 10.0 ms (Table 4.3
   config constant per R5, ± 0 by definition).

**Writing task remaining:** state in §3.8.7/§4.3.1 that dispersion for
holdout estimates is bootstrap-based (single split) and will become
across-fold SD once R9 introduces the scenario-level CV protocol.

---

### R9 — Evidence the scenario-level split (1 day)
**Where:** All 8 `src/*.py` training scripts (`train_test_split` call);
`src/combination_accuracy.py`, `src/hypothesis.py`; Chapter Three §3.13 /
Chapter Four §4.3.3.

**STATUS: CODE COMPLETE + RETRAINED (2026-10-30).** All numbers in the
results chain are now scenario-level and regenerate end-to-end.

**Scenario structure discovered (the missing scenario identifier):**
- Fault rows come in **8-row location sweeps** (Loc = 15..120 km):
  one sweep = one simulation run → **689 fault scenarios**
  (685 complete + 4 truncated at block boundaries).
- The 2,365 no-fault rows are **one continuous steady-state
  recording** (lag-1 autocorr = 1.0, no phase discontinuities,
  ~6.5 fundamental periods) → segmented into **296 non-overlapping
  8-row observation windows** (same window size as a fault sweep).
- Total: **985 scenarios**. Implemented in `src/scenarios.py`
  (`assign_scenario_ids`), deterministic from row order.

**Leakage audit (`src/leakage_audit.py` → `outputs/leakage_audit/`):**
- Before (report_2026-10-28_11-13-09): under the row-level split,
  **814 of 815 touched scenarios (99.9%) spanned train/test** in every
  row-level model — trains on 7 locations of a run, tested on the 8th.
  LSTM 61% (no-fault singletons scattered).
- After (latest report): all 9 methods evaluate on the **same 195 test
  scenarios / 1,554 rows, 0 spanning**.

**New shared split (all 8 training scripts use it):**
- Fault scenarios: **stratified group selection** (per-signature seeded
  20% draw; §3.12.1 stratification + §3.8.2 grouping). Verified 80/20
  within every fault signature.
- No-fault windows: **contiguous temporal blocking** (tail 20% of the
  recording → test) — random scattering would place adjacent samples
  of one continuous signal on opposite sides. Residual boundary
  autocorrelation acknowledged as a limitation.
- Deterministic and identical across scripts → the old 339-row
  common-set problem disappears: **common test rows = 1,554**.
- Each script prints the R9 split-composition block
  (`log_split_info`).

**Retrained results (de-inflated, all on the shared split):**
- Detection (overall) survives: knn 0.9987, ann 0.9981, dtr 0.9961,
  rnn/pinn 0.9955, svm/cnn 0.9949, lstm (see its report) — the
  leakage-protection challenge to >95% accuracies is answerable.
- Per-phase identification degrades: **G is hardest (0.82–0.88)**,
  C/B/A ≥ 0.97 for all models.
- Location MAE: knn 8.82 km, dtr 7.47 km (was 4.80/6.08 under the
  leaky split); the ~30 km cluster (svm/ann/rnn/cnn/pinn) unchanged.
- LSTM: events map 1:1 onto scenarios; same 1,554 test rows.
  G 0.862, C 0.974, B 0.977, A 1.000; Loc MAE 0.49 km (position-prior
  caveat intact).

**K = 7 CV reconciliation (`src/scenario_cv.py` →
`outputs/scenario_cv/`):** StratifiedGroupKFold(7) with fold
composition + spanning verification + descriptive per-fold overall
accuracy for 7 models (builders imported from the canonical scripts;
LSTM excluded with stated reason — its scenario compliance is
established by the shared partition). **The CV is descriptive with
fixed hyperparameters, NOT a hyperparameter search** — the chapter
text must say so. **All 7 models' folds completed 2026-11-12**
(report_2026-11-12_11-40-02.txt + cv_results CSV; the earlier runs had
stopped mid-PINN). PINN: mean detection 0.9944 ± 0.0042, exact-match
0.8092 ± 0.0214 — in line with the other trained nets.

**Tables 4.8/4.9 (`src/combination_accuracy.py`):** now evaluated on
the explicit common 1,554-row set; per-cell n printed (no longer
uniform-by-unexplained); **SVM included** (the earlier exclusion was
an artifact of the SVM output-path bug fixed under R2's bonus fix).

**Writing tasks remaining:**
1. Transcribe new numbers into Ch. 4 (Tables 4.3, 4.8, 4.9; §4.4.1
   hypothesis results; §3.13 protocol description).
2. Amend §3.12.1: state that the K = 7 grouped CV was descriptive;
   hyperparameters were fixed a priori; final evaluation on the
   scenario-level holdout.
3. State the no-fault-recording windowing decision and its
   autocorrelation limitation in §3.8.2/§3.13.
4. Explain the old n = 2,235 uniformity as superseded (new tables
   carry per-cell n).

---

### Cross-cutting: Holm correction (§3.8.5 mandated)
**Where:** `src/hypothesis.py`, `src/evaluator.py`.

**The bug:** `evaluator.py` applies Holm to H₂ fold-F1 post-hoc, but
`hypothesis.py` uses Bonferroni for H₅ and no family correction for H₁.
The corrected Chapter requires Holm everywhere.

**STATUS: CODE COMPLETE (2026-11-12).** All families now use Holm
step-down in `src/hypothesis.py`, each carrying a `p_value_holm` field:
- H₁: relay-vs-model McNemar family (8 comparisons) — done earlier.
- H₂: per-target cross-family McNemar families (16 comparisons per
  target, corrected within each family; omnibus Cochran's Q outside
  the family) — added 2026-11-12, verified in
  `outputs/hypothesis_tests/report_2026-11-12_11-50-38.txt`.
- H₃: relay-vs-model Wilcoxon family (8 comparisons) — done 2026-10-28.
- H₅: per-location Wilcoxon post-hoc families — done earlier.

---

## PRESENTATIONAL — documentation gaps; noted but survivable

### R4 — Enumerate the feature set (½ day to document; re-sim if wavelets wanted)
**Where:** `src/*.py` `FEATURE_COLUMNS`; Chapter Three §3.5.5.

**STATUS: RESOLVED BY DECISION (2026-11-06) — option (a), no code change.**

The codebase computes exactly six features: the instantaneous phase
signals `Ia, Ib, Ic, Va, Vb, Vc` (`FEATURE_COLUMNS` in every training
script; `Loc` is a regression target, never a feature). There are no
RMS, peak, variance, or spectral features anywhere in the pipeline,
and the wavelet claim was already withdrawn (1 kHz sampling → 500 Hz
Nyquist limit).

**Resolution:** amend §3.5.5 to list the six raw signals as the
complete, exhaustive feature set, with the design rationale: the
signals are sampled at 1 kHz, so any derived window statistic (RMS,
peak, variance) or spectral/wavelet feature would either be computed
from a single instantaneous sample (degenerate) or require multi-sample
windows that the one-row-per-measurement data model does not provide.
State explicitly that wavelet/spectral features were considered and
rejected because the Nyquist limit (500 Hz) cannot support them at the
simulation's sampling rate; note that adopting them would require
re-simulation at 5–20 kHz (decision deferred with R1).

**If the examiners' correction expects engineered features instead**
(option b): that means implementing RMS/peak/variance per window in
code AND retraining all 8 models on the new feature set — a full
cascade (all R9-era numbers change). Only take this path if R1
re-simulation happens and produces windowed recordings.

**Original bug (for the record):** `FEATURE_COLUMNS` is just the six
instantaneous phase signals. No RMS, peak magnitude, signal variance,
or spectral components are actually computed — and the wavelet claim
was withdrawn (1 kHz sampling → 500 Hz Nyquist limit cannot support
wavelets).

---

### R8 — Regenerate the TOST numerical output (1 day)
**Where:** `src/evaluator.py` → `evaluate_h4_tost()` / `evaluate_h4()`;
`src/hypothesis.py` → `tost_paired()` / `run_h4()`; Chapter Four §4.4.4.

**STATUS: CODE COMPLETE + RE-RUN COMPLETE (2026-11-12).** The queued
`hypothesis.py` re-run has been executed:
`outputs/hypothesis_tests/report_2026-11-12_11-50-38.txt`. The H4 section
now prints, for all 96 model × target × noise cells, the paired diff +
90% CI + p_lower + p_upper + p_TOST + verdict at the unified 0.05 margin.
Results are discriminating: DTR is non-equivalent at every noise level;
equivalence breaks at 20% noise for ann (C/B/A), cnn (C/B), pinn (C),
rnn (C/A), svm (C/B); equivalence holds broadly at 5% noise.

**Fixed:**
1. Zero-variance guard in BOTH implementations: identical trial
   vectors now yield a degenerate-but-principled verdict (common
   difference strictly inside ±margin → equivalent, p = 0; at/beyond
   a margin → not equivalent, p = 1). The `1e-10` hack and the
   unguarded division are gone.
2. Margin unified at **0.05** everywhere: `hypothesis.py`
   `TOST_MARGIN = 0.05` (unchanged), `evaluator.py`
   `evaluate_h4_tost` default, `evaluate_h4` default and its call
   site all now 0.05, matching the corrected chapter's ±5.0 pp.
3. `tost_paired()` / `evaluate_h4_tost()` now return the 90% CI on
   the paired difference (`ci90_low` / `ci90_high`, t-based, and
   degenerate CI = point value in the zero-variance case);
   `run_h4()` prints diff + 90% CI + p_lower + p_upper + p_TOST +
   verdict for every cell; `evaluate_h4()` report lines carry the
   same fields. Equivalence iff the 90% CI lies wholly inside
   (−margin, +margin).
4. Verified by direct test: zero-variance inside/beyond margin and
   a normal-variance case all behave correctly.

**Original bugs (for the record):**
1. `evaluate_h4_tost()` computed `se = np.std(diffs, ddof=1) / np.sqrt(n)`
   with no zero-variance guard → division by zero when trials are
   identical. (`tost_paired()` in `hypothesis.py` used a `1e-10` hack;
   neither was clean.)
2. Margin inconsistency: `evaluator.py` default margin `0.02`,
   `hypothesis.py` `TOST_MARGIN = 0.05`, corrected Chapter declares
   `±5.0 pp`. Unified on `0.05`.
3. `evaluate_h4_tost()` returned only verdicts; corrected text requires
   both one-sided probabilities and the 90% CI on each paired difference.
   (The verdicts themselves were sound — this was a reporting fix.)
verdicts themselves appear sound — this is a reporting fix.

---

### R10 — Expand the waveform discussion (2 days of writing)
**Where:** Chapter Four §4.2.2; reconcile `0.017 s` relay response vs
`10.000 ms` decision latency in Table 4.3.

**Non-coding:** Twelve figures currently share two sentences. Each fault
class needs a paragraph covering pre-fault steady state, inception
transient and decay, post-fault steady state, sequence-component
signature, and which features the models exploit. Pure writing.

---

## Non-coding (no Python changes)

| Item | What | Where |
|------|------|-------|
| R1 (if not re-simulated) | Declare superseded parameters in §3.15 with corrected derivation | Writing |
| R5 fallback | Declare H₃ as untested against relay | Writing (§3.15) |
| R3 fallback | Declare fault location out of scope (withdraw objective) | Writing (§3.15, possible title change) |
| Chapter 5 rewrite | Rewrite against corrected findings; comparative not absolute claims | Writing |
| Chapter 3/4 editorial sweep | §3.9, §3.10, §3.8.4 new sections; table renumbering 4.1→4.15; section continuity 4.4–4.7; figure captions; future→past tense; duplicate numbering | Already applied in corrected working copy where possible |
| Limitations declarations | Draft wording for R1/R5/R3 in §3.15; viva prep | `corrections/Declaring-Pending_Items.docx` |
| PENDING marker sweep | Resolve or convert all 14 `⟦PENDING R#⟧` markers | Dissertation file |

---

## Recommended sequencing (from Pending Tasks Recommendation)

**Week 1** — Locate per-scenario results file (done, it exists).
- R2: verify relay column; re-run H₁
- R3: diagnose location predictor

**Week 2**
- R5: H₃ vs relay
- R6: regenerate Table 4.3
- R7: add dispersion
- R9: evidence scenario split
- R4: enumerate features
- R8: regenerate TOST output

**Weeks 3–5** (parallel from Week 1)
- R1: correct Simulink parameters; regenerate dataset (14,400 scenarios,
  separate no-fault stratum); re-train; re-run all analyses; report
  before/after comparison

**Week 6**
- R10: waveform discussion
- Rewrite Chapter 5 against corrected findings
- Final consistency pass; PENDING marker sweep

**Total:** ~6 weeks given the results file exists. The second outcome
row ("Fix R2, R6, R9 only") is under three days of combined work and
changes the dissertation category from "major revisions / referral" to
"defensible."
