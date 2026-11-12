<div align="center">

<img src="https://img.shields.io/badge/FiaseFaultML-161%20kV%20Line%20Protection%20Study-0B3D91?style=for-the-badge&logo=python&logoColor=white" alt="FiaseFaultML Banner"/>

<br/>

<img src="https://img.shields.io/badge/Regent%20University%20Ghana-Department%20of%20Engineering-0B3D91?style=flat-square"/>
<img src="https://img.shields.io/badge/Dissertation-Results%20Code%20Base-0B3D91?style=flat-square"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.13%2B-1a73e8?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Keras-v3.15-0B3D91?style=flat-square&logo=tensorflow&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.9-0B3D91?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-v3.0-0B3D91?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

<br/>

*A traceable machine-learning study of transmission-line fault detection: physics-based re-simulation, eight model architectures, a conventional distance-relay baseline, and the full H₁–H₅ statistical cascade.*

</div>

---

## Overview

The purpose of this project is to make **every number in the dissertation's results chapter regenerable from code**. It couples a Python re-simulation of a corrected 161 kV / 120 km transmission-line study (Z₁ = 0.080 + j0.390 Ω/km, Z₀ = 0.228 + j1.352 Ω/km) with eight machine-learning architectures and a numerical distance-relay baseline, then benchmarks all nine methods across five pre-registered evaluation criteria:

* **H₁ — Detection Accuracy**: overall binary fault detection vs the relay baseline (McNemar, Cochran's Q).
* **H₂ — Per-Phase Classification**: multi-label accuracy across phases A, B, C and Ground (per-target Cochran's Q + Holm).
* **H₃ — Reporting Latency**: per-row inference time vs the relay's 10 ms reference (Wilcoxon, Hodges–Lehmann).
* **H₄ — Robustness (TOST)**: accuracy stability under 1–20% injected noise, against a pre-declared 5 pp equivalence margin.
* **H₅ — Fault Location**: location error in km and as % of line length (Friedman + Holm-adjusted Wilcoxon).

*N.B: The datasets and every derived number in this repository were **regenerated on 2026-11-12** from the corrected simulation (`data/classdetectfault_sim.csv`, 5,365 rows; `data/detectfault_sim.csv`, 9,503 rows — 31.6% fault rows). In documents or notebooks where references are made to the superseded original datasets (14,400 rows, 45.8% fault rows, relay accuracy ≈ 0.9+, a 270 ms LSTM latency gap), treat those as legacy references — they no longer describe anything in this repository. If you wish to see such references tracked down and corrected, they are catalogued in [TODO.md](TODO.md) and [corrections/](corrections/).*

<details>
<summary><b>🔧 The Pipeline at a Glance</b></summary>
<br/>

| Stage | Script | Key Artifacts |
|:---|:---|:---|
| **Physics re-simulation** | `src/simulate.py` | Both `_sim` datasets, `relay_measurements_sim.csv`, 8-point verification battery (all PASS) |
| **Relay baseline** | `src/relay.py` | `predictions_relay.csv` — six loop impedances, Zone 1 = 38.22 Ω |
| **Models (8)** | `src/ann.py`, `cnn.py`, `dtr.py`, `knn.py`, `lstm.py`, `pinn.py`, `rnn.py`, `svm.py` | `predictions_<model>.csv`, `latency_per_row_*.csv`, `robustness_*.csv` |
| **Results table** | `src/table_4_3.py` | `table_4_3.csv` — accuracy/recall/F1 ± bootstrap SDs (2,000 resamples, seed 42) |
| **Statistics H₁–H₅** | `src/hypothesis.py` | McNemar / Cochran's Q / Wilcoxon / TOST / Friedman reports + results CSV |
| **Scenario CV** | `src/scenario_cv.py` | K = 7 scenario-grouped CV (671 scenarios, 0 spanning) |
| **Leakage audit** | `src/leakage_audit.py` | Nine-method scenario-overlap audit |
| **Writer's pack** | `src/analytics.py` | A4 PDF report + 14 publication-grade figures + manifest |

</details>

---

## Traceability Without Hand-Copied Numbers

In building this code base, the problem I had to tackle wasn't just producing results — it was making sure that no figure in the dissertation could drift away from the code that produced it. If this project has any use or advantage over a typical results notebook, then it's that.

Every script follows the same discipline: a timestamped `report_<timestamp>.txt` written to `outputs/<stage>/` alongside every CSV, and hard **verification anchors** asserted at load time — the pipeline refuses to build on a mismatch. A downstream script, `src/analytics.py`, demonstrates the reuse contract directly:

```python
"""Reuse the verified physics engine — no re-derivation anywhere."""

from simulate import prefault_state, solve_fault, loop_impedances
from relay import zone1_reach_ohms

pre   = prefault_state(0.75)                      # 75% of 750 A load
fault = solve_fault("AG", 25.0, 60, 0.75, pre)    # AG fault, R_f = 25 Ω, 60 km
z_ga  = loop_impedances(fault["v_r_abc"], fault["i_r_abc"])[0]

print(z_ga < zone1_reach_ohms())                  # inside Zone 1 (38.22 Ω)?
```

For the full regeneration cascade, each stage is a single command (see [Getting Started](#getting-started)), and every stage's report is re-runnable on demand:

- [src/simulate.py](src/simulate.py) — the 8-point verification battery that anchors the physics before any model touches the data
- [src/hypothesis.py](src/hypothesis.py) — H₁–H₅ in one pass, auto-discovering every `predictions_*.csv` file
- [src/analytics.py](src/analytics.py) — asserts the anchors (relay 0.7158, H₅ MAEs, dataset counts) before rendering a single figure

---

## The Writer's Analytics Pack

> **Architecture:** every figure and number is generated from the 2026-11-12 cascade reports only — no re-training, no new physics, no transcription.

The repository produces a self-contained pack for the dissertation writer, split into:

| Artefact | Location | Role |
|:---|:---|:---|
| **A4 PDF report** | `analytics/report_<timestamp>.pdf` | Clean, minimal 10-page document with the writer's companion and traceability appendix |
| **Figure stack** | `analytics/figures/` | 14 figures as 300 dpi PNG (Word) + vector PDF (LaTeX) |
| **Manifest** | `analytics/figures_manifest_<timestamp>.csv` | figure → section → source → md5 |
| **Replacement text** | `corrections/ch3_ch4_replacement_text_<date>.md` | Paste-ready Chapter 3/4 corrections |

To rebuild the pack after any cascade re-run:

```bash
.venv/bin/python src/analytics.py
```

---

## Getting Started

<table>
<tr>
<td width="50%">

### Prerequisites

1. Install [Python 3.13+](https://www.python.org/) and [uv](https://docs.astral.sh/uv/)
2. Create the environment:
   ```bash
   uv venv .venv --python 3.13
   uv pip install --python .venv/bin/python \
       numpy pandas scipy scikit-learn \
       tensorflow matplotlib
   ```
3. A TeX Live install with `pdflatex` — only needed to rebuild the analytics PDF (the figures themselves need only matplotlib)

</td>
<td width="50%">

### Quick Start

1. Regenerate the physics and datasets:
   ```bash
   .venv/bin/python src/simulate.py
   ```
2. Run the models and the relay baseline (`src/<model>.py` for each)
3. Run the statistical cascade:
   ```bash
   .venv/bin/python src/hypothesis.py
   ```
4. Build the writer's pack:
   ```bash
   .venv/bin/python src/analytics.py
   ```

Each stage prints its report to stdout **and** files a timestamped copy under `outputs/`.

</td>
</tr>
</table>

---

## Project Structure

```
Mr Fiase/
├── src/                          # ← Pipeline scripts (one stage per file)
│   ├── simulate.py               #   Line re-simulation + dataset generation
│   ├── relay.py                  #   Distance-relay baseline (6 loops, Zone 1)
│   ├── ann.py, cnn.py, dtr.py, knn.py
│   ├── lstm.py, pinn.py, rnn.py, svm.py
│   ├── table_4_3.py              #   Regenerated results table ± bootstrap SDs
│   ├── hypothesis.py             #   H₁–H₅ statistical tests
│   ├── scenario_cv.py            #   K = 7 scenario-grouped CV
│   ├── combination_accuracy.py   #   Exact type-vector accuracy by combination
│   ├── leakage_audit.py          #   Scenario-overlap audit (9 methods)
│   └── analytics.py              #   Writer-facing PDF + figure pack
│
├── data/                         # ← Regenerated datasets (md5-anchored)
│   ├── classdetectfault_sim.csv      # 5,365 rows — detection + 4 phase targets
│   └── detectfault_sim.csv           # 9,503 rows — detection only
│
├── outputs/<stage>/              # ← Timestamped reports + CSVs per stage
│   ├── simulate/, relay/, ann/, cnn/, dtr/, knn/
│   ├── lstm/, pinn/, rnn/, svm/
│   ├── table_4_3/, hypothesis_tests/, scenario_cv/
│   └── combination_accuracy/, leakage_audit/
│
├── analytics/                    # ← Writer's pack (PDF, figures, manifest)
├── corrections/                  # ← Paste-ready dissertation replacement text
├── docs/                         # ← Simulation caveats & reviewer notes
└── TODO.md                       # ← R1–R10 correction tracker
```

---

## Conventions

- Every script follows the same skeleton: path configuration block → `ReportLogger` (stdout + timestamped report file) → `main()` guard
- Predictions are named `predictions_<model>.csv` so `src/hypothesis.py` discovers them automatically via glob
- All timestamps use `%Y-%m-%d_%H-%M-%S`; metrics print with `:.6f`; counts print as integers
- **Traceability rule:** if a number in the results chapter cannot be regenerated by a script in this repository on demand, it does not go in the chapter

---

## Verified Anchors

This project is anchored to the 2026-11-12 cascade reports, asserted at build time rather than trusted:

<div align="center">

![Physics battery](https://img.shields.io/badge/8--point%20verification%20battery-all%20PASS-2A9D8F?style=for-the-badge)
![Relay anchor](https://img.shields.io/badge/relay%20overall%20accuracy-0.7158-2A9D8F?style=for-the-badge)
![H₅ anchors](https://img.shields.io/badge/H%E2%82%85%20MAE%20asserted-%C2%B10.05%20pp-2A9D8F?style=for-the-badge)
![Leakage audit](https://img.shields.io/badge/spanning%20scenarios-0%20of%20671-2A9D8F?style=for-the-badge)

Headline results: every learned model beats the Zone-1 relay by **23.7–24.3 pp** (all McNemar *p* < 10⁻⁶); the relay is secure but incomplete (TN 100%, TP 49.2%); the LSTM locates faults to **0.46%** of line length.

</div>

---

<div align="center">

**Ing. David Fiase** · Project Owner
**Somtochukwu Stanislus Emeka-Onwuneme** · Research Assistant

Department of Engineering, Regent University Ghana

### Connect with me

<a href="https://twitter.com/vzyengineer">
  <img src="https://img.shields.io/badge/X-000000?style=flat&logo=&logoColor=white" />
</a><a href="https://linkedin.com/in/emekasomto3">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=&logo=linkedin&logoColor=white" />
</a>

</div>
