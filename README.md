# Electrical Line Fault Detection Using Machine Learning

This repository evaluates six machine learning architectures—four classical algorithms and two deep learning models—for real-time fault detection and per-line phase classification (Lines A, B, C, and Ground) in electrical distribution networks.

## Research Purpose

Electrical power systems require rapid, precise fault detection to isolate line disruptions and protect grid infrastructure. This research benchmarks machine learning models against standard relay baselines across four core evaluation criteria:

* **H1: Fault Detection Accuracy**: Overall binary detection accuracy compared to standard relay baselines.
* **H2: Per-Line Classification**: Multi-label classification accuracy across individual phase lines (A, B, C) and Ground (G).
* **H3: Fault Reporting Latency**: Processing time and execution overhead per inference scenario.
* **H4: Robustness (TOST Equivalence)**: Model performance stability under signal noise levels ranging from 1% to 20%.

---

## Repository Structure

```text
├── data/
│   ├── raw/                  # Raw waveform and voltage/current sensor data
│   └── processed/            # Cleaned, aligned, and normalized scenario data
├── models/                   # Saved pretrained model artifacts
│   ├── ann_model.pkl         # Artificial Neural Network weights
│   ├── cnn_model.h5          # 1D Convolutional Neural Network model
│   ├── dtr_model.pkl         # Decision Tree Regressor/Classifier model
│   ├── knn_model.pkl         # K-Nearest Neighbors model
│   ├── lstm_model.h5         # Long Short-Term Memory model
│   └── rnn_model.h5          # Recurrent Neural Network model
├── src/
│   ├── data_loader.py        # Data ingestion and sequence alignment pipelines
│   ├── preprocessing.py     # Feature scaling and signal noise injection
│   ├── train.py              # Training routines for classical and deep learning models
│   ├── predict.py            # Inference wrapper for pretrained models
│   └── evaluate.py           # Statistical testing (Cochran's Q, McNemar, Wilcoxon, TOST)
├── notebooks/                # Exploratory data analysis and statistical reporting
├── main.py                   # Master script to run full training and evaluation pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

```

---

## Model Inventory & Artifact Storage

The project evaluates six distinct machine learning models stored in the `models/` directory:

| Model Name | Model Type | Architecture Class | File Location |
| --- | --- | --- | --- |
| **ANN** | Artificial Neural Network | Classical / Feedforward | `models/ann_model.pkl` |
| **CNN** | Convolutional Neural Network | Deep Learning | `models/cnn_model.h5` |
| **DTR** | Decision Tree | Classical / Non-Parametric | `models/dtr_model.pkl` |
| **KNN** | K-Nearest Neighbors | Classical / Instance-Based | `models/knn_model.pkl` |
| **LSTM** | Long Short-Term Memory | Deep Learning / Sequential | `models/lstm_model.h5` |
| **RNN** | Recurrent Neural Network | Classical / Recurrent | `models/rnn_model.h5` |

---

## Quick Start & Usage

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-org/electrical-fault-detection.git](https://github.com/your-org/electrical-fault-detection.git)
cd electrical-fault-detection
pip install -r requirements.txt

```

### 2. Run Inference with Pretrained Models

To run inference on processed sensor data using saved checkpoints:

```bash
python src/predict.py --model cnn --input data/processed/test_scenarios.csv

```

Available model flags: `ann`, `cnn`, `dtr`, `knn`, `lstm`, `rnn`.

### 3. Train Models

To retrain all models on current datasets:

```bash
python src/train.py --all

```

To train a specific model:

```bash
python src/train.py --model lstm

```

### 4. Run Statistical Evaluation

Execute the evaluation suite to reproduce hypothesis testing (H1–H4), latency comparisons, and TOST robustness checks:

```bash
python src/evaluate.py

```

