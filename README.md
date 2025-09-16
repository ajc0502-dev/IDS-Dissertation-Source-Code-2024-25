# IDS-Dissertation-Source-Code-2024-25
Source code for the submission of my MMU Cyber Security MSc Dissertation
=======
# AI-based Intrusion Detection System (Cowrie Honeypot)

A research implementation for a MMU MSc cyber security dissertation
---

## Features

* **Command-aware detection**: TF–IDF over command strings to flag malicious activity even without successful logins.
* **Multi-model ensemble**: Logistic Regression (text) + Random Forest (numeric) + Gradient Boosting (numeric), stacked via a meta-learner.
* **Baselines included**: Decision Tree model for quick benchmarking.
* **Convolutional Neural Network**: Tunable Deep Learning Mdep;
* **Sessionisation** of Cowrie events with heuristic labels: failed login **or** suspicious commands → malicious.
* **Unified CLI** through `main.py` to train and predict with either model.

---

## Installation

1. **Python**: 3.10+ recommended.

2. **Create a virtualenv** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Data inputs

### A) Raw Cowrie JSONL (preferred)

* One JSON object per line with typical Cowrie fields (e.g., `eventid`, `session`, `src_ip`, `username`, `password`, `input`, `duration`).
* Place at `data/MID.json` or change the path in `src/config.py`.

The loader will build sessions and derive labels with:

* `is_malicious = 1` if **any** of the following are true for the session:

  * a login **failed** event occurred, or
  * commands contain suspicious tokens (e.g., `wget`, `curl`, `nc`, `chmod`, `busybox`, etc.).

### B) Preprocessed CSV (fallback)

If you don’t have valid JSONL, you can use `data/processed.csv`. The **minimum columns** the pipeline can work with are:

* `src_ip` (string)
* `commands` (string; semicolon-separated commands in a session)
* `duration` (float seconds)
* `is_malicious` (0/1) — optional for prediction, required for training

> The feature engineer will also create derived numeric columns like `num_commands`, `suspicious_token_hits`, `avg_command_len`, and an `ip_encoded` label.

---

## Quick start

### Train the **ensemble** (recommended)

```bash
python main.py --model ensemble --train
```

* Saves to `saved_models/ensemble_ids.joblib`.
* Prints a classification report on a held-out test split.

### Train the **decision tree** (baseline)

```bash
python main.py --model dt --train
```

* Saves to `saved_models/decision_tree.joblib`.

### Train the **CNN** (DL Classifier)

```bash
python main.py --model cnn --train
```

* Saves to `saved_models/cnn_tree.joblib`.

### Predict on new data

Prepare a CSV with at least `commands`, `src_ip`, and `duration`. Example:

```csv
session,src_ip,commands,duration
S1,203.0.113.7,"wget http://x; chmod +x a; ./a",35.1
S2,198.51.100.9,"ls; whoami",4.0
```

Run prediction with the chosen model:

```bash
python main.py --model ensemble --predict path/to/new.csv
# or
python main.py --model dt --predict path/to/new.csv
# or
python main.py --model cnn --predict path/to/new.csv
```

Outputs an array of predictions (0 = benign, 1 = malicious).

---

## Configuration

`src/config.py` contains paths. Defaults:

```python
BASE_DIRECTORY = <project root>
DATA_PATH = BASE_DIRECTORY/data/
PROCESSED_DATA_PATH = BASE_DIRECTORY/data/
MODEL_DIRECTORY = BASE_DIRECTORY/saved_models
```

---

## How it works

### 1) Loading & labelling (`src/loader.py`)

* Reads JSONL, groups by `session`.
* Tracks login events and commands.
* Binary label: **failed login OR suspicious commands ⇒ malicious**.

### 2) Feature engineering (`src/f_engineer.py`)

* Keeps raw `commands` text for TF–IDF.
* Builds numeric features: `num_commands`, `suspicious_token_hits`, `avg_command_len`, `duration`, `ip_encoded`.

### 3) Models

* **Text LR** over TF–IDF of `commands` (captures semantics like `wget …; chmod +x …`).
* **Random Forest** over numeric features (non-linear tabular patterns).
* **Gradient Boosting** over numeric features (strong baseline booster).
* **Stacking meta-learner** (Logistic Regression) combines base model probabilities.
* **Decision Tree** provided as a simple baseline and for explainability.
* **CNN** DL classification for sequential features.

---

## Evaluation

* Automatic **train/test split** (25% test, stratified).
* Reports **Precision, Recall, F1** via `sklearn.metrics.classification_report`.
* For the dissertation, include:

  * Class distribution (label counts),
  * Confusion matrix plot,
  * Short error analysis (examples of FP/FN sessions and why).

---

## Reproducibility tips

* Fix random seeds where applicable (`random_state=42`).
* Save `processed.csv` after engineering for auditability.
* Export model artefacts with exact timestamps if running multiple experiments, e.g. `saved_models/ensemble_YYYYMMDD_HHMM.joblib`.

---

## Troubleshooting

* **Invalid JSONL**: the loader will skip malformed lines and log warnings. If many lines are invalid, use `processed.csv` as a fallback.
* **Module import errors**: ensure you run from the project root (where `main.py` lives), or add the root to `PYTHONPATH`.
* **No `saved_models` dir**: it’s created automatically on first train; otherwise create it manually.
* **Mismatched columns on predict**: make sure your CSV has `commands`, `src_ip`, `duration`. Unknown columns are ignored.

---

## Extending the system

* Add **calibration** for reliable alerting thresholds (e.g., `CalibratedClassifierCV`).
* Add external **threat intel features** (URL/domain reputation hits, hashes) as extra numeric columns.
* Add external **attention insight Hhatmaps** for post evaluation analysis.

---

## License

Academic/research use.
>>>>>>> 0a09bda (Initial commit: IDS Project)
