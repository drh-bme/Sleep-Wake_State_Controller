# Sleep/Wake State Controller and Classifier — Real-Time Simulator

A real-time EEG acquisition simulator, that is it emulates a real-time EEG acquisition pipelin, then classifies sleep depth from brain signal using frequency-band analysis. Built as a fun biomedical signal processing and ML modelling project using publicly available PhysioNet data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MNE](https://img.shields.io/badge/MNE-1.12%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![ScikitLearn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)

## Project Overview

The simulator emulates a real-time EEG acquisition pipeline — the kind used in clinical sleep monitoring hardware. On launch it:

1. Downloads a real overnight EEG recording from the [PhysioNet Sleep-EDF dataset](https://physionet.org/content/sleep-edfx/1.0.0/).
2. Streams the signal through a bandpass filter (0.5–30 Hz) to remove noise and DC drift.
3. Trains a logistic regression classifier on 5 subjects from the PhysioNet Sleep-EDF dataset, printing cross-validation metrics and a confusion matrix.
4. Detects movement artifacts (amplitude > 500 µV) and raises a fault flag.
5. Opens a diagnostic comparison window showing the same recording with expert-annotated sleep stages.
6. Streams the EEG signal in real time, classifying each 30-second epoch using the trained model and displaying the predicted sleep stage live.
The sleep stages are classified like:

| Stage | Label | EEG Signature |
|-------|-------|---------------|
|Wake| W | High Alpha/ Beta activity|
| Light Sleep | N1 | Decreasing Alpha, Vertex Sharp Waves |
| Sleep | N2 | Sleep Spindles, K-complexes|
| Deep Sleep | N3 | High-amplitude Delta |
| REM | R | Mixed frequency, low amplitude |
---
## Architecture
```
simulation.py   ← entry point, orchestrates all modules
      │
      ├── mlm.py          ← ML pipeline (training, evaluation, confusion matrix)
      │     └── processing.py  ← EEGProcessor (filter, artifact detection, band power)
      │     └── acquisition.py ← downloads PhysioNet EDF files via MNE
      │
      ├── acquisition.py  ← real-time data stream (emulates hardware interface)
      ├── processing.py   ← signal conditioning
      ├── print_subject.py← offline diagnostic viewer with expert annotations
      └── update.py       ← animation loop, live classification display
```

---
## Signal Processing Pipeline

```
Raw EEG (PhysioNet EDF)
        │
        ▼
Butterworth Bandpass Filter (0.5–30 Hz, order 4)
        │
        ▼
Artifact Detection (threshold: 500 µV peak)
        │
        ▼
Power Spectral Density — Welch's method (2s windows)
        │
        ▼
Relative Band Power:  Delta (0.5–4 Hz) | Alpha (8–12 Hz)
        │
        ▼
Threshold Classifier → Sleep State Label
```

---

## ML Classifier
The classifier in <code>mlm.py</code> uses logistic regression - chosen because  it is interpretable, fast to train, and well-suited to linearly separable spectral features. More complex models (SVM, random forest, LSTM) tend to outperform it, but at the cost of interpretability which matters in a biomedical context.

### Feature vector (6 features per 30s epoch):
| Feature | Description |
|---------|-------------|
|delta_power|Relative power 0.5–3.9 Hz|
|theta_power|Relative power 4–7.9 Hz|
|alpha_power|Relative power 8-11.9 Hz|
|beta_power| Relative power 12-30 Hz|
|delta_ratio| delta/(alpha + beta) - sleep depth ratio|
|log_variance| log of signal variance - amplitude proxy|

### Training details:
* 5 subjects from PhysioNet Sleep-EDF (cassette subset)
* Stages 3 and 4 merged into N3 (AASM 2007 guidelines)
* <code>"Sleep stage ?"</code> epochs excluded (unreliable expert label)
* <code>class_weight='balanced'</code> to correct for N1 underrepresentation
* Evaluated with 5-fold stratified cross-validation
* Metrics reported: per-class precision, recall, F1; confusion matrix

### Expected performance (logistic regression on spectral features, 5-fold CV):

* W and N3 are reliably separated by delta power
* N1 is the hardest class — spectrally similar to both W and N2
* REM is often confused with W (both show low-delta, mixed-frequency EEG on a single Fpz-Cz channel)

### Model Performance

The 5-fold cross-validation yielded a stable average accuracy of **~64%**. Below is the detailed classification report broken down by sleep stage:

| Sleep Stage | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **N1**      | 0.61      | 0.47   | 0.53     | 152     |
| **N2**      | 0.73      | 0.55   | 0.63     | 225     |
| **N3**      | 0.76      | 0.90   | 0.83     | 195     |
| **R (REM)** | 0.50      | 0.86   | 0.63     | 63      |
| **W (Wake)**| 0.41      | 0.46   | 0.44     | 80      |

**Overall Metrics:**
* **Accuracy:** 0.64
* **Macro Avg:** 0.61
* **Weighted Avg:** 0.64

While a 64% accuracy might seem modest at first glance, sleep stage scoring involves 5 highly imbalanced, sequential classes. A random guess would yield 20% accuracy, meaning the model has learned significant physiological patterns from the feature engineering pipeline. The stability across folds (62.9% to 66.4%) also confirms that the model is robust and not suffering from overfitting.

#### Key Strengths:
* **Excellent N3 Detection:** The model excels at identifying slow-wave deep sleep, boasting a 0.83 F1-score and an outstanding 0.90 recall. This indicates that the low-frequency Delta power features (``delta_ratio`` and ``delta`` relative power) are highly effective biomarkers that clearly differentiate deep sleep from other stages.
* **High Sensitivity to R:** With a recall of 0.86, the model is highly sensitive to REM sleep, successfully catching 86% of all true REM periods.

#### Key weaknesses:
* **The REM False Positives:** While the model finds REM easily (high recall), its precision is only 0.50. This means exactly half of the times the model predicts REM, it is a false alarm. In EEG data, REM sleep heavily resembles both Wakefulness (W) and Stage N1 sleep (characterized by low-amplitude, mixed-frequency waves). Without eye-movement (EOG) or muscle-tone (EMG) data, a simple linear model like Logistic Regression struggles to separate them.
* **Severe Underperformance on W and N1:** These are the weakest links with 0.44 a d 0.53 F1-score respectively. The model misses more than half of actual N1 stages, but this could be normal since it is a short, ambiguous transitional state. The poor performance on the Wake is more critical; a precision of 0.41 suggests the model is aggressively misclassifying active wakefulness as sleep, likely due to overlapping spectral features in the alpha/beta bands during relaxed wakefulness.

#### Possible Solutions:
1. **Sequence Modeling:** Since sleep stages do not happen at random, introducing a sequence-aware model (like a Hidden Markov Model, or a Random Forest with time-lagged features) could fix the false positives.
2. **Multimodal Data:** Integrating EOG and EMG recordings  would resolve the overlap between Wake, N1, and REM.
3. **Class Imbalance Handling:** Stage N2 typically dominates sleep datasets. Utilizing techniques like SMOTE (Synthetic Minority Over-sampling Technique) or adjusting the ``class_weight='balanced'`` hyperparameter in Logistic Regression could help boost the precision of the minority classes like Wake and REM.

---

## Repository Structure

```
.
├── README.md
├── acquisition.py      # Data provider — downloads and loads PhysioNet EDF files
├── mlm.py              # ML model - returns better suited values for better state detection
├── print_subject.py    # Diagnostic viewer — offline plot with expert annotations
├── processing.py       # EEGProcessor class — filtering, artifact detection, band power
├── requirements.txt
├── simulation.py       # Entry point — launches the real-time simulator
└── update.py           # Animation loop — updates the EEG plot and state display
```


---

## Dataset

[PhysioNet Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/) — Cassette subset. Each recording is a full overnight PSG (polysomnography) with a single EEG channel (Fpz-Cz), sampled at 100 Hz, with expert-annotated sleep stages in 30-second epochs.

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).

---

## Installation & Setup
```bash
git clone https://github.com/drh-bme/Sleep-Wake_State_Controller.git
cd Sleep-Wake_State_Controller
```
For windows:
```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#Run the simulator
python simulation.py 
```
For Mac/Linux:
```bash
python3 /m venv venv
source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt

#Run the simulator
python3 simulation.py 
```
On first run, MNE will automatically download the PhysioNet Sleep-EDF recording (~50 MB) and cache it locally. Subsequent runs load from cache instantly.

---
## Controls

The simulator window has two sliders at the bottom:

| Slider | Range | Description |
|--------|-------|-------------|
| **Speed** | 0.25× – 4× | Playback speed of the animation |
| **Start (s)** | 0 – end | Jump to any point in the recording |

---

## Requirements

* Python 3.10+
* See <code>requirements.txt</code>

---
