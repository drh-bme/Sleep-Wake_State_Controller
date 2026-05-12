# Sleep/Wake State Controller and Classifier — Real-Time Simulator

A real-time EEG acquisition simulator, that is it emulates a real-time EEG acquisition pipelin, then classifies sleep depth from brain signal using frequency-band analysis. Built as a fun biomedical signal processing and ML modelling project using publicly available PhysioNet data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MNE](https://img.shields.io/badge/MNE-1.12%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview

The simulator emulates a real-time EEG acquisition pipeline — the kind used in clinical sleep monitoring hardware. It:

1. Downloads a real overnight EEG recording from the [PhysioNet Sleep-EDF dataset](https://physionet.org/content/sleep-edfx/1.0.0/)
2. Streams the signal through a bandpass filter (0.5–30 Hz) to remove noise and DC drift
3. Computes **relative band power** (Delta and Alpha) in 30-second epochs using Welch's method
4. Classifies sleep depth in real time using amplitude thresholds:
   * **Deep Sleep (N3)** — high delta power (>50%)
   * **Transition (N1/N2)** — moderate delta power (25–50%)
   * **Wake / Light Sleep** — low delta power (<25%)
5. Detects movement artifacts (amplitude > 500 µV) and raises a fault flag

A comparison window shows the same recording with expert-annotated sleep stages side by side.

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
## Controls

The simulator window has two sliders at the bottom:

| Slider | Range | Description |
|--------|-------|-------------|
| **Speed** | 0.25× – 4× | Playback speed of the animation |
| **Start (s)** | 0 – end | Jump to any point in the recording |

---

## Dataset

[PhysioNet Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/) — Cassette subset. Each recording is a full overnight PSG (polysomnography) with a single EEG channel (Fpz-Cz), sampled at 100 Hz, with expert-annotated sleep stages in 30-second epochs.

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).

---

## Installation & Setup
```bash
git clone https://github.com/drh-bme/Sleep-Wake_State_Controller.git
cd Sleep-Wake_State_Controller

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

#Run the simulator
python simulation.py
```
On first run, MNE will automatically download the PhysioNet Sleep-EDF recording (~50 MB) and cache it locally. Subsequent runs load from cache instantly.

---

## Roadmap

- [ ] Logistic regression classifier trained on epoch-level band features
- [ ] Cross-subject validation and confusion matrix
- [ ] Theta band (4–8 Hz) and sleep spindle detection for N1/N2 distinction
- [ ] Multi-channel support (EOG, EMG)

---
