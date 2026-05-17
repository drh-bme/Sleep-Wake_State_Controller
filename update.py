import numpy as np

# These will hold the references passed from simulation.py
raw = None
proc = None
line = None
bars = None
status_text = None
ax_eeg = None
_model = None
_labels = None

# Track the last second at which state was updated, to properly throttle
# update_signaling_state to once per second instead of every frame.
_last_state_update_sec = -1

# Colour map for each sleep stage label
STAGE_COLORS = {
    "W":  "royalblue",
    "N1": "gold",
    "N2": "orange",
    "N3": "green",
    "R":  "mediumpurple",
}
STAGE_LABELS = {
    "W":  "STATE: WAKE",
    "N1": "STATE: LIGHT SLEEP (N1)",
    "N2": "STATE: SLEEP (N2)",
    "N3": "STATE: DEEP SLEEP (N3)",
    "R":  "STATE: REM",
}

def init_params(_raw, _proc, _line, _bars, _st, _ax, model=None, labels=None):
    """Initializes the update module with the UI and data objects."""
    global raw, proc, line, bars, status_text, ax_eeg, _model, _labels
    raw, proc, line, bars, status_text, ax_eeg, _model, _labels = _raw, _proc, _line, _bars, _st, _ax, model, labels

def _extract_features(signal, fs):
    """Builds the 6-element feature vector expected by the trained pipeline."""
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)

    def bandpower(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.trapezoid(psd[mask], freqs[mask])
    
    # Define rel band powers
    total = bandpower(0.5, 30.0)
    delta = bandpower(0.5, 3.9) / total
    theta = bandpower(4.0, 7.9) / total
    alpha = bandpower(8.0, 11.9) / total
    beta = bandpower(12.0, 30.0) / total

    denom = alpha + beta
    delta_rel = delta / denom if denom > 0 else 0
    log_var = np.log(np.var(signal) + 1e-12)

    return np.array([[delta_rel, alpha, beta, log_var, theta, total]])


def update_signaling_state(idx, fs):
    """Updates the signaling state based on the current EEG chunk. This is the 'Controller Logic' that decides
    what state we're in and updates the status text and band power bars accordingly.
    """
    logic_samples = 30 * fs    
    if idx < logic_samples:
        return

    chunk_logic, _ = raw[0, idx - logic_samples: idx]
    filtered = proc.apply_filter(chunk_logic.flatten())

    if proc.check_for_artifacts(filtered):
        status_text.set_text("SYSTEM FAULT: Artifact Detected")
        status_text.set_color('red')
        return

    powers = proc.get_band_power(filtered)
    bar_keys = ['delta', 'theta', 'alpha', 'beta']
    for bar, key in zip(bars, bar_keys):
        bar.set_height(powers.get(key, 0))

    # Use the trained ML model if available, else fall back to delta-based heuristic
    if _model is not None:
        feats = _extract_features(filtered, fs)
        stage = _model.predict(feats)[0]
        status_text.set_text(STAGE_LABELS.get(stage, f"STATE: {stage}"))
        status_text.set_color(STAGE_COLORS.get(stage, 'black'))
    else:
        # Fallback: heuristic based predictor (no model loaded)
        if powers['delta'] > 0.5:
            status_text.set_text("STATE: DEEP SLEEP (N3)")
            status_text.set_color('green')
        elif 0.25 <= powers['delta'] <= 0.5:
            status_text.set_text("STATE: TRANSITION (N1/N2)")
            status_text.set_color('orange')
        else:
            status_text.set_text("STATE: WAKE / LIGHT")
            status_text.set_color('blue')

def update_eeg(frame, fs):
    global _last_state_update_sec

    display_samples = 60 * fs
    idx = frame
    end_idx = idx + display_samples

    if end_idx >= raw.n_times:
        return line, *bars

    chunk_visual, _ = raw[0, idx:end_idx]
    current_time_start = idx / fs
    current_time_end = end_idx / fs

    # Update the EEG plot window
    ax_eeg.set_xlim(current_time_start, current_time_end)
    line.set_data(
        np.linspace(current_time_start, current_time_end, chunk_visual.shape[1]),
        chunk_visual[0]
    )
    # Update the signaling state once per second based on the current chunk
    current_sec = idx // fs
    if current_sec != _last_state_update_sec:
        _last_state_update_sec = current_sec
        update_signaling_state(idx, fs)

    # Explicitly request a canvas redraw so the x-axis scrolling and suptitle
    # color changes are reflected on all Matplotlib backends.
    ax_eeg.figure.canvas.draw_idle()

    return line, *bars
