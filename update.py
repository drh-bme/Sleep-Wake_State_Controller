import numpy as np

# These will hold the references passed from simulation.py
START_TIME_SEC = 0.0
raw = None
proc = None
line = None
bars = None
status_text = None
ax_eeg = None

# Track the last second at which state was updated, to properly throttle
# update_signaling_state to once per second instead of every frame.
_last_state_update_sec = -1

def init_params(_raw, _proc, _line, _bars, _st, _ax):
    """Initializes the update module with the UI and data objects."""
    global raw, proc, line, bars, status_text, ax_eeg
    raw, proc, line, bars, status_text, ax_eeg = _raw, _proc, _line, _bars, _st, _ax

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
    bars[0].set_height(powers['delta'])
    bars[1].set_height(powers['alpha'])

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