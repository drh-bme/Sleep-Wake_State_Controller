import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from acquisition import get_data_stream
from processing import EEGProcessor
from print_subject import get_subject_data
from mlm import train_classifier 
import update

# --- Subject ---
ID = 10 

# --- Train ML Classifier ---
print("=== Training classifier... ===")
model, labels = train_classifier(n_subjects=5, verbose=True)
print("=== Classifier trained. Starting simulation... ===")

# --- Launch Comparison Window ---
get_subject_data(ID)

# --- Setup Real-Time Simulation Window ---
raw, _ = get_data_stream(ID)
fs = int(raw.info['sfreq'])
proc = EEGProcessor(fs)

fig, (ax_eeg, ax_bars) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
plt.subplots_adjust(bottom=0.25)

line, = ax_eeg.plot([], [], lw=0.75, color='#2c3e50')
ax_eeg.set_ylim(-100e-6, 100e-6)
ax_eeg.set_xlabel("Time (s)")
ax_eeg.set_ylabel("Amplitude (V)")
ax_eeg.set_title("EEG - Real-Time View")

bars = ax_bars.bar(
    ['Delta', 'Theta', 'Alpha', 'Beta'],
    [0, 0, 0, 0],
    color=['#2ecc71', '#f1c40f', '#e67e22', '#9b59b6']
)
ax_bars.set_ylim(0, 1)
ax_bars.set_ylabel("Relative Power")
ax_bars.set_title("Band Power")

status_text = fig.suptitle("Controller Active", fontsize=14, fontweight='bold')

# Initialize the update module
update.init_params(raw, proc, line, bars, status_text, ax_eeg,
                   model = model, labels =  labels)

# --- Sliders ---
ax_speed = plt.axes([0.15, 0.10, 0.3, 0.03])
ax_start = plt.axes([0.15, 0.05, 0.3, 0.03])

total_duration = raw.n_times / fs

slider_speed = Slider(ax_speed, 'Speed', 0.25, 4.0, valinit=1.0, valstep=0.25)
slider_start = Slider(ax_start, 'Start (s)', 0.0, total_duration - 120, valinit=0.0, valstep=1.0)

# --- Speed control state ---
# The interval (timer tick rate) stays fixed; only how far we jump each tick changes.
state = {'frame_idx': 0, 'base_step': fs // 4}

def on_speed_change(val):
    pass  # speed is read live from slider_speed.val inside animate()

def on_start_change(val):
    # Reset the internal frame counter to match the new start position
    state['frame_idx'] = int(slider_start.val * fs)
    
slider_speed.on_changed(on_speed_change)
slider_start.on_changed(on_start_change)

# --- Animation function ---
def animate(_counter):
    """
    Uses an internal frame index instead of FuncAnimation's frame argument.
    Each tick advances frame_idx by base_step * speed, so the speed slider
    directly controls how fast the signal scrolls in signal-time.
    """
    step = int(state['base_step'] * slider_speed.val)
    state['frame_idx'] += step

    if state['frame_idx'] >= raw.n_times:
        state['frame_idx'] = 0  # loop back to start

    return update.update_eeg(state['frame_idx'], fs)

# --- Animation Loop ---
ani = FuncAnimation(
    fig,
    animate,
    frames=itertools.count(),
    interval=50,
    blit=False
)

plt.show()
