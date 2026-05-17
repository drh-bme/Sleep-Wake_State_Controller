import numpy as np
from scipy.signal import butter, filtfilt, welch

class EEGProcessor:
    def __init__(self, fs):
        self.fs = fs
        # Design a Bandpass Filter (0.5 - 30 Hz)
        # to account for DC drift and high-freq noise
        self.b, self.a = butter(4, [0.5, 30], btype='band', fs=fs)

    def apply_filter(self, data):
        return filtfilt(self.b, self.a, data)

    def check_for_artifacts(self, data_chunk):
        """
        The 'Inhibit' signal. If amplitude is too high, 
        the sensor might have moved.
        """
        threshold = 500e-6 # 500 microvolts standard threshold for artifacts
        if np.max(np.abs(data_chunk)) > threshold:
            return True # Artifact detected
        return False

    def get_band_power(self, data_chunk):
        """
        Extracts relative power for all four clinical EEG bands:
            Delta  0.5–3.9  Hz  — slow wave sleep (N3)
            Theta  4–7.9    Hz  — drowsiness, N1
            Alpha  8–11.9   Hz  — relaxed wakefulness
            Beta   12–30  Hz  — active wakefulness, REM
        All values are relative to total power (0.5–30 Hz), so they sum to 1.
        """
        # Calculate PSD using Welch's method
        freqs, psd = welch(data_chunk, self.fs, nperseg=self.fs*2)
        
        def rel_bandpower(low, high):
            mask = (freqs >= low) & (freqs <= high)
            total_mask = (freqs >= 0.5) & (freqs <= 30)
            total_pwr = np.trapezoid(psd[total_mask], freqs[total_mask])
            return np.trapezoid(psd[mask], freqs[mask]) / total_pwr if total_pwr > 0 else 0

        return {
            "delta": rel_bandpower(0.5, 3.9),
            "theta": rel_bandpower(4, 7.9),
            "alpha": rel_bandpower(8, 11.9),
            "beta": rel_bandpower(12, 30)
        }
