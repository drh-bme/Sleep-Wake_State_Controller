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
        Extracts Delta (0.5-4Hz) and Alpha (8-12Hz) ratios.
        """
        # Calculate PSD using Welch's method
        freqs, psd = welch(data_chunk, self.fs, nperseg=self.fs*2)
        
        # Define band masks
        delta_mask = (freqs >= 0.5) & (freqs <= 4)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        total_mask = (freqs >= 0.5) & (freqs <= 30)
        
        # Calculate Relative Power
        total_pwr = np.trapezoid(psd[total_mask])
        delta_pwr = np.trapezoid(psd[delta_mask]) / total_pwr
        alpha_pwr = np.trapezoid(psd[alpha_mask]) / total_pwr
        
        return {"delta": delta_pwr, "alpha": alpha_pwr}