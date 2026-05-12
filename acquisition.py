import mne
from mne.datasets.sleep_physionet.age import fetch_data

def get_data_stream(subject_id=0):
    """
    Acts as the data provider, emulates a hardware interface. In a real system, this would 
    interface with a USB or Bluetooth EEG amplifier. 
    Fetches the PhysioNet EDF files, pairs them with annotations, and provides the raw "sensor" stream.

    Inputs: subject_id (int): The ID of the subject to load (0-19)
    Outputs: raw (mne.io.Raw): The raw EEG data with annotations
    """
    # Fetch data from the web/cache
    files = fetch_data(subjects=[subject_id], recording=[1])
    
    # Load the signal (PSG) and the expert labels (Hypnogram)
    raw = mne.io.read_raw_edf(files[0][0], preload=True)
    annot = mne.read_annotations(files[0][1])
    
    # Sync the labels with the signal
    raw.set_annotations(annot, emit_warning=False)
    
    # Select the primary sensor
    # More channels can be chosen to see them in the simulation
    raw.pick_channels(['EEG Fpz-Cz'])
    
    return raw, annot