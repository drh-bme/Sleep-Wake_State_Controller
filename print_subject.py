import mne
from mne.datasets.sleep_physionet.age import fetch_data

def get_subject_data(subject_id=0):
    """
    Retrieves the recording for the given ID and displays the EEG 
    with expert annotations. This function does not return any data.
    """
    try:
        # Acquisition: Fetch paths from PhysioNet
        files = fetch_data(subjects=[subject_id], recording=[1], verbose=False)
        psg_fname, annot_fname = files[0]

        # Loading: Read raw data and annotations
        raw = mne.io.read_raw_edf(psg_fname, preload=True, verbose=False)
        annotations = mne.read_annotations(annot_fname)
        raw.set_annotations(annotations, emit_warning=False)

        # Conditioning: Select only the target sensor
        raw.pick(['EEG Fpz-Cz'])

        # HMI (Human-Machine Interface): Print specs and launch plot
        print(f"\n--- LOG: Displaying Recording for Subject {subject_id} ---")
        print(f"Sampling Rate: {raw.info['sfreq']} Hz")
        print(f"Channels: {raw.ch_names}")
        
        # Displaying with annotations and auto-scaling
        raw.plot(
            duration=30, 
            n_channels=1, 
            scalings='auto', 
            title=f"Diagnostic View - Subject {subject_id}",
            block=False
        )

    except Exception as e:
        print(f"SYSTEM ERROR: Failed to display subject {subject_id}. Details: {e}")

if __name__ == "__main__":
    # Calling the void function
    get_subject_data(subject_id=0)