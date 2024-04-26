import mne
import numpy as np
from mne.io import read_raw_edf
from mne.datasets import eegbci

def load_eeg_data(subjects, runs, interval):
    '''
    Placeholder function that works for a toy dataset
    '''
    data = []
    max_seq_len = 0

    for subject in subjects:
        for run in runs:
            raw = read_raw_edf(eegbci.load_data(subject, run)[0], preload=True)
            raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
            picks = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
            )
            sfreq = raw.info["sfreq"]
            run_data = np.concatenate(
                [
                    raw.get_data(picks, start=i, stop=i + int(interval * sfreq))
                    for i in range(0, raw.n_times, int(interval * sfreq))
                ],
                axis=1,
            )
            max_seq_len = max(max_seq_len, run_data.shape[1])
            data.append(run_data.T)

    padded_data = []
    for seq in data:
        pad_width = ((0, max_seq_len - seq.shape[0]), (0, 0))
        padded_seq = np.pad(seq, pad_width, mode="constant")
        padded_data.append(padded_seq)

    padded_data = np.array(padded_data)
    return padded_data
