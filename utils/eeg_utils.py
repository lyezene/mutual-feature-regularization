import os
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset
import subprocess
from collections import Counter


def download_eeg_data(url, username, password, output_dir):
    command = f'wget -r -np -nH --cut-dirs=7 --no-clobber --user={username} --password={password} -P {output_dir} {url}'
    subprocess.run(command, shell=True, check=True)


def find_edf_files(root_dir):
    edf_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.edf'):
                edf_files.append(os.path.join(dirpath, filename))
    return edf_files


def load_edf_file(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signals = []
    sampling_rates = []
    for i in range(n):
        sig = f.readSignal(i)
        signals.append(sig)
        fs = f.getSampleFrequency(i)
        sampling_rates.append(fs)
    f._close()

    sampling_rate_counts = Counter(sampling_rates)
    most_common_fs, _ = sampling_rate_counts.most_common(1)[0]

    filtered_data = [
        (sig, label)
        for sig, label, fs in zip(signals, signal_labels, sampling_rates)
        if fs == most_common_fs
    ]

    if not filtered_data:
        raise ValueError(f"No signals with the most common sampling rate found in {file_path}.")

    signals_filtered, signal_labels_filtered = zip(*filtered_data)
    signals_filtered = list(signals_filtered)
    signal_labels_filtered = list(signal_labels_filtered)
    fs = most_common_fs

    min_length = min(len(sig) for sig in signals_filtered)
    signals_filtered = [sig[:min_length] for sig in signals_filtered]

    signals_array = np.array(signals_filtered)
    return signals_array, signal_labels_filtered, fs


def bandpass_filter(signals, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signals = filtfilt(b, a, signals, axis=1)
    return filtered_signals


def segment_signal(signals, fs, segment_length_sec):
    segment_length_samples = int(segment_length_sec * fs)
    n_channels, n_samples = signals.shape
    segments = []
    for start in range(0, n_samples - segment_length_samples + 1, segment_length_samples):
        end = start + segment_length_samples
        segment = signals[:, start:end]
        segments.append(segment)
    remainder = n_samples % segment_length_samples
    if remainder != 0:
        start = n_samples - remainder
        segment = signals[:, start:]
        padding = np.zeros((n_channels, segment_length_samples - remainder))
        segment = np.hstack((segment, padding))
        segments.append(segment)
    return segments


def normalize_segment(segment, epsilon=1e-8):
    means = np.mean(segment, axis=1, keepdims=True)
    stds = np.std(segment, axis=1, keepdims=True)
    stds = np.where(stds < epsilon, epsilon, stds)
    normalized_segment = (segment - means) / stds
    return normalized_segment


def vectorize_segments(segments):
    return [segment.flatten() for segment in segments]


def preprocess_and_save_data(root_dir, processed_data_file, segment_length_sec, lowcut, highcut, filter_order, most_common_fs):
    edf_files = find_edf_files(root_dir)
    n_channels = None
    segments_list = []
    for file_idx, file_path in enumerate(edf_files):
        signals, _, fs = load_edf_file(file_path)
        if fs != most_common_fs:
            print(f"Skipping file {file_path} due to different sampling rate ({fs} Hz).")
            continue

        if n_channels is None:
            n_channels = signals.shape[0]
        elif signals.shape[0] != n_channels:
            print(f"Skipping file {file_path} due to inconsistent number of channels.")
            continue

        filtered_signals = bandpass_filter(signals, lowcut, highcut, fs, filter_order)
        segments = segment_signal(filtered_signals, fs, segment_length_sec)
        normalized_segments = [normalize_segment(segment) for segment in segments]
        vectorized_segments = vectorize_segments(normalized_segments)

        segments_list.extend(vectorized_segments)

    segments_array = np.array(segments_list, dtype=np.float16)
    segments_tensor = torch.from_numpy(segments_array)
    torch.save(segments_tensor, processed_data_file)
    return n_channels


class EEGDataset(Dataset):
    def __init__(self, processed_data_file):
        self.segments_tensor = torch.load(processed_data_file, map_location='cpu')
        self.n_segments = self.segments_tensor.shape[0]
        self.segment_shape = self.segments_tensor.shape[1:]

    def __len__(self):
        return self.n_segments

    def __getitem__(self, idx):
        segment = self.segments_tensor[idx]
        return (segment,)

