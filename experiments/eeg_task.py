import os
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.sae_trainer import SAETrainer
from models.sae import SparseAutoencoder
import wandb
import subprocess


def download_eeg_data(url, username, password, output_dir):
    command = f'wget -r -np -nH --cut-dirs=7 --user={username} --password={password} -P {output_dir} {url}'
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

    if not all(fs == sampling_rates[0] for fs in sampling_rates):
        raise ValueError(f"Signals in {file_path} have different sampling rates.")
    fs = sampling_rates[0]

    min_length = min(len(sig) for sig in signals)
    signals = [sig[:min_length] for sig in signals]

    signals = np.array(signals)
    return signals, signal_labels, fs


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


def create_normalized_shuffled_dataset(root_dir, segment_length_sec, lowcut, highcut, filter_order):
    edf_files = find_edf_files(root_dir)
    dataset = []
    n_channels = None
    for file_path in edf_files:
        signals, _, fs = load_edf_file(file_path)
        if n_channels is None:
            n_channels = signals.shape[0]
        elif signals.shape[0] != n_channels:
            print(f"Skipping file {file_path} due to inconsistent number of channels.")
            continue

        filtered_signals = bandpass_filter(signals, lowcut, highcut, fs, filter_order)
        segments = segment_signal(filtered_signals, fs, segment_length_sec)
        normalized_segments = [normalize_segment(segment) for segment in segments]
        vectorized_segments = vectorize_segments(normalized_segments)
        dataset.extend(vectorized_segments)
    dataset = np.array(dataset)
    dataset = shuffle(dataset, random_state=42)
    return dataset, n_channels


def run(device, config):
    eeg_data_dir = "eeg_data"
    os.makedirs(eeg_data_dir, exist_ok=True)
    download_eeg_data(
        config['data']['eeg_data_url'],
        os.environ['EEG_USERNAME'],
        os.environ['EEG_PASSWORD'],
        eeg_data_dir
    )

    dataset, n_channels = create_normalized_shuffled_dataset(
        eeg_data_dir,
        config['hyperparameters']['segment_length_sec'],
        config['hyperparameters']['lowcut'],
        config['hyperparameters']['highcut'],
        config['hyperparameters']['filter_order']
    )

    segment_length_samples = int(config['hyperparameters']['segment_length_sec'] * config['hyperparameters']['sampling_rate'])
    input_dim = n_channels * segment_length_samples

    config['hyperparameters']['input_size'] = 8250

    tensor_dataset = TensorDataset(torch.FloatTensor(dataset))
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=config['hyperparameters']['training_batch_size'],
        shuffle=True
    )
    model = SparseAutoencoder(config['hyperparameters']).to(device)
    trainer = SAETrainer(model, device, config['hyperparameters'])
    trainer.train(dataloader, config['hyperparameters']['num_epochs'])
    trainer.save_model(config['hyperparameters']['num_epochs'])
    wandb.log({"experiment_completed": True})

