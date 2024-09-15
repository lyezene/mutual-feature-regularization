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
    return signals, signal_labels, sampling_rates

def bandpass_filter(signal, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def segment_signal(signal, fs, segment_length_sec):
    segment_length_samples = int(segment_length_sec * fs)
    n_samples = len(signal)
    segments = []
    for start in range(0, n_samples, segment_length_samples):
        end = start + segment_length_samples
        segment = signal[start:end]
        if len(segment) == segment_length_samples:
            segments.append(segment)
        else:
            segment = np.pad(segment, (0, segment_length_samples - len(segment)), 'constant')
            segments.append(segment)
    return segments

def normalize_segment(segment):
    return (segment - np.mean(segment)) / np.std(segment)

def vectorize_segments(segments):
    return [segment.reshape(-1) for segment in segments]

def create_normalized_shuffled_dataset(root_dir, segment_length_sec, lowcut, highcut, filter_order):
    edf_files = find_edf_files(root_dir)
    dataset = []
    for file_path in edf_files:
        signals, _, sampling_rates = load_edf_file(file_path)
        for sig, fs in zip(signals, sampling_rates):
            filtered_sig = bandpass_filter(sig, lowcut, highcut, fs, filter_order)
            segments = segment_signal(filtered_sig, fs, segment_length_sec)
            normalized_segments = [normalize_segment(segment) for segment in segments]
            vectorized_segments = vectorize_segments(normalized_segments)
            dataset.extend(vectorized_segments)
    dataset = np.array(dataset)
    dataset = shuffle(dataset, random_state=42)
    return dataset

def run(device, config):
    # Download EEG data
    eeg_data_dir = "eeg_data"
    os.makedirs(eeg_data_dir, exist_ok=True)
    download_eeg_data(config['data']['eeg_data_url'], os.environ['EEG_USERNAME'], os.environ['EEG_PASSWORD'], eeg_data_dir)

    # Create dataset
    dataset = create_normalized_shuffled_dataset(
        eeg_data_dir,
        config['hyperparameters']['segment_length_sec'],
        config['hyperparameters']['lowcut'],
        config['hyperparameters']['highcut'],
        config['hyperparameters']['filter_order']
    )

    # Convert to PyTorch dataset and create DataLoader
    tensor_dataset = TensorDataset(torch.FloatTensor(dataset))
    dataloader = DataLoader(tensor_dataset, batch_size=config['hyperparameters']['training_batch_size'], shuffle=True)

    # Initialize model
    model = SparseAutoencoder(config['hyperparameters']).to(device)

    # Initialize trainer
    trainer = SAETrainer(model, device, config['hyperparameters'])

    # Train model
    trainer.train(dataloader, config['hyperparameters']['num_epochs'])

    # Save model
    trainer.save_model(config['hyperparameters']['num_epochs'])

    wandb.log({"experiment_completed": True})
