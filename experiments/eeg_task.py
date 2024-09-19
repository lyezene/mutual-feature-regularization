import os
import torch
from torch.utils.data import DataLoader
from utils.sae_trainer import SAETrainer
from models.sae import SparseAutoencoder
import wandb
from collections import Counter
from utils import eeg_utils
import pyedflib


def run(device, config):
    eeg_data_dir = "eeg_data"
    processed_data_file = "processed_data.pt"
    os.makedirs(eeg_data_dir, exist_ok=True)
    eeg_utils.download_eeg_data(
        config['data']['eeg_data_url'],
        os.environ['EEG_USERNAME'],
        os.environ['EEG_PASSWORD'],
        eeg_data_dir
    )

    edf_files = eeg_utils.find_edf_files(eeg_data_dir)
    sampling_rates = []
    for file_path in edf_files:
        f = pyedflib.EdfReader(file_path)
        n = f.signals_in_file
        for i in range(n):
            fs = f.getSampleFrequency(i)
            sampling_rates.append(fs)
        f._close()

    sampling_rate_counts = Counter(sampling_rates)
    most_common_fs, _ = sampling_rate_counts.most_common(1)[0]
    print(f"Most common sampling rate across all files: {most_common_fs} Hz")

    n_channels = eeg_utils.preprocess_and_save_data(
        eeg_data_dir,
        processed_data_file,
        config['hyperparameters']['segment_length_sec'],
        config['hyperparameters']['lowcut'],
        config['hyperparameters']['highcut'],
        config['hyperparameters']['filter_order'],
        most_common_fs
    )

    segment_length_samples = int(config['hyperparameters']['segment_length_sec'] * most_common_fs)
    input_dim = n_channels * segment_length_samples
    config['hyperparameters']['input_size'] = input_dim

    eeg_dataset = eeg_utils.EEGDataset(processed_data_file)

    dataloader = DataLoader(
        eeg_dataset,
        batch_size=config['hyperparameters']['training_batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = SparseAutoencoder(config['hyperparameters']).to(device)
    trainer = SAETrainer(model, device, config['hyperparameters'])
    trainer.train(dataloader, config['hyperparameters']['num_epochs'])
    trainer.save_model(config['hyperparameters']['num_epochs'])
    wandb.log({"experiment_completed": True})

