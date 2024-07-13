import mne
import numpy as np
from mne.io import read_raw_edf
from mne.datasets import eegbci
import torch
from data.gpt2_dataset import GPT2ActivationDataset
import wandb
from utils.gpt2_utils import stream_data, process_activations
from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_download
import math
import os
from data.synthetic_dataset import SyntheticIterableDataset
from config import get_device


def generate_synthetic_data(
    num_features,
    num_true_features,
    total_data_points,
    num_active_features_per_point,
    batch_size,
    decay_rate=0.99,
    num_feature_groups=12,
    device=get_device(),
    output_dir="synthetic_data_batches"
):
    os.makedirs(output_dir, exist_ok=True)
    true_features = torch.randn(num_features, num_true_features, device=device, dtype=torch.float16)
    group_size = num_true_features // num_feature_groups
    feature_group_indices = [
        torch.arange(i * group_size, (i + 1) * group_size, device=device)
        for i in range(num_feature_groups)
    ]

    group_feature_probs = [
        torch.pow(decay_rate, torch.arange(group_size, device=device, dtype=torch.float16))
        for _ in range(num_feature_groups)
    ]
    for probs in group_feature_probs:
        probs /= probs.sum()

    batch_files = []

    for batch_start in tqdm(
        range(0, total_data_points, batch_size), desc="Generating Batches"
    ):
        batch_end = min(batch_start + batch_size, total_data_points)
        current_batch_size = batch_end - batch_start
        batch_coefficients = torch.zeros(current_batch_size, num_true_features, device=device, dtype=torch.float16)

        selected_groups = torch.randint(num_feature_groups, (current_batch_size,), device=device)
        for i in range(num_feature_groups):
            mask = selected_groups == i
            if mask.any():
                selected_group_indices = feature_group_indices[i]
                selected_probs = group_feature_probs[i]
                selected_features = torch.multinomial(
                    selected_probs, num_active_features_per_point, replacement=False
                )
                indices = selected_group_indices[selected_features]
                batch_coefficients[mask.nonzero(as_tuple=True)[0].unsqueeze(1), indices] = torch.rand(
                    mask.sum(), num_active_features_per_point, device=device, dtype=torch.float16
                )

        batch_data = torch.mm(batch_coefficients, true_features.T)

        batch_file = os.path.join(output_dir, f"batch_{batch_start}.pt")
        torch.save(batch_data.cpu(), batch_file)
        batch_files.append(batch_file)

        del batch_coefficients, batch_data
        torch.cuda.empty_cache()

    data_batches = [torch.load(batch_file) for batch_file in batch_files]
    generated_data = torch.cat(data_batches)
    return generated_data, true_features


def load_synthetic_dataset(cache_dir=None, chunk_size=1000, num_epochs=1):
    repo_id = "lukemarks/synthetic_dataset"
    return SyntheticIterableDataset(repo_id, cache_dir, chunk_size, num_epochs)


def load_true_features(cache_dir=None):
    repo_id = "lukemarks/synthetic_dataset"
    cache_dir = cache_dir or os.path.join(os.getcwd(), 'hf_cache')
    file_path = hf_hub_download(repo_id, "data/true_features.pt", repo_type="dataset", cache_dir=cache_dir)
    return torch.load(file_path).float()


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


def generate_synthetic_dataset(config, device):
    num_features = config.get('num_features', 256)
    num_true_features = config.get('num_ground_features', 512)
    total_data_points = config.get('total_data_points', 100000)
    num_active_features_per_point = config.get('num_active_features_per_point', 42)
    batch_size = config.get('data_batch_size', 1000)

    generated_data, true_features = generate_synthetic_data(
        num_features,
        num_true_features,
        total_data_points,
        num_active_features_per_point,
        batch_size,
        device=device
    )

    train_dataset = torch.utils.data.TensorDataset(generated_data)

    dataset_artifact = wandb.Artifact('synthetic_dataset', type='dataset')

    dataset_path = 'synthetic_dataset.pth'
    torch.save(train_dataset, dataset_path)
    dataset_artifact.add_file(dataset_path)

    true_features_path = 'true_features.pth'
    torch.save(true_features, true_features_path)
    dataset_artifact.add_file(true_features_path)

    wandb.log_artifact(dataset_artifact)


def generate_gpt2_dataset(device):
    activation_dataset = GPT2ActivationDataset(model_name='gpt2', device=device)

    all_data = []
    data_stream = stream_data()
    for entry in data_stream:
        input_text = entry["text"]
        if not input_text.strip():
            continue

        activations, tokens = activation_dataset(input_text)
        processed_activations = process_activations(activations)
        all_data.append((processed_activations, tokens))

    artifact = wandb.Artifact('gpt2_activations_dataset', type='dataset')
    dataset_path = 'gpt2_dataset.pth'
    torch.save(all_data, dataset_path)
    artifact.add_file(dataset_path)
    wandb.log_artifact(artifact)

