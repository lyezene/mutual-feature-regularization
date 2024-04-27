from torch.utils.data import DataLoader, TensorDataset
import torch
from data.gpt2_dataset import GPT2ActivationDataset
from utils.gpt2_utils import stream_data, train_gpt2_sae


def run(device, config):
    activation_dataset = GPT2ActivationDataset("gpt2", device)

    data_stream = stream_data()
    all_activations = []

    for entry in data_stream:
        input_text = entry["text"]
        if not input_text.strip():
            continue

        activations, _ = activation_dataset(input_text)

        for activation in activations:
            all_activations.append(activation)

    train_gpt2_sae(
        all_activations, device, config
    )
