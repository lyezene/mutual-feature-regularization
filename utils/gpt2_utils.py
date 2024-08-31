from torch.cuda.amp import autocast
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
from models.gpt2 import GPT2Shortcut


class GPT2ActivationsDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.activation_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('activations_') and f.endswith('.npy')])
        self.num_files = len(self.activation_files)

        first_file = np.load(os.path.join(self.data_dir, self.activation_files[0]))
        self.batch_size, self.seq_len, self.d_model = first_file.shape

        self.activations_per_file = self.batch_size * self.seq_len
        self.total_samples = self.num_files * self.activations_per_file

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // self.activations_per_file
        item_idx = idx % self.activations_per_file

        batch_idx = (item_idx // self.seq_len) % self.batch_size
        seq_idx = item_idx % self.seq_len

        filename = self.activation_files[file_idx]
        file_path = os.path.join(self.data_dir, filename)

        activations = np.load(file_path)
        return (torch.tensor(activations[batch_idx, seq_idx], dtype=torch.float32),)


def generate_activations(device: str, num_samples: int, batch_size: int, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    dataset = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train", streaming=True)
    config = GPT2Config.from_pretrained("gpt2")
    embeddings = GPT2Model.from_pretrained("gpt2").wte.to(device)
    optimized_layer = GPT2Shortcut(config).to(device)

    dataset_iter = iter(dataset.shuffle(seed=42).take(num_samples))
    for i in tqdm(range(0, num_samples, batch_size), desc="Collecting activations"):
        batch = [next(dataset_iter)["input_ids"] for _ in range(min(batch_size, num_samples - i))]
        max_length = max(len(seq) for seq in batch)
        padded_batch = torch.tensor([seq + [0] * (max_length - len(seq)) for seq in batch], dtype=torch.long).to(device)

        attention_mask = (padded_batch != 0).to(torch.float32).to(device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(padded_batch.size(0), 1, padded_batch.size(1), padded_batch.size(1))
        attention_mask = attention_mask.to(dtype=next(optimized_layer.parameters()).dtype)

        causal_mask = torch.tril(torch.ones(max_length, max_length)).view(1, 1, max_length, max_length).to(device)
        attention_mask = attention_mask * causal_mask
        attention_mask = (1.0 - attention_mask) * -10000.0

        with torch.no_grad():
            hidden_states = embeddings(padded_batch)
            mlp_output = optimized_layer(hidden_states, attention_mask)

        activations = mlp_output.cpu().numpy()
        np.save(f"{data_dir}/activations_{i}.npy", activations)
