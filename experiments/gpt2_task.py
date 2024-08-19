from torch.cuda.amp import autocast
import os
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
from utils.general_utils import calculate_MMCS, get_recent_model_runs, load_sae
from utils.sae_trainer import SAETrainer
from models.sae import SparseAutoencoder
import numpy as np


class GPT2ActivationsDataset(Dataset):
    def __init__(self, layer_index, num_samples=100000, batch_size=1000, device='cuda', data_dir='gpt2_activations'):
        self.layer_index = layer_index
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.dataset = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train", streaming=True)
        self.gpt2 = GPT2Model.from_pretrained("gpt2", config=GPT2Config.from_pretrained("gpt2", output_hidden_states=True))
        self.gpt2.to(self.device).eval()

        self.collect_and_save_activations()
        self.activation_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npy')])
        self.total_samples = sum(np.load(f"{self.data_dir}/{f}").shape[0] for f in self.activation_files)

    def collect_and_save_activations(self):
        dataset_iter = iter(self.dataset.shuffle(seed=42).take(self.num_samples))

        for i in tqdm(range(0, self.num_samples, self.batch_size), desc="Collecting activations"):
            batch = [next(dataset_iter)["input_ids"] for _ in range(min(self.batch_size, self.num_samples - i))]
            max_length = max(len(seq) for seq in batch)
            padded_batch = [seq + [0] * (max_length - len(seq)) for seq in batch]

            with torch.no_grad():
                inputs = torch.tensor(padded_batch).to(self.device)
                outputs = self.gpt2(input_ids=inputs, output_hidden_states=True)
                layer_output = outputs.hidden_states[self.layer_index + 1]
                mlp_output = self.gpt2.h[self.layer_index].mlp(layer_output)

            np.save(f"{self.data_dir}/activations_{i}.npy", mlp_output.cpu().numpy())

            del inputs, outputs, layer_output, mlp_output
            torch.cuda.empty_cache()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // self.batch_size
        item_idx = idx % self.batch_size
        activations = np.load(f"{self.data_dir}/{self.activation_files[file_idx]}")
        if item_idx >= activations.shape[0]:
            file_idx = (file_idx + 1) % len(self.activation_files)
            item_idx = 0
            activations = np.load(f"{self.data_dir}/{self.activation_files[file_idx]}")
        return (torch.tensor(activations[item_idx], dtype=torch.float32),)


def run(device, config):
    params = config['hyperparameters']

    wandb.init(project="gpt2_sae", config=params)

    dataset = GPT2ActivationsDataset(
        layer_index=0,
        num_samples=params['num_samples'],
        batch_size=params['data_collection_batch_size'],
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=params['training_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    model = SparseAutoencoder(params).to(device)
    trainer = SAETrainer(model, device, params, true_features=None)

    trainer.train(dataloader, params['num_epochs'])

    wandb.finish()

    return {"experiment_completed": True}
