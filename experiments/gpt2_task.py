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
from typing import List, Tuple, Dict, Any


class OptimizedGPT2Layer(torch.nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = GPT2Model(config).h[0].ln_1
        self.attn = GPT2Model(config).h[0].attn
        self.ln_2 = GPT2Model(config).h[0].ln_2
        self.mlp = GPT2Model(config).h[0].mlp

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        return mlp_output


class OptimizedGPT2ActivationsDataset(Dataset):
    def __init__(self, num_samples: int = 100000, batch_size: int = 1000, device: str = 'cuda', data_dir: str = 'gpt2_activations'):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.dataset = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train", streaming=True)
        self.config = GPT2Config.from_pretrained("gpt2")
        self.embeddings = GPT2Model.from_pretrained("gpt2").wte.to(device)
        self.optimized_layer = OptimizedGPT2Layer(self.config).to(device)
        
        self.collect_and_save_activations()
        self.activation_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npy')])
        self.total_samples = sum(np.load(f"{self.data_dir}/{f}").shape[0] for f in self.activation_files)

    @torch.no_grad()
    def collect_and_save_activations(self) -> None:
        dataset_iter = iter(self.dataset.shuffle(seed=42).take(self.num_samples))
        for i in tqdm(range(0, self.num_samples, self.batch_size), desc="Collecting activations"):
            batch = [next(dataset_iter)["input_ids"] for _ in range(min(self.batch_size, self.num_samples - i))]
            max_length = max(len(seq) for seq in batch)
            padded_batch = torch.tensor([seq + [0] * (max_length - len(seq)) for seq in batch], dtype=torch.long).to(self.device)
            
            attention_mask = (padded_batch != 0).to(torch.float32).to(self.device)
            
            # Adjust attention mask for GPT2 causal attention
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(padded_batch.size(0), 1, padded_batch.size(1), padded_batch.size(1))
            attention_mask = attention_mask.to(dtype=next(self.optimized_layer.parameters()).dtype)  # Match dtype of model
            
            # Create causal mask
            causal_mask = torch.tril(torch.ones(max_length, max_length)).view(1, 1, max_length, max_length).to(self.device)
            attention_mask = attention_mask * causal_mask
            
            # Adjust values for masked positions
            attention_mask = (1.0 - attention_mask) * -10000.0
            
            hidden_states = self.embeddings(padded_batch)
            mlp_output = self.optimized_layer(hidden_states, attention_mask)
            
            activations = mlp_output.cpu().numpy()
            np.save(f"{self.data_dir}/activations_{i}.npy", activations)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = idx // self.batch_size
        item_idx = idx % self.batch_size
        activations = np.load(f"{self.data_dir}/{self.activation_files[file_idx]}")
        if item_idx >= activations.shape[0]:
            file_idx = (file_idx + 1) % len(self.activation_files)
            item_idx = 0
            activations = np.load(f"{self.data_dir}/{self.activation_files[file_idx]}")
        return torch.tensor(activations[item_idx], dtype=torch.float32)


def run(device: str, config: Dict[str, Any]) -> Dict[str, bool]:
    params = config['hyperparameters']
    wandb.init(project="gpt2_sae", config=params)
    dataset = OptimizedGPT2ActivationsDataset(
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
