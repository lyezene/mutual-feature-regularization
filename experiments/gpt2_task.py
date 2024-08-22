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
from utils.gpt2_utils import GPT2ActivationsDataset, generate_activations


def run(device: str, config: Dict[str, Any]) -> Dict[str, bool]:
    params = config['hyperparameters']
    data_dir = params.get('data_dir', 'gpt2_activations')
    force_regenerate = params.get('force_regenerate', False)

    if not os.path.exists(data_dir) or force_regenerate:
        print("Generating activations...")
        generate_activations(device, params['num_samples'], params['data_collection_batch_size'], data_dir)
    else:
        print("Using pre-existing activations...")

    wandb.init(project="gpt2_sae", config=params)

    dataset = GPT2ActivationsDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=params['training_batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    model = SparseAutoencoder(params).to(device)
    trainer = SAETrainer(model, device, params, true_features=None)
    trainer.train(dataloader, params['num_epochs'])
    print("training model")

    wandb.finish()
    return {"experiment_completed": True}
