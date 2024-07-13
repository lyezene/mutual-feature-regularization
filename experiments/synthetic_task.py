import torch
from config import get_device
import itertools
from utils.synthetic_utils import find_combinations, train_synthetic_sae
from utils.data_utils import load_synthetic_dataset, load_true_features
import wandb
from torch.utils.data import DataLoader
import os


def run(device, config):
    cache_dir = os.path.join(os.getcwd(), 'hf_cache')
    os.makedirs(cache_dir, exist_ok=True)

    chunk_size = 5000000
    num_epochs = config['synthetic_epochs']
    train_dataset = load_synthetic_dataset(cache_dir, chunk_size, num_epochs)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training_batch_size'], num_workers=2)
    
    true_features = load_true_features(cache_dir)

    parameter_grid = {
        'learning_rate': config['learning_rate'],
        'input_size': config['input_size'],
        'l1_coef': config['l1_coef'],
        'num_epochs': config['synthetic_epochs'],
        'hidden_size': config['hidden_size'],
        'ar': config['ar'],
        'beta': config['beta'],
        'num_saes': config['num_saes'],
        'property': config['property'],
        'accumulation_steps': config['accumulation_steps'],
    }

    for key, value in parameter_grid.items():
        if not isinstance(value, list):
            parameter_grid[key] = [value]

    results = []
    for params in find_combinations(parameter_grid):
        epoch_losses, mmcs_scores, cos_sim_matrices = train_synthetic_sae(params, true_features, train_loader)
        results.append((params, epoch_losses, mmcs_scores, cos_sim_matrices))

