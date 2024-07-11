import torch
from config import get_device
from data.synthetic_dataset import generate_synthetic_data
import itertools
from utils.synthetic_utils import find_combinations, train_synthetic_sae
from utils.data_utils import generate_synthetic_dataset, load_synthetic_dataset, load_true_features
import wandb
from torch.utils.data import DataLoader


def run(device, config):
    train_dataset = load_synthetic_dataset()
    train_loader = DataLoader(train_dataset, batch_size=config.get('training_batch_size'), num_workers=4)

    true_features = load_true_features()

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
    }

    for key, value in parameter_grid.items():
        if not isinstance(value, list):
            parameter_grid[key] = [value]

    results = []
    for params in find_combinations(parameter_grid):
        epoch_losses, mmcs_scores, cos_sim_matrices = train_synthetic_sae(params, true_features, train_loader)
        results.append((params, epoch_losses, mmcs_scores, cos_sim_matrices))

