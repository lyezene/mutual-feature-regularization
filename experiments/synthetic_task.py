import torch
from config import get_device
from data.synthetic_dataset import generate_synthetic_data
import itertools
from utils.synthetic_utils import find_combinations, train_synthetic_sae


def run(device, config):
    num_features = config.get('num_features', 256)
    num_true_features = config.get('num_ground_features', 512)
    total_data_points = config.get('total_data_points', 1000)
    num_active_features_per_point = config.get('num_active_features_per_point', 42)
    batch_size = config.get('data_batch_size', 100)

    generated_data, true_features = generate_synthetic_data(
        num_features,
        num_true_features,
        total_data_points,
        num_active_features_per_point,
        batch_size,
        device=device
    )

    train_dataset = torch.utils.data.TensorDataset(generated_data)
    torch.save(train_dataset, 'synthetic_dataset.pth')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.get('training_batch_size'), shuffle=True)

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
