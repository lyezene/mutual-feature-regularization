import torch
from models.sae import SparseAutoencoder, SAETrainer
from config import get_device
from data.synthetic_dataset import generate_toy_data
import itertools

def train_synthetic_sae(params, Fe, train_loader):
    model = SparseAutoencoder(params)
    trainer = SAETrainer(model, params, Fe)
    losses, mmcs_scores, similarity_matrices = trainer.train(
        train_loader, params["num_epochs"]
    )
    return losses, mmcs_scores, similarity_matrices


def find_combinations(grid):
    keys, values = zip(*grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))


def run(device, config):
    h = config.get('h', 256)
    G = config.get('G', 512)
    num_data_points = config.get('num_data_points', 1000)
    num_active_features = config.get('num_active_features', 42)
    batch_size = config.get('batch_size', 100)

    X, Fe = generate_toy_data(h, G, num_data_points, num_active_features, batch_size, device=device)

    train_dataset = torch.utils.data.TensorDataset(X)
    torch.save(train_dataset, 'synthetic_dataset.pth')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True)

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
        epoch_losses, mmcs_scores, similarity_matrices = train_synthetic_sae(params, Fe, train_loader)
        results.append((params, epoch_losses, mmcs_scores, similarity_matrices))
