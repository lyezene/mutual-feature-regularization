import itertools
from models.sae import SparseAutoencoder, SAETrainer


def train_synthetic_sae(params, true_features, train_loader):
    model = SparseAutoencoder(params)
    trainer = SAETrainer(model, params, true_features)
    losses, mmcs_scores, cos_sim_matrices = trainer.train(
        train_loader, params["num_epochs"]
    )
    return losses, mmcs_scores, cos_sim_matrices


def find_combinations(grid):
    keys, values = zip(*grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))
