import itertools
from models.sae import SparseAutoencoder, SAETrainer
from tqdm import tqdm


def train_synthetic_sae(params, true_features, train_loader):
    data_sample = next(iter(train_loader))
    model = SparseAutoencoder(params, data_sample)
    total_samples = len(train_loader) * params["num_epochs"]
    progress_bar = tqdm(total=total_samples, desc="Training SAE")

    trainer = SAETrainer(model, params, true_features)
    
    losses, mmcs_scores, cos_sim_matrices = trainer.train(train_loader, params["num_epochs"], progress_bar)
    
    progress_bar.close()
    return losses, mmcs_scores, cos_sim_matrices


def find_combinations(grid):
    keys, values = zip(*grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))
