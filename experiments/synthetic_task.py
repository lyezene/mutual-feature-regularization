import torch
from config import get_device
import itertools
from utils.sae_trainer import SAETrainer
from utils.data_utils import load_synthetic_dataset, load_true_features
from utils.general_utils import find_combinations
from torch.utils.data import DataLoader
import os
from models.sae import SparseAutoencoder
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


def run(device, config):
    cache_dir = os.path.join(os.getcwd(), 'hf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    train_dataset = load_synthetic_dataset(cache_dir, config['chunk_size'], config['num_epochs'])
    train_loader = DataLoader(train_dataset, batch_size=config['training_batch_size'], num_workers=2)
    true_features = load_true_features(cache_dir)
    
    parameter_grid = {k: [v] if not isinstance(v, list) else v for k, v in config.items() 
                      if k in ['learning_rate', 'input_size', 'k_sparse', 'num_epochs', 'hidden_size', 
                               'ar', 'beta', 'num_saes', 'property', 'accumulation_steps']}
    
    return [train_synthetic_sae(params, true_features, train_loader) 
            for params in find_combinations(parameter_grid)]
