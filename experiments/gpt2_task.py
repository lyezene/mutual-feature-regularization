import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from utils.gpt2_utils import GPT2ActivationsDataset, generate_activations
from utils.sae_trainer import SAETrainer
from models.sae import SparseAutoencoder
from typing import Dict, Any


def stack_collate_fn(batch):
    tensors = [item[0] for item in batch]
    stacked_batch = torch.stack(tensors, dim=0)
    return (stacked_batch,)


def run(device, config):
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
    dataloader = DataLoader(dataset, batch_size=params['training_batch_size'], num_workers=6, pin_memory=True, collate_fn=stack_collate_fn)

    model = SparseAutoencoder(params).to(device)

    trainer = SAETrainer(model, device, params, true_features=None)
    trainer.train(dataloader, params['num_epochs'])

    print("Training model completed")
    wandb.finish()
