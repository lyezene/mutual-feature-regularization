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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size, config: Dict[str, Any]):
    setup(rank, world_size)
    
    params = config['hyperparameters']
    data_dir = params.get('data_dir', 'gpt2_activations')
    force_regenerate = params.get('force_regenerate', False)

    if rank == 0:
        if not os.path.exists(data_dir) or force_regenerate:
            print("Generating activations...")
            generate_activations(f"cuda:{rank}", params['num_samples'], params['data_collection_batch_size'], data_dir)
        else:
            print("Using pre-existing activations...")

    dist.barrier()

    if rank == 0:
        wandb.init(project="gpt2_sae", config=params)

    dataset = GPT2ActivationsDataset(data_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=params['training_batch_size'], sampler=sampler, num_workers=8, pin_memory=True, collate_fn=stack_collate_fn)

    model = SparseAutoencoder(params).to(rank)
    model = DDP(model, device_ids=[rank])
    
    trainer = SAETrainer(model, f"cuda:{rank}", params, true_features=None)
    trainer.train(dataloader, params['num_epochs'])

    if rank == 0:
        print("Training model completed")
        wandb.finish()

    cleanup()
