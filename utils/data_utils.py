import os
import torch
from torch.utils.data import IterableDataset, TensorDataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from data.synthetic_dataset import SyntheticIterableDataset
from config import get_device


def generate_synthetic_data(config, device=None):
    device = device or get_device()
    num_features = config.get('num_features', 256)
    num_true_features = config.get('num_ground_features', 512)
    total_data_points = config.get('total_data_points', 100000)
    num_active_features_per_point = config.get('num_active_features_per_point', 42)
    batch_size = config.get('data_batch_size', 1000)
    decay_rate = config.get('decay_rate', 0.99)
    num_feature_groups = config.get('num_feature_groups', 12)
    output_dir = config.get('output_dir', "synthetic_data_batches")

    os.makedirs(output_dir, exist_ok=True)
    true_features = torch.randn(num_features, num_true_features, device=device, dtype=torch.float16)
    group_size = num_true_features // num_feature_groups
    feature_groups = [torch.arange(i * group_size, (i + 1) * group_size, device=device) for i in range(num_feature_groups)]
    group_probs = [torch.pow(decay_rate, torch.arange(group_size, device=device, dtype=torch.float16)) / (1 - decay_rate) for _ in range(num_feature_groups)]

    batches = []
    for batch_start in tqdm(range(0, total_data_points, batch_size), desc="Generating Batches"):
        batch_size = min(batch_size, total_data_points - batch_start)
        coeffs = torch.zeros(batch_size, num_true_features, device=device, dtype=torch.float16)
        selected_groups = torch.randint(num_feature_groups, (batch_size,), device=device)
        
        for i, (group, probs) in enumerate(zip(feature_groups, group_probs)):
            mask = selected_groups == i
            if mask.any():
                indices = group[torch.multinomial(probs, num_active_features_per_point, replacement=False)]
                coeffs[mask.nonzero(as_tuple=True)[0].unsqueeze(1), indices] = torch.rand(mask.sum(), num_active_features_per_point, device=device, dtype=torch.float16)

        batch_data = torch.mm(coeffs, true_features.T)
        batches.append(batch_data.cpu())
        
        del coeffs, batch_data
        torch.cuda.empty_cache()

    return torch.cat(batches), true_features


def load_synthetic_dataset(cache_dir=None, chunk_size=1000, num_epochs=1):
    repo_id = "lukemarks/synthetic_dataset"
    return SyntheticIterableDataset(repo_id, cache_dir, chunk_size, num_epochs)


def load_true_features(cache_dir=None):
    repo_id = "lukemarks/synthetic_dataset"
    cache_dir = cache_dir or os.path.join(os.getcwd(), 'hf_cache')
    file_path = hf_hub_download(repo_id, "data/true_features.pt", repo_type="dataset", cache_dir=cache_dir)
    return torch.load(file_path).float()
