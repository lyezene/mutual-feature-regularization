import os
import torch
from torch.utils.data import IterableDataset, TensorDataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from config import get_device


def generate_synthetic_data(config, true_features=None, device=None):
    device = device or get_device()
    num_features = config.get('num_features', 256)
    num_true_features = config.get('num_ground_features', 512)
    total_data_points = config.get('total_data_points', 10000)
    num_active_features_per_point = config.get('num_active_features_per_point', 32)
    batch_size = config.get('data_batch_size', 10000)
    decay_rate = config.get('decay_rate', 0.99)
    num_feature_groups = config.get('num_feature_groups', 12)
    output_dir = config.get('output_dir', "synthetic_data_batches")
    os.makedirs(output_dir, exist_ok=True)
    
    if true_features is None:
        true_features = torch.randn(num_features, num_true_features, device=device, dtype=torch.float16)
    else:
        true_features = true_features.to(device=device, dtype=torch.float16)

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
