import torch
from tqdm import tqdm
from config import get_device
import os

def generate_synthetic_data(
    num_features,
    num_true_features,
    total_data_points,
    num_active_features_per_point,
    batch_size,
    decay_rate=0.99,
    num_feature_groups=12,
    device=get_device(),
    output_dir="synthetic_data_batches"
):
    os.makedirs(output_dir, exist_ok=True)
    true_features = torch.randn(num_features, num_true_features, device=device, dtype=torch.float16)
    group_size = num_true_features // num_feature_groups
    feature_group_indices = [
        torch.arange(i * group_size, (i + 1) * group_size, device=device)
        for i in range(num_feature_groups)
    ]

    group_feature_probs = [
        torch.pow(decay_rate, torch.arange(group_size, device=device, dtype=torch.float16))
        for _ in range(num_feature_groups)
    ]
    for probs in group_feature_probs:
        probs /= probs.sum()

    batch_files = []

    for batch_start in tqdm(
        range(0, total_data_points, batch_size), desc="Generating Batches"
    ):
        batch_end = min(batch_start + batch_size, total_data_points)
        current_batch_size = batch_end - batch_start
        batch_coefficients = torch.zeros(current_batch_size, num_true_features, device=device, dtype=torch.float16)

        selected_groups = torch.randint(num_feature_groups, (current_batch_size,), device=device)
        for i in range(num_feature_groups):
            mask = selected_groups == i
            if mask.any():
                selected_group_indices = feature_group_indices[i]
                selected_probs = group_feature_probs[i]
                selected_features = torch.multinomial(
                    selected_probs, num_active_features_per_point, replacement=False
                )
                indices = selected_group_indices[selected_features]
                batch_coefficients[mask.nonzero(as_tuple=True)[0].unsqueeze(1), indices] = torch.rand(
                    mask.sum(), num_active_features_per_point, device=device, dtype=torch.float16
                )

        batch_data = torch.mm(batch_coefficients, true_features.T)

        batch_file = os.path.join(output_dir, f"batch_{batch_start}.pt")
        torch.save(batch_data.cpu(), batch_file)
        batch_files.append(batch_file)

        del batch_coefficients, batch_data
        torch.cuda.empty_cache()

    data_batches = [torch.load(batch_file) for batch_file in batch_files]
    generated_data = torch.cat(data_batches)
    return generated_data, true_features
