import torch
from tqdm import tqdm
from config import get_device

def generate_synthetic_data(
    num_features,
    num_true_features,
    total_data_points,
    num_active_features_per_point,
    batch_size,
    decay_rate=0.99,
    num_feature_groups=12,
    device=get_device(),
):
    true_features = torch.randn(num_features, num_true_features, device=device)
    group_size = num_true_features // num_feature_groups
    feature_group_indices = [
        torch.arange(i * group_size, (i + 1) * group_size, device=device)
        for i in range(num_feature_groups)
    ]

    data_batches = []

    for batch_start in tqdm(
        range(0, total_data_points, batch_size), desc="Generating Batches"
    ):
        batch_end = min(batch_start + batch_size, total_data_points)
        current_batch_size = batch_end - batch_start
        batch_coefficients = torch.zeros(current_batch_size, num_true_features, device=device)

        for i in range(current_batch_size):
            selected_group = torch.randint(num_feature_groups, (1,), device=device).item()
            selected_group_indices = feature_group_indices[selected_group]
            group_feature_probs = torch.pow(
                decay_rate, torch.arange(len(selected_group_indices), device=device)
            )
            group_feature_probs /= group_feature_probs.sum()
            selected_features = torch.multinomial(
                group_feature_probs, num_active_features_per_point, replacement=False
            )
            batch_coefficients[i, selected_group_indices[selected_features]] = torch.rand(
                num_active_features_per_point, device=device
            )

        batch_data = torch.mm(batch_coefficients, true_features.T)
        data_batches.append(batch_data)

    generated_data = torch.cat(data_batches)
    return generated_data, true_features
