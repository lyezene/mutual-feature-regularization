import torch
from tqdm import tqdm
from config import get_device

def generate_toy_data(
    h,
    G,
    num_data_points,
    num_active_features,
    batch_size,
    decay_rate=0.99,
    num_groups=12,
    device=get_device(),
):
    F = torch.randn(h, G, device=device)
    group_size = G // num_groups
    feature_groups = [
        torch.arange(i * group_size, (i + 1) * group_size, device=device)
        for i in range(num_groups)
    ]

    X_batches = []

    for batch_start in tqdm(
        range(0, num_data_points, batch_size), desc="Generating Batches"
    ):
        batch_end = min(batch_start + batch_size, num_data_points)
        current_batch_size = batch_end - batch_start
        coefficients = torch.zeros(current_batch_size, G, device=device)

        for i in range(current_batch_size):
            active_group = torch.randint(num_groups, (1,), device=device).item()
            local_feature_indices = feature_groups[active_group]
            local_feature_probs = torch.pow(
                decay_rate, torch.arange(len(local_feature_indices), device=device)
            )
            local_feature_probs /= local_feature_probs.sum()
            active_features = torch.multinomial(
                local_feature_probs, num_active_features, replacement=False
            )
            coefficients[i, local_feature_indices[active_features]] = torch.rand(
                num_active_features, device=device
            )

        X = torch.mm(coefficients, F.T)
        X_batches.append(X)

    X_generated = torch.cat(X_batches)
    return X_generated, F
