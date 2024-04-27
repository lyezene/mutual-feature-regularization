import torch


def calculate_MMCS(learned_features, true_features, device):
    if not isinstance(true_features, torch.Tensor):
        true_features = torch.tensor(true_features, dtype=torch.float32)

    if learned_features.shape[0] != true_features.shape[0]:
        learned_features = learned_features.t()
        true_features = true_features.t()

    learned_features = learned_features.to(device).float()
    true_features = true_features.to(device).float()

    learned_norm = torch.nn.functional.normalize(learned_features, p=2, dim=0)
    true_norm = torch.nn.functional.normalize(true_features, p=2, dim=0)

    cos_sim_matrix = torch.matmul(learned_norm.t(), true_norm)
    max_cos_sims = torch.max(cos_sim_matrix, dim=0).values

    mmcs = torch.mean(max_cos_sims).item()

    return mmcs, cos_sim_matrix
