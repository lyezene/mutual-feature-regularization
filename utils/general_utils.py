import torch

def calculate_MMCS(learned_features, ground_truth_features, device):
    if not isinstance(ground_truth_features, torch.Tensor):
        ground_truth_features = torch.tensor(ground_truth_features, dtype=torch.float32)

    if learned_features.shape[0] != ground_truth_features.shape[0]:
        learned_features = learned_features.t()
        ground_truth_features = ground_truth_features.t()

    learned_features = learned_features.to(device).float()
    ground_truth_features = ground_truth_features.to(device).float()

    learned_norm = torch.nn.functional.normalize(learned_features, p=2, dim=0)
    ground_truth_norm = torch.nn.functional.normalize(ground_truth_features, p=2, dim=0)

    cos_sims = torch.matmul(learned_norm.t(), ground_truth_norm)
    max_cosine_similarities = torch.max(cos_sims, dim=0).values

    mmcs = torch.mean(max_cosine_similarities).item()

    return mmcs, cos_sims
