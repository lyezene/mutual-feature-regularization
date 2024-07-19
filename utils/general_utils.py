import torch
import itertools
import numpy as np
from typing import List
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


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


def log_sim_matrices(sim_matrices: List[torch.Tensor]):
    first_sim_matrix = sim_matrices[0]
    sim_matrix_np = first_sim_matrix.detach().cpu().numpy()

    row_max = np.max(sim_matrix_np, axis=1)
    col_max = np.max(sim_matrix_np, axis=0)

    max_similarities = np.concatenate([row_max, col_max])

    plt.figure(figsize=(10, 6))
    plt.hist(max_similarities, bins=50, range=(0, 1), edgecolor='black')
    plt.title("Histogram of Maximum Cosine Similarities (First Encoder)")
    plt.xlabel("Maximum Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()

    wandb.log({"Max_Cosine_Similarities_Histogram_First_Encoder": wandb.Image(plt)})
    plt.close()


def find_combinations(grid):
    keys, values = zip(*grid.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))
