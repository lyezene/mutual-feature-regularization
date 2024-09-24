import torch
import itertools
import numpy as np
from typing import List
import wandb
import matplotlib.pyplot as plt
import wandb
import os
from models.sae import SparseAutoencoder
from scipy.optimize import linear_sum_assignment


def calculate_MMCS(learned_features, true_features, device):
    if true_features is None:
        return None, None

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

    cost_matrix = 1 - cos_sim_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    max_cos_sims = cos_sim_matrix[row_ind, col_ind]

    mmcs = torch.mean(max_cos_sims)

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


def load_specific_run(project, run_id):
    api = wandb.Api()
    return api.run(f"{project}/{run_id}")


def load_sae(run, params, device, encoder_idx):
    model_artifact = next(art for art in run.logged_artifacts() if art.type == 'model')
    artifact_dir = model_artifact.download()
    model_path = os.path.join(artifact_dir, f"{run.name}_epoch_1.pth")
    model = SparseAutoencoder(params)
    state_dict = torch.load(model_path, map_location=device)
    
    encoder_state_dict = {
        f"encoders.{encoder_idx}.weight": state_dict['weight'],
        f"encoders.{encoder_idx}.bias": state_dict['bias']
    }
    
    model.load_state_dict(encoder_state_dict, strict=False)
    return model


def load_true_features_from_run(run, device):
    true_features_artifact = next((art for art in run.logged_artifacts() if art.type == 'true_features'), None)
    if true_features_artifact:
        artifact_dir = true_features_artifact.download()
        return torch.load(os.path.join(artifact_dir, 'true_features.pt'), map_location=device)
    raise ValueError(f"No valid true_features artifact found in run {run.id}")


def get_recent_model_runs(project, num_saes):
    api = wandb.Api()
    return [run for run in api.runs(project) if any(art.type == 'model' for art in run.logged_artifacts())][:num_saes]


def load_true_features(project, device):
    api = wandb.Api()
    for run in api.runs(project):
        true_features_artifact = next((art for art in run.logged_artifacts() if art.type == 'true_features'), None)
        if true_features_artifact:
            artifact_dir = true_features_artifact.download()
            return torch.load(os.path.join(artifact_dir, 'true_features.pt'), map_location=device)
    raise ValueError(f"No valid true_features artifact found in project {project}")
