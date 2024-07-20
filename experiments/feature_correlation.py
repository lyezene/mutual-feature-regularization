import torch
from config import get_device
from utils.general_utils import calculate_MMCS
from models.sae import SparseAutoencoder
import wandb
from tqdm import tqdm
import os


def load_sae_model(artifact_path, params, device):
    model = SparseAutoencoder.load_from_pretrained(artifact_path, params, device)
    return model


def analyze_feature_correlations(models, true_features, device):
    results = []
    for i, current_model in enumerate(models):
        current_encoder_weights = current_model.encoders[0].weight.to(device)
        other_decoder_weights = [model.encoders[0].weight.t().to(device) for j, model in enumerate(models) if j != i]

        ground_truth_sim = torch.matmul(current_encoder_weights, true_features.t())
        other_sae_sims = [torch.matmul(current_encoder_weights, decoder_weight) for decoder_weight in other_decoder_weights]

        gt_max_sim = ground_truth_sim.max(dim=1)[0]
        other_max_sims = [sim.max(dim=1)[0] for sim in other_sae_sims]

        for j, other_sim in enumerate(other_max_sims):
            corr_coef = torch.corrcoef(torch.stack([gt_max_sim, other_sim]))[0, 1].item()
            results.append({
                f"Correlation_SAE_{i+1}_vs_SAE_{j+2 if j >= i else j+1}": corr_coef
            })

        wandb.log({
            f"SAE_{i+1}_encoder_norm": torch.norm(current_encoder_weights).item(),
            f"SAE_{i+1}_feature_count": current_encoder_weights.shape[0]
        })

    return results


def run(device, config):
    true_features_path = 'true_features.pt'
    if os.path.exists(true_features_path):
        true_features = torch.load(true_features_path, map_location=device)
    else:
        raise FileNotFoundError("True features file not found. Please generate the synthetic dataset first.")

    artifact_paths = config.get('artifact_paths', [])
    if not artifact_paths:
        raise ValueError("No artifact paths provided in the configuration.")

    models = []
    for path in tqdm(artifact_paths, desc="Loading models"):
        model = load_sae_model(path, config, device)
        models.append(model)

    results = analyze_feature_correlations(models, true_features, device)

    for result in results:
        wandb.log(result)

    wandb.log({"experiment_completed": True})

    return results
