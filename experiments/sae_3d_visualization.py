import os
from models.sae import SparseAutoencoder
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
from torch.utils.data import DataLoader, TensorDataset
from utils.general_utils import calculate_MMCS, load_specific_run, load_true_features_from_run
from utils.data_utils import generate_synthetic_data
from config import get_device
import traceback


def calculate_topk_activation_probability(model, data_loader, device):
    model.eval()
    activation_counts = [torch.zeros(encoder.weight.shape[0], device=device) for encoder in model.encoders]
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            x = x.to(torch.float32)
            batch_size = x.shape[0]
            total_samples += batch_size

            for encoder_idx, encoder in enumerate(model.encoders):
                encoded = encoder(x)
                topk_encoded = model._topk_activation(encoded, encoder_idx)
                activation_counts[encoder_idx] += (topk_encoded != 0).float().sum(dim=0)

    return [counts / total_samples for counts in activation_counts]


def visualize_sae_features_3d(models, true_features, data_loader, device, encoders_to_compare):
    all_gt_max_sim, all_max_sim_across_saes, all_activation_probabilities = [], [], []
    colors = []

    activation_probabilities = calculate_topk_activation_probability(models[0], data_loader, device)

    for i, encoder in enumerate(models[0].encoders):
        current_encoder_weights = encoder.weight.to(device)
        mmcs, ground_truth_sim = calculate_MMCS(current_encoder_weights.t(), true_features, device)
        gt_max_sim = ground_truth_sim.max(dim=1)[0]

        print(f"SAE {i}:")
        print(f"  MMCS: {mmcs:.4f}")
        print(f"  GT Max Sim: min={gt_max_sim.min().item():.4f}, max={gt_max_sim.max().item():.4f}, mean={gt_max_sim.mean().item():.4f}")

        other_sae_sims = [calculate_MMCS(current_encoder_weights.t(), other_encoder.weight.t(), device)[1]
                          for j, other_encoder in enumerate(models[0].encoders) if j != i]

        max_sim_across_saes = torch.stack([other_sim.max(dim=1)[0] for other_sim in other_sae_sims]).max(dim=0)[0]

        print(f"  Max Sim Across SAEs: min={max_sim_across_saes.min().item():.4f}, max={max_sim_across_saes.max().item():.4f}, mean={max_sim_across_saes.mean().item():.4f}")

        min_features = min(gt_max_sim.shape[0], max_sim_across_saes.shape[0])
        all_gt_max_sim.append(gt_max_sim[:min_features].detach().cpu().numpy())
        all_max_sim_across_saes.append(max_sim_across_saes[:min_features].detach().cpu().numpy())
        all_activation_probabilities.append(activation_probabilities[i][:min_features].detach().cpu().numpy())
        colors.extend(['red' if i == 0 else 'blue'] * min_features)

        log_activation_stats(i, 0, activation_probabilities[i])

    return process_and_visualize_data(all_gt_max_sim, all_max_sim_across_saes, all_activation_probabilities, colors)


def log_activation_stats(sae_idx, encoder_idx, activation_probs):
    wandb.log({
        f"SAE_{sae_idx+1}_Encoder_{encoder_idx+1}_avg_activation_prob": activation_probs.mean().item(),
        f"SAE_{sae_idx+1}_Encoder_{encoder_idx+1}_max_activation_prob": activation_probs.max().item(),
        f"SAE_{sae_idx+1}_Encoder_{encoder_idx+1}_min_activation_prob": activation_probs.min().item(),
    })


def process_and_visualize_data(all_gt_max_sim, all_max_sim_across_saes, all_activation_probabilities, colors):
    gt_max_sim = np.concatenate(all_gt_max_sim)
    max_sim_across_saes = np.concatenate(all_max_sim_across_saes)
    activation_probabilities = np.concatenate(all_activation_probabilities)

    create_3d_plot(gt_max_sim, max_sim_across_saes, activation_probabilities, colors, 'SAE Feature 3D Visualization (All Data)', '3d_feature_analysis_all.png')

    outlier_threshold = np.percentile(activation_probabilities, 99)
    mask = activation_probabilities < outlier_threshold
    create_3d_plot(gt_max_sim[mask], max_sim_across_saes[mask], activation_probabilities[mask], np.array(colors)[mask], 
                   'SAE Feature 3D Visualization (Without Outliers)', '3d_feature_analysis_no_outliers.png')

    create_activation_histogram(activation_probabilities)

    log_wandb_results(gt_max_sim, max_sim_across_saes, activation_probabilities)

    return gt_max_sim, max_sim_across_saes, activation_probabilities


def create_3d_plot(x, y, z, colors, title, filename):
    colors = np.array(colors)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    categories = np.unique(colors)
    markers = ['o', 's']
    category_labels = ['SAE 1', 'SAE 2']
    colors_map = ['#1f77b4', '#ff7f0e']

    for i, category in enumerate(categories):
        idx = (colors == category)
        ax.scatter(
            x[idx],
            y[idx],
            z[idx],
            alpha=0.7,
            label=category_labels[i],
            marker=markers[i],
            color=colors_map[i],
            edgecolor='k',
            s=50
        )

    ax.set_xlabel('Similarity with True Feature', fontsize=14, labelpad=10)
    ax.set_ylabel('Similarity with SAE feature', fontsize=14, labelpad=10)
    ax.set_zlabel('Activation Probability', fontsize=14, labelpad=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.view_init(elev=20, azim=120)
    legend = ax.legend(loc='best', fontsize=12)

    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close(fig)


def create_activation_histogram(activation_probabilities):
    plt.figure(figsize=(10, 6))
    plt.hist(activation_probabilities, bins=50, alpha=0.7)
    plt.xlabel('Activation Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Feature Activation Probabilities')
    plt.savefig('activation_probability_histogram.png')
    plt.close()


def log_wandb_results(gt_max_sim, max_sim_across_saes, activation_probabilities):
    wandb.log({
        "3D_Feature_Analysis_All": wandb.Image('3d_feature_analysis_all.png'),
        "3D_Feature_Analysis_No_Outliers": wandb.Image('3d_feature_analysis_no_outliers.png'),
        "Activation_Probability_Histogram": wandb.Image('activation_probability_histogram.png'),
        "Avg_GT_Similarity": np.mean(gt_max_sim),
        "Avg_SAE_Similarity": np.mean(max_sim_across_saes),
        "Avg_Activation_Probability": np.mean(activation_probabilities),
        "Max_Activation_Probability": np.max(activation_probabilities),
        "Min_Activation_Probability": np.min(activation_probabilities),
    })


def create_data_loader(config, true_features=None):
    device = get_device()
    data, true_features = generate_synthetic_data(config, true_features, device)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=2048, shuffle=True), true_features


def run(device, config):
    torch.set_default_dtype(torch.float32)
    run_id = config['run_id']
    specific_run = load_specific_run(wandb.run.project, run_id)
    true_features = load_true_features_from_run(specific_run, device).to(device).to(torch.float32)

    # Load the full model state dict
    model_artifact = next(art for art in specific_run.logged_artifacts() if art.type == 'model')
    artifact_dir = model_artifact.download()
    model_path = os.path.join(artifact_dir, f"{specific_run.name}_epoch_1.pth")
    full_state_dict = torch.load(model_path, map_location=device)

    # Create a single model with both encoders
    model = SparseAutoencoder(config['hyperparameters'])
    model.load_state_dict(full_state_dict)
    model = model.to(device).to(torch.float32)

    data_loader, _ = create_data_loader(config, true_features)

    try:
        results = visualize_sae_features_3d([model], true_features, data_loader, device, range(len(model.encoders)))
        wandb.log({"experiment_completed": True})
        return {
            "gt_max_sim": results[0],
            "max_sim_across_saes": results[1],
            "activation_probabilities": results[2]
        }
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

