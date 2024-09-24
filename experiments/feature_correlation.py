import seaborn as sns
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils.general_utils import calculate_MMCS, load_specific_run, load_sae, load_true_features_from_run
import os
from models.sae import SparseAutoencoder


def analyze_feature_correlations(model, true_features, device, encoders_to_compare):
    results = []
    all_gt_max_sim, all_max_sim_across_saes = [], []
    colors = []

    for i, encoder in enumerate(model.encoders):
        if i not in encoders_to_compare:
            continue

        current_encoder_weights = encoder.weight.to(device)
        mmcs, ground_truth_sim = calculate_MMCS(current_encoder_weights.t(), true_features, device)
        gt_max_sim = ground_truth_sim.max(dim=1)[0]

        print(f"SAE {i}:")
        print(f"  MMCS: {mmcs:.4f}")
        print(f"  GT Max Sim: min={gt_max_sim.min().item():.4f}, max={gt_max_sim.max().item():.4f}, mean={gt_max_sim.mean().item():.4f}")

        other_sae_sims = [calculate_MMCS(current_encoder_weights.t(), other_encoder.weight.t(), device)[1]
                          for j, other_encoder in enumerate(model.encoders) if j != i]

        max_sim_across_saes = torch.stack([other_sim.max(dim=1)[0] for other_sim in other_sae_sims]).max(dim=0)[0]

        print(f"  Max Sim Across SAEs: min={max_sim_across_saes.min().item():.4f}, max={max_sim_across_saes.max().item():.4f}, mean={max_sim_across_saes.mean().item():.4f}")

        min_features = min(gt_max_sim.shape[0], max_sim_across_saes.shape[0])
        all_gt_max_sim.append(gt_max_sim[:min_features].detach().cpu().numpy())
        all_max_sim_across_saes.append(max_sim_across_saes[:min_features].detach().cpu().numpy())
        colors.extend(['red' if i == 0 else 'blue'] * min_features)

        results.append({
            f"SAE_{i+1}_MMCS": mmcs,
            f"SAE_{i+1}_GT_Max_Sim_Min": gt_max_sim.min().item(),
            f"SAE_{i+1}_GT_Max_Sim_Max": gt_max_sim.max().item(),
            f"SAE_{i+1}_GT_Max_Sim_Mean": gt_max_sim.mean().item(),
            f"SAE_{i+1}_Max_Sim_Across_SAEs_Min": max_sim_across_saes.min().item(),
            f"SAE_{i+1}_Max_Sim_Across_SAEs_Max": max_sim_across_saes.max().item(),
            f"SAE_{i+1}_Max_Sim_Across_SAEs_Mean": max_sim_across_saes.mean().item(),
        })

    gt_max_sim = np.concatenate(all_gt_max_sim)
    max_sim_across_saes = np.concatenate(all_max_sim_across_saes)

    colors = np.array(colors)

    plt.figure(figsize=(10, 8))

    categories = np.unique(colors)
    markers = ['o', 's']
    category_labels = ['SAE 1', 'SAE 2']
    colors_map = ['#1f77b4', '#ff7f0e']

    for i, category in enumerate(categories):
        idx = (colors == category)
        plt.scatter(
            gt_max_sim[idx],
            max_sim_across_saes[idx],
            alpha=0.7,
            label=category_labels[i],
            marker=markers[i],
            color=colors_map[i],
            edgecolor='k'
        )

    z = np.polyfit(gt_max_sim, max_sim_across_saes, 1)
    p = np.poly1d(z)
    plt.plot(
        gt_max_sim,
        p(gt_max_sim),
        "k--",
        linewidth=2,
        label='Correlation Line'
    )

    plt.xlabel("Similarity with Input Feature", fontsize=14)
    plt.ylabel("Similarity with SAE Feature", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.linspace(0, 1, 6))
    plt.yticks(np.linspace(0, 1, 6))

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig('sae_similarity_scatter.png', dpi=300)
    plt.close()
 
    wandb.log({"SAE_Similarity_Scatter": wandb.Image('sae_similarity_scatter.png')})

    return results


def run(device, config):
    run_id = config['run_id']
    specific_run = load_specific_run(wandb.run.project, run_id)
    true_features = load_true_features_from_run(specific_run, device).to(device).to(torch.float32)

    model_artifact = next(art for art in specific_run.logged_artifacts() if art.type == 'model')
    artifact_dir = model_artifact.download()
    model_path = os.path.join(artifact_dir, f"{specific_run.name}_epoch_1.pth")
    full_state_dict = torch.load(model_path, map_location=device)

    model = SparseAutoencoder(config['hyperparameters'])
    model.load_state_dict(full_state_dict)
    model = model.to(device).to(torch.float32)

    results = analyze_feature_correlations(model, true_features, device, range(len(model.encoders)))
    wandb.log({"experiment_completed": True})
    return results
