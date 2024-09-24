import seaborn as sns
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils.general_utils import calculate_MMCS, load_specific_run, load_sae, load_true_features_from_run
import os
from models.sae import SparseAutoencoder

def analyze_feature_correlations(models, true_features, device, encoders_to_compare):
    results = []
    all_gt_max_sim, all_max_sim_across_saes, all_frequency_learned = [], [], []
    colors = []

    for i, current_model in enumerate(models):
        for encoder_idx in encoders_to_compare:
            current_encoder_weights = current_model.encoders[encoder_idx].weight.to(device)
            _, ground_truth_sim = calculate_MMCS(current_encoder_weights.t(), true_features, device)
            gt_max_sim = ground_truth_sim.max(dim=1)[0]

            other_sae_sims = [calculate_MMCS(current_encoder_weights.t(), other_model.encoders[other_encoder_idx].weight.t(), device)[1]
                              for j, other_model in enumerate(models) if j != i
                              for other_encoder_idx in encoders_to_compare]

            max_sim_across_saes = torch.zeros_like(gt_max_sim)
            frequency_learned = torch.zeros_like(gt_max_sim)

            for k, other_sim in enumerate(other_sae_sims):
                other_max_sim = other_sim.max(dim=1)[0]
                min_features = min(gt_max_sim.shape[0], other_max_sim.shape[0])
                gt_max_sim_trunc = gt_max_sim[:min_features]
                other_max_sim_trunc = other_max_sim[:min_features]

                max_sim_across_saes[:min_features] = torch.max(max_sim_across_saes[:min_features], other_max_sim_trunc)
                frequency_learned[:min_features] += (other_max_sim_trunc > 0.5).float()

                corr_coef = torch.corrcoef(torch.stack([gt_max_sim_trunc, other_max_sim_trunc]))[0, 1].item()
                results.append({f"Correlation_SAE_{i+1}_Encoder_{encoder_idx+1}_vs_SAE_{k+2}_Encoder_{encoders_to_compare[k%len(encoders_to_compare)]+1}": corr_coef})

            all_gt_max_sim.append(gt_max_sim.detach().cpu().numpy())
            all_max_sim_across_saes.append(max_sim_across_saes.detach().cpu().numpy())
            all_frequency_learned.append(frequency_learned.detach().cpu().numpy())
            colors.extend(['red' if i == 0 else 'blue'] * len(gt_max_sim))

            wandb.log({
                f"SAE_{i+1}_Encoder_{encoder_idx+1}_norm": torch.norm(current_encoder_weights).item(),
                f"SAE_{i+1}_Encoder_{encoder_idx+1}_feature_count": current_encoder_weights.shape[0]
            })

    gt_max_sim = np.concatenate(all_gt_max_sim)
    max_sim_across_saes = np.concatenate(all_max_sim_across_saes)
    frequency_learned = np.concatenate(all_frequency_learned)

    overall_correlation = np.corrcoef([gt_max_sim, max_sim_across_saes, frequency_learned])
    results.append({
        "Overall_Correlation_GT_vs_MaxSim": overall_correlation[0, 1],
        "Overall_Correlation_GT_vs_Frequency": overall_correlation[0, 2],
        "Overall_Correlation_MaxSim_vs_Frequency": overall_correlation[1, 2]
    })

    wandb.log({
        "Max_Similarity_Across_SAEs": np.mean(max_sim_across_saes),
        "Average_Frequency_Learned": np.mean(frequency_learned)
    })

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
    true_features = load_true_features_from_run(specific_run, device)
    
    model_artifact = next(art for art in specific_run.logged_artifacts() if art.type == 'model')
    artifact_dir = model_artifact.download()
    model_path = os.path.join(artifact_dir, f"{specific_run.name}_epoch_1.pth")
    full_state_dict = torch.load(model_path, map_location=device)
    
    models = []
    for encoder_idx in range(2):
        model = SparseAutoencoder(config['hyperparameters'])
        encoder_state_dict = {
            f"encoders.{encoder_idx}.weight": full_state_dict[f'encoders.{encoder_idx}.weight'],
            f"encoders.{encoder_idx}.bias": full_state_dict[f'encoders.{encoder_idx}.bias']
        }
        model.load_state_dict(encoder_state_dict, strict=False)
        models.append(model)
    
    results = analyze_feature_correlations(models, true_features, device, [0, 1])
    for result in results:
        wandb.log(result)
    wandb.log({"experiment_completed": True})
    return results
