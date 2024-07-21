import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils.general_utils import calculate_MMCS, get_recent_model_runs, load_sae, load_true_features


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

    plt.figure(figsize=(10, 8))
    plt.scatter(gt_max_sim, max_sim_across_saes, alpha=0.5, c=colors)
    plt.xlabel("Max Cosine Similarity with True Features")
    plt.ylabel("Max Cosine Similarity between SAEs")
    plt.title("SAE Feature Similarity: True Features vs Inter-SAE")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    z = np.polyfit(gt_max_sim, max_sim_across_saes, 1)
    corr_line, = plt.plot(gt_max_sim, np.poly1d(z)(gt_max_sim), "k--", alpha=0.8)
    plt.text(0.05, 0.95, f'Correlation: {overall_correlation[0, 1]:.3f}', transform=plt.gca().transAxes)

    plt.legend(handles=[corr_line, plt.scatter([], [], c='red', alpha=0.5), plt.scatter([], [], c='blue', alpha=0.5)],
               labels=['Correlation Line', 'SAE 1', 'SAE 2'],
               loc='lower right')

    plt.savefig('sae_similarity_scatter.png')
    plt.close()
    wandb.log({"SAE_Similarity_Scatter": wandb.Image('sae_similarity_scatter.png')})

    return results


def run(device, config):
    true_features = load_true_features(wandb.run.project, device)
    num_saes = config['hyperparameters'].get('num_saes', 1)
    model_runs = get_recent_model_runs(wandb.run.project, num_saes)
    models = [load_sae(run, config['hyperparameters'], device) for run in model_runs]
    results = analyze_feature_correlations(models, true_features, device, list(range(num_saes)))
    for result in results:
        wandb.log(result)
    wandb.log({"experiment_completed": True})
    return results
