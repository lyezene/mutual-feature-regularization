import torch
import wandb
from tqdm import tqdm
import os
import logging
from models.sae import SparseAutoencoder
from utils.general_utils import calculate_MMCS
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_recent_model_runs(project, num_saes):
    api = wandb.Api()
    runs = api.runs(project)
    model_runs = []
    for run in runs:
        artifacts = run.logged_artifacts()
        if any(art.type == 'model' for art in artifacts):
            model_runs.append(run)
        if len(model_runs) == num_saes:
            break
    return model_runs

def load_sae_model(run, params, device):
    artifacts = run.logged_artifacts()
    model_artifact = next((art for art in artifacts if art.type == 'model'), None)
    if not model_artifact:
        raise ValueError(f"No model artifact found in run {run.name}")
    
    artifact_dir = model_artifact.download()
    logger.info(f"Model artifact downloaded to: {artifact_dir}")
    
    model_file_name = f"{run.name}_epoch_1.pth"
    model_path = os.path.join(artifact_dir, model_file_name)
    
    if os.path.exists(model_path):
        try:
            model = SparseAutoencoder(params)
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Successfully loaded model for run {run.name}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def load_true_features(project, device):
    api = wandb.Api()
    runs = api.runs(project)
    for run in runs:
        artifacts = run.logged_artifacts()
        true_features_artifact = next((art for art in artifacts if art.type == 'true_features'), None)
        if true_features_artifact:
            artifact_dir = true_features_artifact.download()
            true_features_path = os.path.join(artifact_dir, 'true_features.pt')
            if os.path.exists(true_features_path):
                true_features = torch.load(true_features_path, map_location=device)
                logger.info("Successfully loaded true features")
                return true_features
            else:
                logger.error(f"True features file not found at {true_features_path}")
    raise ValueError(f"No valid true_features artifact found in project {project}")

def analyze_feature_correlations(models, true_features, device, encoders_to_compare):
    results = []
    all_gt_max_sim = []
    all_max_sim_across_saes = []
    all_frequency_learned = []
    
    for i, current_model in enumerate(models):
        for encoder_idx in encoders_to_compare:
            current_encoder_weights = current_model.encoders[encoder_idx].weight.to(device)
            logger.info(f"SAE_{i+1}_Encoder_{encoder_idx+1} shape: {current_encoder_weights.shape}")
            
            _, ground_truth_sim = calculate_MMCS(current_encoder_weights.t(), true_features, device)
            logger.info(f"Ground truth similarity matrix shape: {ground_truth_sim.shape}")
            
            gt_max_sim = ground_truth_sim.max(dim=1)[0]
            logger.info(f"Ground truth max similarity shape: {gt_max_sim.shape}")
            
            other_sae_sims = []
            for j, other_model in enumerate(models):
                if j != i:
                    for other_encoder_idx in encoders_to_compare:
                        other_encoder_weights = other_model.encoders[other_encoder_idx].weight.to(device)
                        logger.info(f"SAE_{j+1}_Encoder_{other_encoder_idx+1} shape: {other_encoder_weights.shape}")
                        
                        _, sim_matrix = calculate_MMCS(current_encoder_weights.t(), other_encoder_weights.t(), device)
                        logger.info(f"Similarity matrix shape between SAE_{i+1}_Encoder_{encoder_idx+1} and SAE_{j+1}_Encoder_{other_encoder_idx+1}: {sim_matrix.shape}")
                        
                        other_sae_sims.append(sim_matrix)
            
            max_sim_across_saes = torch.zeros_like(gt_max_sim)
            frequency_learned = torch.zeros_like(gt_max_sim)
            
            for k, other_sim in enumerate(other_sae_sims):
                other_max_sim = other_sim.max(dim=1)[0]
                logger.info(f"Other max similarity shape: {other_max_sim.shape}")
                
                min_features = min(gt_max_sim.shape[0], other_max_sim.shape[0])
                gt_max_sim_trunc = gt_max_sim[:min_features]
                other_max_sim_trunc = other_max_sim[:min_features]
                
                max_sim_across_saes[:min_features] = torch.max(max_sim_across_saes[:min_features], other_max_sim_trunc)
                frequency_learned[:min_features] += (other_max_sim_trunc > 0.5).float()
                
                corr_coef = torch.corrcoef(torch.stack([gt_max_sim_trunc, other_max_sim_trunc]))[0, 1].item()
                results.append({
                    f"Correlation_SAE_{i+1}_Encoder_{encoder_idx+1}_vs_SAE_{k+2}_Encoder_{encoders_to_compare[k%len(encoders_to_compare)]+1}": corr_coef
                })
            
            all_gt_max_sim.append(gt_max_sim.detach().cpu().numpy())
            all_max_sim_across_saes.append(max_sim_across_saes.detach().cpu().numpy())
            all_frequency_learned.append(frequency_learned.detach().cpu().numpy())
            
            wandb.log({
                f"SAE_{i+1}_Encoder_{encoder_idx+1}_norm": torch.norm(current_encoder_weights).item(),
                f"SAE_{i+1}_Encoder_{encoder_idx+1}_feature_count": current_encoder_weights.shape[0]
            })
    
    gt_max_sim = np.concatenate(all_gt_max_sim)
    max_sim_across_saes = np.concatenate(all_max_sim_across_saes)
    frequency_learned = np.concatenate(all_frequency_learned)
    
    logger.info(f"Shape of gt_max_sim after concatenation: {gt_max_sim.shape}")
    logger.info(f"Shape of max_sim_across_saes after concatenation: {max_sim_across_saes.shape}")
    logger.info(f"Shape of frequency_learned after concatenation: {frequency_learned.shape}")
    
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
    
    colors = []
    for i, model in enumerate(models):
        for _ in encoders_to_compare:
            if i == 0:  # First SAE will be red
                colors.extend(['red'] * len(all_gt_max_sim[i]))
            else:  # All other SAEs will be blue
                colors.extend(['blue'] * len(all_gt_max_sim[i]))

    plt.figure(figsize=(10, 8))
    plt.scatter(gt_max_sim, max_sim_across_saes, alpha=0.5, c=colors)
    plt.xlabel("Max Cosine Similarity with True Features")
    plt.ylabel("Max Cosine Similarity between SAEs")
    plt.title("SAE Feature Similarity: True Features vs Inter-SAE")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    z = np.polyfit(gt_max_sim, max_sim_across_saes, 1)
    p = np.poly1d(z)
    plt.plot(gt_max_sim, p(gt_max_sim), "k--", alpha=0.8)  # Changed to black dashed line
    plt.text(0.05, 0.95, f'Correlation: {overall_correlation[0, 1]:.3f}', transform=plt.gca().transAxes)

    # Add legend
    plt.legend(['Correlation Line', 'SAE 1', 'Other SAEs'], loc='lower right')

    plt.savefig('sae_similarity_scatter.png')
    plt.close()
    wandb.log({"SAE_Similarity_Scatter": wandb.Image('sae_similarity_scatter.png')})

    return results

def run(device, config):
    project = wandb.run.project
    logger.info(f"Attempting to load true features from project: {project}")
    true_features = load_true_features(project, device)

    num_saes = config['hyperparameters'].get('num_saes', 1)
    logger.info(f"Loading {num_saes} recent model runs")
    model_runs = get_recent_model_runs(project, num_saes)
    
    if not model_runs:
        raise ValueError("No runs with model artifacts found in the project.")

    models = []
    for run in tqdm(model_runs, desc="Loading models"):
        try:
            model = load_sae_model(run, config['hyperparameters'], device)
            models.append(model)
        except Exception as e:
            logger.error(f"Failed to load model for run {run.name}: {str(e)}")

    if not models:
        raise ValueError("No models could be loaded successfully.")

    encoders_to_compare = list(range(num_saes))  # Compare all encoders
    logger.info(f"Analyzing feature correlations for {len(models)} models and {len(encoders_to_compare)} encoders")
    results = analyze_feature_correlations(models, true_features, device, encoders_to_compare)

    for result in results:
        wandb.log(result)

    wandb.log({"experiment_completed": True})
    logger.info("Feature correlation analysis completed")

    return results
