import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
from torch.utils.data import DataLoader, TensorDataset
from utils.general_utils import calculate_MMCS, get_recent_model_runs, load_sae, load_true_features
from utils.data_utils import generate_synthetic_data
from config import get_device

def calculate_topk_activation_probability(model, data_loader, device):
    model.eval()
    activation_counts = [torch.zeros(encoder.weight.shape[0]).to(device) for encoder in model.encoders]
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            elif isinstance(batch, torch.Tensor):
                x = batch.to(device)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")
            
            x = x.to(torch.float32)
            batch_size = x.shape[0]
            total_samples += batch_size

            # Calculate activation probability for each encoder separately
            for encoder_idx, encoder in enumerate(model.encoders):
                # Forward pass for this encoder only
                encoded = encoder(x)
                
                # Apply the custom topk activation
                topk_encoded = model._topk_activation(encoded)
                
                # Count non-zero activations
                activation_counts[encoder_idx] += (topk_encoded != 0).float().sum(dim=0)

    # Convert counts to probabilities
    activation_probabilities = [counts / total_samples for counts in activation_counts]

    return activation_probabilities

def visualize_sae_features_3d(models, true_features, data_loader, device, encoders_to_compare):
    all_gt_max_sim, all_max_sim_across_saes, all_activation_probabilities = [], [], []
    colors = []

    for i, current_model in enumerate(models):
        # Calculate topk activation probability
        activation_probabilities = calculate_topk_activation_probability(current_model, data_loader, device)

        for encoder_idx in encoders_to_compare:
            current_encoder_weights = current_model.encoders[encoder_idx].weight.to(device)
            _, ground_truth_sim = calculate_MMCS(current_encoder_weights.t(), true_features, device)
            gt_max_sim = ground_truth_sim.max(dim=1)[0]

            other_sae_sims = [calculate_MMCS(current_encoder_weights.t(), other_model.encoders[other_encoder_idx].weight.t(), device)[1]
                              for j, other_model in enumerate(models) if j != i
                              for other_encoder_idx in encoders_to_compare]

            max_sim_across_saes = torch.zeros_like(gt_max_sim)

            for other_sim in other_sae_sims:
                other_max_sim = other_sim.max(dim=1)[0]
                min_features = min(gt_max_sim.shape[0], other_max_sim.shape[0])
                gt_max_sim_trunc = gt_max_sim[:min_features]
                other_max_sim_trunc = other_max_sim[:min_features]
                max_sim_across_saes[:min_features] = torch.max(max_sim_across_saes[:min_features], other_max_sim_trunc)

            all_gt_max_sim.append(gt_max_sim_trunc.detach().cpu().numpy())
            all_max_sim_across_saes.append(max_sim_across_saes[:min_features].detach().cpu().numpy())
            all_activation_probabilities.append(activation_probabilities[encoder_idx][:min_features].detach().cpu().numpy())
            colors.extend(['red' if i == 0 else 'blue'] * min_features)

            # Log additional diagnostics
            wandb.log({
                f"SAE_{i+1}_Encoder_{encoder_idx+1}_avg_activation_prob": activation_probabilities[encoder_idx].mean().item(),
                f"SAE_{i+1}_Encoder_{encoder_idx+1}_max_activation_prob": activation_probabilities[encoder_idx].max().item(),
                f"SAE_{i+1}_Encoder_{encoder_idx+1}_min_activation_prob": activation_probabilities[encoder_idx].min().item(),
            })

    # Combine all data
    gt_max_sim = np.concatenate(all_gt_max_sim)
    max_sim_across_saes = np.concatenate(all_max_sim_across_saes)
    activation_probabilities = np.concatenate(all_activation_probabilities)

    # Create 3D visualization with all data points
    create_3d_plot(gt_max_sim, max_sim_across_saes, activation_probabilities, colors, 'SAE Feature 3D Visualization (All Data)', '3d_feature_analysis_all.png')

    # Create 3D visualization without outliers
    outlier_threshold = np.percentile(activation_probabilities, 99)  # Adjust this value to include more or fewer points
    mask = activation_probabilities < outlier_threshold
    create_3d_plot(gt_max_sim[mask], max_sim_across_saes[mask], activation_probabilities[mask], np.array(colors)[mask], 
                   'SAE Feature 3D Visualization (Without Outliers)', '3d_feature_analysis_no_outliers.png')

    # Create histogram of activation probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(activation_probabilities, bins=50, alpha=0.7)
    plt.xlabel('Activation Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Feature Activation Probabilities')
    plt.savefig('activation_probability_histogram.png')
    plt.close()

    # Log to wandb
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

    # Return the data for further analysis if needed
    return gt_max_sim, max_sim_across_saes, activation_probabilities

def create_3d_plot(x, y, z, colors, title, filename):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, alpha=0.6)
    ax.set_xlabel('Max Cosine Similarity with True Features')
    ax.set_ylabel('Max Cosine Similarity between SAEs')
    ax.set_zlabel('Top-k Activation Probability')
    ax.set_title(title)

    # Add a color bar
    color_map = plt.cm.get_cmap('coolwarm')
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('SAE Index (Red: SAE 1, Blue: SAE 2)')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def create_data_loader(config, true_features=None):
    device = get_device()
    data, true_features = generate_synthetic_data(config, true_features, device)

    dataset = TensorDataset(data)
    data_loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=True
    )

    return data_loader, true_features

def run(device, config):
    # Set the default dtype for the entire process
    torch.set_default_dtype(torch.float32)

    true_features = load_true_features(wandb.run.project, device)
    true_features = true_features.to(device).to(torch.float32)  # Ensure true_features is on the correct device and dtype
    num_saes = config['hyperparameters'].get('num_saes', 1)
    model_runs = get_recent_model_runs(wandb.run.project, num_saes)
    models = [load_sae(run, config['hyperparameters'], device) for run in model_runs]
    
    # Ensure all models and their encoders are on the correct device and dtype
    for model in models:
        model.to(device)
        model.to(torch.float32)
        for encoder in model.encoders:
            encoder.to(device)
            encoder.to(torch.float32)

    # Create a single data loader for all models, using the same true_features
    data_loader, true_features = create_data_loader(config, true_features)

    # Add some debug prints
    print(f"Device: {device}")
    print(f"Model device and dtype: {next(models[0].parameters()).device}, {next(models[0].parameters()).dtype}")
    print(f"True features device and dtype: {true_features.device}, {true_features.dtype}")

    # Debug: print information about the data loader
    print(f"Data loader type: {type(data_loader)}")
    print(f"Data loader batch size: {data_loader.batch_size}")
    print(f"Data loader length: {len(data_loader)}")

    # Debug: print information about the first batch
    first_batch = next(iter(data_loader))
    print(f"First batch type: {type(first_batch)}")
    if isinstance(first_batch, list):
        print(f"First batch length: {len(first_batch)}")
        print(f"First item in batch type: {type(first_batch[0])}")
        print(f"First item in batch shape: {first_batch[0].shape}")
        print(f"First item in batch dtype: {first_batch[0].dtype}")

    try:
        gt_max_sim, max_sim_across_saes, reconstruction_importance = visualize_sae_features_3d(models, true_features, data_loader, device, list(range(num_saes)))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

    wandb.log({"experiment_completed": True})

    return {
        "gt_max_sim": gt_max_sim,
        "max_sim_across_saes": max_sim_across_saes,
        "reconstruction_importance": reconstruction_importance
    }
