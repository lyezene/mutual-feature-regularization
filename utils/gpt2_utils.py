import numpy as np
from config import get_device
from datasets import load_dataset
from utils.gpt4_utils import GPT4Helper
from collections import defaultdict
from random import sample
import scipy.stats
import torch
from torch.utils.data import DataLoader, TensorDataset


def stream_data(n=10):
    dataset = load_dataset("HuggingFaceFW/fineweb", split='train', streaming=True)
    sampled_data = dataset.shuffle(seed=42).take(n)
    return sampled_data


def process_activations(activations):
    return [
        [
            np.round(
                np.clip(act, 0, None) / np.max(act) * 10 if np.max(act) > 0 else act
            ).astype(int)
            for act in layer
        ]
        for layer in activations
    ]


def reconstruct(activations, autoencoders, device=get_device()):
    reconstructed_activations_list = []

    with torch.no_grad():
        for layer_num, layer_activations in enumerate(activations):
            layer_reconstructions = []
            for activation in layer_activations:
                activation_tensor = torch.tensor(activation).to(device)
                _, reconstructed = autoencoders[layer_num](activation_tensor)
                layer_reconstructions.append(reconstructed.cpu().numpy())
            reconstructed_activations_list.append(layer_reconstructions)
    return reconstructed_activations_list


def get_feature_explanations(gpt4_helper, data):
    feature_dict = defaultdict(dict)
    for processed_activations, tokens in data:
        for layer_num, layer_activations in enumerate(processed_activations):
            #num_features = layer_activations[0].shape[0]
            num_features = 1
            for feature_index in range(num_features):
                feature_activations = [
                    seq_activations[feature_index]
                    for seq_activations in layer_activations
                ]

                non_zero_activations = [act for act in feature_activations if act != 0]
                sparsity = (
                    len(non_zero_activations) / len(feature_activations)
                    if feature_activations
                    else 0
                )

                if sparsity < 0.2:
                    summed_activation = sum(non_zero_activations)
                    feature_dict[layer_num].setdefault(feature_index, []).append(
                        (summed_activation, tokens, feature_activations)
                    )

    for layer_num, features in feature_dict.items():
        for feature_index, activations_list in features.items():
            activations_list.sort(reverse=True, key=lambda x: x[0])
            top_activations = activations_list[:5]

            formatted_activations = "\n".join(
                f"{token}\t{act:.2f}"
                for _, seq_tokens, feature_activations in top_activations
                for token, act in zip(seq_tokens, feature_activations)
            )

            explanation = gpt4_helper.explain_feature(
                feature_index, formatted_activations
            )
            feature_dict[layer_num][feature_index] = explanation

    return feature_dict


def evaluate_feature_explanations(gpt4_helper, feature_explanations, all_data):
    correlation_scores = defaultdict(lambda: defaultdict(list))

    for layer_num, features in feature_explanations.items():
        for feature_index, description in features.items():
            true_activations_list = []
            predicted_activations_list = []

            sampled_data = sample(all_data, min(1, len(all_data)))

            for processed_activations, tokens in sampled_data:
                if layer_num < len(processed_activations):
                    feature_activations = [
                        layer[feature_index]
                        for layer in processed_activations[layer_num]
                    ]

                    if len(feature_activations) == len(tokens):
                        true_activations_list.extend(feature_activations)
                        predicted_activations_str = gpt4_helper.predict_activations(
                            description, tokens
                        )
                        predicted_activations = [
                            int(line.split("\t")[1])
                            for line in predicted_activations_str.split("\n")
                            if "\t" in line
                        ]
                        predicted_activations_list.extend(predicted_activations)

            if (
                true_activations_list
                and predicted_activations_list
                and len(true_activations_list) == len(predicted_activations_list)
            ):
                correlation_score, _ = scipy.stats.pearsonr(
                    true_activations_list, predicted_activations_list
                )
                correlation_scores[layer_num][feature_index].append(correlation_score)

    averaged_correlation_scores = {
        layer: {
            feature: np.mean(correlation_scores) if correlation_scores else None
            for feature, correlation_scores in features.items()
        }
        for layer, features in correlation_scores.items()
    }
    return averaged_correlation_scores


def train_gpt2_sae(activations, device, config):
    from models.sae import SparseAutoencoder, SAETrainer

    activations_tensor = torch.tensor(activations, dtype=torch.float32).to(device)
    dataset = TensorDataset(activations_tensor)
    train_loader = DataLoader(dataset, batch_size=config["training_batch_size"], shuffle=True)

    sae_hyperparameters = {
        'input_size': config['input_size'],
        'hidden_size': config['hidden_size'],
        'learning_rate': config['learning_rate'],
        'l1_coef': config['l1_coef'],
        'ar': config['ar'],
        'beta': config['beta'],
        'num_saes': config['num_saes'],
        'property': config['property'],
    }

    sae_model = SparseAutoencoder(sae_hyperparameters).to(device)
    sae_trainer = SAETrainer(sae_model, sae_hyperparameters, device=device)
    sae_trainer.train(train_loader, config["epochs"])
