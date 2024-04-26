import numpy as np
from config import get_device
from datasets import load_dataset
from utils.gpt4_utils import GPT4Helper
from collections import defaultdict
from random import sample
import scipy.stats

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


def get_feature_dict(gpt4_helper, data):
    feature_dict = defaultdict(dict)
    for processed_activations, tokens in data:
        for layer_num, layer_activations in enumerate(processed_activations):
            num_features = layer_activations[0].shape[0]
            for feature_index in range(num_features):
                feature_activations = [
                    seq_activations[feature_index]
                    for seq_activations in layer_activations
                ]
                summed_activation = sum(feature_activations)
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

            feature_dict[layer_num][feature_index] = gpt4_helper.explain_feature(
                feature_index, formatted_activations
            )
    return feature_dict


def predict_and_evaluate(gpt4_helper, feature_explanations, all_data):
    rho_scores = defaultdict(lambda: defaultdict(list))

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
                rho, _ = scipy.stats.pearsonr(
                    true_activations_list, predicted_activations_list
                )
                rho_scores[layer_num][feature_index].append(rho)

    averaged_rho_scores = {
        layer: {
            feature: np.mean(rhos) if rhos else None
            for feature, rhos in features.items()
        }
        for layer, features in rho_scores.items()
    }
    return averaged_rho_scores
