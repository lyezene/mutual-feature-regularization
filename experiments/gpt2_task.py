from data.gpt2_dataset import GPT2ActivationDataset
from utils.gpt2_utils import process_activations, get_feature_explanations, evaluate_feature_explanations, stream_data
from utils.gpt4_utils import GPT4Helper
import yaml
import numpy as np


def convert_numpy_scalars(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_scalars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_scalars(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def run(device, config):
    activation_dataset = GPT2ActivationDataset("gpt2", device)
    gpt4_helper = GPT4Helper(config['gpt2_api_key'])

    data_stream = stream_data()

    all_data = []
    for entry in data_stream:
        input_text = entry["text"]
        if not input_text.strip():
            continue

        activations, tokens = activation_dataset(input_text)
        processed_activations = process_activations(activations)
        all_data.append((processed_activations, tokens))

    feature_explanations = get_feature_explanations(gpt4_helper, all_data)
    correlation_scores = evaluate_feature_explanations(gpt4_helper, feature_explanations, all_data)

    converted_correlations = convert_numpy_scalars(correlation_scores)

    with open('correlation_scores.yaml', 'w') as f:
        yaml.dump(converted_correlations, f)
