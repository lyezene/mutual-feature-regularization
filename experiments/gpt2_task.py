import wandb
import json
import torch
from data.gpt2_dataset import GPT2ActivationDataset
from utils.gpt2_utils import get_feature_explanations, evaluate_feature_explanations
from utils.gpt4_utils import GPT4Helper
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
    wandb.init(project="alignment_regularization", entity="your_wandb_entity", job_type="experiment")

    artifact = wandb.use_artifact('gpt2_activations_dataset:latest', type='dataset')
    artifact_dir = artifact.download()
    dataset_path = artifact_dir + '/gpt2_dataset.pth'

    all_data = torch.load(dataset_path)

    gpt4_helper = GPT4Helper(config['gpt2_api_key'])

    feature_explanations = get_feature_explanations(gpt4_helper, all_data)
    correlation_scores = evaluate_feature_explanations(gpt4_helper, feature_explanations, all_data)

    correlation_scores = convert_numpy_scalars(correlation_scores)

    correlation_scores_path = 'correlation_scores.json'
    with open(correlation_scores_path, 'w') as f:
        json.dump(correlation_scores, f)

    correlation_artifact = wandb.Artifact('correlation_scores', type='result')
    correlation_artifact.add_file(correlation_scores_path)
    wandb.log_artifact(correlation_artifact)

    wandb.finish()

    return feature_explanations
