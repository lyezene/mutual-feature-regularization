from config import get_device
from experiments import synthetic_task, gpt2_task, eeg_task, gpt2_sae
import argparse
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config):
    device = get_device()

    if config['experiment'] == 'synthetic':
        print("Running Synthetic Experiment...")
        synthetic_task.run(device, config)
    elif config['experiment'] == 'gpt2':
        print("Running GPT-2 Experiment...")
        gpt2_task.run(device, config)
    elif config['experiment'] == 'eeg':
        print("Running EEG Experiment...")
        eeg_task.run(device, config)
    elif config['experiment'] == 'gpt2-sae':
        print("Training GPT-2 SAE...")
        gpt2_sae.run(device, config)
    else:
        print("Invalid experiment name. Choose from 'synthetic', 'gpt2', or 'eeg'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
