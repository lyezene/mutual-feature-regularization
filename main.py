import argparse
import yaml
import wandb
from config import get_device
from experiments import synthetic_task, gpt2_task, eeg_task, gpt2_sae
from utils.data_utils import generate_synthetic_dataset, generate_gpt2_dataset
import traceback


def load_config(config_path):
    if config_path:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        return {}


def init_wandb(config):
    default_config = {
        'project': 'alignment_regularization',
        'config': config
    }
    wandb.init(**default_config)


def run_experiment(config):
    init_wandb(config)
    device = get_device()

    try:
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
            raise ValueError(f"Invalid experiment name: {config['experiment']}. Choose from 'synthetic', 'gpt2', 'eeg', 'gpt2-sae'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Run experiments or generate datasets')
    parser.add_argument('--config', type=str, help='Optional path to the configuration file for experiments or dataset generation')
    parser.add_argument('--generate-dataset', choices=['synthetic', 'gpt2'], help='Generate a specific dataset')

    args = parser.parse_args()

    if args.generate_dataset:
        config = load_config(args.config)
        device = get_device()
        init_wandb(config)
        print(f"Generating {args.generate_dataset.capitalize()} Dataset...")
        if args.generate_dataset == 'synthetic':
            generate_synthetic_dataset(config, device)
        elif args.generate_dataset == 'gpt2':
            generate_gpt2_dataset(device)
        wandb.finish()
    elif args.config:
        config = load_config(args.config)
        run_experiment(config)
    else:
        print("Error: No operation specified. Please provide a configuration file using --config or use --generate-dataset to generate a dataset.")


if __name__ == '__main__':
    main()
