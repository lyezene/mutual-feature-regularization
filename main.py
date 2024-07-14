import argparse
import yaml
import wandb
from config import get_device
from experiments import synthetic_task
from utils.data_utils import generate_synthetic_data
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
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Run experiments or generate datasets')
    parser.add_argument('--config', type=str, help='Optional path to the configuration file for experiments or dataset generation')
    parser.add_argument('--generate-dataset', choices=['synthetic'], help='Generate a specific dataset')

    args = parser.parse_args()

    if args.generate_dataset:
        config = load_config(args.config)
        device = get_device()
        init_wandb(config)
        print(f"Generating {args.generate_dataset.capitalize()} Dataset...")
        if args.generate_dataset == 'synthetic':
            generate_synthetic_data(config, device)
        wandb.finish()
    elif args.config:
        config = load_config(args.config)
        run_experiment(config)
    else:
        print("Error: No operation specified. Please provide a configuration file using --config or use --generate-dataset to generate a dataset.")


if __name__ == '__main__':
    main()
