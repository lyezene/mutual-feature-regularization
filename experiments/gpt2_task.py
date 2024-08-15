import torch
import wandb
import requests
import json
from torch.utils.data import Dataset, DataLoader
from utils.general_utils import calculate_MMCS, get_recent_model_runs, load_sae
from utils.sae_trainer import SAETrainer
from models.sae import SparseAutoencoder
from tqdm import tqdm


class GPT2ActivationsDataset(Dataset):
    def __init__(self, layer_index, num_neurons):
        self.layer_index = layer_index
        self.num_neurons = num_neurons
        self.base_url = "https://openaipublic.blob.core.windows.net/neuron-explainer/gpt2_small_data/collated-activations"
        self.data = self.load_all_neuron_data()

    def load_all_neuron_data(self):
        all_data = []
        for neuron_idx in tqdm(range(self.num_neurons), desc="Loading neuron data"):
            url = f"{self.base_url}/{self.layer_index}/{neuron_idx}.json"
            response = requests.get(url)
            data = json.loads(response.text)
            activations = data['random_sample'][0]['activations']
            all_data.append(activations)
        return np.array(all_data).T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


def run(device, config):
    params = config['hyperparameters']
    
    wandb.init(project="gpt2_sae", config=params)

    dataset = GPT2ActivationsDataset(layer_index=0, num_neurons=params['input_size'])
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model = SparseAutoencoder(params).to(device)
    trainer = SAETrainer(model, device, params, true_features=None)

    trainer.train(dataloader, params['num_epochs'])

    wandb.finish()

    return {"experiment_completed": True}
