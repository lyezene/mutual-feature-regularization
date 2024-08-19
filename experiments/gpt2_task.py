import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
from utils.general_utils import calculate_MMCS, get_recent_model_runs, load_sae
from utils.sae_trainer import SAETrainer
from models.sae import SparseAutoencoder


class GPT2ActivationsDataset(Dataset):
    def __init__(self, layer_index, num_samples=100, batch_size=1000, device='cuda'):
        self.layer_index = layer_index
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.dataset = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train")
        self.gpt2 = GPT2Model.from_pretrained("gpt2").to(self.device)
        self.gpt2.eval()
        self.activations = self.collect_activations()

    def collect_activations(self):
        all_activations = []
        for i in tqdm(range(0, min(self.num_samples, len(self.dataset)), self.batch_size), desc="Collecting activations"):
            batch = self.dataset[i:min(i+self.batch_size, self.num_samples)]["input_ids"]
            with torch.no_grad():
                inputs = torch.tensor(batch).to(self.device)
                outputs = self.gpt2(input_ids=inputs, output_hidden_states=True)
            layer_output = outputs.hidden_states[self.layer_index + 1]
            mlp_output = self.gpt2.h[self.layer_index].mlp(layer_output)
            all_activations.append(mlp_output.cpu())
        return torch.cat(all_activations, dim=0)

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return (self.activations[idx],)


def run(device, config):
    params = config['hyperparameters']

    wandb.init(project="gpt2_sae", config=params)

    dataset = GPT2ActivationsDataset(layer_index=0, batch_size=params['data_collection_batch_size'], device=device)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model = SparseAutoencoder(params).to(device)
    trainer = SAETrainer(model, device, params, true_features=None)

    trainer.train(dataloader, params['num_epochs'])

    wandb.finish()

    return {"experiment_completed": True}
