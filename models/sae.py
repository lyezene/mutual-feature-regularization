import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_device
from utils.general_utils import calculate_MMCS
from datetime import datetime
import os
from tqdm import tqdm
import wandb
from utils.sae_trainer import SAETrainer


class SparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters):
        super(SparseAutoencoder, self).__init__()
        self.config = hyperparameters
        self.encoders = nn.ModuleList()
        self.initialize_sae()

    def initialize_sae(self):
        input_size = self.config["input_size"]
        hidden_sizes = [
            self.config["hidden_size"] * (i + 1)
            for i in range(self.config.get("num_saes", 1))
        ]
        for hs in hidden_sizes:
            self.encoders.append(nn.Linear(input_size, hs))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        hidden_states = []
        reconstructions = []

        for encoder in self.encoders:
            encoded = F.relu(encoder(x))
            hidden_states.append(encoded)

            normalized_weights = F.normalize(encoder.weight, p=2, dim=1)
            decoded = F.linear(encoded, normalized_weights.t())
            reconstructions.append(decoded)

        if self.config.get("ar", False):
            ar_property = self.config.get("property", "x_hat")
            additional_output = (
                hidden_states
                if ar_property == "hid"
                else [encoder.weight for encoder in self.encoders]
            )
            return hidden_states, reconstructions, additional_output

        return hidden_states, reconstructions

    def save_model(self, run_name: str, alias: str="latest"):
        artifact = wandb.Artifact(run_name, type='model')
        artifact.add_file(self.state_dict(), f'{run_name}.pth')
        wandb.log_artifact(artifact, aliases=[alias])
        print(f"Model checkpoint '{alias}' saved as wandb artifact under {run_name}")

    @classmethod
    def load_from_pretrained(cls, artifact_path: str, hyperparameters, device="cpu"):
        with wandb.init() as run:
            artifact = run.use_artifact(artifact_path, type='model')
            artifact_dir = artifact.download()
            model_path = os.path.join(artifact_dir, f"{artifact_path.split(':')[-1]}.pth")
            model = cls(hyperparameters)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            return model
