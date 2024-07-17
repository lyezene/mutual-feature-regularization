import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
from utils.general_utils import geometric_median

class SparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters, data_sample):
        super(SparseAutoencoder, self).__init__()
        self.config = hyperparameters
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.b_pre = nn.Parameter(torch.zeros(self.config["input_size"]))
        self.k_sparse = self.config["k_sparse"]
        self.initialize_sae(data_sample)

    def initialize_sae(self, data_sample):
        input_size = self.config["input_size"]
        hidden_sizes = [
            self.config["hidden_size"] * (i + 1)
            for i in range(self.config.get("num_saes", 1))
        ]   

        self.b_pre.data = geometric_median(data_sample)

        for hs in hidden_sizes:
            encoder = nn.Linear(input_size, hs, bias=False)
            decoder = nn.Linear(hs, input_size, bias=False)
            self.encoders.append(encoder)
            self.decoders.append(decoder)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            with torch.no_grad():
                m.weight.data = F.normalize(m.weight.data, dim=1)

    def forward(self, x):
        hidden_states = []
        reconstructions = []

        x = x - self.b_pre

        for encoder, decoder in zip(self.encoders, self.decoders):
            encoded = self.topk_activation(encoder(x), self.k_sparse)
            hidden_states.append(encoded)

            normalized_weights = F.normalize(decoder.weight, p=2, dim=1)
            decoded = decoder(encoded)
            reconstructions.append(decoded + self.b_pre)

        if self.config.get("ar", False):
            ar_property = self.config.get("property", "x_hat")
            additional_output = (
                hidden_states
                if ar_property == "hid"
                else torch.stack([encoder.weight for encoder in self.encoders])
            )
            return hidden_states, reconstructions, additional_output

        return hidden_states, reconstructions

    '''
    @staticmethod
    @torch.jit.script
    def topk_activation(x: torch.Tensor, k: int) -> torch.Tensor:
        top_values, _ = torch.topk(x, k, dim=1)
        threshold = top_values[:, -1].unsqueeze(1)
        return torch.where(x >= threshold, x, torch.zeros_like(x))
    '''

    def topk_activation(self, x, k):
        top_values, _ = torch.topk(x, k, dim=1)
        threshold = top_values[:, -1].unsqueeze(1)
        return torch.where(x >= threshold, x, torch.zeros_like(x))

    def normalize_decoder_weights(self):
        with torch.no_grad():
            for decoder in self.decoders:
                decoder.weight.data = F.normalize(decoder.weight.data, dim=0)

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
