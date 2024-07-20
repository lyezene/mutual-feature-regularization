import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import wandb


class SparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = hyperparameters
        self.encoders: nn.ModuleList = nn.ModuleList([
            nn.Linear(self.config["input_size"], self.config["hidden_size"] * (1))
            for i in range(self.config.get("num_saes", 1))
        ])
        self.init_method = torch.rand(self.config.get("num_saes", 1), 1) * 2 - 1
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            encoder_index = next((i for i, encoder in enumerate(self.encoders) if encoder == m), None)
            if encoder_index is not None:
                init_value = self.init_method[encoder_index].item()
                if init_value > 0.5:
                    nn.init.orthogonal_(m.weight)
                elif 0 < init_value <= 0.5:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self._process_layer(encoder, x) for encoder in self.encoders]

    def _process_layer(self, encoder: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self._topk_activation(encoder(x))
        normalized_weights: torch.Tensor = F.normalize(encoder.weight, p=2, dim=1)
        return F.linear(encoded, normalized_weights.t())

    def _topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        k: int = self.config["k_sparse"]
        top_values, _ = torch.topk(x, k, dim=1)
        return x * (x >= top_values[:, -1:])

    def get_pre_acts(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.encoders[layer_idx](x)

    def save_model(self, run_name: str, alias: str="latest"):
        artifact = wandb.Artifact(run_name, type='model')
        temp_path = f'{run_name}_temp.pth'
        torch.save(self.state_dict(), temp_path)
        artifact.add_file(self.state_dict(), f'{run_name}.pth')
        wandb.log_artifact(artifact, aliases=[alias])
        os.remove(temp_path)

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
