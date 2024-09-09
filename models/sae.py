import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import wandb
import tempfile
import os

class SparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = hyperparameters
        self.encoders: nn.ModuleList = nn.ModuleList([
            nn.Linear(self.config["input_size"], self.config["hidden_size"] * (1))
            for i in range(self.config.get("num_saes", 1))
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_with_encoded(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        results = [self._process_layer(encoder, x) for encoder in self.encoders]
        return [r[0] for r in results], [r[1] for r in results]

    def _process_layer(self, encoder: nn.Linear, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded: torch.Tensor = self._topk_activation(encoder(x))
        normalized_weights: torch.Tensor = F.normalize(encoder.weight, p=2, dim=1)
        decoded: torch.Tensor = F.linear(encoded, normalized_weights.t())
        return decoded, encoded

    def _topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        k: int = self.config["k_sparse"]
        top_values, _ = torch.topk(x, k, dim=1)
        return x * (x >= top_values[:, -1:])

    def save_model(self, run_name: str, alias: str="latest"):
        if torch.distributed.get_rank() == 0:
            artifact = wandb.Artifact(run_name, type='model')
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as tmp_file:
                torch.save(self.state_dict(), tmp_file.name)
                artifact.add_file(tmp_file.name, f'{run_name}.pth')
            wandb.log_artifact(artifact, aliases=[alias])
            os.remove(tmp_file.name)

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
