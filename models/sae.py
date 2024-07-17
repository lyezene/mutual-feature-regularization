import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


class SparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = hyperparameters
        self.encoders: nn.ModuleList = nn.ModuleList([
            nn.Linear(self.config["input_size"], self.config["hidden_size"] * (i + 1))
            for i in range(self.config.get("num_saes", 1))
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self._process_layer(encoder, x) for encoder in self.encoders]

    def _process_layer(self, encoder: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self._topk_activation(encoder(x))
        normalized_weights: torch.Tensor = F.normalize(encoder.weight, p=2, dim=1)
        return F.linear(encoded, normalized_weights.t())

    def _topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        k: int = self.config["k_sparse"]
        top_values, _= torch.topk(x, k, dim=1)
        return x * (x >= top_values[:, -1:])
