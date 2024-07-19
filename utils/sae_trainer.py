import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any
import itertools
import wandb
import numpy as np
from utils.general_utils import calculate_MMCS, log_sim_matrices

class SAETrainer:
    def __init__(self, model: nn.Module, device: torch.device, hyperparameters: Dict[str, Any], true_features: torch.Tensor):
        self.model: nn.Module = model.to(device)
        self.device: torch.device = device
        self.config: Dict[str, Any] = hyperparameters
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        self.criterion: nn.MSELoss = nn.MSELoss()
        self.true_features: torch.Tensor = true_features.to(device)
        self.scaler: GradScaler = GradScaler()

    def calculate_ar_loss(self, items: List[torch.Tensor]) -> torch.Tensor:
        pairs: itertools.combinations = itertools.combinations(items, 2)
        mmcs_values: List[torch.Tensor] = [1 - calculate_MMCS(a.flatten(1), b.flatten(1), self.device)[0] for a, b in pairs]
        return self.config["beta"] * (sum(mmcs_values) / len(mmcs_values) if mmcs_values else 0)

    def train(self, train_loader: DataLoader, num_epochs: int) -> Tuple[List[float], List[Tuple[float, ...]], List[torch.Tensor]]:
        losses: List[float] = []
        mmcs_scores: List[Tuple[float, ...]] = []
        
        for epoch in range(num_epochs):
            total_loss: float = 0
            for X_batch, in train_loader:
                X_batch: torch.Tensor = X_batch.to(self.device)
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(X_batch)
                    ar_loss: torch.Tensor = self.calculate_ar_loss(outputs) if self.config.get("beta", 0) > 0 else torch.tensor(0.0, device=self.device)
                    l2_loss: torch.Tensor = sum(self.criterion(output, X_batch) for output in outputs)
                    loss: torch.Tensor = l2_loss + ar_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                mmcs = tuple(calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0]
                         for encoder in self.model.encoders)

                wandb.log({
                    "MMCS": mmcs,
                    "L2_loss": l2_loss.item(),
                    "AR_loss": ar_loss,
                    "total_loss": loss.item(),
                })
            
            losses.append(total_loss / len(train_loader))
            mmcs_scores.append(mmcs)

        final_sim_matrices = [
            calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[1]
            for encoder in self.model.encoders
        ]
        log_sim_matrices(final_sim_matrices)

        return losses, mmcs_scores
