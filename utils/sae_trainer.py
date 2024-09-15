import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Dict, Any
import itertools
import wandb
import numpy as np
from utils.general_utils import calculate_MMCS
import math


class SAETrainer:
    def __init__(self, model: nn.Module, device: str, hyperparameters: Dict[str, Any], true_features: Optional[torch.Tensor] = None):
        self.model = model
        self.device = device
        self.config = hyperparameters
        self.base_model = model
        self.optimizers = [torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"]) for encoder in self.base_model.encoders]
        self.criterion = nn.MSELoss()
        self.true_features = true_features.to(device) if true_features is not None else None
        self.scalers = [GradScaler() for _ in self.base_model.encoders]
        self.ensemble_consistency_weight = hyperparameters.get("ensemble_consistency_weight", 0.1)
        self.use_amp = hyperparameters.get("use_amp", True)
        self.warmup_steps = hyperparameters.get("warmup_steps", 100)
        self.current_step = 0

    def get_warmup_factor(self) -> float:
        if self.current_step >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (self.warmup_steps - self.current_step) / self.warmup_steps))

    def save_true_features(self):
        artifact = wandb.Artifact(f"{wandb.run.name}_true_features", type="true_features")
        with artifact.new_file("true_features.pt", mode="wb") as f:
            torch.save(self.true_features.cpu(), f)
        wandb.log_artifact(artifact)

    def save_model(self, epoch: int):
        run_name = f"{wandb.run.name}_epoch_{epoch}"
        self.base_model.save_model(run_name, alias=f"epoch_{epoch}")

    def calculate_consensus_loss(self, encoder_weights: List[torch.Tensor]) -> torch.Tensor:
        pairs = itertools.combinations(encoder_weights, 2)
        mmcs_values = [1 - calculate_MMCS(a.clone(), b.clone(), self.device)[0] for a, b in pairs]
        unweighted_loss = torch.mean(torch.stack(mmcs_values)) if mmcs_values else torch.tensor(0.0, device=self.device)
        warmup_factor = self.get_warmup_factor()
        return unweighted_loss * self.ensemble_consistency_weight * warmup_factor

    def train(self, train_loader: DataLoader, num_epochs: int = 1):
        for epoch in range(num_epochs):
            for batch_num, (X_batch,) in enumerate(train_loader):
                self.current_step += 1
                X_batch = X_batch.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs, activations = self.base_model.forward_with_encoded(X_batch)
                    encoder_weights = [encoder.weight.t() for encoder in self.base_model.encoders]
                    consensus_loss = self.calculate_consensus_loss(encoder_weights)
                    reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]

                total_losses = [rec_loss + consensus_loss for rec_loss in reconstruction_losses]

                for i, (optimizer, scaler, total_loss) in enumerate(zip(self.optimizers, self.scalers, total_losses)):
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward(retain_graph=(i < len(self.optimizers) - 1))
                    scaler.step(optimizer)
                    scaler.update()

                if self.true_features is not None:
                    with torch.no_grad():
                        mmcs = [calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0] for encoder in self.base_model.encoders]

                wandb.log({
                    "Consensus_loss": consensus_loss,
                    **{f"SAE_{i}_reconstruction_loss": rec_loss.item() for i, rec_loss in enumerate(reconstruction_losses)},
                    **(({f"MMCS_SAE_{i}": mmcs_i for i, mmcs_i in enumerate(mmcs)}) if self.true_features is not None else {})
                })

            self.save_model(epoch + 1)
            if self.true_features is not None:
                self.save_true_features()
