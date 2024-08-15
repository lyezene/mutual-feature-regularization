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
    def __init__(self, model: nn.Module, device: torch.device, hyperparameters: Dict[str, Any], true_features: Optional[torch.Tensor] = None):
        self.model: nn.Module = model.to(device)
        self.device: torch.device = device
        self.config: Dict[str, Any] = hyperparameters
        self.optimizers: List[torch.optim.Adam] = [torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"]) for encoder in self.model.encoders]
        self.criterion: nn.MSELoss = nn.MSELoss()
        self.true_features: Optional[torch.Tensor] = true_features.to(device) if true_features is not None else None
        self.scalers: List[GradScaler] = [GradScaler() for _ in self.model.encoders]
        self.auxiliary_loss_weight = hyperparameters.get("auxiliary_loss_weight", 1.0)
        self.ensemble_consistency_weight = hyperparameters.get("ensemble_consistency_weight", 0.1)
        self.auxiliary_loss_threshold = hyperparameters.get("auxiliary_loss_threshold", 1)
        self.use_amp = hyperparameters.get("use_amp", True)

    def save_true_features(self):
        artifact = wandb.Artifact(f"{wandb.run.name}_true_features", type="true_features")
        with artifact.new_file("true_features.pt", mode="wb") as f:
            torch.save(self.true_features.cpu(), f)
        wandb.log_artifact(artifact)

    def save_model(self, epoch: int):
        run_name = f"{wandb.run.name}_epoch_{epoch}"
        self.model.save_model(run_name, alias=f"epoch_{epoch}")

    def calculate_consensus_loss(self, encoder_weights: List[torch.Tensor]) -> torch.Tensor:
        pairs = itertools.combinations(encoder_weights, 2)
        mmcs_values = [1 - calculate_MMCS(a, b, self.device)[0] for a, b in pairs]
        unweighted_loss = sum(mmcs_values) / len(mmcs_values) if mmcs_values else torch.tensor(0.0, device=self.device)
        return unweighted_loss * self.ensemble_consistency_weight

    def calculate_auxiliary_loss(self, encoded_activations: List[torch.Tensor]) -> List[torch.Tensor]:
        auxiliary_losses = []

        for act in encoded_activations:
            activation_rates = (act != 0).float().mean(dim=0)
            target_rates = torch.full_like(activation_rates, 0.0625)
            relative_diff = torch.abs(activation_rates - target_rates) / target_rates
            sensitive_diff = torch.pow(relative_diff, 3)
            mean_sensitive_diff = sensitive_diff.mean()
            auxiliary_loss = mean_sensitive_diff
            auxiliary_losses.append(auxiliary_loss)

        return [loss * self.auxiliary_loss_weight for loss in auxiliary_losses]

    def reinitialize_sae_weights(self, sae_index: int):
        encoder = self.model.encoders[sae_index]
        dtype = encoder.weight.dtype
        device = encoder.weight.device
        nn.init.orthogonal_(encoder.weight)
        nn.init.zeros_(encoder.bias)
        encoder.weight.data = encoder.weight.data.to(dtype=dtype, device=device)
        encoder.bias.data = encoder.bias.data.to(dtype=dtype, device=device)

        self.optimizers[sae_index] = torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"])
        self.scalers[sae_index] = GradScaler()

    def train(self, train_loader: DataLoader, num_epochs: int = 1) -> Tuple[List[float], List[Tuple[float, ...]], List[torch.Tensor]]:
        for epoch in range(num_epochs):
            total_loss: float = 0
            for batch_num, (X_batch,) in enumerate(train_loader):
                X_batch: torch.Tensor = X_batch.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs, activations = self.model.forward_with_encoded(X_batch)
                    encoder_weights = [encoder.weight.t() for encoder in self.model.encoders]
                    consensus_loss: torch.Tensor = self.calculate_consensus_loss(encoder_weights)
                    reconstruction_losses: List[torch.Tensor] = [self.criterion(output, X_batch) for output in outputs]
                    auxiliary_losses: List[torch.Tensor] = self.calculate_auxiliary_loss(activations)

                reinitialized = [False] * len(self.model.encoders)
                for i, aux_loss in enumerate(auxiliary_losses):
                    if aux_loss.item() > self.auxiliary_loss_threshold:
                        print(f"Auxiliary loss for SAE {i} exceeded threshold. Reinitializing weights.")
                        self.reinitialize_sae_weights(i)
                        reinitialized[i] = True

                if any(reinitialized):
                    with autocast(enabled=self.use_amp):
                        outputs, activations = self.model.forward_with_encoded(X_batch)
                        reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
                        auxiliary_losses = self.calculate_auxiliary_loss(activations)

                sae_losses = [rec_loss for rec_loss in reconstruction_losses]
                total_losses = [sae_loss + consensus_loss for sae_loss in sae_losses]

                for i, (optimizer, scaler, total_loss) in enumerate(zip(self.optimizers, self.scalers, total_losses)):
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward(retain_graph=(i < len(self.optimizers) - 1))
                    scaler.step(optimizer)
                    scaler.update()

                with torch.no_grad():
                    mmcs = [calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0] for encoder in self.model.encoders]

                wandb.log({
                    "Consensus_loss": consensus_loss,
                    **{f"SAE_{i}_loss": sae_loss.item() for i, sae_loss in enumerate(sae_losses)},
                    **{f"SAE_{i}_reconstruction_loss": rec_loss.item() for i, rec_loss in enumerate(reconstruction_losses)},
                    **{f"SAE_{i}_auxiliary_loss": aux_loss.item() for i, aux_loss in enumerate(auxiliary_losses)},
                    **{f"MMCS_SAE_{i}": mmcs_i for i, mmcs_i in enumerate(mmcs)}
                })

        self.save_model(epoch + 1)
        self.save_true_features()
