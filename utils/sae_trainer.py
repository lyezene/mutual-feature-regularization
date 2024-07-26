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
import math

class SAETrainer:
    def __init__(self, model: nn.Module, device: torch.device, hyperparameters: Dict[str, Any], true_features: torch.Tensor):
        self.model: nn.Module = model.to(device)
        self.device: torch.device = device
        self.config: Dict[str, Any] = hyperparameters
        self.optimizers: List[torch.optim.Adam] = [torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"]) for encoder in self.model.encoders]
        self.criterion: nn.MSELoss = nn.MSELoss()
        self.true_features: torch.Tensor = true_features.to(device)
        self.scaler: GradScaler = GradScaler()
        self.feature_activation_warmup_batches = hyperparameters.get("feature_activation_warmup_batches", 1000)

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
        return self.config.get("consensus_weight", 0.1) * (sum(mmcs_values) / len(mmcs_values) if mmcs_values else 0)

    def cosine_warmup(self, current_step: int, warmup_steps: int) -> float:
        if current_step < warmup_steps:
            return 0.5 * (1 + math.cos(math.pi * (current_step / warmup_steps - 1)))
        return 1.0

    def calculate_auxiliary_loss(self, outputs: List[torch.Tensor], encoded_activations: List[torch.Tensor], X_batch: torch.Tensor) -> List[torch.Tensor]:
        auxiliary_losses = []
        reconstruction_contributions = [self.criterion(output, X_batch) for output in outputs]

        for i, act in enumerate(encoded_activations):
            prob_activated = (act != 0).float().mean(dim=0)

            inactive_feature_penalty = (1 - prob_activated).mean()
            auxiliary_loss = reconstruction_contributions[i] + inactive_feature_penalty
            auxiliary_losses.append(auxiliary_loss)

        return auxiliary_losses

    def train(self, train_loader: DataLoader, num_epochs: int = 1) -> Tuple[List[float], List[Tuple[float, ...]], List[torch.Tensor]]:
        losses: List[float] = []
        mmcs_scores: List[Tuple[float, ...]] = []

        for epoch in range(num_epochs):
            total_loss: float = 0
            for batch_num, (X_batch,) in enumerate(train_loader):
                X_batch: torch.Tensor = X_batch.to(self.device)

                with autocast():
                    outputs, activations = self.model.forward_with_encoded(X_batch)
                    encoder_weights = [encoder.weight.t() for encoder in self.model.encoders]

                    consensus_loss: torch.Tensor = self.calculate_consensus_loss(encoder_weights)
                    auxiliary_losses: List[torch.Tensor] = self.calculate_auxiliary_loss(outputs, activations, X_batch)

                    total_auxiliary_loss = sum(auxiliary_losses)
                    loss: torch.Tensor = consensus_loss + total_auxiliary_loss

                self.scaler.scale(loss).backward()

                for optimizer in self.optimizers:
                    self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()

                mmcs = [calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0] for encoder in self.model.encoders]
                warmup_factor = self.cosine_warmup(batch_num, self.feature_activation_warmup_batches)

                wandb.log({
                    "MMCS": mmcs,
                    "Consensus_loss": consensus_loss,
                    "Auxiliary_loss": total_auxiliary_loss.item(),
                    "total_loss": loss.item(),
                    **{f"Auxiliary_loss_SAE_{i}": aux_loss.item() for i, aux_loss in enumerate(auxiliary_losses)},
                    **{f"MMCS_SAE_{i}": mmcs_i for i, mmcs_i in enumerate(mmcs)}
                })

            losses.append(total_loss / len(train_loader))
            mmcs_scores.append(tuple(mmcs))

        self.save_model(epoch + 1)
        self.save_true_features()
        final_sim_matrices = [
            calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[1]
            for encoder in self.model.encoders
        ]
        log_sim_matrices(final_sim_matrices)
