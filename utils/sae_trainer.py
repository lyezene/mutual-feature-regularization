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
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        self.criterion: nn.MSELoss = nn.MSELoss()
        self.true_features: torch.Tensor = true_features.to(device)
        self.scaler: GradScaler = GradScaler()
        self.penalize_proportion = hyperparameters.get("penalize_proportion", 0.1)
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

    def calculate_feature_activation_regularization(self, encoded_activations: List[torch.Tensor], batch_num: int) -> torch.Tensor:
        regularization_losses = []

        for act in encoded_activations:
            # Calculate activation probabilities for this batch
            prob_activated = (act != 0).float().mean(dim=0)
            
            num_features = act.shape[1]
            
            # Calculate the target activation probability
            target_prob = self.config["k_sparse"] / num_features
            
            # Calculate the deviation from the target probability
            prob_deviation = (prob_activated - target_prob).abs()
            
            # Sort features by their deviation from the target
            sorted_deviations, _ = torch.sort(prob_deviation, descending=True)
            
            # Calculate the number of features to penalize
            num_features_to_penalize = max(1, int(self.penalize_proportion * num_features))
            
            # Calculate the loss for the most deviating features
            feature_loss = sorted_deviations[:num_features_to_penalize].mean()
            
            regularization_losses.append(feature_loss)

        regularization_loss = torch.stack(regularization_losses).mean()
        
        # Apply warm-up factor
        warmup_factor = self.cosine_warmup(batch_num, self.feature_activation_warmup_batches)
        
        # Apply log scaling to prevent the loss from dominating
        scaled_loss = torch.log1p(regularization_loss)
        
        return self.config.get("feature_activation_weight", 0.1) * scaled_loss * warmup_factor

    def train(self, train_loader: DataLoader, num_epochs: int = 1) -> Tuple[List[float], List[Tuple[float, ...]], List[torch.Tensor]]:
        losses: List[float] = []
        mmcs_scores: List[Tuple[float, ...]] = []

        for epoch in range(num_epochs):
            total_loss: float = 0
            for batch_num, (X_batch,) in enumerate(train_loader):
                X_batch: torch.Tensor = X_batch.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs, activations = self.model.forward_with_encoded(X_batch)
                    encoder_weights = [encoder.weight.t() for encoder in self.model.encoders]

                    consensus_loss: torch.Tensor = self.calculate_consensus_loss(encoder_weights)
                    feature_activation_loss: torch.Tensor = self.calculate_feature_activation_regularization(activations, batch_num)
                    l2_loss: torch.Tensor = sum(self.criterion(output, X_batch) for output in outputs)
                    loss: torch.Tensor = l2_loss + consensus_loss + feature_activation_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

                mmcs = tuple(calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0]
                         for encoder in self.model.encoders)
                warmup_factor = self.cosine_warmup(batch_num, self.feature_activation_warmup_batches)
                wandb.log({
                    "MMCS": mmcs,
                    "L2_loss": l2_loss.item(),
                    "Consensus_loss": consensus_loss,
                    "Feature_Activation_loss": feature_activation_loss.item(),
                    "total_loss": loss.item(),
                })

            losses.append(total_loss / len(train_loader))
            mmcs_scores.append(mmcs)

        self.save_model(epoch + 1)
        self.save_true_features()
        final_sim_matrices = [
            calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[1]
            for encoder in self.model.encoders
        ]
        log_sim_matrices(final_sim_matrices)
        return losses, mmcs_scores
