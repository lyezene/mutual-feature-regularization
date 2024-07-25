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
        self.penalize_proportion = hyperparameters.get("penalize_proportion", 0.1)

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

    def calculate_feature_activation_regularization(self, activations: List[torch.Tensor]) -> torch.Tensor:
        k = self.config["k_sparse"]
        regularization_losses = []

        for act in activations:
            num_features = act.shape[1]
            
            topk_mask = torch.zeros_like(act, dtype=torch.bool)
            topk_indices = torch.topk(act, k, dim=1).indices
            topk_mask.scatter_(1, topk_indices, True)
            prob_in_topk = topk_mask.float().mean(dim=0)
            
            num_features_to_penalize = max(1, int(self.penalize_proportion * num_features))
            
            _, bottom_indices = torch.topk(-prob_in_topk, num_features_to_penalize)
            
            feature_loss = torch.mean(1 - prob_in_topk[bottom_indices])
            regularization_losses.append(feature_loss)

        regularization_loss = torch.stack(regularization_losses).mean()
        return self.config.get("feature_activation_weight", 0.1) * regularization_loss

    def train(self, train_loader: DataLoader, num_epochs: int) -> Tuple[List[float], List[Tuple[float, ...]], List[torch.Tensor]]:
        losses: List[float] = []
        mmcs_scores: List[Tuple[float, ...]] = []
        
        for epoch in range(num_epochs):
            total_loss: float = 0
            for X_batch, in train_loader:
                X_batch: torch.Tensor = X_batch.to(self.device)
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs, activations = self.model.forward_with_encoded(X_batch)
                    encoder_weights = [encoder.weight.t() for encoder in self.model.encoders]
                    
                    consensus_loss: torch.Tensor = self.calculate_consensus_loss(encoder_weights)
                    feature_activation_loss: torch.Tensor = self.calculate_feature_activation_regularization(activations)
                    l2_loss: torch.Tensor = sum(self.criterion(output, X_batch) for output in outputs)
                    loss: torch.Tensor = l2_loss + consensus_loss + feature_activation_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                mmcs = tuple(calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0]
                         for encoder in self.model.encoders)
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
