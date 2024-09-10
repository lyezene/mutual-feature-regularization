import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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
        self.base_model = model.module if isinstance(model, DDP) else model
        self.optimizers = [torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"]) for encoder in self.base_model.encoders]
        self.criterion = nn.MSELoss()
        self.true_features = true_features.to(device) if true_features is not None else None
        self.scalers = [GradScaler() for _ in self.base_model.encoders]
        self.auxiliary_loss_weight = hyperparameters.get("auxiliary_loss_weight", 1.0)
        self.ensemble_consistency_weight = hyperparameters.get("ensemble_consistency_weight", 0.1)
        self.auxiliary_loss_threshold = hyperparameters.get("auxiliary_loss_threshold", 1)
        self.use_amp = hyperparameters.get("use_amp", True)
        self.warmup_steps = hyperparameters.get("warmup_steps", 100)
        self.current_step = 0

    def get_warmup_factor(self) -> float:
        if self.current_step >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (self.warmup_steps - self.current_step) / self.warmup_steps))

    def save_true_features(self):
        if dist.get_rank() == 0:
            artifact = wandb.Artifact(f"{wandb.run.name}_true_features", type="true_features")
            with artifact.new_file("true_features.pt", mode="wb") as f:
                torch.save(self.true_features.cpu(), f)
            wandb.log_artifact(artifact)

    def save_model(self, epoch: int):
        if dist.get_rank() == 0:
            run_name = f"{wandb.run.name}_epoch_{epoch}"
            self.base_model.save_model(run_name, alias=f"epoch_{epoch}")

    def calculate_consensus_loss(self, encoder_weights: List[torch.Tensor]) -> torch.Tensor:
        pairs = itertools.combinations(encoder_weights, 2)
        mmcs_values = [1 - calculate_MMCS(a.clone(), b.clone(), self.device)[0] for a, b in pairs]
        unweighted_loss = torch.mean(torch.stack(mmcs_values)) if mmcs_values else torch.tensor(0.0, device=self.device)
        warmup_factor = self.get_warmup_factor()
        return unweighted_loss * self.ensemble_consistency_weight * warmup factor

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
        encoder = self.base_model.encoders[sae_index]
        dtype = encoder.weight.dtype
        device = encoder.weight.device
        nn.init.orthogonal_(encoder.weight)
        nn.init.zeros_(encoder.bias)
        encoder.weight.data = encoder.weight.data.to(dtype=dtype, device=device)
        encoder.bias.data = encoder.bias.data.to(dtype=dtype, device=device)
        self.optimizers[sae_index] = torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"])
        self.scalers[sae_index] = GradScaler()

    def train(self, train_loader: DataLoader, num_epochs: int = 1):
        for epoch in range(num_epochs):
            train_loader.sampler.set_epoch(epoch)
            for batch_num, (X_batch,) in enumerate(train_loader):
                self.current_step += 1
                X_batch = X_batch.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs, activations = self.base_model.forward_with_encoded(X_batch)
                    encoder_weights = [encoder.weight.t() for encoder in self.base_model.encoders]
                    consensus_losses: List[torch.Tensor] = self.calculate_consensus_loss(encoder_weights)
                    reconstruction_losses: List[torch.Tensor] = [self.criterion(output, X_batch) for output in outputs]
                    auxiliary_losses: List[torch.Tensor] = self.calculate_auxiliary_loss(activations)

                reinitialized = [False] * len(self.base_model.encoders)
                for i, aux_loss in enumerate(auxiliary_losses):
                    if aux_loss.item() > self.auxiliary_loss_threshold:
                        if dist.get_rank() == 0:
                            print(f"Auxiliary loss for SAE {i} exceeded threshold. Reinitializing weights.")
                        self.reinitialize_sae_weights(i)
                        reinitialized[i] = True

                if any(reinitialized):
                    with autocast(enabled=self.use_amp):
                        outputs, activations = self.base_model.forward_with_encoded(X_batch)
                        reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
                        auxiliary_losses = self.calculate_auxiliary_loss(activations)
                        print(type(auxiliary_losses[0]))

                total_losses = [rec_loss + consensus_loss for rec_loss in reconstruction_losses]

                for i, (optimizer, scaler, total_loss) in enumerate(zip(self.optimizers, self.scalers, total_losses)):
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward(retain_graph=(i < len(self.optimizers) - 1))
                    scaler.step(optimizer)
                    scaler.update()

                if self.true_features is not None:
                    with torch.no_grad():
                        mmcs = [calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0] for encoder in self.base_model.encoders]

                if dist.get_rank() == 0:
                    wandb.log({
                        "Consensus_loss": consensus_loss,
                        **{f"SAE_{i}_reconstruction_loss": rec_loss.item() for i, rec_loss in enumerate(reconstruction_losses)},
                        **{f"SAE_{i}_auxiliary_loss": aux_loss.item() for i, aux_loss in enumerate(auxiliary_losses)},
                        **(({f"MMCS_SAE_{i}": mmcs_i for i, mmcs_i in enumerate(mmcs)}) if self.true_features is not None else {})
                    })

            self.save_model(epoch + 1)
            if self.true_features is not None:
                self.save_true_features()
