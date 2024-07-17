import torch
from tqdm import tqdm
import wandb
from utils.general_utils import calculate_MMCS
from config import get_device
from torch.cuda.amp import autocast, GradScaler


class SAETrainer:
    def __init__(self, model, hyperparameters, true_features=None, device=get_device()):
        self.model = model.to(device)
        self.device = device
        self.config = hyperparameters
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config["learning_rate"]
        )
        self.criterion = torch.nn.MSELoss()
        self.true_features = true_features
        self.scaler = GradScaler()
        self.dead_latents = torch.zeros(self.model.encoders[0].weight.shape[0], dtype=torch.bool, device=device)
        self.latent_activations = torch.zeros_like(self.dead_latents, dtype=torch.long)
        self.aux_k = self.config.get("aux_k", 512)
        self.aux_alpha = self.config.get("aux_alpha", 1/32)
        self.dead_threshold = self.config.get("dead_threshold", 10_000_000)

    def calculate_ar_loss(self, items):
        items_flat = torch.stack([item.flatten(start_dim=1) for item in items])
        dot_product = torch.matmul(items_flat, items_flat.transpose(0, 1))
        norms = torch.norm(items_flat, dim=1, keepdim=True)
        cos_sim = dot_product / (norms * norms.transpose(0, 1))
        ar_loss = (1 - cos_sim.triu(diagonal=1)).mean()
        return self.config["beta"] * ar_loss

    def combined_loss(self, X_batch, hidden_states, reconstructions, ar_items):
        total_loss = 0
        l2_loss = sum(self.criterion(reconstruction, X_batch) for reconstruction in reconstructions)
        ar_loss = self.calculate_ar_loss(ar_items[0]) if ar_items else 0
    
        aux_loss = self.calculate_aux_loss(X_batch, hidden_states[0])
    
        total_loss = l2_loss + ar_loss + self.aux_alpha * aux_loss
        return total_loss, l2_loss, ar_loss, aux_loss

    def calculate_aux_loss(self, X_batch, hidden_states):
        with torch.no_grad():
            # Assuming hidden_states is a list of tensors, one for each encoder
            for hidden_state in hidden_states:
                self.latent_activations += (hidden_state != 0).sum(0)

            self.dead_latents = self.latent_activations < self.dead_threshold

        if not self.dead_latents.any():
            return torch.tensor(0.0, device=self.device)

        reconstruction = self.model.decoders[0](hidden_states[0])
        e = X_batch - reconstruction
        dead_latents = hidden_states[0][:, self.dead_latents]
        top_k_dead, _ = dead_latents.topk(min(self.aux_k, dead_latents.shape[1]), dim=1)
        e_hat = self.model.decoders[0](top_k_dead)

        aux_loss = torch.nn.functional.mse_loss(e_hat, e)

        return aux_loss

    def train(self, train_loader, num_epochs, progress_bar):
        losses, mmcs_scores = [], []
        total_batches = len(train_loader)
        progress_bar.total = total_batches * num_epochs
        log_interval = self.config.get("log_interval", 1)
        mmcs_calculation_interval = self.config.get("mmcs_calculation_interval", 1)

        for epoch in range(num_epochs):
            epoch_loss = 0
            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_loader):
                X_batch = batch[0].to(self.device)
                
                with autocast():
                    hidden_states, reconstructions, *ar_items = self.model(X_batch)
                    total_loss, l2_loss, ar_loss, aux_loss = self.combined_loss(X_batch, hidden_states, reconstructions, ar_items)

                self.scaler.scale(total_loss).backward()
                if (batch_idx + 1) % self.config["accumulation_steps"] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                epoch_loss += total_loss.item()
                progress_bar.update(1)
                self.model.normalize_decoder_weights()

                if batch_idx % log_interval == 0:
                    log_dict = {
                        'L2 Loss': f"{l2_loss.item():.4f}",
                        'AR Loss': f"{ar_loss:.4f}",
                        'Total Loss': f"{total_loss.item():.4f}"
                    }
                    progress_bar.set_postfix(log_dict)
                    wandb.log({
                        "L2_loss": l2_loss.item(),
                        "AR_loss": ar_loss,
                        "Aux Loss": aux_loss.item(),
                        "total_loss": total_loss.item(),
                    })

                if self.true_features is not None and batch_idx % mmcs_calculation_interval == 0:
                    mmcs = self.calculate_mmcs()
                    mmcs_scores.append(mmcs)
                    wandb.log({"MMCS": mmcs})

            losses.append(epoch_loss / len(train_loader))
            wandb.log({
                "epoch": epoch,
                "epoch_loss": epoch_loss / len(train_loader),
            })

        progress_bar.close()
        self.save_model()
        return losses, mmcs_scores

    def calculate_mmcs(self):
        with torch.no_grad():
            mmcs = [
                calculate_MMCS(decoder.weight, self.true_features, self.device)[0]
                for decoder in self.model.decoders
            ]
        return mmcs[0]

    def save_model(self):
        prefix = ""
        self.model.save_model(prefix)
