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
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["learning_rate"]
        )
        self.criterion = torch.nn.MSELoss()
        self.true_features = true_features
        self.scaler = GradScaler()

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
        l1_loss = sum(hidden_state.norm(p=1, dim=1).mean() for hidden_state in hidden_states) * self.config["l1_coef"]
        ar_loss = self.calculate_ar_loss(ar_items[0]) if ar_items else 0
        total_loss = l2_loss + l1_loss + ar_loss
        return total_loss, l2_loss, l1_loss, ar_loss

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
                    total_loss, l2_loss, l1_loss, ar_loss = self.combined_loss(X_batch, hidden_states, reconstructions, ar_items)

                self.scaler.scale(total_loss).backward()
                if (batch_idx + 1) % self.config["accumulation_steps"] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                epoch_loss += total_loss.item()
                progress_bar.update(1)

                if batch_idx % log_interval == 0:
                    log_dict = {
                        'L2 Loss': f"{l2_loss.item():.4f}",
                        'L1 Loss': f"{l1_loss.item():.4f}",
                        'AR Loss': f"{ar_loss:.4f}",
                        'Total Loss': f"{total_loss.item():.4f}"
                    }
                    progress_bar.set_postfix(log_dict)
                    wandb.log({
                        "L2_loss": l2_loss.item(),
                        "L1_loss": l1_loss.item(),
                        "AR_loss": ar_loss,
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
                calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0]
                for encoder in self.model.encoders
            ]
        return mmcs[0]

    def save_model(self):
        prefix = ""
        self.model.save_model(prefix)
