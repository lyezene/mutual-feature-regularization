import torch
from tqdm import tqdm
import wandb
from utils.general_utils import calculate_MMCS
from config import get_device


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

    def calculate_ar_loss(self, items):
        ar_loss = 0.0
        num_pairs = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                mmcs_value, _ = calculate_MMCS(
                    items[i].flatten(start_dim=1),
                    items[j].flatten(start_dim=1),
                    self.device,
                )
                ar_loss += 1 - mmcs_value
                num_pairs += 1

        if num_pairs > 0:
            ar_loss /= num_pairs
        return self.config["beta"] * ar_loss

    def train(self, train_loader, num_epochs, progress_bar):
        losses, mmcs_scores, cos_sim_matrices = [], [], []
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in train_loader:
                X_batch = batch[0].to(self.device)
                self.optimizer.zero_grad()
                hidden_states, reconstructions, *ar_items = self.model(X_batch)
                ar_loss = self.calculate_ar_loss(ar_items[0]) if ar_items else 0

                for idx, (hidden_state, reconstruction) in enumerate(zip(hidden_states, reconstructions)):
                    l2_loss = self.criterion(reconstruction, X_batch)
                    l1_loss = hidden_state.norm(p=1, dim=1).mean() * self.config["l1_coef"]
                    loss = l2_loss + ar_loss + l1_loss
                    loss.backward(retain_graph=True if idx < len(hidden_states) - 1 else False)
                    epoch_loss += loss.item()

                self.optimizer.step()
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'L2 Loss': f"{l2_loss:.4f}",
                    'L1 Loss': f"{l1_loss:.4f}",
                    'AR Loss': f"{ar_loss:.4f}"
                })

            if self.true_features is not None:
                mmcs, cos_sim_matrix = zip(*[
                    calculate_MMCS(encoder.weight.detach().t(), self.true_features, self.device)
                    for encoder in self.model.encoders
                ])
                mmcs_scores.append(mmcs)
                cos_sim_matrices.append(cos_sim_matrix)
                self.progress_bar.update(1)
                progress_bar.set_postfix({
                    'L2 Loss': f"{l2_loss:.4f}",
                    'L1 Loss': f"{l1_loss:.4f}",
                    'AR Loss': f"{ar_loss:.4f}",
                    'MMCS': f"{mmcs}"
                })

            losses.append(epoch_loss / len(train_loader))
            progress_bar.close()
            wandb.log({"loss": epoch_loss, "L2_loss": l2_loss, "L1_loss": l1_loss, "AR_loss": ar_loss})

        self.save_model()
        return losses, mmcs_scores, cos_sim_matrices

    def save_model(self):
        prefix = ""
        self.model.save_model(prefix)

