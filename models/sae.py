import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_device
from utils.general_utils import calculate_MMCS
from datetime import datetime
import os

class SparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters):
        super(SparseAutoencoder, self).__init__()
        self.config = hyperparameters
        self.encoders = nn.ModuleList()
        self.initialize_sae()

    def initialize_sae(self):
        input_size = self.config["input_size"]
        hidden_sizes = [
            self.config["hidden_size"] * (i + 1)
            for i in range(self.config.get("num_saes", 1))
        ]
        for hs in hidden_sizes:
            self.encoders.append(nn.Linear(input_size, hs))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        hidden_states = []
        reconstructions = []

        for encoder in self.encoders:
            encoded = F.relu(encoder(x))
            hidden_states.append(encoded)

            normalized_weights = F.normalize(encoder.weight, p=2, dim=1)
            decoded = F.linear(encoded, normalized_weights.t())
            reconstructions.append(decoded)

        if self.config.get("ar", False):
            ar_property = self.config.get("property", "x_hat")
            additional_output = (
                hidden_states
                if ar_property == "hid"
                else [encoder.weight for encoder in self.encoders]
            )
            return hidden_states, reconstructions, additional_output

        return hidden_states, reconstructions

    def save_model(self, save_dir: str, prefix: str = "sae"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{prefix}_{timestamp}.pth")
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    @classmethod
    def load_from_pretrained(cls, load_path: str, hyperparameters, device="cpu"):
        model = cls(hyperparameters)
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.to(device)
        return model


class SAETrainer:
    def __init__(self, model, hyperparameters, true_features, device=get_device()):
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

    def train(self, train_loader, num_epochs):
        losses, mmcs_scores, cos_sim_matrices = [], [], []
        for epoch in range(num_epochs):
            total_loss = 0
            for (X_batch,) in train_loader:
                X_batch = X_batch.to(self.device)
                self.optimizer.zero_grad()
                hidden_states, reconstructions, *ar_items = self.model(X_batch)
                ar_loss = self.calculate_ar_loss(ar_items[0]) if ar_items else 0
                for idx, (hidden_state, reconstruction) in enumerate(
                    zip(hidden_states, reconstructions)
                ):
                    l2_loss = self.criterion(reconstruction, X_batch)
                    l1_loss = hidden_state.norm(p=1, dim=1).mean() * self.config["l1_coef"]
                    loss = l2_loss + ar_loss + l1_loss
                    loss.backward(
                        retain_graph=True if idx < len(hidden_states) - 1 else False
                    )
                    total_loss += loss.item()

                self.optimizer.step()

            mmcs, cos_sim_matrix = zip(
                *[
                    calculate_MMCS(encoder.weight.detach().t(), self.true_features, self.device)
                    for encoder in self.model.encoders
                ]
            )
            losses.append(total_loss / len(train_loader))
            mmcs_scores.append(mmcs)
            cos_sim_matrices.append(cos_sim_matrix)

            print(
                f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}, L2: {l2_loss}, L1: {l1_loss}, MMCS Scores = {mmcs}, AR Loss = {ar_loss}"
            )

        self.save_model()
        return losses, mmcs_scores, cos_sim_matrices

    def save_model(self):
        """Saves the model using the specified prefix."""
        prefix = ""
        self.model.save_model(prefix)
