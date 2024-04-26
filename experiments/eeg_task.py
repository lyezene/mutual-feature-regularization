from data.eeg_dataset import EEGDataset
from models.mistral import MistralForEEGPrediction
from transformers import MistralConfig, MistralModel
from utils.data_utils import load_eeg_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import get_device

def train(model, dataloader, criterion, optimizer, scheduler, config, device):
    best_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), config["checkpoint_path"])
        else:
            early_stop_counter += 1
            if early_stop_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(config["checkpoint_path"]))
    return model


def run(device, config):
    eeg_data = load_eeg_data(config['eeg_subjects'], config['eeg_runs'], config['eeg_interval'])
    dataset = EEGDataset(eeg_data, normalize=config['eeg_normalize'])
    dataloader = DataLoader(dataset, batch_size=config['eeg_batch_size'], shuffle=True)

    mistral_config = MistralConfig(
        num_layers=config['eeg_num_layers'],
        hidden_size=config['eeg_hidden_size'],
        intermediate_size=config['eeg_hidden_size'] * 4,
        num_attention_heads=config['eeg_num_attention_heads'],
        max_position_embeddings=config['eeg_max_position_embeddings'],
        num_channels=dataset.data.shape[2],
    )
    model = MistralForEEGPrediction(mistral_config).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['eeg_learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config['eeg_lr_patience'], verbose=True)

    train_config = {
        "num_epochs": config['eeg_num_epochs'],
        "checkpoint_path": config['eeg_checkpoint_path'],
        "model_path": config['eeg_model_path'],
        "patience": config['eeg_patience'],
    }

    best_model = train(model, dataloader, criterion, optimizer, scheduler, train_config, device)
    torch.save(best_model.state_dict(), config['eeg_model_path'])
