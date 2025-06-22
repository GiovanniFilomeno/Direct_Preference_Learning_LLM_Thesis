import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys 

module_path = os.path.abspath(os.path.join(os.getcwd(), "../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from maze_env import PolicyNetwork

def train_dpo():

    # Set device to MPS for Mac users, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the preference dataset
    df_preferences = pd.read_parquet("preferences.parquet")

    # --- Split the dataset into train (90%), val (5%), and test (5%) ---
    train_df = df_preferences.sample(frac=0.90, random_state=42)
    temp_df = df_preferences.drop(train_df.index)
    val_df = temp_df.sample(frac=0.50, random_state=42)
    test_df = temp_df.drop(val_df.index)

    # Define the Preference Dataset class
    class PreferenceDataset(Dataset):
        def __init__(self, df):
            # --- all'interno di PreferenceDataset --------------------
            norm_stats_path = "norm_stats.npz" # Adatta il path se necessario
            norm = np.load(norm_stats_path)
            mean = norm["mean"].astype(np.float32)
            std  = norm["std"].astype(np.float32) + 1e-8 # Aggiungi epsilon
            self.x_better = ((df[["x_better","y_better"]].values - mean) / std).astype(np.float32)
            self.x_worse  = ((df[["x_worse", "y_worse"] ].values - mean) / std).astype(np.float32)
            self.labels = df["preference"].values
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return (
                torch.tensor(self.x_better[idx], dtype=torch.float32),
                torch.tensor(self.x_worse[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32)
            )


    def dpo_loss(model, xb, xw, m=0.25):
        return torch.clamp(m - (model(xb) - model(xw)), min=0).mean()

    # Training parameters
    batch_size = 128
    epochs = 150
    learning_rate = 1e-3
    hidden_dim = 256
    num_layers = 4
    dropout_prob = 0.05

    # Early stopping parameter
    early_stopping_patience = 10  # Number of epochs to wait before stopping if no improvement

    # Create Dataset objects for train, val, and test
    train_dataset = PreferenceDataset(train_df)
    val_dataset = PreferenceDataset(val_df)
    test_dataset = PreferenceDataset(test_df)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = PolicyNetwork(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Track best validation loss for saving best model
    best_val_loss = float('inf')
    no_improvement_count = 0  # Tracks epochs without improvement

    # Lists to store losses per epoch
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        for x_better, x_worse, _ in train_dataloader:
            x_better, x_worse = x_better.to(device), x_worse.to(device)
            optimizer.zero_grad()
            loss = dpo_loss(model, x_better, x_worse)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_better, x_worse, _ in val_dataloader:
                x_better, x_worse = x_better.to(device), x_worse.to(device)
                loss = dpo_loss(model, x_better, x_worse)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Step the scheduler with the validation loss
        scheduler.step(avg_val_loss)

        # Check if this is the best validation loss so far; if so, save the model and reset patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_dpo_policy.pth")
            # print(f"New best model saved at epoch {epoch+1} with val loss: {avg_val_loss:.4f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # print(f"Epoch {epoch+1}/{epochs}, "
        #       f"Train Loss: {avg_train_loss:.4f}, "
        #       f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if no_improvement_count >= early_stopping_patience:
            # print(f"Early stopping triggered at epoch {epoch+1}.")
            break


    # Load the best saved model
    model.load_state_dict(torch.load("best_dpo_policy.pth"))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x_better, x_worse, _ in test_dataloader:
            x_better, x_worse = x_better.to(device), x_worse.to(device)
            r_better = model(x_better)
            r_worse = model(x_worse)
            # We consider the prediction correct if r_better > r_worse
            correct += torch.sum(r_better > r_worse).item()
            total += x_better.size(0)

    test_accuracy = correct / total
    print(f"Test accuracy (with best model): {test_accuracy:.4f}")