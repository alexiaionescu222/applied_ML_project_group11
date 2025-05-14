import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import product
from dataset import GTZANSpectrogramDataset
from model_cnn import GenreCNN
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# fixed settings
GENRES        = ["blues","classical","country","disco","hiphop",
                 "jazz","metal","pop","reggae","rock"]
NUM_EPOCHS    = 30
PATIENCE      = 5
N_MELS        = 128
CLIP_DURATION = 10
SR            = 22050
HOP_LENGTH    = 512
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper‐parameter grid
GRID = {
    "lr":         [1e-3, 5e-4, 1e-4],
    "batch_size":[16, 32]
}

# to record results
best_val_acc = 0.0
best_config  = None
results      = []

# grid search
for lr, batch_size in product(GRID["lr"], GRID["batch_size"]):
    print(f"\n Running config: lr={lr}, batch_size={batch_size}")
    # data loaders (recreated per batch_size)
    train_ds = GTZANSpectrogramDataset("audio_split/train", GENRES, n_mels=N_MELS)
    val_ds   = GTZANSpectrogramDataset("audio_split/val",   GENRES, n_mels=N_MELS)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # model, loss, optimizer
    model     = GenreCNN(
                    n_mels=N_MELS,
                    n_genres=len(GENRES),
                    clip_duration=CLIP_DURATION,
                    sr=SR,
                    hop_length=HOP_LENGTH
                ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # early‐stopping trackers
    epochs_no_improve = 0
    config_best_val   = 0.0

    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- train ---
        model.train()
        running_loss = correct = total = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X)
            loss  = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            correct      += (preds.argmax(1) == y).sum().item()
            total        += y.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # --- validate ---
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X)
                val_loss    += criterion(preds, y).item() * X.size(0)
                val_correct += (preds.argmax(1) == y).sum().item()
                val_total   += y.size(0)

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total

        print(
            f"  Epoch {epoch}/{NUM_EPOCHS} "
            f"– train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f} "
            f"| val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}"
        )

        # early stopping per config
        if val_acc > config_best_val:
            config_best_val   = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  → Early stopping at epoch {epoch}")
                break

    # record and compare against global best
    results.append({
        "lr": lr, "batch_size": batch_size, "best_val_acc": config_best_val
    })
    if config_best_val > best_val_acc:
        best_val_acc = config_best_val
        best_config  = {"lr": lr, "batch_size": batch_size}

# summary
print("\n=== GRID SEARCH RESULTS ===")
for r in results:
    print(f" lr={r['lr']:<7} bs={r['batch_size']:<3}  val_acc={r['best_val_acc']:.3f}")
print(f"\n Best config: lr={best_config['lr']}, batch_size={best_config['batch_size']} → val_acc={best_val_acc:.3f}")
