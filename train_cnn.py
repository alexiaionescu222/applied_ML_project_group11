# train_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GTZANSpectrogramDataset
from model_cnn import GenreCNN
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)

# --- HYPERPARAMETERS ---
GENRES       = ["blues","classical","country","disco","hiphop",
                "jazz","metal","pop","reggae","rock"]
BATCH_SIZE   = 16
LR           = 1e-3
NUM_EPOCHS   = 20
PATIENCE     = 3
N_MELS       = 128

# --- DATA LOADERS ---
train_ds = GTZANSpectrogramDataset("audio_split/train", GENRES, n_mels=N_MELS)
val_ds   = GTZANSpectrogramDataset("audio_split/val",   GENRES, n_mels=N_MELS)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# --- MODEL, LOSS, OPTIMIZER ---
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = GenreCNN(n_mels=N_MELS, n_genres=len(GENRES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- STORAGE FOR METRICS ---
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- TRAIN ----
    model.train()
    running_loss = 0.0
    correct = total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward(); optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---- VALIDATE ----
    model.eval()
    val_loss = val_correct = val_total = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss    += loss.item() * X_batch.size(0)
            preds       = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total   += y_batch.size(0)

    val_loss = val_loss / val_total
    val_acc  = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # ---- LOGGING ----
    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}"
    )

    # ---- EARLY STOPPING ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "cnn_best.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"→ Early stopping after {epoch} epochs.")
            break

# --- PLOT & SAVE CURVES ---
epochs = range(1, len(train_losses) + 1)

# Loss curve
plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses,   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("CNN Loss Curve")
plt.legend(); plt.tight_layout()
plt.savefig("plots/cnn_loss_curve.png")

# Accuracy curve
plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs,   label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("CNN Accuracy Curve")
plt.legend(); plt.tight_layout()
plt.savefig("plots/cnn_acc_curve.png")