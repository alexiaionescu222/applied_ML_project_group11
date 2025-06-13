import os
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import GTZANSpectrogramDataset
from model_cnn import GenreCNN


def analyse_fit(
    train_acc_hist, val_acc_hist,
    train_loss_hist, val_loss_hist, cfg_name="run"
):
    train_acc_hist = np.asarray(train_acc_hist)
    val_acc_hist = np.asarray(val_acc_hist)
    train_loss_hist = np.asarray(train_loss_hist)
    val_loss_hist = np.asarray(val_loss_hist)

    gap_acc = train_acc_hist - val_acc_hist

    verdict = "No strong signs of over- or under-fitting."
    if train_acc_hist[-1] > 0.85 and gap_acc[-1] > 0.10:
        verdict = (
            f"Over-fitting detected ({cfg_name}): "
            f"train acc {train_acc_hist[-1]:.2%} "
            f"vs val acc {val_acc_hist[-1]:.2%}."
        )
    elif train_acc_hist[-1] < 0.60 and val_acc_hist[-1] < 0.60:
        verdict = (
            f"Under-fitting detected ({cfg_name}): "
            f"train acc {train_acc_hist[-1]:.2%}, "
            f"val acc {val_acc_hist[-1]:.2%}."
        )

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(6, 3))
    plt.plot(train_acc_hist, label="train acc")
    plt.plot(val_acc_hist, label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/learning_curve_acc_{cfg_name}.png")
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.plot(train_loss_hist, label="train loss")
    plt.plot(val_loss_hist, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/learning_curve_loss_{cfg_name}.png")
    plt.close()

    return verdict


os.makedirs("plots", exist_ok=True)

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
NUM_EPOCHS = 30
PATIENCE = 5
N_MELS = 128
CLIP_DURATION = 10
SR = 22050
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameter grid
GRID = {
    "lr": [1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32]
}


SPLITS = {
    "fold_1": "audio_split/fold_1",
    "fold_2": "audio_split/fold_2",
    "fold_3": "audio_split/fold_3",
    "fold_4": "audio_split/fold_4",
    "fold_5": "audio_split/fold_5",
}
# 5-fold CV
fold_keys = sorted(k for k in SPLITS)

# record results
best_val_acc_overall = 0.0
best_config = None
results = []

# grid search with cross-validation
for lr, batch_size in product(GRID["lr"], GRID["batch_size"]):
    print(f"\nRunning config: lr={lr}, batch_size={batch_size}")
    fold_val_accs = []
    fold_verdicts = []

    for fold_key in fold_keys:
        print(f"{fold_key} is starting now\n")
        val_subset = GTZANSpectrogramDataset(
            SPLITS[fold_key], GENRES, n_mels=N_MELS
        )
        train_ds_list = [
            GTZANSpectrogramDataset(SPLITS[k], GENRES, n_mels=N_MELS)
            for k in fold_keys if k != fold_key
        ]
        train_subset = ConcatDataset(train_ds_list)
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )

        model = GenreCNN(
            n_mels=N_MELS,
            n_genres=len(GENRES),
            clip_duration=CLIP_DURATION,
            sr=SR,
            hop_length=HOP_LENGTH
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs_no_improve = 0
        fold_best_val = 0.0
        train_acc_hist, val_acc_hist = [], []
        train_loss_hist, val_loss_hist = [], []

        for epoch in range(1, NUM_EPOCHS + 1):
            # train
            model.train()
            running_loss = correct = total = 0
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                preds = model(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X.size(0)
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)
            train_loss = running_loss / total
            train_acc = correct / total

            # validate
            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    preds = model(X)
                    val_loss += criterion(preds, y).item() * X.size(0)
                    val_correct += (preds.argmax(1) == y).sum().item()
                    val_total += y.size(0)
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} "
                f"train_loss: {train_loss:.2f}, train_acc: {train_acc:.2f} | "
                f"val_loss: {val_loss:.2f}, val_acc: {val_acc:.2f}"
            )

            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)
            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            if val_acc > fold_best_val:
                fold_best_val = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"  → Early stopping at epoch {epoch}")
                    break

        print(f"  {fold_key}: best_val_acc = {fold_best_val:.2f}")
        fold_val_accs.append(fold_best_val)

        cfg_tag = f"lr{lr}_bs{batch_size}_{fold_key}"
        verdict = analyse_fit(
            train_acc_hist, val_acc_hist,
            train_loss_hist, val_loss_hist,
            cfg_tag
        )
        print(f"    {cfg_tag} verdict: {verdict}")
        fold_verdicts.append(verdict)

    avg_val_acc = sum(fold_val_accs) / len(fold_val_accs)
    print(f"\n → Average CV val_acc = {avg_val_acc:.2f}")

    results.append({
        "lr": lr,
        "batch_size": batch_size,
        "avg_val_acc": avg_val_acc,
        "over_underfitting": fold_verdicts
    })
    if avg_val_acc > best_val_acc_overall:
        best_val_acc_overall = avg_val_acc
        best_config = {"lr": lr, "batch_size": batch_size}

# save fold-wise validation accuracies
np.save("cnn_fold_accuracies.npy", np.array(fold_val_accs))

# summary
print("\n=== CV GRID SEARCH RESULTS ===")
for r in results:
    print(
        f" lr={r['lr']:<7} bs={r['batch_size']:<3}  "
        f"avg_val_acc={r['avg_val_acc']:.3f}  "
        f"verdicts={r['over_underfitting']}"
    )

print(
    f"\nBest config: lr={best_config['lr']}, "
    f"batch_size={best_config['batch_size']}  "
    f"avg_val_acc={best_val_acc_overall:.3f}"
)
