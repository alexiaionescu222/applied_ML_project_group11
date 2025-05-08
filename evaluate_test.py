# evaluate_test.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.metrics       import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data       import DataLoader
from model_cnn              import GenreCNN
from dataset                import GTZANSpectrogramDataset
import os
os.makedirs("plots", exist_ok=True)


GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
BATCH_SIZE = 16
N_MELS     = 128

#k-NN on PCA features
print("=== k-NN Test Evaluation ===")
train = np.load("data/features/train_pca.npz")
test  = np.load("data/features/test_pca.npz")
X_train, y_train = train["X"], train["y"]
X_test,  y_test  = test["X"],  test["y"]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=knn.classes_))

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("k-NN Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(knn.classes_)), knn.classes_, rotation=45, ha="right")
plt.yticks(range(len(knn.classes_)), knn.classes_)
plt.tight_layout()
plt.savefig("plots/knn_test_confusion_matrix.png")
plt.close()

#CNN on raw spectrograms
print("=== CNN Test Evaluation ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GenreCNN(n_mels=N_MELS, n_genres=len(GENRES)).to(device)
model.load_state_dict(torch.load("cnn_best.pth", map_location=device))
criterion = nn.CrossEntropyLoss()

test_ds     = GTZANSpectrogramDataset("audio_split/test", GENRES, n_mels=N_MELS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

model.eval()
all_preds, all_labels = [], []
test_loss = 0.0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

test_loss /= len(test_ds)
y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)
acc = accuracy_score(y_true, y_pred)

print(f"Loss: {test_loss:.3f} | Accuracy: {acc:.3f}")
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=GENRES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("CNN Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
plt.yticks(range(len(GENRES)), GENRES)
plt.tight_layout()
plt.savefig("plots/cnn_test_confusion_matrix.png")
plt.close()