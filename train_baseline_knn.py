import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from glob import glob

os.makedirs("plots", exist_ok=True)

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
BASE_DIR = "audio_split"
FEATURE_DIR = "data/features"

# load PCA‐reduced full features (train+val)
train = np.load(os.path.join(FEATURE_DIR, "train_pca.npz"))
val = np.load(os.path.join(FEATURE_DIR, "val_pca.npz"))
X_full = np.vstack([train["X"], val["X"]])
y_full = np.concatenate([train["y"], val["y"]])

# load cross-validation folds
with open(os.path.join(BASE_DIR, "cv_folds.json"), "r") as f:
    folds = json.load(f)
fold_keys = sorted(folds.keys())

# reconstruct list of all relative paths
all_rel_paths = []
# entries from train
for genre in sorted(GENRES):
    wavs = sorted(glob(os.path.join(BASE_DIR, "train", genre, "*.wav")))
    for w in wavs:
        rel = os.path.normpath(
            os.path.join("train", genre, os.path.basename(w))
        )
        all_rel_paths.append(rel)
# entries from val
for genre in sorted(GENRES):
    wavs = sorted(glob(os.path.join(BASE_DIR, "val", genre, "*.wav")))
    for w in wavs:
        rel = os.path.normpath(os.path.join("val", genre, os.path.basename(w)))
        all_rel_paths.append(rel)

# ensure lengths match
assert len(all_rel_paths) == X_full.shape[0]

# CV evaluation
fold_accs = []
for fold_key in fold_keys:
    print(f"Starting {fold_key}")
    train_list = folds[fold_key]["train"]
    val_list = folds[fold_key]["val"]

    # map to indices
    train_idx = [all_rel_paths.index(os.path.normpath(p)) for p in train_list]
    val_idx = [all_rel_paths.index(os.path.normpath(p)) for p in val_list]

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val,   y_val = X_full[val_idx],   y_full[val_idx]

    # train k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # evaluate
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    fold_accs.append(acc)

    print(f"  {fold_key} Validation accuracy: {acc:.3f}")
    print(classification_report(y_val, y_pred, target_names=GENRES))

    # plot confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=knn.classes_)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=45, ha="right")
    ax.set_yticks(range(len(GENRES)))
    ax.set_yticklabels(GENRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{fold_key} Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"plots/knn_{fold_key}_confusion_matrix.png")
    plt.close()

# summary
avg_acc = np.mean(fold_accs)
print("\n=== k-NN CV Results ===")
for k, acc in zip(fold_keys, fold_accs):
    print(f"  {k}: {acc:.3f}")
print(f"Average CV accuracy: {avg_acc:.3f}")
