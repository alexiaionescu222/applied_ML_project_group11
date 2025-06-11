import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# make sure plots dir exists
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# where your PCA'd features live
FEATURE_DIR = "data/features"

# folds to use in CV 
fold_names = [f"fold_{i}" for i in range(1, 6)]

# load all folds into memory
fold_data = {}
for fold in fold_names:
    arr = np.load(os.path.join(FEATURE_DIR, f"{fold}_pca.npz"))
    fold_data[fold] = (arr["X"], arr["y"])

fold_accs = []
for fold in fold_names:
    print(f"=== CV on {fold} ===")
    # validation = this fold
    X_val, y_val = fold_data[fold]
    # training = all other folds
    X_train = np.vstack([fold_data[f][0] for f in fold_names if f != fold])
    y_train = np.concatenate([fold_data[f][1] for f in fold_names if f != fold])

    # train k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # predict & score
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    fold_accs.append(acc)
    print(f"{fold} accuracy: {acc:.3f}\n")

    # detailed report for this fold
    print(classification_report(
        y_val,
        y_pred,
        target_names=GENRES,
        zero_division=0
    ))

    # confusion matrix plot
    cm = confusion_matrix(y_val, y_pred, labels=knn.classes_)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=45, ha="right")
    ax.set_yticks(range(len(GENRES)))
    ax.set_yticklabels(GENRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{fold} Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"knn_{fold}_confusion_matrix.png"))
    plt.close()

# save fold-wise validation accuracies
np.save("knn_fold_accuracies.npy", np.array(fold_accs))

# summary of all folds
avg_acc = np.mean(fold_accs)
print("\n=== k-NN 5-Fold CV Results ===")
for fold, acc in zip(fold_names, fold_accs):
    print(f"  {fold}: {acc:.3f}")
print(f"Average CV accuracy: {avg_acc:.3f}")