# train_baseline_knn.py
import numpy as np
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.metrics       import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot    as plt
import os
os.makedirs("plots", exist_ok=True)

# 1) load PCA‐reduced features
train = np.load("data/features/train_pca.npz")
X_train, y_train = train["X"], train["y"]
val   = np.load("data/features/val_pca.npz")
X_val, y_val     = val["X"],   val["y"]

# 2) fit k-NN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 3) evaluate on validation set
y_pred = knn.predict(X_val)
acc    = accuracy_score(y_val, y_pred)
print(f"✔️  k-NN(k=5) Validation acc: {acc:.3f}\n")
print("Classification report:")
print(classification_report(y_val, y_pred))

# 4) confusion matrix
labels = knn.classes_
cm     = confusion_matrix(y_val, y_pred, labels=labels)
fig, ax = plt.subplots(1,1,figsize=(7,7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("k-NN Confusion Matrix")
plt.colorbar(im, ax=ax); plt.tight_layout()
plt.savefig("plots/knn_val_confusion_matrix.png")
plt.close()