import numpy as np
from scipy.stats import ttest_rel

# Load per-fold accuracies
knn_accs = np.load("knn_fold_accuracies.npy")
cnn_accs = np.load("cnn_fold_accuracies.npy")

# Print mean ± std
print(f"kNN accuracy: {knn_accs.mean():.3f} ± {knn_accs.std():.3f}")
print(f"CNN accuracy: {cnn_accs.mean():.3f} ± {cnn_accs.std():.3f}")

# Paired t-test
t_stat, p_val = ttest_rel(cnn_accs, knn_accs)
print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

if p_val < 0.05:
    print("CNN significantly outperforms k-NN (p < 0.05)")
else:
    print("No significant difference detected (p >= 0.05)")

