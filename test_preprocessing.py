import os
import sys
import random
import numpy as np
from preprocessing import preprocess_audio_dataset

# Fix randomness for reproducibility
random.seed(42)
np.random.seed(42)

SPLIT_DIR = "audio_split/train"

if not os.path.isdir(SPLIT_DIR):
    print(f"Error: missing {SPLIT_DIR!r}. Run split.py first.")
    sys.exit(1)

audio_files, labels = [], []
for genre in sorted(os.listdir(SPLIT_DIR)):
    genre_folder = os.path.join(SPLIT_DIR, genre)
    wavs = sorted([f for f in os.listdir(genre_folder) if f.endswith(".wav")])
    if not wavs:
        continue
    # Always pick the first file alphabetically
    pick = wavs[0]
    audio_files.append(os.path.join(genre_folder, pick))
    labels.append(genre)

X, y, scaler, pca = preprocess_audio_dataset(audio_files, labels)
#components_range, accuracies = evaluate_pca_performance(X, y, max_components=100)

print("Processed clips:", X.shape[0])
print("Feature dim   :", X.shape[1])
print("Sample labels :", y)
#print("Components range:", components_range)
#print("Accuracies:", accuracies)
