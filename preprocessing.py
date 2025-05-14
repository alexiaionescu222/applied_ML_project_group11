# preprocessing.py
import os
import numpy as np
import joblib
import librosa
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def split_audio_into_clips(y, sr, clip_duration=10):
    clip_samples = sr * clip_duration
    clips = []
    for start in range(0, len(y), clip_samples):
        clip = y[start:start + clip_samples]
        if len(clip) == clip_samples:
            clips.append(clip)
    return clips

def extract_mel_spectrogram(y, sr=22050, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def scale_unit(mel_db):
    mn, mx = mel_db.min(), mel_db.max()
    return (mel_db - mn) / (mx - mn + 1e-6)

def evaluate_pca_performance(X, y, max_components=100):
    from collections import Counter

    label_counts = Counter(y)
    num_classes = len(label_counts)
    min_required = num_classes
    val_size = max(min_required, int(len(y) * 0.15))

    if val_size >= len(y):
        print("Skipping PCA evaluation: not enough data for validation split.")
        return [], []

    stratify_param = y if val_size >= num_classes else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=42,
        stratify=stratify_param
    )

    # Clamp components to min(n_samples, n_features)
    max_possible = min(max_components, X_train.shape[0], X_train.shape[1])
    components_range = range(1, max_possible + 1)

    accuracies = []
    for n in components_range:
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_val_pca)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(components_range, accuracies, marker='o')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Performance vs. Number of PCA Components")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return components_range, accuracies


def preprocess_audio_dataset(audio_paths, labels,
                             clip_duration=10, sr=22050,
                             n_mels=128, pca_components=7):
    X_feats, y_expanded = [], []
    for path, label in zip(audio_paths, labels):
        try:
            y, _ = librosa.load(path, sr=sr)
        except Exception as e:
            print(f"Skipping unreadable file {path!r}: {e}")
            continue

        clips = split_audio_into_clips(y, sr, clip_duration)
        for clip in clips:
            mel = extract_mel_spectrogram(clip, sr, n_mels)
            mel = scale_unit(mel)
            X_feats.append(mel.flatten())
            y_expanded.append(label)

    if not X_feats:
        raise RuntimeError("No audio clips could be processed!")

    X = np.vstack(X_feats)
    y = np.array(y_expanded)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    
    # clamp n_components so it never exceeds n_samples or n_features
    n_samples, n_features = X_scaled.shape
    n_comp = min(pca_components, n_samples, n_features)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, scaler, pca

if __name__ == "__main__":
    SPLITS = {
        "train": "audio_split/train",
        "val":   "audio_split/val",
        "test":  "audio_split/test",
    }
    OUT_DIR = "data/features"
    os.makedirs(OUT_DIR, exist_ok=True)

    for split, split_dir in SPLITS.items():
        if not os.path.isdir(split_dir):
            print(f"Missing folder {split_dir!r}, run split.py first.")
            continue

        audio_paths, labels = [], []
        for genre in os.listdir(split_dir):
            genre_folder = os.path.join(split_dir, genre)
            for fname in os.listdir(genre_folder):
                if fname.endswith(".wav"):
                    audio_paths.append(os.path.join(genre_folder, fname))
                    labels.append(genre)

        X_pca, y, scaler, pca = preprocess_audio_dataset(
            audio_paths, labels,
            clip_duration=10, sr=22050,
            n_mels=128, pca_components=7
        )

        # save features and labels
        np.savez(
            os.path.join(OUT_DIR, f"{split}_pca.npz"),
            X=X_pca,
            y=y
        )
        # dump scaler & PCA models
        joblib.dump(scaler, os.path.join(OUT_DIR, f"{split}_scaler.joblib"))
        joblib.dump(pca,    os.path.join(OUT_DIR, f"{split}_pca_model.joblib"))

        print(f"[{split}] X.shape={X_pca.shape}, y.shape={y.shape}")

    print("All splits processed.")