import os
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reproducibility
random.seed(42)
np.random.seed(42)


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


def gather_all_audio_files(split_dirs):
    audio_paths = []
    labels = []
    for split_name, split_dir in split_dirs.items():
        if not os.path.isdir(split_dir):
            continue
        for genre in os.listdir(split_dir):
            genre_folder = os.path.join(split_dir, genre)
            for fname in os.listdir(genre_folder):
                if fname.endswith(".wav"):
                    audio_paths.append(os.path.join(genre_folder, fname))
                    labels.append(genre)
    return audio_paths, labels


def extract_features(
    audio_paths, labels, sr=22050, clip_duration=10, n_mels=128
):
    X_feats, y_expanded = [], []
    for path, label in zip(audio_paths, labels):
        try:
            y, _ = librosa.load(path, sr=sr)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        clips = split_audio_into_clips(y, sr, clip_duration)
        for clip in clips:
            mel = extract_mel_spectrogram(clip, sr=sr, n_mels=n_mels)
            mel = scale_unit(mel)
            X_feats.append(mel.flatten())
            y_expanded.append(label)
    return np.vstack(X_feats), np.array(y_expanded)


def evaluate_pca(X, y, max_components=100, show_plot=True):
    label_counts = Counter(y)
    num_classes = len(label_counts)
    val_size = max(num_classes, int(len(y) * 0.15))

    stratify_param = y if val_size >= num_classes else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=stratify_param
    )

    max_possible = min(max_components, X_train.shape[0], X_train.shape[1])
    print(f"Max number of components: {max_possible}")
    components_range = range(1, max_possible + 1)

    accuracies = []
    for n in components_range:
        print(f" {n} step in progress")
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_pca, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val_pca))
        accuracies.append(acc)
    print("Components range done!")

    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(components_range, accuracies, marker='o')
        plt.xlabel("Number of PCA Components")
        plt.ylabel("Validation Accuracy")
        plt.title("Global PCA Evaluation (All WAVs)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    best_acc = max(accuracies)
    best_components = [
        components_range[i]
        for i, acc in enumerate(accuracies)
        if acc == best_acc
    ]

    print(
        f"Best components (global): {best_components}, "
        f"Accuracy: {best_acc:.4f}"
    )

    return best_components, dict(zip(components_range, accuracies))


if __name__ == "__main__":
    SPLITS = {
        "train": "audio_split/train",
        "val": "audio_split/val",
        "test": "audio_split/test",
    }

    print("Gathering audio from all splits...")
    paths, labels = gather_all_audio_files(SPLITS)
    print(f"Total files found: {len(paths)}")

    print("Extracting features...")
    X, y = extract_features(paths, labels)

    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("Evaluating PCA performances...")
    evaluate_pca(X_scaled, y, max_components=100)
