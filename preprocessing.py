# preprocessing.py
import os
import numpy as np
import joblib
import librosa
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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

def preprocess_audio_dataset(audio_paths, labels,
                             clip_duration=10, sr=22050,
                             n_mels=128, pca_components=50):
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
            n_mels=128, pca_components=50
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