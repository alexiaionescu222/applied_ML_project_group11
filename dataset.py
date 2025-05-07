# dataset.py

import os
import librosa
import torch
from torch.utils.data import Dataset

class GTZANSpectrogramDataset(Dataset):
    def __init__(self, split_dir, genres, sr=22050, n_mels=128, clip_duration=10):
        self.samples = []
        for label, genre in enumerate(genres):
            folder = os.path.join(split_dir, genre)
            for fname in os.listdir(folder):
                if not fname.endswith(".wav"):
                    continue

                path = os.path.join(folder, fname)
                try:
                    y, _ = librosa.load(path, sr=sr)
                except Exception as e:
                    print(f"⚠️  Skipping unreadable file {path!r}: {e}")
                    continue

                # split into fixed‐length clips
                clip_samples = sr * clip_duration
                for start in range(0, len(y), clip_samples):
                    clip = y[start:start + clip_samples]
                    if len(clip) != clip_samples:
                        continue

                    # compute Mel-spectrogram → dB → [0,1] norm
                    mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=n_mels)
                    mel_db = librosa.power_to_db(mel, ref=mel.max())
                    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

                    # to tensor (1×n_mels×time)
                    tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
                    self.samples.append((tensor, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]