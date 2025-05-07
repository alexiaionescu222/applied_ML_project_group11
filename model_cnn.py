# model_cnn.py
import torch
import torch.nn as nn

class GenreCNN(nn.Module):
    def __init__(
        self,
        n_mels: int       = 128,
        n_genres: int     = 10,
        clip_duration: int = 10,
        sr: int           = 22050,
        hop_length: int   = 512,
    ):
        super().__init__()
        # 1) feature extractor: 2× (Conv → ReLU → MaxPool) + Dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3),
        )

        # 2) figure out flatten size by passing a dummy input
        #    dummy shape = (batch=1, channel=1, freq_bins=n_mels, time_frames)
        n_frames = int((clip_duration * sr) / hop_length)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            feat  = self.features(dummy)
            flatten_dim = feat.numel() // feat.shape[0]  # total features per sample

        # 3) classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_genres),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x