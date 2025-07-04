import torch
import torch.nn as nn


class GenreCNN(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        n_genres: int = 10,
        clip_duration: int = 10,
        sr: int = 22050,
        hop_length: int = 512,
    ):
        super().__init__()
        # feature extractor: 2× (Conv → ReLU → MaxPool) + Dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3),
        )
        # figure out flatten size by passing a dummy input
        n_frames = int((clip_duration * sr) / hop_length)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            feat = self.features(dummy)
            flatten_dim = feat.numel() // feat.shape[0]
            # total features per sample

        # classifier head
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

    def predict_mc_dropout(self, x: torch.Tensor, n_samples: int = 20):
        # perform n_samples stochastic forward passes with dropout active
        self.train()   # enable dropout layers
        # collect predictions
        preds = torch.stack([self(x) for _ in range(n_samples)])
        # [n_samples, batch, n_genres]
        # compute statistics
        mean = preds.mean(dim=0)    # [batch, n_genres]
        var = preds.var(dim=0)     # [batch, n_genres]
        return mean, var
