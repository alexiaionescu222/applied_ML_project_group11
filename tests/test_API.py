import importlib
import sys
import types
import unittest
from io import BytesIO
from unittest import mock

import numpy as np
import torch
from fastapi.testclient import TestClient

_SR = 22_050
_N_MELS = 128
_CLIP_LEN = 10
_FRAMES = 44
_N_GENRES = 10


def fake_torch_load(*args, **kwargs):
    return {
        "fc.weight": torch.randn(10, 128 * 44),  # shape must match FakeCNN
        "fc.bias": torch.randn(10),
    }

mock_load = mock.patch("torch.load", side_effect=fake_torch_load)
mock_load.start()

import API

mock_load.stop()

def _install_fake_librosa():
    """A minimal librosa substitute that does no actual decoding."""
    fake = types.ModuleType("librosa")

    feature = types.ModuleType("librosa.feature")

    def fake_melspectrogram(y, sr, n_mels, hop_length):
        rng = np.random.default_rng(0)
        return rng.random((n_mels, _FRAMES))

    feature.melspectrogram = fake_melspectrogram
    fake.feature = feature

    def fake_load(file_like, sr, duration=None):
        return np.zeros(sr * _CLIP_LEN, dtype=np.float32), sr

    def fake_power_to_db(mel, ref=None):
        return mel

    fake.load = fake_load
    fake.power_to_db = fake_power_to_db

    sys.modules["librosa"] = fake


def _install_fake_model():
    fake_model_mod = types.ModuleType("model_cnn")

    class FakeCNN(torch.nn.Module):
        def __init__(self, n_mels, n_genres, *_, **__):
            super().__init__()
            self.fc = torch.nn.Linear(n_mels * _FRAMES, n_genres)

        def forward(self, x):
            b = x.size(0)
            return self.fc(x.view(b, -1))

    fake_model_mod.GenreCNN = FakeCNN
    sys.modules["model_cnn"] = fake_model_mod


_install_fake_librosa()
_install_fake_model()

_PATCH_CUDA = mock.patch("torch.cuda.is_available", lambda: False)
_PATCH_TLOAD = mock.patch("torch.load", lambda *_, **__: {})

with _PATCH_CUDA, _PATCH_TLOAD:
    API = importlib.import_module("API")

client = TestClient(API.app)


class TestPredictEndpoint(unittest.TestCase):
    def _post_file(self, filename: str, bytes_obj: bytes):
        return client.post(
            "/predict",
            files={"file": (filename, BytesIO(bytes_obj), "audio/wav")},
        )

    def test_predict_success_returns_genre(self):
        resp = self._post_file("clip.wav", b"\x00\x01dummy wav bytes")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("genre", body)
        self.assertIsInstance(body["genre"], str)

    def test_predict_non_wav_rejected(self):
        resp = self._post_file("song.mp3", b"1234")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Only WAV files", resp.text)

    def test_predict_empty_file_rejected(self):
        resp = self._post_file("empty.wav", b"")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("file is empty", resp.text.lower())

    @mock.patch(
        "api.librosa.feature.melspectrogram", side_effect=RuntimeError("boom")
    )
    def test_internal_error_gracefully_500(self, _):
        resp = self._post_file("clip.wav", b"xyz")
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Failed to generate mel spectrogram", resp.text)


if __name__ == "__main__":
    unittest.main()
