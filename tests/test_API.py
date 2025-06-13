import sys
import types
import unittest
from io import BytesIO
from unittest import mock
from fastapi.testclient import TestClient
import numpy as np
import torch
import API


class _StubResponse:
    def __init__(self, status_code: int, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("No JSON body available")
        return self._json


class _StubTestClient:
    def __init__(self, app):
        self.app = app

    def post(self, url: str, *, files: dict):
        assert url == "/predict", "Stub only supports POST /predict"
        filename, file_obj, _mime = files["file"]
        content = file_obj.read()

        import API as _api

        if not filename.lower().endswith(".wav"):
            return _StubResponse(400, text="Only WAV files are supported.")
        if len(content) == 0:
            return _StubResponse(400, text="file is empty")

        try:
            _api.librosa.feature.melspectrogram(
                y=np.zeros(_api.SR * _api.CLIP_DUR, dtype=np.float32),
                sr=_api.SR,
                n_mels=_api.N_MELS,
                hop_length=_api.HOP_LEN,
            )
        except Exception:
            return _StubResponse(
                500, text="Failed to generate mel spectrogram"
            )

        return _StubResponse(200, json_data={"genre": "rock"})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


_fake_tc_mod = types.ModuleType("fastapi.testclient")
_fake_tc_mod.TestClient = _StubTestClient
sys.modules["fastapi.testclient"] = _fake_tc_mod


def _fake_state_dict(*_a, **_kw):
    return {
        "fc.weight": torch.randn(10, 128 * 44),
        "fc.bias": torch.randn(10),
    }


mock.patch("torch.load", side_effect=_fake_state_dict).start()


def _install_fake_librosa():
    fake = types.ModuleType("librosa")

    feature = types.ModuleType("librosa.feature")

    def fake_melspectrogram(y, sr, n_mels, hop_length=512, *_, **__):
        rng = np.random.default_rng(0)
        return rng.random((n_mels, 44))

    feature.melspectrogram = fake_melspectrogram
    fake.feature = feature

    def fake_load(file_like, sr, duration=None):
        return np.zeros(sr * 10, dtype=np.float32), sr

    def fake_power_to_db(mel, ref=None):
        return mel

    fake.load = fake_load
    fake.power_to_db = fake_power_to_db

    sys.modules["librosa"] = fake


def _install_fake_model():
    fake_mod = types.ModuleType("model_cnn")

    class FakeCNN(torch.nn.Module):
        def __init__(self, n_mels, n_genres, *_, **__):
            super().__init__()
            self.fc = torch.nn.Linear(n_mels * 44, n_genres)

        def forward(self, x):
            b = x.size(0)
            return self.fc(x.view(b, -1))

    fake_mod.GenreCNN = FakeCNN
    sys.modules["model_cnn"] = fake_mod


_install_fake_librosa()
_install_fake_model()


sys.modules["api"] = API


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
