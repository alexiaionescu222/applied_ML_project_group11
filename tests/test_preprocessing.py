import unittest
from unittest import mock
import numpy as np

import preprocessing


class TestpreprocessingingHelpers(unittest.TestCase):
    sr = 22_050
    clip_duration = 10
    n_mels = 128

    def test_split_audio_into_clips_exact_multiple(self):
        y = np.zeros(self.sr * self.clip_duration * 3)
        clips = preprocessing.split_audio_into_clips(
            y, self.sr, self.clip_duration
        )
        self.assertEqual(len(clips), 3)
        self.assertTrue(
            all(len(c) == self.sr * self.clip_duration for c in clips)
        )

    def test_split_audio_into_clips_drops_incomplete_tail(self):
        y = np.zeros(self.sr * self.clip_duration * 2 + self.sr * 3)
        clips = preprocessing.split_audio_into_clips(
            y, self.sr, self.clip_duration
        )
        self.assertEqual(len(clips), 2)

    def test_scale_unit_maps_to_0_1(self):
        mat = np.array([[1.0, 4.0], [2.0, 3.0]])
        scaled = preprocessing.scale_unit(mat)
        self.assertAlmostEqual(scaled.min(), 0.0)
        self.assertAlmostEqual(scaled.max(), 1.0, places=6)

    def test_extract_mel_spectrogram_shape(self):
        t = np.linspace(0, 1, self.sr, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        mel_db = preprocessing.extract_mel_spectrogram(
            y, sr=self.sr, n_mels=self.n_mels
        )
        self.assertEqual(mel_db.shape[0], self.n_mels)
        self.assertTrue(mel_db.ndim == 2)

    @mock.patch("preprocessing.librosa.feature.melspectrogram")
    @mock.patch("preprocessing.librosa.load")
    def test_preprocessing_audio_dataset_happy_path(
        self, mock_load, mock_mel
    ):
        rng = np.random.default_rng(42)

        def fake_load(path, sr):
            return rng.standard_normal(self.sr * self.clip_duration), sr

        fake_mel = rng.random((self.n_mels, 431))

        mock_load.side_effect = fake_load
        mock_mel.return_value = fake_mel

        audio_paths = [f"dummy_{i}.wav" for i in range(5)]
        labels = ["rock"] * 5

        X_pca, y, scaler, pca = preprocessing.preprocess_audio_dataset(
            audio_paths,
            labels,
            clip_duration=self.clip_duration,
            sr=self.sr,
            n_mels=self.n_mels,
            pca_components=4,
        )

        self.assertEqual(X_pca.shape[0], len(labels))
        self.assertEqual(len(y), len(labels))
        self.assertEqual(pca.n_components_, 4)
        self.assertGreaterEqual(scaler.data_min_.min(), 0.0)
        self.assertLessEqual(scaler.data_max_.max(), 1.0)

    @mock.patch(
        "preprocessing.librosa.load", side_effect=Exception("Unreadable")
    )
    def test_preprocessing_audio_dataset_raises_if_nothing_processed(
        self, mock_load
    ):
        with self.assertRaises(RuntimeError):
            preprocessing.preprocess_audio_dataset(["broken.wav"], ["rock"])


if __name__ == "__main__":
    unittest.main()
