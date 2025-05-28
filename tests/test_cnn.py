import importlib
import sys
import types
import unittest
from unittest import mock
import torch

_FAKE_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
_N_MELS = 128
_FRAMES = 44


def _make_fake_modules():
    """Insert stub modules so the training script uses them."""
    fake_ds_mod = types.ModuleType("dataset")

    class FakeDataset(torch.utils.data.Dataset):
        """Returns random tensors shaped like (1, 128, 44) + integer label."""
        def __init__(self, root, genres, n_mels):
            self.X = torch.randn(32, 1, _N_MELS, _FRAMES)
            self.y = torch.randint(0, len(genres), (32,))

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    fake_ds_mod.GTZANSpectrogramDataset = FakeDataset
    sys.modules["dataset"] = fake_ds_mod

    fake_model_mod = types.ModuleType("model_cnn")

    class FakeModel(torch.nn.Module):
        """A tiny 1-layer classifier – keeps optimisation *very* fast."""
        def __init__(self, n_mels, n_genres, *_, **__):
            super().__init__()
            self.fc = torch.nn.Linear(n_mels * _FRAMES, n_genres)

        def forward(self, x):
            b = x.size(0)
            return self.fc(x.view(b, -1))

    fake_model_mod.GenreCNN = FakeModel
    sys.modules["model_cnn"] = fake_model_mod


_DISABLE_CUDA_PATCH = mock.patch("torch.cuda.is_available", lambda: False)

_make_fake_modules()
_DISABLE_CUDA_PATCH.start()

TRAINING_SCRIPT = importlib.import_module("train_cnn")


class TestGridSearchScript(unittest.TestCase):
    """Checks that the script’s public outcomes make sense."""

    @classmethod
    def tearDownClass(cls):
        _DISABLE_CUDA_PATCH.stop()
        sys.modules.pop("dataset", None)
        sys.modules.pop("model_cnn", None)

    def test_results_length_equals_hyperparameter_grid(self):
        expected = (len(TRAINING_SCRIPT.GRID["lr"]) *
                    len(TRAINING_SCRIPT.GRID["batch_size"]))
        self.assertEqual(len(TRAINING_SCRIPT.results), expected)

    def test_each_result_entry_has_expected_fields(self):
        for entry in TRAINING_SCRIPT.results:
            self.assertSetEqual(
                set(entry.keys()), {"lr", "batch_size", "best_val_acc"}
            )
            self.assertIsInstance(entry["lr"], float)
            self.assertIsInstance(entry["batch_size"], int)
            self.assertGreaterEqual(entry["best_val_acc"], 0.0)
            self.assertLessEqual(entry["best_val_acc"], 1.0)

    def test_best_config_is_not_none_and_valid(self):
        best_cfg = TRAINING_SCRIPT.best_config
        self.assertIsNotNone(best_cfg)
        self.assertIn(best_cfg["lr"], TRAINING_SCRIPT.GRID["lr"])
        self.assertIn(best_cfg["batch_size"],
                      TRAINING_SCRIPT.GRID["batch_size"])

    def test_best_val_acc_range(self):
        self.assertGreaterEqual(TRAINING_SCRIPT.best_val_acc, 0.0)
        self.assertLessEqual(TRAINING_SCRIPT.best_val_acc, 1.0)


if __name__ == "__main__":
    unittest.main()
