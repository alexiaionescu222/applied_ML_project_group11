import importlib
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
import sklearn.metrics as _skm
import numpy as np
import torch

_rng = np.random.default_rng(0)
_FAKE_X_TRAIN = _rng.random((48, 20))
_FAKE_Y_TRAIN = _rng.integers(0, 10, 48)
_FAKE_X_TEST = _rng.random((16, 20))
_FAKE_Y_TEST = _rng.integers(0, 10, 16)


class _FakeNpz(dict):
    def __init__(self, X, y):
        super().__init__(X=X, y=y)


def _fake_np_load(path, *_, **__):
    if "train" in path:
        return _FakeNpz(_FAKE_X_TRAIN, _FAKE_Y_TRAIN)
    if "test" in path:
        return _FakeNpz(_FAKE_X_TEST, _FAKE_Y_TEST)
    raise FileNotFoundError(path)


_N_MELS = 128
_FRAMES = 44
_N_GENRES = 10


def _install_fake_modules():
    fake_ds_mod = types.ModuleType("dataset")

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, *_args, **_kw):
            self.X = torch.randn(24, 1, _N_MELS, _FRAMES)
            self.y = torch.randint(0, _N_GENRES, (24,))

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    fake_ds_mod.GTZANSpectrogramDataset = FakeDataset
    sys.modules["dataset"] = fake_ds_mod

    fake_model_mod = types.ModuleType("model_cnn")

    class FakeModel(torch.nn.Module):
        def __init__(self, n_mels, n_genres, *_, **__):
            super().__init__()
            self.fc = torch.nn.Linear(n_mels * _FRAMES, n_genres)

        def forward(self, x):
            b = x.size(0)
            return self.fc(x.view(b, -1))

    fake_model_mod.GenreCNN = FakeModel
    sys.modules["model_cnn"] = fake_model_mod


_install_fake_modules()

_NO_CUDA = mock.patch("torch.cuda.is_available", lambda: False)
_PATCH_NPLOAD = mock.patch("numpy.load", side_effect=_fake_np_load)


def _fake_state_dict():
    return {
        "fc.weight": torch.randn(_N_GENRES, _N_MELS * _FRAMES),
        "fc.bias": torch.randn(_N_GENRES),
    }


_PATCH_TLOAD = mock.patch("torch.load", lambda *_, **__: _fake_state_dict())

_SAVEFIG_CALLS = []


def _fake_savefig(fname, *_, **__):
    _SAVEFIG_CALLS.append(fname)


_PATCH_SAVEFIG = mock.patch(
    "matplotlib.pyplot.savefig", side_effect=_fake_savefig
)

_orig_classification_report = _skm.classification_report


def _safe_classification_report(y_true, y_pred, *args, **kwargs):
    try:
        return _orig_classification_report(y_true, y_pred, *args, **kwargs)
    except ValueError as exc:
        if "does not match size of target_names" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("target_names", None)
            kwargs.pop("labels", None)
            return _orig_classification_report(y_true, y_pred, *args, **kwargs)
        raise


_PATCH_CR = mock.patch(
    "sklearn.metrics.classification_report",
    side_effect=_safe_classification_report,
)

with _NO_CUDA, _PATCH_NPLOAD, _PATCH_TLOAD, _PATCH_SAVEFIG, _PATCH_CR:
    buf = StringIO()
    with redirect_stdout(buf):
        EVAL = importlib.import_module("evaluate_test")
    _PRINTED_OUTPUT = buf.getvalue()


class TestEvaluationScript(unittest.TestCase):
    def test_script_prints_two_section_headers(self):
        self.assertIn("=== k-NN Test Evaluation ===", _PRINTED_OUTPUT)
        self.assertIn("=== CNN Test Evaluation ===", _PRINTED_OUTPUT)

    def test_confusion_matrices_were_saved(self):
        self.assertIn("plots/knn_test_confusion_matrix.png", _SAVEFIG_CALLS)
        self.assertIn("plots/cnn_test_confusion_matrix.png", _SAVEFIG_CALLS)

    def test_accuracy_numbers_reasonable(self):
        import re

        accs = re.findall(r"Accuracy:\s+([0-1]\.\d{3})", _PRINTED_OUTPUT)
        self.assertGreaterEqual(len(accs), 1)
        for a in accs:
            v = float(a)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_no_unexpected_errors_seen(self):
        self.assertNotIn("Traceback", _PRINTED_OUTPUT)


if __name__ == "__main__":
    unittest.main()
