import importlib
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class MainScriptTest(unittest.TestCase):
    """Run `main.py` once under heavy patching and then make assertions."""

    @classmethod
    def setUpClass(cls):
        cls._captured_cmds = []

        def _fake_system(cmd):
            cls._captured_cmds.append(cmd)
            return 0

        cls._patch_system = patch("os.system", side_effect=_fake_system)
        cls._patch_system.start()

        cls._stdout = io.StringIO()
        with redirect_stdout(cls._stdout):
            cls._main = importlib.import_module("run_all")

        cls._patch_system.stop()
        cls._printed = cls._stdout.getvalue()

    def test_commands_are_called_in_expected_order(self):
        expected = [
            "python split.py",
            "python preprocessing.py",
            "python train_baseline_knn.py",
            "python train_cnn.py",
            "python evaluate_test.py",
            "python comparison_model.py"
        ]
        self.assertEqual(self._captured_cmds, expected)

    def test_every_step_header_is_printed(self):
        """Each phase should announce itself to the user."""
        for step in [
            "Running split.py",
            "Running preprocessing.py",
            "Running train_baseline_knn.py",
            "Running train_cnn.py",
            "Running evaluate_test.py",
            "Running comparison_model.py"
        ]:
            with self.subTest(step=step):
                self.assertIn(step, self._printed)

    def test_trailing_summary_message_printed(self):
        self.assertIn("All steps completed.", self._printed)

    def test_script_exit_code_success(self):
        """
        Because `os.system` is faked to return 0 and main.py never raises,
        the module import should complete cleanly (no uncaught exception).
        """
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
