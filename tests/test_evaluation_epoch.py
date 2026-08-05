import tempfile
import unittest
from pathlib import Path

from evaluation_epoch import model_fingerprint, resolve_evaluation_epoch


class EvaluationEpochTest(unittest.TestCase):
    def test_same_model_reuses_epoch_and_change_starts_new_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            (root / "src").mkdir()
            source = root / "src" / "optimizer.py"
            source.write_text("a=1\n", encoding="utf-8")
            paths = ("src/optimizer.py",)

            # Patch the module-level file list for an isolated deterministic test.
            import evaluation_epoch
            original = evaluation_epoch.MODEL_SOURCE_FILES
            evaluation_epoch.MODEL_SOURCE_FILES = paths
            try:
                first = resolve_evaluation_epoch(root, output)
                second = resolve_evaluation_epoch(root, output)
                self.assertEqual(first["epoch_id"], 1)
                self.assertEqual(second["epoch_id"], 1)

                source.write_text("a=2\n", encoding="utf-8")
                third = resolve_evaluation_epoch(root, output)
                self.assertEqual(third["epoch_id"], 2)
                self.assertNotEqual(first["model_fingerprint"], third["model_fingerprint"])
            finally:
                evaluation_epoch.MODEL_SOURCE_FILES = original

    def test_fingerprint_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x=1", encoding="utf-8")
            self.assertEqual(
                model_fingerprint(root, ("a.py",)),
                model_fingerprint(root, ("a.py",)),
            )


if __name__ == "__main__":
    unittest.main()
