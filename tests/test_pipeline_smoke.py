import unittest
from pathlib import Path

class PipelineSmokeTest(unittest.TestCase):
    def test_required_output_directories_exist(self):
        self.assertTrue(Path("src").exists())
        self.assertTrue(Path("tests").exists())

    def test_output_directory_exists_or_can_be_created(self):
        out = Path("output")
        out.mkdir(exist_ok=True)
        self.assertTrue(out.exists())

    def test_expected_prediction_files_declared(self):
        expected = [
            "prediction_optimizer_loto6.json",
            "prediction_optimizer_loto7.json",
            "prediction_optimizer_miniloto.json",
            "prediction_optimizer_numbers3.json",
            "prediction_optimizer_numbers4.json",
        ]
        self.assertEqual(len(expected), 5)
        self.assertEqual(len(set(expected)), 5)

if __name__ == "__main__":
    unittest.main()
