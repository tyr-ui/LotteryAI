from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from optimizer_learning import (
    apply_learning_weights,
    load_learning_strength,
    load_learning_weights,
    save_learning_strength_evaluation,
)
from predictor import PredictionWeights


class OptimizerLearningTest(unittest.TestCase):
    def test_load_learning_weights_normalizes_and_caps(self):
        with patch("optimizer_learning._load_analysis", return_value={
            "games": {"loto6": {"all_history": {"features": [
                {"feature": "freq", "average_selection_score_drop_percent": 50},
                {"feature": "delay", "average_selection_score_drop_percent": -200},
            ]}}}
        }):
            result = load_learning_weights("loto6")
        self.assertEqual(result["freq"], 0.05)
        self.assertEqual(result["delay"], -0.1)

    def test_apply_learning_weights_supports_mapping(self):
        result = apply_learning_weights(
            {"freq": 1.0, "delay": 2.0},
            {"freq": 0.1, "delay": -0.1},
            strength=0.5,
        )
        self.assertEqual(result["freq"], 1.05)
        self.assertEqual(result["delay"], 1.9)

    def test_apply_learning_weights_supports_dataclass(self):
        base = PredictionWeights()
        result = apply_learning_weights(base, {"freq": 0.1}, strength=1.0)
        self.assertIsInstance(result, PredictionWeights)
        self.assertGreaterEqual(result.global_frequency, base.global_frequency)

    def test_load_learning_strength_prefers_valid_store(self):
        with patch("optimizer_learning._load_learning_strength_store", return_value={
            "games": {"loto6": {"best_strength": 0.35}}
        }):
            self.assertEqual(load_learning_strength("loto6"), 0.35)

    def test_save_learning_strength_uses_long_term_average(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning_strength.json"
            with (
                patch("optimizer_learning.LEARNING_STRENGTH_PATH", path),
                patch("optimizer_learning.OUTPUT_DIR", Path(directory)),
            ):
                first = save_learning_strength_evaluation(
                    "loto6", 0.2,
                    [
                        {"strength": 0.2, "selection_score": 2.0},
                        {"strength": 0.8, "selection_score": 1.0},
                    ],
                )
                second = save_learning_strength_evaluation(
                    "loto6", 0.8,
                    [
                        {"strength": 0.2, "selection_score": 2.0},
                        {"strength": 0.8, "selection_score": 3.0},
                    ],
                )
                saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(first, 0.2)
        # 0.2 average=2.0, 0.8 average=2.0; weaker strength wins ties.
        self.assertEqual(second, 0.2)
        self.assertEqual(saved["games"]["loto6"]["history_count"], 2)


if __name__ == "__main__":
    unittest.main()
