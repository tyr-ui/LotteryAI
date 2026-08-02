from __future__ import annotations

import unittest

from games import LOTTO_GAMES


EXPECTED_GAME_KEYS = {
    "loto6",
    "loto7",
    "miniloto",
    "numbers3",
    "numbers4",
}

COMBINATION_EXPECTATIONS = {
    "loto6": {
        "display_name": "LOTO6",
        "pick_count": 6,
        "min_num": 1,
        "max_num": 43,
        "bonus_count": 1,
    },
    "loto7": {
        "display_name": "LOTO7",
        "pick_count": 7,
        "min_num": 1,
        "max_num": 37,
        "bonus_count": 2,
    },
    "miniloto": {
        "display_name": "MINILOTO",
        "pick_count": 5,
        "min_num": 1,
        "max_num": 31,
        "bonus_count": 1,
    },
}

NUMBERS_EXPECTATIONS = {
    "numbers3": {
        "display_name": "NUMBERS3",
        "digit_count": 3,
    },
    "numbers4": {
        "display_name": "NUMBERS4",
        "digit_count": 4,
    },
}


class GameDefinitionTest(unittest.TestCase):
    def test_exactly_five_supported_games_are_registered(self) -> None:
        self.assertEqual(EXPECTED_GAME_KEYS, set(LOTTO_GAMES))

    def test_combination_game_definitions(self) -> None:
        for game_key, expected in COMBINATION_EXPECTATIONS.items():
            with self.subTest(game_key=game_key):
                config = LOTTO_GAMES[game_key]

                self.assertEqual(expected["display_name"], config["display_name"])
                self.assertEqual(game_key, config["kind"])
                self.assertNotEqual(
                    "numbers",
                    str(config.get("family", "lotto")).lower(),
                )

                self.assertEqual(expected["pick_count"], config["pick_count"])
                self.assertEqual(expected["min_num"], config["min_num"])
                self.assertEqual(expected["max_num"], config["max_num"])
                self.assertLess(config["min_num"], config["max_num"])

                self.assertEqual(config["pick_count"], len(config["main_cols"]))
                self.assertEqual(
                    config["pick_count"],
                    len(set(config["main_cols"])),
                )
                self.assertEqual(expected["bonus_count"], len(config["bonus_cols"]))

                self.assertEqual(["draw_no", "date"], config["all_columns"][:2])
                self.assertEqual(
                    config["main_cols"] + config["bonus_cols"],
                    config["all_columns"][2:],
                )
                self.assertEqual(
                    f"prediction_optimizer_{game_key}.json",
                    config["prediction_filename"],
                )

    def test_numbers_game_definitions(self) -> None:
        for game_key, expected in NUMBERS_EXPECTATIONS.items():
            with self.subTest(game_key=game_key):
                config = LOTTO_GAMES[game_key]

                self.assertEqual(expected["display_name"], config["display_name"])
                self.assertEqual(game_key, config["kind"])
                self.assertEqual("numbers", config["family"])

                self.assertEqual(expected["digit_count"], config["digit_count"])
                self.assertEqual(0, config["digit_min"])
                self.assertEqual(9, config["digit_max"])

                expected_main_cols = [
                    f"digit{position}"
                    for position in range(1, config["digit_count"] + 1)
                ]
                self.assertEqual(expected_main_cols, config["main_cols"])
                self.assertEqual([], config["bonus_cols"])
                self.assertEqual(
                    ["draw_no", "date", *config["main_cols"]],
                    config["all_columns"],
                )
                self.assertEqual(
                    f"prediction_optimizer_{game_key}.json",
                    config["prediction_filename"],
                )

    def test_prediction_filenames_are_unique(self) -> None:
        filenames = [
            str(config["prediction_filename"])
            for config in LOTTO_GAMES.values()
        ]
        self.assertEqual(len(filenames), len(set(filenames)))


if __name__ == "__main__":
    unittest.main()
