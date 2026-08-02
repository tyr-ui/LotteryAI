from __future__ import annotations

import pytest

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


def test_exactly_five_supported_games_are_registered() -> None:
    assert set(LOTTO_GAMES) == EXPECTED_GAME_KEYS


@pytest.mark.parametrize(
    ("game_key", "expected"),
    COMBINATION_EXPECTATIONS.items(),
)
def test_combination_game_definitions(
    game_key: str,
    expected: dict[str, int | str],
) -> None:
    config = LOTTO_GAMES[game_key]

    assert config["display_name"] == expected["display_name"]
    assert config["kind"] == game_key
    assert str(config.get("family", "lotto")).lower() != "numbers"

    assert config["pick_count"] == expected["pick_count"]
    assert config["min_num"] == expected["min_num"]
    assert config["max_num"] == expected["max_num"]
    assert config["min_num"] < config["max_num"]

    assert len(config["main_cols"]) == config["pick_count"]
    assert len(set(config["main_cols"])) == config["pick_count"]
    assert len(config["bonus_cols"]) == expected["bonus_count"]

    assert config["all_columns"][:2] == ["draw_no", "date"]
    assert config["all_columns"][2:] == (
        config["main_cols"] + config["bonus_cols"]
    )

    assert config["prediction_filename"] == (
        f"prediction_optimizer_{game_key}.json"
    )


@pytest.mark.parametrize(
    ("game_key", "expected"),
    NUMBERS_EXPECTATIONS.items(),
)
def test_numbers_game_definitions(
    game_key: str,
    expected: dict[str, int | str],
) -> None:
    config = LOTTO_GAMES[game_key]

    assert config["display_name"] == expected["display_name"]
    assert config["kind"] == game_key
    assert config["family"] == "numbers"

    assert config["digit_count"] == expected["digit_count"]
    assert config["digit_min"] == 0
    assert config["digit_max"] == 9

    assert config["main_cols"] == [
        f"digit{position}"
        for position in range(1, config["digit_count"] + 1)
    ]
    assert config["bonus_cols"] == []
    assert config["all_columns"] == [
        "draw_no",
        "date",
        *config["main_cols"],
    ]

    assert config["prediction_filename"] == (
        f"prediction_optimizer_{game_key}.json"
    )


def test_prediction_filenames_are_unique() -> None:
    filenames = [
        str(config["prediction_filename"])
        for config in LOTTO_GAMES.values()
    ]

    assert len(filenames) == len(set(filenames))
