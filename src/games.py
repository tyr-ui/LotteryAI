LOTTO_GAMES = {
    "loto6": {
        "display_name": "LOTO6",
        "kind": "loto6",
        "official_url": (
            "https://www.mizuhobank.co.jp/"
            "retail/takarakuji/loto/loto6/csv/loto6.csv"
        ),
        "fallback_kind": "loto6",
        "pick_count": 6,
        "min_num": 1,
        "max_num": 43,
        "main_cols": [
            "main1",
            "main2",
            "main3",
            "main4",
            "main5",
            "main6",
        ],
        "bonus_cols": ["bonus"],
        "all_columns": [
            "draw_no",
            "date",
            "main1",
            "main2",
            "main3",
            "main4",
            "main5",
            "main6",
            "bonus",
        ],
        "block_ranges": [
            (1, 10),
            (11, 21),
            (22, 32),
            (33, 43),
        ],
        "allowed_odd_counts": [2, 3, 4],
        "allowed_low_counts": [2, 3, 4],
        "train_window": 500,
        "tested_periods": 90,
        "backtest_candidates": 300,
        "final_candidates": 10000,
        "prediction_filename": "prediction_optimizer_loto6.json",
    },

    "loto7": {
        "display_name": "LOTO7",
        "kind": "loto7",
        "official_url": (
            "https://www.mizuhobank.co.jp/"
            "retail/takarakuji/loto/loto7/csv/loto7.csv"
        ),
        "fallback_kind": "loto7",
        "pick_count": 7,
        "min_num": 1,
        "max_num": 37,
        "main_cols": [
            "main1",
            "main2",
            "main3",
            "main4",
            "main5",
            "main6",
            "main7",
        ],
        "bonus_cols": ["bonus1", "bonus2"],
        "all_columns": [
            "draw_no",
            "date",
            "main1",
            "main2",
            "main3",
            "main4",
            "main5",
            "main6",
            "main7",
            "bonus1",
            "bonus2",
        ],
        "block_ranges": [
            (1, 9),
            (10, 18),
            (19, 27),
            (28, 37),
        ],
        "allowed_odd_counts": [3, 4],
        "allowed_low_counts": [3, 4],
        "train_window": 240,
        "tested_periods": 90,
        "backtest_candidates": 300,
        "final_candidates": 10000,
        "prediction_filename": "prediction_optimizer_loto7.json",
    },

    "miniloto": {
        "display_name": "MINILOTO",
        "kind": "miniloto",
        "official_url": (
            "https://www.mizuhobank.co.jp/"
            "retail/takarakuji/loto/miniloto/csv/miniloto.csv"
        ),
        "fallback_kind": "miniloto",
        "pick_count": 5,
        "min_num": 1,
        "max_num": 31,
        "main_cols": [
            "main1",
            "main2",
            "main3",
            "main4",
            "main5",
        ],
        "bonus_cols": ["bonus"],
        "all_columns": [
            "draw_no",
            "date",
            "main1",
            "main2",
            "main3",
            "main4",
            "main5",
            "bonus",
        ],
        "block_ranges": [
            (1, 8),
            (9, 16),
            (17, 24),
            (25, 31),
        ],
        "allowed_odd_counts": [2, 3],
        "allowed_low_counts": [2, 3],
        "train_window": 500,
        "tested_periods": 90,
        "backtest_candidates": 300,
        "final_candidates": 10000,
        "prediction_filename": "prediction_optimizer_miniloto.json",
    },
    
    "numbers3": {
        "display_name": "NUMBERS3",
        "family": "numbers",
        "kind": "numbers3",
        "official_url": "",
        "fallback_kind": "numbers3",

        "digit_count": 3,
        "digit_min": 0,
        "digit_max": 9,

        "main_cols": [
            "digit1",
            "digit2",
            "digit3",
        ],

        "bonus_cols": [],
        "all_columns": [
            "draw_no",
            "date",
            "digit1",
            "digit2",
            "digit3",
        ],

        "train_window": 1000,
        "tested_periods": 180,
        "backtest_candidates": 500,
        "final_candidates": 10000,

        "prediction_filename":
            "prediction_optimizer_numbers3.json",
    },

    "numbers4": {
        "display_name": "NUMBERS4",
        "family": "numbers",
        "kind": "numbers4",
        "official_url": "",
        "fallback_kind": "numbers4",

        "digit_count": 4,
        "digit_min": 0,
        "digit_max": 9,

        "main_cols": [
            "digit1",
            "digit2",
            "digit3",
            "digit4",
        ],

        "bonus_cols": [],
        "all_columns": [
            "draw_no",
            "date",
            "digit1",
            "digit2",
            "digit3",
            "digit4",
        ],

        "train_window": 1000,
        "tested_periods": 180,
        "backtest_candidates": 500,
        "final_candidates": 10000,

        "prediction_filename":
            "prediction_optimizer_numbers4.json",
    },
}