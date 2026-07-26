from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Mapping

from backtester import run_backtest
from data_loader import dataframe_to_history, load_game_data
from features import build_model_context
from games import LOTTO_GAMES
from predictor import CandidateScore, predict


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"


def _config_value(
    config: Mapping[str, object],
    name: str,
    default: object,
) -> object:
    return config.get(name, default)


def _simple_stats(
    history: tuple[tuple[int, ...], ...],
    *,
    min_num: int,
    max_num: int,
) -> dict[str, object]:
    number_counter: Counter[int] = Counter()
    pair_counter: Counter[tuple[int, int]] = Counter()
    triplet_counter: Counter[tuple[int, int, int]] = Counter()

    for draw in history:
        number_counter.update(draw)
        pair_counter.update(combinations(draw, 2))
        triplet_counter.update(combinations(draw, 3))

    top_frequency = sorted(
        (
            {
                "number": number,
                "count": int(number_counter[number]),
            }
            for number in range(min_num, max_num + 1)
        ),
        key=lambda item: (item["count"], -item["number"]),
        reverse=True,
    )[:10]

    return {
        "total_draws": len(history),
        "top_frequency": top_frequency,
        "top_pairs": [
            {"pair": list(pair), "count": int(count)}
            for pair, count in pair_counter.most_common(10)
        ],
        "top_triplets": [
            {"triplet": list(triplet), "count": int(count)}
            for triplet, count in triplet_counter.most_common(10)
        ],
    }


def _make_rationale(
    item: CandidateScore,
) -> str:
    components = item.components
    strongest = sorted(
        components.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )[:3]
    strongest_text = "、".join(
        f"{name}={value:.3f}"
        for name, value in strongest
    )

    repeat_text = (
        "、".join(
            f"{depth}回前一致{count}個"
            for depth, count in enumerate(item.repeat_counts, start=1)
        )
        if item.repeat_counts
        else "比較対象なし"
    )

    return (
        f"総合スコア上位。主な評価要素は{strongest_text}。"
        f"直近履歴との重なりは{repeat_text}。"
    )


def _prediction_to_dict(
    selected: tuple[CandidateScore, ...],
    *,
    min_num: int,
    max_num: int,
    pick_count: int,
    backtest_summary: dict[str, object],
) -> list[dict[str, object]]:
    total_combinations = math.comb(max_num - min_num + 1, pick_count)
    estimated_probability = 1 / total_combinations

    result: list[dict[str, object]] = []

    for index, item in enumerate(selected, start=1):
        result.append(
            {
                "pattern_id": f"P{index}",
                "numbers": list(item.candidate),
                "score": round(float(item.total_score), 8),
                "estimated_probability": estimated_probability,
                "rationale": _make_rationale(item),
                "score_detail": {
                    name: round(float(value), 8)
                    for name, value in item.components.items()
                },
                "repeat_counts": list(item.repeat_counts),
                "backtest_summary": backtest_summary,
            }
        )

    return result


def run_game(
    game_name: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    display_name = str(
        _config_value(config, "display_name", game_name.upper())
    )
    main_columns = tuple(
        str(column)
        for column in _config_value(config, "main_cols", ())
    )
    pick_count = int(
        _config_value(config, "pick_count", len(main_columns))
    )
    min_num = int(_config_value(config, "min_num", 1))
    max_num = int(_config_value(config, "max_num", 1))

    loaded = load_game_data(
        game_name,
        config,
        destination=DATA_DIR / f"{game_name}.csv",
    )
    history = dataframe_to_history(loaded.dataframe, config)

    backtest = run_backtest(
        history,
        config,
        train_window=int(
            _config_value(config, "train_window", 100)
        ),
        tested_periods=int(
            _config_value(config, "tested_periods", 30)
        ),
        candidate_count=int(
            _config_value(config, "backtest_candidates", 300)
        ),
        top_k=5,
        seed=2025,
    )
    backtest_dict = backtest.to_dict(include_records=False)

    context = build_model_context(history, config)
    prediction_result = predict(
        context,
        config,
        candidate_count=int(
            _config_value(config, "final_candidates", 10000)
        ),
        top_k=5,
        seed=2025,
    )

    prediction = _prediction_to_dict(
        prediction_result.selected,
        min_num=min_num,
        max_num=max_num,
        pick_count=pick_count,
        backtest_summary=backtest_dict,
    )

    latest_draw_no = loaded.validation.get("latest_draw_no")
    next_draw_no = (
        int(latest_draw_no) + 1
        if latest_draw_no is not None
        else None
    )

    return {
        "display_name": display_name,
        "source": loaded.source,
        "latest_draw_no": latest_draw_no,
        "next_draw_no": next_draw_no,
        "validation": dict(loaded.validation),
        "stats": _simple_stats(
            history,
            min_num=min_num,
            max_num=max_num,
        ),
        "backtest_summary": backtest_dict,
        "prediction": prediction,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "status": "ok",
        "version": "LotteryAI v2",
        "games": {},
    }

    games_result: dict[str, object] = {}

    for game_name, config in LOTTO_GAMES.items():
        print(f"\n=== {config['display_name']} v2 START ===")
        game_result = run_game(game_name, config)
        games_result[game_name] = game_result

        prediction = game_result["prediction"]
        print(
            json.dumps(
                {
                    "latest_draw_no": game_result["latest_draw_no"],
                    "next_draw_no": game_result["next_draw_no"],
                    "prediction": [
                        item["numbers"]
                        for item in prediction
                    ],
                    "backtest": game_result["backtest_summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        prediction_filename = str(
            _config_value(
                config,
                "prediction_filename",
                f"prediction_{game_name}.json",
            )
        )
        (OUTPUT_DIR / prediction_filename).write_text(
            json.dumps(
                prediction,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    result["games"] = games_result

    (OUTPUT_DIR / "analysis_result_v2.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== LOTTERYAI v2 COMPLETE ===")
    print(
        json.dumps(
            {
                game_name: {
                    "latest_draw_no": game_result["latest_draw_no"],
                    "next_draw_no": game_result["next_draw_no"],
                    "prediction": [
                        item["numbers"]
                        for item in game_result["prediction"]
                    ],
                }
                for game_name, game_result in games_result.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
