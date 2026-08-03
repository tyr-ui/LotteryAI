from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


GAME_ORDER = (
    "loto6",
    "loto7",
    "miniloto",
    "numbers3",
    "numbers4",
)

GAME_NAMES = {
    "loto6": "LOTO6",
    "loto7": "LOTO7",
    "miniloto": "ミニロト",
    "numbers3": "Numbers3",
    "numbers4": "Numbers4",
}

LOTO_GAMES = {
    "loto6",
    "loto7",
    "miniloto",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return list(value)
    return []


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _format_decimal(
    value: Any,
    *,
    digits: int = 3,
    signed: bool = False,
) -> str:
    number = _number(value)
    if number is None:
        return "未評価"

    prefix = "+" if signed and float(number) > 0.0 else ""
    return f"{prefix}{float(number):.{digits}f}"


def _format_numbers(
    values: Any,
    *,
    ordered: bool,
) -> str:
    numbers = [
        int(value)
        for value in _list(values)
        if _number(value) is not None
    ]
    if not numbers:
        return "未取得"

    if ordered:
        return "".join(str(value) for value in numbers)

    return " ".join(f"{value:02d}" for value in numbers)


def _current_prediction_lines(
    game_key: str,
    section: Mapping[str, Any],
) -> list[str]:
    lines: list[str] = []
    ordered = game_key not in LOTO_GAMES

    for index, item in enumerate(
        _list(section.get("prediction")),
        start=1,
    ):
        prediction = _mapping(item)
        label = prediction.get("number")

        if not label:
            label = _format_numbers(
                prediction.get(
                    "numbers",
                    prediction.get("digits"),
                ),
                ordered=ordered,
            )

        lines.append(f"{index}. {label}")

    return lines or ["予想なし"]


def _previous_result_lines(
    game_key: str,
    previous: Mapping[str, Any],
) -> list[str]:
    status = str(previous.get("status", "unknown"))

    if status == "pending":
        return [
            "前回予想の対象回は、まだ結果未反映です。",
            str(previous.get("message", "")),
        ]

    if status != "evaluated":
        message = previous.get("message")
        return [
            str(message)
            if message
            else "前回予想の評価データはありません。"
        ]

    ordered = game_key not in LOTO_GAMES
    actual = _format_numbers(
        previous.get("actual_numbers"),
        ordered=ordered,
    )

    lines = [
        f"対象回: 第{previous.get('draw_no')}回",
        f"当選番号: {actual}",
        (
            "最高位置一致: "
            if ordered
            else "5口中最高一致: "
        )
        + str(previous.get("best_match_count", 0)),
        (
            "1口平均位置一致: "
            if ordered
            else "1口平均一致: "
        )
        + _format_decimal(
            previous.get("avg_match_count"),
            digits=2,
        ),
    ]

    if ordered:
        lines.extend(
            [
                (
                    "最高順不同一致: "
                    + str(
                        previous.get(
                            "best_unordered_match_count",
                            0,
                        )
                    )
                ),
                (
                    "ストレート: "
                    + (
                        "的中"
                        if previous.get("straight_hit")
                        else "なし"
                    )
                ),
                (
                    "BOX: "
                    + (
                        "的中"
                        if previous.get("box_hit")
                        else "なし"
                    )
                ),
            ]
        )

    return lines


def _holdout_summary(
    section: Mapping[str, Any],
) -> tuple[str, float | None]:
    holdout = _mapping(section.get("holdout_evaluation"))
    if not holdout:
        return "独立ホールドアウト: 未評価", None

    uplift = _number(holdout.get("random_uplift"))
    periods = holdout.get("holdout_periods")
    average = holdout.get("avg_matches")

    return (
        "独立ホールドアウト: "
        f"{periods}回、平均最高一致 "
        f"{_format_decimal(average, digits=3)}、"
        "一様ランダム比 "
        f"{_format_decimal(uplift, digits=3, signed=True)}",
        float(uplift) if uplift is not None else None,
    )


def _feature_summary(
    section: Mapping[str, Any],
) -> str:
    rows = [
        _mapping(item)
        for item in _list(section.get("feature_ablation"))
        if _mapping(item).get("active", True)
    ]

    if not rows:
        return "特徴量分析: 未評価"

    rows.sort(
        key=lambda row: float(
            row.get("selection_score_drop", 0.0)
            or 0.0
        ),
        reverse=True,
    )

    labels = []
    for row in rows[:3]:
        feature = row.get("feature", "unknown")
        drop = _format_decimal(
            row.get("selection_score_drop"),
            digits=3,
            signed=True,
        )
        labels.append(f"{feature} ({drop})")

    return "寄与上位: " + " / ".join(labels)


def _build_ai_summary(
    output: Mapping[str, Any],
) -> list[str]:
    game_metrics: list[tuple[str, float]] = []

    for game_key in GAME_ORDER:
        section = _mapping(output.get(game_key))
        holdout = _mapping(section.get("holdout_evaluation"))
        uplift = _number(holdout.get("random_uplift"))

        if uplift is not None:
            game_metrics.append((game_key, float(uplift)))

    lines: list[str] = []

    if game_metrics:
        best_key, best_uplift = max(
            game_metrics,
            key=lambda item: item[1],
        )
        lines.append(
            f"・独立ホールドアウトでは"
            f"{GAME_NAMES[best_key]}が最も高く、"
            f"一様ランダム比は{best_uplift:+.3f}です。"
        )

        negative = [
            GAME_NAMES[key]
            for key, uplift in game_metrics
            if uplift < 0.0
        ]
        if negative:
            lines.append(
                "・"
                + "、".join(negative)
                + "は独立ホールドアウトで"
                "一様ランダムを下回っています。"
            )
        else:
            lines.append(
                "・評価可能な全ゲームで、"
                "独立ホールドアウトは"
                "一様ランダム以上でした。"
            )
    else:
        lines.append(
            "・独立ホールドアウトの評価値は"
            "まだ取得できていません。"
        )

    evaluated = _mapping(
        output.get("previous_evaluation")
    )
    completed = [
        GAME_NAMES[key]
        for key in GAME_ORDER
        if _mapping(evaluated.get(key)).get("status")
        == "evaluated"
    ]

    if completed:
        lines.append(
            "・前回結果を確認できたゲーム: "
            + "、".join(completed)
            + "。"
        )
    else:
        lines.append(
            "・今回は前回予想の結果が"
            "まだデータへ反映されていません。"
        )

    lines.append(
        "・キャリーオーバー金額は"
        "取得機能の接続前のため、現在は未取得です。"
    )

    return lines


def build_notification_summary(
    output: Mapping[str, Any],
) -> str:
    lines = [
        "# LotteryAI 予想・振り返り",
        "",
        f"生成日時: {output.get('generated_at', '不明')}",
        "",
        "## AI総評",
        "",
        *_build_ai_summary(output),
        "",
        "---",
        "",
    ]

    previous_by_game = _mapping(
        output.get("previous_evaluation")
    )

    for game_key in GAME_ORDER:
        section = _mapping(output.get(game_key))
        previous = _mapping(
            previous_by_game.get(game_key)
        )

        lines.extend(
            [
                f"## {GAME_NAMES[game_key]}",
                "",
                f"次回: 第{section.get('next_draw_no', '不明')}回",
                (
                    "キャリーオーバー: 未取得"
                    if game_key in {"loto6", "loto7"}
                    else "キャリーオーバー: 対象外"
                ),
                f"採用モデル: {section.get('selected_config', '不明')}",
                "",
                "### 次回予想",
                "",
                *_current_prediction_lines(
                    game_key,
                    section,
                ),
                "",
                "### 前回結果",
                "",
                *_previous_result_lines(
                    game_key,
                    previous,
                ),
                "",
            ]
        )

        if game_key in LOTO_GAMES:
            holdout_text, _ = _holdout_summary(section)
            lines.extend(
                [
                    "### 評価メモ",
                    "",
                    holdout_text,
                    _feature_summary(section),
                    "",
                ]
            )

        lines.extend(
            [
                "---",
                "",
            ]
        )

    lines.extend(
        [
            "※ 数値は過去データに基づく評価であり、"
            "当選を保証するものではありません。",
            "",
        ]
    )

    return "\n".join(lines)


def write_notification_summary(
    output_dir: Path,
    output: dict,
) -> Path:
    path = output_dir / "notification_summary.md"
    path.write_text(
        build_notification_summary(output),
        encoding="utf-8",
    )
    return path


__all__ = [
    "build_notification_summary",
    "write_notification_summary",
]
