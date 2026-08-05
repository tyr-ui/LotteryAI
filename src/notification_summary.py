"""Build human-readable and Discord-ready LotteryAI notifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from storage import load_json, save_json
from statistical_evaluation import build_game_statistical_report


GAME_ORDER = ("loto6", "loto7", "miniloto", "numbers3", "numbers4")
GAME_NAMES = {
    "loto6": "LOTO6",
    "loto7": "LOTO7",
    "miniloto": "ミニロト",
    "numbers3": "Numbers3",
    "numbers4": "Numbers4",
}
LOTO_GAMES = {"loto6", "loto7", "miniloto"}
CARRYOVER_GAMES = {"loto6", "loto7"}

COLOR_NORMAL = 0x2F6FED
COLOR_CARRYOVER = 0xC89B24
COLOR_MUTED = 0x667085
DISCORD_EMBED_TOTAL_LIMIT = 5_800


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


def _format_numbers(values: Any, *, ordered: bool) -> str:
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


def _prediction_lines(
    game_key: str,
    section: Mapping[str, Any],
) -> list[str]:
    ordered = game_key not in LOTO_GAMES
    lines: list[str] = []
    for index, item in enumerate(_list(section.get("prediction")), start=1):
        row = _mapping(item)
        label = str(row.get("number") or "")
        if not label:
            label = _format_numbers(
                row.get("numbers", row.get("digits")),
                ordered=ordered,
            )
        lines.append(f"{index}. {label}")
    return lines or ["予想なし"]



def _previous_prediction_lines(
    game_key: str,
    previous: Mapping[str, Any],
) -> list[str]:
    """Format the predictions that were actually saved for the evaluated draw."""
    ordered = game_key not in LOTO_GAMES
    lines: list[str] = []

    for index, item in enumerate(_list(previous.get("predictions")), start=1):
        row = _mapping(item)
        label = str(row.get("number") or "")
        if not label:
            label = _format_numbers(
                row.get("numbers", row.get("digits")),
                ordered=ordered,
            )

        if ordered:
            match_count = int(row.get("position_matches", 0) or 0)
            match_text = f"位置{match_count}個一致"
        else:
            match_count = int(row.get("matches", 0) or 0)
            match_text = f"{match_count}個一致"

        lines.append(f"{index}. {label}（{match_text}）")

    return lines

def _previous_summary(
    game_key: str,
    previous: Mapping[str, Any],
) -> str:
    status = str(previous.get("status", "unknown"))
    if status == "pending":
        return "結果未反映"
    if status != "evaluated":
        return str(previous.get("message") or "評価データなし")

    ordered = game_key not in LOTO_GAMES
    actual = _format_numbers(previous.get("actual_numbers"), ordered=ordered)
    best_label = "最高位置一致" if ordered else "5口中最高一致"
    average_label = "1口平均位置一致" if ordered else "1口平均一致"
    lines = [
        f"第{previous.get('draw_no')}回 / 当選番号 {actual}",
        f"{best_label} {previous.get('best_match_count', 0)}",
        f"{average_label} {_format_decimal(previous.get('avg_match_count'), digits=2)}",
    ]
    if ordered:
        lines.append(
            "ストレート "
            + ("的中" if previous.get("straight_hit") else "なし")
            + " / ボックス "
            + ("的中" if previous.get("box_hit") else "なし")
        )
    return "\n".join(lines)


def _carryover_row(output: Mapping[str, Any], game_key: str) -> Mapping[str, Any]:
    return _mapping(
        _mapping(_mapping(output.get("carryover")).get("games")).get(game_key)
    )


def _carryover_text(output: Mapping[str, Any], game_key: str) -> str:
    if game_key not in CARRYOVER_GAMES:
        return "対象外"
    row = _carryover_row(output, game_key)
    status = str(row.get("status") or "unknown")
    if status == "fetch_error":
        return "取得できず（公式サイトへの接続エラー）"
    if status == "parse_error":
        return "取得できず（公式ページを解析できませんでした）"
    if status == "stale":
        return "確認待ち（公式情報の回号が未更新）"
    if status == "status_only":
        return str(row.get("display_amount") or "発生中（金額未取得）")
    if status == "cached":
        return str(row.get("display_amount") or "未取得") + "（前回取得値）"
    if status != "ok":
        return "未取得"
    return str(row.get("display_amount") or "未取得")


def _has_carryover(output: Mapping[str, Any], game_key: str) -> bool:
    row = _carryover_row(output, game_key)
    return bool(row.get("status") in {"ok", "cached", "status_only"} and row.get("has_carryover"))




def _statistical_report(output: Mapping[str, Any], game_key: str) -> Mapping[str, Any]:
    reports = _mapping(output.get("statistical_reports"))
    report = _mapping(reports.get(game_key))
    if report:
        return report
    return build_game_statistical_report(game_key, _mapping(output.get(game_key)), [])


def _statistics_text(output: Mapping[str, Any], game_key: str) -> str:
    report = _statistical_report(output, game_key)
    paired = _mapping(report.get("paired_evaluation"))
    operational = _mapping(report.get("operational_evaluation"))
    interval = _mapping(paired.get("confidence_interval_95"))
    permutation_p = _mapping(paired.get("permutation_p_value_reference")).get("value")
    sign_test_p = _mapping(paired.get("sign_test_p_value_reference")).get("value")
    mean_difference = paired.get("mean_difference")
    if mean_difference is None:
        return (
            f"統計判定: {paired.get('judgement', '評価データなし')}\n"
            f"データ量: {paired.get('data_volume', '未評価')}\n"
            f"実運用: {operational.get('evaluated_draws', 0)}回"
        )
    lower = interval.get("lower")
    upper = interval.get("upper")
    ci = "未評価" if lower is None or upper is None else f"{float(lower):+.3f}～{float(upper):+.3f}"
    permutation_text = "未評価" if permutation_p is None else f"{float(permutation_p):.4f}（参考）"
    sign_text = "未評価" if sign_test_p is None else f"{float(sign_test_p):.4f}（参考）"
    started = operational.get("evaluation_started_at") or operational.get("started_at") or "未記録"
    return (
        f"統計判定: {paired.get('judgement')}\n"
        f"データ量: {paired.get('data_volume')}\n"
        f"平均差: {float(mean_difference):+.3f}\n"
        f"95%CI: {ci}\n"
        f"平均差permutation p={permutation_text}\n"
        f"勝敗符号検定p={sign_text}\n"
        f"実運用: {operational.get('evaluated_draws', 0)}回 / 評価開始 {started}"
    )


def _feature_top3(section: Mapping[str, Any]) -> str:
    rows = [
        _mapping(item)
        for item in _list(section.get("feature_ablation"))
        if _mapping(item).get("active", True)
    ]
    if not rows:
        return "未評価"
    rows.sort(
        key=lambda row: float(row.get("selection_score_drop", 0.0) or 0.0),
        reverse=True,
    )
    return " / ".join(
        f"{row.get('feature', 'unknown')} "
        f"({_format_decimal(row.get('selection_score_drop'), signed=True)})"
        for row in rows[:3]
    )


def _lotto_evaluation(section: Mapping[str, Any]) -> str:
    holdout = _mapping(section.get("holdout_evaluation"))
    ranked = _mapping((_list(section.get("ranked_configs")) or [{}])[0])
    holdout_text = "未評価"
    if holdout:
        holdout_text = (
            f"{holdout.get('tested_periods', holdout.get('holdout_periods', '不明'))}回 / "
            f"平均最高一致 {_format_decimal(holdout.get('avg_matches'))} / "
            f"一様ランダム比 {_format_decimal(holdout.get('random_uplift'), signed=True)}"
        )
    model_uplift = _format_decimal(ranked.get("random_uplift"), signed=True)
    production = _mapping(section.get("final_candidate_holdout"))
    production_text = "未評価"
    if production:
        production_text = (
            f"{production.get('holdout_periods', '不明')}回 / "
            f"平均最高一致 {_format_decimal(production.get('avg_matches'))} / "
            f"一様ランダム比 {_format_decimal(production.get('random_uplift'), signed=True)}"
        )
    return (
        f"開発用頑健性評価（300候補・3seed）: {holdout_text}\n"
        f"本番条件ホールドアウト（10,000候補・本番seed）: {production_text}\n"
        f"探索期間の一様ランダム比: {model_uplift}"
    )


def _numbers_evaluation(section: Mapping[str, Any]) -> str:
    holdout = _mapping(section.get("holdout_evaluation"))
    if holdout:
        random_baseline = _mapping(holdout.get("random_baseline"))
        straight = _format_decimal(
            holdout.get("straight_hit_rate"),
            digits=4,
        )
        box = _format_decimal(holdout.get("box_hit_rate"), digits=4)
        random_straight = _format_decimal(
            random_baseline.get("straight_hit_rate"),
            digits=4,
        )
        position_uplift = _format_decimal(
            holdout.get("random_uplift"),
            signed=True,
        )
        return (
            f"開発用ホールドアウト: {holdout.get('tested_periods', holdout.get('holdout_periods', '不明'))}回 / "
            f"平均最高位置一致 {_format_decimal(holdout.get('average_best_position_matches'))} / "
            f"一様ランダム比 {position_uplift}\n"
            f"ストレート率 {straight} / ボックス率 {box} / "
            f"一様ランダムのストレート率 {random_straight}"
        )

    ranked = _mapping((_list(section.get("ranked_configs")) or [{}])[0])
    random_baseline = _mapping(section.get("random_baseline"))
    straight = _format_decimal(ranked.get("straight_hit_rate"), digits=4)
    box = _format_decimal(ranked.get("box_hit_rate"), digits=4)
    random_straight = _format_decimal(
        random_baseline.get("straight_hit_rate"),
        digits=4,
    )
    return (
        "開発用ホールドアウト: 未評価\n"
        f"探索期間のストレート率 {straight} / ボックス率 {box} / "
        f"一様ランダムのストレート率 {random_straight}"
    )


def _ai_summary_lines(output: Mapping[str, Any]) -> list[str]:
    lines: list[str] = [
        "開発用ホールドアウトのゲーム間順位は、指標分布が異なるため表示しません。"
    ]
    for game_key in GAME_ORDER:
        report = _statistical_report(output, game_key)
        lines.append(f"{GAME_NAMES[game_key]}: {report.get('one_line_summary')}")
    carryovers = [
        f"{GAME_NAMES[key]} {_carryover_text(output, key)}"
        for key in ("loto6", "loto7") if _has_carryover(output, key)
    ]
    if carryovers:
        lines.append("キャリーオーバー発生中: " + " / ".join(carryovers) + "。")
    previous = _mapping(output.get("previous_evaluation"))
    evaluated = [GAME_NAMES[key] for key in GAME_ORDER if _mapping(previous.get(key)).get("status") == "evaluated"]
    lines.append("前回結果を反映済み: " + "、".join(evaluated) + "。" if evaluated else "今回は前回予想の結果がまだ反映されていません。")
    return lines


def build_notification_summary(output: Mapping[str, Any]) -> str:
    lines = [
        "# LotteryAI 予想・振り返り",
        "",
        f"生成日時: {output.get('generated_at', '不明')}",
        "",
        "## AI総評",
        "",
        *[f"- {line}" for line in _ai_summary_lines(output)],
        "",
    ]

    previous = _mapping(output.get("previous_evaluation"))
    for game_key in GAME_ORDER:
        section = _mapping(output.get(game_key))
        lines.extend([
            "---",
            "",
            f"## {GAME_NAMES[game_key]}",
            "",
            f"次回: 第{section.get('next_draw_no', '不明')}回",
        ])
        if game_key in CARRYOVER_GAMES:
            lines.append(
                f"キャリーオーバー: {_carryover_text(output, game_key)}"
            )
        lines.extend([
            f"採用モデル: {section.get('selected_config', '不明')}",
            "",
            "### 次回予想",
            "",
            *_prediction_lines(game_key, section),
            "",
        ])
        previous_row = _mapping(previous.get(game_key))
        previous_prediction_lines = _previous_prediction_lines(
            game_key,
            previous_row,
        )
        if previous_prediction_lines:
            lines.extend([
                "### 前回予想",
                "",
                *previous_prediction_lines,
                "",
            ])
        lines.extend([
            "### 前回結果",
            "",
            _previous_summary(game_key, previous_row),
            "",
            "### 評価",
            "",
            (
                _lotto_evaluation(section)
                if game_key in LOTO_GAMES
                else _numbers_evaluation(section)
            ),
            "",
            _statistics_text(output, game_key),
        ])
        if game_key in LOTO_GAMES:
            lines.extend(["", f"主要特徴: {_feature_top3(section)}"])

    lines.extend([
        "",
        "---",
        "",
        "処理結果: 正常完了",
        "",
        "※ 過去データに基づく評価であり、当選を保証するものではありません。",
        "※ 開発用ホールドアウトは開発中に参照済みです。最終的な性能評価は実運用結果を優先してください。",
        "",
    ])
    return "\n".join(lines)


def _embed_char_count(embed: Mapping[str, Any]) -> int:
    count = len(str(embed.get("title", ""))) + len(str(embed.get("description", "")))
    footer = _mapping(embed.get("footer"))
    count += len(str(footer.get("text", "")))
    for field in _list(embed.get("fields")):
        row = _mapping(field)
        count += len(str(row.get("name", ""))) + len(str(row.get("value", "")))
    return count


def _game_embed(
    output: Mapping[str, Any],
    game_key: str,
) -> dict[str, object]:
    section = _mapping(output.get(game_key))
    previous = _mapping(_mapping(output.get("previous_evaluation")).get(game_key))
    carryover = _carryover_text(output, game_key)
    fields: list[dict[str, object]] = [
        {
            "name": "次回予想",
            "value": "```\n" + "\n".join(_prediction_lines(game_key, section)) + "\n```",
            "inline": False,
        },
    ]

    previous_prediction_lines = _previous_prediction_lines(game_key, previous)
    if previous_prediction_lines:
        fields.append({
            "name": "前回予想",
            "value": "```\n" + "\n".join(previous_prediction_lines) + "\n```",
            "inline": False,
        })

    fields.extend([
        {
            "name": "前回結果",
            "value": _previous_summary(game_key, previous),
            "inline": False,
        },
        {
            "name": "評価",
            "value": (
                (
                    _lotto_evaluation(section)
                    if game_key in LOTO_GAMES
                    else _numbers_evaluation(section)
                )
                + "\n\n"
                + _statistics_text(output, game_key)
            ),
            "inline": False,
        },
    ])
    if game_key in LOTO_GAMES:
        fields.append({"name": "主要特徴", "value": _feature_top3(section), "inline": False})

    return {
        "title": f"{GAME_NAMES[game_key]}｜第{section.get('next_draw_no', '不明')}回",
        "description": (
            (f"キャリーオーバー: {carryover}\n" if game_key in CARRYOVER_GAMES else "")
            + f"採用モデル: {section.get('selected_config', '不明')}"
        ),
        "color": COLOR_CARRYOVER if _has_carryover(output, game_key) else COLOR_NORMAL,
        "fields": fields,
    }


def build_discord_payload(output: Mapping[str, Any]) -> dict[str, object]:
    any_carryover = any(_has_carryover(output, key) for key in CARRYOVER_GAMES)
    overview = {
        "title": "LotteryAI 予想・振り返り",
        "description": "\n".join(f"• {line}" for line in _ai_summary_lines(output)),
        "color": COLOR_CARRYOVER if any_carryover else COLOR_NORMAL,
        "footer": {
            "text": (
                "処理結果: 正常完了 / "
                "過去データに基づく評価であり、当選を保証しません。 "
                "ホールドアウトは開発中に参照済みのため、実運用結果を優先します。"
            )
        },
    }
    embeds = [overview, *[_game_embed(output, key) for key in GAME_ORDER]]

    messages: list[dict[str, object]] = []
    current: list[dict[str, object]] = []
    current_chars = 0
    for embed in embeds:
        size = _embed_char_count(embed)
        if current and (len(current) >= 10 or current_chars + size > DISCORD_EMBED_TOTAL_LIMIT):
            messages.append({
                "username": "LotteryAI",
                "allowed_mentions": {"parse": []},
                "embeds": current,
            })
            current = []
            current_chars = 0
        current.append(embed)
        current_chars += size
    if current:
        messages.append({
            "username": "LotteryAI",
            "allowed_mentions": {"parse": []},
            "embeds": current,
        })

    return {
        "schema_version": "2.0",
        "generated_at": output.get("generated_at"),
        "has_carryover": any_carryover,
        "messages": messages,
    }


def write_notification_summary(output_dir: Path, output: dict) -> Path:
    history = load_json(output_dir / "evaluation_history.json", default=[])
    history = history if isinstance(history, list) else []
    enriched_output = dict(output)
    enriched_output["statistical_reports"] = {
        game_key: build_game_statistical_report(
            game_key,
            _mapping(output.get(game_key)),
            history,
        )
        for game_key in GAME_ORDER
    }
    markdown_path = output_dir / "notification_summary.md"
    markdown_path.write_text(build_notification_summary(enriched_output), encoding="utf-8")
    save_json(
        output_dir / "notification_payload.json",
        build_discord_payload(enriched_output),
    )
    return markdown_path


__all__ = [
    "build_discord_payload",
    "build_notification_summary",
    "write_notification_summary",
]
