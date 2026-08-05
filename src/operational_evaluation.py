from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from common import now_iso


def _numbers(row: object) -> list[int]:
    if isinstance(row, Mapping):
        row = row.get("numbers", row.get("digits", []))
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        return []
    return [int(value) for value in row]


def _best_lotto(predictions: object, actual: Sequence[int]) -> dict[str, object]:
    rows = predictions if isinstance(predictions, list) else []
    actual_set = set(int(value) for value in actual)
    matches = [len(actual_set & set(_numbers(row))) for row in rows]
    return {
        "ticket_count": len(matches),
        "best_match_count": max(matches, default=0),
        "average_match_count": round(sum(matches) / len(matches), 6) if matches else 0.0,
        "match_counts": matches,
    }


def _unordered_matches(left: Sequence[int], right: Sequence[int]) -> int:
    a = Counter(int(value) for value in left)
    b = Counter(int(value) for value in right)
    return sum(min(a[key], b[key]) for key in a.keys() | b.keys())


def _best_numbers(predictions: object, actual: Sequence[int]) -> dict[str, object]:
    rows = predictions if isinstance(predictions, list) else []
    digits = [_numbers(row) for row in rows]
    positional = [sum(a == b for a, b in zip(row, actual)) for row in digits]
    unordered = [_unordered_matches(row, actual) for row in digits]
    return {
        "ticket_count": len(digits),
        "best_position_match_count": max(positional, default=0),
        "average_position_match_count": round(sum(positional) / len(positional), 6) if positional else 0.0,
        "best_unordered_match_count": max(unordered, default=0),
        "average_unordered_match_count": round(sum(unordered) / len(unordered), 6) if unordered else 0.0,
        "straight_hit": any(row == list(actual) for row in digits),
        "box_hit": any(sorted(row) == sorted(actual) for row in digits),
    }


def evaluate_operational_controls(
    controls_store: Mapping[str, object],
    datasets: Mapping[str, object],
    game_configs: Mapping[str, Mapping[str, object]],
    existing_history: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Evaluate previously frozen controls whose target draw is now available."""
    merged = [dict(item) for item in existing_history if isinstance(item, Mapping)]
    seen = {
        (item.get("game_key"), item.get("target_draw_no"), item.get("evaluation_epoch"))
        for item in merged
    }
    games = controls_store.get("games", {})
    if not isinstance(games, Mapping):
        return merged

    for game_key, control_obj in games.items():
        if game_key not in datasets or game_key not in game_configs or not isinstance(control_obj, Mapping):
            continue
        control = dict(control_obj)
        target = control.get("target_draw_no")
        if target is None:
            continue
        epoch = control.get("evaluation_epoch")
        key = (game_key, int(target), epoch)
        if key in seen:
            continue

        df = datasets[game_key]
        hit = df[df["draw_no"] == int(target)]
        if hit.empty:
            continue
        config = game_configs[game_key]
        actual = [int(hit.iloc[0][column]) for column in config["main_cols"]]
        family = str(config.get("family", "lotto")).lower()
        record: dict[str, object] = {
            "game_key": game_key,
            "target_draw_no": int(target),
            "evaluation_epoch": epoch,
            "model_version": control.get("model_version"),
            "generated_at": control.get("generated_at"),
            "evaluated_at": now_iso(),
            "generated_before_draw": bool(control.get("generated_before_draw")),
            "actual_numbers": actual,
        }
        if family == "numbers":
            model = _best_numbers(control.get("model_prediction", []), actual)
            uniform = _best_numbers(control.get("uniform_random_control", []), actual)
            model_box = _best_numbers(control.get("model_box_prediction", []), actual)
            matched_box = _best_numbers(control.get("composition_matched_random_box_control", []), actual)
            record.update({
                "model": model,
                "uniform_random_control": uniform,
                "model_box": model_box,
                "composition_matched_random_box_control": matched_box,
                "model_minus_uniform_best_position": model["best_position_match_count"] - uniform["best_position_match_count"],
                "model_box_minus_control_hit": int(bool(model_box["box_hit"])) - int(bool(matched_box["box_hit"])),
            })
        else:
            model = _best_lotto(control.get("model_prediction", []), actual)
            uniform = _best_lotto(control.get("uniform_random_control", []), actual)
            filtered = _best_lotto(control.get("filtered_random_control", []), actual)
            record.update({
                "model": model,
                "uniform_random_control": uniform,
                "filtered_random_control": filtered,
                "model_minus_uniform_best": model["best_match_count"] - uniform["best_match_count"],
                "model_minus_filtered_best": model["best_match_count"] - filtered["best_match_count"],
            })
        merged.append(record)
        seen.add(key)

    merged.sort(key=lambda row: (int(row.get("evaluation_epoch") or 0), str(row.get("game_key", "")), int(row.get("target_draw_no") or 0)))
    return merged
