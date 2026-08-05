"""Paired statistical summaries for LotteryAI evaluation output."""
from __future__ import annotations

from datetime import datetime
from math import comb
from random import Random
from statistics import mean
from typing import Any, Mapping, Sequence


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def data_volume_label(sample_size: int) -> str:
    if sample_size <= 0:
        return "未評価"
    if sample_size < 30:
        return f"標本数が少ない（{sample_size}回）"
    if sample_size < 100:
        return f"暫定評価（{sample_size}回）"
    return f"継続評価可能（{sample_size}回）"


def _bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 20260805,
) -> tuple[float, float] | None:
    if not differences:
        return None
    if len(differences) == 1:
        value = float(differences[0])
        return value, value

    rng = Random(seed)
    size = len(differences)
    boot = []
    for _ in range(iterations):
        boot.append(mean(differences[rng.randrange(size)] for _ in range(size)))
    boot.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(iterations - 1, int(alpha * iterations)))
    upper_index = max(0, min(iterations - 1, int((1.0 - alpha) * iterations) - 1))
    return round(float(boot[lower_index]), 6), round(float(boot[upper_index]), 6)


def _two_sided_sign_test_p_value(differences: Sequence[float]) -> float | None:
    non_zero = [value for value in differences if value != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0 if differences else None
    positives = sum(value > 0 for value in non_zero)
    tail = min(positives, n - positives)
    probability = sum(comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return round(min(1.0, 2.0 * probability), 6)


def paired_difference_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    baseline_key: str,
    bootstrap_seed: int = 20260805,
) -> dict[str, Any]:
    differences: list[float] = []
    wins = ties = losses = 0
    for row in rows:
        raw_model = row.get(model_key)
        raw_baseline = row.get(baseline_key)
        model = float(raw_model) if isinstance(raw_model, bool) else _number(raw_model)
        baseline = float(raw_baseline) if isinstance(raw_baseline, bool) else _number(raw_baseline)
        if model is None or baseline is None:
            continue
        difference = model - baseline
        differences.append(difference)
        if difference > 0:
            wins += 1
        elif difference < 0:
            losses += 1
        else:
            ties += 1

    sample_size = len(differences)
    if not differences:
        return {
            "status": "unavailable",
            "sample_size": 0,
            "data_volume": data_volume_label(0),
            "mean_difference": None,
            "confidence_interval_95": None,
            "p_value_reference": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "judgement": "評価データなし",
        }

    average = round(float(mean(differences)), 6)
    interval = _bootstrap_mean_ci(differences, seed=bootstrap_seed)
    p_value = _two_sided_sign_test_p_value(differences)
    lower, upper = interval if interval is not None else (None, None)

    if lower is not None and lower > 0:
        judgement = "今回の評価期間では正の差を検出"
    elif upper is not None and upper < 0:
        judgement = "今回の評価期間では負の差を検出"
    else:
        judgement = "現時点では差を特定できない"

    return {
        "status": "evaluated",
        "sample_size": sample_size,
        "data_volume": data_volume_label(sample_size),
        "mean_difference": average,
        "confidence_interval_95": {
            "lower": lower,
            "upper": upper,
            "method": "paired_bootstrap_percentile",
            "iterations": 10_000,
        },
        "p_value_reference": {
            "value": p_value,
            "method": "two_sided_exact_sign_test",
            "note": "補助情報。主要判定は95%信頼区間を使用。",
        },
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "judgement": judgement,
    }


def operational_evaluation_info(
    history: Sequence[Mapping[str, Any]],
    game_key: str,
) -> dict[str, Any]:
    rows = [
        row for row in history
        if row.get("draw_type") == game_key and row.get("status") == "evaluated"
    ]
    dates: list[datetime] = []
    for row in rows:
        raw = row.get("evaluated_at")
        if not isinstance(raw, str):
            continue
        try:
            dates.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except ValueError:
            continue
    start = min(dates).date().isoformat() if dates else None
    latest = max(dates).date().isoformat() if dates else None
    return {
        "evaluated_draws": len(rows),
        "started_at": start,
        "latest_evaluated_at": latest,
    }


def build_game_statistical_report(
    game_key: str,
    section: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if game_key in {"loto6", "loto7", "miniloto"}:
        evaluation = section.get("final_candidate_holdout")
        if not isinstance(evaluation, Mapping):
            evaluation = section.get("holdout_evaluation")
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        rows = evaluation.get("paired_draw_results", [])
        rows = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
        paired = paired_difference_summary(
            rows,
            model_key="model_best_match_count",
            baseline_key="uniform_best_match_count",
        )
        metric = "5口中の最大一致数"
        baseline = "一様ランダム"
    else:
        holdout = section.get("holdout_evaluation")
        holdout = holdout if isinstance(holdout, Mapping) else {}
        box = holdout.get("box_dedicated_evaluation")
        box = box if isinstance(box, Mapping) else {}
        rows = box.get("paired_draw_results", [])
        rows = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
        paired = paired_difference_summary(
            rows,
            model_key="model_box_hit",
            baseline_key="random_box_hit",
        )
        metric = "BOX専用候補の的中（0/1）"
        baseline = "BOX専用ランダム"

    operational = operational_evaluation_info(history, game_key)
    summary = (
        f"{paired['judgement']} / 評価{paired['sample_size']}回"
        f"・実運用{operational['evaluated_draws']}回"
    )
    return {
        "game_key": game_key,
        "metric": metric,
        "baseline": baseline,
        "paired_evaluation": paired,
        "operational_evaluation": operational,
        "one_line_summary": summary,
        "holdout_note": (
            "ホールドアウトは開発中に参照済みです。"
            "最終的な性能評価は実運用結果を優先してください。"
        ),
    }


__all__ = [
    "build_game_statistical_report",
    "data_volume_label",
    "operational_evaluation_info",
    "paired_difference_summary",
]
