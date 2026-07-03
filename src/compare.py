from pathlib import Path
import json
import math
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd

from main import (
    LOTO6_URL,
    LOTO7_URL,
    download_text,
    read_csv_text,
    normalize_loto6,
    normalize_loto7,
    validate_lottery,
    build_model_context,
    score_candidate,
    shape_score,
    select_diverse_top,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"

MODEL_PRESETS = {
    "random": {
        "freq": 0.00,
        "recent": 0.00,
        "pair": 0.00,
        "triplet": 0.00,
        "delay": 0.00,
        "dist": 0.00,
        "sampling": "uniform",
    },
    "global_frequency": {
        "freq": 1.00,
        "recent": 0.00,
        "pair": 0.00,
        "triplet": 0.00,
        "delay": 0.00,
        "dist": 0.00,
        "sampling": "global",
    },
    "recent_frequency": {
        "freq": 0.00,
        "recent": 1.00,
        "pair": 0.00,
        "triplet": 0.00,
        "delay": 0.00,
        "dist": 0.00,
        "sampling": "recent",
    },
    "delay": {
        "freq": 0.00,
        "recent": 0.00,
        "pair": 0.00,
        "triplet": 0.00,
        "delay": 1.00,
        "dist": 0.00,
        "sampling": "delay",
    },
    "pair": {
        "freq": 0.10,
        "recent": 0.10,
        "pair": 0.70,
        "triplet": 0.05,
        "delay": 0.00,
        "dist": 0.05,
        "sampling": "hybrid",
    },
    "hybrid_v1": {
        "freq": 0.22,
        "recent": 0.24,
        "pair": 0.22,
        "triplet": 0.08,
        "delay": 0.08,
        "dist": 0.16,
        "sampling": "hybrid",
    },
    "hybrid_no_delay": {
        "freq": 0.24,
        "recent": 0.26,
        "pair": 0.24,
        "triplet": 0.08,
        "delay": 0.00,
        "dist": 0.18,
        "sampling": "hybrid",
    },
}

def local_consecutive_count(nums: tuple[int, ...]) -> int:
    nums = sorted(nums)
    return sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)


def local_block_counts(nums: tuple[int, ...], max_num: int) -> list[int]:
    if max_num == 43:
        blocks = [(1, 10), (11, 21), (22, 32), (33, 43)]
    else:
        blocks = [(1, 9), (10, 18), (19, 27), (28, 37)]

    return [
        sum(1 for n in nums if lo <= n <= hi)
        for lo, hi in blocks
    ]


def is_reasonable_candidate(
    nums: tuple[int, ...],
    max_num: int,
    pick_count: int
) -> bool:
    """
    v4:
    極端な偏りを除外する。
    特にロト7で1桁数字が多すぎる、連番が多すぎる、同一ブロックに寄りすぎる候補を除外。
    """
    nums = tuple(sorted(nums))

    block_counts = local_block_counts(nums, max_num)

    odd_count = sum(n % 2 for n in nums)
    low_count = sum(n <= (max_num // 2) for n in nums)
    con_count = local_consecutive_count(nums)

    # 1つのブロックに4個以上集中する候補は除外
    if max(block_counts) >= 4:
        return False

    if pick_count == 6:
        # ロト6：奇偶・低高が極端な候補を除外
        if odd_count not in [2, 3, 4]:
            return False
        if low_count not in [2, 3, 4]:
            return False

        # ロト6も連番2組以上は除外
        if con_count >= 2:
            return False

    if pick_count == 7:
        # ロト7：1〜9ブロックは最大2個までに制限
        # 例：[3,4,5,...] のような形を避ける
        if block_counts[0] >= 3:
            return False

        # ロト7：奇偶・低高が極端な候補を除外
        if odd_count not in [3, 4]:
            return False
        if low_count not in [3, 4]:
            return False

        # ロト7：連番2組以上を除外
        # 3-4-5 のような3連番は con_count=2 になるため除外される
        if con_count >= 2:
            return False

    return True
    
    def select_diverse_top(scored: list[dict], pick_count: int, top_k: int) -> list[dict]:
    """
    v4:
    上位スコアだけでなく、5口同士が似すぎないようにする。
    ロト6は共通数字3個まで、ロト7も共通数字3個まで。
    """
    selected = []

    max_common = 3

    for item in scored:
        nums = set(item["numbers"])

        if all(len(nums & set(existing["numbers"])) <= max_common for existing in selected):
            selected.append(item)

        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for item in scored:
            if item not in selected:
                selected.append(item)
            if len(selected) >= top_k:
                break

    return selected

def component_scores(nums: tuple[int, ...], max_num: int, ctx: dict) -> dict:
    freq_score = float(np.mean([ctx["global_norm"][n] for n in nums]))
    recent_score = float(np.mean([ctx["recent_norm"][n] for n in nums]))
    delay_score = float(np.mean([ctx["delay_norm"][n] for n in nums]))

    pair_values = [
        ctx["pairs"].get(tuple(sorted(pair)), 0) / ctx["max_pair"]
        for pair in combinations(nums, 2)
    ]
    pair_score = float(np.mean(pair_values)) if pair_values else 0.0

    triple_values = [
        ctx["triples"].get(tuple(sorted(triple)), 0) / ctx["max_triple"]
        for triple in combinations(nums, 3)
    ]
    triplet_score = float(np.mean(triple_values)) if triple_values else 0.0

    dist_score = float(shape_score(nums, max_num, ctx["shape_stats"]))

    return {
        "freq": freq_score,
        "recent": recent_score,
        "pair": pair_score,
        "triplet": triplet_score,
        "delay": delay_score,
        "dist": dist_score,
    }


def model_score(nums: tuple[int, ...], max_num: int, ctx: dict, model_name: str) -> float:
    preset = MODEL_PRESETS[model_name]

    if model_name == "random":
        return 0.0

    comps = component_scores(nums, max_num, ctx)

    return float(
        preset["freq"] * comps["freq"]
        + preset["recent"] * comps["recent"]
        + preset["pair"] * comps["pair"]
        + preset["triplet"] * comps["triplet"]
        + preset["delay"] * comps["delay"]
        + preset["dist"] * comps["dist"]
    )


def sampling_probabilities(ctx: dict, min_num: int, max_num: int, model_name: str) -> np.ndarray:
    preset = MODEL_PRESETS[model_name]
    sampling = preset["sampling"]

    values = []

    for n in range(min_num, max_num + 1):
        if sampling == "uniform":
            v = 1.0
        elif sampling == "global":
            v = ctx["global_norm"][n]
        elif sampling == "recent":
            v = ctx["recent_norm"][n]
        elif sampling == "delay":
            v = ctx["delay_norm"][n]
        else:
            v = (
                0.35 * ctx["global_norm"][n]
                + 0.40 * ctx["recent_norm"][n]
                + 0.25 * ctx["delay_norm"][n]
            )

        values.append(max(float(v), 0.0001))

    arr = np.array(values, dtype=float)
    arr = arr ** 1.25
    return arr / arr.sum()


def generate_model_candidates(
    pick_count: int,
    min_num: int,
    max_num: int,
    ctx: dict,
    model_name: str,
    candidate_count: int,
    seed: int,
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)

    numbers = np.array(list(range(min_num, max_num + 1)))
    probabilities = sampling_probabilities(ctx, min_num, max_num, model_name)

    candidates = set()
    attempts = 0
    max_attempts = candidate_count * 20

    while len(candidates) < candidate_count and attempts < max_attempts:
        selected = rng.choice(numbers, size=pick_count, replace=False, p=probabilities)
        candidates.add(tuple(sorted(int(n) for n in selected)))
        attempts += 1

    return list(candidates)


def predict_by_model(
    df: pd.DataFrame,
    main_cols: list[str],
    min_num: int,
    max_num: int,
    pick_count: int,
    model_name: str,
    candidate_count: int,
    seed: int,
    top_k: int = 5,
) -> list[dict]:
    ctx = build_model_context(df, main_cols, min_num, max_num)

    candidates = generate_model_candidates(
        pick_count=pick_count,
        min_num=min_num,
        max_num=max_num,
        ctx=ctx,
        model_name=model_name,
        candidate_count=candidate_count,
        seed=seed,
    )

    # v3: 極端な分布の候補を除外
    filtered_candidates = [
        nums for nums in candidates
        if is_reasonable_candidate(nums, max_num, pick_count)
    ]

    # フィルターで候補が減りすぎた場合だけ、元候補に戻す
    if len(filtered_candidates) >= top_k:
        candidates_to_score = filtered_candidates
    else:
        candidates_to_score = candidates

    estimated_probability = 1 / math.comb(max_num, pick_count)

    scored = []
    for nums in candidates_to_score:
        if model_name == "random":
            score = 0.0
            detail = component_scores(nums, max_num, ctx)
        else:
            score = model_score(nums, max_num, ctx, model_name)
            detail = component_scores(nums, max_num, ctx)

        scored.append({
            "numbers": list(nums),
            "raw_tuple": nums,
            "score": float(score),
            "score_detail": detail,
            "block_counts": local_block_counts(nums, max_num),
            "consecutive_count": local_consecutive_count(nums),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = select_diverse_top(scored, pick_count, top_k)

    output = []
    for i, item in enumerate(selected, start=1):
        output.append({
            "pattern_id": f"P{i}",
            "numbers": item["numbers"],
            "score": round(float(item["score"]), 6),
            "estimated_probability": estimated_probability,
            "model": model_name,
            "block_counts": item["block_counts"],
            "consecutive_count": item["consecutive_count"],
            "score_detail": {
                k: round(float(v), 6)
                for k, v in item["score_detail"].items()
            },
        })

    return output


def backtest_one_model(
    df: pd.DataFrame,
    main_cols: list[str],
    min_num: int,
    max_num: int,
    pick_count: int,
    model_name: str,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seed: int,
) -> dict:
    if len(df) <= train_window + 5:
        return {
            "model": model_name,
            "tested_periods": 0,
            "avg_matches": None,
            "hit_rate_1match": None,
            "hit_rate_2match": None,
            "hit_rate_3match": None,
        }

    start = max(train_window, len(df) - tested_periods)
    matches = []

    for test_idx in range(start, len(df)):
        train = df.iloc[:test_idx].copy()
        actual = set(df.iloc[test_idx][main_cols].astype(int).tolist())

        pred = predict_by_model(
            df=train,
            main_cols=main_cols,
            min_num=min_num,
            max_num=max_num,
            pick_count=pick_count,
            model_name=model_name,
            candidate_count=candidate_count,
            seed=seed + test_idx,
            top_k=1,
        )

        predicted = set(pred[0]["numbers"])
        matches.append(len(predicted & actual))

    tested = len(matches)

    return {
        "model": model_name,
        "tested_periods": int(tested),
        "avg_matches": round(float(np.mean(matches)), 4) if matches else None,
        "hit_rate_1match": round(sum(m >= 1 for m in matches) / tested, 4) if tested else None,
        "hit_rate_2match": round(sum(m >= 2 for m in matches) / tested, 4) if tested else None,
        "hit_rate_3match": round(sum(m >= 3 for m in matches) / tested, 4) if tested else None,
        "hit_rate_4match": round(sum(m >= 4 for m in matches) / tested, 4) if tested else None,
        "hit_rate_5match": round(sum(m >= 5 for m in matches) / tested, 4) if tested else None,
    }


def compare_models(
    df: pd.DataFrame,
    main_cols: list[str],
    min_num: int,
    max_num: int,
    pick_count: int,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seed: int,
) -> list[dict]:
    results = []

    for model_name in MODEL_PRESETS.keys():
        result = backtest_one_model(
            df=df,
            main_cols=main_cols,
            min_num=min_num,
            max_num=max_num,
            pick_count=pick_count,
            model_name=model_name,
            train_window=train_window,
            tested_periods=tested_periods,
            candidate_count=candidate_count,
            seed=seed,
        )
        results.append(result)

    results.sort(
        key=lambda x: (
            x["avg_matches"] if x["avg_matches"] is not None else -1,
            x["hit_rate_3match"] if x["hit_rate_3match"] is not None else -1,
            x["hit_rate_2match"] if x["hit_rate_2match"] is not None else -1,
        ),
        reverse=True,
    )

    return results


def select_prediction_model(model_comparison: list[dict]) -> str:
    """
    v4:
    avg_matches だけで採用せず、2個以上・3個以上一致率も加味する。
    delay単独は偏りやすいので、明確に優位な場合以外はhybrid系を優先する。
    """
    random_avg = None
    for item in model_comparison:
        if item["model"] == "random":
            random_avg = item["avg_matches"]
            break

    candidates = []

    for item in model_comparison:
        model = item["model"]

        if model == "random":
            continue

        avg = item["avg_matches"] or 0.0
        hit2 = item["hit_rate_2match"] or 0.0
        hit3 = item["hit_rate_3match"] or 0.0
        hit4 = item["hit_rate_4match"] or 0.0

        score = (
            avg
            + 0.25 * hit2
            + 0.70 * hit3
            + 1.00 * hit4
        )

        # ハイブリッドは偏りにくいので加点
        if model in ["hybrid_v1", "hybrid_no_delay"]:
            score += 0.06

        # delay単独は偏りやすいので減点
        if model == "delay":
            score -= 0.12

        # ランダムより明確に悪い場合は減点
        if random_avg is not None and avg < random_avg - 0.05:
            score -= 0.25

        candidates.append({
            "model": model,
            "selection_score": score,
            "avg_matches": avg,
            "hit_rate_2match": hit2,
            "hit_rate_3match": hit3,
            "hit_rate_4match": hit4,
        })

    candidates.sort(key=lambda x: x["selection_score"], reverse=True)

    if not candidates:
        return "hybrid_no_delay"

    best = candidates[0]

    # delayが1位でも、hybrid系との差が小さいならhybrid系を採用
    if best["model"] == "delay":
        hybrid_candidates = [
            c for c in candidates
            if c["model"] in ["hybrid_no_delay", "hybrid_v1"]
        ]

        if hybrid_candidates:
            hybrid_best = max(hybrid_candidates, key=lambda x: x["selection_score"])

            # delayの平均一致優位が0.18未満なら、偏り回避でhybridを採用
            if best["avg_matches"] - hybrid_best["avg_matches"] < 0.18:
                return hybrid_best["model"]

    return best["model"]


def load_lottery_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    loto6_text = download_text(LOTO6_URL)
    loto7_text = download_text(LOTO7_URL)

    loto6_raw = read_csv_text(loto6_text)
    loto7_raw = read_csv_text(loto7_text)

    loto6 = normalize_loto6(loto6_raw)
    loto7 = normalize_loto7(loto7_raw)

    return loto6, loto7


def print_model_table(title: str, results: list[dict]) -> None:
    print(f"\n=== {title} MODEL COMPARISON ===")
    for r in results:
        print(
            f'{r["model"]}: '
            f'avg={r["avg_matches"]}, '
            f'1+={r["hit_rate_1match"]}, '
            f'2+={r["hit_rate_2match"]}, '
            f'3+={r["hit_rate_3match"]}, '
            f'4+={r["hit_rate_4match"]}'
        )


def print_predictions(title: str, predictions: list[dict]) -> None:
    print(f"\n=== {title} NEXT PREDICTION ===")
    for p in predictions:
        print(f'{p["pattern_id"]}: {p["numbers"]} score={p["score"]} model={p["model"]}')


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loto6, loto7 = load_lottery_data()

    loto6_main_cols = ["main1", "main2", "main3", "main4", "main5", "main6"]
    loto7_main_cols = ["main1", "main2", "main3", "main4", "main5", "main6", "main7"]

    loto6_validation = validate_lottery(loto6, loto6_main_cols, ["bonus"], 1, 43)
    loto7_validation = validate_lottery(loto7, loto7_main_cols, ["bonus1", "bonus2"], 1, 37)

    loto6_model_comparison = compare_models(
        df=loto6,
        main_cols=loto6_main_cols,
        min_num=1,
        max_num=43,
        pick_count=6,
        train_window=500,
        tested_periods=60,
        candidate_count=400,
        seed=2025,
    )

    loto7_model_comparison = compare_models(
        df=loto7,
        main_cols=loto7_main_cols,
        min_num=1,
        max_num=37,
        pick_count=7,
        train_window=240,
        tested_periods=60,
        candidate_count=400,
        seed=2025,
    )

    loto6_selected_model = select_prediction_model(loto6_model_comparison)
    loto7_selected_model = select_prediction_model(loto7_model_comparison)

    loto6_prediction = predict_by_model(
        df=loto6,
        main_cols=loto6_main_cols,
        min_num=1,
        max_num=43,
        pick_count=6,
        model_name=loto6_selected_model,
        candidate_count=10000,
        seed=2025,
        top_k=5,
    )

    loto7_prediction = predict_by_model(
        df=loto7,
        main_cols=loto7_main_cols,
        min_num=1,
        max_num=37,
        pick_count=7,
        model_name=loto7_selected_model,
        candidate_count=10000,
        seed=2025,
        top_k=5,
    )

    result = {
        "status": "ok",
        "note": "v2 compares random, frequency, recent, delay, pair, and hybrid models. Prediction uses the best non-random model from recent roll-forward backtest.",
        "loto6": {
            "latest_draw_no": loto6_validation["latest_draw_no"],
            "next_draw_no": loto6_validation["latest_draw_no"] + 1,
            "rows": loto6_validation["rows"],
            "validation": loto6_validation,
            "model_comparison": loto6_model_comparison,
            "selected_model": loto6_selected_model,
            "prediction": loto6_prediction,
        },
        "loto7": {
            "latest_draw_no": loto7_validation["latest_draw_no"],
            "next_draw_no": loto7_validation["latest_draw_no"] + 1,
            "rows": loto7_validation["rows"],
            "validation": loto7_validation,
            "model_comparison": loto7_model_comparison,
            "selected_model": loto7_selected_model,
            "prediction": loto7_prediction,
        },
    }

    (OUTPUT_DIR / "model_comparison_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "prediction_v2_loto6.json").write_text(
        json.dumps(loto6_prediction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "prediction_v2_loto7.json").write_text(
        json.dumps(loto7_prediction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f'LOTO6 latest={result["loto6"]["latest_draw_no"]} next={result["loto6"]["next_draw_no"]} rows={result["loto6"]["rows"]}')
    print_model_table("LOTO6", loto6_model_comparison)
    print(f'LOTO6 selected_model={loto6_selected_model}')
    print_predictions("LOTO6", loto6_prediction)

    print(f'\nLOTO7 latest={result["loto7"]["latest_draw_no"]} next={result["loto7"]["next_draw_no"]} rows={result["loto7"]["rows"]}')
    print_model_table("LOTO7", loto7_model_comparison)
    print(f'LOTO7 selected_model={loto7_selected_model}')
    print_predictions("LOTO7", loto7_prediction)

    print("\n=== SHORT JSON ===")
    print(json.dumps({
        "status": "ok",
        "loto6_latest_draw_no": result["loto6"]["latest_draw_no"],
        "loto6_next_draw_no": result["loto6"]["next_draw_no"],
        "loto6_selected_model": loto6_selected_model,
        "loto6_prediction": [p["numbers"] for p in loto6_prediction],
        "loto6_model_comparison": loto6_model_comparison,
        "loto7_latest_draw_no": result["loto7"]["latest_draw_no"],
        "loto7_next_draw_no": result["loto7"]["next_draw_no"],
        "loto7_selected_model": loto7_selected_model,
        "loto7_prediction": [p["numbers"] for p in loto7_prediction],
        "loto7_model_comparison": loto7_model_comparison,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()