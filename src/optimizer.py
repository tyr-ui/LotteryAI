from pathlib import Path
import json
import math
from itertools import combinations

import numpy as np
import pandas as pd

from games import LOTTO_GAMES
from main import (
    download_game_csv,
    read_csv_text,
    normalize_loto6,
    normalize_loto7,
    normalize_miniloto,
    validate_lottery,
    build_model_context,
    shape_score,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
SEED = 2025

# 自動探索する設定。重み + 分布制約をセットで比較する。
CONFIGS = [
    {"name": "balanced_strict", "w": {"freq": .22, "recent": .24, "pair": .22, "triplet": .08, "delay": .08, "dist": .16}, "s": {"g": .35, "r": .40, "d": .25}, "f": {"max_block": 3, "max_first": 2, "max_con": 1, "max_common": 3}},
    {"name": "balanced_loose", "w": {"freq": .22, "recent": .24, "pair": .22, "triplet": .08, "delay": .08, "dist": .16}, "s": {"g": .35, "r": .40, "d": .25}, "f": {"max_block": 3, "max_first": 3, "max_con": 2, "max_common": 4}},
    {"name": "no_delay_strict", "w": {"freq": .24, "recent": .26, "pair": .24, "triplet": .08, "delay": .00, "dist": .18}, "s": {"g": .45, "r": .55, "d": .00}, "f": {"max_block": 3, "max_first": 2, "max_con": 1, "max_common": 3}},
    {"name": "no_delay_loose", "w": {"freq": .24, "recent": .26, "pair": .24, "triplet": .08, "delay": .00, "dist": .18}, "s": {"g": .45, "r": .55, "d": .00}, "f": {"max_block": 3, "max_first": 3, "max_con": 2, "max_common": 4}},
    {"name": "freq_pair_strict", "w": {"freq": .30, "recent": .18, "pair": .28, "triplet": .06, "delay": .00, "dist": .18}, "s": {"g": .65, "r": .35, "d": .00}, "f": {"max_block": 3, "max_first": 2, "max_con": 1, "max_common": 3}},
    {"name": "recent_pair_strict", "w": {"freq": .14, "recent": .34, "pair": .26, "triplet": .06, "delay": .00, "dist": .20}, "s": {"g": .25, "r": .75, "d": .00}, "f": {"max_block": 3, "max_first": 2, "max_con": 1, "max_common": 3}},
    {"name": "delay_light_strict", "w": {"freq": .18, "recent": .20, "pair": .20, "triplet": .05, "delay": .15, "dist": .22}, "s": {"g": .30, "r": .35, "d": .35}, "f": {"max_block": 3, "max_first": 2, "max_con": 1, "max_common": 3}},
    {"name": "dist_heavy_strict", "w": {"freq": .18, "recent": .20, "pair": .20, "triplet": .04, "delay": .04, "dist": .34}, "s": {"g": .40, "r": .45, "d": .15}, "f": {"max_block": 3, "max_first": 2, "max_con": 1, "max_common": 3}},
]


def consecutive_count(nums):
    nums = sorted(nums)
    return sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)


def block_counts(nums, max_num):
    game_config = next(
        (
            config
            for config in LOTTO_GAMES.values()
            if config["max_num"] == max_num
        ),
        None,
    )

    if game_config is None:
        raise ValueError(f"Unsupported max_num: {max_num}")

    return [
        sum(1 for n in nums if lo <= n <= hi)
        for lo, hi in game_config["block_ranges"]
    ]


def is_reasonable(nums, max_num, pick_count, cfg):
    nums = tuple(sorted(nums))

    game_config = next(
        (
            config
            for config in LOTTO_GAMES.values()
            if config["max_num"] == max_num
            and config["pick_count"] == pick_count
        ),
        None,
    )

    if game_config is None:
        raise ValueError(
            f"Unsupported lottery settings: "
            f"max_num={max_num}, pick_count={pick_count}"
        )

    filters = cfg["f"]
    blocks = block_counts(nums, max_num)

    if max(blocks) > filters["max_block"]:
        return False

    if pick_count == 7 and blocks[0] > filters["max_first"]:
        return False

    if consecutive_count(nums) > filters["max_con"]:
        return False

    odd_count = sum(n % 2 for n in nums)
    low_count = sum(n <= max_num // 2 for n in nums)

    return (
        odd_count in game_config["allowed_odd_counts"]
        and low_count in game_config["allowed_low_counts"]
    )


def component_scores(nums, max_num, ctx):
    pairs = [ctx["pairs"].get(tuple(sorted(p)), 0) / ctx["max_pair"] for p in combinations(nums, 2)]
    triples = [ctx["triples"].get(tuple(sorted(t)), 0) / ctx["max_triple"] for t in combinations(nums, 3)]
    return {
        "freq": float(np.mean([ctx["global_norm"][n] for n in nums])),
        "recent": float(np.mean([ctx["recent_norm"][n] for n in nums])),
        "delay": float(np.mean([ctx["delay_norm"][n] for n in nums])),
        "pair": float(np.mean(pairs)) if pairs else 0.0,
        "triplet": float(np.mean(triples)) if triples else 0.0,
        "dist": float(shape_score(tuple(nums), max_num, ctx["shape_stats"])),
    }


def sampling_probs(ctx, min_num, max_num, cfg=None, uniform=False):
    vals = []
    for n in range(min_num, max_num + 1):
        if uniform or cfg is None:
            v = 1.0
        else:
            s = cfg["s"]
            v = s["g"] * ctx["global_norm"][n] + s["r"] * ctx["recent_norm"][n] + s["d"] * ctx["delay_norm"][n]
        vals.append(max(v, 0.0001))
    arr = np.array(vals, dtype=float) ** 1.25
    return arr / arr.sum()


def generate_candidates(pick_count, min_num, max_num, ctx, cfg, candidate_count, seed, uniform=False):
    rng = np.random.default_rng(seed)
    numbers = np.array(list(range(min_num, max_num + 1)))
    probs = sampling_probs(ctx, min_num, max_num, cfg, uniform=uniform)

    candidates = set()
    attempts = 0
    while len(candidates) < candidate_count and attempts < candidate_count * 80:
        nums = tuple(sorted(int(n) for n in rng.choice(numbers, size=pick_count, replace=False, p=probs)))
        if uniform or cfg is None or is_reasonable(nums, max_num, pick_count, cfg):
            candidates.add(nums)
        attempts += 1
    return list(candidates)


def select_diverse(scored, top_k, max_common, max_number_usage=3):
    """
    Select top candidates while controlling portfolio concentration.

    max_common:
        Maximum shared numbers between any two patterns.

    max_number_usage:
        Maximum times the same number can appear across all selected patterns.
        Example: max_number_usage=3 means number 6 can appear in at most 3 of P1-P5.
    """
    selected = []
    usage = {}

    def can_add(item, strict_usage=True):
        nums = set(item["numbers"])

        # Pairwise overlap control.
        for existing in selected:
            if len(nums & set(existing["numbers"])) > max_common:
                return False

        # Whole-portfolio number concentration control.
        if strict_usage:
            for n in nums:
                if usage.get(n, 0) >= max_number_usage:
                    return False

        return True

    def add_item(item):
        selected.append(item)
        for n in item["numbers"]:
            usage[n] = usage.get(n, 0) + 1

    # First pass: strict overlap + strict number usage.
    for item in scored:
        if can_add(item, strict_usage=True):
            add_item(item)
        if len(selected) >= top_k:
            return selected

    # Second pass: keep overlap control, relax number usage if too few candidates.
    for item in scored:
        if item in selected:
            continue
        if can_add(item, strict_usage=False):
            add_item(item)
        if len(selected) >= top_k:
            return selected

    # Final fallback: always return top_k if possible.
    for item in scored:
        if item not in selected:
            add_item(item)
        if len(selected) >= top_k:
            return selected

    return selected


def predict(
    df,
    main_cols,
    min_num,
    max_num,
    pick_count,
    cfg,
    candidate_count,
    seed,
    top_k=5,
    random_mode=False,
    ctx=None,
):
    if ctx is None:
        ctx = build_model_context(
            df,
            main_cols,
            min_num,
            max_num,
        )

    candidates = generate_candidates(
        pick_count,
        min_num,
        max_num,
        ctx,
        cfg,
        candidate_count,
        seed,
        uniform=random_mode,
    )

    if len(candidates) < top_k:
        candidates = generate_candidates(
            pick_count,
            min_num,
            max_num,
            ctx,
            None,
            candidate_count,
            seed,
            uniform=True,
        )

    scored = []

    for nums in candidates:
        comps = component_scores(
            nums,
            max_num,
            ctx,
        )

        score = (
            0.0
            if random_mode or cfg is None
            else sum(
                cfg["w"][key] * comps[key]
                for key in cfg["w"]
            )
        )

        scored.append({
            "numbers": list(nums),
            "score": float(score),
            "model": (
                "random"
                if random_mode or cfg is None
                else cfg["name"]
            ),
            "block_counts": block_counts(
                nums,
                max_num,
            ),
            "consecutive_count": consecutive_count(nums),
            "score_detail": comps,
            "estimated_probability": (
                1 / math.comb(max_num, pick_count)
            ),
        })

    if not random_mode:
        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

    max_common = (
        3
        if cfg is None
        else cfg["f"]["max_common"]
    )
    max_number_usage = (
        3
        if cfg is None
        else cfg["f"].get(
            "max_number_usage",
            3,
        )
    )

    selected = select_diverse(
        scored,
        top_k,
        max_common,
        max_number_usage=max_number_usage,
    )

    out = []

    for i, item in enumerate(selected, start=1):
        out.append({
            "pattern_id": f"P{i}",
            "numbers": item["numbers"],
            "score": round(
                item["score"],
                6,
            ),
            "model": item["model"],
            "block_counts": item["block_counts"],
            "consecutive_count": (
                item["consecutive_count"]
            ),
            "estimated_probability": (
                item["estimated_probability"]
            ),
        })

    return out


def backtest(
    df,
    main_cols,
    min_num,
    max_num,
    pick_count,
    cfg,
    train_window,
    tested_periods,
    candidate_count,
    seed,
    random_mode=False,
    context_cache=None,
):
    start = max(
        train_window,
        len(df) - tested_periods,
    )
    matches = []

    for idx in range(start, len(df)):
        if context_cache is None:
            train = df.iloc[:idx].copy()
            ctx = build_model_context(
                train,
                main_cols,
                min_num,
                max_num,
            )
        else:
            ctx = context_cache[idx]

        actual = set(
            df.iloc[idx][main_cols]
            .astype(int)
            .tolist()
        )

        pred = predict(
            df=None,
            main_cols=main_cols,
            min_num=min_num,
            max_num=max_num,
            pick_count=pick_count,
            cfg=cfg,
            candidate_count=candidate_count,
            seed=seed + idx,
            top_k=1,
            random_mode=random_mode,
            ctx=ctx,
        )

        matches.append(
            len(
                set(pred[0]["numbers"])
                & actual
            )
        )

    n = len(matches)

    return {
        "config": (
            "random"
            if random_mode or cfg is None
            else cfg["name"]
        ),
        "tested_periods": n,
        "avg_matches": (
            round(float(np.mean(matches)), 4)
            if matches
            else None
        ),
        "hit_rate_1match": (
            round(
                sum(m >= 1 for m in matches) / n,
                4,
            )
            if n
            else None
        ),
        "hit_rate_2match": (
            round(
                sum(m >= 2 for m in matches) / n,
                4,
            )
            if n
            else None
        ),
        "hit_rate_3match": (
            round(
                sum(m >= 3 for m in matches) / n,
                4,
            )
            if n
            else None
        ),
        "hit_rate_4match": (
            round(
                sum(m >= 4 for m in matches) / n,
                4,
            )
            if n
            else None
        ),
        "hit_rate_5match": (
            round(
                sum(m >= 5 for m in matches) / n,
                4,
            )
            if n
            else None
        ),
    }


def selection_score(result, random_avg):
    avg = result["avg_matches"] or 0.0
    h2 = result["hit_rate_2match"] or 0.0
    h3 = result["hit_rate_3match"] or 0.0
    h4 = result["hit_rate_4match"] or 0.0
    uplift = 0.0 if random_avg is None else avg - random_avg

    score = avg + 0.30 * h2 + 0.80 * h3 + 1.20 * h4 + 0.35 * uplift

    name = result["config"]
    if "no_delay" in name or "balanced" in name:
        score += 0.015
    if "loose" in name:
        score -= 0.010
    if random_avg is not None and avg < random_avg - 0.05:
        score -= 0.25

    return round(float(score), 6)


def optimize(
    df,
    main_cols,
    min_num,
    max_num,
    pick_count,
    train_window,
    tested_periods,
    bt_candidates,
    final_candidates,
):
    start = max(
        train_window,
        len(df) - tested_periods,
    )

    context_cache = {}

    for idx in range(start, len(df)):
        train = df.iloc[:idx].copy()

        context_cache[idx] = build_model_context(
            train,
            main_cols,
            min_num,
            max_num,
        )

    random_result = backtest(
        df,
        main_cols,
        min_num,
        max_num,
        pick_count,
        None,
        train_window,
        tested_periods,
        bt_candidates,
        SEED,
        random_mode=True,
        context_cache=context_cache,
    )

    random_avg = random_result["avg_matches"]
    results = []

    for cfg in CONFIGS:
        result = backtest(
            df,
            main_cols,
            min_num,
            max_num,
            pick_count,
            cfg,
            train_window,
            tested_periods,
            bt_candidates,
            SEED,
            context_cache=context_cache,
        )

        result["selection_score"] = selection_score(
            result,
            random_avg,
        )
        result["random_avg"] = random_avg
        result["random_uplift"] = round(
            (result["avg_matches"] or 0.0)
            - (random_avg or 0.0),
            4,
        )
        result["weights"] = cfg["w"]
        result["filters"] = cfg["f"]

        results.append(result)

    results.sort(
        key=lambda item: item["selection_score"],
        reverse=True,
    )

    best_name = results[0]["config"]
    best_cfg = next(
        cfg
        for cfg in CONFIGS
        if cfg["name"] == best_name
    )

    final_ctx = build_model_context(
        df,
        main_cols,
        min_num,
        max_num,
    )

    prediction = predict(
        df=None,
        main_cols=main_cols,
        min_num=min_num,
        max_num=max_num,
        pick_count=pick_count,
        cfg=best_cfg,
        candidate_count=final_candidates,
        seed=SEED,
        top_k=5,
        ctx=final_ctx,
    )

    return {
        "random_baseline": random_result,
        "ranked_configs": results,
        "selected_config": best_name,
        "selected_weights": best_cfg["w"],
        "selected_filters": best_cfg["f"],
        "prediction": prediction,
    }


def load_data(include_miniloto=False):
    loto6 = normalize_loto6(
        read_csv_text(download_game_csv("loto6"))
    )
    loto7 = normalize_loto7(
        read_csv_text(download_game_csv("loto7"))
    )

    if not include_miniloto:
        return loto6, loto7

    miniloto = normalize_miniloto(
        read_csv_text(download_game_csv("miniloto"))
    )

    return loto6, loto7, miniloto


def print_result(title, latest, next_draw, result):
    print(f"\n=== {title} OPTIMIZER RESULT ===")
    print(f"latest={latest} next={next_draw}")
    rb = result["random_baseline"]
    print(f'random: avg={rb["avg_matches"]}, 2+={rb["hit_rate_2match"]}, 3+={rb["hit_rate_3match"]}')

    print("--- TOP CONFIGS ---")
    for r in result["ranked_configs"][:5]:
        print(f'{r["config"]}: selection={r["selection_score"]}, avg={r["avg_matches"]}, uplift={r["random_uplift"]}, 2+={r["hit_rate_2match"]}, 3+={r["hit_rate_3match"]}')

    print(f'selected_config={result["selected_config"]}')
    print(f'selected_weights={json.dumps(result["selected_weights"], ensure_ascii=False)}')
    print(f'selected_filters={json.dumps(result["selected_filters"], ensure_ascii=False)}')

    print(f"--- {title} NEXT PREDICTION ---")
    for p in result["prediction"]:
        print(f'{p["pattern_id"]}: {p["numbers"]} score={p["score"]} blocks={p["block_counts"]} con={p["consecutive_count"]}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loto6, loto7, miniloto = load_data(include_miniloto=True)

    loto6_cols = [
        "main1", "main2", "main3",
        "main4", "main5", "main6",
    ]
    loto7_cols = [
        "main1", "main2", "main3", "main4",
        "main5", "main6", "main7",
    ]
    miniloto_cols = [
        "main1", "main2", "main3",
        "main4", "main5",
    ]

    loto6_val = validate_lottery(
        loto6,
        loto6_cols,
        ["bonus"],
        1,
        43,
    )
    loto7_val = validate_lottery(
        loto7,
        loto7_cols,
        ["bonus1", "bonus2"],
        1,
        37,
    )
    miniloto_val = validate_lottery(
        miniloto,
        miniloto_cols,
        ["bonus"],
        1,
        31,
    )

    loto6_result = optimize(
        loto6,
        loto6_cols,
        1,
        43,
        6,
        500,
        45,
        300,
        10000,
    )
    loto7_result = optimize(
        loto7,
        loto7_cols,
        1,
        37,
        7,
        240,
        45,
        300,
        10000,
    )
    miniloto_result = optimize(
        miniloto,
        miniloto_cols,
        1,
        31,
        5,
        500,
        45,
        300,
        10000,
    )

    output = {
        "status": "ok",
        "note": (
            "optimizer automatically searches multiple weight/filter "
            "configs against a random baseline."
        ),
        "loto6": {
            "latest_draw_no": loto6_val["latest_draw_no"],
            "next_draw_no": loto6_val["latest_draw_no"] + 1,
            "rows": loto6_val["rows"],
            "validation": loto6_val,
            **loto6_result,
        },
        "loto7": {
            "latest_draw_no": loto7_val["latest_draw_no"],
            "next_draw_no": loto7_val["latest_draw_no"] + 1,
            "rows": loto7_val["rows"],
            "validation": loto7_val,
            **loto7_result,
        },
        "miniloto": {
            "latest_draw_no": miniloto_val["latest_draw_no"],
            "next_draw_no": miniloto_val["latest_draw_no"] + 1,
            "rows": miniloto_val["rows"],
            "validation": miniloto_val,
            **miniloto_result,
        },
    }

    (
        OUTPUT_DIR / "optimizer_result.json"
    ).write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR / "prediction_optimizer_loto6.json"
    ).write_text(
        json.dumps(
            loto6_result["prediction"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR / "prediction_optimizer_loto7.json"
    ).write_text(
        json.dumps(
            loto7_result["prediction"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR / "prediction_optimizer_miniloto.json"
    ).write_text(
        json.dumps(
            miniloto_result["prediction"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_result(
        "LOTO6",
        output["loto6"]["latest_draw_no"],
        output["loto6"]["next_draw_no"],
        loto6_result,
    )
    print_result(
        "LOTO7",
        output["loto7"]["latest_draw_no"],
        output["loto7"]["next_draw_no"],
        loto7_result,
    )
    print_result(
        "MINILOTO",
        output["miniloto"]["latest_draw_no"],
        output["miniloto"]["next_draw_no"],
        miniloto_result,
    )

    print("\n=== SHORT JSON ===")
    print(
        json.dumps(
            {
                "status": "ok",
                "loto6_latest_draw_no": (
                    output["loto6"]["latest_draw_no"]
                ),
                "loto6_next_draw_no": (
                    output["loto6"]["next_draw_no"]
                ),
                "loto6_selected_config": (
                    loto6_result["selected_config"]
                ),
                "loto6_prediction": [
                    p["numbers"]
                    for p in loto6_result["prediction"]
                ],
                "loto7_latest_draw_no": (
                    output["loto7"]["latest_draw_no"]
                ),
                "loto7_next_draw_no": (
                    output["loto7"]["next_draw_no"]
                ),
                "loto7_selected_config": (
                    loto7_result["selected_config"]
                ),
                "loto7_prediction": [
                    p["numbers"]
                    for p in loto7_result["prediction"]
                ],
                "miniloto_latest_draw_no": (
                    output["miniloto"]["latest_draw_no"]
                ),
                "miniloto_next_draw_no": (
                    output["miniloto"]["next_draw_no"]
                ),
                "miniloto_selected_config": (
                    miniloto_result["selected_config"]
                ),
                "miniloto_prediction": [
                    p["numbers"]
                    for p in miniloto_result["prediction"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()