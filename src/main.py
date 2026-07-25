from pathlib import Path
import csv
import io
import json
import math
from itertools import combinations
from collections import Counter
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from games import LOTTO_GAMES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"

LOTO6_URL = LOTTO_GAMES["loto6"]["official_url"]
LOTO7_URL = LOTTO_GAMES["loto7"]["official_url"]
MINILOTO_URL = LOTTO_GAMES["miniloto"]["official_url"]

LOTO6_COLUMNS = [
    "draw_no", "date",
    "main1", "main2", "main3", "main4", "main5", "main6",
    "bonus"
]

LOTO7_COLUMNS = [
    "draw_no", "date",
    "main1", "main2", "main3", "main4", "main5", "main6", "main7",
    "bonus1", "bonus2"
]


def decode_content(content: bytes) -> str:
    for encoding in ["utf-8-sig", "cp932", "shift_jis", "utf-8"]:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("unknown", content, 0, 1, "Could not decode response")


def get_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,text/html,*/*",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }


def download_from_mkmode(kind: str) -> str:
    headers = get_headers()
    page_url = f"https://www.mk-mode.com/rails/loto/{kind}"

    page_response = requests.get(page_url, headers=headers, timeout=30)
    page_response.raise_for_status()

    html = decode_content(page_response.content)
    soup = BeautifulSoup(html, "html.parser")

    target_name = f"{kind.upper()}_ALL.csv"

    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        href = a.get("href", "")

        if target_name in text or target_name in href:
            csv_url = urljoin(page_url, href)
            csv_response = requests.get(csv_url, headers=headers, timeout=30)
            csv_response.raise_for_status()
            return decode_content(csv_response.content)

    raise RuntimeError(f"Could not find {target_name} link on {page_url}")


def download_text(url: str) -> str:
    headers = get_headers()
    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 403:
        if "loto6" in url:
            print("Official LOTO6 CSV was blocked. Falling back to mk-mode.")
            return download_from_mkmode("loto6")

        if "loto7" in url:
            print("Official LOTO7 CSV was blocked. Falling back to mk-mode.")
            return download_from_mkmode("loto7")

        raise RuntimeError(f"403 Forbidden: {url}")

    response.raise_for_status()
    return decode_content(response.content)


def read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            raise ValueError("CSV is empty")
        return pd.DataFrame(rows[1:], columns=rows[0])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {
        str(c).replace(" ", "").replace("　", "").lower(): c
        for c in df.columns
    }

    for name in candidates:
        key = name.replace(" ", "").replace("　", "").lower()
        if key in normalized:
            return normalized[key]

    return None


def clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col == "date":
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+)", expand=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[c for c in columns if c != "date"])

    for col in columns:
        if col != "date":
            df[col] = df[col].astype(int)

    return df[columns].sort_values("draw_no").reset_index(drop=True)


def normalize_loto6(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    if all(c in df.columns for c in LOTO6_COLUMNS):
        out = df[LOTO6_COLUMNS].copy()
    else:
        mapping = {
            "draw_no": find_col(df, ["draw_no", "回別", "回号", "開催回"]),
            "date": find_col(df, ["date", "抽せん日", "抽選日"]),
            "main1": find_col(df, ["main1", "本数字1", "第1数字"]),
            "main2": find_col(df, ["main2", "本数字2", "第2数字"]),
            "main3": find_col(df, ["main3", "本数字3", "第3数字"]),
            "main4": find_col(df, ["main4", "本数字4", "第4数字"]),
            "main5": find_col(df, ["main5", "本数字5", "第5数字"]),
            "main6": find_col(df, ["main6", "本数字6", "第6数字"]),
            "bonus": find_col(df, ["bonus", "ボーナス数字", "ボーナス"]),
        }

        if all(mapping.values()):
            out = pd.DataFrame({k: df[v] for k, v in mapping.items()})
        else:
            if df.shape[1] < 9:
                raise ValueError(f"LOTO6 CSV has too few columns: {list(df.columns)}")
            out = df.iloc[:, :9].copy()
            out.columns = LOTO6_COLUMNS

    return clean_numeric(out, LOTO6_COLUMNS)


def normalize_loto7(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    if all(c in df.columns for c in LOTO7_COLUMNS):
        out = df[LOTO7_COLUMNS].copy()
    else:
        mapping = {
            "draw_no": find_col(df, ["draw_no", "回別", "回号", "開催回"]),
            "date": find_col(df, ["date", "抽せん日", "抽選日"]),
            "main1": find_col(df, ["main1", "本数字1", "第1数字"]),
            "main2": find_col(df, ["main2", "本数字2", "第2数字"]),
            "main3": find_col(df, ["main3", "本数字3", "第3数字"]),
            "main4": find_col(df, ["main4", "本数字4", "第4数字"]),
            "main5": find_col(df, ["main5", "本数字5", "第5数字"]),
            "main6": find_col(df, ["main6", "本数字6", "第6数字"]),
            "main7": find_col(df, ["main7", "本数字7", "第7数字"]),
            "bonus1": find_col(df, ["bonus1", "ボーナス数字1", "ボーナス1"]),
            "bonus2": find_col(df, ["bonus2", "ボーナス数字2", "ボーナス2"]),
        }

        if all(mapping.values()):
            out = pd.DataFrame({k: df[v] for k, v in mapping.items()})
        else:
            if df.shape[1] < 11:
                raise ValueError(f"LOTO7 CSV has too few columns: {list(df.columns)}")
            out = df.iloc[:, :11].copy()
            out.columns = LOTO7_COLUMNS

    return clean_numeric(out, LOTO7_COLUMNS)


def validate_lottery(
    df: pd.DataFrame,
    main_cols: list[str],
    bonus_cols: list[str],
    min_num: int,
    max_num: int
) -> dict:
    number_cols = main_cols + bonus_cols

    report = {
        "rows": int(len(df)),
        "latest_draw_no": int(df["draw_no"].max()) if len(df) else None,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_draw_no": int(df["draw_no"].duplicated().sum()),
        "duplicate_date": int(df["date"].duplicated().sum()),
        "out_of_range_cells": 0,
        "duplicate_main_numbers_rows": 0,
    }

    for col in number_cols:
        report["out_of_range_cells"] += int((~df[col].between(min_num, max_num)).sum())

    for _, row in df[main_cols].iterrows():
        nums = row[main_cols].astype(int).tolist()
        if len(nums) != len(set(nums)):
            report["duplicate_main_numbers_rows"] += 1

    report["status"] = "ok" if all([
        report["missing_cells"] == 0,
        report["duplicate_draw_no"] == 0,
        report["out_of_range_cells"] == 0,
        report["duplicate_main_numbers_rows"] == 0,
    ]) else "warning"

    return report


def minmax_map(values: dict[int, float], min_num: int, max_num: int) -> dict[int, float]:
    raw = [values.get(n, 0.0) for n in range(min_num, max_num + 1)]
    lo = min(raw)
    hi = max(raw)

    if hi == lo:
        return {n: 0.5 for n in range(min_num, max_num + 1)}

    return {
        n: (values.get(n, 0.0) - lo) / (hi - lo)
        for n in range(min_num, max_num + 1)
    }


def count_frequency(df: pd.DataFrame, main_cols: list[str], min_num: int, max_num: int) -> dict[int, int]:
    counter = Counter()

    for _, row in df.iterrows():
        nums = row[main_cols].astype(int).tolist()
        counter.update(nums)

    return {n: int(counter[n]) for n in range(min_num, max_num + 1)}


def count_recent_frequency(
    df: pd.DataFrame,
    main_cols: list[str],
    min_num: int,
    max_num: int
) -> dict[int, float]:
    weights = [
        (50, 0.50),
        (100, 0.30),
        (200, 0.20),
    ]

    score = {n: 0.0 for n in range(min_num, max_num + 1)}

    for window, weight in weights:
        recent = df.tail(min(window, len(df)))
        freq = count_frequency(recent, main_cols, min_num, max_num)
        denom = max(1, len(recent))

        for n in range(min_num, max_num + 1):
            score[n] += weight * (freq[n] / denom)

    return score


def current_delay(df: pd.DataFrame, main_cols: list[str], min_num: int, max_num: int) -> dict[int, int]:
    last_seen = {n: None for n in range(min_num, max_num + 1)}

    for idx, (_, row) in enumerate(df.iterrows()):
        nums = row[main_cols].astype(int).tolist()
        for n in nums:
            last_seen[n] = idx

    delays = {}
    last_index = len(df) - 1

    for n in range(min_num, max_num + 1):
        if last_seen[n] is None:
            delays[n] = len(df)
        else:
            delays[n] = last_index - last_seen[n]

    return delays


def pair_counts(df: pd.DataFrame, main_cols: list[str]) -> Counter:
    counter = Counter()

    for _, row in df.iterrows():
        nums = sorted(row[main_cols].astype(int).tolist())
        counter.update(combinations(nums, 2))

    return counter


def triplet_counts(df: pd.DataFrame, main_cols: list[str]) -> Counter:
    counter = Counter()

    for _, row in df.iterrows():
        nums = sorted(row[main_cols].astype(int).tolist())
        counter.update(combinations(nums, 3))

    return counter


def consecutive_count(nums: list[int]) -> int:
    nums = sorted(nums)
    return sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)


def block_tuple(nums: list[int], max_num: int) -> tuple[int, int, int, int]:
    if max_num == 43:
        blocks = [(1, 10), (11, 21), (22, 32), (33, 43)]
    else:
        blocks = [(1, 9), (10, 18), (19, 27), (28, 37)]

    result = []
    for lo, hi in blocks:
        result.append(sum(1 for n in nums if lo <= n <= hi))

    return tuple(result)


def build_shape_stats(df: pd.DataFrame, main_cols: list[str], min_num: int, max_num: int) -> dict:
    sums = []
    odd_counter = Counter()
    low_counter = Counter()
    consecutive_counter = Counter()
    block_counter = Counter()

    threshold = max_num // 2

    for _, row in df.iterrows():
        nums = sorted(row[main_cols].astype(int).tolist())

        sums.append(sum(nums))
        odd_counter.update([sum(n % 2 for n in nums)])
        low_counter.update([sum(n <= threshold for n in nums)])
        consecutive_counter.update([consecutive_count(nums)])
        block_counter.update([block_tuple(nums, max_num)])

    return {
        "sum_mean": float(np.mean(sums)),
        "sum_std": float(np.std(sums) if np.std(sums) > 0 else 1.0),
        "odd_counter": odd_counter,
        "low_counter": low_counter,
        "consecutive_counter": consecutive_counter,
        "block_counter": block_counter,
    }


def normalized_counter_score(counter: Counter, key) -> float:
    if not counter:
        return 0.5

    max_count = max(counter.values())
    if max_count == 0:
        return 0.5

    return counter.get(key, 0) / max_count


def shape_score(nums: tuple[int, ...], max_num: int, shape_stats: dict) -> float:
    nums_list = list(nums)

    total = sum(nums_list)
    sum_mean = shape_stats["sum_mean"]
    sum_std = shape_stats["sum_std"]
    sum_score = math.exp(-abs(total - sum_mean) / sum_std)

    threshold = max_num // 2
    odd_count = sum(n % 2 for n in nums_list)
    low_count = sum(n <= threshold for n in nums_list)
    con_count = consecutive_count(nums_list)
    blocks = block_tuple(nums_list, max_num)

    odd_score = normalized_counter_score(shape_stats["odd_counter"], odd_count)
    low_score = normalized_counter_score(shape_stats["low_counter"], low_count)
    con_score = normalized_counter_score(shape_stats["consecutive_counter"], con_count)
    block_score = normalized_counter_score(shape_stats["block_counter"], blocks)

    return (
        0.30 * sum_score
        + 0.20 * odd_score
        + 0.20 * low_score
        + 0.15 * con_score
        + 0.15 * block_score
    )


def build_model_context(
    df: pd.DataFrame,
    main_cols: list[str],
    min_num: int,
    max_num: int
) -> dict:
    global_freq = count_frequency(df, main_cols, min_num, max_num)
    recent_freq = count_recent_frequency(df, main_cols, min_num, max_num)
    delay = current_delay(df, main_cols, min_num, max_num)

    global_norm = minmax_map(global_freq, min_num, max_num)
    recent_norm = minmax_map(recent_freq, min_num, max_num)
    delay_norm = minmax_map(delay, min_num, max_num)

    pairs = pair_counts(df, main_cols)
    triples = triplet_counts(df, main_cols)

    max_pair = max(pairs.values()) if pairs else 1
    max_triple = max(triples.values()) if triples else 1

    shape_stats = build_shape_stats(df, main_cols, min_num, max_num)

    number_weight = {}
    for n in range(min_num, max_num + 1):
        number_weight[n] = (
            0.35 * global_norm[n]
            + 0.40 * recent_norm[n]
            + 0.25 * delay_norm[n]
        )

    return {
        "global_freq": global_freq,
        "recent_freq": recent_freq,
        "delay": delay,
        "global_norm": global_norm,
        "recent_norm": recent_norm,
        "delay_norm": delay_norm,
        "pairs": pairs,
        "triples": triples,
        "max_pair": max_pair,
        "max_triple": max_triple,
        "shape_stats": shape_stats,
        "number_weight": number_weight,
    }


def score_candidate(
    nums: tuple[int, ...],
    max_num: int,
    ctx: dict
) -> dict:
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
    triple_score = float(np.mean(triple_values)) if triple_values else 0.0

    dist_score = shape_score(nums, max_num, ctx["shape_stats"])

    total_score = (
        0.22 * freq_score
        + 0.24 * recent_score
        + 0.22 * pair_score
        + 0.08 * triple_score
        + 0.08 * delay_score
        + 0.16 * dist_score
    )

    return {
        "score": float(total_score),
        "frequency_score": float(freq_score),
        "recent_score": float(recent_score),
        "pair_score": float(pair_score),
        "triplet_score": float(triple_score),
        "delay_score": float(delay_score),
        "distribution_score": float(dist_score),
    }


def generate_candidates(
    pick_count: int,
    min_num: int,
    max_num: int,
    ctx: dict,
    candidate_count: int,
    seed: int
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)

    numbers = np.array(list(range(min_num, max_num + 1)))
    weights = np.array([ctx["number_weight"][int(n)] for n in numbers], dtype=float)

    weights = np.maximum(weights, 0.0001)
    weights = weights ** 1.35
    probabilities = weights / weights.sum()

    candidates = set()
    attempts = 0
    max_attempts = candidate_count * 20

    while len(candidates) < candidate_count and attempts < max_attempts:
        selected = rng.choice(numbers, size=pick_count, replace=False, p=probabilities)
        candidates.add(tuple(sorted(int(n) for n in selected)))
        attempts += 1

    return list(candidates)


def top_pair_in_candidate(nums: tuple[int, ...], pairs: Counter) -> tuple[list[int], int]:
    best_pair = None
    best_count = -1

    for pair in combinations(nums, 2):
        count = pairs.get(tuple(sorted(pair)), 0)
        if count > best_count:
            best_pair = pair
            best_count = count

    return list(best_pair) if best_pair else [], int(best_count)


def top_triplet_in_candidate(nums: tuple[int, ...], triples: Counter) -> tuple[list[int], int]:
    best_triplet = None
    best_count = -1

    for triple in combinations(nums, 3):
        count = triples.get(tuple(sorted(triple)), 0)
        if count > best_count:
            best_triplet = triple
            best_count = count

    return list(best_triplet) if best_triplet else [], int(best_count)


def make_rationale(nums: tuple[int, ...], max_num: int, ctx: dict) -> str:
    pair, pair_count = top_pair_in_candidate(nums, ctx["pairs"])
    triple, triple_count = top_triplet_in_candidate(nums, ctx["triples"])

    total = sum(nums)
    odd = sum(n % 2 for n in nums)
    low = sum(n <= (max_num // 2) for n in nums)

    return (
        f"頻度・直近傾向・ペア共起・分布形状の総合スコアで上位。"
        f"合計{total}、奇偶{odd}:{len(nums)-odd}、低高{low}:{len(nums)-low}。"
        f"候補内の最頻ペアは{pair}（過去{pair_count}回）、"
        f"最頻トリプルは{triple}（過去{triple_count}回）。"
    )


def select_diverse_top(scored: list[dict], pick_count: int, top_k: int) -> list[dict]:
    selected = []
    max_common = max(3, pick_count - 2)

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


def predict_patterns(
    df: pd.DataFrame,
    draw_type: str,
    main_cols: list[str],
    min_num: int,
    max_num: int,
    pick_count: int,
    candidate_count: int = 10000,
    seed: int = 2025,
    top_k: int = 5,
    backtest_summary: dict | None = None
) -> list[dict]:
    ctx = build_model_context(df, main_cols, min_num, max_num)

    candidates = generate_candidates(
        pick_count=pick_count,
        min_num=min_num,
        max_num=max_num,
        ctx=ctx,
        candidate_count=candidate_count,
        seed=seed,
    )

    scored = []
    estimated_probability = 1 / math.comb(max_num, pick_count)

    for nums in candidates:
        detail = score_candidate(nums, max_num, ctx)
        scored.append({
            "numbers": list(nums),
            "raw_tuple": nums,
            **detail,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = select_diverse_top(scored, pick_count, top_k)

    output = []
    for idx, item in enumerate(selected, start=1):
        nums_tuple = tuple(item["numbers"])

        output.append({
            "pattern_id": f"P{idx}",
            "numbers": item["numbers"],
            "score": round(float(item["score"]), 6),
            "estimated_probability": estimated_probability,
            "rationale": make_rationale(nums_tuple, max_num, ctx),
            "score_detail": {
                "frequency_score": round(item["frequency_score"], 6),
                "recent_score": round(item["recent_score"], 6),
                "pair_score": round(item["pair_score"], 6),
                "triplet_score": round(item["triplet_score"], 6),
                "delay_score": round(item["delay_score"], 6),
                "distribution_score": round(item["distribution_score"], 6),
            },
            "backtest_summary": backtest_summary or {},
        })

    return output


def run_backtest(
    df: pd.DataFrame,
    main_cols: list[str],
    min_num: int,
    max_num: int,
    pick_count: int,
    train_window: int,
    tested_periods: int,
    candidate_count: int,
    seed: int
) -> dict:
    if len(df) <= train_window + 5:
        return {
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

        prediction = predict_patterns(
            df=train,
            draw_type="backtest",
            main_cols=main_cols,
            min_num=min_num,
            max_num=max_num,
            pick_count=pick_count,
            candidate_count=candidate_count,
            seed=seed + test_idx,
            top_k=1,
            backtest_summary={},
        )

        predicted = set(prediction[0]["numbers"])
        matches.append(len(predicted & actual))

    tested = len(matches)

    return {
        "tested_periods": int(tested),
        "avg_matches": round(float(np.mean(matches)), 4) if matches else None,
        "hit_rate_1match": round(sum(m >= 1 for m in matches) / tested, 4) if tested else None,
        "hit_rate_2match": round(sum(m >= 2 for m in matches) / tested, 4) if tested else None,
        "hit_rate_3match": round(sum(m >= 3 for m in matches) / tested, 4) if tested else None,
        "hit_rate_4match": round(sum(m >= 4 for m in matches) / tested, 4) if tested else None,
        "hit_rate_5match": round(sum(m >= 5 for m in matches) / tested, 4) if tested else None,
    }


def simple_stats(df: pd.DataFrame, main_cols: list[str], max_num: int) -> dict:
    all_numbers = []
    for _, row in df.iterrows():
        all_numbers.extend(row[main_cols].astype(int).tolist())

    freq = Counter(all_numbers)
    total_draws = len(df)

    top_frequency = sorted(
        [{"number": n, "count": int(freq[n])} for n in range(1, max_num + 1)],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    pair_counter = pair_counts(df, main_cols)
    triple_counter = triplet_counts(df, main_cols)

    top_pairs = [
        {"pair": list(pair), "count": int(count)}
        for pair, count in pair_counter.most_common(10)
    ]

    top_triplets = [
        {"triplet": list(triplet), "count": int(count)}
        for triplet, count in triple_counter.most_common(10)
    ]

    return {
        "total_draws": int(total_draws),
        "top_frequency": top_frequency,
        "top_pairs": top_pairs,
        "top_triplets": top_triplets,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loto6_text = download_text(LOTO6_URL)
    loto7_text = download_text(LOTO7_URL)

    loto6_raw = read_csv_text(loto6_text)
    loto7_raw = read_csv_text(loto7_text)

    loto6 = normalize_loto6(loto6_raw)
    loto7 = normalize_loto7(loto7_raw)

    loto6.to_csv(DATA_DIR / "loto6.csv", index=False, encoding="utf-8")
    loto7.to_csv(DATA_DIR / "loto7.csv", index=False, encoding="utf-8")

    loto6_main_cols = ["main1", "main2", "main3", "main4", "main5", "main6"]
    loto7_main_cols = ["main1", "main2", "main3", "main4", "main5", "main6", "main7"]

    loto6_validation = validate_lottery(
        loto6,
        loto6_main_cols,
        ["bonus"],
        1,
        43,
    )

    loto7_validation = validate_lottery(
        loto7,
        loto7_main_cols,
        ["bonus1", "bonus2"],
        1,
        37,
    )

    loto6_backtest = run_backtest(
        df=loto6,
        main_cols=loto6_main_cols,
        min_num=1,
        max_num=43,
        pick_count=6,
        train_window=500,
        tested_periods=80,
        candidate_count=800,
        seed=2025,
    )

    loto7_backtest = run_backtest(
        df=loto7,
        main_cols=loto7_main_cols,
        min_num=1,
        max_num=37,
        pick_count=7,
        train_window=240,
        tested_periods=80,
        candidate_count=800,
        seed=2025,
    )

    loto6_prediction = predict_patterns(
        df=loto6,
        draw_type="loto6",
        main_cols=loto6_main_cols,
        min_num=1,
        max_num=43,
        pick_count=6,
        candidate_count=10000,
        seed=2025,
        top_k=5,
        backtest_summary=loto6_backtest,
    )

    loto7_prediction = predict_patterns(
        df=loto7,
        draw_type="loto7",
        main_cols=loto7_main_cols,
        min_num=1,
        max_num=37,
        pick_count=7,
        candidate_count=10000,
        seed=2025,
        top_k=5,
        backtest_summary=loto7_backtest,
    )

    result = {
        "status": "ok",
        "source_note": "Official Mizuho CSV may be blocked from GitHub Actions; fallback source is mk-mode CSV when needed.",
        "loto6": {
            "latest_draw_no": loto6_validation["latest_draw_no"],
            "next_draw_no": loto6_validation["latest_draw_no"] + 1,
            "validation": loto6_validation,
            "stats": simple_stats(loto6, loto6_main_cols, 43),
            "backtest_summary": loto6_backtest,
            "prediction": loto6_prediction,
        },
        "loto7": {
            "latest_draw_no": loto7_validation["latest_draw_no"],
            "next_draw_no": loto7_validation["latest_draw_no"] + 1,
            "validation": loto7_validation,
            "stats": simple_stats(loto7, loto7_main_cols, 37),
            "backtest_summary": loto7_backtest,
            "prediction": loto7_prediction,
        },
    }

    (OUTPUT_DIR / "analysis_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "prediction_loto6.json").write_text(
        json.dumps(loto6_prediction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "prediction_loto7.json").write_text(
        json.dumps(loto7_prediction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "status": "ok",
        "loto6_latest_draw_no": result["loto6"]["latest_draw_no"],
        "loto6_next_draw_no": result["loto6"]["next_draw_no"],
        "loto6_rows": result["loto6"]["validation"]["rows"],
        "loto6_prediction": [p["numbers"] for p in loto6_prediction],
        "loto6_backtest": loto6_backtest,
        "loto7_latest_draw_no": result["loto7"]["latest_draw_no"],
        "loto7_next_draw_no": result["loto7"]["next_draw_no"],
        "loto7_rows": result["loto7"]["validation"]["rows"],
        "loto7_prediction": [p["numbers"] for p in loto7_prediction],
        "loto7_backtest": loto7_backtest,
    }

    print("\n=== LOTO6 NEXT PREDICTION ===")
    for p in loto6_prediction:
        print(f'{p["pattern_id"]}: {p["numbers"]} score={p["score"]}')

    print("\n=== LOTO6 BACKTEST ===")
    print(json.dumps(loto6_backtest, ensure_ascii=False, indent=2))

    print("\n=== LOTO7 NEXT PREDICTION ===")
    for p in loto7_prediction:
        print(f'{p["pattern_id"]}: {p["numbers"]} score={p["score"]}')

    print("\n=== LOTO7 BACKTEST ===")
    print(json.dumps(loto7_backtest, ensure_ascii=False, indent=2))

    print("\n=== FULL SUMMARY JSON ===")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()