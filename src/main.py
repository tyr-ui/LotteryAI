from pathlib import Path
import csv
import io
import json
from itertools import combinations
from collections import Counter
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"

LOTO6_URL = "https://www.mizuhobank.co.jp/retail/takarakuji/loto/loto6/csv/loto6.csv"
LOTO7_URL = "https://www.mizuhobank.co.jp/retail/takarakuji/loto/loto7/csv/loto7.csv"

LOTO6_COLUMNS = ["draw_no", "date", "main1", "main2", "main3", "main4", "main5", "main6", "bonus"]
LOTO7_COLUMNS = ["draw_no", "date", "main1", "main2", "main3", "main4", "main5", "main6", "main7", "bonus1", "bonus2"]


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
    """
    kind: 'loto6' or 'loto7'
    mk-modeの一覧ページから LOTO6_ALL.csv / LOTO7_ALL.csv のリンクを探して取得する。
    """
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
    for encoding_note in ["normal"]:
        try:
            return pd.read_csv(io.StringIO(text))
        except Exception:
            pass

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV is empty")

    return pd.DataFrame(rows[1:], columns=rows[0])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(c).replace(" ", "").replace("　", "").lower(): c for c in df.columns}
    for name in candidates:
        key = name.replace(" ", "").replace("　", "").lower()
        if key in normalized:
            return normalized[key]
    return None


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


def clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col != "date":
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


def validate_lottery(df: pd.DataFrame, main_cols: list[str], bonus_cols: list[str], min_num: int, max_num: int) -> dict:
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


def simple_stats(df: pd.DataFrame, main_cols: list[str], max_num: int) -> dict:
    all_numbers = []
    for _, row in df.iterrows():
        all_numbers.extend(row[main_cols].astype(int).tolist())

    freq = Counter(all_numbers)
    total_draws = len(df)

    frequency = {
        str(n): {
            "count": int(freq[n]),
            "rate_per_draw": round(freq[n] / total_draws, 6) if total_draws else 0
        }
        for n in range(1, max_num + 1)
    }

    top_frequency = sorted(
        [{"number": n, "count": int(freq[n])} for n in range(1, max_num + 1)],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    pair_counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(row[main_cols].astype(int).tolist())
        pair_counter.update(combinations(nums, 2))

    top_pairs = [
        {"pair": list(pair), "count": int(count)}
        for pair, count in pair_counter.most_common(10)
    ]

    return {
        "total_draws": int(total_draws),
        "top_frequency": top_frequency,
        "top_pairs": top_pairs,
        "frequency": frequency,
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

    result = {
        "loto6_validation": validate_lottery(
            loto6,
            ["main1", "main2", "main3", "main4", "main5", "main6"],
            ["bonus"],
            1,
            43,
        ),
        "loto7_validation": validate_lottery(
            loto7,
            ["main1", "main2", "main3", "main4", "main5", "main6", "main7"],
            ["bonus1", "bonus2"],
            1,
            37,
        ),
        "loto6_stats": simple_stats(
            loto6,
            ["main1", "main2", "main3", "main4", "main5", "main6"],
            43,
        ),
        "loto7_stats": simple_stats(
            loto7,
            ["main1", "main2", "main3", "main4", "main5", "main6", "main7"],
            37,
        ),
    }

    (OUTPUT_DIR / "analysis_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "ok",
        "loto6_latest_draw_no": result["loto6_validation"]["latest_draw_no"],
        "loto7_latest_draw_no": result["loto7_validation"]["latest_draw_no"],
        "loto6_rows": result["loto6_validation"]["rows"],
        "loto7_rows": result["loto7_validation"]["rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
