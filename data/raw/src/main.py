from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"


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


def load_csv(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    missing_columns = [c for c in expected_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name}: missing columns: {missing_columns}")

    return df


def validate_lottery(
    df: pd.DataFrame,
    draw_type: str,
    main_cols: list[str],
    bonus_cols: list[str],
    min_num: int,
    max_num: int
) -> dict:
    number_cols = main_cols + bonus_cols

    report = {
        "draw_type": draw_type,
        "rows": int(len(df)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_draw_no": int(df["draw_no"].duplicated().sum()) if "draw_no" in df.columns else None,
        "duplicate_date": int(df["date"].duplicated().sum()) if "date" in df.columns else None,
        "out_of_range_cells": 0,
        "duplicate_main_numbers_rows": 0,
        "status": "ok"
    }

    if len(df) == 0:
        report["status"] = "empty_csv_header_only"
        return report

    for col in number_cols:
        report["out_of_range_cells"] += int((~df[col].between(min_num, max_num)).sum())

    for _, row in df[main_cols].iterrows():
        nums = row.dropna().astype(int).tolist()
        if len(nums) != len(set(nums)):
            report["duplicate_main_numbers_rows"] += 1

    if (
        report["missing_cells"] > 0
        or report["duplicate_draw_no"] > 0
        or report["out_of_range_cells"] > 0
        or report["duplicate_main_numbers_rows"] > 0
    ):
        report["status"] = "warning"

    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loto6 = load_csv(DATA_DIR / "loto6.csv", LOTO6_COLUMNS)
    loto7 = load_csv(DATA_DIR / "loto7.csv", LOTO7_COLUMNS)

    result = {
        "loto6_validation": validate_lottery(
            df=loto6,
            draw_type="loto6",
            main_cols=["main1", "main2", "main3", "main4", "main5", "main6"],
            bonus_cols=["bonus"],
            min_num=1,
            max_num=43
        ),
        "loto7_validation": validate_lottery(
            df=loto7,
            draw_type="loto7",
            main_cols=["main1", "main2", "main3", "main4", "main5", "main6", "main7"],
            bonus_cols=["bonus1", "bonus2"],
            min_num=1,
            max_num=37
        )
    }

    output_path = OUTPUT_DIR / "validation_result.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
