from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"

LOTO6_COLUMNS = ["draw_no", "date", "main1", "main2", "main3", "main4", "main5", "main6", "bonus"]
LOTO7_COLUMNS = ["draw_no", "date", "main1", "main2", "main3", "main4", "main5", "main6", "main7", "bonus1", "bonus2"]

def load_csv(path, expected_columns):
    df = pd.read_csv(path)
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df

def validate(df, main_cols, bonus_cols, min_num, max_num):
    number_cols = main_cols + bonus_cols

    report = {
        "rows": int(len(df)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_draw_no": int(df["draw_no"].duplicated().sum()) if "draw_no" in df.columns else 0,
        "duplicate_date": int(df["date"].duplicated().sum()) if "date" in df.columns else 0,
        "out_of_range_cells": 0,
        "duplicate_main_numbers_rows": 0,
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

    report["status"] = "ok" if all([
        report["missing_cells"] == 0,
        report["duplicate_draw_no"] == 0,
        report["out_of_range_cells"] == 0,
        report["duplicate_main_numbers_rows"] == 0,
    ]) else "warning"

    return report

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loto6 = load_csv(DATA_DIR / "loto6.csv", LOTO6_COLUMNS)
    loto7 = load_csv(DATA_DIR / "loto7.csv", LOTO7_COLUMNS)

    result = {
        "loto6_validation": validate(
            loto6,
            ["main1", "main2", "main3", "main4", "main5", "main6"],
            ["bonus"],
            1,
            43,
        ),
        "loto7_validation": validate(
            loto7,
            ["main1", "main2", "main3", "main4", "main5", "main6", "main7"],
            ["bonus1", "bonus2"],
            1,
            37,
        ),
    }

    (OUTPUT_DIR / "validation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
