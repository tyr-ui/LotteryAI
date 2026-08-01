from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True, slots=True)
class LoadedGameData:
    game_name: str
    dataframe: pd.DataFrame
    validation: Mapping[str, object]
    source: str


def _config_value(
    config: Mapping[str, object] | object,
    *names: str,
    default: object | None = None,
) -> object:
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
    else:
        for name in names:
            if hasattr(config, name):
                return getattr(config, name)
    return default


def _game_family(
    config: Mapping[str, object] | object,
) -> str:
    return str(
        _config_value(
            config,
            "family",
            default="lotto",
        )
    ).lower()


def decode_content(content: bytes) -> str:
    for encoding in (
        "utf-8-sig",
        "cp932",
        "shift_jis",
        "utf-8",
    ):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        content,
        0,
        1,
        "Could not decode response",
    )


def get_headers() -> dict[str, str]:
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
    page_url = (
        f"https://www.mk-mode.com/rails/loto/{kind}"
    )

    response = requests.get(
        page_url,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        decode_content(response.content),
        "html.parser",
    )
    target_name = f"{kind.upper()}_ALL.csv"

    for anchor in soup.find_all("a"):
        text = anchor.get_text(strip=True)
        href = anchor.get("href", "")

        if (
            target_name not in text
            and target_name not in href
        ):
            continue

        csv_url = urljoin(
            page_url,
            href,
        )
        csv_response = requests.get(
            csv_url,
            headers=headers,
            timeout=30,
        )
        csv_response.raise_for_status()

        return decode_content(
            csv_response.content
        )

    raise RuntimeError(
        f"Could not find {target_name} "
        f"link on {page_url}"
    )


def download_game_csv(
    game_name: str,
    config: Mapping[str, object] | object,
) -> tuple[str, str]:
    official_url = str(
        _config_value(
            config,
            "official_url",
            default="",
        )
    )
    fallback_kind = _config_value(
        config,
        "fallback_kind",
        default=None,
    )

    if not official_url:
        raise ValueError(
            "official_url is not configured "
            f"for game: {game_name}"
        )

    response = requests.get(
        official_url,
        headers=get_headers(),
        timeout=30,
    )

    if response.status_code == 403:
        if not fallback_kind:
            raise RuntimeError(
                "Official CSV was blocked and "
                "no fallback is configured: "
                f"{game_name}"
            )

        return (
            download_from_mkmode(
                str(fallback_kind)
            ),
            "mk-mode",
        )

    response.raise_for_status()

    return (
        decode_content(
            response.content
        ),
        "official",
    )


def read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(
            io.StringIO(text),
            dtype=str,
        )
    except Exception:
        rows = list(
            csv.reader(
                io.StringIO(text)
            )
        )

        if not rows:
            raise ValueError(
                "CSV is empty."
            )

        return pd.DataFrame(
            rows[1:],
            columns=rows[0],
        )


def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result.columns = [
        str(column).strip()
        for column in result.columns
    ]
    return result


def find_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    normalized = {
        (
            str(column)
            .replace(" ", "")
            .replace("　", "")
            .lower()
        ): column
        for column in df.columns
    }

    for candidate in candidates:
        key = (
            candidate
            .replace(" ", "")
            .replace("　", "")
            .lower()
        )

        if key in normalized:
            return str(
                normalized[key]
            )

    return None


def clean_numeric(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    result = df.copy()

    for column in columns:
        if column == "date":
            continue

        result[column] = (
            result[column]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False,
            )
            .str.extract(
                r"(\d+)",
                expand=False,
            )
        )
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    required = [
        column
        for column in columns
        if column != "date"
    ]

    result = result.dropna(
        subset=required
    )

    for column in required:
        result[column] = (
            result[column].astype(int)
        )

    return (
        result[list(columns)]
        .sort_values("draw_no")
        .reset_index(drop=True)
    )


def _column_candidates(
    canonical_name: str,
) -> tuple[str, ...]:
    fixed = {
        "draw_no": (
            "draw_no",
            "回別",
            "回号",
            "開催回",
        ),
        "date": (
            "date",
            "抽せん日",
            "抽選日",
        ),
        "number": (
            "number",
            "当選番号",
            "抽せん数字",
            "抽選数字",
            "本数字",
        ),
        "bonus": (
            "bonus",
            "ボーナス数字",
            "ボーナス",
        ),
        "bonus1": (
            "bonus1",
            "ボーナス数字1",
            "ボーナス1",
        ),
        "bonus2": (
            "bonus2",
            "ボーナス数字2",
            "ボーナス2",
        ),
        "digit1": (
            "digit1",
            "数字1",
            "第1数字",
            "1桁目",
            "千の位",
            "百の位",
        ),
        "digit2": (
            "digit2",
            "数字2",
            "第2数字",
            "2桁目",
            "百の位",
            "十の位",
        ),
        "digit3": (
            "digit3",
            "数字3",
            "第3数字",
            "3桁目",
            "十の位",
            "一の位",
        ),
        "digit4": (
            "digit4",
            "数字4",
            "第4数字",
            "4桁目",
            "一の位",
        ),
    }

    if canonical_name in fixed:
        return fixed[canonical_name]

    if canonical_name.startswith(
        "main"
    ):
        suffix = (
            canonical_name.removeprefix(
                "main"
            )
        )

        return (
            canonical_name,
            f"本数字{suffix}",
            f"第{suffix}数字",
        )

    return (canonical_name,)


def _numbers_digit_columns(
    config: Mapping[str, object] | object,
) -> tuple[str, ...]:
    configured = tuple(
        str(column)
        for column in _config_value(
            config,
            "main_cols",
            "main_columns",
            default=(),
        )
    )

    if configured:
        return configured

    digit_count = int(
        _config_value(
            config,
            "digit_count",
            default=0,
        )
        or 0
    )

    return tuple(
        f"digit{index}"
        for index in range(
            1,
            digit_count + 1,
        )
    )


def _extract_number_string(
    value: object,
    *,
    digit_count: int,
) -> str | None:
    text = str(
        value
        if value is not None
        else ""
    )

    digits = "".join(
        character
        for character in text
        if character.isdigit()
    )

    if not digits:
        return None

    if len(digits) > digit_count:
        digits = digits[-digit_count:]

    return digits.zfill(
        digit_count
    )


def _split_numbers_column(
    df: pd.DataFrame,
    *,
    source_column: str,
    digit_columns: Sequence[str],
) -> pd.DataFrame:
    result = df.copy()
    digit_count = len(
        digit_columns
    )

    extracted = result[
        source_column
    ].map(
        lambda value: (
            _extract_number_string(
                value,
                digit_count=digit_count,
            )
        )
    )

    for index, column in enumerate(
        digit_columns
    ):
        result[column] = extracted.map(
            lambda value: (
                value[index]
                if value is not None
                else None
            )
        )

    return result


def _normalize_numbers_dataframe(
    df: pd.DataFrame,
    config: Mapping[str, object] | object,
) -> pd.DataFrame:
    result = normalize_columns(df)

    digit_columns = (
        _numbers_digit_columns(
            config
        )
    )

    if not digit_columns:
        raise ValueError(
            "Numbers digit columns are "
            "not configured."
        )

    required_columns = (
        "draw_no",
        "date",
        *digit_columns,
    )

    direct_mapping = {
        column: find_col(
            result,
            _column_candidates(
                column
            ),
        )
        for column in required_columns
    }

    if all(
        direct_mapping.values()
    ):
        normalized = pd.DataFrame({
            canonical: result[source]
            for canonical, source
            in direct_mapping.items()
            if source is not None
        })

        return clean_numeric(
            normalized,
            required_columns,
        )

    draw_column = find_col(
        result,
        _column_candidates(
            "draw_no"
        ),
    )
    date_column = find_col(
        result,
        _column_candidates(
            "date"
        ),
    )
    number_column = find_col(
        result,
        _column_candidates(
            "number"
        ),
    )

    if (
        draw_column is not None
        and date_column is not None
        and number_column is not None
    ):
        normalized = pd.DataFrame({
            "draw_no": result[
                draw_column
            ],
            "date": result[
                date_column
            ],
            "number": result[
                number_column
            ],
        })

        normalized = (
            _split_numbers_column(
                normalized,
                source_column="number",
                digit_columns=(
                    digit_columns
                ),
            )
        )

        normalized = normalized.drop(
            columns=["number"]
        )

        return clean_numeric(
            normalized,
            required_columns,
        )

    minimum_column_count = (
        len(required_columns)
    )

    if (
        result.shape[1]
        < minimum_column_count
    ):
        missing = [
            column
            for column, source
            in direct_mapping.items()
            if source is None
        ]

        raise ValueError(
            "Numbers CSV has too few "
            "columns or unknown column "
            "names. "
            f"Missing mappings: {missing}; "
            "actual columns: "
            f"{list(result.columns)}"
        )

    normalized = result.iloc[
        :,
        :minimum_column_count,
    ].copy()
    normalized.columns = list(
        required_columns
    )

    return clean_numeric(
        normalized,
        required_columns,
    )


def normalize_game_dataframe(
    df: pd.DataFrame,
    config: Mapping[str, object] | object,
) -> pd.DataFrame:
    if _game_family(config) == "numbers":
        return _normalize_numbers_dataframe(
            df,
            config,
        )

    result = normalize_columns(df)

    all_columns = tuple(
        str(column)
        for column in _config_value(
            config,
            "all_columns",
            default=(),
        )
    )

    if not all_columns:
        main_columns = tuple(
            str(column)
            for column in _config_value(
                config,
                "main_cols",
                "main_columns",
                default=(),
            )
        )
        bonus_columns = tuple(
            str(column)
            for column in _config_value(
                config,
                "bonus_cols",
                "bonus_columns",
                default=(),
            )
        )

        all_columns = (
            "draw_no",
            "date",
            *main_columns,
            *bonus_columns,
        )

    if all(
        column in result.columns
        for column in all_columns
    ):
        normalized = result[
            list(all_columns)
        ].copy()
    else:
        mapping = {
            column: find_col(
                result,
                _column_candidates(
                    column
                ),
            )
            for column in all_columns
        }

        if all(mapping.values()):
            normalized = pd.DataFrame({
                canonical: result[source]
                for canonical, source
                in mapping.items()
                if source is not None
            })
        else:
            if (
                result.shape[1]
                < len(all_columns)
            ):
                missing = [
                    column
                    for column, source
                    in mapping.items()
                    if source is None
                ]

                raise ValueError(
                    "CSV has too few columns "
                    "or unknown column names. "
                    f"Missing mappings: {missing}; "
                    "actual columns: "
                    f"{list(result.columns)}"
                )

            normalized = result.iloc[
                :,
                :len(all_columns),
            ].copy()
            normalized.columns = list(
                all_columns
            )

    return clean_numeric(
        normalized,
        all_columns,
    )


def validate_numbers(
    df: pd.DataFrame,
    config: Mapping[str, object] | object,
) -> dict[str, object]:
    digit_columns = (
        _numbers_digit_columns(
            config
        )
    )
    digit_count = int(
        _config_value(
            config,
            "digit_count",
            default=len(
                digit_columns
            ),
        )
        or len(digit_columns)
    )
    digit_min = int(
        _config_value(
            config,
            "digit_min",
            "min_num",
            "min_number",
            default=0,
        )
        or 0
    )
    digit_max = int(
        _config_value(
            config,
            "digit_max",
            "max_num",
            "max_number",
            default=9,
        )
        or 9
    )

    missing_digit_columns = [
        column
        for column in digit_columns
        if column not in df.columns
    ]

    report: dict[str, object] = {
        "family": "numbers",
        "rows": int(len(df)),
        "latest_draw_no": (
            int(
                df["draw_no"].max()
            )
            if len(df)
            else None
        ),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "duplicate_draw_no": int(
            df["draw_no"]
            .duplicated()
            .sum()
        ),
        "duplicate_date": int(
            df["date"]
            .duplicated()
            .sum()
        ),
        "out_of_range_cells": 0,
        "invalid_digit_count_rows": 0,
        "missing_digit_columns": (
            missing_digit_columns
        ),
    }

    if (
        not digit_columns
        or missing_digit_columns
    ):
        report[
            "invalid_digit_count_rows"
        ] = int(len(df))
        report["status"] = "warning"
        return report

    out_of_range = 0

    for column in digit_columns:
        out_of_range += int(
            (
                ~df[column].between(
                    digit_min,
                    digit_max,
                )
            ).sum()
        )

    invalid_digit_count_rows = (
        0
        if len(digit_columns)
        == digit_count
        else int(len(df))
    )

    report[
        "out_of_range_cells"
    ] = out_of_range
    report[
        "invalid_digit_count_rows"
    ] = invalid_digit_count_rows

    report["status"] = (
        "ok"
        if (
            report["missing_cells"] == 0
            and report[
                "duplicate_draw_no"
            ] == 0
            and report[
                "out_of_range_cells"
            ] == 0
            and report[
                "invalid_digit_count_rows"
            ] == 0
            and not missing_digit_columns
        )
        else "warning"
    )

    return report


def validate_lottery(
    df: pd.DataFrame,
    config: Mapping[str, object] | object,
) -> dict[str, object]:
    if _game_family(config) == "numbers":
        return validate_numbers(
            df,
            config,
        )

    main_columns = tuple(
        str(column)
        for column in _config_value(
            config,
            "main_cols",
            "main_columns",
            default=(),
        )
    )
    bonus_columns = tuple(
        str(column)
        for column in _config_value(
            config,
            "bonus_cols",
            "bonus_columns",
            default=(),
        )
    )
    min_num = int(
        _config_value(
            config,
            "min_num",
            "min_number",
        )
    )
    max_num = int(
        _config_value(
            config,
            "max_num",
            "max_number",
        )
    )

    if not main_columns:
        raise ValueError(
            "main columns are not "
            "configured."
        )

    report: dict[str, object] = {
        "family": "lotto",
        "rows": int(len(df)),
        "latest_draw_no": (
            int(
                df["draw_no"].max()
            )
            if len(df)
            else None
        ),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "duplicate_draw_no": int(
            df["draw_no"]
            .duplicated()
            .sum()
        ),
        "duplicate_date": int(
            df["date"]
            .duplicated()
            .sum()
        ),
        "out_of_range_cells": 0,
        "duplicate_main_numbers_rows": 0,
    }

    number_columns = (
        *main_columns,
        *bonus_columns,
    )
    out_of_range = 0

    for column in number_columns:
        out_of_range += int(
            (
                ~df[column].between(
                    min_num,
                    max_num,
                )
            ).sum()
        )

    duplicate_main_rows = 0

    for row in df[
        list(main_columns)
    ].itertuples(
        index=False,
        name=None,
    ):
        values = [
            int(value)
            for value in row
        ]

        if len(values) != len(
            set(values)
        ):
            duplicate_main_rows += 1

    report[
        "out_of_range_cells"
    ] = out_of_range
    report[
        "duplicate_main_numbers_rows"
    ] = duplicate_main_rows

    report["status"] = (
        "ok"
        if (
            report["missing_cells"] == 0
            and report[
                "duplicate_draw_no"
            ] == 0
            and report[
                "out_of_range_cells"
            ] == 0
            and report[
                "duplicate_main_numbers_rows"
            ] == 0
        )
        else "warning"
    )

    return report


def dataframe_to_history(
    df: pd.DataFrame,
    config: Mapping[str, object] | object,
) -> tuple[tuple[int, ...], ...]:
    main_columns = tuple(
        str(column)
        for column in _config_value(
            config,
            "main_cols",
            "main_columns",
            default=(),
        )
    )

    if not main_columns:
        raise ValueError(
            "main columns are not "
            "configured."
        )

    rows = df[
        list(main_columns)
    ].itertuples(
        index=False,
        name=None,
    )

    if _game_family(config) == "numbers":
        return tuple(
            tuple(
                int(value)
                for value in row
            )
            for row in rows
        )

    return tuple(
        tuple(
            sorted(
                int(value)
                for value in row
            )
        )
        for row in rows
    )


def save_game_dataframe(
    df: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(
        destination,
        index=False,
        encoding="utf-8",
    )


def load_game_data(
    game_name: str,
    config: Mapping[str, object] | object,
    *,
    destination: Path | None = None,
) -> LoadedGameData:
    text, source = download_game_csv(
        game_name,
        config,
    )
    raw = read_csv_text(text)
    dataframe = (
        normalize_game_dataframe(
            raw,
            config,
        )
    )
    validation = validate_lottery(
        dataframe,
        config,
    )

    if destination is not None:
        save_game_dataframe(
            dataframe,
            destination,
        )

    return LoadedGameData(
        game_name=game_name,
        dataframe=dataframe,
        validation=validation,
        source=source,
    )
