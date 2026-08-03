"""Fetch and parse official LOTO6/LOTO7 carryover amounts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping
import unicodedata

import requests
from bs4 import BeautifulSoup

from common import now_iso
from data_loader import decode_content, get_headers


MIZUHO_CARRYOVER_URLS = {
    "loto6": (
        "https://www.mizuhobank.co.jp/"
        "takarakuji/check/loto/loto6/index.html"
    ),
    "loto7": (
        "https://www.mizuhobank.co.jp/"
        "takarakuji/check/loto/loto7/index.html"
    ),
}

GAME_LABELS = {
    "loto6": "ロト6",
    "loto7": "ロト7",
}


@dataclass(frozen=True, slots=True)
class CarryoverResult:
    game_key: str
    status: str
    source: str
    source_url: str
    expected_draw_no: int | None
    draw_no: int | None
    next_draw_no: int | None
    amount_yen: int | None
    has_carryover: bool | None
    display_amount: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def format_yen_japanese(amount_yen: int | None) -> str:
    """Format yen using Japanese 億/万 units without losing exact yen."""
    if amount_yen is None:
        return "未取得"
    if amount_yen <= 0:
        return "なし"

    amount = int(amount_yen)
    oku, remainder = divmod(amount, 100_000_000)
    man, yen = divmod(remainder, 10_000)

    parts: list[str] = []
    if oku:
        parts.append(f"{oku:,}億")
    if man:
        parts.append(f"{man:,}万")
    if yen:
        parts.append(f"{yen:,}")

    return "".join(parts) + "円"


def _normalized_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text)


def parse_mizuho_carryover_html(
    html: str,
    *,
    game_key: str,
    expected_draw_no: int | None = None,
    source_url: str | None = None,
) -> CarryoverResult:
    """Parse the latest official carryover amount from a Mizuho result page."""
    if game_key not in GAME_LABELS:
        raise ValueError(f"Unsupported carryover game: {game_key}")

    url = source_url or MIZUHO_CARRYOVER_URLS[game_key]
    label = GAME_LABELS[game_key]
    text = _normalized_page_text(html)

    marker_pattern = re.compile(
        rf"{re.escape(label)}\s*第\s*(\d+)\s*回"
    )
    markers = list(marker_pattern.finditer(text))

    if not markers:
        return CarryoverResult(
            game_key=game_key,
            status="parse_error",
            source="mizuho_bank",
            source_url=url,
            expected_draw_no=expected_draw_no,
            draw_no=None,
            next_draw_no=(expected_draw_no + 1 if expected_draw_no else None),
            amount_yen=None,
            has_carryover=None,
            display_amount="未取得",
            message="公式ページから抽せん回号を読み取れませんでした。",
        )

    selected_index: int | None = None
    if expected_draw_no is not None:
        for index, marker in enumerate(markers):
            if int(marker.group(1)) == int(expected_draw_no):
                selected_index = index
                break

    if selected_index is None:
        selected_index = max(
            range(len(markers)),
            key=lambda index: int(markers[index].group(1)),
        )

    marker = markers[selected_index]
    draw_no = int(marker.group(1))
    block_end = (
        markers[selected_index + 1].start()
        if selected_index + 1 < len(markers)
        else len(text)
    )
    block = text[marker.start():block_end]

    amount_match = re.search(
        r"キャリーオーバー\s*[.:：]?\s*([0-9,]+)\s*円",
        block,
    )
    if amount_match is None:
        return CarryoverResult(
            game_key=game_key,
            status="parse_error",
            source="mizuho_bank",
            source_url=url,
            expected_draw_no=expected_draw_no,
            draw_no=draw_no,
            next_draw_no=draw_no + 1,
            amount_yen=None,
            has_carryover=None,
            display_amount="未取得",
            message="公式ページからキャリーオーバー金額を読み取れませんでした。",
        )

    amount_yen = int(amount_match.group(1).replace(",", ""))
    status = "ok"
    message = "公式ページから取得しました。"

    if expected_draw_no is not None and draw_no != int(expected_draw_no):
        status = "stale"
        message = (
            "公式ページの最新回号がLotteryAIの最新回号と一致しません。"
        )

    return CarryoverResult(
        game_key=game_key,
        status=status,
        source="mizuho_bank",
        source_url=url,
        expected_draw_no=expected_draw_no,
        draw_no=draw_no,
        next_draw_no=draw_no + 1,
        amount_yen=amount_yen,
        has_carryover=amount_yen > 0,
        display_amount=format_yen_japanese(amount_yen),
        message=message,
    )


def fetch_mizuho_carryover(
    game_key: str,
    *,
    expected_draw_no: int | None = None,
    timeout: float = 30.0,
) -> CarryoverResult:
    """Fetch one official Mizuho result page and parse its carryover."""
    if game_key not in MIZUHO_CARRYOVER_URLS:
        raise ValueError(f"Unsupported carryover game: {game_key}")

    url = MIZUHO_CARRYOVER_URLS[game_key]
    try:
        response = requests.get(
            url,
            headers=get_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        html = decode_content(response.content)
    except requests.RequestException as exc:
        return CarryoverResult(
            game_key=game_key,
            status="fetch_error",
            source="mizuho_bank",
            source_url=url,
            expected_draw_no=expected_draw_no,
            draw_no=None,
            next_draw_no=(expected_draw_no + 1 if expected_draw_no else None),
            amount_yen=None,
            has_carryover=None,
            display_amount="未取得",
            message=f"公式ページの取得に失敗しました: {type(exc).__name__}",
        )

    return parse_mizuho_carryover_html(
        html,
        game_key=game_key,
        expected_draw_no=expected_draw_no,
        source_url=url,
    )


def fetch_carryover_snapshot(
    latest_draw_numbers: Mapping[str, object],
) -> dict[str, object]:
    """Fetch LOTO6/LOTO7 carryovers without failing the prediction pipeline."""
    games: dict[str, object] = {}

    for game_key in MIZUHO_CARRYOVER_URLS:
        raw_draw_no = latest_draw_numbers.get(game_key)
        expected_draw_no = (
            int(raw_draw_no)
            if isinstance(raw_draw_no, (int, float))
            and not isinstance(raw_draw_no, bool)
            else None
        )
        result = fetch_mizuho_carryover(
            game_key,
            expected_draw_no=expected_draw_no,
        )
        games[game_key] = result.to_dict()

    return {
        "source": "mizuho_bank",
        "fetched_at": now_iso(),
        "games": games,
    }


__all__ = [
    "CarryoverResult",
    "MIZUHO_CARRYOVER_URLS",
    "fetch_carryover_snapshot",
    "fetch_mizuho_carryover",
    "format_yen_japanese",
    "parse_mizuho_carryover_html",
]
