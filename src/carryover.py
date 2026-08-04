"""Fetch LOTO6/LOTO7 carryover information with safe fallbacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time
from typing import Mapping
import unicodedata

import requests
from bs4 import BeautifulSoup

from common import now_iso
from data_loader import decode_content, get_headers


MIZUHO_CARRYOVER_URLS = {
    "loto6": "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html",
    "loto7": "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto7/index.html",
}

OFFICIAL_STATUS_URLS = {
    "loto6": "https://www.takarakuji-official.jp/kuji/loto/loto6/",
    "loto7": "https://www.takarakuji-official.jp/kuji/loto/loto7/",
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
    if game_key not in GAME_LABELS:
        raise ValueError(f"Unsupported carryover game: {game_key}")

    url = source_url or MIZUHO_CARRYOVER_URLS[game_key]
    label = GAME_LABELS[game_key]
    text = _normalized_page_text(html)
    marker_pattern = re.compile(rf"{re.escape(label)}\s*第\s*(\d+)\s*回")
    markers = list(marker_pattern.finditer(text))

    if not markers:
        return CarryoverResult(
            game_key, "parse_error", "mizuho_bank", url,
            expected_draw_no, None,
            expected_draw_no + 1 if expected_draw_no else None,
            None, None, "未取得",
            "公式ページから抽せん回号を読み取れませんでした。",
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
            game_key, "parse_error", "mizuho_bank", url,
            expected_draw_no, draw_no, draw_no + 1,
            None, None, "未取得",
            "公式ページからキャリーオーバー金額を読み取れませんでした。",
        )

    amount_yen = int(amount_match.group(1).replace(",", ""))
    status = "ok"
    message = "みずほ銀行の公式ページから取得しました。"
    if expected_draw_no is not None and draw_no != int(expected_draw_no):
        status = "stale"
        message = "公式ページの最新回号がLotteryAIの最新回号と一致しません。"

    return CarryoverResult(
        game_key, status, "mizuho_bank", url,
        expected_draw_no, draw_no, draw_no + 1,
        amount_yen, amount_yen > 0,
        format_yen_japanese(amount_yen), message,
    )


def parse_official_status_html(
    html: str,
    *,
    game_key: str,
    expected_draw_no: int | None = None,
    source_url: str | None = None,
) -> CarryoverResult:
    """Read only an explicit current-sales carryover banner from official site."""
    if game_key not in GAME_LABELS:
        raise ValueError(f"Unsupported carryover game: {game_key}")

    url = source_url or OFFICIAL_STATUS_URLS[game_key]
    soup = BeautifulSoup(html, "html.parser")
    explicit_signal = False

    for tag in soup.find_all(["img", "source"]):
        candidates = [
            tag.get("alt"),
            tag.get("title"),
            tag.get("aria-label"),
        ]
        if any(
            value and "キャリーオーバー発生中" in unicodedata.normalize("NFKC", str(value))
            for value in candidates
        ):
            explicit_signal = True
            break

    if not explicit_signal:
        for tag in soup.find_all(attrs={"data-carryover": True}):
            value = str(tag.get("data-carryover", "")).lower()
            if value in {"1", "true", "yes", "active"}:
                explicit_signal = True
                break

    if explicit_signal:
        return CarryoverResult(
            game_key, "status_only", "takarakuji_official", url,
            expected_draw_no, expected_draw_no,
            expected_draw_no + 1 if expected_draw_no else None,
            None, True, "発生中（金額未取得）",
            "宝くじ公式サイトで発生中を確認しましたが、金額は取得できませんでした。",
        )

    return CarryoverResult(
        game_key, "status_unknown", "takarakuji_official", url,
        expected_draw_no, expected_draw_no,
        expected_draw_no + 1 if expected_draw_no else None,
        None, None, "未取得",
        "宝くじ公式サイトで明示的な発生表示を確認できませんでした。",
    )


def _get_with_retries(
    url: str,
    *,
    timeout: float,
    attempts: int = 3,
) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=get_headers(), timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_mizuho_carryover(
    game_key: str,
    *,
    expected_draw_no: int | None = None,
    timeout: float = 20.0,
) -> CarryoverResult:
    if game_key not in MIZUHO_CARRYOVER_URLS:
        raise ValueError(f"Unsupported carryover game: {game_key}")
    url = MIZUHO_CARRYOVER_URLS[game_key]
    try:
        response = _get_with_retries(url, timeout=timeout)
        html = decode_content(response.content)
    except requests.RequestException as exc:
        return CarryoverResult(
            game_key, "fetch_error", "mizuho_bank", url,
            expected_draw_no, None,
            expected_draw_no + 1 if expected_draw_no else None,
            None, None, "未取得",
            f"みずほ銀行の公式ページ取得に失敗しました: {type(exc).__name__}",
        )
    return parse_mizuho_carryover_html(
        html,
        game_key=game_key,
        expected_draw_no=expected_draw_no,
        source_url=url,
    )


def fetch_official_status(
    game_key: str,
    *,
    expected_draw_no: int | None = None,
    timeout: float = 20.0,
) -> CarryoverResult:
    url = OFFICIAL_STATUS_URLS[game_key]
    try:
        response = _get_with_retries(url, timeout=timeout, attempts=2)
        html = decode_content(response.content)
    except requests.RequestException as exc:
        return CarryoverResult(
            game_key, "fetch_error", "takarakuji_official", url,
            expected_draw_no, None,
            expected_draw_no + 1 if expected_draw_no else None,
            None, None, "未取得",
            f"宝くじ公式サイトの取得に失敗しました: {type(exc).__name__}",
        )
    return parse_official_status_html(
        html,
        game_key=game_key,
        expected_draw_no=expected_draw_no,
        source_url=url,
    )


def _cached_result(
    previous_snapshot: Mapping[str, object] | None,
    game_key: str,
    expected_draw_no: int | None,
) -> CarryoverResult | None:
    if not previous_snapshot or expected_draw_no is None:
        return None
    games = previous_snapshot.get("games")
    if not isinstance(games, Mapping):
        return None
    row = games.get(game_key)
    if not isinstance(row, Mapping):
        return None
    if int(row.get("draw_no") or -1) != int(expected_draw_no):
        return None
    amount = row.get("amount_yen")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return None
    amount_yen = int(amount)
    return CarryoverResult(
        game_key, "cached", "previous_success", str(row.get("source_url") or ""),
        expected_draw_no, expected_draw_no, expected_draw_no + 1,
        amount_yen, amount_yen > 0,
        format_yen_japanese(amount_yen),
        "公式サイトへ接続できなかったため、同じ回号で以前取得した金額を使用しました。",
    )


def fetch_carryover_snapshot(
    latest_draw_numbers: Mapping[str, object],
    *,
    previous_snapshot: Mapping[str, object] | None = None,
) -> dict[str, object]:
    games: dict[str, object] = {}

    for game_key in MIZUHO_CARRYOVER_URLS:
        raw_draw_no = latest_draw_numbers.get(game_key)
        expected_draw_no = (
            int(raw_draw_no)
            if isinstance(raw_draw_no, (int, float)) and not isinstance(raw_draw_no, bool)
            else None
        )

        primary = fetch_mizuho_carryover(
            game_key,
            expected_draw_no=expected_draw_no,
        )
        if primary.status in {"ok", "stale"}:
            result = primary
        else:
            cached = _cached_result(previous_snapshot, game_key, expected_draw_no)
            if cached is not None:
                result = cached
            else:
                fallback = fetch_official_status(
                    game_key,
                    expected_draw_no=expected_draw_no,
                )
                result = fallback if fallback.status == "status_only" else primary
                if result is primary and fallback.status == "fetch_error":
                    result = CarryoverResult(
                        primary.game_key, primary.status, primary.source,
                        primary.source_url, primary.expected_draw_no,
                        primary.draw_no, primary.next_draw_no,
                        primary.amount_yen, primary.has_carryover,
                        primary.display_amount,
                        primary.message + " / " + fallback.message,
                    )
        games[game_key] = result.to_dict()

    return {
        "source": "official_multi_source",
        "fetched_at": now_iso(),
        "games": games,
    }


__all__ = [
    "CarryoverResult",
    "MIZUHO_CARRYOVER_URLS",
    "OFFICIAL_STATUS_URLS",
    "fetch_carryover_snapshot",
    "fetch_mizuho_carryover",
    "fetch_official_status",
    "format_yen_japanese",
    "parse_mizuho_carryover_html",
    "parse_official_status_html",
]
