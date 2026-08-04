"""Send LotteryAI Discord Embed payloads through a webhook."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class NotificationConfigurationError(RuntimeError):
    """Raised when required notification configuration is missing."""


class NotificationDeliveryError(RuntimeError):
    """Raised when Discord rejects or cannot receive a notification."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return list(value)
    return []


def _required_env(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(name, "")).strip()
    if not value:
        raise NotificationConfigurationError(
            f"必要な環境変数が設定されていません: {name}"
        )
    return value


def load_discord_payload(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Discord notification payload was not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    messages = _list(_mapping(raw).get("messages"))
    if not messages:
        raise ValueError("Discord notification payload contains no messages.")

    validated: list[dict[str, object]] = []
    for index, item in enumerate(messages, start=1):
        message = dict(_mapping(item))
        embeds = _list(message.get("embeds"))
        content = str(message.get("content", "")).strip()
        if not embeds and not content:
            raise ValueError(f"Discord message {index} has neither embeds nor content.")
        if len(embeds) > 10:
            raise ValueError(f"Discord message {index} exceeds the 10-embed limit.")
        message.setdefault("username", "LotteryAI")
        message.setdefault("allowed_mentions", {"parse": []})
        validated.append(message)

    return validated


def _post_discord_payload(
    webhook_url: str,
    payload: Mapping[str, object],
    *,
    timeout: float = 20.0,
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "LotteryAI-Notifier/2.0",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 204))
            if status not in {200, 204}:
                raise NotificationDeliveryError(
                    f"Discord webhook returned HTTP {status}."
                )
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        suffix = f" Response: {detail[:500]}" if detail else ""
        raise NotificationDeliveryError(
            f"Discord webhook returned HTTP {exc.code}.{suffix}"
        ) from exc
    except URLError as exc:
        raise NotificationDeliveryError(
            f"Discord webhook request failed: {exc.reason}"
        ) from exc


def send_notification_discord(
    payload_path: Path,
    *,
    webhook_url: str,
) -> int:
    messages = load_discord_payload(payload_path)
    for message in messages:
        _post_discord_payload(webhook_url, message)
    return len(messages)


def send_from_environment(
    payload_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if environ is None else environ
    webhook_url = _required_env("LOTTERY_DISCORD_WEBHOOK", source)
    return send_notification_discord(payload_path, webhook_url=webhook_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LotteryAIのDiscordカード通知を送信します。"
    )
    parser.add_argument(
        "payload_path",
        nargs="?",
        default="output/notification_payload.json",
    )
    args = parser.parse_args()

    count = send_from_environment(Path(args.payload_path))
    print(f"Discord通知を送信しました: {count}件")


if __name__ == "__main__":
    main()
