"""Send the generated LotteryAI notification summary by email."""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Mapping


DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 465


class NotificationConfigurationError(RuntimeError):
    """Raised when required mail configuration is missing."""


def _required_env(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(name, "")).strip()
    if not value:
        raise NotificationConfigurationError(
            f"Required environment variable is missing: {name}"
        )
    return value


def build_email_subject(
    summary_text: str,
    *,
    generated_at: datetime | None = None,
) -> str:
    """Build a concise subject without claiming unavailable carryover data."""
    timestamp = generated_at or datetime.now()
    date_text = timestamp.strftime("%Y/%m/%d")

    pending = "まだ結果未反映" in summary_text
    suffix = "（一部結果待ち）" if pending else ""
    return f"【LotteryAI】{date_text} 次回予想・前回結果{suffix}"


def build_email_message(
    *,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body, subtype="plain", charset="utf-8")
    return message


def send_notification_email(
    summary_path: Path,
    *,
    sender: str,
    app_password: str,
    recipient: str,
    smtp_host: str = DEFAULT_SMTP_HOST,
    smtp_port: int = DEFAULT_SMTP_PORT,
) -> str:
    """Send a Markdown summary as a UTF-8 plain-text email."""
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Notification summary was not found: {summary_path}"
        )

    body = summary_path.read_text(encoding="utf-8").strip()
    if not body:
        raise ValueError("Notification summary is empty.")

    subject = build_email_subject(body)
    message = build_email_message(
        sender=sender,
        recipient=recipient,
        subject=subject,
        body=body,
    )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(
        smtp_host,
        int(smtp_port),
        context=context,
    ) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(message)

    return subject


def send_from_environment(
    summary_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environ is None else environ
    sender = _required_env("LOTTERY_MAIL_USERNAME", source)
    app_password = _required_env("LOTTERY_MAIL_APP_PASSWORD", source)
    recipient = _required_env("LOTTERY_MAIL_TO", source)

    smtp_host = str(
        source.get("LOTTERY_MAIL_SMTP_HOST", DEFAULT_SMTP_HOST)
    ).strip() or DEFAULT_SMTP_HOST
    smtp_port = int(
        str(source.get("LOTTERY_MAIL_SMTP_PORT", DEFAULT_SMTP_PORT))
    )

    return send_notification_email(
        summary_path,
        sender=sender,
        app_password=app_password,
        recipient=recipient,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send the LotteryAI notification summary by email."
    )
    parser.add_argument(
        "summary_path",
        nargs="?",
        default="output/notification_summary.md",
    )
    args = parser.parse_args()

    subject = send_from_environment(Path(args.summary_path))
    print(f"Notification email sent: {subject}")


if __name__ == "__main__":
    main()
