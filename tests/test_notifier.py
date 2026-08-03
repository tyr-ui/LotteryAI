import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from notifier import (
    NotificationConfigurationError,
    load_discord_payload,
    send_from_environment,
    send_notification_discord,
)


class NotifierTest(unittest.TestCase):

    def test_missing_environment_is_rejected(self):
        with self.assertRaises(NotificationConfigurationError):
            send_from_environment(Path("missing.json"), environ={})

    def test_invalid_empty_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text('{"messages": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_discord_payload(path)

    @patch("notifier.urlopen")
    def test_send_posts_embed_json_without_real_webhook(self, mock_urlopen):
        response = MagicMock()
        response.status = 204
        mock_urlopen.return_value.__enter__.return_value = response

        payload = {
            "messages": [
                {
                    "username": "LotteryAI",
                    "allowed_mentions": {"parse": []},
                    "embeds": [
                        {
                            "title": "LotteryAI",
                            "description": "通知テスト",
                            "color": 12345,
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notification_payload.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            count = send_notification_discord(
                path,
                webhook_url="https://discord.com/api/webhooks/test/token",
            )

        self.assertEqual(count, 1)
        request = mock_urlopen.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["embeds"][0]["title"], "LotteryAI")
        self.assertEqual(sent["allowed_mentions"], {"parse": []})


if __name__ == "__main__":
    unittest.main()
