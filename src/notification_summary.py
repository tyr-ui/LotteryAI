from pathlib import Path
from storage import save_json

def write_notification_summary(output_dir: Path, output: dict)->Path:
    p=output_dir/'notification_summary.md'
    lines=['# LotteryAI Notification','','This is a placeholder summary.']
    p.write_text("\n".join(lines),encoding='utf-8')
    return p
