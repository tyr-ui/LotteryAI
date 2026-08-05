from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

from common import now_iso
from storage import load_json, save_json

SCHEMA_VERSION = "1.0"
MODEL_SOURCE_FILES = (
    "src/optimizer.py",
    "src/optimizer_search.py",
    "src/optimizer_evolution.py",
    "src/optimizer_adaptation.py",
    "src/optimizer_evaluation.py",
    "src/predictor.py",
    "src/numbers_optimizer.py",
    "src/numbers_predictor.py",
    "src/numbers_features.py",
    "src/games.py",
)


def model_fingerprint(root: Path, paths: Iterable[str] = MODEL_SOURCE_FILES) -> str:
    digest = sha256()
    for relative in sorted(paths):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_evaluation_epoch(root: Path, output_dir: Path) -> dict[str, object]:
    """Return the active evaluation epoch, starting a new one on model changes."""
    path = output_dir / "evaluation_epoch.json"
    store = load_json(path, {})
    if not isinstance(store, dict):
        store = {}

    fingerprint = model_fingerprint(root)
    epochs = store.get("epochs", [])
    if not isinstance(epochs, list):
        epochs = []

    current = store.get("current")
    if isinstance(current, dict) and current.get("model_fingerprint") == fingerprint:
        return current

    created_at = now_iso()
    if isinstance(current, dict):
        closed = dict(current)
        closed["status"] = "closed"
        closed["ended_at"] = created_at
        epochs = [item for item in epochs if not (
            isinstance(item, dict) and item.get("epoch_id") == current.get("epoch_id")
        )]
        epochs.append(closed)
        next_id = int(current.get("epoch_id", 0)) + 1
        reason = "model_fingerprint_changed"
    else:
        next_id = 1
        reason = "initial_epoch"

    current = {
        "epoch_id": next_id,
        "model_version": f"epoch-{next_id}",
        "model_fingerprint": fingerprint,
        "started_at": created_at,
        "status": "active",
        "reason": reason,
    }
    save_json(path, {
        "schema_version": SCHEMA_VERSION,
        "current": current,
        "epochs": epochs,
    })
    return current
