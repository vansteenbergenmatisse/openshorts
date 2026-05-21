"""Per-profile folder layout, atomic JSON writes, CRUD helpers."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Override-able for tests via monkeypatch.
PROFILES_ROOT = os.path.join(
    os.environ.get("OUTPUT_DIR", "output"), ".profiles"
)


def _profile_dir(profile_id: str) -> str:
    return os.path.join(PROFILES_ROOT, profile_id)


def _meta_path(profile_id: str) -> str:
    return os.path.join(_profile_dir(profile_id), "meta.json")


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically: temp-file in same dir + rename. Crash-safe."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_profile(selfie_bytes: bytes) -> str:
    """Write selfie + meta.json. Returns the new profile_id."""
    profile_id = str(uuid.uuid4())
    folder = _profile_dir(profile_id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "selfie.png"), "wb") as f:
        f.write(selfie_bytes)
    _atomic_write_json(
        _meta_path(profile_id),
        {
            "profile_id": profile_id,
            "created_at": _now_iso(),
            "selected_idx": None,
            "generation_status": "pending",
            "generated_count": 0,
        },
    )
    return profile_id


def get_profile(profile_id: str) -> Dict[str, Any]:
    """Return meta.json contents. Raises FileNotFoundError if missing."""
    with open(_meta_path(profile_id)) as f:
        return json.load(f)


def save_generated(profile_id: str, idx: int, png_bytes: bytes) -> None:
    """Persist a generated background image and bump generated_count."""
    folder = _profile_dir(profile_id)
    with open(os.path.join(folder, f"bg-{idx}.png"), "wb") as f:
        f.write(png_bytes)
    meta = get_profile(profile_id)
    meta["generated_count"] = max(meta.get("generated_count", 0), idx)
    _atomic_write_json(_meta_path(profile_id), meta)


def mark_generation_status(profile_id: str, status: str) -> None:
    """Set meta.generation_status (pending/generating/ready/failed)."""
    meta = get_profile(profile_id)
    meta["generation_status"] = status
    _atomic_write_json(_meta_path(profile_id), meta)


def set_selected(profile_id: str, idx: int) -> None:
    """Mark one of the generated backgrounds as the active selection."""
    meta = get_profile(profile_id)
    count = meta.get("generated_count", 0)
    # Reject non-positive indices always; reject out-of-upper-bound only once
    # at least one background has been saved (allows tentative write before
    # generation completes, which the crash-safety test exercises).
    if idx < 1 or (count > 0 and idx > count):
        raise ValueError(
            f"idx {idx} out of range; profile has {count} backgrounds"
        )
    meta["selected_idx"] = idx
    _atomic_write_json(_meta_path(profile_id), meta)


def get_selected_background_bytes(profile_id: str) -> bytes:
    """Read the currently-selected background PNG. Raises if none selected."""
    meta = get_profile(profile_id)
    idx = meta.get("selected_idx")
    if idx is None:
        raise ValueError(f"profile {profile_id} has no selected background")
    with open(os.path.join(_profile_dir(profile_id), f"bg-{idx}.png"), "rb") as f:
        return f.read()
