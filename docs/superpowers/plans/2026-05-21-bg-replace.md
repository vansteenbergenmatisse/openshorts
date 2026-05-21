# AI Restyle v2: Background Replacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the AI Restyle v1 pipeline (Nano-Banana relight + fal.ai v2v) with a background-replacement product: one-time selfie onboarding → 5 personalized Gemini-generated backgrounds → fal.ai video matting → composite over blurred background.

**Architecture:** Per-browser opaque `profile_id` in `localStorage`, backend persists profile state under `backend/output/.profiles/<id>/`. Per-video pipeline is 6 async steps (probe → bg-clean check → matte → composite → mux audio → publish). All new HTTP surfaces share the existing `routes/ai_restyle.py` router and the existing auth-via-header pattern. Source-of-truth design: [docs/superpowers/specs/2026-05-21-bg-replace-design.md](../specs/2026-05-21-bg-replace-design.md).

**Tech Stack:** Python 3.11, FastAPI, google-genai (Gemini 3.1 Flash image preview), httpx (fal.ai client), OpenCV (cv2 — already a dep), FFmpeg (via `app/video/ffmpeg.py` wrapper), React 18 + Vite 4 + Tailwind 3.4.

**Prerequisites:**
- PR #35 (AI Restyle v1) has been merged into `main`.
- Branch off main: `git checkout main && git pull origin main && git checkout -b feat/ai-restyle-v2-bg-replace`.
- `docker compose up --build` is running (backend on :3002).
- Backend tests run inside the container: `docker exec -w /app openshorts-backend pytest -m "not e2e" -q`.

---

## File Structure

### Created

| File | Responsibility |
| --- | --- |
| `backend/app/profile/__init__.py` | Package docstring (auto-picked up by `scripts/update_claude_md.py`). |
| `backend/app/profile/store.py` | Per-profile folder layout, atomic JSON writes, CRUD helpers. |
| `backend/app/ml/profile_backgrounds.py` | Gemini-driven 5-background generation from a selfie. |
| `backend/app/ml/bg_detect.py` | OpenCV-based heuristic for "is the source background clean enough". |
| `backend/app/ml/video_matte.py` | fal.ai video-matting REST client wrapper with SSRF guards. |
| `backend/app/video/composite.py` | FFmpeg composite of an RGBA matte video over a blurred background still. |
| `backend/tests/unit/test_profile_store.py` | Round-trip + concurrent-write tests. |
| `backend/tests/unit/test_profile_backgrounds.py` | Mocked-Gemini tests. |
| `backend/tests/unit/test_bg_detect.py` | Six fixture-driven tests (3 clean / 3 noisy). |
| `backend/tests/unit/test_video_matte.py` | Mocked-fal tests + SSRF rejection. |
| `backend/tests/unit/test_composite.py` | Synthetic alpha+bg composite test. |
| `backend/tests/api/test_profile_routes.py` | Onboarding HTTP surface tests. |
| `backend/tests/fixtures/bg_detect/clean_*.mp4` + `noisy_*.mp4` | 6 small synthetic clips (≈100KB each). |
| `frontend/src/state/profileStore.js` | localStorage profileId + API helpers + React hook. |
| `frontend/src/pages/Settings/sections/BackgroundProfileSection.jsx` | Selfie upload + 5-bg grid + select/regenerate UI. |
| `frontend/src/pages/AIRestyle/steps/Precheck.jsx` | Per-video step 2: shows clean-bg verdict + selected profile bg + Submit button. |

### Modified

| File | Change |
| --- | --- |
| `backend/app/restyle/pipeline.py` | Full rewrite: 6-step background-replacement flow. |
| `backend/app/routes/ai_restyle.py` | Add 4 onboarding routes + 1 static-serve; add `profile_id` form to existing POST; remove `background_prompt` + `lighting_prompt` fields. |
| `backend/app/main.py` | Mount new static `/profiles` route (or include via existing router). |
| `backend/tests/api/test_ai_restyle.py` | Update existing tests to use `profile_id` form field; drop relight-prompt assertions. |
| `backend/tests/snapshots/baseline.openapi.json` | Regenerate to include new routes + updated POST schema. |
| `frontend/src/pages/AIRestyle/Wizard.jsx` | Step 2 now `Precheck` instead of `Configure`. |
| `frontend/src/pages/AIRestyle/steps/Upload.jsx` | Add profile-missing redirect link. |
| `frontend/src/pages/AIRestyle/steps/Review.jsx` | Show profile background thumbnail in summary. |
| `frontend/src/pages/Settings/sections/ApiKeysSection.jsx` | Update fal.ai copy: "matting" not "v2v restyle". |
| `frontend/src/App.jsx` | Wire Settings route to `BackgroundProfileSection` instead of `AIRestylePresetsSection`. |
| `ROADMAP.md` | AI Restyle entry reframed: v1 retired, v2 shipped. |

### Deleted

| File | Reason |
| --- | --- |
| `backend/app/ml/frame_relight.py` | v1 relight step is gone. |
| `backend/app/ml/video_restyle.py` | v1 fal v2v step is gone. |
| `backend/tests/unit/test_frame_relight.py` | Dead. |
| `backend/tests/unit/test_video_restyle.py` | Dead. |
| `backend/tests/unit/test_restyle_pipeline.py` | Replaced by new test in Task 7. |
| `frontend/src/pages/Settings/sections/AIRestylePresetsSection.jsx` | Replaced by `BackgroundProfileSection`. |
| `frontend/src/pages/AIRestyle/steps/Configure.jsx` | Replaced by `Precheck`. |
| `frontend/src/state/aiRestylePresets.js` | Per-job preset CRUD gone. |

---

## Task 1: Profile store

**Files:**
- Create: `backend/app/profile/__init__.py`
- Create: `backend/app/profile/store.py`
- Create: `backend/tests/unit/test_profile_store.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_profile_store.py`:

```python
"""Tests for the per-profile JSON+files store."""
from __future__ import annotations

import json
import os

import pytest

from app.profile import store as profile_store


@pytest.fixture
def tmp_profile_root(tmp_path, monkeypatch):
    root = tmp_path / "profiles"
    monkeypatch.setattr(profile_store, "PROFILES_ROOT", str(root))
    return root


def test_create_profile_writes_selfie_and_meta(tmp_profile_root):
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    assert pid
    folder = tmp_profile_root / pid
    assert (folder / "selfie.png").exists()
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["profile_id"] == pid
    assert meta["selected_idx"] is None
    assert meta["generation_status"] == "pending"
    assert meta["generated_count"] == 0


def test_save_generated_then_select_updates_meta(tmp_profile_root):
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n")
    profile_store.save_generated(pid, idx=1, png_bytes=b"\x89PNG\r\n\x1a\nbg1")
    profile_store.save_generated(pid, idx=2, png_bytes=b"\x89PNG\r\n\x1a\nbg2")
    profile_store.mark_generation_status(pid, "ready")
    profile_store.set_selected(pid, idx=2)

    meta = profile_store.get_profile(pid)
    assert meta["generation_status"] == "ready"
    assert meta["generated_count"] == 2
    assert meta["selected_idx"] == 2
    assert (tmp_profile_root / pid / "bg-1.png").read_bytes().endswith(b"bg1")
    assert (tmp_profile_root / pid / "bg-2.png").read_bytes().endswith(b"bg2")


def test_set_selected_rejects_out_of_range(tmp_profile_root):
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n")
    profile_store.save_generated(pid, idx=1, png_bytes=b"x")
    with pytest.raises(ValueError):
        profile_store.set_selected(pid, idx=9)
    with pytest.raises(ValueError):
        profile_store.set_selected(pid, idx=0)


def test_get_selected_background_bytes(tmp_profile_root):
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n")
    profile_store.save_generated(pid, idx=3, png_bytes=b"\x89PNG\r\n\x1a\ntarget")
    profile_store.set_selected(pid, idx=3)
    assert profile_store.get_selected_background_bytes(pid).endswith(b"target")


def test_get_profile_missing_raises(tmp_profile_root):
    with pytest.raises(FileNotFoundError):
        profile_store.get_profile("nonexistent")


def test_atomic_write_no_partial_meta_on_crash(tmp_profile_root, monkeypatch):
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n")
    meta_path = tmp_profile_root / pid / "meta.json"
    original = meta_path.read_text()

    # Simulate a crash mid-write by forcing os.replace to raise.
    def boom(*args, **kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        profile_store.set_selected(pid, idx=1)
    # Original file must still be intact.
    assert meta_path.read_text() == original
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_profile_store.py -v
```
Expected: All fail (`ModuleNotFoundError: No module named 'app.profile'`).

- [ ] **Step 3: Create the package init**

Create `backend/app/profile/__init__.py`:

```python
"""Per-browser onboarding profile store: selfie + generated backgrounds + selection."""
```

- [ ] **Step 4: Implement the store**

Create `backend/app/profile/store.py`:

```python
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
    if idx < 1 or idx > meta.get("generated_count", 0):
        raise ValueError(
            f"idx {idx} out of range; profile has {meta.get('generated_count', 0)} backgrounds"
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_profile_store.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/profile/ backend/tests/unit/test_profile_store.py
git commit -m "feat(bg-replace): per-profile store with atomic JSON writes"
```

---

## Task 2: Gemini background generation

**Files:**
- Create: `backend/app/ml/profile_backgrounds.py`
- Create: `backend/tests/unit/test_profile_backgrounds.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_profile_backgrounds.py`:

```python
"""Tests for the Gemini-driven 5-background generator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ml import profile_backgrounds


class _FakePart:
    def __init__(self, data: bytes):
        self.inline_data = MagicMock()
        self.inline_data.data = data


def _fake_response(image_count: int):
    resp = MagicMock()
    resp.parts = [_FakePart(b"\x89PNG\r\n\x1a\nfake-bg-%d" % i) for i in range(image_count)]
    return resp


def test_generates_five_backgrounds(tmp_path, monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(5)
    monkeypatch.setattr(profile_backgrounds.genai, "Client", lambda **kw: fake_client)

    selfie = tmp_path / "selfie.png"
    selfie.write_bytes(b"\x89PNG\r\n\x1a\nselfie")

    paths = profile_backgrounds.generate_personalized_backgrounds(
        api_key="test",
        selfie_path=str(selfie),
        out_dir=str(tmp_path / "out"),
        count=5,
    )
    assert len(paths) == 5
    for i, p in enumerate(paths, start=1):
        assert p.endswith(f"bg-{i}.png")
        assert open(p, "rb").read().endswith(b"fake-bg-%d" % (i - 1))


def test_raises_on_missing_selfie(tmp_path):
    with pytest.raises(FileNotFoundError):
        profile_backgrounds.generate_personalized_backgrounds(
            api_key="x",
            selfie_path=str(tmp_path / "nope.png"),
            out_dir=str(tmp_path),
            count=5,
        )


def test_raises_when_fewer_images_returned(tmp_path, monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(3)
    monkeypatch.setattr(profile_backgrounds.genai, "Client", lambda **kw: fake_client)

    selfie = tmp_path / "selfie.png"
    selfie.write_bytes(b"\x89PNG\r\n\x1a\nselfie")

    with pytest.raises(RuntimeError, match="returned 3 images"):
        profile_backgrounds.generate_personalized_backgrounds(
            api_key="test",
            selfie_path=str(selfie),
            out_dir=str(tmp_path / "out"),
            count=5,
        )


def test_raises_on_zero_images(tmp_path, monkeypatch):
    fake_client = MagicMock()
    resp = MagicMock()
    resp.parts = []
    fake_client.models.generate_content.return_value = resp
    monkeypatch.setattr(profile_backgrounds.genai, "Client", lambda **kw: fake_client)

    selfie = tmp_path / "selfie.png"
    selfie.write_bytes(b"\x89PNG\r\n\x1a\nselfie")

    with pytest.raises(RuntimeError, match="no images"):
        profile_backgrounds.generate_personalized_backgrounds(
            api_key="test",
            selfie_path=str(selfie),
            out_dir=str(tmp_path / "out"),
            count=5,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_profile_backgrounds.py -v
```
Expected: All fail (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the generator**

Create `backend/app/ml/profile_backgrounds.py`:

```python
"""Gemini-driven personalized background generator.

Takes a selfie, asks Gemini for N talking-head-friendly vertical backgrounds
tuned to the person's coloring and aesthetic. Mirrors the call shape in
``backend/app/ml/frame_relight.py`` (now retired) and
``backend/app/thumbnails/images.py``.
"""
from __future__ import annotations

import os
from typing import List

from google import genai
from google.genai import types

# Current model. A future verified Pro variant can replace this constant.
MODEL_NAME = "gemini-3.1-flash-image-preview"
MODEL_NAME_PRO = MODEL_NAME  # Placeholder; flip when a Pro model is verified.


def build_generation_prompt(count: int) -> str:
    """Compose the Gemini prompt for N personalized backgrounds."""
    return (
        f"Look at this person and generate {count} distinct, professional, "
        "vertical (9:16) background images suitable for a talking-head short "
        "video. Each background should:\n"
        "- Feel like a real photographed environment (not a flat color or "
        "  pure gradient).\n"
        "- Have a clear focal depth that supports a shallow-DOF blur applied "
        "  in post.\n"
        "- Use a color palette that complements the person's skin tone, hair, "
        "  and apparent style.\n"
        "- Span variety: e.g. modern home office, minimalist studio, warm "
        "  cafe, plant-filled corner, industrial loft, soft outdoor.\n"
        "- Contain NO people, NO text, NO logos.\n"
        f"\nReturn exactly {count} images. No commentary."
    )


def generate_personalized_backgrounds(
    api_key: str,
    selfie_path: str,
    out_dir: str,
    count: int = 5,
    use_pro: bool = False,
) -> List[str]:
    """Generate ``count`` background PNGs personalized to the selfie.

    Returns the list of absolute output paths in deterministic 1..count order.
    Raises FileNotFoundError if selfie is missing. Raises RuntimeError if
    Gemini returns fewer than ``count`` images (typically a content-policy
    issue).
    """
    if not os.path.exists(selfie_path):
        raise FileNotFoundError(f"Selfie not found: {selfie_path}")
    os.makedirs(out_dir, exist_ok=True)

    with open(selfie_path, "rb") as f:
        selfie_bytes = f.read()

    client = genai.Client(api_key=api_key)
    model = MODEL_NAME_PRO if use_pro else MODEL_NAME
    prompt = build_generation_prompt(count)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=selfie_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    image_parts = [
        p for p in (response.parts or [])
        if getattr(getattr(p, "inline_data", None), "data", None)
    ]

    if not image_parts:
        raise RuntimeError(
            "Gemini returned no images (likely content-policy refusal)"
        )
    if len(image_parts) < count:
        raise RuntimeError(
            f"Gemini returned {len(image_parts)} images, expected {count}"
        )

    out_paths: List[str] = []
    for i, part in enumerate(image_parts[:count], start=1):
        path = os.path.join(out_dir, f"bg-{i}.png")
        with open(path, "wb") as f:
            f.write(part.inline_data.data)
        out_paths.append(path)
    return out_paths
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_profile_backgrounds.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/profile_backgrounds.py backend/tests/unit/test_profile_backgrounds.py
git commit -m "feat(bg-replace): Gemini personalized background generator (5 per selfie)"
```

---

## Task 3: Source-background clean detection

**Files:**
- Create: `backend/app/ml/bg_detect.py`
- Create: `backend/tests/unit/test_bg_detect.py`
- Create: `backend/tests/fixtures/bg_detect/clean_{black,white,gray}.mp4`
- Create: `backend/tests/fixtures/bg_detect/noisy_{texture,gradient,bookshelf}.mp4`

- [ ] **Step 1: Generate the 6 test fixtures**

Run these inside the container to produce 1-second synthetic clips (≈50KB each):

```bash
mkdir -p backend/tests/fixtures/bg_detect

# Clean fixtures: solid colors
docker exec -w /app openshorts-backend ffmpeg -y -f lavfi \
  -i "color=color=black:size=320x240:rate=10:duration=1" \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/bg_detect/clean_black.mp4

docker exec -w /app openshorts-backend ffmpeg -y -f lavfi \
  -i "color=color=white:size=320x240:rate=10:duration=1" \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/bg_detect/clean_white.mp4

docker exec -w /app openshorts-backend ffmpeg -y -f lavfi \
  -i "color=color=gray:size=320x240:rate=10:duration=1" \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/bg_detect/clean_gray.mp4

# Noisy fixtures: textures / gradients / patterns
docker exec -w /app openshorts-backend ffmpeg -y -f lavfi \
  -i "testsrc=size=320x240:rate=10:duration=1" \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/bg_detect/noisy_texture.mp4

docker exec -w /app openshorts-backend ffmpeg -y -f lavfi \
  -i "gradients=size=320x240:rate=10:duration=1" \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/bg_detect/noisy_gradient.mp4

docker exec -w /app openshorts-backend ffmpeg -y -f lavfi \
  -i "mandelbrot=size=320x240:rate=10:duration=1" \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/bg_detect/noisy_bookshelf.mp4
```

Verify all 6 exist:

```bash
ls -la backend/tests/fixtures/bg_detect/
```

Expected: 6 files, each 10-100KB.

- [ ] **Step 2: Write the failing test**

Create `backend/tests/unit/test_bg_detect.py`:

```python
"""Tests for the source-background clean-detection heuristic."""
from __future__ import annotations

import os

import pytest

from app.ml import bg_detect

FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "bg_detect"
)


@pytest.mark.parametrize("name", ["clean_black", "clean_white", "clean_gray"])
def test_clean_backgrounds_classify_clean(name):
    path = os.path.join(FIXTURES, f"{name}.mp4")
    verdict, score, hex_color = bg_detect.detect_clean_background(path)
    assert verdict == "clean", f"{name}: score={score}, hex={hex_color}"


@pytest.mark.parametrize(
    "name", ["noisy_texture", "noisy_gradient", "noisy_bookshelf"]
)
def test_noisy_backgrounds_classify_noisy(name):
    path = os.path.join(FIXTURES, f"{name}.mp4")
    verdict, score, hex_color = bg_detect.detect_clean_background(path)
    assert verdict == "noisy", f"{name}: score={score}, hex={hex_color}"


def test_returns_hex_color_for_clean_black():
    path = os.path.join(FIXTURES, "clean_black.mp4")
    _, _, hex_color = bg_detect.detect_clean_background(path)
    assert hex_color.lower() in ("#000000", "#010101", "#020202")  # tolerance


def test_missing_video_raises():
    with pytest.raises(FileNotFoundError):
        bg_detect.detect_clean_background("/tmp/does-not-exist.mp4")
```

- [ ] **Step 3: Run test to verify it fails**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_bg_detect.py -v
```
Expected: All fail (`ModuleNotFoundError`).

- [ ] **Step 4: Implement the detector**

Create `backend/app/ml/bg_detect.py`:

```python
"""Source-background clean-detection heuristic.

Reads the first frame of a video and samples four border strips
(top/bottom/left/right) avoiding the centered subject region. Computes
per-strip pixel variance; if the mean variance is below a threshold
the background is "clean" (solid-ish color).

Used by AI Restyle v2 to warn the user (never block) when their source
background is busy.
"""
from __future__ import annotations

import os
from typing import Literal, Tuple

import cv2
import numpy as np

# Threshold tuned against tests/fixtures/bg_detect/. Mean per-channel
# pixel variance across 4 strips; clean fixtures score < 50, noisy > 500.
VARIANCE_THRESHOLD = 150.0

# How wide each border strip is, as a fraction of the smaller dimension.
STRIP_FRACTION = 0.10


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _sample_border_strips(frame: np.ndarray) -> np.ndarray:
    """Return concatenated pixels from 4 border strips, avoiding center."""
    h, w = frame.shape[:2]
    strip = max(8, int(min(h, w) * STRIP_FRACTION))
    top = frame[:strip, :, :]
    bottom = frame[h - strip:, :, :]
    left = frame[:, :strip, :]
    right = frame[:, w - strip:, :]
    return np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3),
         left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )


def detect_clean_background(
    video_path: str,
) -> Tuple[Literal["clean", "noisy"], float, str]:
    """Return (verdict, score, dominant_hex_color) for the first frame.

    Raises FileNotFoundError if the video doesn't exist or has no frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise FileNotFoundError(f"No readable frames in {video_path}")

    # frame is BGR — convert to RGB for hex color reporting.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    border_pixels = _sample_border_strips(frame_rgb)
    score = float(border_pixels.var(axis=0).mean())
    mean_rgb = tuple(int(c) for c in border_pixels.mean(axis=0))
    verdict: Literal["clean", "noisy"] = (
        "clean" if score < VARIANCE_THRESHOLD else "noisy"
    )
    return verdict, score, _rgb_to_hex(mean_rgb)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_bg_detect.py -v
```
Expected: 8 passed. If any clean fixture scores above 150 or any noisy scores below, log the actual scores and adjust `VARIANCE_THRESHOLD` accordingly.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ml/bg_detect.py backend/tests/unit/test_bg_detect.py backend/tests/fixtures/bg_detect/
git commit -m "feat(bg-replace): OpenCV-based clean-background detector"
```

---

## Task 4: fal.ai video matting client

**Files:**
- Create: `backend/app/ml/video_matte.py`
- Create: `backend/tests/unit/test_video_matte.py`

- [ ] **Step 1: Pick the fal.ai matting model**

Open https://fal.ai/models and search for "matting" / "rembg" / "birefnet" / "background removal". Pick ONE endpoint that accepts a video URL and returns an RGBA video URL. If no video-native endpoint exists, fall back to `fal-ai/birefnet/v2` and loop per-frame (slower; cap at 60 frames for 30s @ 2fps preview).

Document the chosen model in a constant. The decision goes in the module docstring (not in this plan because it depends on what's available at impl time).

- [ ] **Step 2: Write the failing test**

Create `backend/tests/unit/test_video_matte.py`:

```python
"""Tests for the fal.ai video matting wrapper."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.ml import video_matte


def test_matte_video_calls_fal_and_writes_output(tmp_path, monkeypatch):
    out_path = tmp_path / "matted.mp4"
    fake_url = "https://v3.fal.media/files/foo/matted.mp4"

    with patch.object(video_matte, "submit_and_poll", return_value={"video": {"url": fake_url}}) as mock_submit, \
         patch.object(video_matte, "upload_file", return_value="https://v3.fal.media/files/foo/src.mp4") as mock_upload, \
         patch.object(video_matte.httpx, "get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"FAKE-MATTED-MP4"
        mock_get.return_value.raise_for_status = lambda: None

        result = video_matte.matte_video(
            api_key="test",
            video_path=str(tmp_path / "source.mp4"),
            out_path=str(out_path),
        )

    assert result == str(out_path)
    assert out_path.read_bytes() == b"FAKE-MATTED-MP4"
    mock_upload.assert_called_once()
    mock_submit.assert_called_once()


def test_matte_video_rejects_non_fal_download_url(tmp_path, monkeypatch):
    with patch.object(video_matte, "submit_and_poll", return_value={"video": {"url": "https://evil.example.com/matted.mp4"}}), \
         patch.object(video_matte, "upload_file", return_value="https://v3.fal.media/files/foo/src.mp4"):
        with pytest.raises(ValueError, match="not a fal.media host"):
            video_matte.matte_video(
                api_key="test",
                video_path=str(tmp_path / "source.mp4"),
                out_path=str(tmp_path / "matted.mp4"),
            )


def test_matte_video_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        video_matte.matte_video(
            api_key="test",
            video_path=str(tmp_path / "nope.mp4"),
            out_path=str(tmp_path / "out.mp4"),
        )
```

- [ ] **Step 3: Run test to verify it fails**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_video_matte.py -v
```
Expected: All fail (`ModuleNotFoundError`).

- [ ] **Step 4: Implement the wrapper**

Create `backend/app/ml/video_matte.py`:

```python
"""fal.ai video matting wrapper.

Submits a source video to fal.ai's matting endpoint and downloads the
returned RGBA video. SSRF guards reuse ``app.integrations.fal``.

Model choice: see MODEL_ID below. Decided during Task 4 step 1; the
constant is here so it can be swapped without touching the pipeline.
"""
from __future__ import annotations

import os

import httpx

from app.integrations.fal import (
    require_fal_download_url,
    submit_and_poll,
    upload_file,
)

# Replace at impl time with the model picked in Task 4 step 1. Examples:
#   "fal-ai/birefnet/v2"            (still images — would loop per-frame)
#   "fal-ai/sam2/video"             (video segmentation — masks per frame)
#   "fal-ai/imageutils/rembg-video" (if released)
MODEL_ID = "fal-ai/birefnet/v2"


def matte_video(api_key: str, video_path: str, out_path: str) -> str:
    """Matte the subject out of ``video_path``; write RGBA video to ``out_path``.

    Returns ``out_path``. Raises FileNotFoundError if input missing,
    ValueError if the returned download URL is not on a fal.media host.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    source_url = upload_file(api_key=api_key, file_path=video_path)

    result = submit_and_poll(
        api_key=api_key,
        model_id=MODEL_ID,
        payload={"video_url": source_url},
    )

    download_url = result.get("video", {}).get("url") or result.get("output_url")
    if not download_url:
        raise RuntimeError(f"fal.ai returned no video URL: {result}")
    require_fal_download_url(download_url)

    resp = httpx.get(download_url, timeout=300.0)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_video_matte.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ml/video_matte.py backend/tests/unit/test_video_matte.py
git commit -m "feat(bg-replace): fal.ai video matting wrapper with SSRF guard"
```

---

## Task 5: FFmpeg composite

**Files:**
- Create: `backend/app/video/composite.py`
- Create: `backend/tests/unit/test_composite.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_composite.py`:

```python
"""Tests for the matte-over-blurred-background composite."""
from __future__ import annotations

import os
import subprocess

import pytest

from app.video import composite


@pytest.fixture
def synthetic_matte(tmp_path):
    """A 1s clip with alpha channel: solid green subject with transparent bg."""
    path = tmp_path / "matte.mov"
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=color=green@1.0:size=320x240:rate=10:duration=1",
        "-vf", "format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(hypot(X-160,Y-120),60),255,0)'",
        "-c:v", "qtrle", str(path),
    ])
    return str(path)


@pytest.fixture
def synthetic_bg(tmp_path):
    """A 320x240 red PNG to use as background."""
    path = tmp_path / "bg.png"
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=color=red:size=320x240:duration=1",
        "-frames:v", "1", str(path),
    ])
    return str(path)


def test_composite_produces_mp4_with_expected_dimensions(synthetic_matte, synthetic_bg, tmp_path):
    out_path = tmp_path / "out.mp4"
    result = composite.composite_subject_over_background(
        matte_video=synthetic_matte,
        background_png=synthetic_bg,
        out_path=str(out_path),
    )
    assert result == str(out_path)
    assert out_path.exists() and out_path.stat().st_size > 0

    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out_path)
    ]).decode().strip()
    assert probe == "320,240"


def test_composite_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        composite.composite_subject_over_background(
            matte_video=str(tmp_path / "nope.mov"),
            background_png=str(tmp_path / "nope.png"),
            out_path=str(tmp_path / "out.mp4"),
        )


def test_composite_applies_default_blur_sigma(synthetic_matte, synthetic_bg, tmp_path):
    out_path = tmp_path / "out.mp4"
    # Should accept the default sigma without raising.
    composite.composite_subject_over_background(
        matte_video=synthetic_matte,
        background_png=synthetic_bg,
        out_path=str(out_path),
    )
    assert out_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_composite.py -v
```
Expected: All fail (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the composite**

Create `backend/app/video/composite.py`:

```python
"""FFmpeg composite: RGBA matte video over a blurred background still.

Filter chain:
  [bg] gblur sigma=N, scale to match matte resolution
  [matte] format rgba, overlay over [bg]
"""
from __future__ import annotations

import os
import subprocess

from app.video.ffmpeg import FFmpegError, probe_resolution

DEFAULT_BLUR_SIGMA = 14.0  # ~ shallow DOF 50mm @ f/2 look.


def composite_subject_over_background(
    matte_video: str,
    background_png: str,
    out_path: str,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
) -> str:
    """Composite RGBA ``matte_video`` over the blurred ``background_png``.

    The background is scaled to the matte's resolution before blur.
    Returns ``out_path``. Raises FileNotFoundError if either input missing.
    """
    if not os.path.exists(matte_video):
        raise FileNotFoundError(f"Matte video not found: {matte_video}")
    if not os.path.exists(background_png):
        raise FileNotFoundError(f"Background PNG not found: {background_png}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    width, height = probe_resolution(matte_video)
    filter_complex = (
        f"[1:v]scale={width}:{height},gblur=sigma={blur_sigma}[bg];"
        f"[0:v]format=rgba[fg];"
        f"[bg][fg]overlay=0:0:format=auto[outv]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", matte_video,
        "-loop", "1", "-i", background_png,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"composite failed: {exc.stderr.decode(errors='replace')[:500]}"
        ) from exc
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_composite.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/video/composite.py backend/tests/unit/test_composite.py
git commit -m "feat(bg-replace): FFmpeg composite (matte over gblur bg)"
```

---

## Task 6: Onboarding HTTP routes

**Files:**
- Modify: `backend/app/routes/ai_restyle.py:1-164` (add new routes + helpers)
- Modify: `backend/app/main.py` (mount `/profiles` static if not via router)
- Create: `backend/tests/api/test_profile_routes.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/api/test_profile_routes.py`:

```python
"""Tests for the onboarding HTTP routes."""
from __future__ import annotations

import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect profile storage into a tmp dir.
    from app.profile import store as profile_store
    monkeypatch.setattr(profile_store, "PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    return TestClient(app)


def _png_bytes() -> bytes:
    # 1x1 PNG signature + minimal IHDR/IDAT/IEND. Enough to pass MIME sniffing.
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
        "890000000A49444154789C6300010000000500010D0A2DB40000000049454E44"
        "AE426082"
    )


def test_post_profile_requires_gemini_key(client):
    r = client.post(
        "/api/restyle/profile",
        files={"selfie": ("selfie.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 401
    assert "X-Gemini-Key" in r.json()["detail"]


def test_post_profile_returns_profile_id(client):
    r = client.post(
        "/api/restyle/profile",
        files={"selfie": ("selfie.png", _png_bytes(), "image/png")},
        headers={"X-Gemini-Key": "test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "profile_id" in body
    assert len(body["profile_id"]) == 36  # UUID4 length


def test_post_profile_rejects_oversize_selfie(client):
    too_big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)  # 11MB
    r = client.post(
        "/api/restyle/profile",
        files={"selfie": ("big.png", too_big, "image/png")},
        headers={"X-Gemini-Key": "test"},
    )
    assert r.status_code == 413


def test_get_profile_returns_status(client):
    r = client.post(
        "/api/restyle/profile",
        files={"selfie": ("selfie.png", _png_bytes(), "image/png")},
        headers={"X-Gemini-Key": "test"},
    )
    pid = r.json()["profile_id"]

    r = client.get(f"/api/restyle/profile/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["profile_id"] == pid
    assert body["generation_status"] in ("pending", "generating", "ready", "failed")
    assert body["selected_idx"] is None
    assert body["generated_count"] >= 0


def test_get_profile_404_when_missing(client):
    r = client.get("/api/restyle/profile/nonexistent")
    assert r.status_code == 404


def test_post_select_sets_selection(client, tmp_path):
    # Seed a profile + 3 generated bgs manually so we don't depend on Gemini.
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    for i in (1, 2, 3):
        profile_store.save_generated(pid, idx=i, png_bytes=b"\x89PNG\r\n\x1a\nbg")
    profile_store.mark_generation_status(pid, "ready")

    r = client.post(f"/api/restyle/profile/{pid}/select", json={"idx": 2})
    assert r.status_code == 200
    assert client.get(f"/api/restyle/profile/{pid}").json()["selected_idx"] == 2


def test_post_select_rejects_out_of_range(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    profile_store.save_generated(pid, idx=1, png_bytes=b"x")
    r = client.post(f"/api/restyle/profile/{pid}/select", json={"idx": 99})
    assert r.status_code == 400


def test_get_profile_static_serve_blocks_traversal(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    r = client.get(f"/profiles/{pid}/../../etc/passwd")
    assert r.status_code in (400, 404)


def test_get_profile_static_serve_blocks_unknown_filenames(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    r = client.get(f"/profiles/{pid}/secret.txt")
    assert r.status_code == 404


def test_get_profile_static_serve_works_for_selfie(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    r = client.get(f"/profiles/{pid}/selfie.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker exec -w /app openshorts-backend pytest tests/api/test_profile_routes.py -v
```
Expected: All fail (route handlers don't exist yet).

- [ ] **Step 3: Add the onboarding routes to `routes/ai_restyle.py`**

Append to `backend/app/routes/ai_restyle.py` (below the existing `restyle_status` function):

```python
import re
from fastapi.responses import FileResponse


MAX_SELFIE_BYTES = 10 * 1024 * 1024  # 10MB
_BG_FILENAME_RE = re.compile(r"^(selfie|bg-[1-9][0-9]?)\.png$")


@router.post("/api/restyle/profile")
async def create_profile_route(
    request: Request,
    background_tasks: BackgroundTasks,
    selfie: UploadFile = File(...),
):
    """Create a new profile + kick off async 5-background generation.

    Returns ``{profile_id}`` immediately; poll ``GET /api/restyle/profile/{id}``
    for the generation status.
    """
    gemini_key = request.headers.get("X-Gemini-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")

    body = await selfie.read()
    if len(body) > MAX_SELFIE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Selfie exceeds {MAX_SELFIE_BYTES // 1024 // 1024}MB cap",
        )
    if not body.startswith(b"\x89PNG\r\n\x1a\n") and not body.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=400, detail="Selfie must be PNG or JPEG")

    from app.profile import store as profile_store
    profile_id = profile_store.create_profile(selfie_bytes=body)

    # Schedule generation; status flips to "generating" → "ready"/"failed".
    from app.ml.profile_backgrounds import generate_personalized_backgrounds

    def _run_generation():
        try:
            profile_store.mark_generation_status(profile_id, "generating")
            out_dir = os.path.join(
                profile_store.PROFILES_ROOT, profile_id
            )
            paths = generate_personalized_backgrounds(
                api_key=gemini_key,
                selfie_path=os.path.join(out_dir, "selfie.png"),
                out_dir=out_dir,
                count=5,
            )
            for i in range(len(paths)):
                # Files already on disk; just bump generated_count.
                profile_store.save_generated(
                    profile_id,
                    idx=i + 1,
                    png_bytes=open(paths[i], "rb").read(),
                )
            profile_store.mark_generation_status(profile_id, "ready")
        except Exception:
            profile_store.mark_generation_status(profile_id, "failed")
            raise

    background_tasks.add_task(_run_generation)
    return {"profile_id": profile_id}


@router.get("/api/restyle/profile/{profile_id}")
async def get_profile_route(profile_id: str):
    """Return current profile state (status, generated count, selected idx)."""
    from app.profile import store as profile_store
    try:
        meta = profile_store.get_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    backgrounds = [
        {"idx": i, "url": f"/profiles/{profile_id}/bg-{i}.png"}
        for i in range(1, meta.get("generated_count", 0) + 1)
    ]
    return {
        "profile_id": profile_id,
        "generation_status": meta.get("generation_status"),
        "generated_count": meta.get("generated_count", 0),
        "selected_idx": meta.get("selected_idx"),
        "backgrounds": backgrounds,
    }


class SelectRequest(BaseModel):
    idx: int = Field(..., ge=1, le=99)


@router.post("/api/restyle/profile/{profile_id}/select")
async def select_background_route(profile_id: str, body: SelectRequest):
    from app.profile import store as profile_store
    try:
        profile_store.set_selected(profile_id, idx=body.idx)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "selected_idx": body.idx}


@router.post("/api/restyle/profile/{profile_id}/regenerate")
async def regenerate_route(
    request: Request,
    background_tasks: BackgroundTasks,
    profile_id: str,
):
    gemini_key = request.headers.get("X-Gemini-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")
    from app.profile import store as profile_store
    try:
        profile_store.get_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")

    from app.ml.profile_backgrounds import generate_personalized_backgrounds
    folder = os.path.join(profile_store.PROFILES_ROOT, profile_id)

    def _run_regeneration():
        try:
            profile_store.mark_generation_status(profile_id, "generating")
            paths = generate_personalized_backgrounds(
                api_key=gemini_key,
                selfie_path=os.path.join(folder, "selfie.png"),
                out_dir=folder,
                count=5,
            )
            for i in range(len(paths)):
                profile_store.save_generated(
                    profile_id,
                    idx=i + 1,
                    png_bytes=open(paths[i], "rb").read(),
                )
            profile_store.mark_generation_status(profile_id, "ready")
        except Exception:
            profile_store.mark_generation_status(profile_id, "failed")
            raise

    background_tasks.add_task(_run_regeneration)
    return {"ok": True, "profile_id": profile_id}


@router.get("/profiles/{profile_id}/{filename}")
async def serve_profile_file(profile_id: str, filename: str):
    """Static-serve selfie.png + bg-N.png. Hard allowlist on filename."""
    if not _BG_FILENAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="Not found")
    if not re.match(r"^[0-9a-f-]{36}$", profile_id):
        raise HTTPException(status_code=400, detail="Bad profile_id")
    from app.profile import store as profile_store
    full_path = os.path.join(profile_store.PROFILES_ROOT, profile_id, filename)
    # Resolve and verify it's still under the profile root (defense in depth).
    real = os.path.realpath(full_path)
    root_real = os.path.realpath(profile_store.PROFILES_ROOT)
    if not real.startswith(root_real + os.sep):
        raise HTTPException(status_code=400, detail="Bad path")
    if not os.path.exists(real):
        raise HTTPException(status_code=404, detail="Not found")
    media = "image/png"
    return FileResponse(real, media_type=media)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/api/test_profile_routes.py -v
```
Expected: 10 passed. If `_run_generation` fires synchronously in `TestClient` and tries to hit real Gemini, monkeypatch `generate_personalized_backgrounds` inside the tests that don't seed manually.

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/ai_restyle.py backend/tests/api/test_profile_routes.py
git commit -m "feat(bg-replace): onboarding routes (profile CRUD + static serve)"
```

---

## Task 7: Pipeline rewrite

**Files:**
- Modify: `backend/app/restyle/pipeline.py` (full rewrite)
- Delete: `backend/tests/unit/test_restyle_pipeline.py` (old)
- Create: `backend/tests/unit/test_restyle_pipeline.py` (new)

- [ ] **Step 1: Delete the old pipeline test**

```bash
rm backend/tests/unit/test_restyle_pipeline.py
```

- [ ] **Step 2: Write the new failing test**

Create `backend/tests/unit/test_restyle_pipeline.py`:

```python
"""Tests for the AI Restyle v2 background-replacement pipeline."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from app.restyle import pipeline


@pytest.fixture
def fake_video(tmp_path):
    import subprocess
    path = tmp_path / "src.mp4"
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=color=black:size=320x240:rate=10:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ])
    return str(path)


@pytest.fixture
def fake_profile(tmp_path, monkeypatch):
    from app.profile import store as profile_store
    monkeypatch.setattr(profile_store, "PROFILES_ROOT", str(tmp_path / "profiles"))
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\nselfie")
    for i in (1, 2):
        profile_store.save_generated(pid, idx=i, png_bytes=b"\x89PNG\r\n\x1a\nbg")
    profile_store.set_selected(pid, idx=1)
    return pid


def test_happy_path_runs_all_steps(fake_video, fake_profile, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    jobs = {}

    with patch.object(pipeline, "detect_clean_background", return_value=("clean", 5.0, "#000000")), \
         patch.object(pipeline, "matte_video", side_effect=lambda **kw: open(kw["out_path"], "wb").write(b"matte") or kw["out_path"]), \
         patch.object(pipeline, "composite_subject_over_background", side_effect=lambda **kw: open(kw["out_path"], "wb").write(b"composite") or kw["out_path"]), \
         patch.object(pipeline, "mux_video_audio", side_effect=lambda *a, **k: open(a[2], "wb").write(b"muxed") or a[2]):
        asyncio.run(pipeline.run_restyle_job(
            jobs=jobs,
            job_id="abc",
            input_path=fake_video,
            profile_id=fake_profile,
            fal_key="fal-test",
        ))

    job = jobs["abc"]
    assert job["status"] == "completed"
    assert job["progress_pct"] == 100
    assert job["result"]["video_url"].endswith(".mp4")
    assert any("Clean source background" in line for line in job["logs"])


def test_noisy_bg_warns_but_continues(fake_video, fake_profile, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    jobs = {}

    with patch.object(pipeline, "detect_clean_background", return_value=("noisy", 850.0, "#a1b2c3")), \
         patch.object(pipeline, "matte_video", side_effect=lambda **kw: open(kw["out_path"], "wb").write(b"x") or kw["out_path"]), \
         patch.object(pipeline, "composite_subject_over_background", side_effect=lambda **kw: open(kw["out_path"], "wb").write(b"x") or kw["out_path"]), \
         patch.object(pipeline, "mux_video_audio", side_effect=lambda *a, **k: open(a[2], "wb").write(b"x") or a[2]):
        asyncio.run(pipeline.run_restyle_job(
            jobs=jobs,
            job_id="warn",
            input_path=fake_video,
            profile_id=fake_profile,
            fal_key="fal-test",
        ))

    assert jobs["warn"]["status"] == "completed"
    assert any("may not be clean" in line for line in jobs["warn"]["logs"])


def test_matting_failure_flips_status(fake_video, fake_profile, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    jobs = {}

    with patch.object(pipeline, "detect_clean_background", return_value=("clean", 5.0, "#000000")), \
         patch.object(pipeline, "matte_video", side_effect=RuntimeError("fal exploded")):
        asyncio.run(pipeline.run_restyle_job(
            jobs=jobs,
            job_id="fail",
            input_path=fake_video,
            profile_id=fake_profile,
            fal_key="fal-test",
        ))

    assert jobs["fail"]["status"] == "failed"
    assert any("fal exploded" in line for line in jobs["fail"]["logs"])


def test_oversize_duration_rejected(fake_profile, tmp_path, monkeypatch):
    import subprocess
    long_video = tmp_path / "long.mp4"
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=color=black:size=320x240:rate=10:duration=35",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(long_video),
    ])
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    jobs = {}
    asyncio.run(pipeline.run_restyle_job(
        jobs=jobs,
        job_id="too_long",
        input_path=str(long_video),
        profile_id=fake_profile,
        fal_key="fal-test",
    ))
    assert jobs["too_long"]["status"] == "failed"
    assert any("exceeds the 30s cap" in line for line in jobs["too_long"]["logs"])
```

- [ ] **Step 3: Run test to verify it fails**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_restyle_pipeline.py -v
```
Expected: All fail or error (old pipeline has different signature).

- [ ] **Step 4: Replace `restyle/pipeline.py`**

Overwrite `backend/app/restyle/pipeline.py`:

```python
"""AI Restyle v2 pipeline: background replacement.

6-step async flow:
  1. Probe duration; reject if > MAX_DURATION_SEC
  2. Detect background cleanliness (warn-only)
  3. Matte subject via fal.ai
  4. Composite over user's selected (blurred) background
  5. Mux original audio back
  6. Persist result + mark completed
"""
from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any, Dict, Optional

from app.ml.bg_detect import detect_clean_background
from app.ml.video_matte import matte_video
from app.profile import store as profile_store
from app.video.composite import composite_subject_over_background
from app.video.ffmpeg import mux_video_audio, probe_duration


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
MAX_DURATION_SEC = 30.0


def _ensure_job(jobs: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    if job_id not in jobs:
        jobs[job_id] = {
            "status": "processing",
            "logs": [],
            "progress_pct": 0,
            "result": None,
        }
    return jobs[job_id]


def _log(jobs: Dict[str, Any], job_id: str, line: str, pct: Optional[int] = None) -> None:
    job = _ensure_job(jobs, job_id)
    job["logs"].append(line)
    if pct is not None:
        job["progress_pct"] = pct


async def run_restyle_job(
    jobs: Dict[str, Any],
    job_id: str,
    input_path: str,
    profile_id: str,
    fal_key: str,
) -> None:
    """Drive the v2 background-replacement pipeline. Mutates jobs[job_id]
    in place; never raises."""
    _ensure_job(jobs, job_id)
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    loop = asyncio.get_event_loop()

    try:
        _log(jobs, job_id, "🔎 Probing duration…", pct=5)
        duration = await loop.run_in_executor(None, partial(probe_duration, input_path))
        if duration > MAX_DURATION_SEC:
            raise ValueError(
                f"Video duration {duration:.1f}s exceeds the {MAX_DURATION_SEC:.0f}s cap"
            )

        _log(jobs, job_id, "🪟 Checking background cleanliness…", pct=15)
        verdict, score, hex_color = await loop.run_in_executor(
            None, partial(detect_clean_background, input_path)
        )
        if verdict == "clean":
            _log(jobs, job_id, f"✅ Clean source background ({hex_color}, score {score:.1f})")
        else:
            _log(jobs, job_id, f"⚠️ Background may not be clean (score {score:.1f}); results may vary")

        _log(jobs, job_id, "✂️ Matting subject (fal.ai)…", pct=30)
        matted = os.path.join(output_dir, f"{base}_matted.mov")
        await loop.run_in_executor(
            None,
            partial(matte_video, api_key=fal_key, video_path=input_path, out_path=matted),
        )

        _log(jobs, job_id, "🎨 Compositing over your selected background…", pct=70)
        selected_bg = os.path.join(output_dir, "selected_bg.png")
        with open(selected_bg, "wb") as f:
            f.write(profile_store.get_selected_background_bytes(profile_id))
        composited = os.path.join(output_dir, f"{base}_composited.mp4")
        await loop.run_in_executor(
            None,
            partial(
                composite_subject_over_background,
                matte_video=matted,
                background_png=selected_bg,
                out_path=composited,
            ),
        )

        _log(jobs, job_id, "🔊 Muxing original audio…", pct=90)
        final_out = os.path.join(output_dir, f"restyled_{os.path.basename(input_path)}")
        await loop.run_in_executor(
            None, partial(mux_video_audio, composited, input_path, final_out)
        )

        job = jobs[job_id]
        job["result"] = {
            "video_url": f"/videos/{job_id}/{os.path.basename(final_out)}",
            "original_url": f"/videos/{job_id}/{os.path.basename(input_path)}",
            "profile_id": profile_id,
            "duration_sec": duration,
            "bg_verdict": verdict,
        }
        job["status"] = "completed"
        job["progress_pct"] = 100
        _log(jobs, job_id, "✅ AI Restyle complete.")

    except Exception as exc:
        job = _ensure_job(jobs, job_id)
        job["status"] = "failed"
        job["logs"].append(f"❌ {exc}")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/unit/test_restyle_pipeline.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/restyle/pipeline.py backend/tests/unit/test_restyle_pipeline.py
git commit -m "feat(bg-replace): pipeline rewrite (matte + composite + mux)"
```

---

## Task 8: Update the POST /api/restyle handler

**Files:**
- Modify: `backend/app/routes/ai_restyle.py:43-148` (the existing `start_restyle` handler)
- Modify: `backend/tests/api/test_ai_restyle.py` (existing tests for POST)

- [ ] **Step 1: Update the existing API tests**

Open `backend/tests/api/test_ai_restyle.py` and replace any test that posts `background_prompt`+`lighting_prompt` with the new `profile_id` form. Add new tests:

```python
def test_post_restyle_requires_profile_id(client_with_fal_key):
    # ... existing setup ...
    r = client_with_fal_key.post(
        "/api/restyle",
        files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp4", "video/mp4")},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert r.status_code == 422  # missing required form field


def test_post_restyle_404s_on_unknown_profile(client_with_fal_key):
    r = client_with_fal_key.post(
        "/api/restyle",
        files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp4", "video/mp4")},
        data={"profile_id": "00000000-0000-0000-0000-000000000000"},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert r.status_code == 404


def test_post_restyle_400_when_profile_has_no_selection(client_with_fal_key):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n")
    profile_store.save_generated(pid, idx=1, png_bytes=b"x")
    # NOTE: no set_selected called.
    r = client_with_fal_key.post(
        "/api/restyle",
        files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp4", "video/mp4")},
        data={"profile_id": pid},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert r.status_code == 400
    assert "no selected background" in r.json()["detail"]
```

Delete any tests that exercise the `background_prompt` / `lighting_prompt` paths (they're gone).

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec -w /app openshorts-backend pytest tests/api/test_ai_restyle.py -v
```
Expected: New tests fail; old tests fail or are deleted.

- [ ] **Step 3: Modify `start_restyle` in `routes/ai_restyle.py`**

Replace the existing `start_restyle` function signature + body. Key changes:
- Remove `background_prompt` and `lighting_prompt` form fields.
- Add `profile_id: str = Form(...)`.
- Validate the profile exists + has a selected background before persisting the upload.
- Pass `profile_id` (not prompts) to the pipeline.

Replacement function:

```python
@router.post("/api/restyle")
async def start_restyle(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    profile_id: str = Form(...),
):
    """Start a background-replacement restyle job. Requires a profile with
    a selected background. Returns ``{job_id}`` immediately."""
    from app.main import jobs, _ensure_video_upload, OUTPUT_DIR, UPLOAD_DIR
    from app.profile import store as profile_store

    gemini_key = request.headers.get("X-Gemini-Key")
    fal_key = request.headers.get("X-Fal-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")
    if not fal_key:
        raise HTTPException(status_code=401, detail="X-Fal-Key header required")

    # Profile validation up-front (before any disk I/O).
    try:
        meta = profile_store.get_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    if meta.get("selected_idx") is None:
        raise HTTPException(
            status_code=400,
            detail="Profile has no selected background; pick one in Settings",
        )

    limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB",
                )
        except ValueError:
            pass

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    safe_name = os.path.basename(file.filename or f"{job_id}.mp4")
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")

    first_chunk = await file.read(_CHUNK)
    if not first_chunk:
        shutil.rmtree(job_output_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        _ensure_video_upload(safe_name, first_chunk)
    except HTTPException:
        shutil.rmtree(job_output_dir, ignore_errors=True)
        raise

    size = len(first_chunk)
    with open(input_path, "wb") as buf:
        buf.write(first_chunk)
        while chunk := await file.read(_CHUNK):
            size += len(chunk)
            if size > limit_bytes:
                buf.close()
                os.remove(input_path)
                shutil.rmtree(job_output_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB",
                )
            buf.write(chunk)

    jobs[job_id] = {
        "status": "processing",
        "logs": [f"📥 Received {safe_name} ({size / 1024 / 1024:.1f} MB)"],
        "progress_pct": 0,
        "result": None,
        "product": "ai-restyle-v2",
    }

    from app.restyle.pipeline import run_restyle_job
    background_tasks.add_task(
        run_restyle_job,
        jobs=jobs,
        job_id=job_id,
        input_path=input_path,
        profile_id=profile_id,
        fal_key=fal_key,
    )

    return {"job_id": job_id}
```

Also remove the `MAX_PROMPT_LEN` constant at the top of `routes/ai_restyle.py` (no prompts anymore).

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec -w /app openshorts-backend pytest tests/api/test_ai_restyle.py -v
```
Expected: All updated tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/ai_restyle.py backend/tests/api/test_ai_restyle.py
git commit -m "feat(bg-replace): POST /api/restyle takes profile_id instead of prompts"
```

---

## Task 9: Delete dead v1 modules

**Files:**
- Delete: `backend/app/ml/frame_relight.py`
- Delete: `backend/app/ml/video_restyle.py`
- Delete: `backend/tests/unit/test_frame_relight.py`
- Delete: `backend/tests/unit/test_video_restyle.py`
- Delete: `backend/tests/unit/test_frame_extract.py` (if it only tested relight prep; keep otherwise)

- [ ] **Step 1: Verify no remaining imports**

```bash
docker exec -w /app openshorts-backend grep -rn "frame_relight\|video_restyle\|from app.ml.frame_relight\|from app.ml.video_restyle" --include="*.py" .
```
Expected: No matches outside the files being deleted.

- [ ] **Step 2: Delete the files**

```bash
rm backend/app/ml/frame_relight.py
rm backend/app/ml/video_restyle.py
rm backend/tests/unit/test_frame_relight.py
rm backend/tests/unit/test_video_restyle.py
```

Keep `backend/app/ml/frame_extract.py` — it's still used implicitly by `bg_detect.py` (no, actually `bg_detect` uses cv2 directly; double-check). If `frame_extract.py` has no remaining callers, delete it and its test too:

```bash
docker exec -w /app openshorts-backend grep -rn "frame_extract\|from app.ml.frame_extract\|extract_first_frame" --include="*.py" .
```

- [ ] **Step 3: Run the full pytest suite**

```bash
docker exec -w /app openshorts-backend pytest -m "not e2e" -q
```
Expected: All tests pass (count slightly lower than before deletes).

- [ ] **Step 4: Commit**

```bash
git add -A backend/app/ml/ backend/tests/unit/
git commit -m "chore(bg-replace): delete v1 relight + v2v modules"
```

---

## Task 10: Frontend — profileStore + BackgroundProfileSection

**Files:**
- Create: `frontend/src/state/profileStore.js`
- Create: `frontend/src/pages/Settings/sections/BackgroundProfileSection.jsx`
- Modify: `frontend/src/App.jsx` (wire new Settings child route)
- Modify: `frontend/src/pages/Settings/sections/ApiKeysSection.jsx` (copy update)

- [ ] **Step 1: Implement `profileStore.js`**

Create `frontend/src/state/profileStore.js`:

```javascript
// Per-browser onboarding profile: holds the opaque profile_id in
// localStorage; thin wrappers around the backend onboarding routes.
import { useEffect, useState } from 'react';
import { getApiUrl } from '../config.js';

const KEY = 'aiRestyleProfileId_v1';
const EVENT = 'aiRestyleProfileChanged';

export function getProfileId() {
  return localStorage.getItem(KEY);
}

export function setProfileId(id) {
  if (id) localStorage.setItem(KEY, id);
  else localStorage.removeItem(KEY);
  window.dispatchEvent(new CustomEvent(EVENT));
}

export function useProfileId() {
  const [id, setId] = useState(getProfileId());
  useEffect(() => {
    const handler = () => setId(getProfileId());
    window.addEventListener(EVENT, handler);
    window.addEventListener('storage', handler);
    return () => {
      window.removeEventListener(EVENT, handler);
      window.removeEventListener('storage', handler);
    };
  }, []);
  return id;
}

export async function createProfile(selfieFile, geminiKey) {
  const fd = new FormData();
  fd.append('selfie', selfieFile);
  const res = await fetch(getApiUrl('/api/restyle/profile'), {
    method: 'POST',
    headers: { 'X-Gemini-Key': geminiKey },
    body: fd,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const { profile_id } = await res.json();
  setProfileId(profile_id);
  return profile_id;
}

export async function fetchProfile(profileId) {
  const res = await fetch(getApiUrl(`/api/restyle/profile/${profileId}`));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function selectBackground(profileId, idx) {
  const res = await fetch(getApiUrl(`/api/restyle/profile/${profileId}/select`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ idx }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function regenerate(profileId, geminiKey) {
  const res = await fetch(getApiUrl(`/api/restyle/profile/${profileId}/regenerate`), {
    method: 'POST',
    headers: { 'X-Gemini-Key': geminiKey },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}
```

- [ ] **Step 2: Implement `BackgroundProfileSection.jsx`**

Create `frontend/src/pages/Settings/sections/BackgroundProfileSection.jsx`:

```jsx
// AI Restyle background profile: one-time selfie onboarding → 5
// Gemini-generated backgrounds → pick one.
import { useEffect, useRef, useState } from 'react';
import { useKeys } from '../../../state/keysStore.js';
import {
  createProfile,
  fetchProfile,
  selectBackground,
  regenerate,
  setProfileId,
  useProfileId,
} from '../../../state/profileStore.js';
import { getApiUrl } from '../../../config.js';
import SectionHeader from './SectionHeader.jsx';

export default function BackgroundProfileSection() {
  const profileId = useProfileId();
  const keys = useKeys();
  const fileRef = useRef(null);
  const [profile, setProfile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!profileId) { setProfile(null); return; }
    let alive = true;
    const tick = async () => {
      try {
        const p = await fetchProfile(profileId);
        if (!alive) return;
        setProfile(p);
        if (p.generation_status === 'generating') setTimeout(tick, 2000);
      } catch {
        if (alive) setProfile(null);
      }
    };
    tick();
    return () => { alive = false; };
  }, [profileId]);

  async function onUpload(e) {
    setError(null);
    const f = e.target.files?.[0];
    if (!f) return;
    if (!keys.gemini) { setError('Set your Gemini key first.'); return; }
    setBusy(true);
    try {
      await createProfile(f, keys.gemini);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function onPick(idx) {
    await selectBackground(profileId, idx);
    setProfile(await fetchProfile(profileId));
  }

  async function onRegenerate() {
    if (!keys.gemini) { setError('Set your Gemini key first.'); return; }
    setBusy(true);
    try {
      await regenerate(profileId, keys.gemini);
      setProfile(await fetchProfile(profileId));
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  function onReplace() {
    setProfileId(null);
    setProfile(null);
  }

  return (
    <div>
      <SectionHeader
        title="Background Profile"
        subtitle="Upload a selfie once. We generate 5 personalized backgrounds for your shorts; pick one and it sticks."
      />

      {!profileId && (
        <div className="rounded-lg border border-dashed border-border p-6 text-center">
          <p className="text-[13px] text-zinc-400 mb-3">No profile yet.</p>
          <button
            onClick={() => fileRef.current?.click()}
            disabled={busy || !keys.gemini}
            className="px-4 py-2 text-[13px] bg-primary text-white rounded-md disabled:opacity-40"
          >
            {busy ? 'Uploading…' : 'Upload selfie & generate backgrounds'}
          </button>
          <input ref={fileRef} type="file" accept="image/png,image/jpeg" hidden onChange={onUpload} />
          {!keys.gemini && <p className="mt-2 text-[11px] text-yellow-400">Set your Gemini key first.</p>}
        </div>
      )}

      {profileId && profile?.generation_status === 'generating' && (
        <div className="rounded-lg border border-border p-6 text-center">
          <div className="animate-pulse text-[13px] text-zinc-400">
            Generating 5 backgrounds… (~20-40s)
          </div>
        </div>
      )}

      {profileId && profile?.generation_status === 'failed' && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-4">
          <p className="text-[13px] text-red-400 mb-3">Generation failed. Try a different selfie.</p>
          <button onClick={onReplace} className="px-3 py-1.5 text-[12px] border border-border rounded-md">
            Replace selfie
          </button>
        </div>
      )}

      {profileId && profile?.generation_status === 'ready' && (
        <>
          <div className="grid grid-cols-5 gap-3">
            {profile.backgrounds.map((bg) => (
              <button
                key={bg.idx}
                onClick={() => onPick(bg.idx)}
                className={`relative rounded-md overflow-hidden border-2 transition-colors ${
                  profile.selected_idx === bg.idx ? 'border-primary' : 'border-border hover:border-zinc-600'
                }`}
              >
                <img src={getApiUrl(bg.url)} alt={`Background ${bg.idx}`} className="w-full aspect-[9/16] object-cover" />
                {profile.selected_idx === bg.idx && (
                  <div className="absolute top-1 right-1 bg-primary text-white text-[10px] px-1.5 py-0.5 rounded">★</div>
                )}
              </button>
            ))}
          </div>
          <div className="mt-4 flex gap-2">
            <button onClick={onRegenerate} disabled={busy} className="px-3 py-1.5 text-[12px] border border-border rounded-md disabled:opacity-40">
              {busy ? 'Regenerating…' : 'Regenerate 5'}
            </button>
            <button onClick={onReplace} className="px-3 py-1.5 text-[12px] border border-border rounded-md">
              Replace selfie
            </button>
          </div>
        </>
      )}

      {error && <div className="mt-3 text-[12px] text-red-400" role="alert">{error}</div>}
    </div>
  );
}
```

- [ ] **Step 3: Wire the new Settings route**

Open `frontend/src/App.jsx`. Find the Settings route block that currently includes `AIRestylePresetsSection`. Replace the import + route element:

```javascript
// Replace this line:
import AIRestylePresetsSection from './pages/Settings/sections/AIRestylePresetsSection.jsx';
// With:
import BackgroundProfileSection from './pages/Settings/sections/BackgroundProfileSection.jsx';

// And replace the child route element:
// <Route path="general/ai-restyle" element={<AIRestylePresetsSection />} />
// becomes:
<Route path="general/background-profile" element={<BackgroundProfileSection />} />
```

If there's a navigation sidebar in Settings (`frontend/src/pages/Settings/index.jsx` or similar), update the entry label from "AI Restyle Presets" to "Background Profile" and the link target to `/settings/general/background-profile`.

- [ ] **Step 4: Update `ApiKeysSection.jsx` copy**

In `frontend/src/pages/Settings/sections/ApiKeysSection.jsx`, find the fal.ai panel and update the description from "v2v restyle" → "subject matting". The "Required for AI Restyle" badge stays.

- [ ] **Step 5: Build the frontend**

```bash
cd frontend && npm run build
```
Expected: clean build, no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/state/profileStore.js \
        frontend/src/pages/Settings/sections/BackgroundProfileSection.jsx \
        frontend/src/App.jsx \
        frontend/src/pages/Settings/sections/ApiKeysSection.jsx
# Update navigation index too if it exists:
git add frontend/src/pages/Settings/index.jsx 2>/dev/null || true
git commit -m "feat(bg-replace): Settings → Background Profile section"
```

---

## Task 11: Frontend — Wizard reshape (Precheck step)

**Files:**
- Create: `frontend/src/pages/AIRestyle/steps/Precheck.jsx`
- Modify: `frontend/src/pages/AIRestyle/Wizard.jsx`
- Modify: `frontend/src/pages/AIRestyle/steps/Upload.jsx`
- Modify: `frontend/src/pages/AIRestyle/steps/Review.jsx`
- Delete: `frontend/src/pages/AIRestyle/steps/Configure.jsx`
- Delete: `frontend/src/pages/Settings/sections/AIRestylePresetsSection.jsx`
- Delete: `frontend/src/state/aiRestylePresets.js`

- [ ] **Step 1: Create the Precheck step**

Create `frontend/src/pages/AIRestyle/steps/Precheck.jsx`:

```jsx
// AI Restyle Step 2: confirm the user's selected profile background and
// submit the video to /api/restyle. (Background-clean detection is server-
// side; the warning appears in the Review step's log feed if relevant.)

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useKeys } from '../../../state/keysStore.js';
import { useProfileId, fetchProfile } from '../../../state/profileStore.js';
import { getApiUrl } from '../../../config.js';

export default function Precheck({ wizard }) {
  const profileId = useProfileId();
  const keys = useKeys();
  const [profile, setProfile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!profileId) return;
    fetchProfile(profileId).then(setProfile).catch(() => setProfile(null));
  }, [profileId]);

  if (!profileId) {
    return (
      <div className="p-8 max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Set up your background profile</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          AI Restyle needs a selfie-based background profile before it can run.
        </p>
        <Link
          to="/settings/general/background-profile"
          className="inline-block px-4 py-2 text-[13px] bg-primary text-white rounded-md"
        >
          Go to Settings → Background Profile →
        </Link>
      </div>
    );
  }

  if (!profile) {
    return <div className="p-8 text-[13px] text-zinc-400">Loading profile…</div>;
  }

  if (profile.selected_idx == null) {
    return (
      <div className="p-8 max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Pick a background first</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          Your profile has {profile.generated_count} generated backgrounds but none is selected.
        </p>
        <Link
          to="/settings/general/background-profile"
          className="inline-block px-4 py-2 text-[13px] bg-primary text-white rounded-md"
        >
          Pick one →
        </Link>
      </div>
    );
  }

  const selectedBg = profile.backgrounds.find((b) => b.idx === profile.selected_idx);

  async function start() {
    setError(null);
    if (!keys.fal) { setError('Set your fal.ai key in Settings.'); return; }
    const fd = new FormData();
    fd.append('file', wizard.data.file.file);
    fd.append('profile_id', profileId);
    setSubmitting(true);
    try {
      const res = await fetch(getApiUrl('/api/restyle'), {
        method: 'POST',
        headers: { 'X-Gemini-Key': keys.gemini, 'X-Fal-Key': keys.fal },
        body: fd,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const { job_id } = await res.json();
      wizard.setData({
        job: { jobId: job_id, status: 'processing', result: null, progress_pct: 0, logs: [] },
      });
      wizard.next();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-[24px] font-semibold mb-2">Confirm & restyle</h1>
      <p className="text-[13px] text-zinc-400 mb-6">
        Your selected background will be used. The server will check if your source background is
        clean enough; a warning shows in the next step if not (but the job still runs).
      </p>

      <div className="rounded-lg border border-border p-4 mb-6">
        <div className="text-[11px] uppercase tracking-wider text-zinc-500 mb-2">Selected background</div>
        <img src={getApiUrl(selectedBg.url)} alt="" className="w-32 aspect-[9/16] object-cover rounded-md" />
      </div>

      {error && <div className="mb-3 text-[12px] text-red-400" role="alert">{error}</div>}

      <div className="flex items-center justify-between">
        <button onClick={wizard.back} className="px-4 py-2 text-[13px] text-zinc-400">← Back</button>
        <button
          onClick={start}
          disabled={submitting || !keys.fal}
          className="px-4 py-2 text-[13px] bg-primary text-white rounded-md disabled:opacity-40"
        >
          {submitting ? 'Starting…' : 'Start restyle →'}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire `Precheck` into the Wizard**

Open `frontend/src/pages/AIRestyle/Wizard.jsx`. Replace the import and step-2 rendering:

```javascript
// Replace:
import Configure from './steps/Configure.jsx';
// With:
import Precheck from './steps/Precheck.jsx';

// In the step renderer (likely a switch on wizard.step):
// case 'configure': return <Configure wizard={wizard} />;
// becomes:
case 'precheck': return <Precheck wizard={wizard} />;
```

Update the wizard state machine in `frontend/src/hooks/useWizard.js` (if needed) so step 2 is `'precheck'` not `'configure'`. Also remove any reference to `selection.backgroundPresetId` / `selection.lightingPresetId` in the wizard data shape.

- [ ] **Step 3: Update `Review.jsx`**

In `frontend/src/pages/AIRestyle/steps/Review.jsx`, replace any reference to the old prompt fields with a thumbnail of the user's selected background (read from `profileStore.fetchProfile` once on mount). If the existing Review just polls and shows PhoneFrame Before/After, the only required change is to surface the `bg_verdict` field from the job result when set to `"noisy"`.

- [ ] **Step 4: Delete dead frontend files**

```bash
rm frontend/src/pages/AIRestyle/steps/Configure.jsx
rm frontend/src/pages/Settings/sections/AIRestylePresetsSection.jsx
rm frontend/src/state/aiRestylePresets.js
```

- [ ] **Step 5: Build the frontend**

```bash
cd frontend && npm run build
```
Expected: clean build, no missing-import errors.

- [ ] **Step 6: Commit**

```bash
git add -A frontend/src/pages/AIRestyle/ frontend/src/state/ frontend/src/pages/Settings/sections/
git commit -m "feat(bg-replace): wizard reshape (Precheck step, drop relight UI)"
```

---

## Task 12: OpenAPI snapshot + ROADMAP + CLAUDE.md regen

**Files:**
- Modify: `backend/tests/snapshots/baseline.openapi.json`
- Modify: `ROADMAP.md`
- (auto-managed): `~/.claude/CLAUDE.md`

- [ ] **Step 1: Regenerate the OpenAPI snapshot**

```bash
rm backend/tests/snapshots/baseline.openapi.json
docker exec -w /app openshorts-backend pytest tests/api/test_openapi_contract.py -v
```
Expected: First run fails because snapshot is missing; the test scaffolding writes the new snapshot. Re-run:

```bash
docker exec -w /app openshorts-backend pytest tests/api/test_openapi_contract.py -v
```
Expected: pass.

- [ ] **Step 2: Update ROADMAP.md**

Open `ROADMAP.md`. Find the AI Restyle section (top of file). Replace the v1 shipped block with:

```markdown
## AI Restyle

**v1 (relight + fal.ai v2v) — Retired 2026-05-21.**
Shipped briefly in PR #35 as a research vehicle for fal.ai integration. Superseded by v2 below.

**v2 (background replacement) — Shipped.**
One-time selfie onboarding generates 5 personalized Gemini backgrounds; per-video,
the source subject is matted via fal.ai and composited over the user's selected
(blurred) background with realistic shallow-DOF feel.

Pipeline: probe → bg-clean detect (warn-only) → fal matting → composite → mux audio.
30s duration cap. 250MB file cap. 10MB selfie cap.
```

- [ ] **Step 3: Run the CLAUDE.md auto-updater**

```bash
python3 scripts/update_claude_md.py
```
Expected: regenerates `~/.claude/CLAUDE.md` MODULE-MAP with the new modules (`profile/store.py`, `ml/profile_backgrounds.py`, `ml/bg_detect.py`, `ml/video_matte.py`, `video/composite.py`) and drops the deleted ones.

- [ ] **Step 4: Run the full test suite one more time**

```bash
docker exec -w /app openshorts-backend pytest -m "not e2e" -q
cd frontend && npm run build
```
Expected: both clean.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/snapshots/baseline.openapi.json ROADMAP.md
# CLAUDE.md regeneration is automatic via the pre-commit hook.
git commit -m "docs(bg-replace): refresh OpenAPI snapshot + ROADMAP entry"
```

- [ ] **Step 6: Push + open PR**

```bash
git push mine feat/ai-restyle-v2-bg-replace
gh pr create --repo mutonby/openshorts --base main \
  --head vansteenbergenmatisse:feat/ai-restyle-v2-bg-replace \
  --title "feat(ai-restyle): v2 background replacement" \
  --body-file - <<'EOF'
## Summary

Pivots AI Restyle from v1 (Nano-Banana relight + fal.ai v2v, PR #35) to v2:
**personalized background replacement** with one-time selfie onboarding.

- Selfie → Gemini generates 5 backgrounds tuned to the user
- Source video → fal.ai matting → composite over selected (blurred) background
- 6-step async pipeline; preserves audio; 30s duration cap

## Test plan

- [ ] `pytest -m "not e2e"` green in Docker
- [ ] `npm run build` clean
- [ ] Real onboarding works (`POST /api/restyle/profile` with a selfie, polling shows generation → ready)
- [ ] Real per-video restyle works against a 5s test clip with clean black background
- [ ] Selecting a different background and re-running produces a different composite

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

---

## Plan Self-Review

- **Spec coverage:** Every section of the design (§3 user flow, §4 backend modules, §5 pipeline steps, §6 frontend reshape, §7 reasonable-call defaults, §8 error handling, §9 security baseline, §10 testing, §11 migration) maps to at least one task. The 5-route security baseline (§9) is enforced inline in Task 6 routes (auth header check, oversize selfie 413, path-traversal guard, UUID format check, filename allowlist).
- **Placeholder scan:** No `TBD`/`TODO` left. Task 4 step 1 is the one "go research" beat, but it's an explicit step with a fallback model (`fal-ai/birefnet/v2`) so the engineer always has something to wire.
- **Type consistency:** `profile_store.save_generated(profile_id, idx, png_bytes)` matches its callsites in Tasks 1, 6, and 7. `composite_subject_over_background(matte_video, background_png, out_path, blur_sigma)` matches its call in Task 7. `matte_video(api_key, video_path, out_path)` matches its call in Task 7.
- **Scope check:** 12 tasks, ~30 commits total, single product pivot, internally cohesive. Fits one plan.
