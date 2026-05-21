"""E2E smoke test: chain all 5 AI Restyle v2 routes end-to-end.

POST /api/restyle/profile → GET /api/restyle/profile/{id} (poll until ready) →
POST /api/restyle/profile/{id}/select → POST /api/restyle → GET /api/restyle/{job_id}
(poll until completed).

Mocks only the external boundaries: Gemini image generation (writes real PNG
bytes), fal.ai matting, and the heavy FFmpeg ops in the pipeline (probe_duration,
bg_detect, composite, mux). Everything in between — routes, profile_store,
BackgroundTasks, state machine — runs for real. The point is to catch
cross-route contract drift (response shapes, header propagation, state
transitions, profile_id flow) that per-module mocks would miss.
"""
from __future__ import annotations

import asyncio
import io
import os
import time
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.e2e


def _png_bytes() -> bytes:
    """Minimum valid 1x1 RGBA PNG — accepted by routes' magic-byte sniff."""
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
        "890000000A49444154789C6300010000000500010D0A2DB40000000049454E44"
        "AE426082"
    )


def _mp4_bytes(size: int = 256) -> bytes:
    """Minimum-viable MP4 header to pass _ensure_video_upload's ftyp check."""
    head = b"\x00\x00\x00\x18ftypisom"
    return head + b"\x00" * max(0, size - len(head))


@pytest.fixture
def e2e_client(tmp_path, monkeypatch):
    """TestClient with PROFILES_ROOT + OUTPUT_DIR + UPLOAD_DIR all isolated."""
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "profiles").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)

    from app.profile import store as profile_store
    monkeypatch.setattr(profile_store, "PROFILES_ROOT", str(tmp_path / "profiles"))

    # Stub Gemini: write 5 real PNG bytes per call. The profile_id is captured
    # from the surrounding directory so we use the call's out_dir.
    def fake_gen(api_key, selfie_path, out_dir, count=5, use_pro=False):
        paths = []
        for i in range(1, count + 1):
            p = os.path.join(out_dir, f"bg-{i}.png")
            with open(p, "wb") as f:
                f.write(_png_bytes())
            paths.append(p)
        return paths

    monkeypatch.setattr(
        "app.ml.profile_backgrounds.generate_personalized_backgrounds",
        fake_gen,
    )

    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def _poll_until(fn, predicate, *, timeout=5.0, interval=0.05):
    """Poll fn() until predicate(result) is True or timeout. Returns last result."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = fn()
        if predicate(last):
            return last
        time.sleep(interval)
    return last


def test_full_onboarding_to_restyle_chain(e2e_client):
    """The golden path: onboard, pick a bg, restyle a clip, poll done."""
    from app.restyle import pipeline

    # ── Route 1: POST /api/restyle/profile ────────────────────────────────
    r = e2e_client.post(
        "/api/restyle/profile",
        files={"selfie": ("me.png", _png_bytes(), "image/png")},
        headers={"X-Gemini-Key": "g-secret"},
    )
    assert r.status_code == 200, r.text
    profile_id = r.json()["profile_id"]
    assert len(profile_id) == 36, "profile_id should be a UUID4 string"

    # ── Route 2: GET /api/restyle/profile/{id} — poll until ready ────────
    # FastAPI's TestClient runs BackgroundTasks synchronously after the
    # response returns, so by the time the POST resolves the bg-gen task
    # has already finished. The poll still exercises the GET contract.
    def _get():
        rr = e2e_client.get(f"/api/restyle/profile/{profile_id}")
        assert rr.status_code == 200
        return rr.json()

    state = _poll_until(_get, lambda s: s["generation_status"] == "ready")
    assert state["generation_status"] == "ready", state
    assert state["generated_count"] == 5
    assert state["selected_idx"] is None
    assert len(state["backgrounds"]) == 5
    assert all(b["url"].startswith(f"/profiles/{profile_id}/bg-") for b in state["backgrounds"])

    # ── Route 3: POST /api/restyle/profile/{id}/select ────────────────────
    r = e2e_client.post(
        f"/api/restyle/profile/{profile_id}/select",
        json={"idx": 2},
        headers={"X-Gemini-Key": "g-secret"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["selected_idx"] == 2

    # Confirm selection persisted via GET.
    state = e2e_client.get(f"/api/restyle/profile/{profile_id}").json()
    assert state["selected_idx"] == 2

    # ── Route 4: POST /api/restyle ────────────────────────────────────────
    # Mock the heavy pipeline ops — they're exhaustively unit-tested already;
    # E2E value is in route chaining + state transitions.
    with patch.object(pipeline, "probe_duration", return_value=5.0), \
         patch.object(pipeline, "detect_clean_background", return_value=("clean", 10.0, "#000000")), \
         patch.object(
             pipeline, "matte_video",
             side_effect=lambda **kw: (open(kw["out_path"], "wb").write(b"matte"), kw["out_path"])[1],
         ), \
         patch.object(
             pipeline, "composite_subject_over_background",
             side_effect=lambda **kw: (open(kw["out_path"], "wb").write(b"composite"), kw["out_path"])[1],
         ), \
         patch.object(
             pipeline, "mux_video_audio",
             side_effect=lambda *a, **k: (open(a[2], "wb").write(b"muxed"), a[2])[1],
         ):
        r = e2e_client.post(
            "/api/restyle",
            files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
            data={"profile_id": profile_id},
            headers={"X-Gemini-Key": "g-secret", "X-Fal-Key": "f-secret"},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        assert len(job_id) == 36

        # ── Route 5: GET /api/restyle/{job_id} — poll until completed ───
        # BackgroundTasks runs the async pipeline after the POST returns;
        # under TestClient that's still synchronous, so completion is fast.
        def _status():
            rr = e2e_client.get(f"/api/restyle/{job_id}")
            assert rr.status_code == 200
            return rr.json()

        final = _poll_until(_status, lambda s: s["status"] in ("completed", "failed"))

    assert final["status"] == "completed", f"pipeline failed: {final['logs']}"
    assert final["progress_pct"] == 100
    assert final["result"]["video_url"].startswith(f"/videos/{job_id}/")
    assert final["result"]["profile_id"] == profile_id
    assert final["result"]["bg_verdict"] == "clean"
    # Pipeline emits the four canonical milestone logs even with all ops mocked.
    log_blob = " ".join(final["logs"])
    assert "Probing duration" in log_blob
    assert "Matting subject" in log_blob
    assert "Compositing over your selected background" in log_blob
    assert "Muxing original audio" in log_blob


def test_chain_rejects_restyle_with_no_selection(e2e_client):
    """Route 4 must 400 if onboarding ran but the user never picked a bg."""
    r = e2e_client.post(
        "/api/restyle/profile",
        files={"selfie": ("me.png", _png_bytes(), "image/png")},
        headers={"X-Gemini-Key": "g"},
    )
    profile_id = r.json()["profile_id"]

    # Skip the /select step entirely. /api/restyle must reject.
    r = e2e_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        data={"profile_id": profile_id},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert r.status_code == 400, r.text
    assert "no selected background" in r.json()["detail"]


def test_regenerate_resets_state_then_continues(e2e_client):
    """Regeneration must clear selected_idx + replenish 5 fresh bgs."""
    # Initial onboarding
    r = e2e_client.post(
        "/api/restyle/profile",
        files={"selfie": ("me.png", _png_bytes(), "image/png")},
        headers={"X-Gemini-Key": "g"},
    )
    profile_id = r.json()["profile_id"]

    # Wait for first generation to land, then pick one.
    state = _poll_until(
        lambda: e2e_client.get(f"/api/restyle/profile/{profile_id}").json(),
        lambda s: s["generation_status"] == "ready",
    )
    assert state["generated_count"] == 5

    e2e_client.post(
        f"/api/restyle/profile/{profile_id}/select",
        json={"idx": 3},
        headers={"X-Gemini-Key": "g"},
    )

    # Regenerate
    r = e2e_client.post(
        f"/api/restyle/profile/{profile_id}/regenerate",
        headers={"X-Gemini-Key": "g"},
    )
    assert r.status_code == 200

    # After regenerate, selected_idx must be cleared and 5 fresh bgs present.
    state = _poll_until(
        lambda: e2e_client.get(f"/api/restyle/profile/{profile_id}").json(),
        lambda s: s["generation_status"] == "ready" and s["selected_idx"] is None,
    )
    assert state["selected_idx"] is None, "regenerate must clear stale selection"
    assert state["generated_count"] == 5, "regenerate must replenish to 5 bgs"
