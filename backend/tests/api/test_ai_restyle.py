"""Contract tests for /api/restyle and /api/restyle/{job_id}."""
from __future__ import annotations

import io

import pytest


@pytest.fixture
def restyle_client(tmp_path, monkeypatch):
    """TestClient against the production FastAPI app, isolated to tmp_path."""
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def seeded_profile(tmp_path, monkeypatch):
    """A profile with one selected background, ready to use as profile_id."""
    from app.profile import store as profile_store
    monkeypatch.setattr(profile_store, "PROFILES_ROOT", str(tmp_path / "profiles"))
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n")
    profile_store.save_generated(pid, idx=1, png_bytes=b"x")
    profile_store.set_selected(pid, idx=1)
    return pid


def _mp4_bytes(size: int = 256) -> bytes:
    """Minimum-viable MP4 header to pass _ensure_video_upload's ftyp check."""
    head = b"\x00\x00\x00\x18ftypisom"  # 12 bytes; ftyp at offset 4
    return head + b"\x00" * max(0, size - len(head))


def test_post_restyle_requires_gemini_key(restyle_client, seeded_profile):
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        data={"profile_id": seeded_profile},
        headers={"X-Fal-Key": "fake"},
    )
    assert res.status_code == 401
    assert "Gemini" in res.json()["detail"]


def test_post_restyle_requires_fal_key(restyle_client, seeded_profile):
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        data={"profile_id": seeded_profile},
        headers={"X-Gemini-Key": "fake"},
    )
    assert res.status_code == 401
    assert "Fal" in res.json()["detail"]


def test_post_restyle_rejects_non_mp4_upload(restyle_client, seeded_profile):
    """Non-MP4 bytes must be rejected by _ensure_video_upload (ftyp check)."""
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 256), "video/mp4")},
        data={"profile_id": seeded_profile},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert res.status_code == 415


def test_post_restyle_happy_path_enqueues_job(restyle_client, seeded_profile, monkeypatch):
    """Happy path: returns job_id, seeds jobs[id], schedules background task.

    Mocks ``run_restyle_job`` so the test doesn't actually call Gemini / fal.
    """
    captured = {}

    async def fake_run(jobs, job_id, **kwargs):
        captured["job_id"] = job_id
        captured["kwargs"] = kwargs
        # Don't mutate; the route should already have seeded jobs[id].

    monkeypatch.setattr("app.restyle.pipeline.run_restyle_job", fake_run)

    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        data={"profile_id": seeded_profile},
        headers={"X-Gemini-Key": "g-secret", "X-Fal-Key": "f-secret"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "job_id" in body
    job_id = body["job_id"]

    from app.main import jobs
    assert job_id in jobs
    assert jobs[job_id]["status"] == "processing"
    assert jobs[job_id]["product"] == "ai-restyle-v2"
    assert "logs" in jobs[job_id]
    assert jobs[job_id]["progress_pct"] == 0


def test_post_restyle_rejects_oversize_content_length(restyle_client, seeded_profile, monkeypatch):
    """Codex HIGH-3: bodies declared larger than the AI Restyle cap must
    be rejected via Content-Length preflight BEFORE the file is written
    to disk (otherwise a 2GB upload could disk-DoS the host even if the
    30s duration check would later reject it)."""
    monkeypatch.setattr("app.routes.ai_restyle.MAX_FILE_SIZE_MB", 1)
    big_body = b"\x00\x00\x00\x18ftypisom" + b"\x00" * (2 * 1024 * 1024)
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(big_body), "video/mp4")},
        data={"profile_id": seeded_profile},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert res.status_code == 413


def test_get_restyle_status_not_found(restyle_client):
    res = restyle_client.get("/api/restyle/nonexistent")
    assert res.status_code == 404


def test_get_restyle_status_returns_seeded_job(restyle_client):
    """GET surfaces status/logs/progress_pct/result from the in-memory dict."""
    from app.main import jobs
    jobs["seeded-job-id"] = {
        "status": "completed",
        "logs": ["seeded line"],
        "progress_pct": 100,
        "result": {"video_url": "/videos/seeded-job-id/out.mp4"},
        "product": "ai-restyle-v2",
    }
    try:
        res = restyle_client.get("/api/restyle/seeded-job-id")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "completed"
        assert body["logs"] == ["seeded line"]
        assert body["progress_pct"] == 100
        assert body["result"] == {"video_url": "/videos/seeded-job-id/out.mp4"}
    finally:
        jobs.pop("seeded-job-id", None)


# ---------------------------------------------------------------------------
# New v2 profile-validation tests
# ---------------------------------------------------------------------------

def test_post_restyle_requires_profile_id(restyle_client):
    """Missing profile_id form field → 422 (FastAPI validation)."""
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp4\x00\x00\x00\x00mp42mp41" + b"\x00" * 32, "video/mp4")},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert res.status_code == 422, res.text


def test_post_restyle_404s_on_unknown_profile(restyle_client):
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp4\x00\x00\x00\x00mp42mp41" + b"\x00" * 32, "video/mp4")},
        data={"profile_id": "00000000-0000-0000-0000-000000000000"},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert res.status_code == 404, res.text


def test_post_restyle_400_when_profile_has_no_selection(tmp_path, monkeypatch):
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    from fastapi.testclient import TestClient
    from app.main import app
    from app.profile import store as profile_store
    monkeypatch.setattr(profile_store, "PROFILES_ROOT", str(tmp_path / "profiles"))
    pid = profile_store.create_profile(selfie_bytes=b"\x89PNG\r\n\x1a\n")
    profile_store.save_generated(pid, idx=1, png_bytes=b"x")
    # NOTE: no set_selected called — profile has no active background.
    client = TestClient(app)
    res = client.post(
        "/api/restyle",
        files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp4\x00\x00\x00\x00mp42mp41" + b"\x00" * 32, "video/mp4")},
        data={"profile_id": pid},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert res.status_code == 400, res.text
    assert "no selected background" in res.json()["detail"]
