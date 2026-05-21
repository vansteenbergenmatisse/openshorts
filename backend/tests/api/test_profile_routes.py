"""Tests for the onboarding HTTP routes."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
        "890000000A49444154789C6300010000000500010D0A2DB40000000049454E44"
        "AE426082"
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.profile import store as profile_store
    monkeypatch.setattr(profile_store, "PROFILES_ROOT", str(tmp_path / "profiles"))
    # Stub Gemini generation so background tasks don't try a real API call.
    def fake_gen(api_key, selfie_path, out_dir, count=5, use_pro=False):
        paths = []
        for i in range(1, count + 1):
            p = os.path.join(out_dir, f"bg-{i}.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\nfake-bg-%d" % i)
            paths.append(p)
        return paths
    monkeypatch.setattr(
        "app.ml.profile_backgrounds.generate_personalized_backgrounds",
        fake_gen,
    )
    return TestClient(app)


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
    assert len(body["profile_id"]) == 36


def test_post_profile_rejects_oversize_selfie(client):
    too_big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
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
    assert body["generated_count"] >= 0


def test_get_profile_404_when_missing(client):
    r = client.get("/api/restyle/profile/nonexistent")
    assert r.status_code == 404


def test_post_select_sets_selection(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    for i in (1, 2, 3):
        profile_store.save_generated(pid, idx=i, png_bytes=b"\x89PNG\r\n\x1a\nbg")
    profile_store.mark_generation_status(pid, "ready")

    r = client.post(
        f"/api/restyle/profile/{pid}/select",
        json={"idx": 2},
        headers={"X-Gemini-Key": "test"},
    )
    assert r.status_code == 200
    assert client.get(f"/api/restyle/profile/{pid}").json()["selected_idx"] == 2


def test_post_select_rejects_out_of_range(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    profile_store.save_generated(pid, idx=1, png_bytes=b"x")
    r = client.post(
        f"/api/restyle/profile/{pid}/select",
        json={"idx": 99},
        headers={"X-Gemini-Key": "test"},
    )
    assert r.status_code == 400


def test_post_select_requires_gemini_key(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    profile_store.save_generated(pid, idx=1, png_bytes=b"x")
    r = client.post(f"/api/restyle/profile/{pid}/select", json={"idx": 1})
    assert r.status_code == 401
    assert "X-Gemini-Key" in r.json()["detail"]


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


def test_post_regenerate_requires_gemini_key(client):
    from app.profile import store as profile_store
    pid = profile_store.create_profile(selfie_bytes=_png_bytes())
    r = client.post(f"/api/restyle/profile/{pid}/regenerate")
    assert r.status_code == 401
