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
