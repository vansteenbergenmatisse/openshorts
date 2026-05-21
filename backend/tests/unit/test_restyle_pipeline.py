"""Tests for the AI Restyle v2 background-replacement pipeline."""
from __future__ import annotations

import asyncio
import os
import subprocess
from unittest.mock import patch

import pytest

from app.restyle import pipeline


@pytest.fixture
def fake_video(tmp_path):
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
         patch.object(pipeline, "matte_video", side_effect=lambda **kw: (open(kw["out_path"], "wb").write(b"matte"), kw["out_path"])[1]), \
         patch.object(pipeline, "composite_subject_over_background", side_effect=lambda **kw: (open(kw["out_path"], "wb").write(b"composite"), kw["out_path"])[1]), \
         patch.object(pipeline, "mux_video_audio", side_effect=lambda *a, **k: (open(a[2], "wb").write(b"muxed"), a[2])[1]):
        asyncio.run(pipeline.run_restyle_job(
            jobs=jobs,
            job_id="abc",
            input_path=fake_video,
            profile_id=fake_profile,
            fal_key="fal-test",
        ))

    job = jobs["abc"]
    assert job["status"] == "completed", f"logs: {job['logs']}"
    assert job["progress_pct"] == 100
    assert job["result"]["video_url"].endswith(".mp4")
    assert any("Clean source background" in line for line in job["logs"])


def test_noisy_bg_warns_but_continues(fake_video, fake_profile, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    jobs = {}

    with patch.object(pipeline, "detect_clean_background", return_value=("noisy", 850.0, "#a1b2c3")), \
         patch.object(pipeline, "matte_video", side_effect=lambda **kw: (open(kw["out_path"], "wb").write(b"x"), kw["out_path"])[1]), \
         patch.object(pipeline, "composite_subject_over_background", side_effect=lambda **kw: (open(kw["out_path"], "wb").write(b"x"), kw["out_path"])[1]), \
         patch.object(pipeline, "mux_video_audio", side_effect=lambda *a, **k: (open(a[2], "wb").write(b"x"), a[2])[1]):
        asyncio.run(pipeline.run_restyle_job(
            jobs=jobs,
            job_id="warn",
            input_path=fake_video,
            profile_id=fake_profile,
            fal_key="fal-test",
        ))

    assert jobs["warn"]["status"] == "completed", f"logs: {jobs['warn']['logs']}"
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
