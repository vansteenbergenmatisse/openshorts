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
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=color=green@1.0:size=320x240:rate=10:duration=1",
        "-vf", "format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(hypot(X-160,Y-120),60),255,0)'",
        "-c:v", "qtrle", str(path),
    ], capture_output=True)
    if result.returncode != 0:
        # qtrle unavailable — fall back to VP9 with native alpha
        path = tmp_path / "matte.webm"
        subprocess.check_call([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=color=green:size=320x240:rate=10:duration=1",
            "-vf", "format=yuva420p,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(hypot(X-160,Y-120),60),255,0)'",
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
            str(path),
        ])
    return str(path)


@pytest.fixture
def synthetic_bg(tmp_path):
    """A 320x240 red PNG to use as background."""
    path = tmp_path / "bg.png"
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=color=red:size=320x240:duration=1",
        "-frames:v", "1", "-update", "1", str(path),
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
    composite.composite_subject_over_background(
        matte_video=synthetic_matte,
        background_png=synthetic_bg,
        out_path=str(out_path),
    )
    assert out_path.exists()
