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
    assert hex_color.lower() in ("#000000", "#010101", "#020202")


def test_missing_video_raises():
    with pytest.raises(FileNotFoundError):
        bg_detect.detect_clean_background("/tmp/does-not-exist.mp4")
