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
