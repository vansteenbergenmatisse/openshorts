"""FFmpeg composite: RGBA matte video over a blurred background still."""
from __future__ import annotations

import os

from app.video.ffmpeg import probe_duration, probe_resolution, run

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
    duration = probe_duration(matte_video)
    filter_complex = (
        f"[1:v]scale={width}:{height},gblur=sigma={blur_sigma}[bg];"
        f"[0:v]format=rgba[fg];"
        f"[bg][fg]overlay=0:0:format=auto:eof_action=endall[outv]"
    )

    cmd = [
        "-y",
        "-i", matte_video,
        "-loop", "1", "-i", background_png,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-t", str(duration),
        out_path,
    ]
    run(cmd)
    return out_path
