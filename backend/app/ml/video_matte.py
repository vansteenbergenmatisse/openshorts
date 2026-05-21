"""fal.ai video matting wrapper.

Submits a source video to fal.ai's matting endpoint and downloads the
returned RGBA video. SSRF guards reuse ``app.integrations.fal``.

Model choice: see MODEL_ID below. ``fal-ai/birefnet/v2`` is the fallback;
swap after the Phase 0 spike confirms the best available video-matting
endpoint.
"""
from __future__ import annotations

import os

import httpx

from app.integrations.fal import (
    require_fal_download_url,
    submit_and_poll,
    upload_file,
)

MODEL_ID = "fal-ai/birefnet/v2"


def matte_video(api_key: str, video_path: str, out_path: str) -> str:
    """Matte the subject out of ``video_path``; write RGBA video to ``out_path``.

    Returns ``out_path``. Raises FileNotFoundError if input missing.
    Raises ``app.integrations.fal.FalError`` if the returned download URL is
    not on a fal-controlled host (SSRF guard).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    source_url = upload_file(file_path=video_path, fal_key=api_key)

    result = submit_and_poll(
        model_id=MODEL_ID,
        input_data={"video_url": source_url},
        fal_key=api_key,
    )

    download_url = result.get("video", {}).get("url") or result.get("output_url")
    if not download_url:
        raise RuntimeError(f"fal.ai returned no video URL: {result}")

    # SSRF guard: reject any URL not on a fal-controlled host.
    require_fal_download_url(download_url)

    resp = httpx.get(download_url, timeout=300.0)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path
