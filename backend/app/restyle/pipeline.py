"""AI Restyle v2 pipeline: background replacement.

6-step async flow:
  1. Probe duration; reject if > MAX_DURATION_SEC
  2. Detect background cleanliness (warn-only, never blocks)
  3. Matte subject via fal.ai
  4. Composite over user's selected (blurred) background
  5. Mux original audio back
  6. Persist result + mark completed
"""
from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any, Dict, Optional

from app.ml.bg_detect import detect_clean_background
from app.ml.video_matte import matte_video
from app.profile import store as profile_store
from app.video.composite import composite_subject_over_background
from app.video.ffmpeg import mux_video_audio, probe_duration


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
MAX_DURATION_SEC = 30.0


def _ensure_job(jobs: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    if job_id not in jobs:
        jobs[job_id] = {
            "status": "processing",
            "logs": [],
            "progress_pct": 0,
            "result": None,
        }
    return jobs[job_id]


def _log(jobs: Dict[str, Any], job_id: str, line: str, pct: Optional[int] = None) -> None:
    job = _ensure_job(jobs, job_id)
    job["logs"].append(line)
    if pct is not None:
        job["progress_pct"] = pct


async def run_restyle_job(
    jobs: Dict[str, Any],
    job_id: str,
    input_path: str,
    profile_id: str,
    fal_key: str,
) -> None:
    """Drive the v2 background-replacement pipeline. Mutates jobs[job_id]
    in place; never raises."""
    _ensure_job(jobs, job_id)
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    loop = asyncio.get_event_loop()

    try:
        _log(jobs, job_id, "🔎 Probing duration…", pct=5)
        duration = await loop.run_in_executor(None, partial(probe_duration, input_path))
        if duration > MAX_DURATION_SEC:
            raise ValueError(
                f"Video duration {duration:.1f}s exceeds the {MAX_DURATION_SEC:.0f}s cap"
            )

        _log(jobs, job_id, "🪟 Checking background cleanliness…", pct=15)
        verdict, score, hex_color = await loop.run_in_executor(
            None, partial(detect_clean_background, input_path)
        )
        if verdict == "clean":
            _log(jobs, job_id, f"✅ Clean source background ({hex_color}, score {score:.1f})")
        else:
            _log(jobs, job_id, f"⚠️ Background may not be clean (score {score:.1f}); results may vary")

        _log(jobs, job_id, "✂️ Matting subject (fal.ai)…", pct=30)
        matted = os.path.join(output_dir, f"{base}_matted.mov")
        await loop.run_in_executor(
            None,
            partial(matte_video, api_key=fal_key, video_path=input_path, out_path=matted),
        )

        _log(jobs, job_id, "🎨 Compositing over your selected background…", pct=70)
        selected_bg = os.path.join(output_dir, "selected_bg.png")
        with open(selected_bg, "wb") as f:
            f.write(profile_store.get_selected_background_bytes(profile_id))
        composited = os.path.join(output_dir, f"{base}_composited.mp4")
        await loop.run_in_executor(
            None,
            partial(
                composite_subject_over_background,
                matte_video=matted,
                background_png=selected_bg,
                out_path=composited,
            ),
        )

        _log(jobs, job_id, "🔊 Muxing original audio…", pct=90)
        final_out = os.path.join(output_dir, f"restyled_{os.path.basename(input_path)}")
        await loop.run_in_executor(
            None, partial(mux_video_audio, composited, input_path, final_out)
        )

        job = jobs[job_id]
        job["result"] = {
            "video_url": f"/videos/{job_id}/{os.path.basename(final_out)}",
            "original_url": f"/videos/{job_id}/{os.path.basename(input_path)}",
            "profile_id": profile_id,
            "duration_sec": duration,
            "bg_verdict": verdict,
        }
        job["status"] = "completed"
        job["progress_pct"] = 100
        _log(jobs, job_id, "✅ AI Restyle complete.")

    except Exception as exc:
        job = _ensure_job(jobs, job_id)
        job["status"] = "failed"
        job["logs"].append(f"❌ {exc}")
