"""AI Restyle FastAPI router.

Endpoints:
- POST /api/restyle          start a restyle job
- GET  /api/restyle/{job_id} poll status

Job state is the same in-memory dict as the rest of main.py (``jobs``);
imports from ``app.main`` are deferred to avoid a circular import at
module load time. The pipeline itself lives in ``app.restyle.pipeline``.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


router = APIRouter()


# AI Restyle caps at 30s of video; 250MB is generous (8.3MB/s for 30s,
# ~67 Mbps which is well above streaming-quality bitrates). Tighter than
# main.py's 2GB cap on /api/process because a 30s clip never approaches
# multi-GB unless it's pathological. (Codex HIGH-3 — upload-before-reject
# disk-DoS hardening.) Overrideable for testing / specialty deployments
# via the AI_RESTYLE_MAX_FILE_SIZE_MB env var.
MAX_FILE_SIZE_MB = int(os.environ.get("AI_RESTYLE_MAX_FILE_SIZE_MB", "250"))
_CHUNK = 1024 * 1024


class RestyleStatus(BaseModel):
    status: str
    logs: list[str]
    progress_pct: int = Field(default=0, ge=0, le=100)
    result: Optional[dict] = None


@router.post("/api/restyle")
async def start_restyle(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    profile_id: str = Form(...),
):
    """Start a background-replacement restyle job. Requires a profile with
    a selected background. Returns {job_id} immediately; poll
    GET /api/restyle/{job_id} for status."""
    from app.main import jobs, _ensure_video_upload, OUTPUT_DIR, UPLOAD_DIR
    from app.profile import store as profile_store

    # C1 auth — kept for codebase-wide consistency; fal_key is the only one
    # the v2 pipeline actually uses, but gemini_key is the standard auth gate
    # across all mutating routes.
    gemini_key = request.headers.get("X-Gemini-Key")
    fal_key = request.headers.get("X-Fal-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")
    if not fal_key:
        raise HTTPException(status_code=401, detail="X-Fal-Key header required")

    # Profile validation up-front (after auth, before any disk I/O).
    try:
        meta = profile_store.get_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    if meta.get("selected_idx") is None:
        raise HTTPException(
            status_code=400,
            detail="Profile has no selected background; pick one in Settings",
        )

    limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    # Content-Length preflight: reject before allocating any disk space.
    # (Codex HIGH-3 — disk-DoS via upload-before-duration-check.) Clients
    # can lie, but the streaming check below is the backstop.
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB",
                )
        except ValueError:
            pass  # malformed header — let the streaming check do its job

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    safe_name = os.path.basename(file.filename or f"{job_id}.mp4")
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")

    # Read first chunk, validate signature before persisting anything.
    first_chunk = await file.read(_CHUNK)
    if not first_chunk:
        shutil.rmtree(job_output_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        _ensure_video_upload(safe_name, first_chunk)
    except HTTPException:
        shutil.rmtree(job_output_dir, ignore_errors=True)
        raise

    size = len(first_chunk)
    with open(input_path, "wb") as buf:
        buf.write(first_chunk)
        while chunk := await file.read(_CHUNK):
            size += len(chunk)
            if size > limit_bytes:
                buf.close()
                os.remove(input_path)
                shutil.rmtree(job_output_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB",
                )
            buf.write(chunk)

    jobs[job_id] = {
        "status": "processing",
        "logs": [f"📥 Received {safe_name} ({size / 1024 / 1024:.1f} MB)"],
        "progress_pct": 0,
        "result": None,
        "product": "ai-restyle-v2",
    }

    # Schedule the async pipeline. FastAPI BackgroundTasks awaits async
    # callables natively, so no asyncio.create_task wrapper needed.
    from app.restyle.pipeline import run_restyle_job
    background_tasks.add_task(
        run_restyle_job,
        jobs=jobs,
        job_id=job_id,
        input_path=input_path,
        profile_id=profile_id,
        fal_key=fal_key,
    )

    return {"job_id": job_id}


@router.get("/api/restyle/{job_id}", response_model=RestyleStatus)
async def restyle_status(job_id: str):
    """Poll the status of a restyle job."""
    from app.main import jobs
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return RestyleStatus(
        status=job["status"],
        logs=job.get("logs", []),
        progress_pct=job.get("progress_pct", 0),
        result=job.get("result"),
    )


# ---------------------------------------------------------------------------
# Onboarding profile routes (Task 6)
# ---------------------------------------------------------------------------

MAX_SELFIE_BYTES = 10 * 1024 * 1024  # 10MB
_BG_FILENAME_RE = re.compile(r"^(selfie|bg-[1-9][0-9]?)\.png$")
_PROFILE_ID_RE = re.compile(r"^[0-9a-f-]{36}$")


def _run_background_generation(profile_id: str, gemini_key: str) -> None:
    """Shared background-task body for create_profile + regenerate routes.

    Calls Gemini to generate 5 personalized backgrounds for the saved selfie,
    persists each via the profile store, and updates generation_status. Any
    exception flips status to "failed" and re-raises. Clears prior bg-*.png
    files before regenerating so a fresh generation never inherits stale state.
    """
    from app.profile import store as profile_store
    from app.ml import profile_backgrounds
    try:
        profile_store.mark_generation_status(profile_id, "generating")
        out_dir = os.path.join(profile_store.PROFILES_ROOT, profile_id)
        # Clear stale bg-*.png files (regenerate path) so generated_count
        # and selected_idx don't end up referencing dead files.
        for name in os.listdir(out_dir):
            if name.startswith("bg-") and name.endswith(".png"):
                os.remove(os.path.join(out_dir, name))
        meta = profile_store.get_profile(profile_id)
        meta["generated_count"] = 0
        meta["selected_idx"] = None
        profile_store._atomic_write_json(profile_store._meta_path(profile_id), meta)
        # Now generate fresh batch
        paths = profile_backgrounds.generate_personalized_backgrounds(
            api_key=gemini_key,
            selfie_path=os.path.join(out_dir, "selfie.png"),
            out_dir=out_dir,
            count=5,
        )
        for i, path in enumerate(paths, start=1):
            with open(path, "rb") as f:
                profile_store.save_generated(profile_id, idx=i, png_bytes=f.read())
        profile_store.mark_generation_status(profile_id, "ready")
    except Exception:
        profile_store.mark_generation_status(profile_id, "failed")
        raise


@router.post("/api/restyle/profile")
async def create_profile_route(
    request: Request,
    background_tasks: BackgroundTasks,
    selfie: UploadFile = File(...),
):
    """Create profile + kick off async 5-background generation. Returns {profile_id}."""
    gemini_key = request.headers.get("X-Gemini-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")

    # Streaming read with size cap: 413 as soon as the cap is exceeded;
    # never buffer more than 10MB in memory.
    chunks = []
    total = 0
    first_chunk = await selfie.read(_CHUNK)
    if not first_chunk:
        raise HTTPException(status_code=400, detail="Uploaded selfie is empty")
    # MIME magic-byte sniff on the first chunk.
    if not first_chunk.startswith(b"\x89PNG\r\n\x1a\n") and not first_chunk.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=400, detail="Selfie must be PNG or JPEG")
    chunks.append(first_chunk)
    total = len(first_chunk)
    while chunk := await selfie.read(_CHUNK):
        total += len(chunk)
        if total > MAX_SELFIE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Selfie exceeds {MAX_SELFIE_BYTES // 1024 // 1024}MB cap",
            )
        chunks.append(chunk)
    body = b"".join(chunks)

    from app.profile import store as profile_store
    profile_id = profile_store.create_profile(selfie_bytes=body)

    background_tasks.add_task(_run_background_generation, profile_id, gemini_key)
    return {"profile_id": profile_id}


@router.get("/api/restyle/profile/{profile_id}")
async def get_profile_route(profile_id: str):
    """Return current profile state (status, generated count, selected idx, bg URLs)."""
    from app.profile import store as profile_store
    try:
        meta = profile_store.get_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    backgrounds = [
        {"idx": i, "url": f"/profiles/{profile_id}/bg-{i}.png"}
        for i in range(1, meta.get("generated_count", 0) + 1)
    ]
    return {
        "profile_id": profile_id,
        "generation_status": meta.get("generation_status"),
        "generated_count": meta.get("generated_count", 0),
        "selected_idx": meta.get("selected_idx"),
        "backgrounds": backgrounds,
    }


class SelectRequest(BaseModel):
    idx: int = Field(..., ge=1, le=99)


@router.post("/api/restyle/profile/{profile_id}/select")
async def select_background_route(request: Request, profile_id: str, body: SelectRequest):
    """Mark one of the generated backgrounds as active. 400 if idx out of range."""
    # X-Gemini-Key required for codebase-wide consistency with other mutating
    # routes; the key itself is not validated downstream (this route only
    # writes metadata, no LLM call). The profile_id UUID is the capability.
    gemini_key = request.headers.get("X-Gemini-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")
    from app.profile import store as profile_store
    try:
        profile_store.set_selected(profile_id, idx=body.idx)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "selected_idx": body.idx}


@router.post("/api/restyle/profile/{profile_id}/regenerate")
async def regenerate_route(
    request: Request,
    background_tasks: BackgroundTasks,
    profile_id: str,
):
    """Re-run the 5-background generation against the saved selfie."""
    gemini_key = request.headers.get("X-Gemini-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")
    from app.profile import store as profile_store
    try:
        profile_store.get_profile(profile_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")

    background_tasks.add_task(_run_background_generation, profile_id, gemini_key)
    return {"ok": True, "profile_id": profile_id}


@router.get("/profiles/{profile_id}/{filename}")
async def serve_profile_file(profile_id: str, filename: str):
    """Static-serve selfie.png + bg-N.png with allowlist + path traversal guard.

    No X-Gemini-Key required: the profile_id UUID (128-bit random) is the
    capability. Consistent with GET /api/restyle/{job_id} (also unauthenticated;
    job_id is the credential).
    """
    if not _BG_FILENAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="Not found")
    if not _PROFILE_ID_RE.match(profile_id):
        raise HTTPException(status_code=400, detail="Bad profile_id")
    from app.profile import store as profile_store
    full_path = os.path.join(profile_store.PROFILES_ROOT, profile_id, filename)
    real = os.path.realpath(full_path)
    root_real = os.path.realpath(profile_store.PROFILES_ROOT)
    if not real.startswith(root_real + os.sep):
        raise HTTPException(status_code=400, detail="Bad path")
    if not os.path.exists(real):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(real, media_type="image/png")
