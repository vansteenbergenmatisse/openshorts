# AI Restyle v2: Background Replacement — Design

**Status:** Brainstorm-approved, awaiting user review.
**Date:** 2026-05-21
**Supersedes:** `2026-05-20-ai-restyle-design.md` (the Nano-Banana-relight + fal.ai-v2v pipeline ships in PR #35 and is then retired in favor of this spec).

---

## 1. Goal

Pivot the AI Restyle product from "relight first frame + fal.ai v2v" to **background replacement on solid-colored source video** with personalized, pre-generated backgrounds. The user records talking-head video against a clean backdrop; we matte them out, drop them onto a generated background, and add realistic camera blur to the background plate.

Three things make this different from generic green-screen:

1. **One-time onboarding personalizes the backgrounds.** The user uploads a single selfie; we generate 5 background options tuned to their appearance and saved environment. They pick one and it sticks across sessions.
2. **Source-friendly matting.** We expect solid/low-noise source backgrounds and use a high-quality fal.ai matting model rather than a general video-style-transfer model.
3. **Camera-blur composite.** The new background gets a Gaussian / shallow-DOF blur before the subject is composited on top, so the result reads as a real shallow-depth-of-field shot rather than a hard cut-out.

## 2. Non-goals (v1)

- No lighting adjustment of the subject (out of scope per brief).
- No auth/user model — `profile_id` lives in browser `localStorage`, server treats it as opaque.
- No "Banana Pro" hardware tier — uses `gemini-3.1-flash-image-preview` (same model the codebase already uses for thumbnails). A separate `MODEL_NAME_PRO` constant is reserved so we can flip when a verified Pro variant becomes available.
- No per-job background override. v1 only lets you pick from the 5 saved options; if you want different backgrounds, re-onboard.
- No multi-profile support per browser (one selfie = one set of 5 = one selected bg, replaceable by re-onboarding).

## 3. User flow

```
                           ┌──────────────────────────────┐
                           │ 1. ONBOARDING (one-time)     │
                           │   Settings → Background      │
                           │   Profile section            │
                           │                              │
                           │   - upload selfie            │
                           │   - server generates 5 bgs   │
                           │   - user picks 1             │
                           │   - profile_id → localStorage│
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │ 2. WIZARD (per video)        │
                           │   /ai-restyle/wizard         │
                           │                              │
                           │   Step 1: Upload video       │
                           │   Step 2: Background-check + │
                           │           confirm bg choice  │
                           │   Step 3: Review result      │
                           └──────────────────────────────┘
```

If a user hits the wizard without a profile, Step 1 is replaced by a "Set up your profile first" panel that links to Settings → Background Profile.

## 4. Backend architecture

### 4.1 New modules

| Module | Purpose |
| --- | --- |
| `backend/app/profile/store.py` | Per-profile folder layout, atomic writes, JSON load/save. Public: `create_profile(selfie_bytes) -> profile_id`, `save_generated(profile_id, idx, png_bytes)`, `set_selected(profile_id, idx)`, `get_profile(profile_id) -> ProfileMeta`, `get_selected_background(profile_id) -> bytes`. |
| `backend/app/ml/profile_backgrounds.py` | Generate 5 personalized backgrounds from a selfie using Gemini image preview. Mirrors the call shape in `frame_relight.py`. Prompt asks for 5 distinct, talking-head-friendly backgrounds tuned to the person's coloring. |
| `backend/app/ml/bg_detect.py` | Heuristic for "is the source background clean enough". Reads the first frame, samples four border strips (top, bottom, left, right — center is the subject). Returns `(verdict: Literal["clean","noisy"], score: float, dominant_hex: str)`. Implemented via OpenCV (already a dep): per-strip color variance → mean. Threshold tuned via fixtures. |
| `backend/app/ml/video_matte.py` | Call fal.ai matting on the source video. Returns an `.mp4` (or per-frame PNG sequence — chosen at impl time after probing fal.ai's current video matting endpoint). Public: `matte_video(api_key, video_path, out_path) -> str`. SSRF guard reuses `integrations.fal.require_fal_download_url`. |
| `backend/app/video/composite.py` | FFmpeg-based composite: take the matted subject (alpha channel) and lay it over the pre-rendered blurred background. Uses the existing `video/ffmpeg.py` wrapper. Public: `composite_subject_over_background(matte_video, background_png, out_path) -> str`. |

### 4.2 Repurposed modules

| Module | Change |
| --- | --- |
| `backend/app/restyle/pipeline.py` | New 6-step async flow (see §5). Drops `relight_frame` and `restyle_video` calls. Old code deleted, not feature-flagged — pivot is clean. |
| `backend/app/routes/ai_restyle.py` | Existing `POST /api/restyle` + `GET /api/restyle/{job_id}` stay. New form field `profile_id: str = Form(...)`. Validates profile exists; 404 if not. New routes for onboarding (§4.3). |
| `backend/app/ml/frame_relight.py` | **Deleted.** No relight in this product. |
| `backend/app/ml/video_restyle.py` | **Deleted.** No v2v in this product. |
| `backend/app/ml/frame_extract.py` | Kept and reused by `bg_detect.py` to pull the first frame. |

### 4.3 New onboarding HTTP surface

All routes live in `backend/app/routes/ai_restyle.py`. All require `X-Gemini-Key` header (no fal key needed for onboarding — generation is Gemini-only).

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/restyle/profile` | Multipart upload: selfie image (PNG/JPG, ≤10MB). Returns `{profile_id}`. Server-side: save selfie, kick off background-generation job, return profile_id immediately. |
| `GET` | `/api/restyle/profile/{profile_id}` | Returns `{status, generated_count, selected_idx, backgrounds: [{idx, url}]}`. Status: `generating`, `ready`, `failed`. |
| `POST` | `/api/restyle/profile/{profile_id}/select` | Body: `{idx: int}`. Sets the selected background. |
| `POST` | `/api/restyle/profile/{profile_id}/regenerate` | Re-runs the 5-bg generation (e.g. user doesn't like any of them). |
| `GET` | `/profiles/{profile_id}/{filename}` | Static-file serve for selfie + generated PNGs. Read-only. |

### 4.4 Per-profile folder layout

Root: `backend/output/.profiles/<profile_id>/` (gitignored; `.profiles/` prefix keeps it visually separate from job output).

```
.profiles/<profile_id>/
├── meta.json              # {profile_id, created_at, selected_idx, generation_status}
├── selfie.png             # original upload (max 10MB, resized to 1024px long-edge before save)
├── bg-1.png ... bg-5.png  # Gemini-generated, 1080x1920 each (vertical for shorts)
└── (regeneration appends bg-6..bg-10, etc. — selected_idx tracks the canonical pick)
```

`meta.json` is the source of truth. Atomic writes via temp-file + rename, no `_job_lock` since each profile has at most one in-flight write (the onboarding job).

## 5. Job pipeline (per video upload)

```
1. Validate inputs           profile_id exists + has selected bg, file is MP4/MOV, duration ≤30s
2. Probe duration            reuse video/ffmpeg.py:probe_duration
3. Detect background clean   ml/bg_detect.detect_clean_background(input_video) → verdict
   ├─ "clean"  → log "✅ Clean source background"
   └─ "noisy"  → log "⚠️ Background may not be clean; results may vary"
                 (warn only, do NOT block)
4. Matte subject             ml/video_matte.matte_video(input_video) → matted_video.mp4 (with alpha)
5. Composite                 video/composite.composite_subject_over_background(
                                matted_video,
                                blurred(profile.selected_background),
                                final_out
                             )
   - blur is applied at composite time, not stored — keeps the source bg sharp for re-use
   - blur: FFmpeg gblur sigma=14 (≈ shallow DOF 50mm @ f/2 look). Constant for v1.
6. Mux audio                 video/ffmpeg.mux_video_audio(final_out, input_video, final_with_audio)
7. Persist + mark complete   jobs[job_id]["result"] = {video_url, original_url, profile_id, selected_bg_url}
```

All steps log to `jobs[job_id]["logs"]` with a `progress_pct` update (matches the existing AI Restyle telemetry the frontend polls).

## 6. Frontend changes

### 6.1 New Settings section: Background Profile

`frontend/src/pages/Settings/sections/BackgroundProfileSection.jsx`. Replaces `AIRestylePresetsSection.jsx` (relight presets are dead with the pivot).

Layout:
- **No profile yet:** drag-and-drop selfie upload + "Generate my backgrounds" button.
- **Generation in progress:** spinner + log feed (polled from `/api/restyle/profile/{id}`).
- **Ready:** grid of 5 backgrounds, click to select, star icon on the currently-selected one. "Regenerate" button (re-runs generation). "Replace selfie" button (full re-onboard).

`profile_id` stored in `localStorage` under `aiRestyleProfileId_v1`. Single value, single profile per browser.

### 6.2 Wizard reshape

`frontend/src/pages/AIRestyle/Wizard.jsx` and `steps/`:

| Step | Old | New |
| --- | --- | --- |
| 1 | Upload | Upload (unchanged, still client-side duration cap @ 30s) |
| 2 | Configure (preset + prompt textareas) | **Pre-check + confirm:** server detects bg cleanliness, shows the result. UI shows the selected profile background as a preview thumbnail. Button: "Restyle with this background". |
| 3 | Review | Review (unchanged: PhoneFrame Before/After, Send-to-Short-form sessionStorage handoff) |

If `localStorage.aiRestyleProfileId_v1` is missing, Step 2 shows a "Set up your profile first" empty state with a link to Settings.

### 6.3 Sidebar / routing

Unchanged. AI Restyle stays at `/ai-restyle/*` with the Wand2 icon.

### 6.4 Files retired

- `frontend/src/state/aiRestylePresets.js` — deleted; the per-job preset CRUD is gone.
- `frontend/src/pages/Settings/sections/AIRestylePresetsSection.jsx` — replaced by `BackgroundProfileSection.jsx`.
- `frontend/src/pages/AIRestyle/steps/Configure.jsx` — replaced by a new `Precheck.jsx` step.

## 7. Reasonable-call defaults

These are decisions I made during brainstorming without explicit user input. The user requested I proceed without further questions; flagging here so they're easy to flip during spec review.

| Decision | Choice | Why / Flip-if |
| --- | --- | --- |
| Background-generation model | `gemini-3.1-flash-image-preview` (same as elsewhere in the codebase). Constant `MODEL_NAME_PRO` defined alongside but currently equal. | No verified "Pro" model exists in this account. Flip the constant when one is confirmed. |
| Matting model | fal.ai endpoint TBD at impl time — probe `fal-ai/birefnet/v2` (still-image, loop per-frame) vs `fal-ai/sam2/video` and pick. Spec it as `BIREFNET_MODEL_ID` / `SAM2_VIDEO_MODEL_ID` constants in `ml/video_matte.py`. | Per-frame loop is slow; if SAM2 gives temporally consistent video matting it's the better default. Decision deferred to spike. |
| BG-clean detection threshold | OpenCV per-strip color variance. Tuned via 6 fixtures (3 clean: black/white/gray; 3 noisy: bookshelf/window/poster). Warn-only never block. | If false-positives become annoying, drop to warn=never and remove from pipeline. |
| Camera-blur strength | FFmpeg `gblur sigma=14`. Constant in `video/composite.py`. | Trivial to expose as a per-job slider later if users ask. |
| Selfie file cap | 10MB, resized to 1024px long-edge before generation. | Matches existing thumbnail upload pattern. |
| Number of generated backgrounds | 5 (per brief). Hard-coded. | If users want more, "Regenerate" appends instead of replacing — already in design. |
| Profile cleanup | None in v1. Profiles persist forever in `output/.profiles/`. | Add a "Delete profile" button + reaper if disk usage becomes an issue. |
| Selected-background storage | Stored as the index (1-5) in `meta.json`. The matted background image is regenerated (blur applied) per job. | Avoids stale blur params. |
| Composite mode | Subject over background only. No edge-feather, no env-halo. | User chose "just a blur, that is it". |
| Audio | Original audio always preserved (mux back at step 6). Same as PR #35. | Talking-head video is the target use case. |

## 8. Error handling

| Failure | Behavior |
| --- | --- |
| Profile doesn't exist | 404 from POST `/api/restyle`. Frontend redirects user to Settings → Background Profile. |
| Profile has no selected background | 400 from POST `/api/restyle`. Frontend prompts user to pick one in Settings. |
| Gemini bg-generation refusal (content policy) | Onboarding job → `status=failed`, logs surface the refusal reason. Frontend shows "Try a different selfie" + Retry button. |
| fal.ai matting failure | Job → `status=failed`, logs include the fal error code. Frontend shows the error + Retry. |
| Source video has noisy background | Pipeline logs a warning but proceeds. UI surfaces the warning on the Review screen alongside the result. |
| Duration > 30s | Same as today: 400 from POST `/api/restyle`. |
| Disk-DoS protections | Reuse existing: 250MB cap on video, 10MB cap on selfie (new). Content-Length preflight on both. |

## 9. Security baseline (from `securing-http-and-llm-endpoints` skill)

5 new HTTP surfaces. Tier classification:

| Endpoint | Tier | Required controls (subset) |
| --- | --- | --- |
| `POST /api/restyle/profile` | LLM-CALL (kicks off Gemini generation) | auth (X-Gemini-Key), input validation (file MIME + size), timeout/retry on Gemini, output rate limit (per-IP if exposed), audit logging |
| `GET /api/restyle/profile/{id}` | AUTHENTICATED-READ (profile_id is the cred) | basic input validation (UUID format) |
| `POST /api/restyle/profile/{id}/select` | STATE-MUTATING | input validation (idx in [1..N]), idempotency (set is idempotent), audit logging |
| `POST /api/restyle/profile/{id}/regenerate` | LLM-CALL + STATE-MUTATING | auth, rate limit (per profile_id; cap regenerations / hour), audit logging, cost cap |
| `GET /profiles/{id}/{filename}` | AUTHENTICATED-READ | path traversal guard (no `..`, normalize before serve), basename allowlist (`selfie.png`, `bg-N.png` only) |

Full audit happens in the implementation plan, not here. The skill MUST fire when the routes are added.

## 10. Testing strategy

| Layer | Tests |
| --- | --- |
| Unit: `bg_detect.py` | 6 fixtures (3 clean / 3 noisy). Assert verdict + score range. |
| Unit: `profile/store.py` | Round-trip create → save_generated → set_selected → get_profile. Concurrent write safety (two save_generated calls in flight). |
| Unit: `video_matte.py` | Mocked fal.ai client; assert SSRF guard rejects non-fal.media URLs, assert retry behavior on 5xx. |
| Unit: `composite.py` | Synthetic 1s clip with chromakey background + synthetic PNG bg; assert output exists and has expected dimensions. |
| API: `routes/ai_restyle.py` | Happy path (onboard → select → restyle). Error paths (missing profile, missing key, oversize selfie). Path traversal guard on `/profiles/{id}/{filename}`. |
| Snapshot: `baseline.openapi.json` | Regenerate; new routes appended. |
| E2E (skipped on CI): Full real-fal smoke against `demo-openshorts.mp4`. Requires FAL_KEY + GEMINI_API_KEY. |

Target: all unit + API tests green before merge. `pytest -m "not e2e"` stays at 100%.

## 11. Migration / rollout

PR #35 (current AI Restyle) ships first and is merged. This pivot lands as a follow-up PR that:

1. Deletes `frame_relight.py`, `video_restyle.py`, `aiRestylePresets.js`, `AIRestylePresetsSection.jsx`, `steps/Configure.jsx`.
2. Adds the new modules + routes + frontend section.
3. Updates the OpenAPI snapshot.
4. Updates `ROADMAP.md`: AI Restyle Shipped section reframed to "v1 (relight + v2v, retired)" + "v2 (background replacement, shipped)".
5. Updates `~/.claude/CLAUDE.md` MODULE-MAP via `scripts/update_claude_md.py`.

No backwards-compatibility shim. Users with in-flight v1 jobs will see them fail on next status poll — acceptable given the product is pre-launch.

## 12. Open items deferred to implementation plan

- Exact fal.ai matting model (BiRefNet per-frame vs SAM2 video) — pick during Task 4.
- The Gemini prompt template for background generation — draft + iterate during Task 2.
- BG-clean detection threshold tuning — calibrate against the 6 fixtures during Task 3.
- Whether the regenerate endpoint should cap at N total backgrounds (5? 20?) — pick a number in plan.
