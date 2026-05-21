"""Gemini-driven personalized background generator.

Takes a selfie, asks Gemini for N talking-head-friendly vertical backgrounds
tuned to the person's coloring and aesthetic. Mirrors the call shape in
``backend/app/thumbnails/images.py``.
"""
from __future__ import annotations

import os
from typing import List

from google import genai
from google.genai import types

# Current model. A future verified Pro variant can replace this constant.
MODEL_NAME = "gemini-3.1-flash-image-preview"
MODEL_NAME_PRO = MODEL_NAME  # Placeholder; flip when a Pro model is verified.


def build_generation_prompt(count: int) -> str:
    """Compose the Gemini prompt for N personalized backgrounds."""
    return (
        f"Look at this person and generate {count} distinct, professional, "
        "vertical (9:16) background images suitable for a talking-head short "
        "video. Each background should:\n"
        "- Feel like a real photographed environment (not a flat color or "
        "  pure gradient).\n"
        "- Have a clear focal depth that supports a shallow-DOF blur applied "
        "  in post.\n"
        "- Use a color palette that complements the person's skin tone, hair, "
        "  and apparent style.\n"
        "- Span variety: e.g. modern home office, minimalist studio, warm "
        "  cafe, plant-filled corner, industrial loft, soft outdoor.\n"
        "- Contain NO people, NO text, NO logos.\n"
        f"\nReturn exactly {count} images. No commentary."
    )


def generate_personalized_backgrounds(
    api_key: str,
    selfie_path: str,
    out_dir: str,
    count: int = 5,
    use_pro: bool = False,
) -> List[str]:
    """Generate ``count`` background PNGs personalized to the selfie.

    Returns the list of absolute output paths in deterministic 1..count order.
    Raises FileNotFoundError if selfie is missing. Raises RuntimeError if
    Gemini returns fewer than ``count`` images (typically a content-policy
    issue).
    """
    if not os.path.exists(selfie_path):
        raise FileNotFoundError(f"Selfie not found: {selfie_path}")
    os.makedirs(out_dir, exist_ok=True)

    with open(selfie_path, "rb") as f:
        selfie_bytes = f.read()

    client = genai.Client(api_key=api_key)
    model = MODEL_NAME_PRO if use_pro else MODEL_NAME
    prompt = build_generation_prompt(count)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=selfie_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    image_parts = [
        p for p in (response.parts or [])
        if getattr(getattr(p, "inline_data", None), "data", None)
    ]

    if not image_parts:
        raise RuntimeError(
            "Gemini returned no images (likely content-policy refusal)"
        )
    if len(image_parts) < count:
        raise RuntimeError(
            f"Gemini returned {len(image_parts)} images, expected {count}"
        )

    out_paths: List[str] = []
    for i, part in enumerate(image_parts[:count], start=1):
        path = os.path.join(out_dir, f"bg-{i}.png")
        with open(path, "wb") as f:
            f.write(part.inline_data.data)
        out_paths.append(path)
    return out_paths
