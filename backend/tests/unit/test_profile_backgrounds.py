"""Tests for the Gemini-driven 5-background generator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ml import profile_backgrounds


class _FakePart:
    def __init__(self, data: bytes):
        self.inline_data = MagicMock()
        self.inline_data.data = data


def _fake_response(image_count: int):
    resp = MagicMock()
    resp.parts = [_FakePart(b"\x89PNG\r\n\x1a\nfake-bg-%d" % i) for i in range(image_count)]
    return resp


def test_generates_five_backgrounds(tmp_path, monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(5)
    monkeypatch.setattr(profile_backgrounds.genai, "Client", lambda **kw: fake_client)

    selfie = tmp_path / "selfie.png"
    selfie.write_bytes(b"\x89PNG\r\n\x1a\nselfie")

    paths = profile_backgrounds.generate_personalized_backgrounds(
        api_key="test",
        selfie_path=str(selfie),
        out_dir=str(tmp_path / "out"),
        count=5,
    )
    assert len(paths) == 5
    for i, p in enumerate(paths, start=1):
        assert p.endswith(f"bg-{i}.png")
        assert open(p, "rb").read().endswith(b"fake-bg-%d" % (i - 1))


def test_raises_on_missing_selfie(tmp_path):
    with pytest.raises(FileNotFoundError):
        profile_backgrounds.generate_personalized_backgrounds(
            api_key="x",
            selfie_path=str(tmp_path / "nope.png"),
            out_dir=str(tmp_path),
            count=5,
        )


def test_raises_when_fewer_images_returned(tmp_path, monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(3)
    monkeypatch.setattr(profile_backgrounds.genai, "Client", lambda **kw: fake_client)

    selfie = tmp_path / "selfie.png"
    selfie.write_bytes(b"\x89PNG\r\n\x1a\nselfie")

    with pytest.raises(RuntimeError, match="returned 3 images"):
        profile_backgrounds.generate_personalized_backgrounds(
            api_key="test",
            selfie_path=str(selfie),
            out_dir=str(tmp_path / "out"),
            count=5,
        )


def test_raises_on_zero_images(tmp_path, monkeypatch):
    fake_client = MagicMock()
    resp = MagicMock()
    resp.parts = []
    fake_client.models.generate_content.return_value = resp
    monkeypatch.setattr(profile_backgrounds.genai, "Client", lambda **kw: fake_client)

    selfie = tmp_path / "selfie.png"
    selfie.write_bytes(b"\x89PNG\r\n\x1a\nselfie")

    with pytest.raises(RuntimeError, match="no images"):
        profile_backgrounds.generate_personalized_backgrounds(
            api_key="test",
            selfie_path=str(selfie),
            out_dir=str(tmp_path / "out"),
            count=5,
        )
