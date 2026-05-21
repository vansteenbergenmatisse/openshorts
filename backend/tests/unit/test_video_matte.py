"""Tests for the fal.ai video matting wrapper."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.ml import video_matte


def test_matte_video_calls_fal_and_writes_output(tmp_path, monkeypatch):
    out_path = tmp_path / "matted.mp4"
    fake_url = "https://v3.fal.media/files/foo/matted.mp4"
    src = tmp_path / "source.mp4"
    src.write_bytes(b"FAKE-MP4")

    with patch.object(video_matte, "submit_and_poll", return_value={"video": {"url": fake_url}}) as mock_submit, \
         patch.object(video_matte, "upload_file", return_value="https://v3.fal.media/files/foo/src.mp4") as mock_upload, \
         patch.object(video_matte.httpx, "get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"FAKE-MATTED-MP4"
        mock_get.return_value.raise_for_status = lambda: None

        result = video_matte.matte_video(
            api_key="test",
            video_path=str(src),
            out_path=str(out_path),
        )

    assert result == str(out_path)
    assert out_path.read_bytes() == b"FAKE-MATTED-MP4"
    mock_upload.assert_called_once()
    mock_submit.assert_called_once()


def test_matte_video_rejects_non_fal_download_url(tmp_path, monkeypatch):
    src = tmp_path / "source.mp4"
    src.write_bytes(b"FAKE-MP4")
    with patch.object(video_matte, "submit_and_poll", return_value={"video": {"url": "https://evil.example.com/matted.mp4"}}), \
         patch.object(video_matte, "upload_file", return_value="https://v3.fal.media/files/foo/src.mp4"):
        with pytest.raises((ValueError, RuntimeError), match="untrusted fal download URL host"):
            video_matte.matte_video(
                api_key="test",
                video_path=str(src),
                out_path=str(tmp_path / "matted.mp4"),
            )


def test_matte_video_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        video_matte.matte_video(
            api_key="test",
            video_path=str(tmp_path / "nope.mp4"),
            out_path=str(tmp_path / "out.mp4"),
        )
