from __future__ import annotations

import subprocess
from types import SimpleNamespace

from videowipe import propainter_wipe


def test_propainter_requires_explicit_checkout(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("VIDEOWIPE_PROPINTER_DIR", raising=False)

    result = propainter_wipe.main([
        str(tmp_path / "video.mp4"),
        str(tmp_path / "mask.png"),
        str(tmp_path / "output"),
    ])

    assert result == 2
    assert "not configured" in capsys.readouterr().err


def test_propainter_discards_child_output_by_default(
    monkeypatch, tmp_path, capsys
):
    checkout = tmp_path / "ProPainter"
    checkout.mkdir()
    (checkout / "inference_propainter.py").write_text("", encoding="utf-8")
    video = tmp_path / "video.mp4"
    mask = tmp_path / "mask.png"
    video.write_bytes(b"video")
    mask.write_bytes(b"mask")

    monkeypatch.setattr(propainter_wipe, "extract_frames", lambda *args: 24)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(propainter_wipe.subprocess, "run", fake_run)

    result = propainter_wipe.main([
        str(video),
        str(mask),
        str(tmp_path / "output"),
        "--propainter-dir",
        str(checkout),
    ])

    assert result == 1
    assert captured["stdout"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.DEVNULL
    assert "capture_output" not in captured
    assert "exited with code 7" in capsys.readouterr().err
