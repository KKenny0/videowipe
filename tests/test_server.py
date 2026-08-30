import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from videowipe.api import ProgressEvent, WipeResult
from videowipe.cli import _build_parser
from videowipe.plan import build_wipe_plan, compute_source, load_wipe_plan, save_wipe_plan
from videowipe.server import jobs
from videowipe.server import app as server_app


def _write_test_video(path, width=96, height=64, frames=8):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4,
        (width, height),
    )
    for _ in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[50:60, 10:86] = 200
        writer.write(frame)
    writer.release()


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _add_audio(video, output):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(output),
        ],
        check=True,
    )


class FakeEngine:
    def __init__(self):
        self.calls = []

    def plan(self, request, on_progress=None):
        self.calls.append(
            {
                "method": "plan",
                "video": request.video,
                "output": request.output_dir,
                "preview": request.preview,
                "intent": request.intent,
                "mask": request.mask,
                "plan": request.plan,
            }
        )
        if on_progress is not None:
            on_progress(ProgressEvent("detect", 0, 0))
            on_progress(ProgressEvent("persist", 1, 1))
        output_path = Path(request.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        candidates = {
            "candidates": [
                {
                    "id": "c1", "type": "subtitle", "label": "bottom subtitle",
                    "bbox": [10, 50, 86, 60], "confidence": 0.9,
                    "frame_fraction": 1.0, "reason": "wide bottom text",
                    "default_remove": True, "text_samples": ["subtitle"],
                    "selected": True,
                },
                {
                    "id": "c2", "type": "watermark", "label": "top watermark",
                    "bbox": [4, 4, 28, 16], "confidence": 0.6,
                    "frame_fraction": 1.0, "reason": "edge text",
                    "default_remove": False, "text_samples": [], "selected": False,
                },
            ]
        }
        (output_path / "clean_candidates.json").write_text(
            json.dumps(candidates), encoding="utf-8",
        )
        preview_image = np.zeros((64, 96, 3), dtype=np.uint8)
        preview_image[50:60, 10:86] = (0, 200, 0)
        cv2.imwrite(str(output_path / "clean_preview.jpg"), preview_image)
        mask_image = np.zeros((64, 96), dtype=np.uint8)
        mask_image[50:60, 10:86] = 255
        cv2.imwrite(str(output_path / "auto_mask.png"), mask_image)
        return self._write_plan(request.video, output_path)

    def run(self, request, on_progress=None):
        plan = load_wipe_plan(request.plan, video_path=request.video)
        self.calls.append(
            {
                "method": "run",
                "video": request.video,
                "output": request.output_dir,
                "preview": request.preview,
                "intent": request.intent,
                "mask": request.mask,
                "plan": plan,
            }
        )
        if on_progress is not None:
            on_progress(ProgressEvent("inpaint", 4, 8))
            on_progress(ProgressEvent("inpaint", 8, 8))
        output_path = Path(request.output_dir)
        result = output_path / "input_clean.mp4"
        shutil.copyfile(request.video, result)
        return WipeResult(
            output_path=str(result), backend="fake", mask_source="auto",
            timings={"inpaint": 0.01}, warnings=("fake warning",),
        )

    @staticmethod
    def _write_plan(video, output_path):
        """Write a real wipe_plan.json + .npz bound to *video* (c1 remove, c2 keep)."""
        def _candidate(cid, type_, label, bbox, confidence, default_remove):
            x1, y1, x2, y2 = bbox
            mask = np.zeros((64, 96), dtype=np.uint8)
            mask[y1:y2 + 1, x1:x2 + 1] = 1
            return SimpleNamespace(
                id=cid,
                type=type_,
                label=label,
                bbox=tuple(bbox),
                confidence=confidence,
                default_remove=default_remove,
                mask=mask,
                presence_frames=[0, 2, 4, 6],
            )

        plan = build_wipe_plan(
            [
                _candidate("c1", "subtitle", "bottom subtitle", [10, 50, 86, 60], 0.9, True),
                _candidate("c2", "watermark", "top watermark", [4, 4, 28, 16], 0.6, False),
            ],
            sample_indices=[0, 2, 4, 6],
            n_valid=4,
            source=compute_source(video),
            frame_shape=(64, 96),
        )
        save_wipe_plan(plan, str(output_path))
        return plan

    def cleanup(self):
        pass


@pytest.fixture()
def client(tmp_path, monkeypatch):
    jobs.reset_jobs()
    fake = FakeEngine()
    monkeypatch.setenv("VIDEOWIPE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(server_app, "_engine", fake)
    with TestClient(server_app.app) as test_client:
        yield test_client, fake
    jobs.reset_jobs()
    monkeypatch.setattr(server_app, "_engine", None)


def _post_video(client, video_path, intent="remove bottom subtitles"):
    with open(video_path, "rb") as fh:
        return client.post(
            "/jobs",
            data={"intent": intent},
            files={"video": ("input.mp4", fh, "video/mp4")},
        )


def _wait_for_state(client, job_id, state, timeout=5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/jobs/{job_id}").json()
        if last["state"] == state:
            return last
        if last["state"] == "error":
            raise AssertionError(last["error"])
        time.sleep(0.05)
    raise AssertionError(f"job did not reach {state}: {last}")


def test_cli_exposes_serve_command():
    parser = _build_parser()
    args = parser.parse_args(["serve", "--port", "9000"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 9000


def test_create_job_returns_pending(client, tmp_path):
    test_client, _ = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    response = _post_video(test_client, video)

    assert response.status_code == 200
    body = response.json()
    assert body["id"]
    assert body["state"] == "pending"
    assert body["phase"] in {"upload", "plan", "detect", "persist"}
    assert isinstance(body["warnings"], list)
    assert isinstance(body["timings"], dict)


def test_second_job_while_busy_returns_409(client, tmp_path):
    test_client, _ = client
    jobs.create_job(output_base=str(tmp_path / "jobs"))

    video = tmp_path / "input.mp4"
    _write_test_video(video)
    response = _post_video(test_client, video)

    assert response.status_code == 409


def test_cancel_current_preview_releases_busy_slot(client, tmp_path):
    test_client, _ = client
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    _write_test_video(first_video)
    _write_test_video(second_video)

    first_response = _post_video(test_client, first_video)
    first_job_id = first_response.json()["id"]
    _wait_for_state(test_client, first_job_id, "preview_ready")

    busy_response = _post_video(test_client, second_video)
    assert busy_response.status_code == 409

    cancel_response = test_client.delete("/jobs/current")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["state"] == "cancelled"

    second_response = _post_video(test_client, second_video)
    assert second_response.status_code == 200


def test_cancel_current_running_job_returns_409(client, tmp_path):
    test_client, _ = client
    job = jobs.create_job(output_base=str(tmp_path / "jobs"))
    with job.lock:
        job.state = "running"

    response = test_client.delete("/jobs/current")

    assert response.status_code == 409
    assert "running" in response.json()["detail"]


def test_preview_returns_candidates(client, tmp_path):
    test_client, fake = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    create_response = _post_video(test_client, video, intent="remove bottom subtitles")
    job_id = create_response.json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")

    response = test_client.get(f"/jobs/{job_id}/preview")

    assert response.status_code == 200
    body = response.json()
    assert body["preview_url"] == f"/jobs/{job_id}/preview-image"
    assert [candidate["id"] for candidate in body["candidates"]] == ["c1", "c2"]
    assert body["default_selected_ids"] == ["c1"]
    assert fake.calls[0]["intent"] == "remove bottom subtitles"
    # Phase B: preview also surfaces the plan's tracks (with actions + segments).
    tracks = body["tracks"]
    assert [track["id"] for track in tracks] == ["c1", "c2"]
    assert {track["id"]: track["action"] for track in tracks} == {
        "c1": "remove",
        "c2": "keep",
    }
    assert all(track.get("segments") is not None for track in tracks)
    snapshot = test_client.get(f"/jobs/{job_id}").json()
    assert snapshot["phase"] == "preview"
    assert snapshot["timings"]["upload_s"] >= 0
    assert snapshot["timings"]["plan_s"] >= 0


def test_confirm_runs_and_progress_sse(client, tmp_path):
    test_client, fake = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")

    confirm = test_client.post(
        f"/jobs/{job_id}/confirm",
        json={"selected_ids": ["c1"]},
    )
    assert confirm.status_code == 200

    progress = test_client.get(f"/jobs/{job_id}/progress")
    assert progress.status_code == 200
    assert "data:" in progress.text
    assert '"state": "done"' in progress.text
    assert '"progress": 1.0' in progress.text

    # Phase B: confirm executes via the plan (precise NPZ masks), not a
    # bbox-approximated mask. The bbox path is gone entirely.
    assert not hasattr(server_app, "_mask_from_selected_bboxes")
    confirm_call = fake.calls[-1]
    assert confirm_call["mask"] is None
    assert confirm_call["plan"] is not None
    actions = {track.id: track.action for track in confirm_call["plan"].tracks}
    assert actions == {"c1": "remove", "c2": "keep"}
    snapshot = test_client.get(f"/jobs/{job_id}").json()
    assert snapshot["phase"] == "complete"
    assert snapshot["warnings"] == ["fake warning"]
    assert snapshot["timings"]["inpaint"] == 0.01
    assert snapshot["timings"]["run_wall_s"] >= 0


def test_confirm_rejects_id_present_only_in_candidates_json(client, tmp_path):
    test_client, _ = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")

    candidates_path = tmp_path / "jobs" / job_id / "clean_candidates.json"
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    payload["candidates"].append({"id": "candidate-only", "selected": True})
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")

    response = test_client.post(
        f"/jobs/{job_id}/confirm",
        json={"selected_ids": ["candidate-only"]},
    )

    assert response.status_code == 400
    assert "candidate-only" in response.json()["detail"]
    assert test_client.get(f"/jobs/{job_id}").json()["state"] == "preview_ready"


def test_confirm_rejects_corrupt_plan_without_starting_job(client, tmp_path):
    test_client, _ = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    plan_path = tmp_path / "jobs" / job_id / "wipe_plan.json"
    plan_path.write_text("{", encoding="utf-8")

    response = test_client.post(
        f"/jobs/{job_id}/confirm",
        json={"selected_ids": ["c1"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "job plan is invalid"
    assert test_client.get(f"/jobs/{job_id}").json()["state"] == "preview_ready"


def test_web_page_contains_refresh_recovery_contract(client):
    test_client, _ = client

    page = test_client.get("/").text

    assert 'sessionStorage.getItem(jobStorageKey)' in page
    assert 'fetch("/jobs/current")' in page
    assert 'fetch(`/jobs/${storedId}`)' in page
    assert "async function restoreJob()" in page
    assert "startProgressStream(status.id)" in page
    assert 'status.state === "preview_ready") {\n        submitBtn.disabled = true;' in page
    assert 'status.state === "running") {\n        submitBtn.disabled = true;' in page
    assert "renderTracks(preview.tracks || [])" in page
    assert "preview.candidates.map" not in page


def test_confirm_toggles_remove_on_all_keep_default_plan(client, tmp_path):
    """An all-keep default plan (e.g. a video whose only overlay is a
    safety-kept logo) must still let the user toggle a track to remove and run.
    Confirm must apply the selection before any require_remove check, not
    reject the default plan up front."""
    test_client, fake = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")

    # Replace the preview's plan with an all-keep one bound to the same video.
    job_dir = tmp_path / "jobs" / job_id

    def _keep_candidate(cid, label, bbox):
        x1, y1, x2, y2 = bbox
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[y1:y2 + 1, x1:x2 + 1] = 1
        return SimpleNamespace(
            id=cid,
            type="logo",
            label=label,
            bbox=tuple(bbox),
            confidence=0.9,
            default_remove=False,
            mask=mask,
            presence_frames=[0, 2, 4, 6],
        )

    all_keep = build_wipe_plan(
        [
            _keep_candidate("c1", "top logo", [4, 4, 28, 16]),
            _keep_candidate("c2", "corner logo", [60, 4, 80, 16]),
        ],
        sample_indices=[0, 2, 4, 6],
        n_valid=4,
        source=compute_source(str(job_dir / "input.mp4")),
        frame_shape=(64, 96),
    )
    save_wipe_plan(all_keep, str(job_dir))

    confirm = test_client.post(
        f"/jobs/{job_id}/confirm",
        json={"selected_ids": ["c1"]},
    )
    assert confirm.status_code == 200
    _wait_for_state(test_client, job_id, "done")

    executed = fake.calls[-1]["plan"]
    assert {track.id: track.action for track in executed.tracks} == {
        "c1": "remove",
        "c2": "keep",
    }


def test_confirm_without_selected_ids_uses_current_wipe_plan_actions(client, tmp_path):
    """A no-body confirm follows an Agent-edited plan, not stale job defaults."""
    test_client, fake = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    create_response = _post_video(test_client, video)
    job_id = create_response.json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")

    plan_path = tmp_path / "jobs" / job_id / "wipe_plan.json"
    plan = load_wipe_plan(str(plan_path))
    for track in plan.tracks:
        track.action = "remove" if track.id == "c2" else "keep"
    save_wipe_plan(plan, str(plan_path.parent))

    confirm = test_client.post(f"/jobs/{job_id}/confirm", json={})
    assert confirm.status_code == 200
    _wait_for_state(test_client, job_id, "done")

    executed = fake.calls[-1]["plan"]
    assert {track.id: track.action for track in executed.tracks} == {
        "c1": "keep",
        "c2": "remove",
    }


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
def test_download_returns_mp4_with_audio(client, tmp_path):
    test_client, _ = client
    raw_video = tmp_path / "input_raw.mp4"
    audio_video = tmp_path / "input.mp4"
    _write_test_video(raw_video)
    _add_audio(raw_video, audio_video)

    job_id = _post_video(test_client, audio_video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    test_client.post(f"/jobs/{job_id}/confirm", json={"selected_ids": ["c1"]})
    _wait_for_state(test_client, job_id, "done")

    response = test_client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    downloaded = tmp_path / "download.mp4"
    downloaded.write_bytes(response.content)
    probe = subprocess.run(
        ["ffmpeg", "-i", str(downloaded), "-hide_banner"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "Audio:" in probe.stderr + probe.stdout
