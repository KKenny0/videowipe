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
    """Write a test video with a bright region at rows 50-59, cols 10-85.

    Uses ffmpeg (rawvideo→libx264) when available so that both
    cv2.CAP_PROP_FRAME_COUNT and ffprobe avg_frame_rate are reliably set on
    all CI platforms (Ubuntu, Windows, macOS).  Falls back to cv2+mp4v when
    ffmpeg is absent.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[50:60, 10:86] = 200
    raw_pixels = (frame.tobytes()) * frames  # BGR, same frame repeated

    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
                "-r", "4", "-i", "pipe:0",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                str(path),
            ],
            input=raw_pixels,
            check=True,
        )
        return

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4,
        (width, height),
    )
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def test_recovery_reads_utf8_with_legacy_system_encoding(client, tmp_path, monkeypatch):
    test_client, _ = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    job.commit(original_filename="字幕.mp4")
    jobs.reset_jobs()
    read_text = Path.read_text

    def legacy_read(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", legacy_read)
    restored = test_client.get("/jobs/current").json()
    assert restored["id"] == job_id
    assert restored["original_filename"] == "字幕.mp4"


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
        from types import SimpleNamespace
        self._task_impl = SimpleNamespace(feather_radius=4)
        self.calls = []

    def plan(self, request, on_progress=None, cancellation=None, on_candidates=None):
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
        cv2.imwrite(str(output_path / "clean_preview_source.jpg"), preview_image)
        mask_image = np.zeros((64, 96), dtype=np.uint8)
        mask_image[50:60, 10:86] = 255
        cv2.imwrite(str(output_path / "auto_mask.png"), mask_image)
        plan = self._write_plan(request.video, output_path)
        self._write_plan(request.video, output_path / "coarse")
        (output_path / "refinement_evidence.json").write_text("{}")
        if on_candidates:
            on_candidates({"source": plan.source.to_dict(), "candidates": candidates["candidates"],
                           "default_selected_ids": ["c1"], "evidence": {}})
        return plan

    def _trial_identity(self):
        return {"backend": "fake", "device": "cpu", "weight_sha256": "fake", "precision": "float32"}

    def _refine_review(self, video, machine, plan, evidence, **kwargs):
        return plan

    def run(self, request, on_progress=None, cancellation=None):
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
        if request.trial_range:
            reader = cv2.VideoCapture(str(request.video))
            width, height = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(result), cv2.VideoWriter_fourcc(*"mp4v"),
                                     reader.get(cv2.CAP_PROP_FPS), (width, height * 2))
            for index in range(request.trial_range[1]):
                ok, frame = reader.read()
                assert ok
                if index >= request.trial_range[0]:
                    writer.write(np.vstack([frame, frame]))
            reader.release()
            writer.release()
        else:
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
    assert body["editable_preview_url"] == f"/jobs/{job_id}/editable-preview-image"
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
    editable = test_client.get(body["editable_preview_url"])
    assert editable.status_code == 200
    assert editable.headers["content-type"].startswith("image/jpeg")


def test_confirm_bbox_override_replaces_only_edited_track_mask(client, tmp_path):
    test_client, fake = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job_dir = tmp_path / "jobs" / job_id
    before = load_wipe_plan(str(Path(jobs.get_job(job_id).plan_dir) / "wipe_plan.json"))
    before_sha = before.mask_asset.sha256
    c2_before = next(track for track in before.tracks if track.id == "c2").mask.copy()

    response = test_client.post(
        f"/jobs/{job_id}/confirm",
        json={
            "selected_ids": ["c1"],
            "bbox_overrides": {"c1": [8, 48, 90, 62]},
        },
    )

    assert response.status_code == 200
    _wait_for_state(test_client, job_id, "done")
    executed = fake.calls[-1]["plan"]
    c1 = next(track for track in executed.tracks if track.id == "c1")
    c2 = next(track for track in executed.tracks if track.id == "c2")
    expected = np.zeros((64, 96), dtype=np.uint8)
    expected[48:63, 8:91] = 1
    assert c1.bbox == (8, 48, 90, 62)
    assert c1.decision_reason == "user-confirm:remove:bbox-override"
    assert np.array_equal(c1.mask, expected)
    assert np.array_equal(c2.mask, c2_before)
    assert executed.mask_asset.sha256 != before_sha


@pytest.mark.parametrize(
    ("payload", "status_code"),
    [
        ({"selected_ids": ["c1"], "bbox_overrides": {"missing": [0, 0, 2, 2]}}, 400),
        ({"selected_ids": ["c1"], "bbox_overrides": {"c2": [4, 4, 28, 16]}}, 400),
        ({"selected_ids": ["c1"], "bbox_overrides": {"c1": [8, 8, 7, 12]}}, 400),
        ({"selected_ids": ["c1"], "bbox_overrides": {"c1": [8, 8, 8, 12]}}, 400),
        ({"selected_ids": ["c1"], "bbox_overrides": {"c1": [8, 8, 96, 12]}}, 400),
        ({"selected_ids": ["c1"], "bbox_overrides": {"c1": [8, 8, 12]}}, 422),
        ({"selected_ids": ["c1"], "bbox_overrides": {"c1": [8.5, 8, 12, 12]}}, 422),
    ],
)
def test_invalid_bbox_override_keeps_plan_unchanged(
    client, tmp_path, payload, status_code,
):
    test_client, fake = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job_dir = tmp_path / "jobs" / job_id
    plan_path = Path(jobs.get_job(job_id).plan_dir) / "wipe_plan.json"
    mask_path = plan_path.parent / "wipe_plan_masks.npz"
    before = (plan_path.read_bytes(), mask_path.read_bytes())

    response = test_client.post(f"/jobs/{job_id}/confirm", json=payload)

    assert response.status_code == status_code
    assert test_client.get(f"/jobs/{job_id}").json()["state"] == "preview_ready"
    assert (plan_path.read_bytes(), mask_path.read_bytes()) == before
    assert [call["method"] for call in fake.calls] == ["plan"]


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

    candidates_path = Path(jobs.get_job(job_id).plan_dir) / "clean_candidates.json"
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
    plan_path = Path(jobs.get_job(job_id).plan_dir) / "wipe_plan.json"
    plan_path.write_text("{", encoding="utf-8")

    response = test_client.post(
        f"/jobs/{job_id}/confirm",
        json={"selected_ids": ["c1"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "job plan is invalid"
    assert test_client.get(f"/jobs/{job_id}").json()["state"] == "preview_ready"


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

    plan_path = Path(jobs.get_job(job_id).plan_dir) / "wipe_plan.json"
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


def test_trial_isolated_plan_retry_and_media(client, tmp_path, monkeypatch):
    test_client, fake = client
    video = tmp_path / "source.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    canonical = Path(job.plan_dir) / "wipe_plan.json"
    original = canonical.read_bytes()
    body = {"selected_ids": ["c1"], "bbox_overrides": {"c1": [12, 50, 80, 60]},
            "start_seconds": 1.25, "duration_seconds": 3}
    assert test_client.post(f"/jobs/{job_id}/trial", json=body).status_code == 200
    state = _wait_for_state(test_client, job_id, "preview_ready")
    assert state["trial"]["ready"]
    assert state["trial"]["frame_range"] == [5, 8]  # truncated at source end
    assert canonical.read_bytes() == original
    assert fake.calls[-1]["plan"].tracks[0].bbox == (12, 50, 80, 60)
    assert test_client.get("/jobs/current").json()["id"] == job_id
    trial_id = state["trial"]["id"]
    response = test_client.get(f"/jobs/{job_id}/trial-video?trial_id={trial_id}",
                               headers={"Range": "bytes=0-15"})
    assert response.status_code == 206
    assert len(response.content) == 16
    assert test_client.get(f"/jobs/{job_id}/trial-video?trial_id=old").status_code == 409
    assert test_client.get(f"/jobs/{job_id}/download").status_code == 409
    original_run = fake.run

    def fail(*args, **kwargs):
        raise RuntimeError("test inference failure")

    monkeypatch.setattr(fake, "run", fail)
    body["bbox_overrides"]["c1"] = [14, 50, 80, 60]
    test_client.post(f"/jobs/{job_id}/trial", json=body)
    failed = _wait_for_state(test_client, job_id, "preview_ready")
    assert failed["trial"]["id"] == trial_id  # previous successful trial survives
    assert "test inference failure" in failed["error"]
    assert canonical.read_bytes() == original
    monkeypatch.setattr(fake, "run", original_run)
    assert test_client.post(f"/jobs/{job_id}/trial", json=body).status_code == 200
    assert _wait_for_state(test_client, job_id, "preview_ready")["trial"]["ready"]
    # Trial must not silently carry its edits into a subsequent full run.
    assert test_client.post(f"/jobs/{job_id}/confirm", json={"selected_ids": ["c1"]}).status_code == 200
    _wait_for_state(test_client, job_id, "done")
    assert fake.calls[-1]["plan"].tracks[0].bbox == (10, 50, 86, 60)


@pytest.mark.parametrize("body,code", [
    ({"selected_ids": []}, 400),
    ({"selected_ids": ["unknown"]}, 400),
    ({"start_seconds": -1}, 422),
    ({"start_seconds": "NaN"}, 422),
    ({"start_seconds": "Infinity"}, 422),
    ({"start_seconds": 2}, 400),
    ({"start_seconds": 1e308}, 400),
    ({"duration_seconds": 0}, 422),
    ({"duration_seconds": 6}, 422),
    ({"bbox_overrides": {"c1": [0, 0, 100, 60]}}, 400),
])
def test_trial_rejects_invalid_inputs(client, tmp_path, body, code):
    test_client, _fake = client
    video = tmp_path / "source.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    assert test_client.post(f"/jobs/{job_id}/trial", json=body).status_code == code
    assert test_client.get(f"/jobs/{job_id}").json()["state"] == "preview_ready"


def test_trial_serializes_work_and_rejects_inactive_interval(client, tmp_path, monkeypatch):
    import threading
    from videowipe.plan import Segment
    test_client, fake = client
    video = tmp_path / "source.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    path = Path(job.plan_dir) / "wipe_plan.json"
    plan = load_wipe_plan(str(path))
    plan.tracks[0].segments = [Segment(4, 8)]
    save_wipe_plan(plan, job.plan_dir)
    assert test_client.post(f"/jobs/{job_id}/trial", json={"duration_seconds": 0.5}).status_code == 400
    release = threading.Event()
    original_run = fake.run

    def blocked(*args, **kwargs):
        assert release.wait(5)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(fake, "run", blocked)
    try:
        assert test_client.post(f"/jobs/{job_id}/trial", json={}).status_code == 200
        assert test_client.get(f"/jobs/{job_id}").json()["state"] == "trial_running"
        assert test_client.post(f"/jobs/{job_id}/trial", json={}).status_code == 409
        assert test_client.post(f"/jobs/{job_id}/confirm", json={}).status_code == 409
        assert test_client.delete("/jobs/current").status_code == 409
        assert _post_video(test_client, video).status_code == 409
    finally:
        release.set()
    _wait_for_state(test_client, job_id, "preview_ready")



@pytest.mark.parametrize("intervals,expected", [
    ([(300, 500)], 14.5),  # first subtitle at 12s, never recommend the empty intro
    ([(10, 20)], 0.0),  # less than three seconds
    ([(495, 500)], 17.0),  # clip end
    ([(50, 100), (150, 200)], 1.5),  # earliest interval wins ties
    ([(0, 50), (40, 125), (200, 250)], 1.0),  # merge overlaps first
])
def test_recommended_trial_contains_real_execution_mask(intervals, expected):
    from videowipe.plan import Segment
    track = SimpleNamespace(segments=[Segment(a, b) for a, b in intervals], mask=np.ones((2, 2)))
    plan = SimpleNamespace(source=SimpleNamespace(fps=25, frame_count=500), remove_tracks=[track])
    recommendation = server_app._recommended_trial(plan)
    assert recommendation["start_seconds"] == expected
    assert recommendation["frame_range"] == list(server_app._trial_range(plan, expected, 3))
    track.mask[:] = 0
    assert server_app._recommended_trial(plan) is None
    with pytest.raises(server_app.HTTPException):
        server_app._trial_range(plan, expected, 3)
    plan.remove_tracks = []
    assert server_app._recommended_trial(plan) is None


def test_media_is_job_bound_and_evidence_uses_observed_frame(client, tmp_path):
    test_client, _ = client
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (96, 64))
    for i in range(8):
        writer.write(np.full((64, 96, 3), i * 28, dtype=np.uint8))
    writer.release()
    with video.open("rb") as fh:
        created = test_client.post("/jobs", files={"video": ("C:\\假目录\\中文片段.mp4", fh, "video/mp4")}).json()
    job_id = created["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    assert job.original_filename == "中文片段.mp4"
    assert Path(job.video_path).name == "input.mp4"
    candidates_path = Path(job.plan_dir) / "clean_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates["candidates"][0]["presence_frames"] = [5, 3, 7]
    candidates_path.write_text(json.dumps(candidates))
    preview = test_client.get(f"/jobs/{job_id}/preview").json()
    first, fallback = preview["tracks"]
    assert (first["evidence_frame"], first["evidence_kind"]) == (3, "observed")
    assert fallback["evidence_kind"] == "position_reference"
    assert first["has_mask"] is True
    frame_response = test_client.get(first["evidence_url"])
    image = cv2.imdecode(np.frombuffer(frame_response.content, np.uint8), cv2.IMREAD_COLOR)
    assert abs(float(image.mean()) - 84) < 8
    cache = Path(job.output_dir) / "evidence-3.jpg"
    stamp = cache.stat().st_mtime_ns
    assert test_client.get(first["evidence_url"]).content == frame_response.content
    assert cache.stat().st_mtime_ns == stamp
    for frame, code in [("-1", 422), ("3.5", 422), ("8", 400)]:
        assert test_client.get(f"/jobs/{job_id}/frame?frame_index={frame}").status_code == code
    original = test_client.get(f"/jobs/{job_id}/source-video", headers={"Range": "bytes=0-15"})
    assert original.status_code == 206 and original.content == video.read_bytes()[:16]
    assert test_client.get(f"/jobs/{job_id}/result-video").status_code == 409
    test_client.post(f"/jobs/{job_id}/confirm", json={"selected_ids": ["c1"]})
    _wait_for_state(test_client, job_id, "done")
    assert "runs" in Path(job.result_path).parts
    assert test_client.get(f"/jobs/{job_id}/preview").json()["confirmed_review"] == {"selected_ids": ["c1"], "bbox_overrides": {}}
    assert test_client.get(f"/jobs/{job_id}/result-video", headers={"Range": "bytes=0-15"}).status_code == 206
    download = test_client.get(f"/jobs/{job_id}/download")
    from urllib.parse import unquote
    assert "中文片段_clean.mp4" in unquote(download.headers["content-disposition"])
    # Never select an arbitrary glob match or read another job's/symlinked media.
    job.result_path = None
    assert test_client.get(f"/jobs/{job_id}/download").status_code == 404
    job.result_path = str(video)
    assert test_client.get(f"/jobs/{job_id}/result-video").status_code == 409
    outside = Path(job.output_dir) / "outside.mp4"
    outside.symlink_to(video)
    job.result_path = str(outside)
    assert test_client.get(f"/jobs/{job_id}/result-video").status_code == 409


@pytest.mark.parametrize("name, expected", [
    ("../片段.mp4", "片段.mp4"), ("a\\b\\片段.mp4", "片段.mp4"),
    ("\r\n\x00", "input.mp4"), ("x" * 200, "x" * 120),
])
def test_display_name_is_not_a_storage_path(name, expected):
    assert server_app._display_filename(name) == expected


@pytest.mark.skipif(not shutil.which("node"), reason="requires node")
def test_workspace_recommendation_and_stale_trial_state():
    page = server_app._web_index().read_text(encoding="utf-8")
    recommendation = page.split("    function recommend() {", 1)[1].split("    function invalidateTrial()", 1)[0]
    transition = page.split("    async function applyStatus(data) {", 1)[1].split("    async function showDone", 1)[0]
    signature = page.split("    function signature(request) {", 1)[1].split("    function saveReview(", 1)[0]
    harness = r'''
const assert = require("node:assert/strict");
const nodes = {start:{value:0},duration:{value:3}};
const $ = id => nodes[id] ||= {classList:{},removeAttribute(){}};
const source = {fps:25,frame_count:500};
const duration = () => 20;
let mask=true; const bboxDrafts={};
const selectedTracks = () => [{id:"c1",has_mask:mask,segments:[[300,500]]}];
let currentJobId="job", state="trial_running", latestState, reviewLoaded=false, trial=null;
let finalPlanReady=false,previewLoading=false,resultRevision=null;
let events=[];
const renderActions=()=>{}, renderTargets=()=>{}, renderTimeline=()=>{}, renderOverlays=()=>{};
const closeStream=()=>events.push("closed"), notify=(text)=>events.push(text), showWarnings=()=>{};
const loadPreview=async()=>{reviewLoaded=true;events.push("preview");};
const showDone=async()=>events.push("done"), setMedia=()=>events.push("media"), storageSet=()=>{};
const reviewRequest=()=>({selected_ids:["c1"],start_seconds:14.5,duration_seconds:3});
'''
    assertions = r'''
(async()=>{
recommend();assert.equal(nodes.start.value,"14.5");
mask=false;nodes.start.value="0";recommend();assert.equal(nodes.start.value,"0");
bboxDrafts.c1=[1,1,5,5];recommend();assert.equal(nodes.start.value,"14.5");
await applyStatus({id:"other",state:"done"});assert.deepEqual(events,[]);
await applyStatus({id:"job",state:"preview_ready",trial:{ready:true,request:{selected_ids:["c2"]}}});
assert.deepEqual(events,["closed","preview"]);assert.equal(trial,null);
events=[];await applyStatus({id:"job",state:"done"});assert.deepEqual(events,["closed","done"]);
events=[];await applyStatus({id:"job",state:"preview_ready",error:"encode failure"});
assert.equal(state,"preview_ready");assert.equal(trial,null);assert(events[1].includes("可以重试"));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run(["node", "-e", harness + "function signature(request) {" + signature
                    + "function recommend() {" + recommendation
                    + "async function applyStatus(data) {" + transition + assertions], check=True)


def test_review_and_evidence_paths_reject_symlink_escape(client, tmp_path):
    test_client, _ = client
    video = tmp_path / "source.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(job.output_dir) / "runs").symlink_to(outside, target_is_directory=True)
    assert test_client.post(f"/jobs/{job_id}/confirm", json={}).status_code == 409
    assert job.state == "preview_ready" and not list(outside.iterdir())
    (Path(job.output_dir) / "evidence-0.jpg").symlink_to(video)
    assert test_client.get(f"/jobs/{job_id}/frame?frame_index=0").status_code == 409
    assert len(server_app._display_filename("中" * 200).encode("utf-8")) <= 180
    plan_path = Path(job.plan_dir) / "wipe_plan.json"
    plan_path.rename(plan_path.with_suffix(".bak"))
    assert test_client.get(f"/jobs/{job_id}/frame?frame_index=0").status_code == 409


@pytest.mark.skipif(not shutil.which("node"), reason="requires node")
@pytest.mark.parametrize("confirmed", [False, True])
def test_restored_edited_trial_still_loads_source_video(confirmed):
    html = server_app._web_index().read_text(encoding="utf-8")
    function = html.split("    async function loadPreview(jobId) {", 1)[1].split("    function closeStream()", 1)[0]
    harness = r'''
const assert=require("node:assert/strict");
const preview={source:{fps:25,width:100,height:100,frame_count:500},original_filename:"clip.mp4",
tracks:[{id:"c1",bbox:[1,1,50,50],action:"remove",evidence_frame:300}],
trial:{ready:true,start_seconds:13,request:{selected_ids:["c1"],bbox_overrides:{},start_seconds:12,duration_seconds:3}}};
const saved={selected_ids:["c1"],bbox_overrides:{c1:[2,2,51,51]},start_seconds:12,duration_seconds:3};
let source,tracks=[],selectedIds=[],bboxDrafts={},manualTime,reviewLoaded,activeTrackId,reviewRevision,finalPlanReady;
let segmentDrafts={},protections=[],savedReview=null,undoStack=[];const renderProtections=()=>{};
let currentJobId="job",state="preview_ready";
const nodes={};const $=id=>nodes[id]||=({replaceChildren(){}});
const request=async()=>preview;
const storageGet=key=>key.includes("review")?JSON.stringify(saved):"13";
const signature=JSON.stringify,reviewRequest=()=>saved;
const recommend=()=>{},updateMeta=()=>{},renderTargets=()=>{},renderTimeline=()=>{},renderOverlays=()=>{};
const events=[];const setMedia=(view,at)=>events.push([view,at]);
'''
    assertions = r'''
(async()=>{await loadPreview("job");assert.deepEqual(events,[["source",13]]);assert.deepEqual(bboxDrafts,saved.bbox_overrides);})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    if not confirmed:
        harness += "preview.overrides=saved;"
    if confirmed:
        harness += "preview.confirmed_review={...saved,bbox_overrides:{c1:[3,3,52,52]}};"
        assertions = assertions.replace("saved.bbox_overrides", "preview.confirmed_review.bbox_overrides")
    subprocess.run(["node", "-e", harness + "async function loadPreview(jobId) {" + function + assertions], check=True)


@pytest.mark.parametrize("suffix", ["html", "svg"])
def test_source_media_never_serves_active_document(client, tmp_path, suffix):
    test_client, _ = client
    video = tmp_path / "source.mp4"
    _write_test_video(video)
    created = test_client.post("/jobs", files={
        "video": (f"clip.{suffix}", video.read_bytes(), "application/octet-stream"),
    }).json()
    _wait_for_state(test_client, created["id"], "preview_ready")
    response = test_client.get(f"/jobs/{created['id']}/source-video")
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.skipif(not shutil.which("node"), reason="requires node")
def test_workspace_media_identity_transitions():
    html = server_app._web_index().read_text(encoding="utf-8")
    pick = html.split("    async function pickFile(file) {", 1)[1].split("    async function submit()", 1)[0]
    inspect = html.split("    function inspect(id) {", 1)[1].split("    function imageMetrics()", 1)[0]
    visibility = html.split("    function updateOverlayVisibility() {", 1)[1].split("    function renderOverlays()", 1)[0]
    declarations = html.split('    const busyStates = ', 1)[1].split('    const evidenceImages', 1)[0]
    script = r'''
const assert=require("node:assert/strict");
const events=[];sourceSupported=false;
const busy=()=>busyStates.has(state);
const renderTargets=()=>{},setMedia=(view,at)=>events.push([view,at]);
const position=()=>29/25;
const box={dataset:{trackId:"c1"},hidden:true};const $=()=>({children:[box]});
'''
    assertions = r'''
(async()=>{
await pickFile({name:"other.mp4"});assert.equal(selectedFile,null);
source={fps:25};tracks=[{id:"c1",evidence_frame:29,segments:[[29,30]]}];
inspect("c1");assert.deepEqual(events,[["source",29/25]]);
updateOverlayVisibility();assert.equal(box.hidden,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    # Exercise the source-supported branch separately for the seconds/frame round-trip.
    assertions = assertions.replace('updateOverlayVisibility();', 'sourceSupported=true;updateOverlayVisibility();')
    subprocess.run(["node", "-e", "const busyStates = " + declarations + script
                    + "async function pickFile(file) {" + pick
                    + "function inspect(id) {" + inspect
                    + "function updateOverlayVisibility() {" + visibility + assertions], check=True)


def test_early_review_cas_survives_refinement(client, tmp_path, monkeypatch):
    import threading
    test_client, fake = client
    video = tmp_path / "early.mp4"
    _write_test_video(video)
    ready, finish = threading.Event(), threading.Event()
    original = fake.plan

    def blocked(request, **kwargs):
        plan = original(request, **kwargs)
        ready.set()
        assert finish.wait(5)
        return plan

    monkeypatch.setattr(fake, "plan", blocked)
    job_id = _post_video(test_client, video).json()["id"]
    assert ready.wait(5)
    try:
        snapshot = test_client.get(f"/jobs/{job_id}").json()
        assert snapshot["state"] == "pending" and snapshot["review_ready"]
        assert not snapshot["final_plan_ready"]
        assert test_client.get(f"/jobs/{job_id}/preview").status_code == 200
        body = {"expected_revision": 0, "overrides": {
            "selected_ids": ["c1"], "bbox_overrides": {"c1": [12, 50, 80, 60]},
        }}
        saved = test_client.patch(f"/jobs/{job_id}/review", json=body)
        assert saved.status_code == 200 and saved.json()["review_revision"] == 1
        assert test_client.patch(f"/jobs/{job_id}/review", json=body).status_code == 409
        assert test_client.post(f"/jobs/{job_id}/trial", json={}).status_code == 409
    finally:
        finish.set()
    snapshot = _wait_for_state(test_client, job_id, "preview_ready")
    assert snapshot["overrides"] == body["overrides"]
    assert snapshot["final_plan_ready"] and snapshot["review_revision"] == 1
    assert test_client.post(f"/jobs/{job_id}/confirm", json={"expected_revision": 0}).status_code == 409


def test_saved_trial_cache_and_result_survive_restart(client, tmp_path):
    test_client, fake = client
    video = tmp_path / "resume.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    body = {"selected_ids": ["c1"], "start_seconds": 0, "duration_seconds": 1}
    test_client.post(f"/jobs/{job_id}/trial", json=body)
    first = _wait_for_state(test_client, job_id, "preview_ready")
    assert first["trial"]["ready"] and not first["trial"]["cache_hit"]
    calls = len(fake.calls)
    test_client.post(f"/jobs/{job_id}/trial", json=body)
    second = _wait_for_state(test_client, job_id, "preview_ready")
    assert second["trial"]["cache_hit"] and len(fake.calls) == calls
    test_client.post(f"/jobs/{job_id}/confirm", json={"selected_ids": ["c1"]})
    done = _wait_for_state(test_client, job_id, "done")
    manifest = json.loads((Path(jobs.get_job(job_id).output_dir) / "job.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert not Path(manifest["plan_dir"]).is_absolute()
    jobs.reset_jobs()
    restored = test_client.get("/jobs/current").json()
    assert restored["state"] == "done" and restored["id"] == job_id
    assert restored["trial"]["id"] == done["trial"]["id"]
    assert test_client.get(f"/jobs/{job_id}/result-video").status_code == 200
    assert test_client.get(f"/jobs/{job_id}/trial-video?trial_id={done['trial']['id']}").status_code == 200


@pytest.mark.parametrize("damage", ["schema", "escape", "symlink", "source", "pair", "json"])
def test_recovery_rejects_corrupt_or_foreign_manifest(client, tmp_path, damage):
    test_client, _ = client
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    path = Path(job.output_dir) / "job.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if damage == "schema":
        data["schema_version"] = 99
    elif damage == "escape":
        data["video_path"] = "../outside.mp4"
    elif damage == "symlink":
        Path(job.video_path).unlink()
        Path(job.video_path).symlink_to(video)
    elif damage == "source":
        Path(job.video_path).write_bytes(b"replaced input")
    elif damage == "pair":
        (Path(job.plan_dir) / "wipe_plan_masks.npz").write_bytes(b"incomplete masks")
    if damage == "json":
        path.write_text("{")
    else:
        path.write_text(json.dumps(data))
    jobs.reset_jobs()
    response = test_client.get("/jobs/current").json()
    assert response["state"] == "idle" and response["recovery_error"]
    assert jobs.get_job(job_id) is None


def test_cancel_holds_slot_and_duplicate_operation_does_not_start_worker(client, tmp_path, monkeypatch):
    import threading
    test_client, fake = client
    video = tmp_path / "cancel.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    entered, finish = threading.Event(), threading.Event()
    calls = []

    def blocked(request, on_progress=None, cancellation=None):
        calls.append(request)
        entered.set()
        assert finish.wait(5)
        cancellation.raise_if_cancelled()
        raise AssertionError("cancellation must be observed")

    monkeypatch.setattr(fake, "run", blocked)
    body = {"operation_id": "one", "selected_ids": ["c1"]}
    assert test_client.post(f"/jobs/{job_id}/confirm", json=body).status_code == 200
    assert entered.wait(5)
    try:
        assert test_client.post(f"/jobs/{job_id}/confirm", json=body).status_code == 200
        assert len(calls) == 1
        for _ in range(2):
            assert test_client.post(f"/jobs/{job_id}/cancel").json()["state"] == "cancelling"
        assert _post_video(test_client, video).status_code == 409
        assert test_client.delete("/jobs/current").status_code == 409
    finally:
        finish.set()
    snapshot = _wait_for_state(test_client, job_id, "preview_ready")
    assert snapshot["overrides"]["selected_ids"] == ["c1"]
    assert not snapshot["result_path"]


def test_manifest_atomic_failure_preserves_previous_bytes(client, tmp_path, monkeypatch):
    test_client, _ = client
    video = tmp_path / "atomic.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    manifest = Path(job.output_dir) / "job.json"
    before = manifest.read_bytes()
    monkeypatch.setattr(jobs.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        job.save()
    assert manifest.read_bytes() == before
    assert not list(Path(job.output_dir).glob(".job.json.*.tmp"))


def test_cache_key_tracks_execution_factors_and_playback_pins(client, tmp_path):
    from copy import deepcopy
    test_client, fake = client
    video = tmp_path / "cache.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    plan = load_wipe_plan(str(Path(job.plan_dir) / "wipe_plan.json"))
    identity = fake._trial_identity()
    original = server_app._trial_key(plan, (0, 4), identity)
    for key in ["backend", "device", "weight_sha256", "precision", "gap", "ref_length", "neighbor_stride", "feather", "encoding", "implementation_sha256"]:
        changed = dict(identity, **{key: "different"})
        assert server_app._trial_key(plan, (0, 4), changed) != original
    assert server_app._trial_key(plan, (1, 4), identity) != original
    changed = deepcopy(plan)
    changed.tracks[0].mask[50, 10] = 0
    assert server_app._trial_key(changed, (0, 4), identity) != original
    # A held browser playback lease excludes its file even after a newer trial.
    for index in range(4):
        test_client.post(f"/jobs/{job_id}/trial", json={"duration_seconds": (index + 1) / 4})
        state = _wait_for_state(test_client, job_id, "preview_ready")
        if index < 3:
            assert test_client.post(f"/jobs/{job_id}/playback", json={
                "lease_id": str(index), "trial_id": state["trial"]["id"],
            }).status_code == 200
    assert len(job.trial_cache) == 3  # fourth result succeeds without caching
    assert all(jobs.contained_path(job.output_dir, row["path"]).is_file() for row in job.trial_cache)


def test_retry_uses_new_run_and_last_success_remains_available(client, tmp_path, monkeypatch):
    test_client, fake = client
    video = tmp_path / "retry.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    original = fake.run
    attempted = []

    def fail(request, **kwargs):
        attempted.append(request.output_dir)
        raise RuntimeError("encoder failure")

    monkeypatch.setattr(fake, "run", fail)
    test_client.post(f"/jobs/{job_id}/confirm", json={"selected_ids": ["c1"]})
    failed = _wait_for_state(test_client, job_id, "error")
    assert failed["overrides"]["selected_ids"] == ["c1"]
    monkeypatch.setattr(fake, "run", original)
    assert test_client.post(f"/jobs/{job_id}/retry", json={"operation_id": "retry-full"}).status_code == 200
    done = _wait_for_state(test_client, job_id, "done")
    assert Path(done["result_path"]).parent != Path(attempted[0])
    assert test_client.get(f"/jobs/{job_id}/download").status_code == 200
    assert test_client.delete("/jobs/current").status_code == 200
    jobs.reset_jobs()
    assert test_client.get("/jobs/current").json()["state"] == "idle"


def test_interrupted_detection_requires_new_review_and_only_restores_last(client, tmp_path):
    test_client, _ = client
    video = tmp_path / "interrupted.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    test_client.patch(f"/jobs/{job_id}/review", json={"expected_revision": 0,
        "overrides": {"selected_ids": ["c2"], "bbox_overrides": {}}})
    job.state, job.phase = "pending", "refine"
    job.plan_dir, job.final_plan_ready = None, False
    job.save()
    jobs.reset_jobs()
    restored = test_client.get("/jobs/current").json()
    assert restored["state"] == "interrupted" and restored["overrides"]["selected_ids"] == ["c2"]
    assert test_client.post(f"/jobs/{job_id}/retry", json={"operation_id": "detect-again"}).status_code == 200
    ready = _wait_for_state(test_client, job_id, "preview_ready")
    assert ready["overrides"] is None
    assert ready["previous_review"]["overrides"]["selected_ids"] == ["c2"]
    assert ready["selected_ids"] == ["c1"]
    assert test_client.post(f"/jobs/{job_id}/retry", json={"operation_id": "detect-again"}).status_code == 200


def test_progress_estimate_requires_comparable_completed_segments(tmp_path, monkeypatch):
    job = jobs.Job("test", "", str(tmp_path))
    tick = iter(range(20))
    monkeypatch.setattr(server_app.time, "perf_counter", lambda: next(tick))
    for index in (25, 50, 75):
        server_app._update_progress(job, ProgressEvent("inpaint", index, 150, "bands=1"))
        assert job.remaining_seconds is None
    server_app._update_progress(job, ProgressEvent("inpaint", 100, 150, "bands=1"))
    assert job.remaining_seconds == [2, 2]
    server_app._update_progress(job, ProgressEvent("inpaint", 125, 150, "bands=0"))
    assert job.remaining_seconds is None
    server_app._update_progress(job, ProgressEvent("encode", 0, 1))
    assert job.remaining_seconds is None and job.completed == 0 and job.total == 1


@pytest.mark.skipif(not shutil.which("node"), reason="requires node")
def test_native_sse_reconnect_and_missing_job_stop():
    html = server_app._web_index().read_text(encoding="utf-8")
    function = html.split("    function startProgressStream(jobId) {", 1)[1].split("    async function restoreJob()", 1)[0]
    harness = r'''
const assert=require("node:assert/strict");
let stream=null,state="pending",missing=false;const messages=[],snapshots=[];
class EventSource {constructor(url){this.url=url;this.closed=false;}close(){this.closed=true;}}
const notify=message=>messages.push(message),applyStatus=data=>snapshots.push(data);
function closeStream(){if(stream){stream.close();stream=null;}}
const request=async()=>{if(missing)throw {status:404,message:"missing"};return {id:"job",state:"pending"};};
const settle=()=>new Promise(resolve=>setImmediate(resolve));
'''
    assertions = r'''
(async()=>{
startProgressStream("job");const first=stream;
first.onerror();await settle();assert.equal(first.closed,false);assert.equal(stream,first);
first.onopen();await settle();assert.equal(snapshots.length,1);
state="preview_ready";const before=messages.length;first.onerror();assert.equal(messages.length,before);
state="pending";missing=true;first.onerror();await settle();assert.equal(first.closed,true);assert.equal(stream,null);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run(["node", "-e", harness + "function startProgressStream(jobId) {" + function + assertions], check=True)


def test_revision_protection_mask_restore_and_result_marks(client, tmp_path):
    test_client, fake = client
    video = tmp_path / 'revision.mp4'; _write_test_video(video)
    job_id = _post_video(test_client, video).json()['id']
    ready = _wait_for_state(test_client, job_id, 'preview_ready')
    edits = {'selected_ids': ['c1'], 'bbox_overrides': {},
             'segment_overrides': {'c1': [[4, 8], [0, 3], [2, 5]]},
             'protections': [{'id': 'p_one', 'bbox': [20, 45, 30, 63], 'segments': [[0, 1], [7, 8]]}]}
    saved = test_client.patch(f'/jobs/{job_id}/review', json={'expected_revision': ready['review_revision'], 'overrides': edits})
    assert saved.status_code == 200
    snapshot = saved.json(); revision = snapshot['review_revision']
    assert snapshot['overrides']['segment_overrides']['c1'] == [[0, 8]]
    response = test_client.get(f'/jobs/{job_id}/mask?revision={revision}&frame_index=0')
    assert response.status_code == 200
    alpha = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert not alpha[45:64, 20:31].any() and alpha[52, 40] == 255
    assert test_client.get(f'/jobs/{job_id}/mask?revision={revision-1}&frame_index=0').status_code == 409
    assert test_client.get(f'/jobs/{job_id}/mask?revision={revision}&frame_index=8').status_code == 400
    assert test_client.post(f'/jobs/{job_id}/confirm', json={**edits, 'expected_revision': revision}).status_code == 200
    done = _wait_for_state(test_client, job_id, 'done')
    assert done['result_revision'] and fake.calls[-1]['plan'].schema_version == 2
    data = test_client.get(f'/jobs/{job_id}/result-review').json()
    assert len(data['windows']) <= 12 and data['result_revision'] == done['result_revision']
    mark = {'result_revision': done['result_revision'], 'frame_index': 3, 'status': 'issue'}
    assert test_client.post(f'/jobs/{job_id}/result-review', json=mark).status_code == 200
    assert test_client.post(f'/jobs/{job_id}/result-review', json={**mark, 'result_revision': 'stale'}).status_code == 409
    old_result = jobs.get_job(job_id).result_path
    # Saving decisions on a completed job still allows another export.
    assert test_client.post(f'/jobs/{job_id}/confirm', json=edits).status_code == 200
    newer = _wait_for_state(test_client, job_id, 'done')
    assert newer['result_revision'] != done['result_revision']
    assert newer['review_marks'] == [{'frame_index': 3, 'status': 'issue'}]
    assert Path(old_result).is_file()
    jobs.JOBS.clear(); jobs._current_job = None
    restored = test_client.get('/jobs/current').json()
    assert restored['result_revision'] == newer['result_revision']
    assert restored['overrides']['protections'] == edits['protections']
    assert test_client.get(f'/jobs/{job_id}/mask?revision={restored["review_revision"]}&frame_index=7').status_code == 200


def test_empty_effective_removal_and_invalid_edits_do_not_execute(client, tmp_path):
    test_client, fake = client
    video = tmp_path / 'empty.mp4'; _write_test_video(video)
    job_id = _post_video(test_client, video).json()['id']
    _wait_for_state(test_client, job_id, 'preview_ready')
    edits = {'selected_ids': ['c1'], 'protections': [
        {'id': 'p_all', 'bbox': [0, 0, 95, 63], 'segments': [[0, 8]]}]}
    assert test_client.post(f'/jobs/{job_id}/confirm', json=edits).status_code == 400
    assert test_client.post(f'/jobs/{job_id}/trial', json=edits).status_code == 400
    assert not any(c['method'] == 'run' for c in fake.calls)
    for intervals in ([], [[0, 0]], [[-1, 2]], [[0, 9]], [[True, 2]], [[1.2, 3]]):
        response = test_client.patch(f'/jobs/{job_id}/review', json={
            'expected_revision': 0, 'overrides': {'selected_ids': ['c1'], 'segment_overrides': {'c1': intervals}}})
        assert response.status_code in (400, 422)
    assert test_client.post(f'/jobs/{job_id}/confirm', json={}).status_code == 200
    _wait_for_state(test_client, job_id, 'done')
    directory = Path(jobs.get_job(job_id).output_dir) / 'predictions'; directory.mkdir()
    (directory / ('a'*64 + '.npz')).write_bytes(b'cache')
    (directory / 'keep.txt').write_text('unrelated')
    temporary = directory / ('.' + 'b'*32 + '.tmp'); temporary.write_bytes(b'orphan')
    assert test_client.delete(f'/jobs/{job_id}/prediction-cache').status_code == 200
    assert (directory / 'keep.txt').is_file() and not list(directory.glob('*.npz')) and not temporary.exists()


def test_editor_seconds_roundtrip_preserves_frame_boundaries():
    from types import SimpleNamespace
    from videowipe.server.review import seconds_to_frames
    for fps in (25, 29.97, 30000/1001):
        source = SimpleNamespace(fps=fps, frame_count=1000)
        for first in range(999):
            assert seconds_to_frames(first/fps, (first+1)/fps, source) == (first, first+1)
        assert seconds_to_frames(1.25/fps, 4.75/fps, source) == (1, 5)
    if shutil.which('node'):
        html = server_app._web_index().read_text(encoding="utf-8")
        function = html.split('    function parseIntervals(text) {', 1)[1].split('    function edited()', 1)[0]
        code = 'const assert=require("node:assert/strict");let source={fps:25,frame_count:1000};const duration=()=>source.frame_count/source.fps;'
        code += 'function parseIntervals(text) {' + function
        code += 'for(const fps of [25,29.97,30000/1001]){source.fps=fps;for(let n=0;n<999;n++)assert.deepEqual(parseIntervals(`${n/fps}, ${(n+1)/fps}`),[[n,n+1]]);}'
        subprocess.run(['node', '-e', code], check=True)


def test_interrupted_trial_retry_keeps_latest_interval_and_never_exports_full(client, tmp_path, monkeypatch):
    test_client, _ = client
    video = tmp_path/'trial-retry.mp4'
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()['id']
    _wait_for_state(test_client, job_id, 'preview_ready')
    test_client.post(f'/jobs/{job_id}/trial', json={'start_seconds': 0, 'duration_seconds': .25})
    old = _wait_for_state(test_client, job_id, 'preview_ready')['trial']
    # Persist a different pending request, then emulate loss of its worker.
    monkeypatch.setattr(server_app.threading.Thread, 'start', lambda self: None)
    test_client.post(f'/jobs/{job_id}/trial', json={'start_seconds': .5, 'duration_seconds': .5})
    jobs.reset_jobs()
    restored = test_client.get('/jobs/current').json()
    assert restored['state'] == 'interrupted' and restored['trial']['id'] == old['id']
    calls = []
    def trial(job_id, body):
        calls.append(body)
        return {'retried': 'trial'}
    monkeypatch.setattr(server_app, 'trial', trial)
    monkeypatch.setattr(server_app, 'confirm', lambda *args: pytest.fail('trial retry launched full export'))
    assert server_app.retry(job_id, server_app.RetryRequest(operation_id='retry-trial')) == {'retried': 'trial'}
    assert len(calls) == 1 and calls[0].start_seconds == .5 and calls[0].duration_seconds == .5
    assert calls[0].operation_id == 'retry-trial'
