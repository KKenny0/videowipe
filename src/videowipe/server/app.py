"""FastAPI app for the local-first videowipe web UI."""
from __future__ import annotations

import json
import math
import mimetypes
import uuid
import os
import subprocess
import threading
import time
import unicodedata
from pathlib import Path
from fractions import Fraction

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, StrictInt

from videowipe.api import ProgressEvent, WipeRequest
from videowipe.engine import WipeEngine
from videowipe.errors import InvalidInputError
from videowipe.plan import (
    JSON_FILENAME,
    load_wipe_plan,
    save_wipe_plan,
    validate_plan,
)
from videowipe.server.jobs import (
    Job,
    JobBusy,
    JobNotCancellable,
    cancel_current_job,
    create_job,
    get_current_job,
    get_job,
    release_job,
)

app = FastAPI(title="videowipe")

_engine: WipeEngine | None = None
_engine_lock = threading.Lock()


class ConfirmRequest(BaseModel):
    selected_ids: list[str] | None = None
    bbox_overrides: dict[
        str, tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    ] | None = None


class TrialRequest(ConfirmRequest):
    start_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    duration_seconds: float = Field(default=3, gt=0, le=5, allow_inf_nan=False)


def _jobs_root() -> str:
    return os.environ.get("VIDEOWIPE_JOBS_DIR", "jobs")


def _web_index() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "index.html"


def _get_engine() -> WipeEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = WipeEngine(task="clean")
        return _engine


def _set_error(job: Job, exc: Exception) -> None:
    with job.lock:
        job.state = "error"
        job.phase = "error"
        job.error = str(exc)
    release_job(job.id)


def _update_progress(job: Job, event: ProgressEvent) -> None:
    with job.lock:
        job.phase = event.phase
        if event.fraction is not None:
            job.progress = event.fraction


def _load_candidates(job: Job) -> list[dict]:
    path = Path(job.output_dir) / "clean_candidates.json"
    if not path.exists():
        raise FileNotFoundError("preview candidates are not ready")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("candidates", [])


def _display_filename(name: str | None) -> str:
    name = (name or "input.mp4").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if not unicodedata.category(c).startswith("C"))
    return name.strip(" .")[:120].encode("utf-8")[:180].decode("utf-8", errors="ignore") or "input.mp4"


def _job_path(job: Job, path: str | Path) -> Path:
    root = Path(job.output_dir).resolve()
    resolved = Path(path).resolve()
    if not root.is_relative_to(Path(_jobs_root()).resolve()) or not resolved.is_relative_to(root):
        raise HTTPException(status_code=409, detail="media path escapes the job")
    return resolved


def _job_file(job: Job, path: str | Path) -> Path:
    resolved = _job_path(job, path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="media file not found")
    return resolved


def _trial_range(plan, start: float, duration: float) -> tuple[int, int]:
    fps = plan.source.fps
    if start >= plan.source.frame_count / fps:
        raise HTTPException(status_code=400, detail="trial starts outside the video")
    first = math.floor(start * fps)
    last = min(plan.source.frame_count, math.ceil((start + duration) * fps))
    if not 0 <= first < last <= plan.source.frame_count:
        raise HTTPException(status_code=400, detail="trial starts outside the video")
    if not any(
        track.mask is not None and track.mask.any()
        and any(s.start < last and s.end > first for s in track.segments)
        for track in plan.remove_tracks
    ):
        raise HTTPException(status_code=400, detail="No selected target is active in this interval")
    return first, last


def _recommended_trial(plan) -> dict | None:
    intervals = sorted(
        (s.start, s.end) for track in plan.remove_tracks
        if track.mask is not None and track.mask.any() for s in track.segments
    )
    merged: list[list[int]] = []
    for first, last in intervals:
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    if not merged:
        return None
    first, last = min(merged, key=lambda row: (-(row[1] - row[0]), row[0]))
    fps = plan.source.fps
    duration = min(3.0, plan.source.frame_count / fps)
    start = max(0.0, min(plan.source.frame_count / fps - duration,
                         (first + last) / (2 * fps) - duration / 2))
    frame_range = _trial_range(plan, start, duration)
    return {"start_seconds": start, "duration_seconds": duration,
            "frame_range": list(frame_range)}


def _run_preview(job: Job, intent: str | None) -> None:
    try:
        started = time.perf_counter()
        try:
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=color_transfer,r_frame_rate,avg_frame_rate", "-of", "json", job.video_path,
            ], timeout=30, stderr=subprocess.DEVNULL))["streams"][0]
            if probe.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
                job.input_warnings.append("HDR 视频不在本次 SDR 清理验收范围内，请检查导出颜色。")
            nominal = float(Fraction(probe.get("r_frame_rate", "0")))
            average = float(Fraction(probe.get("avg_frame_rate", "0")))
            if nominal and average and not math.isclose(nominal, average, rel_tol=0.001):
                job.input_warnings.append("视频帧率可能不恒定；精确时间定位只对恒定帧率视频验收。")
        except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError, subprocess.SubprocessError):
            job.input_warnings.append("未能核实视频帧率与色彩信息，请检查试擦和导出效果。")
        plan = _get_engine().plan(
            WipeRequest(
                video=job.video_path,
                output_dir=job.output_dir,
                intent=intent or None,
                preview=True,
            ),
            on_progress=lambda event: _update_progress(job, event),
        )
        default_selected = [track.id for track in plan.remove_tracks]
        with job.lock:
            job.default_selected_ids = default_selected
            job.selected_ids = list(default_selected)
            job.progress = 0.0
            job.phase = "preview"
            job.warnings = job.input_warnings + list(plan.warnings)
            job.timings["plan_s"] = time.perf_counter() - started
            job.state = "preview_ready"
    except Exception as exc:
        _set_error(job, exc)


def _run_inpaint(job: Job, run_dir: Path) -> None:
    try:
        plan_path = str(run_dir / JSON_FILENAME)
        started = time.perf_counter()
        result = _get_engine().run(
            WipeRequest(
                video=job.video_path,
                output_dir=str(run_dir),
                plan=plan_path,
            ),
            on_progress=lambda event: _update_progress(job, event),
        )
        _job_file(job, result.output_path)
        with job.lock:
            job.result_path = result.output_path
            job.progress = 1.0
            job.phase = "complete"
            job.warnings = job.input_warnings + list(result.warnings)
            job.timings.update(result.timings)
            job.timings["run_wall_s"] = time.perf_counter() - started
            job.state = "done"
    except Exception as exc:
        _set_error(job, exc)
        return
    release_job(job.id)


@app.get("/")
def index():
    return FileResponse(_web_index())


@app.post("/jobs")
async def create(
    video: UploadFile = File(...),
    intent: str | None = Form(None),
):
    try:
        job = create_job(output_base=_jobs_root())
    except JobBusy:
        raise HTTPException(status_code=409, detail="server busy, wait for current job")

    original_filename = _display_filename(video.filename)
    suffix = Path(original_filename).suffix.lower()
    if not suffix[1:].isascii() or not suffix[1:].isalnum() or len(suffix) > 10:
        suffix = ".mp4"
    input_path = Path(job.output_dir) / f"input{suffix}"
    try:
        started = time.perf_counter()
        with input_path.open("wb") as fh:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        with job.lock:
            job.video_path = str(input_path)
            job.original_filename = original_filename
            job.phase = "plan"
            job.timings["upload_s"] = time.perf_counter() - started
        threading.Thread(target=_run_preview, args=(job, intent), daemon=True).start()
        return job.snapshot()
    except Exception as exc:
        _set_error(job, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/jobs/current")
def current_job():
    job = get_current_job()
    if job is None:
        return {"state": "idle"}
    return job.snapshot()


@app.delete("/jobs/current")
def cancel_current():
    try:
        job = cancel_current_job()
    except JobNotCancellable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        return {"state": "idle"}
    return job.snapshot()


@app.get("/jobs/{job_id}")
def status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.snapshot()


@app.get("/jobs/{job_id}/preview")
def preview(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    snapshot = job.snapshot()
    if snapshot["state"] == "error" and not (Path(job.output_dir) / JSON_FILENAME).exists():
        raise HTTPException(status_code=409, detail=snapshot["error"])
    if snapshot["state"] == "pending":
        raise HTTPException(status_code=409, detail="preview is not ready")
    try:
        candidates = _load_candidates(job)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    plan = load_wipe_plan(str(Path(job.output_dir) / JSON_FILENAME))
    by_id = {candidate["id"]: candidate for candidate in candidates}
    tracks = []
    for track in plan.tracks:
        evidence = sorted(n for n in by_id.get(track.id, {}).get("presence_frames", [])
                          if type(n) is int and 0 <= n < plan.source.frame_count)
        row = track.to_dict()
        row["has_mask"] = bool(track.mask is not None and track.mask.any())
        row["evidence_frame"] = evidence[0] if evidence else 0
        row["evidence_kind"] = "observed" if evidence else "position_reference"
        row["evidence_url"] = f"/jobs/{job.id}/frame?frame_index={row['evidence_frame']}"
        tracks.append(row)
    return {
        "id": job.id,
        "state": snapshot["state"],
        "candidates": candidates,
        "tracks": tracks,
        "source": plan.source.to_dict(),
        "original_filename": snapshot["original_filename"],
        "source_url": f"/jobs/{job.id}/source-video",
        "recommended_trial": _recommended_trial(plan),
        "confirmed_review": snapshot["confirmed_review"],
        "trial": snapshot["trial"],
        "preview_url": f"/jobs/{job.id}/preview-image",
        "editable_preview_url": f"/jobs/{job.id}/editable-preview-image",
        "default_selected_ids": snapshot["default_selected_ids"],
    }


@app.get("/jobs/{job_id}/preview-image")
def preview_image(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = Path(job.output_dir) / "clean_preview.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="preview image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/editable-preview-image")
def editable_preview_image(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = Path(job.output_dir) / "clean_preview_source.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="editable preview image not found")
    return FileResponse(path, media_type="image/jpeg")


def _reviewed_plan(job: Job, body: ConfirmRequest):
    try:
        plan = load_wipe_plan(str(Path(job.output_dir) / JSON_FILENAME))
    except (InvalidInputError, OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="job plan is invalid") from exc
    selected_ids = (
        list(body.selected_ids)
        if body.selected_ids is not None
        else [track.id for track in plan.remove_tracks]
    )
    if not selected_ids:
        raise HTTPException(status_code=400, detail="select at least one target")
    known_ids = {track.id for track in plan.tracks}
    unknown_ids = sorted(set(selected_ids) - known_ids)
    if unknown_ids:
        raise HTTPException(
            status_code=400,
            detail=f"unknown candidate id: {', '.join(unknown_ids)}",
        )
    selected = set(selected_ids)
    overrides = body.bbox_overrides or {}
    unknown_override_ids = sorted(set(overrides) - known_ids)
    if unknown_override_ids:
        raise HTTPException(
            status_code=400,
            detail=f"unknown bbox override id: {', '.join(unknown_override_ids)}",
        )
    unselected_override_ids = sorted(set(overrides) - selected)
    if unselected_override_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "bbox override requires selected target: "
                f"{', '.join(unselected_override_ids)}"
            ),
        )
    validated_overrides: dict[str, tuple[int, int, int, int]] = {}
    for track_id, bbox in overrides.items():
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            raise HTTPException(
                status_code=400,
                detail=f"bbox override for {track_id} is inverted or empty",
            )
        if x2 - x1 + 1 < 2 or y2 - y1 + 1 < 2:
            raise HTTPException(
                status_code=400,
                detail=f"bbox override for {track_id} must be at least 2x2 pixels",
            )
        if x1 < 0 or y1 < 0 or x2 >= plan.source.width or y2 >= plan.source.height:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"bbox override for {track_id} exceeds source "
                    f"{plan.source.width}x{plan.source.height}"
                ),
            )
        validated_overrides[track_id] = (x1, y1, x2, y2)

    for track in plan.tracks:
        track.action = "remove" if track.id in selected else "keep"
        track.decision_reason = f"user-confirm:{track.action}"
        if track.id in validated_overrides:
            x1, y1, x2, y2 = validated_overrides[track.id]
            mask = np.zeros(
                (plan.source.height, plan.source.width), dtype=np.uint8
            )
            mask[y1:y2 + 1, x1:x2 + 1] = 1
            track.bbox = (x1, y1, x2, y2)
            track.mask = mask
            track.decision_reason = "user-confirm:remove:bbox-override"
    try:
        validate_plan(plan, require_remove=True)
    except InvalidInputError as exc:
        raise HTTPException(status_code=409, detail="job plan is invalid") from exc
    return plan, selected_ids


@app.post("/jobs/{job_id}/confirm")
def confirm(job_id: str, body: ConfirmRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if job.state != "preview_ready":
            raise HTTPException(status_code=409, detail=f"job is {job.state}")
        plan, selected_ids = _reviewed_plan(job, body)
        run_dir = _job_path(job, Path(job.output_dir) / "runs" / uuid.uuid4().hex)
        save_wipe_plan(plan, str(run_dir))
        job.selected_ids = selected_ids
        job.confirmed_review = {"selected_ids": selected_ids, "bbox_overrides": body.bbox_overrides or {}}
        job.error = None
        job.trial = None
        job.trial_path = None
        job.progress = 0.0
        job.phase = "prepare"
        job.state = "running"
    threading.Thread(target=_run_inpaint, args=(job, run_dir), daemon=True).start()
    return job.snapshot()


def _run_trial(job: Job, plan, frame_range: tuple[int, int], trial_dir: Path) -> None:
    started = time.perf_counter()
    try:
        save_wipe_plan(plan, str(trial_dir))
        result = _get_engine().run(
            WipeRequest(
                video=job.video_path, output_dir=str(trial_dir),
                plan=str(trial_dir / JSON_FILENAME), trial_range=frame_range,
            ),
            on_progress=lambda event: _update_progress(job, event),
        )
        with job.lock:
            job.trial_path = result.output_path
            job.trial["ready"] = True
            job.trial["elapsed_s"] = round(time.perf_counter() - started, 3)
            job.trial["backend"] = result.backend
            job.warnings = job.input_warnings + list(result.warnings)
            job.progress = 1.0
    except Exception as exc:
        with job.lock:
            job.error = str(exc)
            job.trial = None
            job.trial_path = None
    finally:
        with job.lock:
            job.state = "preview_ready"
            job.phase = "preview"


@app.post("/jobs/{job_id}/trial")
def trial(job_id: str, body: TrialRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if job.state != "preview_ready":
            raise HTTPException(status_code=409, detail=f"job is {job.state}")
        plan, _selected_ids = _reviewed_plan(job, body)
        fps = plan.source.fps
        first, last = _trial_range(plan, body.start_seconds, body.duration_seconds)
        trial_id = uuid.uuid4().hex
        trial_dir = _job_path(job, Path(job.output_dir) / "trials" / trial_id)
        job.trial = {
            "id": trial_id, "ready": False, "request": body.model_dump(),
            "frame_range": [first, last], "start_seconds": first / fps,
            "duration_seconds": (last - first) / fps,
        }
        job.trial_path = None
        job.error = None
        job.progress = 0.0
        job.phase = "trial"
        job.state = "trial_running"
    threading.Thread(
        target=_run_trial, args=(job, plan, (first, last), trial_dir), daemon=True,
    ).start()
    return job.snapshot()


@app.get("/jobs/{job_id}/trial-video")
def trial_video(job_id: str, trial_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if (not job.trial or not job.trial["ready"]
                or job.trial["id"] != trial_id or not job.trial_path):
            raise HTTPException(status_code=409, detail="trial is not available")
        path = job.trial_path
    return FileResponse(_job_file(job, path), media_type="video/mp4", headers={"Cache-Control": "no-store"})


@app.get("/jobs/{job_id}/progress")
def progress_sse(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    def _events():
        while True:
            snapshot = job.snapshot()
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["state"] in {"done", "error", "cancelled", "preview_ready"}:
                break
            time.sleep(0.5)

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return FileResponse(
        _result_file(job), media_type="video/mp4",
        filename=f"{Path(job.original_filename).stem}_clean.mp4",
    )


def _result_file(job: Job) -> Path:
    snapshot = job.snapshot()
    if snapshot["state"] != "done":
        raise HTTPException(status_code=409, detail="job is not done")
    result_path = snapshot["result_path"]
    if not result_path:
        raise HTTPException(status_code=404, detail="result video not found")
    return _job_file(job, result_path)


@app.get("/jobs/{job_id}/source-video")
def source_video(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = _job_file(job, job.video_path)
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if not media_type.startswith("video/"):
        media_type = "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"X-Content-Type-Options": "nosniff"})


@app.get("/jobs/{job_id}/result-video")
def result_video(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return FileResponse(_result_file(job), media_type="video/mp4")


@app.get("/jobs/{job_id}/frame")
def source_frame(job_id: str, frame_index: int = Query(ge=0)):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    source = _job_file(job, job.video_path)
    if not (Path(job.output_dir) / JSON_FILENAME).is_file():
        raise HTTPException(status_code=409, detail="preview is not ready")
    plan = load_wipe_plan(str(Path(job.output_dir) / JSON_FILENAME), load_masks=False)
    if frame_index >= plan.source.frame_count:
        raise HTTPException(status_code=400, detail="frame is outside the video")
    # Only evidence frames are cached: a free scrub must not fill the disk.
    evidence_frames = {0}
    for candidate in _load_candidates(job):
        present = [n for n in candidate.get("presence_frames", [])
                   if type(n) is int and 0 <= n < plan.source.frame_count]
        if present:
            evidence_frames.add(min(present))
    with job.media_lock:
        cache = Path(job.output_dir) / f"evidence-{frame_index}.jpg"
        if frame_index in evidence_frames and cache.exists():
            return FileResponse(_job_file(job, cache), media_type="image/jpeg")
        reader = cv2.VideoCapture(str(source))
        try:
            # ponytail: decode from zero for exact frame identity; index keyframes if long-video evidence becomes slow.
            for _ in range(frame_index):
                if not reader.grab():
                    raise HTTPException(status_code=422, detail="cannot decode requested frame")
            ok, frame = reader.read()
            if not ok:
                raise HTTPException(status_code=422, detail="cannot decode requested frame")
        finally:
            reader.release()
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=422, detail="cannot encode requested frame")
        if frame_index in evidence_frames:
            if cache.is_symlink():
                raise HTTPException(status_code=409, detail="unsafe evidence cache")
            temporary = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(cache)
            return FileResponse(cache, media_type="image/jpeg")
        return Response(encoded.tobytes(), media_type="image/jpeg")
