"""FastAPI app for the local-first videowipe web UI."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, StrictInt

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


def _load_tracks(job: Job) -> list[dict]:
    """Return validated metadata-only tracks from the job's WipePlan."""
    path = Path(job.output_dir) / JSON_FILENAME
    if not path.exists():
        return []
    plan = load_wipe_plan(str(path), load_masks=False)
    return [track.to_dict() for track in plan.tracks]


def _run_preview(job: Job, intent: str | None) -> None:
    try:
        started = time.perf_counter()
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
            job.warnings = list(plan.warnings)
            job.timings["plan_s"] = time.perf_counter() - started
            job.state = "preview_ready"
    except Exception as exc:
        _set_error(job, exc)


def _run_inpaint(job: Job) -> None:
    try:
        plan_path = str(Path(job.output_dir) / JSON_FILENAME)
        started = time.perf_counter()
        result = _get_engine().run(
            WipeRequest(
                video=job.video_path,
                output_dir=job.output_dir,
                plan=plan_path,
            ),
            on_progress=lambda event: _update_progress(job, event),
        )
        with job.lock:
            job.result_path = result.output_path
            job.progress = 1.0
            job.phase = "complete"
            job.warnings = list(result.warnings)
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

    suffix = Path(video.filename or "").suffix or ".mp4"
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
    if snapshot["state"] == "error":
        raise HTTPException(status_code=409, detail=snapshot["error"])
    if snapshot["state"] != "preview_ready":
        raise HTTPException(status_code=409, detail="preview is not ready")
    try:
        candidates = _load_candidates(job)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "id": job.id,
        "state": snapshot["state"],
        "candidates": candidates,
        "tracks": _load_tracks(job),
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


@app.post("/jobs/{job_id}/confirm")
def confirm(job_id: str, body: ConfirmRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if job.state != "preview_ready":
            raise HTTPException(status_code=409, detail=f"job is {job.state}")
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
        save_wipe_plan(plan, job.output_dir)
        job.selected_ids = selected_ids
        job.progress = 0.0
        job.phase = "prepare"
        job.state = "running"
    threading.Thread(target=_run_inpaint, args=(job,), daemon=True).start()
    return job.snapshot()


@app.get("/jobs/{job_id}/progress")
def progress_sse(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    def _events():
        while True:
            snapshot = job.snapshot()
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["state"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.5)

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    snapshot = job.snapshot()
    if snapshot["state"] != "done":
        raise HTTPException(status_code=409, detail="job is not done")
    result_path = snapshot["result_path"]
    if not result_path or not os.path.exists(result_path):
        matches = sorted(Path(job.output_dir).glob("*_clean.mp4"))
        result_path = str(matches[-1]) if matches else None
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="result video not found")
    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=Path(result_path).name,
    )
