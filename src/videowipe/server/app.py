"""FastAPI app for the local-first videowipe web UI."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from videowipe.engine import WipeEngine
from videowipe.plan import JSON_FILENAME, load_wipe_plan, save_wipe_plan
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
        job.error = str(exc)
    release_job(job.id)


def _load_candidates(job: Job) -> list[dict]:
    path = Path(job.output_dir) / "clean_candidates.json"
    if not path.exists():
        raise FileNotFoundError("preview candidates are not ready")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("candidates", [])


def _load_tracks(job: Job) -> list[dict]:
    """Metadata-only track view of the plan for the UI; empty if absent.

    Read as plain JSON (no SHA/mask validation) — this is display only. The
    plan is validated when it is executed, not when it is shown.
    """
    path = Path(job.output_dir) / JSON_FILENAME
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("tracks", [])


def _run_preview(job: Job, intent: str | None) -> None:
    try:
        _get_engine().process(
            video=job.video_path,
            output=job.output_dir,
            intent=intent or None,
            preview=True,
        )
        candidates = _load_candidates(job)
        # WipePlan is the single source of truth for remove/keep once it
        # exists. Derive the default confirmation selection from its track
        # actions so a stale clean_candidates.json (whose `selected` field
        # predates the plan) cannot silently flip a safety-kept track — e.g.
        # a persistent top logo the plan keeps — to remove on a no-body
        # confirm. The candidate `selected` field stays as a fallback for
        # jobs that predate the plan. Check file existence, not `if tracks`,
        # so a present-but-empty plan is not mistaken for an old job.
        plan_path = Path(job.output_dir) / JSON_FILENAME
        if plan_path.exists():
            default_selected = [
                track["id"]
                for track in _load_tracks(job)
                if track.get("action") == "remove"
            ]
        else:
            default_selected = [
                candidate["id"]
                for candidate in candidates
                if candidate.get("selected")
            ]
        with job.lock:
            job.default_selected_ids = default_selected
            job.selected_ids = list(default_selected)
            job.progress = 0.0
            job.state = "preview_ready"
    except Exception as exc:
        _set_error(job, exc)


def _run_inpaint(job: Job) -> None:
    try:
        with job.lock:
            selected_ids = list(job.selected_ids)

        # Phase A wrote wipe_plan.json + .npz during preview. Confirm overrides
        # each track's action from the selection (chosen ids → remove, the rest
        # → keep), persists the confirmed plan, and executes it. This uses the
        # plan's precise per-track NPZ masks and temporal segments instead of
        # reconstructing a mask from bboxes, so a changed selection no longer
        # degrades to an approximation and the default selection runs temporally
        # (closing subtitle-gap false erasures on the web path too).
        #
        # Load without video_path: load_wipe_plan otherwise enforces
        # require_remove on the DEFAULT plan (before the selection is applied),
        # which would reject the legitimate case of toggling a track to remove
        # when the default plan is all-keep (e.g. a video whose only overlay is
        # a safety-kept logo). engine.process re-validates the mutated plan and
        # re-derives the source SHA, so video binding and NPZ integrity are
        # preserved; load_wipe_plan still verifies the NPZ sha here.
        plan = load_wipe_plan(str(Path(job.output_dir) / JSON_FILENAME))
        selected = set(selected_ids)
        for track in plan.tracks:
            track.action = "remove" if track.id in selected else "keep"
            track.decision_reason = f"user-confirm:{track.action}"
        save_wipe_plan(plan, job.output_dir)

        def _progress(done: int, total: int) -> None:
            with job.lock:
                job.progress = done / total if total else 0.0

        result_path = _get_engine().process(
            video=job.video_path,
            plan=plan,
            output=job.output_dir,
            progress=_progress,
        )
        with job.lock:
            job.result_path = result_path
            job.progress = 1.0
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
        with input_path.open("wb") as fh:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        with job.lock:
            job.video_path = str(input_path)
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


@app.post("/jobs/{job_id}/confirm")
def confirm(job_id: str, body: ConfirmRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if job.state != "preview_ready":
            raise HTTPException(status_code=409, detail=f"job is {job.state}")
        selected_ids = (
            list(body.selected_ids)
            if body.selected_ids is not None
            else list(job.default_selected_ids)
        )
        if not selected_ids:
            raise HTTPException(status_code=400, detail="select at least one target")
        known_ids = {candidate["id"] for candidate in _load_candidates(job)}
        unknown_ids = sorted(set(selected_ids) - known_ids)
        if unknown_ids:
            raise HTTPException(
                status_code=400,
                detail=f"unknown candidate id: {', '.join(unknown_ids)}",
            )
        job.selected_ids = selected_ids
        job.progress = 0.0
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
