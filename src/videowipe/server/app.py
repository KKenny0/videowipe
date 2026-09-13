"""FastAPI app for the local-first videowipe web UI."""
from __future__ import annotations

import json
import hashlib
import math
import mimetypes
import re
import uuid
import os
import subprocess
import threading
import time
import unicodedata
from pathlib import Path
from fractions import Fraction
from contextlib import asynccontextmanager, contextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, StrictInt

from videowipe.api import CancellationToken, ProgressEvent, WipeRequest
from videowipe.engine import WipeEngine
from videowipe.errors import InvalidInputError, ProcessingCancelledError
from videowipe.plan import (
    JSON_FILENAME,
    load_wipe_plan,
    save_wipe_plan,
    validate_plan,
    execution_masks,
)
from videowipe.server.review import compile_review, seconds_to_frames, review_windows
from videowipe.server.jobs import (
    Job,
    JobBusy,
    JobNotCancellable,
    cancel_current_job,
    create_job,
    get_current_job,
    get_job,
    release_job,
    reserve_job,
    restore_last,
    publish_last,
    source_hash,
    relative_path,
    contained_path,
    atomic_json,
)

@asynccontextmanager
async def _lifespan(_app):
    try:
        restore_last(_jobs_root())
    except (OSError, ValueError, KeyError, TypeError, InvalidInputError):
        pass  # /jobs/current returns the concrete recovery error to the UI.
    yield


app = FastAPI(title="videowipe", lifespan=_lifespan)

_engine: WipeEngine | None = None
_engine_lock = threading.Lock()


class ProtectionRequest(BaseModel):
    id: str = Field(pattern=r"^p_[a-zA-Z0-9_-]{1,64}$")
    bbox: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    segments: list[tuple[StrictInt, StrictInt]] = Field(min_length=1, max_length=512)


class ConfirmRequest(BaseModel):
    expected_revision: StrictInt | None = Field(default=None, ge=0)
    operation_id: str | None = Field(default=None, min_length=1, max_length=128)
    selected_ids: list[str] | None = None
    bbox_overrides: dict[
        str, tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    ] | None = None


    segment_overrides: dict[str, list[tuple[StrictInt, StrictInt]]] | None = None
    protections: list[ProtectionRequest] | None = Field(default=None, max_length=64)


class TrialRequest(ConfirmRequest):
    start_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    duration_seconds: float = Field(default=3, gt=0, le=5, allow_inf_nan=False)


class ReviewRequest(BaseModel):
    expected_revision: StrictInt = Field(ge=0)
    overrides: ConfirmRequest


class RetryRequest(BaseModel):
    operation_id: str = Field(min_length=1, max_length=128)


class PlaybackRequest(BaseModel):
    lease_id: str = Field(min_length=1, max_length=128)
    trial_id: str | None = None


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
        cancelled = isinstance(exc, ProcessingCancelledError)
        job.state = ("preview_ready" if job.final_plan_ready else "cancelled") if cancelled else "error"
        job.phase = "preview" if job.state == "preview_ready" else job.state
        job.error = "已取消，已保存的审阅决定保留。" if cancelled else str(exc)
        try:
            job.save()
        except (OSError, ValueError) as save_error:
            job.state = job.phase = "error"
            job.error += f"；任务状态未能保存：{save_error}"
    if job.state != "preview_ready":
        release_job(job.id)


def _update_progress(job: Job, event: ProgressEvent) -> None:
    with job.lock:
        now = time.perf_counter()
        if job.phase != event.phase:
            if {job.phase, event.phase} != {"predict", "compose"}:
                job.progress_samples = []
            job.remaining_seconds = None
        job.phase = event.phase
        job.phase_counts[event.phase] = [event.completed, event.total]
        job.completed, job.total = event.completed, event.total
        if event.fraction is not None:
            job.progress = event.fraction
        # Compose endpoints include the intervening prediction work, including
        # cache reads. Sampling both phases would count each batch twice.
        if event.phase in {"inpaint", "compose"} and event.completed > 0:
            if job.progress_samples and (job.progress_samples[-1][2] != event.message
                                         or job.progress_samples[-1][1] >= event.completed):
                job.progress_samples = []
                job.remaining_seconds = None
            job.progress_samples.append((now, event.completed, event.message))
            job.progress_samples = job.progress_samples[-32:]
            samples = job.progress_samples
            if len(samples) >= 4:
                steps = [n1 - n0 for (_, n0, _), (_, n1, _) in zip(samples, samples[1:]) if n1 > n0]
                comparable = max(set(steps), key=steps.count) if steps else 0
                rates = [(t1 - t0) / (n1 - n0)
                         for (t0, n0, _), (t1, n1, message) in zip(samples, samples[1:])
                         if n1 > n0 and n1 - n0 == comparable
                         and message == event.message and message not in (None, "bands=0")]
                job.remaining_seconds = None
                if len(rates) >= 3:
                    remaining = max(0, event.total - event.completed)
                    job.remaining_seconds = [remaining * min(rates), remaining * max(rates)]


def _plan_directory(job: Job) -> Path:
    return _job_path(job, job.plan_dir or job.draft_dir or job.output_dir)


def _plan_path(job: Job) -> Path:
    directory = _plan_directory(job)
    if job.draft_dir and not job.plan_dir:
        directory = directory / "coarse"
    path = _job_path(job, directory / JSON_FILENAME)
    if not path.is_file():
        raise HTTPException(status_code=409, detail="preview is not ready")
    return path


def _load_candidates(job: Job) -> list[dict]:
    path = _plan_directory(job) / "clean_candidates.json"
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
    try:
        first, last = seconds_to_frames(start, min(start + duration, plan.source.frame_count / fps), plan.source)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        plan_dir = _job_path(job, Path(job.output_dir) / "plans" / uuid.uuid4().hex)
        with job.lock:
            job.failed_stage = "detect"
            job.save()
        def candidates_ready(snapshot):
            with job.lock:
                job.draft_dir = str(plan_dir)
                snapshot["evidence"] = {name: relative_path(job.output_dir, path)
                                        for name, path in snapshot["evidence"].items()}
                job.candidates_snapshot = snapshot
                job.review_ready = True
                job.default_selected_ids = snapshot["default_selected_ids"]
                job.phase = "refine"
                job.timings["candidates_s"] = time.perf_counter() - started
                job.save()
        plan = _get_engine().plan(
            WipeRequest(
                video=job.video_path,
                output_dir=str(plan_dir),
                intent=intent or None,
                preview=True,
            ),
            on_progress=lambda event: _update_progress(job, event),
            cancellation=job.token, on_candidates=candidates_ready,
        )
        default_selected = [track.id for track in plan.remove_tracks]
        with job.lock:
            selected = list(default_selected) if job.overrides is None else job.overrides["selected_ids"]
            conflicts = sorted(set(selected) - {track.id for track in plan.tracks if track.segments})
            job.commit(default_selected_ids=default_selected, selected_ids=selected,
                       plan_dir=str(plan_dir), review_ready=True, final_plan_ready=True,
                       conflicts=conflicts, progress=0.0, phase="preview", state="preview_ready",
                       warnings=job.input_warnings + list(plan.warnings),
                       timings=dict(job.timings, plan_s=time.perf_counter() - started))
    except Exception as exc:
        _set_error(job, exc)


def _run_inpaint(job: Job, run_dir: Path) -> None:
    try:
        plan_path = str(run_dir / JSON_FILENAME)
        started = time.perf_counter()
        plan = _prepare_execution(job, load_wipe_plan(plan_path), run_dir)
        save_wipe_plan(plan, str(run_dir))
        _update_progress(job, ProgressEvent("prepare", 0, 1))
        job.device = _get_engine()._trial_identity()
        result = _get_engine().run(
            WipeRequest(
                video=job.video_path,
                output_dir=str(run_dir),
                plan=plan_path,
                prediction_cache_dir=_prediction_directory(job),
            ),
            on_progress=lambda event: _update_progress(job, event),
            cancellation=job.token,
        )
        _job_file(job, result.output_path)
        with job.lock:
            job.token.raise_if_cancelled()
            retained = _retained_marks(job, plan)
            job.commit(result_path=result.output_path, result_revision=run_dir.name,
                       result_review_revision=job.review_revision, result_identity=job.device, review_marks=retained, progress=1.0,
                       phase="complete", state="done",
                       warnings=job.input_warnings + list(result.warnings),
                       timings={**job.timings, **result.timings,
                                "run_wall_s": time.perf_counter() - started})
    except Exception as exc:
        _set_error(job, exc)
        return
    release_job(job.id)


def _prediction_directory(job):
    return str(_job_path(job, Path(job.output_dir) / "predictions"))


def _retained_marks(job, plan):
    if not job.result_path or not job.review_marks or job.result_identity != job.device:
        return []
    old = load_wipe_plan(str(_job_file(job, Path(job.result_path).parent / JSON_FILENAME)))
    radius = _get_engine()._task_impl.feather_radius
    from videowipe.inpainters.sttn import get_inpaint_mode
    old_union, old_frame = execution_masks(old, radius)
    new_union, new_frame = execution_masks(plan, radius)
    split = int(plan.source.width * 3 / 16)
    if get_inpaint_mode(plan.source.height, split, old_union) != get_inpaint_mode(plan.source.height, split, new_union):
        return []
    return [mark for mark in job.review_marks if mark['status'] == 'issue' and np.array_equal(
        np.squeeze(old_frame(mark['frame_index']) if old_frame else old_union),
        np.squeeze(new_frame(mark['frame_index']) if new_frame else new_union))]


class ResultMarkRequest(BaseModel):
    result_revision: str
    frame_index: StrictInt = Field(ge=0)
    status: str = Field(pattern=r"^(seen|issue)$")


@app.get('/jobs/{job_id}/mask')
def actual_mask(job_id: str, frame_index: int = Query(ge=0), revision: int = Query(ge=0)):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='job not found')
    with job.lock:
        if revision != job.review_revision or not job.final_plan_ready:
            raise HTTPException(status_code=409, detail='mask revision is not ready')
        if job.reviewed_plan_path and job.compiled_review_revision == revision:
            plan = load_wipe_plan(str(_job_file(job, job.reviewed_plan_path)))
        else:
            plan, _ = _reviewed_plan(job, ConfirmRequest(**(job.overrides or {})), allow_empty=True)
            machine = load_wipe_plan(str(_plan_path(job)))
            if {t.id for t in plan.remove_tracks} - {t.id for t in machine.remove_tracks}:
                raise HTTPException(status_code=409, detail='新增目标需先试擦确认出现时间')
        if frame_index >= plan.source.frame_count:
            raise HTTPException(status_code=400, detail='frame outside source')
        union, temporal = execution_masks(plan, _get_engine()._task_impl.feather_radius)
        alpha = temporal(frame_index) if temporal else union[:, :, 0]
        ok, png = cv2.imencode('.png', np.rint(alpha * 255).astype(np.uint8))
        if not ok:
            raise HTTPException(status_code=500, detail='mask encoding failed')
        return Response(png.tobytes(), media_type='image/png', headers={'Cache-Control': 'no-store', 'X-Has-Removal': str(bool(union.any())).lower()})


@app.get('/jobs/{job_id}/result-review')
def result_review(job_id: str, track_id: str | None = None):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='job not found')
    with job.lock:
        if not job.result_path:
            raise HTTPException(status_code=409, detail='no result to review')
        plan = load_wipe_plan(str(_job_file(job, Path(job.result_path).parent / JSON_FILENAME)))
        windows = review_windows(plan, job.review_marks, job.warnings, track_id)
        return {'result_revision': job.result_revision, 'marks': job.review_marks,
                'windows': windows if track_id else windows[:12], 'total': len(windows)}


@app.post('/jobs/{job_id}/result-review')
def mark_result(job_id: str, body: ResultMarkRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='job not found')
    with job.lock:
        if not job.result_path or body.result_revision != job.result_revision or job.state != 'done':
            raise HTTPException(status_code=409, detail='result changed; refresh review')
        plan = load_wipe_plan(str(_job_file(job, Path(job.result_path).parent / JSON_FILENAME)))
        if body.frame_index >= plan.source.frame_count:
            raise HTTPException(status_code=400, detail='frame outside result')
        marks = [m for m in job.review_marks if m['frame_index'] != body.frame_index]
        if len(marks) >= 512:
            raise HTTPException(status_code=400, detail='too many review markers')
        marks.append({'frame_index': body.frame_index, 'status': body.status})
        job.commit(review_marks=marks)
    return result_review(job_id)


@app.delete('/jobs/{job_id}/prediction-cache')
def clear_prediction_cache(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='job not found')
    with job.lock:
        if job.state in {'pending', 'running', 'trial_running', 'cancelling'}:
            raise HTTPException(status_code=409, detail='wait for processing to finish')
        directory = _owned_cache_path(job, 'predictions')
        for path in directory.glob('*'):
            if re.fullmatch(r'(?:[0-9a-f]{64}\.npz|\.[0-9a-f]{32}\.tmp)', path.name):
                _owned_cache_path(job, f'predictions/{path.name}').unlink()
    return {'cleared': True}


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
            job.intent = intent
            job.source_sha256 = source_hash(input_path)
            job.timings["upload_s"] = time.perf_counter() - started
        publish_last(job)
        threading.Thread(target=_run_preview, args=(job, intent), daemon=True).start()
        return job.snapshot()
    except Exception as exc:
        _set_error(job, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/jobs/current")
def current_job():
    job = get_current_job()
    if job is None:
        try:
            job = restore_last(_jobs_root())
        except (OSError, ValueError, KeyError, TypeError, InvalidInputError) as exc:
            return {"state": "idle", "recovery_error": str(exc)}
    if job is None:
        return {"state": "idle"}
    return job.snapshot()


@app.delete("/jobs/current")
def cancel_current():
    try:
        job = cancel_current_job()
    except JobNotCancellable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    pointer = Path(_jobs_root()) / "last_job.json"
    if pointer.exists():
        atomic_json(pointer, {"id": None})
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
    if snapshot["state"] == "error" and not (_plan_directory(job) / JSON_FILENAME).exists():
        raise HTTPException(status_code=409, detail=snapshot["error"])
    if snapshot["state"] == "pending" and not snapshot["review_ready"]:
        raise HTTPException(status_code=409, detail="preview is not ready")
    try:
        candidates = _load_candidates(job)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    plan = load_wipe_plan(str(_plan_path(job)))
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
        "review_ready": snapshot["review_ready"],
        "final_plan_ready": snapshot["final_plan_ready"],
        "review_revision": snapshot["review_revision"],
        "overrides": snapshot["overrides"],
        "previous_review": snapshot["previous_review"],
        "conflicts": snapshot["conflicts"],
    }


@app.get("/jobs/{job_id}/preview-image")
def preview_image(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = _plan_directory(job) / "clean_preview.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="preview image not found")
    return FileResponse(_job_file(job, path), media_type="image/jpeg")


@app.get("/jobs/{job_id}/editable-preview-image")
def editable_preview_image(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = _plan_directory(job) / "clean_preview_source.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="editable preview image not found")
    return FileResponse(_job_file(job, path), media_type="image/jpeg")


def _reviewed_plan(job: Job, body: ConfirmRequest, *, allow_empty=False):
    try:
        plan = load_wipe_plan(str(_plan_path(job)))
    except (InvalidInputError, OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="job plan is invalid") from exc
    try:
        return compile_review(plan, body.selected_ids, body.bbox_overrides, body.segment_overrides,
                              [region.model_dump() for region in body.protections or []], allow_empty=allow_empty)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _check_revision(job: Job, body: ConfirmRequest) -> None:
    if body.expected_revision is not None and body.expected_revision != job.review_revision:
        raise HTTPException(status_code=409, detail="review changed; refresh current decisions")


def _save_review(job: Job, body: ConfirmRequest, *, allow_empty=False):
    _check_revision(job, body)
    plan, selected = _reviewed_plan(job, body, allow_empty=allow_empty)
    if not allow_empty and not execution_masks(plan, _get_engine()._task_impl.feather_radius)[0].any():
        raise HTTPException(status_code=400, detail="没有需要处理的区域")
    overrides = {"selected_ids": selected, "bbox_overrides": body.bbox_overrides or {}}
    if body.segment_overrides is not None:
        overrides["segment_overrides"] = {t.id: [s.to_dict() for s in t.segments]
                                          for t in plan.remove_tracks if t.id in body.segment_overrides}
    if body.protections is not None:
        overrides["protections"] = [{"id": t.id, "bbox": list(t.bbox), "segments": [s.to_dict() for s in t.segments]}
                                    for t in plan.tracks if t.action == "protect"]
    job.commit(overrides=overrides, selected_ids=selected,
               review_revision=job.review_revision + int(job.overrides != overrides),
               conflicts=[track.id for track in plan.remove_tracks if not track.segments])
    return plan, selected


@app.patch("/jobs/{job_id}/review")
def review(job_id: str, body: ReviewRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if not job.review_ready or job.state not in {"pending", "preview_ready", "done", "error", "interrupted"}:
            raise HTTPException(status_code=409, detail="review is not editable")
        body.overrides.expected_revision = body.expected_revision
        before = job.review_revision
        _save_review(job, body.overrides, allow_empty=True)
        if job.review_revision == before:
            job.commit(review_revision=job.review_revision + 1)
    return job.snapshot()


def _prepare_execution(job: Job, plan, run_dir):
    job.token.raise_if_cancelled()
    machine = load_wipe_plan(str(_plan_path(job)))
    if source_hash(job.video_path) != machine.source.sha256:
        raise InvalidInputError("source sha256 mismatch; input changed")
    edited_segments = {t.id: t.segments for t in plan.remove_tracks if ":segment-override" in t.decision_reason}
    new_ids = {track.id for track in plan.remove_tracks} - {
        track.id for track in machine.remove_tracks
    }
    if new_ids:
        plan = _get_engine()._refine_review(
            job.video_path, machine, plan,
            _job_file(job, job.refinement_path or _plan_directory(job) / "refinement_evidence.json"),
            on_progress=lambda event: _update_progress(job, event), cancellation=job.token,
            output_dir=str(run_dir),
        )
        evidence_path = run_dir / "refinement_evidence.json"
        if evidence_path.is_file():
            with job.lock:
                job.refinement_path = str(_job_file(job, evidence_path))
                job.save()
    for track in plan.remove_tracks:
        if track.id in edited_segments:
            track.segments = edited_segments[track.id]
    if not execution_masks(plan, _get_engine()._task_impl.feather_radius)[0].any():
        raise InvalidInputError("没有需要处理的区域")
    if any(not track.segments for track in plan.remove_tracks):
        raise InvalidInputError("Selected target has no confirmed active frames; review required")
    save_wipe_plan(plan, str(run_dir))
    with job.lock:
        job.commit(reviewed_plan_path=str(run_dir / JSON_FILENAME), compiled_review_revision=job.review_revision)
    return plan


def _reserve_execution(job: Job, body: ConfirmRequest, stage: str):
    _check_revision(job, body)
    if stage != "detect" and not job.final_plan_ready:
        raise HTTPException(status_code=409, detail="temporal refinement is not complete")
    if job.conflicts:
        raise HTTPException(status_code=409, detail="selected target evidence changed; review required")
    try:
        reserve_job(job)
    except JobBusy:
        raise HTTPException(status_code=409, detail="server busy, wait for current job")
    job.token = CancellationToken()
    job.phase_counts = {}
    job.operation_id = body.operation_id
    job.failed_stage = stage


@contextmanager
def _worker_start(job: Job):
    """Roll back a launch failure while the caller still holds the job lock."""
    previous = vars(job).copy()
    already_reserved = get_current_job() is job
    try:
        yield
    except Exception as exc:
        vars(job).update(previous)
        try:
            job.save()
        except (OSError, ValueError) as rollback_error:
            raise exc from rollback_error
        finally:
            if not already_reserved:
                release_job(job.id)
        raise


@app.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if job.state in {"pending", "running", "trial_running", "cancelling"}:
            job.token.cancel()
            job.state = "cancelling"
            job.save()
        elif job.state == "preview_ready":
            job.state = "cancelled"
            job.save()
            release_job(job.id)
    return job.snapshot()


@app.post("/jobs/{job_id}/retry")
def retry(job_id: str, body: RetryRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if job.operation_id == body.operation_id:
            return job.snapshot()
        if job.state not in {"error", "interrupted", "cancelled"}:
            raise HTTPException(status_code=409, detail="job is not retryable")
        if job.final_plan_ready and job.failed_stage != "detect":
            with _worker_start(job):
                job.state = "preview_ready"
                if job.failed_stage == "trial":
                    request = job.trial_request or (job.trial or {}).get("request")
                    if not request:
                        raise HTTPException(status_code=409, detail="trial interval missing; start a new trial")
                    return trial(job_id, TrialRequest(**(job.overrides or {}),
                                 start_seconds=request["start_seconds"], duration_seconds=request["duration_seconds"],
                                 expected_revision=job.review_revision, operation_id=body.operation_id))
                return confirm(job_id, ConfirmRequest(**(job.overrides or {}),
                               expected_revision=job.review_revision, operation_id=body.operation_id))
        with _worker_start(job):
            _reserve_execution(job, ConfirmRequest(operation_id=body.operation_id), "detect")
            # Old IDs must not silently acquire old boxes after another detection.
            job.previous_review = {"overrides": job.overrides,
                                   "candidates": (job.candidates_snapshot or {}).get("candidates", []),
                                   "revision": job.review_revision}
            job.overrides = None
            job.review_revision += 1
            job.plan_dir = None
            job.refinement_path = None
            job.final_plan_ready = False
            job.review_ready = False
            job.state = "pending"
            job.error = None
            job.save()
            threading.Thread(target=_run_preview, args=(job, job.intent), daemon=True).start()
    return job.snapshot()


@app.post("/jobs/{job_id}/confirm")
def confirm(job_id: str, body: ConfirmRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if body.operation_id and job.operation_id == body.operation_id:
            return job.snapshot()
        if job.state not in {"preview_ready", "done"}:
            raise HTTPException(status_code=409, detail=f"job is {job.state}")
        plan, selected_ids = _save_review(job, body)
        with _worker_start(job):
            _reserve_execution(job, body.model_copy(update={"expected_revision": job.review_revision}), "full")
            run_dir = _job_path(job, Path(job.output_dir) / "runs" / uuid.uuid4().hex)
            save_wipe_plan(plan, str(run_dir))
            job.selected_ids = selected_ids
            job.confirmed_review = dict(job.overrides)
            job.error = None
            job.progress = 0.0
            job.phase = "prepare"
            job.state = "running"
            job.save()
            threading.Thread(target=_run_inpaint, args=(job, run_dir), daemon=True).start()
    return job.snapshot()


def _run_trial(job: Job, plan, frame_range: tuple[int, int], trial_dir: Path) -> None:
    started = time.perf_counter()
    try:
        plan = _prepare_execution(job, plan, trial_dir)
        _trial_range(plan, frame_range[0] / plan.source.fps,
                     (frame_range[1] - frame_range[0]) / plan.source.fps)
        _update_progress(job, ProgressEvent("prepare", 0, 1))
        identity = _get_engine()._trial_identity()
        job.token.raise_if_cancelled()
        with job.lock:
            job.device = identity
        key = _trial_key(plan, frame_range, identity)
        cached = _cached_trial(job, key)
        if cached:
            try:
                _validate_trial_file(contained_path(job.output_dir, cached["path"]), plan, frame_range)
            except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
                with job.lock:
                    job.trial_cache.remove(cached)
                cached = None
        warnings = list(job.warnings)
        if cached:
            output_path = str(contained_path(job.output_dir, cached["path"]))
            backend = identity["backend"]
        else:
            save_wipe_plan(plan, str(trial_dir))
            result = _get_engine().run(
                WipeRequest(video=job.video_path, output_dir=str(trial_dir),
                            plan=str(trial_dir / JSON_FILENAME), trial_range=frame_range,
                            prediction_cache_dir=_prediction_directory(job)),
                on_progress=lambda event: _update_progress(job, event), cancellation=job.token,
            )
            _job_file(job, result.output_path)
            _validate_trial_file(result.output_path, plan, frame_range)
            output_path, backend = result.output_path, result.backend
            warnings = job.input_warnings + list(result.warnings)
        with job.lock:
            job.token.raise_if_cancelled()
            if not cached:
                _cache_trial(job, key, output_path)
            completed = dict(job.trial, ready=True, cache_hit=bool(cached),
                             elapsed_s=round(time.perf_counter() - started, 3), backend=backend)
            job.commit(trial=completed, trial_path=output_path,
                       last_trial=completed, last_trial_path=output_path,
                       warnings=warnings, progress=1.0, state="preview_ready", phase="preview")
    except Exception as exc:
        with job.lock:
            try:
                job.commit(error="已取消，已保存的审阅决定保留。" if isinstance(exc, ProcessingCancelledError) else str(exc),
                           trial=dict(job.last_trial) if job.last_trial else None,
                           trial_path=job.last_trial_path, state="preview_ready", phase="preview")
            except (OSError, ValueError) as save_error:
                _set_error(job, save_error)


def _trial_key(plan, frame_range, identity) -> str:
    tracks = sorted(({
        "segments": [[segment.start, segment.end] for segment in track.segments],
        "mask": hashlib.sha256(np.ascontiguousarray(track.mask).tobytes()).hexdigest(),
        "dtype": str(track.mask.dtype), "shape": list(track.mask.shape),
        "action": track.action,
        **({"spatial_segments": track.spatial_segments} if track.spatial_segments is not None else {}),
    } for track in plan.tracks if track.action != "keep"), key=lambda item: json.dumps(item, sort_keys=True))
    payload = {"source": plan.source.to_dict(), "tracks": tracks,
               "frames": list(frame_range), "runtime": identity}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _validate_trial_file(path, plan, frame_range):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path),
    ], stderr=subprocess.DEVNULL, timeout=60))["streams"][0]
    if (int(probe["width"]), int(probe["height"]), int(probe["nb_read_frames"])) != (
        plan.source.width, plan.source.height * 2, frame_range[1] - frame_range[0]
    ):
        raise ValueError("Trial output is incomplete or has unexpected dimensions")


def _cached_trial(job: Job, key: str):
    with job.lock:
        for entry in list(job.trial_cache):
            if entry["key"] != key:
                continue
            try:
                path = _job_file(job, contained_path(job.output_dir, entry["path"]))
                if path.stat().st_size != entry["size"] or source_hash(path) != entry["sha256"]:
                    raise ValueError("cache content changed")
            except (OSError, ValueError, HTTPException):
                job.trial_cache.remove(entry)
                return None
            return entry
    return None


def _owned_cache_path(job: Job, relative: str) -> Path:
    """Containment alone does not prove ownership: never follow cache links."""
    path = Path(job.output_dir).resolve() / relative
    if path != contained_path(job.output_dir, relative):
        raise ValueError("cache path must not contain symlinks")
    return path


def _cache_trial(job: Job, key: str, path: str):
    """Bounded MP4 reuse; only evict explicitly owned, idle cached files."""
    try:
        relative = relative_path(job.output_dir, path)
        size = Path(path).stat().st_size
        if size > 128 * 1024 * 1024:
            return
        while len(job.trial_cache) >= 3 or sum(row["size"] for row in job.trial_cache) + size > 128 * 1024 * 1024:
            victim = next((row for row in job.trial_cache
                           if str(contained_path(job.output_dir, row["path"])) not in {job.trial_path, job.last_trial_path}
                           and not job.readers.get(row["path"])
                           and row["path"] not in job.playback.values()), None)
            if victim is None:
                return
            owned = _owned_cache_path(job, victim["path"])
            # Never delete unknown legacy artifacts or an entire run directory.
            if not re.fullmatch(r"trials/[0-9a-f]{32}/[^/]+\.mp4", victim["path"]):
                return
            owned.unlink(missing_ok=True)
            job.trial_cache.remove(victim)
        job.trial_cache.append({"key": key, "path": relative, "size": size,
                                "sha256": source_hash(path)})
    except (OSError, ValueError):
        # ponytail: cache capacity and write failures only disable reuse; the verified output remains usable.
        return


@app.post("/jobs/{job_id}/trial")
def trial(job_id: str, body: TrialRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if body.operation_id and job.operation_id == body.operation_id:
            return job.snapshot()
        if job.state not in {"preview_ready", "done"}:
            raise HTTPException(status_code=409, detail=f"job is {job.state}")
        plan, _selected_ids = _save_review(job, body)
        fps = plan.source.fps
        first, last = _trial_range(plan, body.start_seconds, body.duration_seconds)
        with _worker_start(job):
            _reserve_execution(job, body.model_copy(update={"expected_revision": job.review_revision}), "trial")
            trial_id = uuid.uuid4().hex
            trial_dir = _job_path(job, Path(job.output_dir) / "trials" / trial_id)
            job.trial = {
                "id": trial_id, "ready": False, "request": body.model_dump(),
                "frame_range": [first, last], "start_seconds": first / fps,
                "duration_seconds": (last - first) / fps,
                "review_revision": job.review_revision,
            }
            job.trial_request = {"start_seconds": body.start_seconds, "duration_seconds": body.duration_seconds}
            job.trial_path = None
            job.error = None
            job.progress = 0.0
            job.phase = "trial"
            job.state = "trial_running"
            job.save()
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
        relative = relative_path(job.output_dir, path)
        job.readers[relative] = job.readers.get(relative, 0) + 1
    def release_reader():
        with job.lock:
            job.readers[relative] = max(0, job.readers.get(relative, 0) - 1)
    return FileResponse(_job_file(job, path), media_type="video/mp4",
                        headers={"Cache-Control": "no-store"}, background=BackgroundTask(release_reader))


@app.post("/jobs/{job_id}/playback")
def playback(job_id: str, body: PlaybackRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        if body.trial_id is None:
            job.playback.pop(body.lease_id, None)
        elif job.trial and job.trial.get("ready") and job.trial["id"] == body.trial_id and job.trial_path:
            job.playback[body.lease_id] = relative_path(job.output_dir, job.trial_path)
        else:
            raise HTTPException(status_code=409, detail="trial is not available")
    return {"ok": True}


@app.get("/jobs/{job_id}/progress")
def progress_sse(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    def _events():
        while True:
            snapshot = job.snapshot()
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["state"] in {"done", "error", "cancelled", "preview_ready", "interrupted"}:
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
    if snapshot["state"] != "done" and not snapshot["result_path"]:
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
    plan = load_wipe_plan(str(_plan_path(job)), load_masks=False)
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
