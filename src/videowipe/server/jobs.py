"""Single local execution slot and atomic, source-bound task manifests."""
from __future__ import annotations

import os
import json
import hashlib
import math
import re
import threading
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional

from videowipe.api import CancellationToken
from videowipe.plan import load_wipe_plan

JobState = Literal["pending", "preview_ready", "running", "trial_running", "done", "error", "cancelled", "cancelling", "interrupted"]


@dataclass
class Job:
    id: str
    video_path: str
    output_dir: str
    original_filename: str = "input.mp4"
    state: JobState = "pending"
    progress: float = 0.0
    phase: str = "upload"
    warnings: list[str] = field(default_factory=list)
    input_warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    selected_ids: list[str] = field(default_factory=list)
    default_selected_ids: list[str] = field(default_factory=list)
    result_path: Optional[str] = None
    confirmed_review: Optional[dict] = None
    result_revision: Optional[str] = None
    result_review_revision: Optional[int] = None
    result_identity: Optional[dict] = None
    review_marks: list[dict] = field(default_factory=list)
    reviewed_plan_path: Optional[str] = None
    compiled_review_revision: Optional[int] = None
    trial: Optional[dict] = None
    trial_request: Optional[dict] = None
    trial_path: Optional[str] = None
    last_trial: Optional[dict] = None
    last_trial_path: Optional[str] = None
    source_sha256: str = ""
    plan_dir: Optional[str] = None
    draft_dir: Optional[str] = None
    refinement_path: Optional[str] = None
    review_ready: bool = False
    final_plan_ready: bool = False
    candidates_snapshot: Optional[dict] = None
    review_revision: int = 0
    overrides: Optional[dict] = None
    previous_review: Optional[dict] = None
    conflicts: list[str] = field(default_factory=list)
    intent: Optional[str] = None
    failed_stage: str = "detect"
    operation_id: Optional[str] = None
    phase_counts: dict = field(default_factory=dict)
    completed: int = 0
    total: int = 0
    remaining_seconds: Optional[list[float]] = None
    device: Optional[dict] = None
    trial_cache: list[dict] = field(default_factory=list)
    token: CancellationToken = field(default_factory=CancellationToken, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    media_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    progress_samples: list[tuple[float, int, str | None]] = field(default_factory=list, repr=False)
    readers: dict[str, int] = field(default_factory=dict, repr=False)
    playback: dict[str, str] = field(default_factory=dict, repr=False)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "original_filename": self.original_filename,
                "confirmed_review": self.confirmed_review,
                "result_revision": self.result_revision,
                "result_review_revision": self.result_review_revision,
                "review_marks": self.review_marks,
                "state": self.state,
                "progress": self.progress,
                "phase": self.phase,
                "warnings": list(self.warnings),
                "timings": dict(self.timings),
                "error": self.error,
                "selected_ids": list(self.selected_ids),
                "default_selected_ids": list(self.default_selected_ids),
                "result_path": self.result_path,
                "trial": dict(self.trial) if self.trial else None,
                "review_ready": self.review_ready,
                "final_plan_ready": self.final_plan_ready,
                "review_revision": self.review_revision,
                "overrides": self.overrides,
                "previous_review": self.previous_review,
                "conflicts": list(self.conflicts),
                "phase_counts": dict(self.phase_counts),
                "completed": self.completed,
                "total": self.total,
                "remaining_seconds": self.remaining_seconds,
                "device": self.device,
            }

    def save(self) -> None:
        with self.lock:
            data = {key: getattr(self, key) for key in (
                "id", "original_filename", "source_sha256", "state", "phase",
                "review_revision", "overrides", "confirmed_review", "trial",
                "warnings", "input_warnings", "error", "intent", "failed_stage",
                "default_selected_ids", "selected_ids", "review_ready", "final_plan_ready",
                "candidates_snapshot", "conflicts", "operation_id", "timings", "trial_cache",
                "last_trial", "trial_request",
                "previous_review", "result_revision", "result_review_revision", "result_identity", "review_marks", "compiled_review_revision",
            )}
            for key in ("video_path", "plan_dir", "draft_dir", "refinement_path", "result_path", "trial_path", "last_trial_path", "reviewed_plan_path"):
                value = getattr(self, key)
                data[key] = relative_path(self.output_dir, value) if value else None
            data.update(schema_version=1, updated_at=time.time())
            atomic_json(Path(self.output_dir) / "job.json", data)

    def commit(self, **changes) -> None:
        """Publish state only if its durable manifest replacement succeeds."""
        with self.lock:
            previous = {key: getattr(self, key) for key in changes}
            try:
                for key, value in changes.items():
                    setattr(self, key, value)
                self.save()
            except Exception:
                for key, value in previous.items():
                    setattr(self, key, value)
                raise


def source_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(root, value) -> str:
    return str(Path(value).resolve().relative_to(Path(root).resolve()))


def contained_path(root, value) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("invalid manifest path")
    path = (Path(root) / value).resolve()
    path.relative_to(Path(root).resolve())
    return path


def atomic_json(path: Path, data) -> None:
    if path.is_symlink():
        raise ValueError("unsafe manifest path")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_last(job: Job) -> None:
    job.save()
    atomic_json(Path(job.output_dir).parent / "last_job.json", {"id": job.id})


def restore_last(output_base: str) -> Optional[Job]:
    """Only import the explicit last manifest; corrupt or foreign data fails closed."""
    global _current_job
    root = Path(output_base).resolve()
    if not (root / "last_job.json").exists():
        return None
    with _current_lock:
        pointer = json.loads(contained_path(root, "last_job.json").read_text())
        job_id = pointer["id"]
        if job_id is None:
            return None
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("invalid last job id")
        if job_id in JOBS:
            return JOBS[job_id]
        directory = contained_path(root, job_id)
        data = json.loads(contained_path(directory, "job.json").read_text())
        if data.get("schema_version") != 1 or data.get("id") != job_id:
            raise ValueError("unsupported or mismatched job manifest")
        if data.get("state") not in {"pending", "preview_ready", "running", "trial_running", "done", "error", "cancelled", "cancelling", "interrupted"}:
            raise ValueError("invalid job state")
        if type(data.get("review_revision")) is not int or data["review_revision"] < 0:
            raise ValueError("invalid review revision")
        for key in ("review_ready", "final_plan_ready"):
            if type(data.get(key)) is not bool:
                raise ValueError(f"invalid {key}")
        for key in ("warnings", "input_warnings", "selected_ids", "default_selected_ids", "conflicts"):
            if not isinstance(data.get(key), list) or any(not isinstance(value, str) for value in data[key]):
                raise ValueError(f"invalid {key}")
        if not isinstance(data.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", data["source_sha256"]):
            raise ValueError("invalid source identity")
        if not isinstance(data.get("timings"), dict) or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in data["timings"].values()
        ):
            raise ValueError("invalid timings")
        overrides = data.get("overrides")
        if overrides is not None:
            if not isinstance(overrides, dict) or not isinstance(overrides.get("selected_ids"), list) or not isinstance(overrides.get("bbox_overrides"), dict):
                raise ValueError("invalid review overrides")
            if any(not isinstance(value, str) for value in overrides["selected_ids"]):
                raise ValueError("invalid review selection")
            for candidate_id, box in overrides["bbox_overrides"].items():
                if candidate_id not in overrides["selected_ids"] or not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box):
                    raise ValueError("invalid review box")
        for key in ("video_path", "plan_dir", "draft_dir", "refinement_path", "result_path", "trial_path", "last_trial_path", "reviewed_plan_path"):
            if data.get(key) is not None:
                data[key] = str(contained_path(directory, data[key]))
        if source_hash(data["video_path"]) != data["source_sha256"]:
            raise ValueError("source sha256 mismatch; task input changed")
        review_dir = data.get("plan_dir") or (
            str(contained_path(data["draft_dir"], "coarse")) if data.get("draft_dir") else None
        )
        if data["final_plan_ready"] and not data.get("plan_dir"):
            raise ValueError("complete plan missing")
        if review_dir:
            plan = load_wipe_plan(str(contained_path(review_dir, "wipe_plan.json")))
            if plan.source.sha256 != data["source_sha256"]:
                raise ValueError("plan source mismatch")
            if overrides:
                from videowipe.server.review import compile_review
                from copy import deepcopy
                compile_review(deepcopy(plan), **overrides, allow_empty=True)
                known = {track.id for track in plan.tracks}
                if set(overrides["selected_ids"]) - known:
                    raise ValueError("unknown review target")
                for x1, y1, x2, y2 in overrides["bbox_overrides"].values():
                    if not (0 <= x1 < x2 < plan.source.width and 0 <= y1 < y2 < plan.source.height):
                        raise ValueError("review box outside source")
        elif data.get("final_plan_ready"):
            raise ValueError("complete plan missing")
        for key in ("result_path", "trial_path", "last_trial_path"):
            if data.get(key) and not Path(data[key]).is_file():
                raise ValueError("successful artifact missing")
        if data.get("result_path") and not data.get("result_revision"):
            data["result_revision"] = Path(data["result_path"]).parent.name
        if data.get("result_revision") is not None and not re.fullmatch(r"[0-9a-f]{32}", data["result_revision"]):
            raise ValueError("invalid result revision")
        for key in ("compiled_review_revision", "result_review_revision"):
            if data.get(key) is not None and (type(data[key]) is not int or data[key] < 0):
                raise ValueError("invalid compiled revision")
        marks = data.get("review_marks", [])
        if (marks and not data.get("plan_dir")) or not isinstance(marks, list) or len(marks) > 512 or any(
            not isinstance(m, dict) or type(m.get("frame_index")) is not int
            or not 0 <= m["frame_index"] < plan.source.frame_count
            or m.get("status") not in {"seen", "issue"} for m in marks
        ):
            raise ValueError("invalid review markers")
        if data.get("reviewed_plan_path"):
            compiled = load_wipe_plan(data["reviewed_plan_path"])
            if compiled.source.sha256 != data["source_sha256"]:
                raise ValueError("compiled plan source mismatch")
        cache = data.get("trial_cache")
        if not isinstance(cache, list) or len(cache) > 3:
            raise ValueError("invalid trial cache")
        for entry in cache:
            if not isinstance(entry, dict) or type(entry.get("size")) is not int or entry["size"] < 0:
                raise ValueError("invalid cache entry")
            if not isinstance(entry.get("path"), str) or not re.fullmatch(r"trials/[0-9a-f]{32}/[^/]+\.mp4", entry["path"]):
                raise ValueError("invalid cache path")
            if any(not isinstance(entry.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", entry[key]) for key in ("key", "sha256")):
                raise ValueError("invalid cache identity")
            contained_path(directory, entry["path"])
        if sum(entry["size"] for entry in cache) > 128 * 1024 * 1024:
            raise ValueError("cache exceeds capacity")
        data.pop("schema_version")
        data.pop("updated_at")
        if data.get("trial_request") is None and data.get("failed_stage") == "trial":
            request = (data.get("trial") or {}).get("request")
            if request:
                data["trial_request"] = {key: request[key] for key in ("start_seconds", "duration_seconds")}
        request = data.get("trial_request")
        if request is not None and (
            not isinstance(request, dict) or set(request) != {"start_seconds", "duration_seconds"}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in request.values())
            or request["start_seconds"] < 0 or not 0 < request["duration_seconds"] <= 5
        ):
            raise ValueError("invalid saved trial request")
        job = Job(output_dir=str(directory), **data)
        if job.state in {"pending", "running", "trial_running", "cancelling"}:
            job.state = "interrupted"
            job.phase = "interrupted"
            job.error = ("处理已中断，可使用已保存的计划重试。" if job.final_plan_ready
                         else "检测已中断，审阅草稿已保留，请重新检测并核对目标。")
            if job.last_trial:
                job.trial = dict(job.last_trial)
                job.trial_path = job.last_trial_path
            job.save()
        JOBS[job_id] = job
        if job.state == "preview_ready":
            _current_job = job
        return job


def reserve_job(job: Job) -> None:
    global _current_job
    with _current_lock:
        if _current_job is not None and _current_job is not job:
            raise JobBusy()
        _current_job = job


_current_job: Optional[Job] = None
_current_lock = threading.Lock()
JOBS: dict[str, Job] = {}


class JobBusy(Exception):
    """Raised when the local server is already processing a job."""


class JobNotCancellable(Exception):
    """Raised when the current job cannot be safely cancelled."""


def create_job(video_path: str = "", output_base: str = "jobs") -> Job:
    """Reserve the single local job slot and return the new job."""
    global _current_job
    with _current_lock:
        if _current_job is not None:
            raise JobBusy()
        job_id = uuid.uuid4().hex
        output_dir = os.path.join(output_base, job_id)
        os.makedirs(output_dir, exist_ok=True)
        job = Job(id=job_id, video_path=video_path, output_dir=output_dir)
        _current_job = job
        JOBS[job_id] = job
        return job


def release_job(job_id: str | None = None) -> None:
    """Clear the busy slot after a job reaches done/error."""
    global _current_job
    with _current_lock:
        if _current_job is None:
            return
        if job_id is None or _current_job.id == job_id:
            _current_job = None


def get_current_job() -> Optional[Job]:
    with _current_lock:
        return _current_job


def cancel_current_job() -> Optional[Job]:
    """Release the current slot when a preview is waiting for user input."""
    job = get_current_job()
    if job is None:
        return None

    with job.lock:
        if job.state in {"pending", "running", "trial_running", "cancelling"}:
            raise JobNotCancellable(f"job is {job.state}")
        job.state = "cancelled"
        job.error = "cancelled"
        if job.source_sha256:
            job.save()

    release_job(job.id)
    return job


def get_job(job_id: str) -> Optional[Job]:
    return JOBS.get(job_id)


def reset_jobs() -> None:
    """Reset in-memory jobs for tests."""
    global _current_job
    with _current_lock:
        _current_job = None
        JOBS.clear()
