"""In-memory job state for the local-first web server."""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

JobState = Literal["pending", "preview_ready", "running", "done", "error", "cancelled"]


@dataclass
class Job:
    id: str
    video_path: str
    output_dir: str
    state: JobState = "pending"
    progress: float = 0.0
    phase: str = "upload"
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    selected_ids: list[str] = field(default_factory=list)
    default_selected_ids: list[str] = field(default_factory=list)
    result_path: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "state": self.state,
                "progress": self.progress,
                "phase": self.phase,
                "warnings": list(self.warnings),
                "timings": dict(self.timings),
                "error": self.error,
                "selected_ids": list(self.selected_ids),
                "default_selected_ids": list(self.default_selected_ids),
                "result_path": self.result_path,
            }


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
        if job.state in {"pending", "running"}:
            raise JobNotCancellable(f"job is {job.state}")
        job.state = "cancelled"
        job.error = "cancelled"

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
