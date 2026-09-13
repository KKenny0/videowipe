"""A failed launch must leave a retryable task and its saved review intact."""
import json
from types import SimpleNamespace

import pytest

from videowipe.server import app, jobs


@pytest.mark.parametrize('stage,failure', [
    ('confirm', 'save'), ('confirm', 'start'), ('confirm', 'plan'),
    ('trial', 'save'), ('trial', 'start'),
    ('retry', 'save'), ('retry', 'start'),
    ('retry_full', 'save'), ('retry_full', 'start'),
])
def test_failed_worker_launch_preserves_review(tmp_path, monkeypatch, stage, failure):
    jobs.reset_jobs()
    monkeypatch.setenv('VIDEOWIPE_JOBS_DIR', str(tmp_path))
    job = jobs.create_job(output_base=str(tmp_path))
    job.state = 'error' if stage.startswith('retry') else 'done'
    job.final_plan_ready = stage != 'retry'
    job.failed_stage = 'detect' if stage == 'retry' else 'full'
    job.review_ready = True
    job.overrides = {'selected_ids': ['c1'], 'bbox_overrides': {}}
    job.review_revision = 2
    job.operation_id = 'previous'
    job.result_path = str(tmp_path / job.id / 'successful.mp4')
    job.save()
    before = dict(vars(job))
    jobs.release_job(job.id)
    monkeypatch.setattr(app, '_save_review', lambda *args: (
        SimpleNamespace(source=SimpleNamespace(fps=25)), ['c1']))
    monkeypatch.setattr(app, '_trial_range', lambda *args: (0, 75))
    monkeypatch.setattr(app, 'save_wipe_plan', lambda *args: None)
    original_save = job.save
    def save():
        if failure == 'save' and job.state in {'running', 'trial_running', 'pending'}:
            raise OSError('disk full')
        original_save()
    monkeypatch.setattr(job, 'save', save)
    def fail(*args, **kwargs):
        raise OSError('launch failed')
    monkeypatch.setattr(app.threading.Thread, 'start', fail)
    if failure == 'plan':
        monkeypatch.setattr(app, 'save_wipe_plan', fail)
    endpoint = app.retry if stage.startswith('retry') else getattr(app, stage)
    body = (app.RetryRequest(operation_id='new') if stage.startswith('retry')
            else (app.TrialRequest if stage == 'trial' else app.ConfirmRequest)(operation_id='new'))
    try:
        with pytest.raises(OSError):
            endpoint(job.id, body)
        for name in ('state', 'overrides', 'review_revision', 'result_path', 'operation_id',
                     'token', 'final_plan_ready', 'review_ready', 'failed_stage'):
            assert getattr(job, name) == before[name], name
        assert jobs.get_current_job() is None
        persisted = json.loads((tmp_path / job.id / 'job.json').read_text())
        assert persisted['state'] == before['state']
        assert persisted['overrides'] == before['overrides']
        assert persisted['operation_id'] == before['operation_id']
    finally:
        jobs.reset_jobs()


def test_launch_rollback_does_not_release_another_task(tmp_path, monkeypatch):
    jobs.reset_jobs()
    owner = jobs.create_job(output_base=str(tmp_path))
    other = jobs.Job('other', '', str(tmp_path))
    monkeypatch.setattr(other, 'save', lambda: None)
    try:
        with pytest.raises(jobs.JobBusy), app._worker_start(other):
            jobs.reserve_job(other)
        assert jobs.get_current_job() is owner
        with pytest.raises(OSError), app._worker_start(owner):
            raise OSError('launch failed')
        assert jobs.get_current_job() is owner
    finally:
        jobs.reset_jobs()
