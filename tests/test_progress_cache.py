"""ETA uses complete batches across the cache pipeline's interleaved phases."""
import pytest

from videowipe.api import ProgressEvent
from videowipe.server import app as web
from videowipe.server.jobs import Job


@pytest.mark.parametrize('batch_seconds,cached', [(2, True), (10, False)])
def test_interleaved_prediction_composition_estimates_complete_batches(monkeypatch, batch_seconds, cached):
    job = Job('test', '', '')
    now = [0]
    monkeypatch.setattr(web.time, 'perf_counter', lambda: now[0])

    def batch(number, bands=1, completed=None):
        completed = number * 25 if completed is None else completed
        now[0] = number * batch_seconds - 1
        web._update_progress(job, ProgressEvent(
            'predict', completed, 250,
            f'hits={number if cached else 0};misses={0 if cached else number}'))
        assert job.remaining_seconds is None
        now[0] += 1
        web._update_progress(job, ProgressEvent('compose', completed, 250, f'bands={bands}'))

    for number in (1, 2, 3):
        batch(number)
        assert job.remaining_seconds is None
    batch(4)
    assert job.remaining_seconds == [6 * batch_seconds] * 2
    # Changed crop counts need their own comparable completed batches.
    for number in (5, 6, 7):
        batch(number, bands=2)
        assert job.remaining_seconds is None
    batch(8, bands=2)
    assert job.remaining_seconds == [2 * batch_seconds] * 2
    batch(9, bands=0)
    assert job.remaining_seconds is None
    web._update_progress(job, ProgressEvent('encode', 0, 1))
    assert not job.progress_samples and job.remaining_seconds is None
