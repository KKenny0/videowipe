from pathlib import Path
import pytest
from fastapi import HTTPException
from videowipe.server import app as web
from videowipe.server.jobs import Job


@pytest.mark.parametrize('linked_parent', [False, True])
def test_trial_eviction_never_follows_links_to_input(tmp_path, linked_parent):
    source = tmp_path/'input.mp4'; source.write_bytes(b'original')
    run = tmp_path/'trials'/('a'*32); run.parent.mkdir()
    if linked_parent:
        run.symlink_to(tmp_path, target_is_directory=True)
        relative = f'trials/{run.name}/input.mp4'
    else:
        run.mkdir(); (run/'old.mp4').symlink_to(source)
        relative = f'trials/{run.name}/old.mp4'
    new = tmp_path/'new.mp4'; new.write_bytes(b'result')
    job = Job('a'*32, str(source), str(tmp_path), state='done')
    job.trial_cache = [{'key':str(i), 'path':relative, 'size':8} for i in range(3)]
    web._cache_trial(job, 'new', str(new))
    assert source.read_bytes() == b'original'
    assert new.read_bytes() == b'result'


def test_prediction_cleanup_rejects_links_and_still_clears_owned_files(tmp_path, monkeypatch):
    source = tmp_path/'input.mp4'; source.write_bytes(b'original')
    directory=tmp_path/'predictions'; directory.mkdir()
    entry=directory/('a'*64+'.npz'); entry.symlink_to(source)
    job=Job('a'*32,str(source),str(tmp_path),state='done')
    monkeypatch.setattr(web,'get_job',lambda _:job)
    monkeypatch.setattr(web,'_job_path',lambda j,p:Path(p).resolve())
    with pytest.raises((ValueError,HTTPException)):web.clear_prediction_cache(job.id)
    assert source.read_bytes()==b'original'
    entry.unlink();entry.write_bytes(b'cache')
    assert web.clear_prediction_cache(job.id)=={'cleared':True}
    assert not entry.exists() and source.read_bytes()==b'original'
    directory.rmdir();directory.symlink_to(tmp_path,target_is_directory=True)
    victim=tmp_path/('b'*64+'.npz');victim.write_bytes(b'not-cache')
    with pytest.raises((ValueError,HTTPException)):web.clear_prediction_cache(job.id)
    assert victim.read_bytes()==b'not-cache'
