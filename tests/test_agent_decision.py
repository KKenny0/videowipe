"""Agent participation is observable without persisting prompts or subprocess output."""
import json
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe import WipeEngine, WipeRequest
from videowipe import agent as agents
from videowipe.detect import TextBox
from videowipe.plan import load_wipe_plan

SECRET = 'do-not-persist-this-secret'


def source_and_detector(tmp_path):
    path = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 8, (160, 100))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((100, 160, 3), np.uint8))
    writer.release()
    box = TextBox(np.array([[35, 80], [125, 80], [125, 88], [35, 88]]), .95)
    return path, SimpleNamespace(detect=lambda frame: [box])


@pytest.mark.parametrize('status,output,returncode', [
    ('not_requested', None, 0),
    ('success', '{"remove":["c1"]}', 0),
    ('success', '{"remove":[]}', 0),
    ('command_unavailable', None, 0),
    ('timeout', None, 0),
    ('execution_failed', SECRET, 2),
    ('invalid_output', '{"remove":["unknown"]}', 0),
    ('invalid_output', '[1,2]', 0),
    ('invalid_output', '{}', 0),
])
def test_sdk_and_saved_diagnostics_distinguish_outcomes(tmp_path, monkeypatch, capsys, status, output, returncode):
    source, detector = source_and_detector(tmp_path)
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        if status == 'timeout':
            raise subprocess.TimeoutExpired(SECRET, 30, output=SECRET, stderr=SECRET)
        return SimpleNamespace(returncode=returncode, stdout=output, stderr=SECRET)
    monkeypatch.setattr(agents.subprocess, 'run', run)
    monkeypatch.setattr(agents.shutil, 'which', lambda _: None if status == 'command_unavailable' else '/agent')
    output_dir = tmp_path / 'plan'
    with WipeEngine(task='clean', detector=detector, ocr='off') as engine:
        result = engine.run(WipeRequest(source, output_dir, preview=True, intent='remove subtitles',
                                        agent=None if status == 'not_requested' else 'codex'))
    decision = result.agent_decision
    assert decision['status'] == status
    assert decision['selection_source'] == ('agent' if status == 'success' else 'local_rules')
    saved = json.loads((output_dir / 'agent_decision.json').read_text())
    plan = load_wipe_plan(str(output_dir / 'wipe_plan.json'))
    assert decision == saved == plan.request['agent_decision'] == result.to_dict()['agent_decision']
    assert bool(plan.remove_tracks) == (output != '{"remove":[]}')
    if status in {'not_requested', 'command_unavailable'}:
        assert calls == []
    else:
        assert calls == [1]
    captured = capsys.readouterr()
    assert SECRET not in json.dumps(decision) + captured.out + captured.err


def test_reused_sdk_does_not_reuse_previous_agent_status(tmp_path, monkeypatch):
    source, detector = source_and_detector(tmp_path)
    monkeypatch.setattr(agents.shutil, 'which', lambda _: '/agent')
    monkeypatch.setattr(agents.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='{"remove":[]}', stderr=''))
    with WipeEngine(task='clean', detector=detector, ocr='off') as engine:
        first = engine.run(WipeRequest(source, tmp_path / 'one', preview=True,
                                       agent='codex', intent='keep all'))
        second = engine.run(WipeRequest(source, tmp_path / 'two', preview=True))
    assert first.agent_decision['status'] == 'success'
    assert second.agent_decision['status'] == 'not_requested'
    assert first.agent_decision['status'] == 'success'


@pytest.mark.parametrize('code,status', [
    ('print(\'{"remove":["c1","c1"]}\')', 'success'),
    ('print(\'{"remove":[]}\')', 'success'),
    ('print(\'[{"remove":[]}]\')', 'invalid_output'),
    ('import os; os.write(1, b"\\xff")', 'invalid_output'),
    ('import os; os.write(2, b"\\xff"); print(\'{"remove":[]}\')', 'success'),
    ('import os, sys; os.write(1, b"\\xff"); sys.exit(3)', 'execution_failed'),
    ('import sys; print("'+SECRET+'", file=sys.stderr); sys.exit(3)', 'execution_failed'),
    ('import time; time.sleep(5)', 'timeout'),
])
def test_real_subprocess_outcomes_and_legacy_return(code, status, capsys):
    import shlex
    import sys
    candidate = SimpleNamespace(id='c1', to_dict=lambda: {'id': 'c1'})
    command = f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'
    result = agents.select_with_agent_result(command, [candidate], 'remove subtitle', timeout=.1)
    assert result.status == status
    old = agents.select_with_agent(command, [candidate], 'remove subtitle', timeout=.1)
    assert old == (list(result.selected_ids) if status == 'success' else None)
    captured = capsys.readouterr()
    assert SECRET not in json.dumps(result.diagnostic()) + captured.out + captured.err


def test_command_unavailable_and_bad_syntax_have_safe_reasons():
    for command in ('videowipe-agent-command-that-does-not-exist', 'bad "'+SECRET):
        result = agents.select_with_agent_result(command, [], SECRET)
        assert result.status == 'command_unavailable'
        assert SECRET not in json.dumps(result.diagnostic())


def test_saved_plan_and_manual_mask_do_not_claim_current_agent_participation(tmp_path, monkeypatch):
    source, detector = source_and_detector(tmp_path)
    monkeypatch.setattr(agents.shutil, 'which', lambda _: '/agent')
    monkeypatch.setattr(agents.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='{"remove":["c1"]}', stderr=''))
    first_dir = tmp_path / 'first'
    with WipeEngine(task='clean', detector=detector, ocr='off') as engine:
        original = engine.plan(WipeRequest(source, first_dir, agent='codex', intent='remove subtitles'))
        first = original.request['agent_decision'].copy()
        monkeypatch.setattr(agents.subprocess, 'run', lambda *a, **k: pytest.fail('must not call Agent'))
        replay = engine.run(WipeRequest(source, tmp_path / 'replay', plan=original, preview=True))
        assert replay.agent_decision['status'] == 'not_requested'
        assert replay.agent_decision['selection_source'] == 'saved_plan'
        assert original.request['agent_decision'] == first and first['status'] == 'success'
        mask = tmp_path / 'mask.png'
        assert cv2.imwrite(str(mask), np.ones((100, 160), np.uint8)*255)
        manual = engine.run(WipeRequest(source, tmp_path / 'manual', mask=mask, preview=True))
        assert manual.agent_decision['status'] == 'not_requested'
        assert manual.agent_decision['selection_source'] == 'manual_mask'


def test_confirmation_is_final_selection_source(tmp_path, monkeypatch):
    source, detector = source_and_detector(tmp_path)
    monkeypatch.setattr(agents.shutil, 'which', lambda _: '/agent')
    monkeypatch.setattr(agents.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='{"remove":["c1"]}', stderr=''))
    monkeypatch.setattr(WipeEngine, '_confirm_candidates', staticmethod(lambda *a: []))
    with WipeEngine(task='clean', detector=detector, ocr='off') as engine:
        result = engine.run(WipeRequest(source, tmp_path / 'plan', preview=True,
                                        agent='codex', intent='remove subtitles', confirm=True))
    assert result.agent_decision['status'] == 'success'
    assert result.agent_decision['selection_source'] == 'user_selection'
    assert not load_wipe_plan(str(tmp_path / 'plan/wipe_plan.json')).remove_tracks


def test_cli_reports_actual_outcome_without_subprocess_output(tmp_path):
    import os
    import shlex
    import sys
    from pathlib import Path
    source, _ = source_and_detector(tmp_path)
    code = 'import sys; print("'+SECRET+'", file=sys.stderr); print(\'{"remove":[]}\')'
    command = f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'cli'
    proc = subprocess.run([sys.executable, '-m', 'videowipe.cli', 'clean', str(source),
                           '--preview', '--target', 'region', '--region', 'bottom', '--ocr', 'off',
                           '--detect-mode', 'fast', '--intent', 'remove bottom region',
                           '--agent', command, '--output', str(output)],
                          env={**os.environ, 'PYTHONPATH': str(root / 'src')},
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert 'status=success; source=agent' in proc.stderr
    assert SECRET not in proc.stdout + proc.stderr + (output / 'agent_decision.json').read_text()


@pytest.mark.parametrize('stage', ['source_identity', 'artifacts', 'confirmation'])
def test_agent_status_survives_a_later_planning_failure(tmp_path, monkeypatch, stage):
    from videowipe import ProcessingError
    from videowipe import planning
    source, detector = source_and_detector(tmp_path)
    monkeypatch.setattr(agents.shutil, 'which', lambda _: '/agent')
    monkeypatch.setattr(agents.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b'{"remove":[]}', stderr=b''))
    def fail(*args, **kwargs):
        raise OSError('later planning probe failure')
    if stage == 'source_identity':
        monkeypatch.setattr(planning, 'compute_source', fail)
    elif stage == 'artifacts':
        monkeypatch.setattr(WipeEngine, '_write_clean_artifacts', staticmethod(fail))
    else:
        monkeypatch.setattr(WipeEngine, '_confirm_candidates', staticmethod(fail))
    with WipeEngine(task='clean', detector=detector, ocr='off') as engine:
        with pytest.raises(ProcessingError, match='later planning probe failure'):
            engine.run(WipeRequest(source, tmp_path / 'plan', preview=True, confirm=True,
                                   agent='codex', intent='remove subtitles'))
        assert engine.agent_decision['status'] == 'success'
    assert json.loads((tmp_path / 'plan/agent_decision.json').read_text())['status'] == 'success'
