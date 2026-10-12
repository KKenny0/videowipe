"""Effect acceptance must not turn successful export into claimed visual success."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'verify_cleanup_effect', Path(__file__).parents[1] / 'scripts/verify_cleanup_effect.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_effect_review_requires_playback_and_exact_artifacts():
    row = {'source_sha256': 'source', 'full_sha256': 'output', 'source': {'frame_count': 100}}
    review = {'status': 'pass', 'reviewer': 'human', 'note': 'Viewed entire result at normal speed',
              'source_sha256': 'source', 'output_sha256': 'output',
              'normal_playback': True, 'entire_output_reviewed': True,
              'wrong_target': False, 'residue': False, 'false_removal': False,
              'unusable_distortion': False}
    module.validate_review(review, row)
    for change in ({'normal_playback': False}, {'output_sha256': 'old-output'},
                   {'false_removal': True}, {'entire_output_reviewed': False}):
        with pytest.raises(ValueError):
            module.validate_review(dict(review, **change), row)
    failure = dict(review, status='fail', problems=[{'frames': [10, 12], 'kind': 'residue', 'description': 'visible text'}])
    module.validate_review(failure, row)
    with pytest.raises(ValueError):
        module.validate_review(dict(failure, problems=[{'frames': [99, 101], 'kind': 'residue', 'description': 'text'}]), row)
    module.validate_review({'status': 'unverified'}, row)


def test_explicit_protection_check_is_independent_of_the_execution_mask():
    import numpy as np
    original = np.zeros((6, 8, 3), dtype=np.uint8)
    rendered = original.copy()
    region = {'bbox': [1, 1, 3, 3], 'segments': [[2, 4]]}
    assert module.check_protected_pixels(original, rendered, [region], 2) == 9
    rendered[2, 2] = 255
    with pytest.raises(AssertionError, match='protected pixels'):
        module.check_protected_pixels(original, rendered, [region], 2)
    assert module.check_protected_pixels(original, rendered, [region], 4) == 0


@pytest.mark.parametrize('presence', [None, 0.25, 1.0])
def test_small_edge_text_requires_explicit_removal(presence):
    from videowipe.detect import CleanCandidate, _classify_region, select_clean_candidates
    from videowipe.plan import _resolve_action

    bbox = (5, 75, 35, 85)
    kind, reason, remove = _classify_region(bbox, 'middle-left', [], 320, 180,
                                             presence_fraction=presence)
    candidate = CleanCandidate('edge', kind, 'edge', bbox, .9, presence or 0,
                               reason, remove)
    assert kind == 'watermark'  # Keep the explicit watermark-target route.
    assert select_clean_candidates([candidate]) == []
    assert 'persistent' not in reason  # Position alone cannot establish persistence.
    assert select_clean_candidates([candidate], targets=['watermark']) == [candidate]
    assert _resolve_action(candidate, presence or 0, 180, set(), set(), {})[0] == 'keep'
    assert _resolve_action(candidate, presence or 0, 180, {'edge'}, set(), {})[0] == 'remove'
    assert _classify_region((50, 140, 270, 160), 'bottom', [], 320, 180)[::2] == ('subtitle', True)
    assert _classify_region(bbox, 'middle-left', ['www.example.com'], 320, 180)[::2] == ('watermark', True)


def test_annotations_require_manifest_content_identity(tmp_path, monkeypatch):
    import json
    root = tmp_path / 'repo'
    fixtures = root / 'input/detext_examples'
    fixtures.mkdir(parents=True)
    video = fixtures / 'english1.mp4'
    video.write_bytes(b'original fixture')
    expected = {'file': video.name, 'source_sha256': module.sha256(video),
                'objects': [{'action': 'remove'}], 'frames': []}
    (fixtures / 'fact_baseline.json').write_text(json.dumps({'videos': [expected]}))
    monkeypatch.setattr(module, 'ROOT', root)
    assert module.case_annotations(video, module.sha256(video)) == expected
    external = tmp_path / video.name
    external.write_bytes(b'different input with the same name')
    assert module.case_annotations(external, module.sha256(external)) is None
    video.write_bytes(b'replaced repository fixture')
    assert module.case_annotations(video, module.sha256(video)) is None


@pytest.mark.parametrize('case', ['screenshot_scene', 'real_video'])
def test_failed_upload_retains_case_identity_in_report(tmp_path, monkeypatch, case):
    import json
    import sys
    from fastapi.testclient import TestClient

    # Exercise Windows' default decoding even on UTF-8 development machines.
    original_read_text = Path.read_text
    def windows_read_text(path, *args, **kwargs):
        if not args and kwargs.get('encoding') is None:
            kwargs['encoding'] = 'cp1252'
        return original_read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', windows_read_text)

    video = tmp_path / 'source.mp4'
    video.write_bytes(b'input identity survives failed processing')
    output = tmp_path / 'acceptance'
    argv = ['verify_cleanup_effect.py', '--output', str(output),
            '--case', case, '--source', str(video), '--material-note', 'test-only failure input']
    expectations = {'source_sha256': module.sha256(video), 'request_goal': 'Remove captions; keep the logo',
                    'objects': [{'action': 'remove', 'description': 'Bilingual captions'},
                                {'action': 'keep', 'description': 'Corner logo'}]}
    if case == 'real_video':
        expectation_path = tmp_path / 'expectations.json'
        expectation_path.write_text(json.dumps(expectations), encoding='utf-8')
        argv += ['--expectations', str(expectation_path)]
    monkeypatch.setattr(sys, 'argv', argv)
    def fail_upload(*args, **kwargs):
        raise RuntimeError('upload failed deliberately')
    monkeypatch.setattr(TestClient, 'post', fail_upload)
    monkeypatch.setenv('VIDEOWIPE_JOBS_DIR', str(output / 'jobs'))
    with pytest.raises(RuntimeError, match='upload failed deliberately'):
        module.main()
    row = json.loads((output / 'report.json').read_text(encoding='utf-8'))['cases'][0]
    assert row['source_sha256'] == module.sha256(video)
    assert row['source_path'] == str(video)
    assert row['expected_objects']
    assert row['stage'] == 'upload'
    assert row['effect_review']['status'] == 'unverified'
    assert 'full_sha256' not in row
    assert json.loads((output / case / 'case.json').read_text(encoding='utf-8')) == row
    if case == 'real_video':
        assert row['expected_objects'] == expectations['objects']
        assert row['request_goal'] == expectations['request_goal']
        assert row['review_expectations'] == expectations


@pytest.mark.parametrize('invalid', ['missing', 'wrong-source', 'no-keep'])
def test_real_case_rejects_missing_or_unbound_expectations(tmp_path, monkeypatch, invalid):
    import json
    import sys
    video = tmp_path / 'real.mp4'
    video.write_bytes(b'independent real input')
    output = tmp_path / 'result'
    argv = ['runner', '--case', 'real_video', '--source', str(video),
            '--material-note', 'real input', '--output', str(output)]
    if invalid != 'missing':
        value = {'source_sha256': 'wrong' if invalid == 'wrong-source' else module.sha256(video),
                 'request_goal': 'Remove captions',
                 'objects': [{'action': 'remove', 'description': 'Captions'}]}
        path = tmp_path / 'expectations.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        argv += ['--expectations', str(path)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize('edits', [{}, {'english1': {}}])
def test_correction_rejects_empty_edits_before_creating_output(tmp_path, monkeypatch, edits):
    import json
    import sys
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'cases': [{'case': 'english1'}]}))
    edits_path = tmp_path / 'edits.json'
    edits_path.write_text(json.dumps(edits))
    output = tmp_path / 'corrected'
    monkeypatch.setattr(sys, 'argv', ['verify_cleanup_effect.py', '--output', str(output),
        '--baseline', str(baseline), '--edits', str(edits_path)])
    from fastapi.testclient import TestClient
    def unexpected_upload(*args, **kwargs):
        raise AssertionError('invalid correction must not upload')
    monkeypatch.setattr(TestClient, 'post', unexpected_upload)
    monkeypatch.setenv('VIDEOWIPE_JOBS_DIR', str(output / 'jobs'))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not output.exists()
