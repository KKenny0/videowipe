"""Run real cleanup routes and record effects separately from execution success.

Maintenance only: default selection -> recommended trial -> full export, optionally
one saved review edit for a previously failed case. No model/detector mocks.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from videowipe.plan import load_wipe_plan

ROOT = Path(__file__).resolve().parents[1]
CASES = ('chinese1', 'english1', 'others', 'similar_scene', 'screenshot_scene', 'real_video')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def case_annotations(video, source_sha256):
    facts = json.loads((ROOT / 'input/detext_examples/fact_baseline.json').read_text(encoding='utf-8'))
    return next((v for v in facts['videos']
                 if v['file'] == video.name and v.get('source_sha256') == source_sha256), None)


def load_expectations(path, source):
    """Read source-bound review expectations, never execution overrides."""
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('source_sha256') != sha256(source):
        raise ValueError('expectations must identify the exact source')
    objects = value.get('objects')
    if (not isinstance(value.get('request_goal'), str) or not value['request_goal'].strip()
            or not isinstance(objects, list) or not objects
            or any(not isinstance(obj, dict) or obj.get('action') not in {'remove', 'keep'}
                   or not isinstance(obj.get('description'), str) or not obj['description'].strip()
                   for obj in objects)
            or {obj['action'] for obj in objects} != {'remove', 'keep'}):
        raise ValueError('expectations require a goal and described remove/keep objects')
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def checked(response):
    response.raise_for_status()
    return response.json()


def wait_job(client, job_id, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = checked(client.get(f'/jobs/{job_id}'))
        if row.get('error') or row['state'] in {'error', 'cancelled', 'interrupted'}:
            raise RuntimeError(row)
        if row['state'] in {'preview_ready', 'done'}:
            return row
        time.sleep(0.5)
    raise TimeoutError(f'job {job_id} did not complete within {timeout}s')


def plan_identity(plan):
    # Source path and persisted NPZ checksum are not execution semantics.
    digest = hashlib.sha256()
    for track in sorted(plan.tracks, key=lambda t: t.id):
        digest.update(json.dumps(track.to_dict(), sort_keys=True).encode())
        digest.update(np.ascontiguousarray(track.mask).tobytes())
    return digest.hexdigest()


def validate_review(review, row):
    """Never infer usable output from a successful export or a still image."""
    if review['status'] not in {'pass', 'fail', 'unverified'}:
        raise ValueError('review status must be pass, fail, or unverified')
    if review['status'] == 'unverified':
        return
    if not review.get('reviewer') or not review.get('note'):
        raise ValueError('reviewer and note are required')
    if review.get('source_sha256') != row['source_sha256'] or review.get('output_sha256') != row['full_sha256']:
        raise ValueError('review must identify the exact source and output')
    if review['status'] == 'pass':
        if not review.get('normal_playback') or not review.get('entire_output_reviewed'):
            raise ValueError('pass requires normal playback of the entire output')
        if any(review.get(key) is not False for key in ('wrong_target', 'residue', 'false_removal', 'unusable_distortion')):
            raise ValueError('pass requires explicit absence of effect failures')
    elif not review.get('problems'):
        raise ValueError('fail requires problem intervals and descriptions')
    for problem in review.get('problems', []):
        first, last = problem['frames']
        if not (type(first) is int and type(last) is int and 0 <= first < last <= row['source']['frame_count']):
            raise ValueError('problem frames must be a valid half-open source interval')
        if not problem.get('kind') or not problem.get('description'):
            raise ValueError('problem kind and description required')


def check_protected_pixels(original, rendered, protections, frame_index):
    checked_pixels = 0
    for region in protections:
        if any(first <= frame_index < last for first, last in region['segments']):
            x1, y1, x2, y2 = region['bbox']
            assert np.array_equal(original[y1:y2+1, x1:x2+1], rendered[y1:y2+1, x1:x2+1]), 'protected pixels changed'
            checked_pixels += (x2-x1+1) * (y2-y1+1)
    return checked_pixels


def media_check(path, width, height, frames):
    cap = cv2.VideoCapture(str(path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
        count = 0
        while cap.read()[0]:
            count += 1
        assert count == frames, (path, count, frames)
    finally:
        cap.release()
    return count


def render_page(output, report):
    cards = []
    for row in report['cases']:
        if row.get('error'):
            cards.append(f'<h2>{html.escape(row["case"])}</h2><pre>{html.escape(row["error"])}</pre>')
            continue
        def local(path):
            return html.escape(os.path.relpath(path, output), quote=True)
        cards.append(f'''<section><h2>{html.escape(row['case'])}</h2>
<p>{html.escape(row['request_goal'])}。来源：{html.escape(row['material_kind'])}。</p>
<p>当前效果判断：{html.escape(row['effect_review']['status'])}。上方原片、下方结果。请正常播放完整视频检查漏擦与误擦。</p>
<video controls preload="metadata" src="{local(row['comparison'])}"></video>
<p><a href="{local(row['source_path'])}">原片</a> · <a href="{local(row['full'])}">完整输出</a> · <a href="{local(row['trial'])}">推荐试擦</a></p>
<p>修改轮数：{row['correction_rounds']}；导出与试擦编码前一致：{row['checks']['trial_full_preencode_equal']}。</p></section>''')
    (output / 'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>VideoWipe 清理效果验收</title>
<style>body{font:17px/1.6 system-ui;max-width:1050px;margin:30px auto;padding:0 18px;background:#151719;color:#eee}a{color:#9bd2ff}video{width:100%;max-height:82vh;background:black}section{border-top:1px solid #555;margin-top:32px}pre{white-space:pre-wrap}</style>
<h1>清理效果验收</h1><p>去掉该去的，保留该留的。执行成功不代表效果通过。轻微纹理差异不单独判失败，明显影响使用的变形仍需记录。</p>
<p>本页是维护验收材料；素材性质以各案例来源为准，没有自动质量评分。</p>''' + ''.join(cards), encoding='utf-8')


def run_case(client, web, name, output, capture, observation, baseline=None, edit=None, source=None, material_note=None, row=None, expectations=None):
    row = {} if row is None else row
    row.update(case=name, stage='input', correction_rounds=0, edit=edit,
               request_goal='移除字幕，保留非目标内容及顶部标记',
               material_kind=material_note or (baseline['material_kind'] if baseline else 'synthetic' if name == 'similar_scene' else 'repository calibration sample'),
               effect_review={'status': 'unverified', 'note': 'Requires effect review; execution checks are separate.'})
    directory = output / name
    directory.mkdir()
    if baseline:
        video = Path(baseline['source_path'])
    elif source:
        video = source.resolve()
    elif name == 'similar_scene':
        from verify_similar_subtitles import generate
        video, _, _ = generate(directory / 'synthetic')
    else:
        video = ROOT / 'input/detext_examples' / f'{name}.mp4'
    row['source_path'] = str(video.resolve())
    row['source_sha256'] = sha256(video)
    annotated = case_annotations(video, row['source_sha256'])
    row['annotation_status'] = 'manifest content verified' if annotated else 'unannotated; expectations require review'
    row['expected_objects'] = annotated['objects'] if annotated else [
        {'type': 'subtitle', 'action': 'remove', 'description': 'Bottom overlay subtitles'},
        {'type': 'background', 'action': 'keep', 'description': 'Scene content, printed labels, and pixels outside the subtitle target'}]
    if expectations is None and baseline:
        expectations = baseline.get('review_expectations')
    if expectations:
        assert expectations['source_sha256'] == row['source_sha256'], 'expectation source changed'
        row.update(request_goal=expectations['request_goal'], expected_objects=expectations['objects'],
                   review_expectations=expectations,
                   annotation_status='source-bound review expectations; not execution overrides')
    if name == 'real_video' and not expectations:
        raise ValueError('real_video requires source-bound review expectations')
    if baseline:
        assert row['source_sha256'] == baseline['source_sha256'], 'baseline source changed'
    write_json(directory / 'case.json', row)
    started = time.monotonic()
    row['stage'] = 'upload'
    with video.open('rb') as stream:
        created = checked(client.post('/jobs', files={'video': (video.name, stream, 'video/mp4')}))
    job_id = created['id']
    row.update(job_id=job_id, stage='planning')
    ready = wait_job(client, job_id)
    preview = checked(client.get(f'/jobs/{job_id}/preview'))
    write_json(directory / 'preview.json', preview)
    job_dir = output / 'jobs' / job_id
    manifest = json.loads((job_dir / 'job.json').read_text(encoding='utf-8'))
    machine_path = job_dir / manifest['plan_dir'] / 'wipe_plan.json'
    machine = load_wipe_plan(str(machine_path))
    row.update(source=machine.source.to_dict(),
               machine_plan=str(machine_path), machine_plan_identity=plan_identity(machine),
               default_selected_ids=preview['default_selected_ids'],
               timings={'planning_s': time.monotonic() - started, **ready['timings']})
    assert row['source_sha256'] == machine.source.sha256, 'source changed during planning'
    body = {'selected_ids': preview['default_selected_ids']}
    revision = preview['review_revision']
    if edit:
        row['stage'] = 'correction'
        validate_review(baseline['effect_review'], baseline)
        assert baseline['effect_review']['status'] == 'fail', 'correction requires a real failure'
        assert baseline['correction_rounds'] == 0, 'only one correction round'
        assert row['machine_plan_identity'] == baseline['machine_plan_identity'], 'default plan drifted'
        body.update(edit)
        saved = checked(client.patch(f'/jobs/{job_id}/review', json={'expected_revision': revision, 'overrides': body}))
        revision = saved['review_revision']
        row['correction_rounds'] = 1
    row['stage'] = 'trial'
    interval = preview['recommended_trial']
    if not interval:
        raise RuntimeError('default plan has no recommended nonempty trial; record as a default-flow failure')
    payload = dict(body, start_seconds=interval['start_seconds'], duration_seconds=interval['duration_seconds'],
                   expected_revision=revision, operation_id=f'{name}-trial')
    row['recommended_trial'] = interval
    row['trial_request'] = payload
    observation.update(offset=interval['frame_range'][0], protections=body.get('protections', []))
    capture.clear()
    checked(client.post(f'/jobs/{job_id}/trial', json=payload))
    trial_done = wait_job(client, job_id)
    trial_capture = list(capture)
    assert trial_done['trial'] and trial_done['trial']['ready']
    assert not trial_done['trial']['cache_hit'], 'fresh acceptance must capture real trial frames'
    row['timings']['first_effect_s'] = time.monotonic() - started
    trial_manifest = json.loads((job_dir / 'job.json').read_text(encoding='utf-8'))
    trial_plan = load_wipe_plan(str(job_dir / trial_manifest['reviewed_plan_path']))
    observation['offset'] = 0
    capture.clear()
    row['stage'] = 'export'
    row['export_request'] = dict(body, expected_revision=trial_done['review_revision'],
                                 operation_id=f'{name}-full')
    checked(client.post(f'/jobs/{job_id}/confirm', json=row['export_request']))
    done = wait_job(client, job_id)
    assert done['state'] == 'done'
    row['timings']['full_ready_s'] = time.monotonic() - started
    row['stage'] = 'execution_checks'
    full_capture = list(capture)
    manifest = json.loads((job_dir / 'job.json').read_text(encoding='utf-8'))
    executed_path = job_dir / manifest['reviewed_plan_path']
    executed = load_wipe_plan(str(executed_path))
    first, last = interval['frame_range']
    parity = full_capture[first:last] == trial_capture
    assert len(full_capture) == machine.source.frame_count
    assert len(trial_capture) == last - first
    assert plan_identity(trial_plan) == plan_identity(executed), 'trial/full execution plans differ'
    trial_path = job_dir / manifest['trial_path']
    full_path = Path(done['result_path'])
    media_check(trial_path, machine.source.width, machine.source.height * 2, last-first)
    media_check(full_path, machine.source.width, machine.source.height, machine.source.frame_count)
    for endpoint, params in [('result-video', {}), ('trial-video', {'trial_id': done['trial']['id']})]:
        assert client.get(f'/jobs/{job_id}/{endpoint}', params=params, headers={'Range':'bytes=0-15'}).status_code == 206
    checks = {'trial_full_preencode_equal': parity, 'trial_full_plan_equal': True,
              'parity_mismatch_frames': [i + first for i, (a, b) in enumerate(zip(full_capture[first:last], trial_capture)) if a != b],
              'zero_alpha_preserved': True, 'full_decoded_frames': len(full_capture),
              'empty_execution_frames': sum(n['alpha_pixels'] == 0 for n in full_capture),
              'explicit_protected_pixels_checked': sum(n['protected_pixels_checked'] for n in full_capture)}
    # Only content-bound fixture annotations can support semantic keep checks.
    assert row['source_sha256'] == sha256(video), 'source changed during the run'
    if annotated:
        from videowipe.plan import execution_masks
        keep_ids = [v['id'] for v in annotated['objects'] if v['action'] == 'keep']
        static_alpha, alpha_at = execution_masks(executed, web._get_engine()._task_impl.feather_radius)
        overlaps = []
        for frame in annotated['frames']:
            labels = cv2.imread(str(ROOT / 'input/detext_examples' / frame['mask']), cv2.IMREAD_UNCHANGED)
            keep = np.isin(labels, keep_ids)
            alpha = np.asarray(alpha_at(frame['frame']) if alpha_at else static_alpha).squeeze()
            overlaps.append({'frame': frame['frame'], 'keep_pixels': int(keep.sum()),
                             'keep_alpha_pixels': int(((alpha > 0) & keep).sum())})
        checks['annotated_keep_overlap'] = overlaps
    row.update(full=str(full_path), full_sha256=sha256(full_path), trial=str(trial_path),
               executed_plan=str(executed_path), executed_plan_sha256=sha256(executed_path),
               runtime=done['device'], warnings=done['warnings'], checks=checks, stage='comparison')
    comparison = directory / 'comparison.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video), '-i', str(full_path),
        '-filter_complex', '[0:v]scale=960:-2[a];[1:v]scale=960:-2[b];[a][b]vstack=inputs=2[v]',
        '-map', '[v]', '-map', '0:a?', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', str(comparison)], check=True)
    row.update(comparison=str(comparison), stage='complete')
    write_json(directory / 'frames.json', {'trial_range': [first,last], 'trial': trial_capture, 'full': full_capture})
    write_json(directory / 'case.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, action='append')
    parser.add_argument('--source', type=Path, help='Explicit source for one non-repository case')
    parser.add_argument('--material-note', help='Required provenance description for an explicit source')
    parser.add_argument('--expectations', type=Path, help='Source-bound JSON goal and remove/keep objects for review only')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--edits', type=Path, help='JSON mapping case names to existing review overrides')
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.edits):
        parser.error('--baseline and --edits must be provided together')
    output = args.output.resolve()
    edits = json.loads(args.edits.read_text(encoding='utf-8')) if args.edits else {}
    baselines = {row['case']: row for row in json.loads(args.baseline.read_text(encoding='utf-8'))['cases']} if args.baseline else {}
    if args.edits and (not isinstance(edits, dict) or not edits
                       or any(not isinstance(edit, dict) or not edit for edit in edits.values())):
        parser.error('correction requires a nonempty mapping of nonempty review edits')
    names = args.case or list(edits) or list(CASES[:4])
    if args.source and (len(names) != 1 or not args.material_note or args.baseline):
        parser.error('--source requires one case and --material-note, without --baseline')
    if any(name in names for name in ('screenshot_scene', 'real_video')) and not (args.source or args.baseline):
        parser.error('external cases require an explicit source or baseline')
    if args.expectations and (not args.source or args.baseline):
        parser.error('--expectations requires an explicit source without a baseline')
    if 'real_video' in names and not (args.expectations or args.baseline):
        parser.error('real_video requires --expectations')
    expectations = None
    if args.expectations:
        try:
            expectations = load_expectations(args.expectations, args.source)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    if edits and set(names) != set(edits):
        parser.error('each correction case must have exactly one saved review edit')
    if any(name not in CASES for name in names) or len(set(names)) != len(names):
        parser.error('cases must be distinct supported case names')
    if args.baseline:
        if any(name not in baselines for name in names):
            parser.error('each correction needs a matching baseline case')
        for name in names:
            baseline = baselines[name]
            validate_review(baseline['effect_review'], baseline)
            if baseline['effect_review']['status'] != 'fail' or baseline['correction_rounds'] != 0:
                parser.error('correction requires a reviewed default failure with zero previous corrections')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['VIDEOWIPE_JOBS_DIR'] = str(output / 'jobs')
    from fastapi.testclient import TestClient
    from videowipe.server import app as web
    from videowipe.inpainters import sttn
    implementation = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'src/videowipe').rglob('*.py'))}
    report = {'scope': 'real product HTTP handlers via in-process transport; real detector and STTN; not a browser usability test',
              'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'git_status': subprocess.check_output(['git','status','--short'], text=True),
              'runner_sha256': sha256(__file__), 'implementation': implementation,
              'baseline': str(args.baseline.resolve()) if args.baseline else None, 'cases': []}
    (output / 'runner.py').write_bytes(Path(__file__).read_bytes())
    capture = []
    observation = {'offset': 0, 'protections': []}
    original_blend = sttn._blend_frame_regions
    def observe(original, predictions, modes, alpha):
        frame = original_blend(original, predictions, modes, alpha)
        zero = np.asarray(alpha).squeeze() == 0
        assert np.array_equal(frame[zero], original[zero]), 'zero-alpha pixels changed before encoding'
        protected = check_protected_pixels(original, frame, observation['protections'], observation['offset'] + len(capture))
        capture.append({'protected_pixels_checked': protected, 'sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                        'alpha_sha256': hashlib.sha256(np.ascontiguousarray(alpha).tobytes()).hexdigest(),
                        'alpha_pixels': int(np.count_nonzero(alpha))})
        return frame
    sttn._blend_frame_regions = observe
    try:
        with TestClient(web.app) as client:
            for name in names:
                print(f'START {name}', flush=True)
                row = {'case': name, 'stage': 'input', 'effect_review': {'status': 'unverified'}}
                try:
                    run_case(client, web, name, output, capture, observation, baselines.get(name), edits.get(name), args.source, args.material_note, row=row, expectations=expectations)
                except Exception as exc:
                    row['error'] = repr(exc)
                    report['cases'].append(row)
                    if (output / name).is_dir():
                        write_json(output / name / 'case.json', row)
                    write_json(output/'report.json', report)
                    render_page(output, report)
                    raise
                report['cases'].append(row)
                write_json(output/'report.json', report)
                render_page(output, report)
                print(f'DONE {name}: {row["timings"]["full_ready_s"]:.1f}s, effect review pending', flush=True)
        assert implementation == {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'src/videowipe').rglob('*.py'))}
    finally:
        sttn._blend_frame_regions = original_blend
        if web._engine:
            web._engine.cleanup()


if __name__ == '__main__':
    main()
