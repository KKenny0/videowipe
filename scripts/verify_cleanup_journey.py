"""Measure real local HTTP cleanup journeys with interleaved old/new runs.

The three repository samples test ordinary journeys. --long-seconds generates
one repeated English pressure input; it tests duration/resources, not diversity.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

import cv2
import httpx
import numpy as np

from videowipe.plan import compute_source, load_wipe_plan

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def plan_fingerprint(plan):
    digest = hashlib.sha256()
    for track in sorted(plan.tracks, key=lambda track: track.id):
        digest.update(json.dumps(track.to_dict(), sort_keys=True).encode())
        digest.update(np.ascontiguousarray(track.mask).tobytes())
    return digest.hexdigest()


def package_identity():
    import videowipe
    package = Path(videowipe.__file__).parent
    return {str(path.relative_to(package)): sha256(path)
            for path in sorted(package.rglob("*.py"))}


def serve(args):
    """Test-only counters; these routes never enter the distributed app."""
    import uvicorn
    from videowipe.server import app as web
    from videowipe.inpainters import sttn
    from videowipe.detect import DBNetDetector
    os.environ["VIDEOWIPE_JOBS_DIR"] = str(args.output / args.serve / "jobs")
    implementation = package_identity()
    calls = 0
    detector_calls = 0
    original_detect = DBNetDetector.detect
    original = sttn._process_segment

    def count_segment(*values, **options):
        nonlocal calls
        calls += 1
        return original(*values, **options)

    def count_detection(detector, frame):
        nonlocal detector_calls
        detector_calls += 1
        return original_detect(detector, frame)

    DBNetDetector.detect = count_detection
    sttn._process_segment = count_segment

    @web.app.get("/__acceptance/identity")
    def identity():
        backend = web._engine._task_impl.backend if web._engine else None
        runtime = None
        if backend is not None:
            runtime = backend.benchmark_metadata() if hasattr(backend, "benchmark_metadata") else {
                "device": backend.encoder_session.get_providers(),
                "weight_sha256": {part: sha256(getattr(backend, part + "_session")._model_path)
                                  for part in ("encoder", "transformer", "decoder")},
            }
            runtime["backend"] = type(backend).__name__
            runtime["precision"] = "float16-autocast" if str(runtime["device"]).startswith("cuda") else "float32"
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory = {"peak_rss_bytes": rss * (1 if sys.platform == "darwin" else 1024)}
        if backend is not None and str(getattr(backend, "device", "")) == "mps":
            memory["mps_allocated_bytes"] = backend._torch.mps.current_allocated_memory()
            memory["mps_driver_bytes"] = backend._torch.mps.driver_allocated_memory()
        return {"model_segment_calls": calls, "detector_calls": detector_calls, "runtime": runtime, "memory": memory,
                "implementation": implementation, "source_unchanged": implementation == package_identity()}

    uvicorn.run(web.app, host="127.0.0.1", port=args.port, log_level="warning")


def wait_job(client, job_id):
    while True:
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        data = response.json()
        if data["state"] in {"preview_ready", "done"}:
            if data.get("error"):
                raise RuntimeError(data)
            return data
        if data["state"] in {"error", "cancelled", "interrupted"}:
            raise RuntimeError(data)
        time.sleep(0.5)  # Match the product SSE cadence, using one persistent HTTP client.


def media_check(path, source, frames=None, dual=False):
    reader = cv2.VideoCapture(str(path))
    try:
        actual = [int(reader.get(key)) for key in (
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)]
        assert actual == [source.width, source.height * (2 if dual else 1), frames or source.frame_count], actual
        reader.set(cv2.CAP_PROP_POS_FRAMES, actual[2] - 1)
        assert reader.read()[0], path
    finally:
        reader.release()


def journey(client, label, name, video, repetition, cold, output):
    started = time.perf_counter()
    marks = {"file_selected_s": 0.0}
    with video.open("rb") as stream:
        created = client.post("/jobs", files={"video": (video.name, stream, "video/mp4")})
    created.raise_for_status()
    job_id = created.json()["id"]
    marks["copy_complete_s"] = time.perf_counter() - started
    ready = wait_job(client, job_id)
    preview = client.get(f"/jobs/{job_id}/preview").json()
    marks["final_plan_s"] = marks["copy_complete_s"] + ready["timings"]["plan_s"]
    marks["candidates_reviewable_s"] = (marks["copy_complete_s"] + ready["timings"]["candidates_s"]
                                        if label == "current" else marks["final_plan_s"])
    if label == "current":
        assert marks["candidates_reviewable_s"] < marks["final_plan_s"]
    interval = preview["recommended_trial"]
    assert interval, name
    body = {"selected_ids": preview["default_selected_ids"],
            "start_seconds": interval["start_seconds"], "duration_seconds": interval["duration_seconds"],
            "operation_id": f"first-{name}-{repetition}"}
    client.post(f"/jobs/{job_id}/trial", json=body).raise_for_status()
    trial = wait_job(client, job_id)["trial"]
    assert trial and trial["ready"] and not trial.get("cache_hit", False)
    playable = client.get(f"/jobs/{job_id}/trial-video", params={"trial_id": trial["id"]}, headers={"Range": "bytes=0-15"})
    assert playable.status_code == 206
    marks["first_trial_playable_s"] = time.perf_counter() - started
    cached_s, cache_calls, cache_detector_calls = 0.0, None, None
    if label == "current":
        before = client.get("/__acceptance/identity").json()
        cache_started = time.perf_counter()
        body["operation_id"] = f"repeat-{name}-{repetition}"
        client.post(f"/jobs/{job_id}/trial", json=body).raise_for_status()
        hit = wait_job(client, job_id)["trial"]
        cached_s = time.perf_counter() - cache_started
        after = client.get("/__acceptance/identity").json()
        cache_calls = after["model_segment_calls"] - before["model_segment_calls"]
        cache_detector_calls = after["detector_calls"] - before["detector_calls"]
        assert hit["cache_hit"] and cache_calls == cache_detector_calls == 0, hit
    client.post(f"/jobs/{job_id}/confirm", json={"selected_ids": body["selected_ids"],
        "operation_id": f"full-{name}-{repetition}"}).raise_for_status()
    done = wait_job(client, job_id)
    assert done["state"] == "done"
    assert client.get(f"/jobs/{job_id}/result-video", headers={"Range": "bytes=0-15"}).status_code == 206
    marks["full_playable_s"] = time.perf_counter() - started - cached_s
    folder = output / label / "jobs" / job_id
    if label == "current":
        manifest = json.loads((folder / "job.json").read_text())
        plan_path = folder / manifest["plan_dir"] / "wipe_plan.json"
        trial_path = folder / manifest["trial_path"]
    else:
        plan_path = folder / "wipe_plan.json"
        trial_path = next((folder / "trials" / trial["id"]).glob("*.mp4"))
    plan = load_wipe_plan(str(plan_path))
    media_check(trial_path, plan.source, interval["frame_range"][1] - interval["frame_range"][0], True)
    media_check(done["result_path"], plan.source)
    identity = client.get("/__acceptance/identity").json()
    assert identity["source_unchanged"], "implementation changed during acceptance"
    row = {"sample": name, "repeat": repetition, "temperature": "cold" if cold else "warm",
           "source": plan.source.to_dict(), "marks": marks, "cache_hit_seconds": cached_s,
           "cache_model_calls": cache_calls, "cache_detector_calls": cache_detector_calls, "plan_json_sha256": sha256(plan_path),
           "execution_evidence_sha256": plan_fingerprint(plan), "runtime": identity["runtime"],
           "memory": identity["memory"], "implementation": identity["implementation"],
           "trial": str(trial_path), "full": done["result_path"]}
    golden = ROOT / "input/detext_examples/mask" / f"{name}_mask.png"
    if golden.is_file():
        expected = cv2.imread(str(golden), cv2.IMREAD_GRAYSCALE) > 0
        predicted = cv2.imread(str(plan_path.parent / "auto_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        row["golden_mask_iou"] = float(np.logical_and(expected, predicted).sum() / max(1, np.logical_or(expected, predicted).sum()))
    print(f"{label} {name} #{repetition}: {marks['full_playable_s']:.2f}s; cache {cached_s:.2f}s", flush=True)
    return row


def revision_acceptance(args):
    """Paired SDK exports from automatic plans; hash actual frames before FFmpeg."""
    from copy import deepcopy
    from videowipe import WipeEngine, WipeRequest
    from videowipe.plan import save_wipe_plan, execution_masks
    from videowipe.server.review import compile_review
    from videowipe.inpainters import sttn
    implementation = package_identity()
    calls = 0
    original_prediction, original_blend = sttn._process_segment, sttn._blend_frame_regions
    def count_prediction(*values, **options):
        nonlocal calls
        calls += 1
        return original_prediction(*values, **options)
    sttn._process_segment = count_prediction
    report = {'mode': 'revision', 'repeat': args.repeat, 'implementation': implementation,
              'scope': 'automatic plans; warm loaded STTN; cold/warm prediction cache; paired SDK exports',
              'rows': [], 'comparisons': []}
    def export(engine, plan, directory, cache):
        nonlocal calls
        directory.mkdir(parents=True, exist_ok=True)
        saved, _ = save_wipe_plan(plan, str(directory))
        digest, frame_index = hashlib.sha256(), 0
        protected_pixels = 0
        def capture(frame_ori, crops, modes, alpha):
            nonlocal frame_index, protected_pixels
            frame = original_blend(frame_ori, crops, modes, alpha)
            for region in plan.tracks:
                if region.action == 'protect' and any(segment.contains(frame_index) for segment in region.segments):
                    protected = region.mask.astype(bool)
                    assert np.array_equal(frame[protected], frame_ori[protected]), ('protection changed', frame_index)
                    protected_pixels += int(protected.sum())
            digest.update(frame.tobytes())
            frame_index += 1
            return frame
        sttn._blend_frame_regions = capture
        before = calls
        started = time.perf_counter()
        result = engine.run(WipeRequest(video=video, plan=saved, output_dir=directory, prediction_cache_dir=cache))
        elapsed = time.perf_counter() - started
        assert frame_index == plan.source.frame_count
        media_check(result.output_path, plan.source)
        return {'seconds': elapsed, 'model_segment_calls': calls-before, 'preencode_sha256': digest.hexdigest(),
                'protected_pixels_checked': protected_pixels, 'timings': dict(result.timings), 'full': result.output_path,
                'plan_sha256': sha256(saved), 'execution_evidence_sha256': plan_fingerprint(plan)}
    try:
        with WipeEngine(task='clean') as engine:
            for name in ('english1', 'chinese1', 'others'):
                video = ROOT / 'input/detext_examples' / f'{name}.mp4'
                automatic = engine.plan(WipeRequest(video=video, output_dir=args.output / name / 'automatic'))
                identity = engine._trial_identity()
                target = max(automatic.remove_tracks, key=lambda track: sum(s.end-s.start for s in track.segments))
                intervals = [segment.to_dict() for segment in target.segments]
                longest = max(range(len(intervals)), key=lambda i: intervals[i][1]-intervals[i][0])
                assert intervals[longest][1] - intervals[longest][0] >= 3
                intervals[longest][0] += 1
                selected = [track.id for track in automatic.remove_tracks]
                x1,y1,x2,y2 = target.bbox
                px1, px2 = x1 + (x2-x1)//3, x1 + 2*(x2-x1)//3
                assert px2 > px1 and y2 > y1
                protection = {'id': 'p_acceptance', 'bbox': [px1,y1,px2,y2],
                              'segments': [[intervals[longest][0], min(intervals[longest][1], intervals[longest][0]+max(1,round(automatic.source.fps)))]]}
                edited, _ = compile_review(deepcopy(automatic), selected,
                    segment_overrides={target.id: intervals}, protections=[protection])
                first_union = execution_masks(automatic, engine._task_impl.feather_radius)[0]
                edited_union = execution_masks(edited, engine._task_impl.feather_radius)[0]
                split = int(automatic.source.width * 3 / 16)
                assert sttn.get_inpaint_mode(automatic.source.height, split, first_union) == sttn.get_inpaint_mode(edited.source.height, split, edited_union)
                for repetition in range(args.repeat):
                    cache = args.output / 'cache' / f'{name}-{repetition}' / 'predictions'
                    assert not cache.exists(), 'use a new output directory for cold-cache acceptance'
                    directory = args.output / name / str(repetition)
                    row = {'sample': name, 'repeat': repetition, 'source': automatic.source.to_dict(), 'runtime': identity,
                           'edits': {'segment_overrides': {target.id: intervals}, 'protections': [protection]}}
                    order = ('baseline', 'cold_cache') if repetition % 2 == 0 else ('cold_cache', 'baseline')
                    for label in order:
                        row[label] = export(engine, automatic, directory / label, cache if label == 'cold_cache' else None)
                    assert row['baseline']['preencode_sha256'] == row['cold_cache']['preencode_sha256']
                    order = ('edited_cached', 'edited_fresh') if repetition % 2 == 0 else ('edited_fresh', 'edited_cached')
                    for label in order:
                        row[label] = export(engine, edited, directory / label, cache if label == 'edited_cached' else None)
                    assert row['edited_cached']['model_segment_calls'] == 0
                    assert row['edited_cached']['preencode_sha256'] == row['edited_fresh']['preencode_sha256']
                    assert row['edited_cached']['protected_pixels_checked'] > 0
                    row['cache_bytes'] = sum(path.stat().st_size for path in cache.glob('*.npz'))
                    report['rows'].append(row)
                    write_report(args.output / 'report.json', report)
                    print(f"{name} round {repetition+1}: cached {row['edited_cached']['seconds']:.2f}s, fresh {row['edited_fresh']['seconds']:.2f}s", flush=True)
                rows = [row for row in report['rows'] if row['sample'] == name]
                medians = {label: statistics.median(row[label]['seconds'] for row in rows)
                           for label in ('baseline','cold_cache','edited_cached','edited_fresh')}
                report['comparisons'].append({'sample': name, 'medians': medians,
                    'cold_overhead_ratio': medians['cold_cache']/medians['baseline'],
                    'revision_ratio': medians['edited_cached']/medians['edited_fresh'],
                    'within_10_percent': medians['cold_cache'] <= medians['baseline'] * 1.1,
                    'at_least_30_percent_faster': medians['edited_cached'] <= medians['edited_fresh'] * .7})
                write_report(args.output / 'report.json', report)
        assert package_identity() == implementation, 'source changed during acceptance'
        report['source_unchanged'] = True
        report['enable_prediction_cache'] = all(row['within_10_percent'] and row['at_least_30_percent_faster'] for row in report['comparisons'])
        write_report(args.output / 'report.json', report)
    finally:
        sttn._process_segment, sttn._blend_frame_regions = original_prediction, original_blend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["journey", "revision"], default="journey")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--long-seconds", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("result/product-journey"))
    parser.add_argument("--baseline-ref", default="90f5fa6")
    parser.add_argument("--serve", choices=["baseline", "current"], help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1 or args.long_seconds < 0:
        parser.error("repeat must be positive and long-seconds nonnegative")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "revision":
        if args.long_seconds or args.serve:
            parser.error("revision mode uses the three ordinary samples")
        revision_acceptance(args)
        return
    if args.serve:
        serve(args)
        return
    samples = [(name, ROOT / "input/detext_examples" / f"{name}.mp4") for name in ("english1", "chinese1", "others")]
    labels = ["baseline", "current"]
    if args.long_seconds:
        pressure = args.output / "english-repeat-pressure.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(samples[0][1]),
            "-t", str(args.long_seconds), "-vf", "fps=25", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", str(pressure)], check=True)
        assert compute_source(str(pressure)).frame_count == args.long_seconds * 25
        audio = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", str(pressure)])
        assert json.loads(audio)["streams"], "pressure input lost audio"
        samples, labels = [("english-repeat-pressure", pressure)], ["current"]
    ref = subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip()
    rows = {label: [] for label in labels}
    processes, clients, logs = [], {}, []
    with tempfile.TemporaryDirectory(prefix="videowipe-journey-") as temporary:
        package = Path(temporary)
        if "baseline" in labels:
            archive = subprocess.check_output(["git", "archive", ref, "src/videowipe"], cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                for member in tar:
                    path = (package / member.name).resolve()
                    path.relative_to(package.resolve())
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(tar.extractfile(member).read())
                    else:
                        raise ValueError("baseline contains non-regular source entries")
        try:
            for label in labels:
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                source_root = package / "src" if label == "baseline" else ROOT / "src"
                log = (args.output / f"{label}-server.log").open("w")
                logs.append(log)
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--serve", label,
                    "--port", str(port), "--output", str(args.output)],
                    env=dict(os.environ, PYTHONPATH=str(source_root)), stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60)
                clients[label] = client
                deadline = time.monotonic() + 30
                while True:
                    try:
                        client.get("/jobs/current").raise_for_status()
                        client.delete("/jobs/current").raise_for_status()
                        break
                    except httpx.HTTPError:
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(f"{label} service failed; inspect its log")
                        time.sleep(0.1)
            for name, video in samples:
                for repetition in range(args.repeat):
                    order = labels if repetition % 2 == 0 else list(reversed(labels))
                    for label in order:
                        rows[label].append(journey(clients[label], label, name, video, repetition,
                                                   not rows[label], args.output))
                        write_report(args.output / f"{label}.json", rows[label])
        finally:
            for client in clients.values():
                client.close()
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
    comparisons = []
    if "baseline" in rows:
        for name, _ in samples:
            before = [row for row in rows["baseline"] if row["sample"] == name]
            after = [row for row in rows["current"] if row["sample"] == name]
            for a, b in zip(before, after):
                assert a["source"]["sha256"] == b["source"]["sha256"]
                assert a["execution_evidence_sha256"] == b["execution_evidence_sha256"]
                assert a["runtime"] == b["runtime"]
            old = statistics.median(row["marks"]["full_playable_s"] for row in before)
            new = statistics.median(row["marks"]["full_playable_s"] for row in after)
            comparisons.append({"sample": name, "baseline_median_s": old, "current_median_s": new,
                                "ratio": new / old, "within_5_percent": new <= old * 1.05})
    report = {"mode": "journey", "baseline_commit": ref if comparisons else None,
              "repeat": args.repeat, "pressure_only": bool(args.long_seconds), "comparisons": comparisons,
              **rows, "script_sha256": sha256(__file__),
              "measurement": "Both versions use persistent HTTP services and 0.5s polling. Paired order alternates. Full journey excludes the separate cache-hit check; browser rendering and user thinking time are not simulated. Pressure input tests only duration/resources; the ordinary samples carry the regression gate."}
    write_report(args.output / "report.json", report)
    assert all(row["within_5_percent"] for row in comparisons), comparisons


if __name__ == "__main__":
    main()
