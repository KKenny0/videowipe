import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_pipeline.py"
    spec = importlib.util.spec_from_file_location("benchmark_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_peak_rss_is_optional_without_resource(monkeypatch):
    script = _load_script()
    monkeypatch.setattr(script, "resource", None)
    assert script._process_peak_rss_so_far_mib() is None


def test_file_input_repeat_isolated_and_median_aggregated(tmp_path, monkeypatch):
    script = _load_script()
    video = tmp_path / "chinese1.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4,
        (32, 24),
    )
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    assert script._find_videos(str(tmp_path)) == [str(video)]

    calls = []

    class StubEngine:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def process(self, video, mask, output):
            run = int(Path(output).name.split("-")[-1])
            Path(output).mkdir(parents=True, exist_ok=True)
            values = {1: 10, 2: 30, 3: 20}
            value = values[run]
            (Path(output) / "benchmark.json").write_text(
                json.dumps(
                    {
                        "timing": {
                            "total_s": value,
                            "detection_s": value / 10,
                            "model_load_s": value / 5,
                            "inpainting_s": value / 2,
                        }
                    }
                )
            )

        def cleanup(self):
            pass

    output = tmp_path / "benchmark"
    monkeypatch.setattr(script, "WipeEngine", StubEngine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_pipeline.py",
            str(video),
            "--output-dir",
            str(output),
            "--repeat",
            "3",
            "--gap",
            "17",
            "--ocr",
            "off",
        ],
    )

    script.main()

    report = json.loads((output / "benchmark_report.json").read_text())
    result = report["results"][0]
    assert [Path(run["output_dir"]).name for run in result["runs"]] == [
        "run-001",
        "run-002",
        "run-003",
    ]
    assert result["median_timing_s"] == {
        "total_s": 20.0,
        "detection_s": 2.0,
        "model_load_s": 4.0,
        "inpainting_s": 10.0,
    }
    assert result["input"] == {
        "path": str(video),
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "width": 32,
        "height": 24,
        "frame_count": 2,
        "fps": 4.0,
    }
    assert report["config"] == {
        "detect_mode": "balanced",
        "ocr": "off",
        "gap": 17,
        "repeat": 3,
    }
    assert all(
        run["process_peak_rss_so_far_mib"] > 0 for run in result["runs"]
    )
    assert "peak_rss_mib" not in result["runs"][0]
    assert report["metric_notes"] == {
        "process_peak_rss_so_far_mib": (
            "Current benchmark process lifetime high-water RSS; repeated runs "
            "share this value, so repeat > 1 is not a per-run memory measurement. "
            "Null means the platform does not expose resource.getrusage."
        ),
    }
    assert all(call["gap"] == 17 for call in calls)
