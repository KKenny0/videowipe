import importlib.util
import json
import pathlib
import subprocess

import cv2
import numpy as np
import pytest

from videowipe.detect import CleanCandidate, CleanDetectionResult


def _load_eval_module():
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "eval_clean_detection.py"
    spec = importlib.util.spec_from_file_location("eval_clean_detection", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_video(path, width=12, height=8, frames=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def _candidate(mask, candidate_id="candidate-1", target_type="subtitle", default_remove=True):
    return CleanCandidate(
        id=candidate_id,
        type=target_type,
        label=target_type,
        bbox=(0, 0, mask.shape[1] - 1, mask.shape[0] - 1),
        confidence=1.0,
        frame_fraction=1.0,
        reason="test",
        default_remove=default_remove,
        mask=mask[..., None],
    )


def _write_manifest(root, mask, objects=None, frame=1):
    annotations = root / "annotations" / "sample"
    annotations.mkdir(parents=True)
    mask_path = annotations / f"{frame:06d}.png"
    cv2.imwrite(str(mask_path), mask)
    payload = {
        "schema_version": 1,
        "videos": [{
            "file": "sample.mp4",
            "objects": objects or [
                {"id": 1, "type": "subtitle", "action": "remove", "description": "remove"},
                {"id": 2, "type": "watermark", "action": "keep", "description": "keep"},
            ],
            "frames": [{"frame": frame, "mask": str(mask_path.relative_to(root))}],
        }],
    }
    manifest = root / "fact_baseline.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _fake_detector(module, predicted, candidates=None, selected=None):
    candidates = candidates or [_candidate(predicted)]
    selected = candidates if selected is None else selected
    module._detect_video = lambda *args, **kwargs: (
        CleanDetectionResult(candidates, predicted.shape), selected, predicted.astype(bool), "fake_detector"
    )


def test_jaccard_and_davis_boundary_reference_vectors():
    module = _load_eval_module()
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    disjoint = np.zeros_like(mask)
    disjoint[:3, :3] = 1
    assert module._compute_mask_iou(mask, mask) == pytest.approx(1.0)
    assert module._compute_mask_iou(mask, disjoint) == pytest.approx(0.0)
    assert module.compute_boundary_f(mask, mask) == pytest.approx(1.0)
    assert module.compute_boundary_f(mask, disjoint) == pytest.approx(0.0)

    # Official DAVIS 2017 reference convention: 100x100, 50px square,
    # translated diagonally by (2, 2), with ceil(0.008 * diagonal) = 2.
    reference = np.zeros((100, 100), dtype=np.uint8)
    shifted = np.zeros_like(reference)
    reference[20:70, 20:70] = 1
    shifted[22:72, 22:72] = 1
    assert module.compute_boundary_f(shifted, reference) == pytest.approx(0.985)


def test_evaluator_rejects_unsupported_opencv5(monkeypatch):
    module = _load_eval_module()
    monkeypatch.setattr(module.cv2, "__version__", "5.0.0")
    with pytest.raises(RuntimeError, match="OpenCV 5 is not currently supported"):
        module._validate_supported_opencv()


def test_default_opencv_dependency_excludes_unsupported_major_version():
    pyproject = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"opencv-python-headless>=4.5,<5"' in pyproject


def test_fact_baseline_separates_remove_keep_and_writes_schema(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[3:6, 3:7] = 1
    indexed[0:2, 0:2] = 2
    manifest = _write_manifest(tmp_path, indexed)
    predicted = indexed == 1
    _fake_detector(module, predicted)
    report = module.evaluate_fact_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "report.json"), str(tmp_path / "previews")
    )
    frame = report["videos"][0]["frames"][0]
    assert frame["remove_region_jaccard"] == pytest.approx(1.0)
    assert frame["keep_prediction_coverage"] == pytest.approx(0.0)
    assert frame["visible_annotation_matches"][0]["classifier_semantic_match"] is True
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["schema_version"] == 1
    assert set(saved["macro_average"]) == {
        "frame_remove_union_region_jaccard",
        "frame_remove_union_boundary_f",
        "visible_remove_object_region_jaccard",
        "visible_remove_object_boundary_f",
        "visible_keep_prediction_coverage",
        "no_remove_false_removal_area_ratio",
        "visible_annotation_semantic_match_rate",
        "visible_annotation_selection_intent_match_rate",
    }


def test_semantic_match_uses_all_candidates_and_only_visible_objects(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[0:2, 0:2] = 2
    objects = [
        {"id": 1, "type": "subtitle", "action": "remove", "description": "absent"},
        {"id": 2, "type": "watermark", "action": "keep", "description": "visible"},
    ]
    manifest = _write_manifest(tmp_path, indexed, objects=objects)
    candidate_mask = indexed == 2
    watermark = _candidate(candidate_mask, target_type="watermark", default_remove=False)
    _fake_detector(module, np.zeros_like(candidate_mask), candidates=[watermark], selected=[])
    report = module.evaluate_fact_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "report.json"), str(tmp_path / "previews")
    )
    match = report["videos"][0]["frames"][0]["visible_annotation_matches"]
    assert len(match) == 1
    assert match[0]["classifier_semantic_match"] is True
    assert match[0]["selection_intent_match"] is True
    assert report["macro_average"]["visible_annotation_semantic_match_rate"] == 1.0


def test_object_metrics_do_not_penalize_other_valid_remove_objects(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[3:5, 1:3] = 1
    indexed[3:5, 8:10] = 2
    objects = [
        {"id": 1, "type": "subtitle", "action": "remove", "description": "first"},
        {"id": 2, "type": "subtitle", "action": "remove", "description": "second"},
    ]
    manifest = _write_manifest(tmp_path, indexed, objects=objects)
    first = _candidate(indexed == 1, candidate_id="first")
    second = _candidate(indexed == 2, candidate_id="second")
    _fake_detector(module, indexed > 0, candidates=[first, second], selected=[first, second])
    report = module.evaluate_fact_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "report.json"), str(tmp_path / "previews")
    )
    metrics = report["videos"][0]["frames"][0]["object_metrics"]
    assert [metric["region_jaccard"] for metric in metrics] == [1.0, 1.0]
    assert report["macro_average"]["visible_remove_object_region_jaccard"] == 1.0


def test_keep_injury_and_no_target_false_removal(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[0:2, 0:2] = 2
    manifest = _write_manifest(tmp_path, indexed)
    predicted = np.zeros((8, 12), dtype=bool)
    predicted[0:2, 0:2] = True
    _fake_detector(module, predicted)
    report = module.evaluate_fact_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "report.json"), str(tmp_path / "previews")
    )
    frame = report["videos"][0]["frames"][0]
    assert frame["remove_region_jaccard"] is None
    assert frame["keep_prediction_coverage"] == pytest.approx(1.0)
    assert frame["no_remove_false_removal_area_ratio"] == pytest.approx(4 / 96, abs=1e-6)


def test_manifest_duplicate_ids_fail(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    manifest = _write_manifest(tmp_path, indexed)
    payload = json.loads(manifest.read_text())
    payload["videos"][0]["objects"].append({"id": 1, "type": "logo", "action": "keep"})
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        module._load_fact_manifest(str(manifest), str(tmp_path))


def test_manifest_missing_mask_and_video_frame_fail(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    manifest = _write_manifest(tmp_path, np.zeros((8, 12), dtype=np.uint8))
    payload = json.loads(manifest.read_text())
    payload["videos"][0]["frames"][0]["mask"] = "annotations/sample/missing.png"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing mask"):
        module._load_fact_manifest(str(manifest), str(tmp_path))
    payload["videos"][0]["frames"][0]["mask"] = "annotations/sample/000001.png"
    payload["videos"][0]["frames"][0]["frame"] = 99
    manifest.write_text(json.dumps(payload))
    _fake_detector(module, np.zeros((8, 12), dtype=bool))
    with pytest.raises(ValueError, match="missing annotated frame"):
        module.evaluate_fact_baseline(
            str(tmp_path), str(manifest), str(tmp_path / "report.json"), str(tmp_path / "previews")
        )


@pytest.mark.parametrize("kind", ["unknown_id", "wrong_size"])
def test_annotation_validation_failures(tmp_path, kind):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    if kind == "unknown_id":
        indexed[2, 2] = 9
    else:
        indexed = np.zeros((7, 12), dtype=np.uint8)
    manifest = _write_manifest(tmp_path, indexed)
    _fake_detector(module, np.zeros((8, 12), dtype=bool))
    with pytest.raises(ValueError, match="unknown object IDs|shape mismatch"):
        module.evaluate_fact_baseline(
            str(tmp_path), str(manifest), str(tmp_path / "report.json"), str(tmp_path / "previews")
        )


def test_manifest_rejects_escape_paths_and_symlinks(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    manifest = _write_manifest(tmp_path, np.zeros((8, 12), dtype=np.uint8))
    payload = json.loads(manifest.read_text())
    payload["videos"][0]["file"] = "../outside.mp4"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parent traversal|escapes"):
        module._load_fact_manifest(str(manifest), str(tmp_path))

    payload["videos"][0]["file"] = "sample.mp4"
    payload["videos"][0]["frames"][0]["mask"] = str((tmp_path / "absolute.png").resolve())
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="relative"):
        module._load_fact_manifest(str(manifest), str(tmp_path))

    outside = tmp_path.parent / "outside-mask.png"
    cv2.imwrite(str(outside), np.zeros((8, 12), dtype=np.uint8))
    link = tmp_path / "annotations" / "sample" / "link.png"
    link.symlink_to(outside)
    payload["videos"][0]["frames"][0]["mask"] = "annotations/sample/link.png"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="escapes"):
        module._load_fact_manifest(str(manifest), str(tmp_path))


def test_preview_output_rejects_escaping_symlink(tmp_path):
    module = _load_eval_module()
    root = tmp_path / "previews"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact_id = "sample-deadbeef"
    (root / artifact_id).symlink_to(outside, target_is_directory=True)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="symlink"):
        module._write_previews(
            root, artifact_id, 1, frame, np.zeros((8, 12), dtype=np.uint8),
            np.zeros((8, 12), dtype=bool), {},
        )


def test_preview_output_rejects_inside_root_symlink(tmp_path):
    module = _load_eval_module()
    root = tmp_path / "previews"
    root.mkdir()
    other = root / "other-video"
    other.mkdir()
    artifact_id = "sample-deadbeef"
    (root / artifact_id).symlink_to(other, target_is_directory=True)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="symlink"):
        module._write_previews(
            root, artifact_id, 1, frame, np.zeros((8, 12), dtype=np.uint8),
            np.zeros((8, 12), dtype=bool), {},
        )
    assert list(other.iterdir()) == []


def test_report_output_rejects_symlink_and_input_alias(tmp_path):
    module = _load_eval_module()
    protected = tmp_path / "manifest.json"
    protected.write_text("{}")
    destination = tmp_path / "report.json"
    destination.symlink_to(protected)
    with pytest.raises(ValueError, match="symlink"):
        module._write_json_report(destination, {}, [protected])
    destination.unlink()
    with pytest.raises(ValueError, match="overwrite an input"):
        module._write_json_report(protected, {}, [protected])
    assert protected.read_text() == "{}"


def test_hash_uses_logical_name_not_invocation_path(tmp_path):
    module = _load_eval_module()
    first = tmp_path / "first.bin"
    second = tmp_path / "nested" / "second.bin"
    second.parent.mkdir()
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    assert module._sha256_named_files([("sample.mp4", first)]) == module._sha256_named_files([("sample.mp4", second)])


def test_git_provenance_records_dirty_worktree(monkeypatch):
    module = _load_eval_module()
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        if "rev-parse" in command:
            return "abcdef\n"
        assert "status" in command
        if "--untracked-files=no" in command:
            return " M scripts/eval_clean_detection.py\n"
        return " M scripts/eval_clean_detection.py\n?? NEXT_WORK.md\n"

    monkeypatch.setattr(module.subprocess, "check_output", check_output)
    provenance = module._git_provenance()
    assert provenance["git_head"] == "abcdef"
    assert provenance["tracked_worktree_is_clean"] is False
    assert provenance["untracked_paths"] == ["NEXT_WORK.md"]
    assert provenance["worktree_status_sha256"]
    assert all(command[:2] == ["git", "-C"] for command in commands)
    assert all(str(module._project_root()) in command for command in commands)


def test_formal_baseline_refuses_dirty_worktree(monkeypatch):
    module = _load_eval_module()
    monkeypatch.setattr(
        module,
        "_git_provenance",
        lambda: {
            "git_head": "abcdef",
            "tracked_worktree_is_clean": False,
            "untracked_paths": [],
            "worktree_status_sha256": "hash",
        },
    )
    with pytest.raises(RuntimeError, match="clean tracked git worktree"):
        module._require_clean_git_worktree([])


def test_formal_baseline_allows_unrelated_untracked_file(monkeypatch):
    module = _load_eval_module()
    monkeypatch.setattr(
        module,
        "_git_provenance",
        lambda: {
            "git_head": "abcdef",
            "tracked_worktree_is_clean": True,
            "untracked_paths": ["NEXT_WORK.md"],
            "worktree_status_sha256": "hash",
        },
    )
    module._require_clean_git_worktree([])


def test_formal_baseline_rejects_untracked_relevant_input(tmp_path, monkeypatch):
    module = _load_eval_module()
    monkeypatch.setattr(module, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_git_provenance",
        lambda: {
            "git_head": "abcdef",
            "tracked_worktree_is_clean": True,
            "untracked_paths": ["fact_baseline.json"],
            "worktree_status_sha256": "hash",
        },
    )
    relevant = tmp_path / "fact_baseline.json"
    relevant.write_text("{}")

    def not_tracked(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(module.subprocess, "check_output", not_tracked)
    with pytest.raises(RuntimeError, match="not tracked by git"):
        module._require_clean_git_worktree([relevant])


def test_regression_compare_detects_removed_video_in_legacy_snapshot(tmp_path):
    module = _load_eval_module()
    report_a = {"video": "a.mp4", "candidate_count": 1, "empty_detection": False, "candidates": [{"type": "subtitle"}]}
    report_b = {"video": "b.mp4", "candidate_count": 1, "empty_detection": False, "candidates": [{"type": "subtitle"}]}
    legacy = tmp_path / "detection_baseline.json"
    legacy.write_text(json.dumps([report_a, report_b]))
    assert module._compare_against_regression_snapshot([report_a], str(legacy)) is False


def test_legacy_regression_cli_reads_and_writes_legacy_artifact(tmp_path, monkeypatch):
    module = _load_eval_module()
    _write_video(tmp_path / "a.mp4")
    report = {
        "video": "a.mp4", "candidate_count": 1, "empty_detection": False,
        "candidates": [{"type": "subtitle"}], "legacy_calibration_metric": {"available": False},
        "generated_mask_area_ratio": 0.0, "frame_shape": [8, 12], "selected_count": 1,
    }
    monkeypatch.setattr(module, "_eval_video", lambda *args, **kwargs: report)
    monkeypatch.setattr(module, "_print_report", lambda report: None)
    monkeypatch.setattr(module.sys, "argv", ["eval_clean_detection.py", str(tmp_path), "--write-baseline"])
    module.main()
    legacy = tmp_path / "detection_baseline.json"
    assert legacy.exists()
    (tmp_path / "detection_regression_snapshot.json").write_text(
        json.dumps([{**report, "candidate_count": 99}])
    )
    monkeypatch.setattr(module.sys, "argv", ["eval_clean_detection.py", str(tmp_path), "--compare-baseline"])
    module.main()


def test_regression_cli_continues_after_one_video_error(tmp_path, monkeypatch):
    module = _load_eval_module()
    _write_video(tmp_path / "a.mp4")
    _write_video(tmp_path / "b.mp4")
    visited = []

    def evaluate(path, *args, **kwargs):
        visited.append(pathlib.Path(path).name)
        if pathlib.Path(path).name == "a.mp4":
            raise RuntimeError("broken")
        return {
            "video": "b.mp4",
            "candidate_count": 0,
            "selected_count": 0,
            "empty_detection": True,
            "candidates": [],
            "legacy_calibration_metric": {"available": False},
            "generated_mask_area_ratio": 0.0,
            "frame_shape": [8, 12],
        }

    monkeypatch.setattr(module, "_eval_video", evaluate)
    monkeypatch.setattr(module, "_print_report", lambda report: None)
    monkeypatch.setattr(module.sys, "argv", ["eval_clean_detection.py", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert exit_info.value.code == 1
    assert visited == ["a.mp4", "b.mp4"]


def test_legacy_golden_missing_keeps_regression_snapshot_compatible(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    module._detect_video = lambda *args, **kwargs: (
        CleanDetectionResult([], (8, 12)), [], np.zeros((8, 12), dtype=bool), "fake_detector"
    )
    report = module._eval_video(str(tmp_path / "sample.mp4"), mask_dir=str(tmp_path / "masks"))
    assert report["report_kind"] == "regression_snapshot"
    assert report["legacy_calibration_metric"]["available"] is False
    assert "golden_mask_missing" in report


def test_legacy_golden_is_reported_as_calibration_metric(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    golden_dir = tmp_path / "masks"
    golden_dir.mkdir()
    golden = np.zeros((8, 12), dtype=np.uint8)
    golden[3:6, 3:7] = 255
    cv2.imwrite(str(golden_dir / "sample_mask.png"), golden)
    generated = golden > 0
    module._detect_video = lambda *args, **kwargs: (
        CleanDetectionResult([], (8, 12)), [], generated, "fake_detector"
    )
    report = module._eval_video(str(tmp_path / "sample.mp4"), mask_dir=str(golden_dir))
    assert report["legacy_calibration_metric"] == {
        "name": "legacy_calibration_metric",
        "available": True,
        "golden_area_ratio": pytest.approx(12 / 96),
        "iou": 1.0,
    }
