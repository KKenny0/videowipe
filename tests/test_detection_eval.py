import gc
import importlib.util
import json
import pathlib
import subprocess
import weakref

import cv2
import numpy as np
import pytest

import videowipe.detect as detect_module
from videowipe.detect import (
    CleanCandidate,
    CleanDetectionResult,
    TextBox,
    detect_clean_candidates,
    refine_temporal_presence,
)
from videowipe.plan import Segment, compute_source
from videowipe.planning import CleanPlanDraft, finalize


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
        detector_backed=True,
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


def _write_decision_manifest(root, cases):
    manifest = root / "decision_baseline.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "set_kind": "decision_calibration_set",
            "fact_manifest": "fact_baseline.json",
            "cases": cases,
        }),
        encoding="utf-8",
    )
    return manifest


def _fake_detector(module, predicted, candidates=None, selected=None):
    candidates = candidates or [_candidate(predicted)]
    selected = candidates if selected is None else selected

    def detect(video_path, *_args, **_kwargs):
        result = CleanDetectionResult(candidates, predicted.shape)
        draft = CleanPlanDraft(
            video_path,
            result,
            compute_source(video_path),
            [candidate.id for candidate in selected],
            {},
            False,
        )
        return draft, selected, predicted.astype(bool), "fake_detector"

    module._detect_video = detect


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


def test_evaluator_detection_enters_clean_planning(monkeypatch):
    module = _load_eval_module()
    mask = np.ones((8, 12), dtype=np.uint8)
    candidate = _candidate(mask)
    draft = type(
        "Draft",
        (),
        {
            "candidates": (candidate,),
            "proposed_remove_ids": frozenset({candidate.id}),
            "frame_shape": mask.shape,
        },
    )()
    calls = []
    monkeypatch.setattr(
        module,
        "prepare",
        lambda video, **kwargs: calls.append((video, kwargs)) or draft,
    )

    detected, selected, generated, path = module._detect_video(
        "input.mp4", "sensitive", "off",
    )

    assert detected is draft
    assert selected == [candidate]
    assert generated.all()
    assert path == "dbnet_default"
    assert calls == [("input.mp4", {"detect_mode": "sensitive", "ocr": "off"})]


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
    for relative in frame["previews"].values():
        preview = tmp_path / "previews" / relative
        assert preview.is_file()
        assert preview.stem.rsplit("-", 1)[-1] == module.hashlib.sha256(
            preview.read_bytes()
        ).hexdigest()
    # schema v2: predictions are per-frame (built from a WipePlan), not a
    # replayed static mask.
    assert saved["schema_version"] == 2
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


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_case",
        "unknown_video",
        "missing_action",
        "invalid_action",
        "unknown_target",
        "blank_target",
        "blank_intent",
        "non_object_case",
    ],
)
def test_decision_manifest_validation(tmp_path, kind):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    _write_manifest(tmp_path, indexed)
    case = {
        "id": "default",
        "file": "sample.mp4",
        "request": {},
        "expected_actions": {"1": "remove", "2": "keep"},
    }
    cases = [case]
    if kind == "duplicate_case":
        cases.append(dict(case))
    elif kind == "unknown_video":
        case["file"] = "unknown.mp4"
    elif kind == "missing_action":
        case["expected_actions"].pop("2")
    elif kind == "invalid_action":
        case["expected_actions"]["2"] = "ignore"
    elif kind == "unknown_target":
        case["request"] = {"targets": ["watermak"]}
    elif kind == "blank_target":
        case["request"] = {"targets": [" "]}
    elif kind == "blank_intent":
        case["request"] = {"intent": " "}
    else:
        cases = [None]
    manifest = _write_decision_manifest(tmp_path, cases)

    expected_error = TypeError if kind == "non_object_case" else ValueError
    with pytest.raises(
        expected_error,
        match=(
            "unique id|unknown video|every object action|remove or keep|"
            "unsupported target|non-empty string|must be an object"
        ),
    ):
        module._load_decision_manifest(str(manifest), str(tmp_path))


def test_decision_manifest_root_must_be_an_object(tmp_path):
    module = _load_eval_module()
    manifest = tmp_path / "decision_baseline.json"
    manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="root must be an object"):
        module._load_decision_manifest(str(manifest), str(tmp_path))


def test_decision_baseline_scores_default_and_targeted_selection(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[4:7, 2:8] = 1
    indexed[0:2, 0:3] = 2
    _write_manifest(tmp_path, indexed)
    subtitle = _candidate(indexed == 1, candidate_id="subtitle", target_type="subtitle")
    watermark = _candidate(
        indexed == 2,
        candidate_id="watermark",
        target_type="watermark",
        default_remove=False,
    )
    _fake_detector(module, indexed > 0, candidates=[subtitle, watermark])
    manifest = _write_decision_manifest(tmp_path, [
        {
            "id": "default",
            "file": "sample.mp4",
            "request": {},
            "expected_actions": {"1": "remove", "2": "keep"},
        },
        {
            "id": "watermark-only",
            "file": "sample.mp4",
            "request": {"targets": ["watermark"]},
            "expected_actions": {"1": "keep", "2": "remove"},
        },
    ])

    report = module.evaluate_decision_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "decision-report.json")
    )

    assert [case["selected_candidate_ids"] for case in report["cases"]] == [
        ["subtitle"], ["watermark"],
    ]
    assert report["macro_average"] == {
        "visible_remove_candidate_availability_rate": 1.0,
        "visible_remove_selection_recall_given_candidate": 1.0,
        "visible_keep_false_selection_rate": 0.0,
        "visible_annotation_action_match_rate": 1.0,
        "visible_annotation_candidate_type_match_rate": 1.0,
        "annotated_candidate_semantic_match_rate": 1.0,
        "mixed_object_candidate_rate": 0.0,
        "mixed_semantic_candidate_rate": 0.0,
        "case_exact_action_match_rate": 1.0,
    }


def test_decision_baseline_separates_missing_candidate_from_selection_miss(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[2:4, 1:4] = 1
    indexed[5:7, 8:11] = 2
    objects = [
        {"id": 1, "type": "subtitle", "action": "remove", "description": "detected"},
        {"id": 2, "type": "logo", "action": "keep", "description": "missing"},
    ]
    _write_manifest(tmp_path, indexed, objects=objects)
    subtitle = _candidate(indexed == 1, candidate_id="subtitle", target_type="subtitle")
    _fake_detector(module, indexed == 1, candidates=[subtitle])
    manifest = _write_decision_manifest(tmp_path, [{
        "id": "remove-both",
        "file": "sample.mp4",
        "request": {"targets": ["logo"]},
        "expected_actions": {"1": "remove", "2": "remove"},
    }])

    report = module.evaluate_decision_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "decision-report.json")
    )
    objects_report = report["cases"][0]["frames"][0]["objects"]

    assert [obj["failure_kind"] for obj in objects_report] == [
        "selection_miss", "candidate_missing",
    ]
    assert report["macro_average"]["visible_remove_candidate_availability_rate"] == 0.5
    assert report["macro_average"]["visible_remove_selection_recall_given_candidate"] == 0.0


def test_decision_baseline_attributes_cross_type_candidate_as_mixed(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[2:4, 1:4] = 1
    indexed[2:4, 4:7] = 2
    objects = [
        {"id": 1, "type": "subtitle", "action": "remove", "description": "text"},
        {"id": 2, "type": "logo", "action": "keep", "description": "logo"},
    ]
    _write_manifest(tmp_path, indexed, objects=objects)
    mixed = _candidate(indexed > 0, candidate_id="mixed", target_type="subtitle")
    _fake_detector(module, indexed > 0, candidates=[mixed])
    manifest = _write_decision_manifest(tmp_path, [{
        "id": "subtitle-only",
        "file": "sample.mp4",
        "request": {"targets": ["subtitle"]},
        "expected_actions": {"1": "remove", "2": "keep"},
    }])

    report = module.evaluate_decision_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "decision-report.json")
    )

    assert report["cases"][0]["frames"][0]["objects"][1]["failure_kind"] == "candidate_mixed"
    assert report["macro_average"]["mixed_object_candidate_rate"] == 1.0
    assert report["macro_average"]["mixed_semantic_candidate_rate"] == 1.0
    assert report["videos"][0]["candidates"][0] == {
        "candidate_id": "mixed",
        "candidate_type": "subtitle",
        "matched_object_ids": [1, 2],
        "matched_object_coverage": {"1": 1.0, "2": 1.0},
        "matched_annotation_types": ["logo", "subtitle"],
        "semantic_type_match": True,
        "mixed_object": True,
        "mixed_semantic": True,
    }


def test_candidate_purity_ignores_incidental_cross_type_overlap(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[0:3, 0:6] = 1
    indexed[3:7, 6:11] = 2
    objects = [
        {"id": 1, "type": "watermark", "action": "keep", "description": "text"},
        {"id": 2, "type": "logo", "action": "keep", "description": "logo"},
    ]
    _write_manifest(tmp_path, indexed, objects=objects)
    mask = indexed == 2
    mask[2, 5] = True
    logo = _candidate(mask, candidate_id="logo", target_type="logo")
    _fake_detector(module, mask, candidates=[logo])
    manifest = _write_decision_manifest(tmp_path, [{
        "id": "default",
        "file": "sample.mp4",
        "request": {},
        "expected_actions": {"1": "keep", "2": "keep"},
    }])

    report = module.evaluate_decision_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "decision-report.json")
    )

    evidence = report["videos"][0]["candidates"][0]
    assert evidence["matched_object_ids"] == [2]
    assert evidence["mixed_semantic"] is False
    assert report["macro_average"]["mixed_semantic_candidate_rate"] == 0.0


def test_decision_baseline_detects_each_video_once_and_is_deterministic(tmp_path, monkeypatch):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[3:6, 3:7] = 1
    indexed[0, 0] = 2
    _write_manifest(tmp_path, indexed)
    candidate = _candidate(indexed == 1)
    result = CleanDetectionResult([candidate], indexed.shape)
    calls = 0

    def detect_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        draft = CleanPlanDraft(
            args[0], result, compute_source(args[0]), [candidate.id], {}, False,
        )
        return draft, [candidate], indexed == 1, "fake_detector"

    monkeypatch.setattr(module, "_detect_video", detect_once)
    cases = [
        {
            "id": case_id,
            "file": "sample.mp4",
            "request": {},
            "expected_actions": {"1": "remove", "2": "keep"},
        }
        for case_id in ("first", "second")
    ]
    manifest = _write_decision_manifest(tmp_path, cases)
    first = module.evaluate_decision_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "first.json")
    )
    assert calls == 1
    calls = 0
    second = module.evaluate_decision_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "second.json")
    )

    assert calls == 1
    assert first == second


def test_decision_baseline_rejects_declared_object_missing_from_annotations(tmp_path):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[3:6, 3:7] = 1
    _write_manifest(tmp_path, indexed)
    _fake_detector(module, indexed == 1)
    manifest = _write_decision_manifest(tmp_path, [{
        "id": "default",
        "file": "sample.mp4",
        "request": {},
        "expected_actions": {"1": "remove", "2": "keep"},
    }])

    with pytest.raises(ValueError, match="missing from every annotation frame: \\[2\\]"):
        module.evaluate_decision_baseline(
            str(tmp_path), str(manifest), str(tmp_path / "report.json")
        )


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


def test_fact_baseline_uses_temporal_refinement_for_a_single_frame_gap(tmp_path, monkeypatch):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    manifest = _write_manifest(tmp_path, np.zeros((8, 12), dtype=np.uint8))
    candidate = _candidate(np.ones((8, 12), dtype=np.uint8))
    candidate.presence_frames = [0, 2]

    class Detector:
        calls = 0

        def detect(self, _frame):
            self.calls += 1
            if self.calls == 2:
                return []
            return [TextBox(
                points=np.array([[0, 0], [11, 0], [11, 7], [0, 7]]), confidence=1.0,
            )]

    result = CleanDetectionResult(
        [candidate], (8, 12), sample_indices=[0, 2], detector=Detector(),
    )
    draft = CleanPlanDraft(
        str(tmp_path / "sample.mp4"),
        result,
        compute_source(str(tmp_path / "sample.mp4")),
        [candidate.id],
        {"detect_mode": "balanced"},
        False,
    )
    monkeypatch.setattr(
        module, "_detect_video",
        lambda *args, **kwargs: (
            draft, [candidate], candidate.mask.squeeze().astype(bool), "fake_detector",
        ),
    )
    report = module.evaluate_fact_baseline(
        str(tmp_path), str(manifest), str(tmp_path / "report.json"), str(tmp_path / "previews"),
    )

    assert report["videos"][0]["frames"][0]["no_remove_false_removal_area_ratio"] == 0.0


def test_band_fallback_candidate_keeps_coarse_semantics(tmp_path):
    video = tmp_path / "input.mp4"
    _write_video(video, width=96, height=64)

    class BandOnlyDetector:
        calls = 0

        def detect(self, frame):
            self.calls += 1
            h, w = frame.shape[:2]
            if h < 50:
                return [TextBox(
                    points=np.array([[2, h - 10], [w - 2, h - 10], [w - 2, h - 2], [2, h - 2]]),
                    confidence=0.9,
                )]
            return []

    result = detect_clean_candidates(
        str(video), detector=BandOnlyDetector(), sample_count=3, subtitle_fallback="light",
    )
    candidate = result.candidates[0]
    calls_before_plan = result.detector.calls
    plan = finalize(
        CleanPlanDraft(
            str(video), result, compute_source(str(video)), [candidate.id], {}, False,
        ),
        refine=True,
    )

    assert not candidate.detector_backed
    assert result.detector.calls == calls_before_plan
    assert candidate.temporal_sample_indices == []
    assert candidate.presence_frames == []
    assert [(segment.start, segment.end) for segment in plan.remove_tracks[0].segments] == [(0, 3)]
    assert any("no per-frame presence evidence" in warning for warning in plan.warnings)


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


def test_preview_custody_rejects_unsafe_artifact_id_without_writing(tmp_path):
    module = _load_eval_module()
    root = tmp_path / "previews"
    custody = module._PreviewCustody(root)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsafe preview artifact id"):
        custody.stage(
            "../escape", 1, frame, np.zeros((8, 12), dtype=np.uint8),
            np.zeros((8, 12), dtype=bool), {},
        )
    assert not root.exists()


def test_preview_custody_rejects_artifact_root_symlink(tmp_path):
    module = _load_eval_module()
    root = tmp_path / "previews"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact directory must not be a symlink"):
        module._PreviewCustody(root)
    assert list(outside.iterdir()) == []


def test_preview_custody_rejects_run_destination_symlink(tmp_path, monkeypatch):
    module = _load_eval_module()
    root = tmp_path / "previews"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(module, "_new_preview_run_id", lambda: "run-fixed")
    custody = module._PreviewCustody(root)
    (root / "run-fixed").symlink_to(outside, target_is_directory=True)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    custody.stage(
        "sample-deadbeef", 1, frame, np.zeros((8, 12), dtype=np.uint8),
        np.zeros((8, 12), dtype=bool), {},
    )

    with pytest.raises(ValueError, match="run destination already exists"):
        custody.publish()
    assert list(outside.iterdir()) == []
    assert [path.name for path in root.iterdir()] == ["run-fixed"]


def test_preview_custody_cleans_private_stage_when_publish_fails(tmp_path, monkeypatch):
    module = _load_eval_module()
    root = tmp_path / "previews"
    custody = module._PreviewCustody(root)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    custody.stage(
        "sample-deadbeef", 1, frame, np.zeros((8, 12), dtype=np.uint8),
        np.zeros((8, 12), dtype=bool), {},
    )
    def fail_rename(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(module.os, "rename", fail_rename)

    with pytest.raises(OSError, match="boom"):
        custody.publish()

    assert list(root.iterdir()) == []


def test_preview_custody_publishes_without_dir_fd_support(tmp_path, monkeypatch):
    module = _load_eval_module()
    root = tmp_path / "previews"
    monkeypatch.setattr(module.os, "supports_dir_fd", set())
    monkeypatch.setattr(module, "_new_preview_run_id", lambda: "run-fixed")
    custody = module._PreviewCustody(root)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    paths = custody.stage(
        "sample-deadbeef", 1, frame, np.zeros((8, 12), dtype=np.uint8),
        np.zeros((8, 12), dtype=bool), {},
    )

    custody.publish()

    assert (root / paths["annotations"]).is_file()
    assert not list(root.glob(".*"))


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


def test_report_final_check_runs_after_temp_write_before_replace(tmp_path, monkeypatch):
    module = _load_eval_module()
    destination = tmp_path / "report.json"
    events = []
    real_replace = module.os.replace

    def before_replace():
        temporary = list(tmp_path.glob(".report.json.*.tmp"))
        assert len(temporary) == 1
        assert json.loads(temporary[0].read_text()) == {"ready": True}
        assert not destination.exists()
        events.append("check")

    def replace(source, target):
        events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", replace)
    module._write_json_report(
        destination, {"ready": True}, [], before_replace=before_replace,
    )

    assert events == ["check", "replace"]
    assert json.loads(destination.read_text()) == {"ready": True}


def test_formal_final_check_runs_after_preview_publish(tmp_path, monkeypatch):
    module = _load_eval_module()
    root = tmp_path / "previews"
    previews = module._PreviewCustody(root)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    paths = previews.stage(
        "sample-deadbeef", 1, frame, np.zeros((8, 12), dtype=np.uint8),
        np.zeros((8, 12), dtype=bool), {},
    )
    events = []
    monkeypatch.setattr(module, "_require_clean_git_worktree", lambda _paths: None)
    monkeypatch.setattr(module, "_baseline_provenance", lambda *_args: {})

    def check_state(*_args):
        assert (root / paths["annotations"]).is_file()
        events.append("state")

    def check_weights(*_args):
        events.append("weights")

    monkeypatch.setattr(module, "_assert_formal_state_unchanged", check_state)
    monkeypatch.setattr(module, "_assert_detector_weights_unchanged", check_weights)
    custody = module._BaselineCustody([], [], [], "balanced", "off", True)

    custody.publish(str(tmp_path / "report.json"), {}, previews)

    assert events == ["state", "weights"]


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


def test_formal_baseline_rejects_state_changed_during_evaluation(monkeypatch):
    module = _load_eval_module()
    events = []
    monkeypatch.setattr(
        module, "_require_clean_git_worktree", lambda paths: events.append("validate")
    )

    def capture():
        events.append("capture")
        return {"git": {"git_head": "after"}}

    with pytest.raises(RuntimeError, match="changed during evaluation"):
        module._assert_formal_state_unchanged(
            {"git": {"git_head": "before"}},
            capture,
            [],
        )
    assert events == ["validate", "capture"]


@pytest.mark.parametrize("baseline_kind", ["fact", "decision"])
def test_formal_baseline_state_change_prevents_report_publish(
    tmp_path, monkeypatch, baseline_kind
):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[3:6, 3:7] = 1
    indexed[0:2, 0:2] = 2
    fact_manifest = _write_manifest(tmp_path, indexed)
    _fake_detector(module, indexed == 1)
    monkeypatch.setattr(module, "_require_clean_git_worktree", lambda _paths: None)
    states = iter([{"state": "before"}, {"state": "after"}])
    monkeypatch.setattr(module, "_baseline_provenance", lambda *_args: next(states))

    output = tmp_path / f"{baseline_kind}-report.json"
    if baseline_kind == "fact":
        def evaluate():
            return module.evaluate_fact_baseline(
                str(tmp_path),
                str(fact_manifest),
                str(output),
                str(tmp_path / "previews"),
                require_clean_git=True,
            )
    else:
        decision_manifest = _write_decision_manifest(
            tmp_path,
            [
                {
                    "id": "default",
                    "file": "sample.mp4",
                    "request": {},
                    "expected_actions": {"1": "remove", "2": "keep"},
                }
            ],
        )
        def evaluate():
            return module.evaluate_decision_baseline(
                str(tmp_path),
                str(decision_manifest),
                str(output),
                require_clean_git=True,
            )

    with pytest.raises(RuntimeError, match="changed during evaluation"):
        evaluate()
    assert not output.exists()


def test_formal_fact_custody_failure_preserves_existing_report_and_previews(
    tmp_path, monkeypatch
):
    module = _load_eval_module()
    _write_video(tmp_path / "sample.mp4")
    indexed = np.zeros((8, 12), dtype=np.uint8)
    indexed[3:6, 3:7] = 1
    manifest = _write_manifest(tmp_path, indexed)
    _fake_detector(module, indexed == 1)
    monkeypatch.setattr(module, "_require_clean_git_worktree", lambda _paths: None)
    states = iter([{"state": "before"}, {"state": "after"}])
    monkeypatch.setattr(module, "_baseline_provenance", lambda *_args: next(states))

    output = tmp_path / "report.json"
    output.write_bytes(b'{"previews":["sample-old/frame-old.png"]}\n')
    old_preview = tmp_path / "previews" / "sample-old" / "frame-old.png"
    old_preview.parent.mkdir(parents=True)
    old_preview.write_bytes(b"old preview")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        module.evaluate_fact_baseline(
            str(tmp_path),
            str(manifest),
            str(output),
            str(tmp_path / "previews"),
            require_clean_git=True,
        )

    report_bytes = output.read_bytes()
    assert report_bytes == b'{"previews":["sample-old/frame-old.png"]}\n'
    assert old_preview.read_bytes() == b"old preview"
    orphan_runs = [
        path for path in (tmp_path / "previews").iterdir()
        if path.is_dir() and path.name.startswith("run-")
    ]
    assert len(orphan_runs) == 1
    assert orphan_runs[0].name.encode() not in report_bytes
    orphan_files = list(orphan_runs[0].iterdir())
    assert orphan_files
    for preview in orphan_files:
        assert preview.stem.rsplit("-", 1)[-1] == module.hashlib.sha256(
            preview.read_bytes()
        ).hexdigest()


def test_preview_report_alias_is_rejected_before_publish(tmp_path, monkeypatch):
    module = _load_eval_module()
    root = tmp_path / "previews"
    monkeypatch.setattr(module, "_new_preview_run_id", lambda: "run-fixed")
    previews = module._PreviewCustody(root)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    paths = previews.stage(
        "sample-deadbeef", 1, frame, np.zeros((8, 12), dtype=np.uint8),
        np.zeros((8, 12), dtype=bool), {},
    )
    custody = module._BaselineCustody([], [], [], "balanced", "off", False)
    output = root / paths["annotations"]

    with pytest.raises(ValueError, match="must not overwrite preview evidence"):
        custody.publish(str(output), {}, previews)

    assert not root.exists()


def test_report_write_failure_keeps_existing_evidence_and_orphan_run(
    tmp_path, monkeypatch
):
    module = _load_eval_module()
    root = tmp_path / "previews"
    old_preview = root / "run-old" / "old.png"
    old_preview.parent.mkdir(parents=True)
    old_preview.write_bytes(b"old preview")
    output = tmp_path / "report.json"
    output.write_bytes(b"old report")
    monkeypatch.setattr(module, "_new_preview_run_id", lambda: "run-new")
    previews = module._PreviewCustody(root)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    paths = previews.stage(
        "sample-deadbeef", 1, frame, np.zeros((8, 12), dtype=np.uint8),
        np.zeros((8, 12), dtype=bool), {},
    )
    custody = module._BaselineCustody([], [], [], "balanced", "off", False)

    def fail_report(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(module, "_write_json_report", fail_report)

    with pytest.raises(OSError, match="boom"):
        custody.publish(str(output), {}, previews)

    assert output.read_bytes() == b"old report"
    assert old_preview.read_bytes() == b"old preview"
    assert (root / paths["annotations"]).is_file()
    assert not list(root.glob(".*.tmp"))


def test_detector_weight_provenance_detects_replacement(tmp_path):
    module = _load_eval_module()
    weight = tmp_path / "detector.onnx"
    weight.write_bytes(b"first")
    draft = type("Draft", (), {"detector_weight": (str(weight), None)})()
    path, provenance = module._detector_weight_state(draft)

    assert provenance["filename"] == "detector.onnx"
    assert len(provenance["sha256"]) == 64
    weight.write_bytes(b"second")
    with pytest.raises(RuntimeError, match="Detector weight changed"):
        module._record_detector_weight({path: provenance}, draft)
    with pytest.raises(RuntimeError, match="Detector weight changed"):
        module._assert_detector_weights_unchanged({path: provenance})


def test_detector_weight_provenance_uses_load_time_digest(tmp_path):
    module = _load_eval_module()
    weight = tmp_path / "detector.onnx"
    weight.write_bytes(b"loaded")
    loaded = module._weight_provenance(weight)
    draft = type(
        "Draft", (), {"detector_weight": (str(weight), loaded["sha256"])}
    )()
    weight.write_bytes(b"replacement")

    with pytest.raises(RuntimeError, match="Detector weight changed"):
        module._record_detector_weight({}, draft)


def test_baseline_custody_does_not_overwrite_detector_weight(tmp_path, monkeypatch):
    module = _load_eval_module()
    weight = tmp_path / "detector.onnx"
    weight.write_bytes(b"loaded")
    draft = type("Draft", (), {"detector_weight": (str(weight), None)})()
    monkeypatch.setattr(module, "_baseline_provenance", lambda *_args: {})
    custody = module._BaselineCustody([], [], [], "balanced", "off", False)
    custody.record_detector_weight(draft)

    with pytest.raises(ValueError, match="overwrite an input"):
        custody.publish(str(weight), {})
    assert weight.read_bytes() == b"loaded"


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


@pytest.mark.parametrize(
    ("mode_flag", "evaluator_name", "expected_output"),
    [
        ("--manifest", "evaluate_fact_baseline", "result/fact-baseline/report.json"),
        (
            "--decision-manifest",
            "evaluate_decision_baseline",
            "result/decision-baseline/report.json",
        ),
    ],
)
def test_baseline_cli_uses_mode_specific_default_output(
    tmp_path, monkeypatch, mode_flag, evaluator_name, expected_output
):
    module = _load_eval_module()
    captured = {}

    def evaluate(input_dir, manifest, output, *args, **kwargs):
        captured["output"] = output
        return {"macro_average": {}}

    monkeypatch.setattr(module, evaluator_name, evaluate)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["eval_clean_detection.py", str(tmp_path), mode_flag, "manifest.json"],
    )
    module.main()

    assert captured["output"] == expected_output


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


# ── WipePlan Phase A / C2: temporal presence capture ─────────────────────────

def _write_presence_video(path, frames=12, width=64, height=64):
    """Subtitle band white on even frames, black on odd frames."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if i % 2 == 0:
            frame[50:60, :] = 255  # subtitle band
        writer.write(frame)
    writer.release()


def test_detect_captures_per_frame_presence_and_sample_indices(tmp_path):
    from videowipe.detect import TextBox, detect_clean_candidates

    video = tmp_path / "input.mp4"
    _write_presence_video(video, frames=12)

    class FakeDetector:
        def detect(self, frame):
            # subtitle box present only when the band is white on this frame
            if frame[55, 32].max() > 0:
                return [TextBox(
                    points=np.array([[8, 50], [56, 50], [56, 60], [8, 60]]),
                    confidence=0.9,
                    text="subtitle",
                )]
            return []

    result = detect_clean_candidates(str(video), detector=FakeDetector(), sample_count=12)

    # sample_indices carries real frame indices of successful samples
    assert result.sample_indices, "expected sampled frames"
    assert all(isinstance(i, int) for i in result.sample_indices)

    subtitle = [c for c in result.candidates if c.type == "subtitle"]
    assert subtitle, "expected a subtitle candidate"
    sub = subtitle[0]

    # presence_frames is the subset of sample_indices where the subtitle band was active (even frames)
    expected = {idx for idx in result.sample_indices if idx % 2 == 0}
    assert set(sub.presence_frames) == expected
    assert sub.presence_frames != result.sample_indices  # genuinely temporal, not full-video


def test_clean_detection_streams_main_then_fallback_with_exact_output(monkeypatch):
    calls = []
    frame_refs = []

    def sample_frames(_path, count):
        for index in range(count):
            frame = np.full((100, 100, 3), index, dtype=np.uint8)
            frame_refs.append(weakref.ref(frame))
            yield index, frame

    class FakeDetector:
        def detect(self, frame):
            calls.append(frame.shape)
            if frame.shape[0] != 100:
                return []
            return [TextBox(
                points=np.array([[20, 80], [80, 80], [80, 90], [20, 90]]),
                confidence=0.9,
                text="sub",
            )]

    monkeypatch.setattr(
        detect_module, "_iter_sample_frames_with_indices", sample_frames,
    )
    result = detect_clean_candidates(
        "unused.mp4", FakeDetector(), sample_count=3,
        subtitle_fallback="light",
    )

    assert calls == [(100, 100, 3)] * 3 + [(40, 100, 3)] * 6
    assert result.sample_indices == [0, 1, 2]
    assert sorted(result.sampled_frame_boxes) == [0, 1, 2]
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert (candidate.bbox, candidate.type, candidate.default_remove) == (
        (0, 76, 99, 94), "subtitle", True,
    )
    assert candidate.presence_frames == [0, 1, 2]
    expected_mask = np.zeros((100, 100, 1), dtype=np.uint8)
    expected_mask[76:95] = 1
    np.testing.assert_array_equal(candidate.mask, expected_mask)

    gc.collect()
    assert frame_refs and all(ref() is None for ref in frame_refs)


def test_temporal_refinement_reuses_sampled_frame_boxes(tmp_path):
    video = tmp_path / "input.mp4"
    _write_video(video, width=100, height=100, frames=3)

    class FakeDetector:
        calls = 0

        def detect(self, _frame):
            self.calls += 1
            return [TextBox(
                points=np.array([[20, 80], [80, 80], [80, 90], [20, 90]]),
                confidence=0.9,
            )]

    detector = FakeDetector()
    result = detect_clean_candidates(
        str(video), detector, sample_count=3,
    )
    candidate = result.candidates[0]
    assert detector.calls == 3

    warnings = refine_temporal_presence(
        str(video), result, {candidate.id: [Segment(0, 3)]}, 3,
    )

    assert warnings == []
    assert detector.calls == 3
    assert candidate.temporal_sample_indices == [0, 1, 2]
    assert candidate.presence_frames == [0, 1, 2]


def test_candidate_to_dict_exposes_presence_frames():
    from videowipe.detect import CleanCandidate

    c = CleanCandidate(
        id="c1", type="subtitle", label="sub", bbox=(0, 0, 9, 9),
        confidence=0.9, frame_fraction=0.5, reason="x", default_remove=True,
        presence_frames=[3, 7, 11],
    )
    d = c.to_dict()
    assert d["presence_frames"] == [3, 7, 11]
