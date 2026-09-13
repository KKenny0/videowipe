"""Protection wins after feathering, only during its declared frame intervals."""
from copy import deepcopy

import numpy as np
import pytest

from videowipe.errors import InvalidInputError
from videowipe.inpainters.sttn import _blend_frame_regions
from videowipe.plan import (
    MaskAsset, Segment, Source, TemporalResolution, Track, WipePlan,
    execution_masks, is_temporal, load_wipe_plan, predicted_mask_at,
    remove_union_mask, save_wipe_plan, validate_plan,
)


def protected_plan():
    remove = np.zeros((32, 48), dtype=np.uint8)
    remove[10:22, 10:38] = 1
    protect = np.zeros_like(remove)
    protect[8:24, 20:30] = 1
    def track(name, action, mask, segments):
        return Track(name, 'region', name, action, (0, 0, 47, 31), 1, 1,
                     'user', segments, name, mask)
    return WipePlan('wipe_plan', 2, Source('x.mp4', 'a'*64, 48, 32, 10, 10), {},
                    TemporalResolution(0, 0, 0), MaskAsset('wipe_plan_masks.npz', ''),
                    [track('remove', 'remove', remove, [Segment(0, 10)]),
                     track('protect', 'protect', protect, [Segment(0, 1), Segment(9, 10)])])


def test_protection_projection_roundtrip_and_pixel_identity(tmp_path):
    plan = protected_plan()
    assert is_temporal(plan)
    union, frame_mask = execution_masks(plan, 4)
    protected = plan.tracks[1].mask.astype(bool)
    assert union[15, 25] == 1  # protection at two boundaries is not whole-video protection
    assert not frame_mask(0)[protected].any()
    assert frame_mask(1)[15, 25] == 1
    assert not frame_mask(9)[protected].any()
    assert frame_mask(0)[8, 15] > 0  # feather outside remove survives away from protection
    assert not predicted_mask_at(plan, 9)[protected].any()
    assert remove_union_mask(plan)[15, 25]
    source = np.arange(32*48*3, dtype=np.uint8).reshape(32, 48, 3)
    result = _blend_frame_regions(source, [np.ones_like(source)*237], [(0, 32)],
                                  frame_mask(9)[:, :, None])
    assert np.array_equal(result[protected], source[protected])
    path, _ = save_wipe_plan(plan, str(tmp_path))
    loaded = load_wipe_plan(path)
    assert loaded.schema_version == 2
    assert np.array_equal(execution_masks(loaded, 4)[1](9), frame_mask(9))
    other = deepcopy(plan.tracks[1]); other.id = other.mask_key = 'other'
    plan.tracks.append(other)
    assert np.array_equal(execution_masks(plan, 4)[1](9), frame_mask(9))
    plan.tracks[1].segments = [Segment(0, 10)]
    assert not remove_union_mask(plan)[15, 25]
    plan.tracks[1].mask[:] = 1
    assert not execution_masks(plan, 4)[0].any()


def test_protection_validation_and_v1_compatibility(tmp_path):
    plan = protected_plan()
    for version in (0, 4, 2.5, True):
        plan.schema_version = version
        with pytest.raises(InvalidInputError): validate_plan(plan)
    plan.schema_version = 1
    with pytest.raises(InvalidInputError, match='require'): validate_plan(plan)
    plan.schema_version = 2
    plan.tracks[1].segments = []
    with pytest.raises(InvalidInputError): validate_plan(plan)
    plan.tracks[1].segments = [Segment(0, 10)]
    plan.tracks[1].mask[:] = 0
    with pytest.raises(InvalidInputError): validate_plan(plan)
    plan.tracks.pop()
    plan.schema_version = 1
    path, _ = save_wipe_plan(plan, str(tmp_path))
    assert load_wipe_plan(path).schema_version == 1
    assert execution_masks(plan)[1] is None


def test_file_backends_reject_protection_before_loading(tmp_path, monkeypatch):
    import cv2
    from videowipe import WipeEngine, WipeRequest
    from videowipe.plan import compute_source
    video = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (48, 32))
    for _ in range(10): writer.write(np.zeros((32, 48, 3), np.uint8))
    writer.release()
    plan = protected_plan(); plan.source = compute_source(str(video))
    path, _ = save_wipe_plan(plan, str(tmp_path / 'plan'))
    for options in ({'external_command': 'unused'}, {'model': 'propainter'}):
        with WipeEngine(task='clean', **options) as engine:
            monkeypatch.setattr(engine, '_ensure_model', lambda: pytest.fail('must reject before model load'))
            with pytest.raises(InvalidInputError, match='protect WipePlan v2'):
                engine.run(WipeRequest(video=video, plan=path, output_dir=tmp_path / 'out'))
