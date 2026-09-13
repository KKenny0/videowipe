"""Manual detector geometry must follow the content, not sentence length or padding."""
from types import SimpleNamespace

import cv2
import numpy as np

from videowipe.detect import DBNetDetector


def detector_for(prob, unclip=1.5):
    detector = object.__new__(DBNetDetector)
    detector._input_w = detector._input_h = 256
    detector._scale = 1 / 255
    detector._mean = (0, 0, 0)
    detector._bin_thresh = .3
    detector._box_thresh = .5
    detector._unclip_ratio = unclip
    detector._net = SimpleNamespace(setInput=lambda blob: None, forward=lambda: prob[None, None])
    return detector


def test_manual_unclip_uses_equal_pixel_margins_for_long_and_rotated_lines():
    frame = np.zeros((256, 256, 3), np.uint8)
    for angle in (0, 30, 90):
        prob = np.zeros((256, 256), np.float32)
        cv2.fillPoly(prob, [cv2.boxPoints(((128, 128), (160, 20), angle)).astype(np.int32)], .9)
        raw = cv2.findContours((prob > .3).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        before = np.sort(cv2.minAreaRect(raw)[1])
        boxes = detector_for(prob)._detect_manual(frame)
        assert len(boxes) == 1
        after = np.sort(cv2.minAreaRect(boxes[0].points)[1])
        np.testing.assert_allclose(after - before, np.repeat(before[0] * .5, 2), atol=1e-4)


def test_manual_probability_map_discards_letterbox_before_source_projection():
    for shape, rect, padding_rect in (
        ((128, 256), (40, 40, 200, 60), (40, 180, 200, 200)),
        ((256, 128), (40, 40, 60, 200), (180, 40, 200, 200)),
    ):
        prob = np.zeros((256, 256), np.float32)
        for x1, y1, x2, y2 in (rect, padding_rect):
            prob[y1:y2+1, x1:x2+1] = .9
        boxes = detector_for(prob, unclip=1)._detect_manual(np.zeros((*shape, 3), np.uint8))
        assert len(boxes) == 1  # Padding is not image evidence.
        points = boxes[0].points
        np.testing.assert_allclose([*points.min(axis=0), *points.max(axis=0)], rect, atol=1e-4)


def test_persistent_top_right_logo_classification_does_not_depend_on_box_padding():
    from videowipe.detect import _classify_region
    for bbox in ((1584, 61, 1885, 119), (1642, 61, 1872, 119)):
        kind, _, remove = _classify_region(bbox, 'top-right', [], 1920, 1080,
                                          presence_fraction=.92, appearance_stability=1.)
        assert kind == 'logo' and not remove
    # Recognized text remains stronger evidence than position.
    assert _classify_region((1642, 61, 1872, 119), 'top-right', ['www.example.com'],
                            1920, 1080, presence_fraction=.92, appearance_stability=1.)[0] == 'watermark'
