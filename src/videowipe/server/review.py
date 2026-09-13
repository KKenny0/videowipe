"""Compile saved human decisions without changing detector evidence."""
from __future__ import annotations

import math
import re
import numpy as np

from videowipe.errors import InvalidInputError
from videowipe.plan import Segment, Track, validate_plan


def seconds_to_frames(start, end, source):
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end)):
        raise InvalidInputError("time must be finite")
    if not 0 <= start < end <= source.frame_count / source.fps:
        raise InvalidInputError("time interval exceeds the video")
    frames = [value * source.fps for value in (start, end)]
    frames = [round(value) if abs(value-round(value)) < 1e-9 else value for value in frames]
    return math.floor(frames[0]), min(source.frame_count, math.ceil(frames[1]))


def normalize_segments(values, frame_count):
    if not isinstance(values, (list, tuple)) or not values or len(values) > 512:
        raise InvalidInputError("provide 1–512 non-empty frame intervals")
    rows = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(n) is not int for n in value):
            raise InvalidInputError("segments require integer frame pairs")
        first, last = value
        if not 0 <= first < last <= frame_count:
            raise InvalidInputError("segment exceeds source frame range")
        rows.append((first, last))
    merged = []
    for first, last in sorted(set(rows)):
        if merged and first <= merged[-1].end:
            merged[-1].end = max(last, merged[-1].end)
        else:
            merged.append(Segment(first, last))
    return merged


def rectangle_mask(box, source):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(n) is not int for n in box):
        raise InvalidInputError("bbox must contain integer coordinates")
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise InvalidInputError("bbox is inverted or empty")
    if x2 - x1 + 1 < 2 or y2 - y1 + 1 < 2:
        raise InvalidInputError("bbox must be at least 2x2 pixels")
    if not (0 <= x1 < x2 < source.width and 0 <= y1 < y2 < source.height):
        raise InvalidInputError("bbox exceeds source dimensions")
    mask = np.zeros((source.height, source.width), dtype=np.uint8)
    mask[y1:y2+1, x1:x2+1] = 1
    return mask


def compile_review(plan, selected_ids=None, bbox_overrides=None, segment_overrides=None,
                   protections=None, *, allow_empty=False):
    selected = set(selected_ids if selected_ids is not None else (t.id for t in plan.remove_tracks))
    known = {t.id for t in plan.tracks}
    boxes, segments = bbox_overrides or {}, segment_overrides or {}
    if selected - known:
        raise InvalidInputError("unknown candidate id: " + ", ".join(sorted(selected - known)))
    if not selected and not allow_empty:
        raise InvalidInputError("select at least one target")
    for values in (boxes, segments):
        if not isinstance(values, dict) or set(values) - known:
            raise InvalidInputError("unknown override id")
        if set(values) - selected:
            raise InvalidInputError("override requires selected target")
    for track in plan.tracks:
        track.action = 'remove' if track.id in selected else 'keep'
        track.decision_reason = f'user-confirm:{track.action}'
        if track.id in boxes:
            track.spatial_segments = None
            track.mask = rectangle_mask(boxes[track.id], plan.source)
            track.bbox = tuple(boxes[track.id])
            track.decision_reason += ':bbox-override'
        if track.id in segments:
            track.segments = normalize_segments(segments[track.id], plan.source.frame_count)
            track.decision_reason += ':segment-override'
    if protections is not None and (not isinstance(protections, list) or len(protections) > 64):
        raise InvalidInputError("at most 64 protection regions")
    for region in protections or []:
        if not isinstance(region, dict) or set(region) != {'id', 'bbox', 'segments'}:
            raise InvalidInputError("invalid protection region")
        name = region['id']
        if not isinstance(name, str) or not re.fullmatch(r'p_[a-zA-Z0-9_-]{1,64}', name) or name in known:
            raise InvalidInputError("invalid or duplicate protection id")
        known.add(name)
        mask = rectangle_mask(region['bbox'], plan.source)
        intervals = normalize_segments(region['segments'], plan.source.frame_count)
        plan.tracks.append(Track(name, 'protection', '保护区域', 'protect', tuple(region['bbox']),
                                 1, 1, 'user-protect', intervals, name, mask))
    plan.schema_version = (3 if any(t.spatial_segments is not None for t in plan.tracks)
                           else 2 if any(t.action == "protect" for t in plan.tracks) else 1)
    validate_plan(plan, require_remove=not allow_empty)
    return plan, sorted(selected)


def review_windows(plan, marks, warnings, track_id=None):
    padding = max(1, round(plan.source.fps))
    windows = []
    def add(frame, reason, target=None, priority=2):
        if 0 <= frame < plan.source.frame_count:
            windows.append({'start': max(0, frame-padding), 'end': min(plan.source.frame_count, frame+padding+1),
                            'priority': priority, 'origins': [{'frame': frame, 'reason': reason, 'track_id': target}]})
    for mark in marks:
        add(mark['frame_index'], '用户标记', priority=0)
    for warning in warnings:
        for match in re.finditer(r'\bframes?[ :=]+(\d+)\b', warning, re.I):
            add(int(match[1]), warning, priority=1)
    for track in plan.remove_tracks:
        if track_id is not None and track.id != track_id:
            continue
        if track_id:
            for segment in track.segments:
                add(segment.start, '开始生效', track.id)
                add(segment.end-1, '结束生效', track.id)
        elif track.segments:
            add(track.segments[0].start, '首次生效', track.id)
            add(track.segments[-1].end-1, '末次生效', track.id)
    merged = []
    for row in sorted(windows, key=lambda row: row['start']):
        if merged and row['start'] <= merged[-1]['end']:
            previous = merged[-1]
            previous['end'] = max(previous['end'], row['end'])
            previous['priority'] = min(previous['priority'], row['priority'])
            previous['origins'].extend(o for o in row['origins'] if o not in previous['origins'])
        else:
            merged.append(row)
    return sorted(merged, key=lambda row: (row['priority'], row['start']))
