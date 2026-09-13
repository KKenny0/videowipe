"""STTN inpainter — the default built-in model.

The segment loop and its helpers were moved verbatim from ``tasks/detext.py``;
only their location changed. STTN's three-stage backend
(``encode``/``transform``/``decode``) is a private implementation detail of
this inpainter, not a public contract.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
from pathlib import Path
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from videowipe.backends import load_backend
from videowipe.inpainters.base import InpaintJob, InpaintOutcome

_VIDEO_CAPTURE_TYPE = cv2.VideoCapture


def get_ref_index(neighbor_ids, length, ref_length):
    """Select reference frames at regular intervals, excluding neighbors."""
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def blend_frames(comp_frames, pred_img, neighbor_ids, counts):
    """Blend predicted frames into the composite buffer.

    Pure-numpy replacement for the former @njit version.
    Accumulate repeated frame IDs; normalize once after all windows.
    """
    for i, idx in enumerate(neighbor_ids):
        img = pred_img[i].astype(np.float32)
        comp_frames[idx] += img
        counts[idx] += 1


def _blend_frame_regions(frame_ori, comp_frames, modes, mask):
    """Blend prepared model crops without converting the whole frame to float."""
    active = [
        (mode, comp)
        for mode, comp in zip(modes, comp_frames)
        if comp is not None
    ]
    frame = frame_ori.copy()
    if not active:
        return frame

    work_from = min(mode[0] for mode, _comp in active)
    work_to = max(mode[1] for mode, _comp in active)
    work = frame_ori[work_from:work_to].astype(np.float32)
    for (from_h, to_h), comp in active:
        mask_area = mask[from_h:to_h].astype(np.float32)
        relative = slice(from_h - work_from, to_h - work_from)
        work[relative] = (
            mask_area * comp
            + (1.0 - mask_area) * work[relative]
        )
    frame[work_from:work_to] = np.clip(work, 0, 255).astype(np.uint8)
    return frame


def _process_segment(frames, backend, w, h, ref_length, neighbor_stride):
    """Inpaint a batch of frames through the model (backend-agnostic)."""
    video_length = len(frames)
    feats = backend.encode(backend.preprocess(frames))

    comp_frames_np = np.zeros((video_length, h, w, 3), dtype=np.float32)
    counts = np.zeros((video_length,), dtype=np.int32)

    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [
            i for i in range(
                max(0, f - neighbor_stride),
                min(video_length, f + neighbor_stride + 1),
            )
        ]
        ref_ids = get_ref_index(neighbor_ids, video_length, ref_length)

        ids = neighbor_ids + ref_ids
        selected = feats[ids]
        transformed = backend.transform(selected)
        decoded = backend.decode(transformed[: len(neighbor_ids)])

        blend_frames(comp_frames_np, decoded, np.array(neighbor_ids), counts)

    comp_frames = []
    for idx in range(video_length):
        if counts[idx]:
            comp_frames.append(np.clip(comp_frames_np[idx] / counts[idx], 0, 255).astype(np.uint8))
        else:
            comp_frames.append(None)
    return comp_frames


def get_inpaint_mode(H, h, mask):
    """Determine inpainting segments based on mask position."""
    mode = []
    frontier = H
    while frontier > 0:
        base_start = max(0, frontier - h)
        unprocessed = mask[base_start:frontier, :]
        if not np.all(unprocessed == 0) and np.sum(unprocessed) > 10:
            from_H, to_H = base_start, frontier
            if frontier != H and base_start != 0:
                move = 0
                while frontier + move < H and not np.all(
                    mask[frontier + move, :] == 0
                ):
                    move += 1
                # Shift only when it neither runs off the frame nor drops mask
                # pixels from the still-unprocessed top edge.
                if (
                    0 < move < h
                    and frontier + move < H
                    and np.all(mask[base_start:base_start + move, :] == 0)
                ):
                    to_H += move
                    from_H += move
            mode.append((from_H, to_H))
        frontier = base_start
    return mode


class STTNInpainter:
    """Built-in STTN inpainter.

    Loads weights via :mod:`videowipe.backends` and runs the segment loop over
    the whole video. The loaded :class:`~videowipe.backends.InpaintBackend` is
    exposed as :attr:`backend` so callers (e.g. the engine benchmark) can read
    its class name.
    """

    name = "sttn"

    def __init__(self, ref_length: int = 5, neighbor_stride: int = 5):
        self.ref_length = ref_length
        self.neighbor_stride = neighbor_stride
        self.backend = None

    def load(self, weight_path: str, device: str = "auto") -> None:
        self.backend = load_backend(weight_path, device=device)
        print(
            f"Loaded weight: {weight_path} "
            f"(backend: {type(self.backend).__name__})"
        )

    def inpaint(self, job: InpaintJob) -> InpaintOutcome:
        if self.backend is None:
            raise RuntimeError(
                "STTNInpainter.load() must be called before inpaint()"
            )
        if job.reader is None:
            raise ValueError(
                "STTNInpainter requires job.reader (a cv2.VideoCapture)"
            )

        w, h = 640, 120
        video_length = job.frame_count
        ori_w, ori_h = job.width, job.height
        fps = job.fps
        gap = job.gap
        first, last = job.trial_range or (0, video_length)
        if not 0 <= first < last <= video_length:
            raise ValueError("trial_range exceeds the source video")
        first_segment = first // gap
        last_segment = (last + gap - 1) // gap
        context_start = first_segment * gap
        reader = job.reader

        video_name = (
            os.path.splitext(os.path.basename(job.video_path))[0]
            if job.video_path
            else "output"
        )
        video_out_path = os.path.join(
            job.output_dir, f"{video_name}_{job.output_suffix}{'_trial' if job.trial_range else ''}.mp4"
        )
        out_w = ori_w
        out_h = ori_h * 2 if job.dual else ori_h

        split_h = int(ori_w * 3 / 16)
        mode = get_inpaint_mode(ori_h, split_h, job.mask)
        if not mode:
            raise ValueError(
                "Mask has no inpaintable regions. "
                "The auto-detected mask is empty — the text detector may have "
                "failed to find subtitles. Try providing a mask manually with -m."
            )

        cache = None
        if job.prediction_cache_dir is not None:
            from videowipe.inpainters.prediction_cache import PredictionCache
            from videowipe.plan import _sha256_file
            runtime = self.backend.benchmark_metadata()
            implementation = hashlib.sha256()
            package = Path(__file__).resolve().parents[1]
            for name in ('inpainters/sttn.py', 'backends.py', 'models/sttn.py', 'core/spectral_norm.py'):
                implementation.update((package / name).read_bytes())
            identity = dict(runtime, source_sha256=_sha256_file(job.video_path),
                            source_size=[ori_w, ori_h], input_size=[w, h],
                            backend=type(self.backend).__name__,
                            precision='float16-autocast' if str(runtime['device']).startswith('cuda') else 'float32',
                            ref_length=self.ref_length, neighbor_stride=self.neighbor_stride,
                            implementation_sha256=implementation.hexdigest())
            cache = PredictionCache(job.prediction_cache_dir, identity)
        prediction_s = composition_s = 0.0
        output_reader = None

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error", "-nostats",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ]
        # Attach the original video as a second input so its audio stream (and
        # metadata) survives. The trailing "?" on "1:a?" makes ffmpeg tolerate
        # videos with no audio track instead of erroring out.
        if job.video_path:
            if job.trial_range is not None:
                ffmpeg_cmd += ["-ss", str(first / fps)]
            ffmpeg_cmd += ["-i", job.video_path]
        ffmpeg_cmd += [
            "-map", "0:v",
        ]
        if job.video_path:
            ffmpeg_cmd += ["-map", "1:a?"]
        ffmpeg_cmd += [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ]
        if job.video_path:
            ffmpeg_cmd += ["-c:a", "aac"]
        if job.trial_range is not None:
            ffmpeg_cmd += ["-t", str((last - first) / fps)]
        ffmpeg_cmd += [
            "-movflags", "+faststart",
            video_out_path,
        ]
        stderr_file = tempfile.TemporaryFile()
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=stderr_file)
        stdin_closed = False
        processing_failed = False
        try:
            if job.video_path and isinstance(reader, _VIDEO_CAPTURE_TYPE):
                candidate_reader = cv2.VideoCapture(job.video_path)
                if candidate_reader.isOpened():
                    output_reader = candidate_reader
                else:
                    candidate_reader.release()
            if context_start:
                if not reader.set(cv2.CAP_PROP_POS_FRAMES, context_start):
                    raise ValueError("Cannot seek source video for trial")
                if output_reader is not None and not output_reader.set(
                    cv2.CAP_PROP_POS_FRAMES, context_start
                ):
                    raise ValueError("Cannot seek comparison video for trial")

            # Backend instances are shared runtime objects; keep inference
            # serialized until the backend declares a thread-safety contract.
            t_inpaint_start = time.monotonic()
            output_decoded = context_start
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                for i in range(first_segment, last_segment):
                    start_f = i * gap
                    end_f = min((i + 1) * gap, video_length)
                    print(f"Processing frames {start_f + 1}-{end_f}/{video_length}")

                    # Cache only emitted masks. Keep all context frames whenever
                    # any emitted pixel in a band needs a prediction.
                    segment_masks = {}
                    active_modes = set(range(len(mode)))
                    if job.frame_mask is not None:
                        active_modes = set()
                        for index in range(max(start_f, first), min(end_f, last)):
                            alpha = np.asarray(job.frame_mask(index))
                            # Only the plan producer guarantees stable storage.
                            # Even readonly views from custom callbacks can alias
                            # a buffer that the next callback invocation changes.
                            if not getattr(job.frame_mask, "_videowipe_stable_masks", False):
                                alpha = alpha.copy()
                            if alpha.ndim == 2:
                                alpha = alpha[:, :, None]
                            segment_masks[index] = alpha
                            for k, (top, bottom) in enumerate(mode):
                                if k not in active_modes and np.any(alpha[top:bottom] != 0):
                                    active_modes.add(k)

                    prediction_started = time.monotonic()
                    comps = {}
                    keys = {}
                    if cache is not None:
                        for k in active_modes:
                            keys[k] = cache.key(start_f, end_f, *mode[k])
                            cached = cache.read(keys[k], end_f - start_f)
                            if cached is not None:
                                comps[k] = cached
                    missing_modes = active_modes - comps.keys()
                    frames_hr = [] if output_reader is None else None
                    frames = {k: [] for k in range(len(mode))}

                    for local_index in range(end_f - start_f):
                        success, image = reader.read()
                        if not success:
                            raise ValueError(
                                f"STTN inference decoded {start_f + local_index} "
                                f"frames; expected {video_length}"
                            )
                        if frames_hr is not None:
                            frames_hr.append(image)
                        for k in missing_modes:
                            from_h, to_h = mode[k]
                            frames[k].append(cv2.resize(
                                image[from_h:to_h], (w, h),
                                interpolation=cv2.INTER_LINEAR,
                            ).astype(np.float32))

                    futures = {
                        k: executor.submit(
                            _process_segment,
                            frames[k], self.backend, w, h,
                            self.ref_length, self.neighbor_stride,
                        )
                        for k in range(len(mode))
                        if frames[k]
                    }
                    for k, future in futures.items():
                        comps[k] = future.result()
                        if cache is not None:
                            cache.write(keys[k], comps[k])
                    prediction_s += time.monotonic() - prediction_started
                    if cache is not None and job.phase_progress is not None:
                        job.phase_progress('predict', min(end_f, last) - first, last - first,
                                           f'hits={cache.hits};misses={cache.misses}')
                    composition_started = time.monotonic()

                    for j in range(end_f - start_f):
                        if output_reader is not None:
                            success, frame_ori = output_reader.read()
                            if not success:
                                raise ValueError(
                                    f"STTN output decoded {output_decoded} frames; "
                                    f"expected {video_length}"
                                )
                            output_decoded += 1
                        else:
                            frame_ori = frames_hr[j]
                        if not first <= start_f + j < last:
                            continue
                        # Per-frame temporal mask (global index start_f + j) when a
                        # WipePlan supplies one; else the static whole-video mask.
                        if job.frame_mask is not None:
                            per_frame = segment_masks[start_f + j]
                        frame_comps = []
                        for k in range(len(mode)):
                            if comps.get(k) and j < len(comps[k]) and comps[k][j] is not None:
                                mode_height = mode[k][1] - mode[k][0]
                                comp = cv2.resize(
                                    comps[k][j], (ori_w, mode_height),
                                )
                                comp = cv2.cvtColor(
                                    np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB
                                ).astype(np.float32)
                                frame_comps.append(comp)
                            else:
                                frame_comps.append(None)
                        # Soft alpha blend: mask is float32 in [0,1] when
                        # feather_radius > 0, else uint8 in {0,1}.
                        blend_mask = per_frame if job.frame_mask is not None else job.mask
                        frame = _blend_frame_regions(
                            frame_ori, frame_comps, mode, blend_mask,
                        )
                        if job.dual:
                            frame = np.vstack([frame_ori, frame])
                        pipe.stdin.write(frame.tobytes())

                    composition_s += time.monotonic() - composition_started
                    if job.phase_progress is not None:
                        job.phase_progress("compose" if cache is not None else "inpaint", min(end_f, last) - first,
                                           last - first, f"bands={len(active_modes)}")
                    elif job.progress is not None:
                        job.progress(min(end_f, last) - first, last - first)

            if job.phase_progress is not None:
                job.phase_progress("encode", 0, 1, None)
            encoding_started = time.monotonic()
            pipe.stdin.close()
            stdin_closed = True
            pipe.wait()

            if isinstance(job.metrics, dict):
                job.metrics.update(prediction_s=round(prediction_s, 4),
                                   composition_s=round(composition_s, 4),
                                   encoding_finalize_s=round(time.monotonic() - encoding_started, 4))
                if cache is not None:
                    job.metrics.update(prediction_hits=cache.hits, prediction_misses=cache.misses,
                                       prediction_corrupt=cache.corrupt, prediction_writes=cache.writes)
                job.metrics["inpainting_s"] = round(
                    time.monotonic() - t_inpaint_start, 3
                )
        except Exception:
            processing_failed = True
            raise
        finally:
            if output_reader is not None:
                output_reader.release()
            if not stdin_closed and pipe.stdin is not None:
                pipe.stdin.close()
            if pipe.poll() is None:
                pipe.terminate()
                pipe.wait()
            if processing_failed:
                stderr_file.close()
        if pipe.returncode != 0:
            stderr_file.seek(0)
            stderr = stderr_file.read().decode(errors="replace")
            stderr_file.close()
            raise RuntimeError(
                f"FFmpeg exited with code {pipe.returncode}:\n"
                f"{stderr}"
            )
        stderr_file.close()
        if job.phase_progress is not None:
            job.phase_progress("encode", 1, 1, None)
        print(f"Saved to {video_out_path}")
        return InpaintOutcome(
            output_path=video_out_path,
            backend=type(self.backend).__name__,
        )

    def cleanup(self) -> None:
        if self.backend is not None:
            self.backend.cleanup()
            self.backend = None
