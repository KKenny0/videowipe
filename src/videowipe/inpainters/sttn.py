"""STTN inpainter — the default built-in model.

The segment loop and its helpers were moved verbatim from ``tasks/detext.py``;
only their location changed. STTN's three-stage backend
(``encode``/``transform``/``decode``) is a private implementation detail of
this inpainter, not a public contract.
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from videowipe.backends import load_backend
from videowipe.inpainters.base import InpaintJob, InpaintOutcome


def get_ref_index(neighbor_ids, length, ref_length):
    """Select reference frames at regular intervals, excluding neighbors."""
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def blend_frames(comp_frames, pred_img, neighbor_ids, mask):
    """Blend predicted frames into the composite buffer.

    Pure-numpy replacement for the former @njit version.
    neighbor_ids may contain duplicates across calls; for first hit the
    frame is written directly, for subsequent hits it is averaged.
    """
    for i, idx in enumerate(neighbor_ids):
        img = pred_img[i].astype(np.float32)
        if not mask[idx]:
            comp_frames[idx] = img
            mask[idx] = True
        else:
            comp_frames[idx] = 0.5 * comp_frames[idx] + 0.5 * img


def _process_segment(frames, backend, w, h, ref_length, neighbor_stride):
    """Inpaint a batch of frames through the model (backend-agnostic)."""
    video_length = len(frames)
    feats = backend.encode(backend.preprocess(frames))

    comp_frames_np = np.zeros((video_length, h, w, 3), dtype=np.float32)
    mask = np.zeros((video_length,), dtype=np.bool_)

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

        blend_frames(comp_frames_np, decoded, np.array(neighbor_ids), mask)

    comp_frames = []
    for idx in range(video_length):
        if mask[idx]:
            comp_frames.append(np.clip(comp_frames_np[idx], 0, 255).astype(np.uint8))
        else:
            comp_frames.append(None)
    return comp_frames


def get_inpaint_mode(H, h, mask):
    """Determine inpainting segments based on mask position."""
    mode = []
    to_H = from_H = H
    while from_H != 0:
        if to_H - h < 0:
            from_H = 0
            to_H = h
        else:
            from_H = to_H - h
        if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
            if to_H != H:
                move = 0
                while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                    move += 1
                if to_H + move < H and move < h:
                    to_H += move
                    from_H += move
            mode.append((from_H, to_H))
        to_H -= h
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
        reader = job.reader

        video_name = (
            os.path.splitext(os.path.basename(job.video_path))[0]
            if job.video_path
            else "output"
        )
        video_out_path = os.path.join(
            job.output_dir, f"{video_name}_{job.output_suffix}.mp4"
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
        ffmpeg_cmd += [
            "-movflags", "+faststart",
            video_out_path,
        ]
        stderr_file = tempfile.TemporaryFile()
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=stderr_file)
        stdin_closed = False
        processing_failed = False
        try:
            rec_time = (
                video_length // gap
                if video_length % gap == 0
                else video_length // gap + 1
            )

            # Backend instances are shared runtime objects; keep inference
            # serialized until the backend declares a thread-safety contract.
            t_inpaint_start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                for i in range(rec_time):
                    start_f = i * gap
                    end_f = min((i + 1) * gap, video_length)
                    print(f"Processing frames {start_f + 1}-{end_f}/{video_length}")

                    frames_hr = []
                    frames = {k: [] for k in range(len(mode))}

                    for j in range(start_f, end_f):
                        success, image = reader.read()
                        if not success:
                            break
                        frames_hr.append(image)
                        for k in range(len(mode)):
                            image_crop = image[mode[k][0]:mode[k][1], :, :]
                            image_resize = cv2.resize(image_crop, (w, h))
                            frames[k].append(image_resize)

                    if not frames_hr:
                        break

                    futures = {
                        k: executor.submit(
                            _process_segment,
                            frames[k], self.backend, w, h,
                            self.ref_length, self.neighbor_stride,
                        )
                        for k in range(len(mode))
                        if frames[k]
                    }
                    comps = {k: futures[k].result() for k in futures}

                    for j in range(len(frames_hr)):
                        frame_ori = frames_hr[j].copy()
                        frame = frames_hr[j].astype(np.float32)
                        for k in range(len(mode)):
                            if comps.get(k) and j < len(comps[k]):
                                comp = cv2.resize(comps[k][j], (ori_w, split_h))
                                comp = cv2.cvtColor(
                                    np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB
                                ).astype(np.float32)
                                # Soft alpha blend: mask is float32 in [0,1]
                                # when feather_radius > 0, else uint8 in {0,1}.
                                mask_area = job.mask[mode[k][0]:mode[k][1], :].astype(np.float32)
                                blended = (
                                    mask_area * comp
                                    + (1.0 - mask_area) * frame[mode[k][0]:mode[k][1], :, :]
                                )
                                frame[mode[k][0]:mode[k][1], :, :] = blended
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                        if job.dual:
                            frame = np.vstack([frame_ori, frame])
                        pipe.stdin.write(frame.tobytes())

                    if job.progress is not None:
                        job.progress(end_f, video_length)

            pipe.stdin.close()
            stdin_closed = True
            pipe.wait()

            if isinstance(job.metrics, dict):
                job.metrics["inpainting_s"] = round(
                    time.monotonic() - t_inpaint_start, 3
                )
        except Exception:
            processing_failed = True
            raise
        finally:
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
        print(f"Saved to {video_out_path}")
        return InpaintOutcome(
            output_path=video_out_path,
            backend=type(self.backend).__name__,
        )

    def cleanup(self) -> None:
        if self.backend is not None:
            self.backend.cleanup()
            self.backend = None
