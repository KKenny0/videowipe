"""Subtitle removal task."""
import concurrent.futures
import os

import cv2
import numpy as np
from numba import njit

from videowipe.tasks.base import BaseTask, read_mask, read_frame_info


def get_ref_index(neighbor_ids, length, ref_length):
    """Select reference frames at regular intervals, excluding neighbors."""
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


@njit
def blend_frames(comp_frames, pred_img, neighbor_ids, mask):
    for i in range(len(neighbor_ids)):
        idx = neighbor_ids[i]
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


class DetextTask(BaseTask):
    """Remove hardcoded subtitles from video."""

    def process_video(self, reader, frame_info, mask, output_dir: str,
                      video_path: str = "") -> str:
        w, h = 640, 120
        video_length = frame_info["len"]
        ori_w, ori_h = frame_info["W_ori"], frame_info["H_ori"]
        fps = frame_info["fps"]

        video_name = os.path.splitext(os.path.basename(video_path))[0] if video_path else "output"
        video_out_path = os.path.join(output_dir, f"{video_name}_detext.mp4")
        writer = cv2.VideoWriter(
            video_out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (ori_w, ori_h) if not self.dual else (ori_w, ori_h * 2),
        )

        split_h = int(ori_w * 3 / 16)
        mode = get_inpaint_mode(ori_h, split_h, mask)
        if not mode:
            writer.release()
            reader.release()
            raise ValueError(
                "Mask has no inpaintable regions. "
                "The auto-detected mask is empty — the text detector may have "
                "failed to find subtitles. Try providing a mask manually with -m."
            )

        rec_time = (
            video_length // self.gap
            if video_length % self.gap == 0
            else video_length // self.gap + 1
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(mode)) as executor:
            for i in range(rec_time):
                start_f = i * self.gap
                end_f = min((i + 1) * self.gap, video_length)
                print(f"Processing frames {start_f + 1}-{end_f}/{video_length}")

                frames_hr = []
                frames = {k: [] for k in range(len(mode))}
                comps = {}

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
                    frame = frames_hr[j]
                    for k in range(len(mode)):
                        if comps.get(k) and j < len(comps[k]):
                            comp = cv2.resize(comps[k][j], (ori_w, split_h))
                            comp = cv2.cvtColor(
                                np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB
                            )
                            mask_area = mask[mode[k][0]:mode[k][1], :]
                            frame[mode[k][0]:mode[k][1], :, :] = (
                                mask_area * comp
                                + (1 - mask_area) * frame[mode[k][0]:mode[k][1], :, :]
                            )
                    if self.dual:
                        frame = np.vstack([frame_ori, frame])
                    writer.write(frame)

        writer.release()
        reader.release()
        print(f"Saved to {video_out_path}")
        return video_out_path
