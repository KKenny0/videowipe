"""Compare fixed-mask STTN context padding; diagnostic only, no default changes.

PYTHONPATH=src python scripts/verify_temporal_context.py --sample english1
Output clips are silent, cropped comparison strips: source / 0 / 5 / 25 padding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from videowipe import WipeEngine
from videowipe.inpainters import sttn
from videowipe.plan import load_wipe_plan, execution_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=['english1', 'others', 'chinese1'], required=True)
    parser.add_argument('--output', type=Path, default=Path('result/temporal-context'))
    args = parser.parse_args()
    out = args.output / args.sample
    out.mkdir(parents=True, exist_ok=True)
    video = Path(f'input/detext_examples/{args.sample}.mp4')
    plan_path = Path(f'result/local-subtitle-tightening/{args.sample}/final-plan/wipe_plan.json')
    plan = load_wipe_plan(str(plan_path))
    static, alpha = execution_masks(plan, 4)
    modes = sttn.get_inpaint_mode(plan.source.height, int(plan.source.width * 3 / 16), static)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for index in range(225):
        ok, frame = cap.read()
        assert ok, f'Cannot decode frame {index}'
        if index >= 100:
            frames.append(frame)
    cap.release()
    first, last = 140, 185
    sources = frames[first - 100:last - 100]
    masks = [alpha(i) for i in range(first, last)]
    outputs, rows = [], []
    with WipeEngine(task='clean', device='mps') as engine:
        engine._ensure_model()
        backend = engine._task_impl.inpainter.backend
        for padding in (0, 5, 25):
            started = time.monotonic()
            result = []
            for start in (125, 150, 175):
                lo, hi = start - padding, start + 25 + padding
                crops = []
                for top, bottom in modes:
                    batch = [cv2.resize(f[top:bottom], (640, 120)).astype(np.float32)
                             for f in frames[lo - 100:hi - 100]]
                    predictions = sttn._process_segment(batch, backend, 640, 120, 5, 5)
                    crops.append(predictions)
                for index in range(max(first, start), min(last, start + 25)):
                    prepared = [cv2.cvtColor(cv2.resize(pred[index - lo],
                                (plan.source.width, bottom - top)), cv2.COLOR_RGB2BGR)
                                for pred, (top, bottom) in zip(crops, modes)]
                    source = frames[index - 100]
                    mask = masks[index - first]
                    if mask.ndim == 2:
                        mask = mask[:, :, None]
                    cleaned = sttn._blend_frame_regions(source, prepared, modes, mask)
                    assert np.array_equal(cleaned[mask[:, :, 0] == 0], source[mask[:, :, 0] == 0])
                    result.append(cleaned)
            assert len(result) == last - first
            outputs.append(result)
            rows.append(dict(padding=padding, inference_and_composition_s=time.monotonic() - started))
            print(args.sample, rows[-1], flush=True)
        runtime = backend.benchmark_metadata()
    # Motion comes from the source, never from each candidate's reconstruction.
    for result, row in zip(outputs, rows):
        errors, boundaries = [], []
        for j in range(1, len(result)):
            gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sources[j-1:j+1]]
            flow = cv2.calcOpticalFlowFarneback(gray[1], gray[0], None, .5, 3, 15, 3, 5, 1.2, 0)
            yy, xx = np.indices(gray[0].shape, dtype=np.float32)
            warped = cv2.remap(result[j-1], xx + flow[:, :, 0], yy + flow[:, :, 1], cv2.INTER_LINEAR)
            previous_mask = cv2.remap(np.squeeze(masks[j-1]), xx + flow[:, :, 0], yy + flow[:, :, 1], cv2.INTER_NEAREST)
            region = (np.squeeze(masks[j]) > .5) & (previous_mask > .5)
            if region.any():
                error = float(np.abs(result[j].astype(np.float32) - warped)[region].mean())
                errors.append(error)
                if (first + j) % 25 == 0:
                    boundaries.append(dict(frame=first+j, mae=error))
        row.update(motion_aligned_mae=float(np.mean(errors)), boundaries=boundaries,
                   preencode_sha256=hashlib.sha256(b''.join(f.tobytes() for f in result)).hexdigest())
    y = int(plan.source.height * .7) // 2 * 2
    height = (plan.source.height - y) * 4
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{plan.source.width}x{height}', '-r', str(fps), '-i', '-',
               '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(out / 'comparison.mp4')]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for j, source in enumerate(sources):
            strips = [source[y:].copy()] + [r[j][y:].copy() for r in outputs]
            for strip, label in zip(strips, ['source', 'padding 0', 'padding 5', 'padding 25']):
                cv2.putText(strip, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            image = np.vstack(strips)
            pipe.stdin.write(image.tobytes())
            if first+j in (149, 150, 152, 174, 175):
                cv2.imwrite(str(out / f'frame-{first+j}.jpg'), image)
    finally:
        pipe.stdin.close()
        assert pipe.wait() == 0
    report = dict(sample=args.sample, frames=[first, last], runtime=runtime, variants=rows,
                  alpha_zero_unchanged=True, source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  implementation_sha256=hashlib.sha256(Path(sttn.__file__).read_bytes()).hexdigest(),
                  caveat='Source optical flow includes subtitles and occlusions; diagnostic MAE is not quality acceptance. Clips are silent cropped comparisons.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
