"""Score lossless ProPainter frames against the existing synthetic oracle.

Run after inference with --mask_dilation 0 --save_frames, passing sample and
output/frames/frames (ProPainter names its output after the input directory).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from verify_reconstruction_truth import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sample', choices=['english1', 'others', 'chinese1'])
    p.add_argument('frames', type=Path)
    args = p.parse_args()
    oracle = Path('result/reconstruction-truth') / args.sample / 'oracle.npz'
    with np.load(oracle) as a:
        truth, dirty, glyph, mask = [a[k] for k in ('truth', 'input', 'glyph', 'mask')]
    files = sorted(args.frames.glob('*.png'))
    if len(files) != len(truth):
        raise ValueError(f'Expected {len(truth)} frames, found {len(files)}')
    result = np.asarray([cv2.imread(str(f)) for f in files])
    if result.shape != truth.shape or result.dtype != np.uint8:
        raise ValueError('Candidate frame dimensions/dtype differ from oracle')
    metrics = score(result, truth, dirty, glyph, mask)
    # Report outside-mask drift rather than silently repairing the candidate.
    metrics['outside_mae'] = float(np.abs(result.astype(float)-truth)[~mask].mean())
    metrics['sample'] = args.sample
    destination = args.frames.parent / 'truth-score.json'
    destination.write_text(json.dumps(metrics, indent=2))
    for j in (10, 24, 25, 35):
        panels = [truth[j].copy(), dirty[j].copy(), result[j].copy()]
        for panel, label in zip(panels, ('truth', 'subtitle input', 'ProPainter')):
            cv2.putText(panel, label, (5,16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,255), 1)
        cv2.imwrite(str(args.frames.parent / f'comparison-{j}.png'), np.vstack(panels))
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
