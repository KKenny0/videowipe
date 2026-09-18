"""Fixed synthetic subtitle oracle at STTN input resolution; no detector involved.

Run: PYTHONPATH=src python scripts/verify_reconstruction_truth.py
Uses decoded upper bands from existing samples, preserving them as exact truth.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from videowipe import WipeEngine
from videowipe.inpainters import sttn


def score(result, truth, dirty, glyph, mask):
    error = result.astype(np.float32) - truth
    delta = dirty.astype(np.float32) - truth
    background = mask & ~glyph
    return {
        'glyph_mae': float(np.abs(error)[glyph].mean()),
        'background_mae': float(np.abs(error)[background].mean()),
        'subtitle_error_projection': float((error[glyph] * delta[glyph]).sum() /
                                            np.square(delta[glyph]).sum()),
        'outside_equal': bool(np.array_equal(result[~mask], truth[~mask])),
    }


def main():
    out = Path('result/reconstruction-truth')
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    with WipeEngine(task='clean', device='mps') as engine:
        engine._ensure_model()
        backend = engine._task_impl.inpainter.backend
        for sample in ('english1', 'others', 'chinese1'):
            folder = out / sample
            folder.mkdir(exist_ok=True)
            video = Path(f'input/detext_examples/{sample}.mp4')
            cap = cv2.VideoCapture(str(video))
            truth = []
            for index in range(200):
                ok, frame = cap.read()
                assert ok
                if index >= 100:
                    y = int(frame.shape[0] * .2)
                    height = int(frame.shape[1] * 3 / 16)
                    truth.append(cv2.resize(frame[y:y+height], (640, 120)))
            cap.release()
            truth = np.asarray(truth)
            dirty = truth.copy()
            glyph = np.zeros(truth.shape[:3], bool)
            mask = np.zeros_like(glyph)
            for index in range(100):
                if 32 <= index < 68:
                    text = 'Are you serious?' if index < 50 else 'Serious about what?'
                    stencil = np.zeros((120, 640), np.uint8)
                    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
                    x, y = (640-w)//2, 75
                    cv2.putText(stencil, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 5, cv2.LINE_AA)
                    cv2.putText(dirty[index], text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 5, cv2.LINE_AA)
                    cv2.putText(dirty[index], text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)
                    glyph[index] = stencil > 0
                    mask[index, y-h-5:y+base+5, x-5:x+w+5] = True
            assert not np.any(glyph & ~mask)
            target = slice(25,75)
            clean, contaminated = truth[target], dirty[target]
            g, m = glyph[target], mask[target]
            variants = {'identity': contaminated, 'oracle': clean}
            for padding in (0,5):
                predictions = []
                for start in (25,50):
                    lo, hi = start-padding, start+25+padding
                    batch = [f.astype(np.float32) for f in dirty[lo:hi]]
                    raw = sttn._process_segment(batch, backend, 640, 120, 5, 5)
                    predictions.extend(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in raw[padding:padding+25])
                predictions = np.asarray(predictions)
                variants[f'padding_{padding}'] = np.where(m[:,:,:,None], predictions, contaminated)
            scores = {name:score(result, clean, contaminated, g, m) for name,result in variants.items()}
            assert scores['oracle']['glyph_mae'] == scores['oracle']['background_mae'] == 0
            assert scores['identity']['glyph_mae'] > 0
            assert abs(scores['identity']['subtitle_error_projection'] - 1) < 1e-6
            assert scores['identity']['background_mae'] == 0
            assert all(row['outside_equal'] for row in scores.values())
            # Clean reconstruction truth also defines the expected temporal delta.
            for name,result in variants.items():
                error = result.astype(np.float32)-clean
                common = m[1:] & m[:-1]
                scores[name]['temporal_error_delta_mae'] = float(np.abs(np.diff(error,axis=0))[common].mean())
            np.savez_compressed(folder/'oracle.npz', truth=clean, input=contaminated, glyph=g, mask=m)
            command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','640x480','-r','25','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(folder/'comparison.mp4')]
            pipe=subprocess.Popen(command,stdin=subprocess.PIPE)
            try:
                for j in range(50):
                    panels=[clean[j].copy(),contaminated[j].copy(),variants['padding_0'][j].copy(),variants['padding_5'][j].copy()]
                    for panel,label in zip(panels,['truth','subtitle input','padding 0','padding 5']):
                        cv2.putText(panel,label,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                    image=np.vstack(panels)
                    pipe.stdin.write(image.tobytes())
                    if j in (10,24,25,35):cv2.imwrite(str(folder/f'frame-{j}.png'),image)
            finally:
                pipe.stdin.close()
                assert pipe.wait()==0
            row=dict(sample=sample,scores=scores,source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),truth_sha256=hashlib.sha256(clean.tobytes()).hexdigest())
            reports.append(row)
            print(json.dumps(row),flush=True)
        metadata=backend.benchmark_metadata()
    report=dict(runtime=metadata,samples=reports,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Synthetic Latin subtitles at model resolution; bypasses detector, resizing and real encoding damage. Projection is correlation with injected text error, not OCR residual rate. Temporal error delta is not motion aligned.')
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
