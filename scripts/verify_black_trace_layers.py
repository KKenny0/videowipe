"""Audit English frame 152 without changing the plan or model.

PYTHONPATH=src python scripts/verify_black_trace_layers.py
The full-alpha ROI is a diagnostic counterfactual, never a production mask.
"""
import json
from pathlib import Path
import hashlib

import cv2
import numpy as np
from videowipe import WipeEngine
from videowipe.plan import load_wipe_plan, execution_masks
from videowipe.inpainters import sttn


def main():
    out = Path('result/black-trace-layers')
    out.mkdir(parents=True, exist_ok=True)
    path = Path('result/local-subtitle-tightening/english1/final-plan/wipe_plan.json')
    plan = load_wipe_plan(str(path))
    static, alpha = execution_masks(plan, 4)
    modes = sttn.get_inpaint_mode(480, int(852*3/16), static)
    assert len(modes) == 1
    top, bottom = modes[0]
    cap = cv2.VideoCapture('input/detext_examples/english1.mp4')
    frames = []
    for i in range(175):
        ok, frame = cap.read()
        assert ok
        if i >= 150:
            frames.append(frame)
    cap.release()
    with WipeEngine(task='clean', device='mps') as engine:
        engine._ensure_model()
        predictions = sttn._process_segment(
            [cv2.resize(f[top:bottom], (640,120)).astype(np.float32) for f in frames],
            engine._task_impl.inpainter.backend, 640, 120, 5, 5)
        runtime = engine._task_impl.inpainter.backend.benchmark_metadata()
    source = frames[2]
    prediction = cv2.cvtColor(cv2.resize(predictions[2], (852,bottom-top)), cv2.COLOR_RGB2BGR)
    mask = np.squeeze(alpha(152))[:, :, None]
    actual = sttn._blend_frame_regions(source, [prediction], modes, mask)
    raw = source.copy()
    raw[top:bottom] = prediction
    # ROI contains the lower outline of the 'y' in the first subtitle line.
    roi = (slice(412,416), slice(370,396))
    fixed_mask = mask.copy()
    fixed_mask[roi] = 1
    counterfactual = sttn._blend_frame_regions(source, [prediction], modes, fixed_mask)
    expected = np.clip(mask * raw + (1-mask) * source, 0,255).astype(np.uint8)
    assert np.array_equal(expected, actual)
    assert np.array_equal(counterfactual[roi], raw[roi])
    outside = np.ones(mask.shape[:2], bool)
    outside[roi] = False
    assert np.array_equal(counterfactual[outside], actual[outside])
    rows=[]
    for y in range(412,416):
        for x in range(370,396):
            rows.append(dict(x=x,y=y,alpha=float(mask[y,x,0]),source=source[y,x].tolist(),
                             prediction=raw[y,x].tolist(),composite=actual[y,x].tolist()))
    panels=[]
    for label,img in [('source',source),('raw prediction',raw),('actual blend',actual),('ROI alpha=1 (probe)',counterfactual)]:
        zoom=cv2.resize(img[402:422,366:400],(680,400),interpolation=cv2.INTER_NEAREST)
        cv2.putText(zoom,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
        panels.append(zoom)
    cv2.imwrite(str(out/'layers.png'),np.vstack(panels))
    np.savez_compressed(out/'layers.npz',source=source,prediction=raw,alpha=mask,actual=actual,counterfactual=counterfactual)
    report=dict(frame=152,roi=[370,412,396,416],pixels=rows,runtime=runtime,
                plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                blend_formula_exact=True,outside_probe_unchanged=True,
                roi_mean_bgr={k:float(v[roi].mean()) for k,v in [('source',source),('prediction',raw),('actual',actual),('counterfactual',counterfactual)]})
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(report['roi_mean_bgr'])


if __name__ == '__main__':
    main()
