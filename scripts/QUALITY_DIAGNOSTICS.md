# Quality diagnostics

These maintenance scripts investigate reconstruction and compositing; they do not change production defaults or certify visual quality. Run from the repository root with `PYTHONPATH=src`, the Torch optional dependencies, FFmpeg, and an MPS-capable Mac. Inputs are the three videos under `input/detext_examples/`. Outputs stay in ignored `result/` directories.

| Script | Purpose and prerequisites |
| --- | --- |
| `verify_reconstruction_truth.py` | Generates synthetic Latin subtitles over known backgrounds and compares STTN predictions with the clean truth. Creates `result/reconstruction-truth/*/oracle.npz`. |
| `score_propainter_truth.py SAMPLE FRAMES` | Scores lossless ProPainter PNG frames against that oracle. Run the synthetic oracle first; generate candidate frames separately with matching inputs and `--mask_dilation 0 --save_frames`. Does not install or invoke ProPainter. |
| `verify_temporal_context.py --sample SAMPLE` | Compares temporal padding 0/5/25 with a fixed mask. Requires the local investigation plan at `result/local-subtitle-tightening/SAMPLE/final-plan/wipe_plan.json` and its mask sidecar. |
| `verify_black_trace_layers.py` | Reproduces the English frame-152 alpha-blending diagnosis. Requires the English investigation plan above. Full-alpha output is a diagnostic counterfactual, not a production mask. |

The two plan-dependent scripts reproduce a specific investigation snapshot. Those generated plans are not shipped; a fresh checkout can run the synthetic oracle but needs the original plan artifacts to reproduce those historical probes exactly. Synthetic scores do not replace real-video review, and optical-flow diagnostics are not calibrated flicker or OCR scores.

Example:

```sh
PYTHONPATH=src python scripts/verify_reconstruction_truth.py
PYTHONPATH=src python scripts/verify_temporal_context.py --sample english1
```

Current release blockers are documented in the root README and the delivery plan. The diagnostic code is committed for reproducibility; model weights, generated frames and processed videos are not.
