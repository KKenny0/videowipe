## What's new

- Added a structured Python SDK with request, result, progress, cancellation, and stable error contracts.
- Made `WipeEngine` reusable across sequential jobs so one loaded model can serve a long-running worker or batch.
- Added a public Inpainter registry and runnable examples for integrating third-party video-repair backends.
- Switched the base package to headless OpenCV and kept ONNX, PyTorch, OCR, Web, and ProPainter support optional.
- Added a packaged ProPainter adapter without redistributing its separately licensed source code or model weights.
- Hardened wheel and source archive contents with isolated install checks and explicit package verification.

## Verification

- 101 tests passed on Windows.
- Scoped ruff checks passed.
- Wheel and source archives passed content checks and a clean-environment install smoke test.

## Notes

- Python 3.10 or newer is now required.
- VideoWipe is distributed under GPL-3.0; review the license before embedding or redistributing it.
- PyPI publishing remains out of scope for this release; install from source or use the published container images.
- ProPainter remains an optional external integration with its own license and setup requirements.
