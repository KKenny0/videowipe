# Agent Instructions

## Project Map

- `src/videowipe/` is the only supported package source.
- `src/videowipe/engine.py` owns the public `WipeEngine` and `remove_text` API.
- `src/videowipe/tasks/` owns task orchestration and video processing.
- `src/videowipe/backends.py` owns Torch and ONNX runtime loading.
- `src/videowipe/detect.py` owns auto mask generation.
- `src/videowipe/external.py` owns the external-model subprocess adapter
  (`run_external`, `ExternalModelError`) used by `--external-command`.
- `src/videowipe/ocr.py` owns optional RapidOCR text recognition.
- `src/videowipe/agent.py` owns optional local-agent candidate selection.
- `src/videowipe/weights.py` owns default-weight download from GitHub Releases.
- `scripts/` is for maintenance tooling only; scripts must import from `src/videowipe`.

## Boundaries

- Do not reintroduce root-level runtime packages such as `core/`, `model/`, or `demo.py`.
- Keep generated artifacts, exported weights, inspection frames, and processed videos out of git.
- Do not expose CLI commands for placeholder tasks.
- Library code should raise exceptions; CLI code should translate exceptions into process exits.

## Verification

- Run `make check` before handing off code changes.
- For mask-quality changes, also compare generated masks against `input/detext_examples/mask/*.png`.

## Agent skills

### Issue tracker
Specs and tickets use GitHub Issues in KKenny0/videowipe.
See docs/agents/issue-tracker.md.

### Domain docs
Single-context: CONTEXT.md and docs/adr/.
See docs/agents/domain.md.
