# Cleanup workspace details

[Back to README](../README.md)

The local workspace plays the selected file immediately. As soon as candidates appear, target
cards show a frame where each target was observed and its time ranges; unchecked
targets remain available for inspection. A three-second trial is recommended from
the longest continuous interval containing selected targets, skipping empty intros.
You can also choose a manual 3/5-second window or start the full cleanup directly.
Short clips and windows near the end are shortened to fit.

Trials place the original above the cleaned frames in one synchronized video,
with the interval's audio. Changing targets, boxes, or time invalidates the trial;
failed trials can be retried without uploading again. Full cleanup starts only
when requested. The result plays in the same workspace at the last trial position,
with paused original/result switching and an original-name `_clean.mp4` download.
Full runs write to separate directories before publishing a successful result.

The workspace supports narrow screens, keyboard box editing, and light/dark themes.
Target and box edits are saved when an interaction ends, even while timing refinement
is running; trial and full cleanup wait for refinement to finish. Refresh or a server
restart restores the last saved task after validating its source and artifacts. An
interrupted run requires an explicit retry; it does not resume inference automatically.
Cancel waits for the worker to exit before another task can start. Closing a tab does
not cancel processing. Progress shows actual phase counts and, when enough comparable
work has finished, a remaining-time range and the loaded device.

Identical trials reuse validated results (up to three entries and 128 MiB per task).
Changing an execution setting invalidates the cache; currently playing results are
protected from eviction. Concurrent tabs use revision checks to prevent silent overwrite.
If a source codec cannot play in the browser, detection and target-frame inspection
remain available, without automatic proxy transcoding. Acceptance covers CFR SDR
video; detected HDR or possibly variable frame rates produce a notice. A successful
trial does not guarantee the whole video's quality; review the result before export.

Result review windows use first/last active boundaries, manual marks, and warnings with frame references. Up to 12 suggestions appear by default; expand a target for all boundaries and loop a window. These are navigation evidence, not automatic quality verdicts. Seen/issue marks belong to a result revision; only issue marks with unchanged runtime identity, crop bands, and per-frame alpha carry into a new result.

Edit target time intervals or draw a protection rectangle; coordinate and time inputs also support keyboard use. Times become half-open frame intervals (start rounded down, end rounded up), sorted and merged. Edits can be undone. “Actual processing area” displays the server's current alpha, including feathering and protection. Protected pixels match the source exactly before encoding; lossy re-encoding can still change decoded pixels.

Balanced/sensitive planning reuses dense detections to give subtitle targets frame-local rectangles, with height stabilized within 0.25 seconds on either side without crossing gaps. Fast mode and fallback-only candidates retain static masks. These plans use v3 and retain one reviewable target per subtitle track. V1/v2 plans keep their original behavior; static plans with protection use v2, otherwise v1. Editing time intervals clips local evidence; it does not invent masks for absent frames. A manual rectangle replaces local masks for that target. Unobserved frames introduced by time-boundary interpolation are checked before saving; known empty/error frames stay untouched. Saved plans replay without detection; older applications reject v3 rather than flattening it. Protection overrides removal, while keep only deselects a candidate. External models/ProPainter reject protection plans instead of dropping protection. Detection evidence, reviewed plans, and successful exports remain separate; edits require a new trial or export.

The web UI enables per-task prediction reuse after the three-sample, three-repeat speed and pixel-equivalence gate. This is not a detector or visual-quality acceptance claim.

SDK callers can opt into STTN prediction reuse with `WipeRequest(prediction_cache_dir=..., ...)`; it defaults to `None`. Each export decodes the original, recomposites, and re-encodes. Source, crop bands, global frame segment, weights, device/precision, and implementation identify a prediction. The cap is 512 MiB per directory and 2 GiB across sibling task directories with the same cache-directory name; use a common `task-root/task-id/predictions` layout to share the total quota. Full caches stop admitting new entries, and corrupt entries are recomputed. HTTP clients cannot supply cache paths; the UI can clear this task's predictions.

Trials use the built-in STTN backend and full-run context. SDK callers can use `WipeRequest(trial_range=(start_frame,
end_frame), ...)` for a comparison video with a half-open frame interval.
`preview=True` still means detection-only and cannot be combined with a trial.
