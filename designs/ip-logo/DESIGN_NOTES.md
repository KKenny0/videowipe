# VideoWipe IP Logo — Design Notes

Date: 2026-11-15 · Skill: [ip-as-logo](https://github.com/s1dashu/ip-as-logo-skill) · Model: GPT Image 2 (codex/gpt-image-2 via Magpie)

## Chosen mark

**A3** — baby elephant, slate blue-grey body, warm vivid orange eraser-cap band
near the trunk tip + orange belly, gently muted warm cream background.
Lower-left emergence, 1254×1254 native.

Concept: the trunk *is* a soft eraser — "gently and thoroughly wipes video clean".
Orange keeps continuity with the previous `#FF6B35` brand accent (also the web UI
`--accent`); blue-grey reads calm and reliable; cream keeps it warm and print-friendly.

## Batch history

| Batch | Labels | Note |
|---|---|---|
| Round 1 — 3 directions × 2 | A1–C2 | elephant / owl / ghost, user picked elephant |
| Round 2 — eraser emphasis | A3–A6 | A3 selected (band + blunt tip reads as eraser cap) |
| Round 3 — A3 tweaks | A3a–A3d | single-variable edits; user returned to A3 |

## 32×32 legibility check (quantitative)

- Background mean luminance 232/255 vs subject mean 161/255 → ~28% separation.
- Top quantized colors at 32px: slate blue-grey (140,157,182), cream (252,238,212),
  orange-family (172,140,126 quantized) — all three semantic families survive downsize.

## Assets shipped

- `assets/logo.png` — 1254×1254 master (A3 as-generated)
- `assets/logo-256/128/64.png` — resized masters (two-step LANCZOS via 512 +
  UnsharpMask radius 1.4 / percent 90 for crisp small sizes)
- `assets/logo.svg` — legacy mark, kept for backward compatibility
- `src/videowipe/web/index.html` — base64-inlined favicon (full square, no crop,
  same crisp pipeline) + topbar logo (33×33 rounded), no static-file routes needed
- `README.md` / `README_CN.md` — header image now `assets/logo.png` (100×100)

Only the chosen mark (`assets/`) and these notes are tracked in git; the other
candidates in this directory remain local exploration material.

## Regeneration

All prompts follow the ip-as-logo skill skeleton (three semantic colors, 4–7
shapes, lower-left emergence, Constraints-in-main-prompt for GPT Image 2).
Prompts for every candidate are reproducible from the batch tables above;
see `/tmp/vwgen-*.log` during the session that generated them.
