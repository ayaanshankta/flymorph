# FlyMorph — design spec

Date: 2026-09-26 · Status: approved in conversation · Owner: Ayaan Shankta

## 1. Goal

Rebuild FlyAreaDetect from scratch as a portfolio-grade TypeScript project that measures
Drosophila **larva area** and **adult head / thorax / abdomen area** in mm² from stereo-microscope
images, plus an explainer website Ayaan uses to rebuild the project himself, milestone by milestone.

Success criteria
- Larva: given an image + scale image, outputs one row per larva (px, mm²) and an overlay; matches the
  old `segment.py` areas on the same images (after scale correction) within 2%, and rejects specks.
- Scale: automatic px/mm from the stage-micrometer image (1 mm bar, 100 ticks); manual 2-click fallback.
- Adult: per fly, per part (head, thorax, abdomen) area mean ± SD in mm², running in the browser.
  Accuracy (Dice, % area error) reported only on Ayaan-approved masks from held-out imaging sessions.
- Same core runs in the browser (React) and in Node (CLI: folder → CSV).
- Explainer site: one chapter per git-tagged milestone (m01–m10) + ML deep-dive + interview prep.

Non-goals: backend/database, auth, wing or leg measurement, touching-fly instance separation,
auto larva/adult detection (user picks the mode).

## 2. Inputs (read-only)

Original data stays in `~/Downloads/FlyAreaDetect/` and is **never modified or copied wholesale**
(16 GB free disk). Referenced by path via `ml/data/manifest.csv`.
- Larva: 161 jpg in `data/larva/**`, bright larvae on dark background, 3584×2746.
- Adult: 219 jpg in `data/adult/**` + 592 in `ML-data/Microscope Images/**` (39 scale images).
  White or black background, 1–2 flies per image, legs/wings present.
- Scale images: stage micrometer, 1 mm bar with 0.01 mm ticks, ≈914 px/mm at the usual zoom.

## 3. Architecture

npm-workspaces monorepo:

```
flymorph/
  packages/core/   pure TS, no DOM: image ops, larva pipeline, calibration, adult inference, measurements
  apps/web/        Vite + React + TS: upload, canvas overlay, results table, CSV, /review page
  apps/cli/        Node: `flymorph larva|adult <dir> --scale <img> --out results.csv`
  ml/              Python: SAM drafting, draft server for review, training, eval, ONNX export
  site/            standalone HTML explainer (opened in Chrome; GitHub Pages later)
  docs/            spec, plan
```

Core types (the only shared interface):
```ts
type Gray = { w: number; h: number; data: Uint8Array };          // 1 channel
type RGB  = { w: number; h: number; data: Uint8ClampedArray };   // RGBA as from canvas / sharp
type Calibration = { pxPerMm: number; confidence: number; method: 'fft' | 'manual' };
type Measurement = { id: number; part: 'larva'|'head'|'thorax'|'abdomen'|'body';
                     areaPx: number; areaMm2: number; sdMm2?: number;
                     bbox: [x: number, y: number, w: number, h: number]; flag?: string };
```
Image decoding is the only platform-specific piece: canvas in web, `sharp` in CLI. Model runtime:
`onnxruntime-web` (WebGPU → WASM fallback) and `onnxruntime-node`, both behind one `runModel()`
function taking a Float32Array.

## 4. Pipelines

**Larva (classical CV, pure TS)**: grayscale (V channel) → Gaussian blur → Otsu threshold (old
code used fixed V≥95; Otsu with that as fallback floor) → open 5 → close 9 → fill holes (flood
from border) → connected components (8-conn) → filters: area ≥ 0.05 mm², aspect ≥ 1.3,
fill ratio ≥ 0.25 → Measurement per larva.

**Scale calibration (FFT)**: grayscale → find the bar row band (rows with the strongest
horizontal high-frequency energy) → average those rows into a 1-D profile → detrend → FFT → dominant
period p (px per 0.01 mm tick) → pxPerMm = 100 / p; bar extent (first/last tick) as cross-check.
Confidence = peak-to-median spectral ratio. Low confidence → UI asks for two clicks on bar ends.

**Adult (ML)**: resize fly image so long side = 384 (letterbox) → normalise → U-Net with
MC-dropout, T = 8 passes → per-pixel mean softmax → argmax label map → connected components of
non-background = fly instances (min area filter) → per instance per part: area mean ± SD across
the T passes, scaled back to original pixels, then to mm². `body` = head+thorax+abdomen.
Flag if SD/mean > 10% ("check this fly").

## 5. ML

- **Teacher drafts (SAM 2 small, MPS)**: saturation/colour threshold → fly blobs → principal axis via
  moments → eye end found by red/orange hue → point prompts at ~15% / 45% / 75% along the axis →
  SAM masks for head / thorax / abdomen → resolve overlaps (smaller mask wins) → draft PNG.
- **Review**: `/review` page lists drafts; keys A approve / R reject / click-to-fix (sends points to
  local `ml/draft_server.py`, which re-prompts SAM). Approved masks → `ml/data/masks/<hash>.png`.
  Only approved masks are ground truth. Target ≈ 60 approved images.
- **Student**: U-Net, MobileNetV3-small encoder (`segmentation_models_pytorch`), dropout 0.2 in
  decoder, 384×384, 4 classes, Dice + CE loss, augment (flip, rotate, colour jitter, bg inversion).
- **Split**: by imaging session (top-level dated folder), 70/15/15; test-set leakage checked by
  SHA-1 of image bytes.
- **Export**: ONNX opset 17, dropout kept active via a `mc` graph variant; int8 quantisation only if
  test Dice drops < 0.01. Target < 3 MB.
- **Compute**: Mac MPS first; Windows RTX 3060 if an epoch takes > 2 min.

## 6. Error handling

UI surfaces: unreadable file; no scale found / low confidence → manual 2-click; no larva/fly found;
model failed to load (WebGPU unavailable → WASM, reported); high uncertainty flag per fly. CLI exits
non-zero with the file name on any per-file failure, continuing the batch and listing failures at end.

## 7. Testing

- Vitest on synthetic images: ellipse area within 1%, CCL on hand grids, fill-holes, Otsu on bimodal
  data, FFT calibration on a synthetic ruler of known period (±1%).
- Regression: larva mm² vs old `segment.py` CSVs on ≥3 images.
- ML: metrics only on approved held-out masks; a too-good score (Dice > 0.98) is treated as a leak until
  disproven.
- Look at every overlay/page shipped before calling it done.

## 8. Milestones = site chapters

m01 scaffold · m02 image I/O + canvas viewer · m03 larva pipeline · m04 FFT calibration ·
m05 measurements table + CSV · m06 Node CLI · m07 SAM drafts + review page · m08 train/eval/export ·
m09 adult model in browser + uncertainty · m10 polish, deploy, README.

Each chapter: goal, concepts, annotated key code, "try it yourself", `git diff mNN-1 mNN`, pitfalls.

## 9. Conventions

Commits as AyaanShankta only, no co-author trailer; repo private `ayaanshankta/flymorph`.
Large artefacts (SAM weights, drafts) are git-ignored. Claude-drafted masks are labelled drafts until approved.
