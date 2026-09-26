# FlyMorph

Measure *Drosophila* larvae and adult flies in mm² from stereo-microscope photos, in the browser or from the command line.

- **Larvae**: a classical computer-vision pipeline in plain TypeScript (Otsu threshold → morphology → hole filling → connected components → shape checks).
- **Adult flies**: head / thorax / abdomen area from **FlyNet**, a 1.66 M-parameter U-Net distilled from Meta's Segment Anything (SAM) via human-reviewed drafts. It runs on WebGPU in the browser and reports every area as **mean ± SD** from Monte Carlo dropout.
- **Scale**: pixels per mm read automatically from a stage-micrometer photo with an FFT of its tick spacing, with a two-click fallback.

![FlyMorph measuring larvae](docs/img/larvae.png)

![FlyMorph measuring adult flies on WebGPU](docs/img/adult.png)

## Quickstart

```bash
npm install
npm run dev                 # web app → http://localhost:5173
npm test                    # 36 unit tests on the core

# batch a whole imaging session to CSV
npx tsx apps/cli/src/main.ts larva "path/to/session" --scale "path/to/session/scale.jpg" --out larvae.csv
npx tsx apps/cli/src/main.ts adult "path/to/session" --scale "path/to/session/scale.jpg" --out flies.csv
```

## How it's built

```
packages/core   pure TypeScript over typed arrays: image ops, larva pipeline, FFT calibration, adult post-processing
apps/web        Vite + React: measure page (Web Worker for pixel loops) and #/review label page
apps/cli        Node: same core, sharp for decoding, onnxruntime-node for the model
ml/             Python: manifest → SAM drafts → review server → train → evaluate → ONNX export
site/           the step-by-step build course (node site/build.mjs)
```

The ML loop:

1. `ml/manifest.py` indexes every photo by content hash and groups them into imaging sessions.
2. `ml/sam_draft.py` finds each fly, works out its long axis and which end the head is on, then prompts SAM for each part.
3. `ml/draft_server.py` plus the web app's `#/review` page let a person approve, reject, flip or re-prompt each draft. **Only approved masks are ground truth.**
4. `ml/train.py` trains FlyNet (ImageNet-pretrained MobileNetV3-small encoder, cross-entropy + Dice loss) on a split by imaging session.
5. `ml/evaluate.py` reports Dice, median per-fly area error and ±2 SD coverage on held-out sessions.
6. `ml/export.py` exports ONNX, checks it against PyTorch, and copies it into the web app.

## Results

The shipped model (v0) was trained on **unreviewed SAM drafts** to prove the pipeline end to end, so these are not accuracy claims. Scored against drafts on 24 images from 6 held-out sessions (`ml/results/v0-drafts-metrics.json`):

| Part | Dice | Median area error | Inside ±2 SD |
|---|---:|---:|---:|
| head | 0.54 | 47% | 18% |
| thorax | 0.63 | 27% | 24% |
| abdomen | 0.57 | 51% | 17% |
| body | 0.81 | 21% | 19% |

Two things to fix before trusting it: part boundaries need reviewed labels, and the error bars are over-confident (±2 SD should hold ~95% of true areas, not ~20%). Retrain with `ml/train.py --labels masks` after approving ≥ 60 masks in `#/review`.

Larva pipeline, checked on real images:
- The old fixed threshold (V > 95) cut into translucent larvae; Otsu follows the true edge (+7–15% area on the same image).
- Repeat photos of the same larvae agree within ~1%.
- Scale calibration succeeds on 63 of 75 real micrometer images with zero false accepts; the rest fall back to two clicks.

## Limitations

- Touching flies are measured as one fly. Larvae touching background objects are flagged `irregular`, not separated.
- The scale photo must be taken at the same zoom as the specimen photo.
- FlyNet has only seen one lab's microscope and backgrounds. A new setup needs a few new labels.
