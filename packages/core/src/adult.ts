import { type RGB, resizeBilinear, gray } from './image';
import { components } from './ccl';
import { open, dilate } from './morph';
import { anatomyIssues } from './anatomy';
import type { Calibration, Measurement } from './larva';

// Contract with the ONNX model (ml/export.py):
//   image [T,3,S,S] float32, ImageNet-normalised · drop [T,576,1,1] float32 · → probs [T,4,S,S] softmax
export const S = 512, DROP_C = 576, PARTS = ['head', 'thorax', 'abdomen'] as const;
const GROUP = 12; // px at 512: mask pieces this close belong to the same fly
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export type Letterbox = { scale: number; padX: number; padY: number; w: number; h: number };
export type RunModel = (image: Float32Array, drop: Float32Array, T: number) => Promise<Float32Array>;

// Shrink so the long side is S, centre on a black S×S square (keeps the aspect ratio), normalise.
export function preprocess(img: RGB) {
  const scale = S / Math.max(img.w, img.h), w = Math.round(img.w * scale), h = Math.round(img.h * scale);
  const padX = (S - w) >> 1, padY = (S - h) >> 1, r = resizeBilinear(img, w, h), tensor = new Float32Array(3 * S * S);
  for (let c = 0; c < 3; c++) tensor.fill(-MEAN[c] / STD[c], c * S * S, (c + 1) * S * S);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < 3; c++)
        tensor[c * S * S + (y + padY) * S + x + padX] = (r.data[(y * w + x) * 4 + c] / 255 - MEAN[c]) / STD[c];
  return { tensor, box: { scale, padX, padY, w, h } };
}

// Monte Carlo dropout: each pass switches off a random 20% of the bottleneck channels (and scales the rest
// up so the average is unchanged). The spread of answers across passes is the model's uncertainty.
export function dropMasks(T: number, p = 0.2, rand = Math.random) {
  return Float32Array.from({ length: T * DROP_C }, () => (rand() < p ? 0 : 1 / (1 - p)));
}

export function postprocess(probs: Float32Array, T: number, box: Letterbox, cal: Calibration) {
  const N = S * S, perPass = new Uint8Array(T * N), labels = new Uint8Array(N), anyFly = gray(S, S);
  for (let i = 0; i < N; i++) {
    const mean = [0, 0, 0, 0];
    for (let t = 0; t < T; t++) {
      let best = 0;
      for (let k = 0; k < 4; k++) {
        const v = probs[(t * 4 + k) * N + i];
        mean[k] += v;
        if (v > probs[(t * 4 + best) * N + i]) best = k;
      }
      perPass[t * N + i] = best;
      if (best) anyFly.data[i] = 1;
    }
    labels[i] = mean.indexOf(Math.max(...mean)); // what the overlay shows
  }

  // Flies = blobs of "some pass thought this was fly" (so every pass is counted in the same region), grouped
  // with a GROUP-px tolerance so a fly with a gap in its mask is still measured as one fly.
  const fly = open(anyFly, 2), { labels: inst, comps } = components(dilate(fly, GROUP));
  const problems = anatomyIssues(labels, S, S, GROUP); // body-plan rules on the averaged prediction
  const toPx = 1 / box.scale ** 2, px2 = cal.pxPerMm ** 2, out: Measurement[] = [];
  let id = 0;
  for (const c of comps) {
    const counts = Array.from({ length: T }, () => [0, 0, 0, 0]);
    let x0 = S, y0 = S, x1 = -1, y1 = -1, pixels = 0;
    for (let y = c.y; y < c.y + c.h; y++)
      for (let x = c.x; x < c.x + c.w; x++) {
        const i = y * S + x;
        if (inst[i] !== c.id || !fly.data[i]) continue;
        pixels++; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
        for (let t = 0; t < T; t++) counts[t][perPass[t * N + i]]++;
      }
    if (pixels <= 0.002 * N) continue; // crumbs
    id++;
    const edge = x0 === 0 || y0 === 0 || x1 === S - 1 || y1 === S - 1;
    const issues = problems.find((p) => p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1)?.issues ?? [];
    const bbox: Measurement['bbox'] = [(x0 - box.padX) / box.scale, (y0 - box.padY) / box.scale, (x1 - x0 + 1) / box.scale, (y1 - y0 + 1) / box.scale];
    const series = [1, 2, 3].map((k) => counts.map((n) => n[k])).concat([counts.map((n) => n[1] + n[2] + n[3])]);
    (['head', 'thorax', 'abdomen', 'body'] as const).forEach((part, k) => {
      const mean = series[k].reduce((a, b) => a + b, 0) / T;
      if (!mean) return;
      const sd = Math.sqrt(series[k].reduce((a, b) => a + (b - mean) ** 2, 0) / T);
      const flags = [edge && 'edge', ...(part === 'body' ? [sd / mean > 0.1 && 'uncertain', ...issues.map((i) => `${i}?`)] : [])]
        .filter(Boolean);
      out.push({ id, part, areaPx: Math.round(mean * toPx), areaMm2: (mean * toPx) / px2, sdMm2: (sd * toPx) / px2,
        bbox, ...(flags.length ? { flag: flags.join(',') } : {}) });
    });
  }
  return { measurements: out, labels };
}

// For drawing: look up each original pixel's label in the S×S map (nearest neighbour through the letterbox).
export function labelsToImage(labels: Uint8Array, box: Letterbox, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(S - 1, Math.floor(y * box.scale) + box.padY);
    for (let x = 0; x < w; x++) out[y * w + x] = labels[sy * S + Math.min(S - 1, Math.floor(x * box.scale) + box.padX)];
  }
  return out;
}

// Run T stochastic passes in one batch and turn them into per-fly, per-part areas ± SD.
export async function measureAdult(img: RGB, cal: Calibration, run: RunModel, T = 8) {
  const { tensor, box } = preprocess(img), image = new Float32Array(T * tensor.length);
  for (let t = 0; t < T; t++) image.set(tensor, t * tensor.length);
  const probs = await run(image, dropMasks(T), T);
  return { ...postprocess(probs, T, box, cal), box };
}
