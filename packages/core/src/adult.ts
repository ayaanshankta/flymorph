import { type RGB, resizeBilinear, gray } from './image';
import { components } from './ccl';
import { open } from './morph';
import type { Calibration, Measurement } from './larva';

// Contract with the ONNX model (ml/export.py):
//   image [T,3,S,S] float32, ImageNet-normalised · drop [T,576,1,1] float32 · → probs [T,4,S,S] softmax
export const S = 512, DROP_C = 576, PARTS = ['head', 'thorax', 'abdomen'] as const;
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

  // Flies = connected blobs of "some pass thought this was fly" (so every pass is counted in the same region).
  const { labels: inst, comps } = components(open(anyFly, 2));
  const toPx = 1 / box.scale ** 2, px2 = cal.pxPerMm ** 2, out: Measurement[] = [];
  comps.filter((c) => c.area > 0.002 * N).forEach((c, f) => {
    const counts = Array.from({ length: T }, () => [0, 0, 0, 0]);
    for (let y = c.y; y < c.y + c.h; y++)
      for (let x = c.x; x < c.x + c.w; x++) {
        const i = y * S + x;
        if (inst[i] === c.id) for (let t = 0; t < T; t++) counts[t][perPass[t * N + i]]++;
      }
    const bbox: Measurement['bbox'] = [(c.x - box.padX) / box.scale, (c.y - box.padY) / box.scale, c.w / box.scale, c.h / box.scale];
    const series = [1, 2, 3].map((k) => counts.map((n) => n[k])).concat([counts.map((n) => n[1] + n[2] + n[3])]);
    (['head', 'thorax', 'abdomen', 'body'] as const).forEach((part, k) => {
      const mean = series[k].reduce((a, b) => a + b, 0) / T;
      if (!mean) return;
      const sd = Math.sqrt(series[k].reduce((a, b) => a + (b - mean) ** 2, 0) / T);
      const flags = [c.edge && 'edge', part === 'body' && sd / mean > 0.1 && 'uncertain'].filter(Boolean);
      out.push({ id: f + 1, part, areaPx: Math.round(mean * toPx), areaMm2: (mean * toPx) / px2, sdMm2: (sd * toPx) / px2,
        bbox, ...(flags.length ? { flag: flags.join(',') } : {}) });
    });
  });
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
