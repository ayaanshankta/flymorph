import { type RGB, resizeBilinear, gray } from './image';
import { components } from './ccl';
import { open } from './morph';
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

  // Flies = blobs of "some pass thought this was fly" (so every pass is counted in the same region).
  // A blob missing a part (e.g. an abdomen cut off by a gap in the mask) joins its nearest neighbour within
  // 2·GROUP px; complete flies are never merged, however close they stand.
  const fly = open(anyFly, 2), { labels: blob, comps } = components(fly);
  const has = comps.map(() => [0, 0, 0, 0]);
  for (let i = 0; i < N; i++) if (blob[i]) has[blob[i] - 1][labels[i]]++;
  const root = comps.map((_, j) => j);
  const find = (j: number): number => (root[j] === j ? j : (root[j] = find(root[j])));
  const gap = (a: (typeof comps)[number], b: (typeof comps)[number]) => Math.hypot(
    Math.max(0, a.x - (b.x + b.w), b.x - (a.x + a.w)), Math.max(0, a.y - (b.y + b.h), b.y - (a.y + a.h)));
  comps.forEach((a, j) => {
    if (has[j][1] && has[j][2] && has[j][3]) return;
    let best = -1;
    comps.forEach((b, k) => { if (k !== j && gap(a, b) <= 2 * GROUP && (best < 0 || gap(a, b) < gap(a, comps[best]))) best = k; });
    if (best >= 0) root[find(j)] = find(best);
  });
  const inst = new Int32Array(N);
  for (let i = 0; i < N; i++) if (blob[i]) inst[i] = find(blob[i] - 1) + 1;
  const groups = [...new Set(comps.map((_, j) => find(j)))].map((r) => {
    const members = comps.filter((_, j) => find(j) === r);
    const x = Math.min(...members.map((m) => m.x)), y = Math.min(...members.map((m) => m.y));
    return { id: r + 1, x, y, w: Math.max(...members.map((m) => m.x + m.w)) - x, h: Math.max(...members.map((m) => m.y + m.h)) - y };
  }).sort((a, b) => a.y - b.y || a.x - b.x);
  const problems = anatomyIssues(labels, S, S, inst); // body-plan rules on the averaged prediction
  const toPx = 1 / box.scale ** 2, px2 = cal.pxPerMm ** 2, out: Measurement[] = [];
  let id = 0;
  for (const c of groups) {
    const counts = Array.from({ length: T }, () => [0, 0, 0, 0]);
    let x0 = S, y0 = S, x1 = -1, y1 = -1, pixels = 0;
    for (let y = c.y; y < c.y + c.h; y++)
      for (let x = c.x; x < c.x + c.w; x++) {
        const i = y * S + x;
        if (inst[i] !== c.id) continue;
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
