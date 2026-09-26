import { expect, test } from 'vitest';
import { postprocess, preprocess, dropMasks, S } from '../src/adult';
import { rgb } from './synth';

const cal = { pxPerMm: 100, confidence: 99, method: 'manual' as const };
const box = { scale: 1, padX: 0, padY: 0, w: S, h: S };

// Fake model output: probability 1 for the class `fly(x, y, pass)` returns, 0 elsewhere. Layout [T, 4, S, S].
function fake(T: number, fly: (x: number, y: number, t: number) => number) {
  const p = new Float32Array(T * 4 * S * S);
  for (let t = 0; t < T; t++) for (let i = 0; i < S * S; i++) p[(t * 4 + fly(i % S, Math.floor(i / S), t)) * S * S + i] = 1;
  return p;
}

test('no fly -> empty list', () => {
  expect(postprocess(fake(2, () => 0), 2, box, cal).measurements).toEqual([]);
});

test('one fly: each part, body = sum, sd from disagreement between passes', () => {
  // x 100–119: head rows 200–219, thorax 220–259, abdomen 260–319 (pass 0) or 260–323 (pass 1)
  const fly = (x: number, y: number, t: number) =>
    x < 100 || x >= 120 || y < 200 ? 0 : y < 220 ? 1 : y < 260 ? 2 : y < (t ? 324 : 320) ? 3 : 0;
  const by = Object.fromEntries(postprocess(fake(2, fly), 2, box, cal).measurements.map((m) => [m.part, m]));
  expect(by.head.areaPx).toBe(400);
  expect(by.thorax.areaPx).toBe(800);
  expect(by.abdomen.areaPx).toBe(1240); // mean of 1200 and 1280
  expect(by.abdomen.sdMm2).toBeCloseTo(40 / 1e4, 6); // population sd 40 px at 100 px/mm
  expect(by.body.areaPx).toBe(2440);
  expect(by.head.areaMm2).toBeCloseTo(0.04, 6);
});

test('two separate flies get ids 1 and 2', () => {
  const ms = postprocess(fake(1, (x, y) => (y > 100 && y < 200 && ((x > 20 && x < 80) || x > 400) ? 2 : 0)), 1, box, cal).measurements;
  expect(new Set(ms.map((m) => m.id))).toEqual(new Set([1, 2]));
});

test('areas are converted back to original-image pixels through the letterbox', () => {
  const half = { scale: 0.5, padX: 0, padY: 0, w: S, h: S }; // model saw the image at half size
  const ms = postprocess(fake(1, (x, y) => (x >= 100 && x < 120 && y >= 100 && y < 140 ? 2 : 0)), 1, half, cal).measurements;
  expect(ms.find((m) => m.part === 'thorax')!.areaPx).toBe(800 * 4);
  expect(ms[0].bbox).toEqual([200, 200, 40, 80]);
});

test('a fly whose passes disagree a lot is flagged uncertain', () => {
  const fly = (x: number, y: number, t: number) => (x < 100 || x >= 140 || y < 100 || y >= (t ? 220 : 140) ? 0 : 2);
  const body = postprocess(fake(2, fly), 2, box, cal).measurements.find((m) => m.part === 'body')!;
  expect(body.flag).toBe('uncertain');
});

test('drop masks are 0 or 1/(1-p) with mean ≈ 1', () => {
  const d = dropMasks(8);
  expect(Math.abs(d.reduce((a, b) => a + b, 0) / d.length - 1)).toBeLessThan(0.1);
  expect(new Set(d).size).toBe(2);
});

test('preprocess letterboxes a wide image into S×S, centred vertically', () => {
  const { tensor, box } = preprocess(rgb(1024, 512, () => [255, 255, 255]));
  expect(box).toEqual({ scale: 0.5, padX: 0, padY: 128, w: 512, h: 256 });
  expect(tensor.length).toBe(3 * S * S);
  expect(tensor[0]).toBeCloseTo((0 - 0.485) / 0.229, 4); // padding row = black
  expect(tensor[200 * S + 10]).toBeCloseTo((1 - 0.485) / 0.229, 4); // image row = white
});

test('labelsToImage maps the S×S label map back onto the original image through the letterbox', async () => {
  const { labelsToImage } = await import('../src/adult');
  const lab = new Uint8Array(S * S);
  lab[(128 + 10) * S + 20] = 3; // model pixel (20, 138) ← image pixel (40, 20) at scale 0.5, padY 128
  const out = labelsToImage(lab, { scale: 0.5, padX: 0, padY: 128, w: 512, h: 256 }, 1024, 512);
  expect(out.length).toBe(1024 * 512);
  expect(out[20 * 1024 + 40]).toBe(3);
  expect(out[20 * 1024 + 44]).toBe(0);
});

test('a fly whose head and abdomen are on the same side of the thorax is flagged anatomy?', () => {
  // column 100–119: head rows 200–219, abdomen 220–259, thorax 260–299 → thorax is not in the middle
  const fly = (x: number, y: number) => (x < 100 || x >= 120 || y < 200 || y >= 300 ? 0 : y < 220 ? 1 : y < 260 ? 3 : 2);
  const body = postprocess(fake(1, fly), 1, box, cal).measurements.find((m) => m.part === 'body')!;
  expect(body.flag).toContain('anatomy?');
});
