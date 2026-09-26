import { expect, test } from 'vitest';
import { measureLarvae } from '../src/larva';
import { rgb, ellipse } from './synth';

const cal = { pxPerMm: 100, confidence: 99, method: 'manual' as const };
const opt = { minAreaMm2: 0.05 };
const LARVA: [number, number, number] = [230, 240, 200];
const BG: [number, number, number] = [15, 15, 20];

test('ellipse area within 1% and specks rejected', () => {
  const e = ellipse(100, 150, 25, 80); // π·25·80 = 6283 px = 0.628 mm²
  const speck = (x: number, y: number) => x > 250 && x < 254 && y > 20 && y < 24;
  const { measurements } = measureLarvae(rgb(300, 300, (x, y) => (e(x, y) || speck(x, y) ? LARVA : BG)), cal, opt);
  expect(measurements).toHaveLength(1);
  expect(Math.abs(measurements[0].areaMm2 - 0.6283) / 0.6283).toBeLessThan(0.01);
});

test('larva touching the border is flagged edge', () => {
  const e = ellipse(20, 150, 25, 80);
  const { measurements } = measureLarvae(rgb(300, 300, (x, y) => (e(x, y) ? LARVA : BG)), cal, opt);
  expect(measurements[0].flag).toContain('edge');
});

test('a single large larva is NOT flagged irregular', () => {
  const es = [ellipse(40, 150, 20, 70), ellipse(100, 150, 20, 70), ellipse(160, 150, 20, 70), ellipse(240, 150, 40, 110)];
  const img = rgb(300, 300, (x, y) => (es.some((f) => f(x, y)) ? LARVA : BG));
  expect(measureLarvae(img, cal, opt).measurements.map((m) => m.flag ?? '')).toEqual(['', '', '', '']);
});

test('a non-larva shape (two blobs joined in an L) is flagged irregular', () => {
  const es = [ellipse(60, 150, 20, 70), ellipse(170, 110, 20, 70), (x: number, y: number) => x >= 150 && x <= 260 && y >= 170 && y <= 200];
  const img = rgb(300, 300, (x, y) => (es.some((f) => f(x, y)) ? LARVA : BG));
  expect(measureLarvae(img, cal, opt).measurements.map((m) => m.flag ?? '')).toEqual(['irregular', '']); // raster order: the L starts higher up
});

test('no larvae -> empty list', () => {
  expect(measureLarvae(rgb(50, 50, () => [10, 10, 10]), cal, opt).measurements).toEqual([]);
});
