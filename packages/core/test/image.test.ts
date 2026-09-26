import { expect, test } from 'vitest';
import { toValue, gaussianBlur, resizeBilinear } from '../src/image';
import { rgb } from './synth';

test('toValue takes the max channel', () => {
  const img = rgb(2, 1, (x) => (x === 0 ? [10, 200, 30] : [77, 77, 77]));
  expect(Array.from(toValue(img).data)).toEqual([200, 77]);
});

test('blur preserves a constant image', () => {
  const g = toValue(rgb(9, 9, () => [120, 0, 0]));
  expect(gaussianBlur(g).data.every((v) => v === 120)).toBe(true);
});

test('blur spreads a single bright pixel', () => {
  const g = toValue(rgb(5, 5, (x, y) => (x === 2 && y === 2 ? [255, 0, 0] : [0, 0, 0])));
  const b = gaussianBlur(g).data;
  expect(b[12]).toBeLessThan(255);
  expect(b[11]).toBeGreaterThan(0);
});

test('resize halves dimensions and keeps colour', () => {
  const r = resizeBilinear(rgb(8, 6, () => [50, 100, 150]), 4, 3);
  expect([r.w, r.h, r.data[0], r.data[1], r.data[2]]).toEqual([4, 3, 50, 100, 150]);
});
