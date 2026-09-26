import { expect, test } from 'vitest';
import { otsu, binarize } from '../src/threshold';

test('otsu splits a bimodal image between the modes', () => {
  const data = new Uint8Array(1000).map((_, i) => (i < 600 ? 20 + (i % 10) : 200 + (i % 10)));
  const t = otsu({ w: 1000, h: 1, data });
  expect(t).toBeGreaterThanOrEqual(29);
  expect(t).toBeLessThan(200);
});

test('binarize marks pixels strictly above t', () => {
  expect(Array.from(binarize({ w: 3, h: 1, data: Uint8Array.from([5, 10, 11]) }, 10).data)).toEqual([0, 0, 1]);
});
