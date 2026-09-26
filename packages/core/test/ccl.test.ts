import { expect, test } from 'vitest';
import { components } from '../src/ccl';
import { mask } from './synth';

test('8-connected components with stats', () => {
  // block A 3x3 at (1,1); pixel (4,4) touches A only diagonally; block B touches the image edge
  const m = mask(10, 6, (x, y) => (x >= 1 && x <= 3 && y >= 1 && y <= 3) || (x === 4 && y === 4) || (x >= 7 && y <= 1));
  const { comps, labels } = components(m);
  // raster order: block B starts on row 0, so it is component 1
  expect(comps.map((c) => c.area)).toEqual([6, 10]);
  expect(comps[1]).toMatchObject({ x: 1, y: 1, w: 4, h: 4, edge: false });
  expect(comps[0].edge).toBe(true);
  expect(labels[4 * 10 + 4]).toBe(2);
});
