import { expect, test } from 'vitest';
import { open, close } from '../src/morph';
import { fillHoles } from '../src/fill';
import { mask } from './synth';

const sum = (m: { data: Uint8Array }) => m.data.reduce((a, b) => a + b, 0);

test('open removes specks smaller than the kernel', () => {
  const m = mask(20, 20, (x, y) => (x === 3 && y === 3) || (x >= 8 && x < 16 && y >= 8 && y < 16));
  const o = open(m, 1);
  expect(o.data[3 * 20 + 3]).toBe(0);
  expect(sum(o)).toBe(64);
});

test('close bridges a 1px gap', () => {
  const m = mask(20, 5, (x, y) => y === 2 && x !== 10 && x > 2 && x < 17);
  expect(close(m, 1).data[2 * 20 + 10]).toBe(1);
});

test('fillHoles fills an enclosed hole but not the outside', () => {
  const m = mask(10, 10, (x, y) => x >= 2 && x <= 7 && y >= 2 && y <= 7 && !(x >= 4 && x <= 5 && y >= 4 && y <= 5));
  const f = fillHoles(m);
  expect(sum(f)).toBe(36);
  expect(f.data[0]).toBe(0);
});
