import { expect, test } from 'vitest';
import { misorderedFlies } from '../src/anatomy';

// 20×30 grid, one fly in column 5–14. Rows: head 0–4, thorax 10–19, abdomen 20–29 (or abdomen above).
function fly(abdomenAbove: boolean) {
  const w = 20, h = 30, l = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 5; x < 15; x++) {
      const part = abdomenAbove
        ? (y < 5 ? 1 : y < 10 ? 3 : y < 20 ? 2 : 0)   // head, abdomen, thorax: head & abdomen both above thorax
        : (y < 5 ? 1 : y < 10 ? 0 : y < 20 ? 2 : 3);  // head, gap, thorax, abdomen
      l[y * w + x] = part;
    }
  // join the head to the rest in the normal fly so it is one connected blob
  if (!abdomenAbove) for (let y = 5; y < 10; y++) l[y * w + 10] = 2;
  return { l, w, h };
}

test('thorax between head and abdomen is fine', () => {
  const { l, w, h } = fly(false);
  expect(misorderedFlies(l, w, h)).toEqual([]);
});

test('head and abdomen on the same side of the thorax is reported', () => {
  const { l, w, h } = fly(true);
  expect(misorderedFlies(l, w, h)).toHaveLength(1);
});

test('a fly missing a part is not judged', () => {
  const l = new Uint8Array(100).fill(2);
  expect(misorderedFlies(l, 10, 10)).toEqual([]);
});
